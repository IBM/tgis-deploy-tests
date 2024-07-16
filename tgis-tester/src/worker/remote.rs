use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use futures::{future::try_join_all, stream, StreamExt};
use k8s_openapi::{
    api::core::v1::{
        EmptyDirVolumeSource, EnvVar, PersistentVolumeClaimVolumeSource, Pod, ResourceRequirements,
        SecretVolumeSource, Service, Volume, VolumeMount,
    },
    apimachinery::pkg::api::resource::Quantity,
};
use kube::{
    api::{DeleteParams, LogParams, PostParams},
    runtime::wait::await_condition,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;
use tonic::{
    transport::{Channel, ClientTlsConfig},
    Request,
};
use tracing::{debug, error, info, warn};

use crate::{
    config::{DeploymentFramework, GpuSpec, Mig},
    pb::generation_service_client::GenerationServiceClient,
    route::Route,
    utils::worker_name,
    Config, GenerationResponse, TgisRequest,
};

use super::*;

/// Represents a remote worker running in a Kubernetes cluster.
pub struct RemoteWorker {
    name: String,
    state: WorkerState,
    grpc_addr: String,
    config: Arc<Config>,
    job: Job,
    kube: kube::Client,
    pod: Pod,
    service: Service,
    route: Route,
    client: Option<GenerationServiceClient<Channel>>,
    logs: Option<String>,
    shutdown: CancellationToken,
    create_start: Option<Instant>,
    scheduled_duration: Option<Duration>,
    ready_duration: Option<Duration>,
}

impl RemoteWorker {
    /// Creates a new worker.
    pub fn new(
        kube: kube::Client,
        config: Arc<Config>,
        job: Job,
        shutdown: CancellationToken,
    ) -> Self {
        let name = worker_name(&job);
        let grpc_addr = format!(
            "https://{}-{}.{}",
            &name,
            kube.default_namespace(),
            &config.worker.ingress_hostname,
        );
        let pod = worker_pod(&name, &job, &config);
        let service = worker_service(&name, &job);
        let route = worker_route(&name, &job);
        Self {
            name,
            state: WorkerState::New,
            grpc_addr,
            config,
            job,
            kube,
            pod,
            service,
            route,
            client: None,
            logs: None,
            shutdown,
            create_start: None,
            scheduled_duration: None,
            ready_duration: None,
        }
    }

    pub async fn create(&mut self) -> Result<(), Error> {
        info!(name = self.name, "creating worker");
        let start = Instant::now();
        self.create_start = Some(start);
        let pod_api = kube::Api::<Pod>::default_namespaced(self.kube.clone());
        match pod_api.create(&PostParams::default(), &self.pod).await {
            Ok(pod) => {
                self.pod = pod;
                self.state = WorkerState::Pending;

                let name = self.name.clone();

                // Await pod scheduled or cancellation
                let pod_scheduled = tokio::time::timeout(
                    Duration::from_secs(self.config.worker.scheduled_timeout_secs),
                    await_condition(pod_api.clone(), &name, is_pod_scheduled()),
                );

                tokio::select! {
                    scheduled = pod_scheduled => {
                        if scheduled.is_err() {
                            error!(
                                name = self.name,
                                "timeout waiting for pod to be scheduled"
                            );
                            self.state = WorkerState::Failed(FailedReason::PodScheduledTimeout);
                            return Err(Error::PodScheduledTimeout);
                        } else {
                            self.scheduled_duration = Some(start.elapsed());
                        }
                    },
                    _ = self.shutdown.cancelled() => {
                        warn!(
                            name = self.name,
                            "received shutdown signal"
                        );
                        self.state = WorkerState::Cancelled;
                        return Err(Error::Cancelled);
                    },
                }

                // Await pod ready/fail or cancellation
                let pod_ready = tokio::time::timeout(
                    Duration::from_secs(self.config.worker.ready_timeout_secs),
                    await_condition(pod_api.clone(), &name, is_pod_ready()),
                );
                let pod_failed = await_condition(pod_api.clone(), &name, is_pod_failed());
                let pod_evicted = await_condition(pod_api.clone(), &name, is_pod_evicted());
                tokio::select! {
                    ready = pod_ready => {
                        match ready {
                            Ok(ready_result) => {
                                match ready_result {
                                    Ok(_pod) => {
                                        info!(
                                            name = self.name,
                                            "worker ready in {} seconds",
                                            (Instant::now() - start).as_secs()
                                        );
                                        self.ready_duration = Some(start.elapsed());
                                        self.state = WorkerState::Running;

                                        let svc_api = kube::Api::<Service>::default_namespaced(self.kube.clone());
                                        match svc_api.create(&PostParams::default(), &self.service).await {
                                            Ok(service) => self.service = service,
                                            Err(error) => {
                                                error!(name = self.name, ?error, "create service failed");
                                                self.state = WorkerState::Failed(FailedReason::CreateServiceFailed);
                                                return Err(Error::CreateServiceFailed(error.to_string()));
                                            }
                                        }
                                        let route_api = kube::Api::<Route>::default_namespaced(self.kube.clone());
                                        match route_api.create(&PostParams::default(), &self.route).await {
                                            Ok(route) => self.route = route,
                                            Err(error) => {
                                                error!(name = self.name, ?error, "create route failed");
                                                self.state = WorkerState::Failed(FailedReason::CreateRouteFailed);
                                                return Err(Error::CreateRouteFailed(error.to_string()));
                                            }
                                        }
                                        debug!(name = self.name, "worker created");
                                        Ok(())
                                    },
                                    Err(error) => {
                                        error!(
                                            name = self.name,
                                            ?error,
                                            "create pod failed"
                                        );
                                        self.state = WorkerState::Failed(FailedReason::CreatePodFailed);
                                        Err(Error::CreatePodFailed(error.to_string()))
                                    },
                                }
                            },
                            Err(_) => {
                                error!(name = self.name, "pod ready timeout");
                                self.state = WorkerState::Failed(FailedReason::PodReadyTimeout);
                                Err(Error::PodReadyTimeout)
                            },
                        }
                    },
                    _ = pod_failed => {
                        error!(name = self.name, "pod failed during startup");
                        self.state = WorkerState::Failed(FailedReason::PodFailed);
                        Err(Error::PodFailedDuringStartup)
                    },
                    _ = pod_evicted => {
                        error!(name = self.name, "pod evicted");
                        // TODO
                        // let pod_conditions = pod_api
                        //     .get_status(&self.name)
                        //     .await
                        //     .unwrap()
                        //     .status
                        //     .unwrap()
                        //     .conditions
                        //     .unwrap();
                        // let evicted_message = pod_conditions
                        //     .iter()
                        //     .find(|&c| c.reason == Some("Evicted".to_string()))
                        //     .map(|c| c.message.clone().unwrap());
                        let evicted_message = "".to_string();
                        self.state = WorkerState::Failed(FailedReason::PodEvicted(evicted_message.clone()));
                        Err(Error::PodEvicted(evicted_message))
                    },
                    _ = self.shutdown.cancelled() => {
                        warn!(name = self.name, "received shutdown signal");
                        self.state = WorkerState::Cancelled;
                        Err(Error::Cancelled)
                    },
                }
            }
            Err(err) => {
                error!(name = self.name, ?err, "creating worker failed");
                self.state = WorkerState::Failed(FailedReason::CreatePodFailed);
                Err(Error::CreatePodFailed(err.to_string()))
            }
        }
    }

    pub async fn connect(&mut self) -> Result<(), Error> {
        debug!(name = self.name, "connecting to worker");
        if !matches!(&self.state, &WorkerState::Running) {
            error!(name = self.name, "connecting to worker failed: not running");
            return Err(Error::NotRunning);
        }
        let name = self.name.clone();
        let tls = if let Some(ca_cert) = &self.job.ca_cert {
            ClientTlsConfig::new().ca_certificate(ca_cert.clone())
        } else {
            ClientTlsConfig::default()
        };
        let endpoint = Channel::from_shared(self.grpc_addr.clone())
            .unwrap()
            .connect_timeout(Duration::from_secs(10))
            .tls_config(tls)
            .unwrap();
        let max_connection_retries = self.config.worker.max_connection_retries;
        let connect = tokio::spawn(async move {
            let mut retries = 0;
            loop {
                if let Ok(channel) = endpoint.connect().await {
                    break Ok(channel);
                } else {
                    if retries == max_connection_retries {
                        break Err(Error::ConnectMaxRetriesExceeded);
                    }
                    debug!(name = name, "connecting to worker failed, retrying...");
                    retries += 1;
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        });
        // Await connection or cancellation
        tokio::select! {
            channel = connect => {
                let channel = channel.map_err(|_| Error::Connect("connecting to worker failed: connect task error".to_string()))?;
                match channel {
                    Ok(channel) => {
                        debug!(name = self.name, "connected to worker");
                        let client = GenerationServiceClient::new(channel);
                        self.client = Some(client);
                        Ok(())
                    }
                    Err(error) => {
                        error!(
                            name = self.name,
                            grpc_addr = self.grpc_addr,
                            ?error,
                            "connecting to worker failed"
                        );
                        self.state = WorkerState::Failed(FailedReason::ConnectMaxRetriesExceeded);
                        Err(error)
                    }
                }
            },
            _ = self.shutdown.cancelled() => {
                warn!(
                    name = self.name,
                    "received shutdown signal"
                );
                self.state = WorkerState::Cancelled;
                Err(Error::Cancelled)
            },
        }
    }

    pub async fn process(&mut self) -> Result<(), Error> {
        debug!(name = self.name, "processing");
        if let Some(client) = &mut self.client {
            let cases = self.job.cases.clone();
            let tasks = stream::iter(cases.clone())
                .map(|case| {
                    let mut client = client.clone();
                    tokio::spawn(async move {
                        match case.request {
                            TgisRequest::Unary(req) => {
                                match client.generate(Request::new(req.into())).await {
                                    Ok(resp) => {
                                        let resp: BatchedGenerationResponse =
                                            resp.into_inner().into();
                                        Ok(resp.into())
                                    }
                                    Err(status) => Err(status.into()),
                                }
                            }
                            TgisRequest::Streaming(req) => {
                                match client.generate_stream(Request::new(req.into())).await {
                                    Ok(resp) => {
                                        let resp: Vec<GenerationResponse> = resp
                                            .into_inner()
                                            .map(|r| r.unwrap().into())
                                            .collect::<Vec<_>>()
                                            .await;
                                        Ok(resp.into())
                                    }
                                    Err(status) => Err(status.into()),
                                }
                            }
                        }
                    })
                })
                .collect::<Vec<_>>()
                .await;
            let responses = try_join_all(tasks).await.unwrap();
            self.job.results = cases.into_iter().zip(responses).collect();
            self.state = WorkerState::Completed;
            Ok(())
        } else {
            error!(name = self.name, "processing failed: not connected");
            Err(Error::NotConnected)
        }
    }

    pub async fn delete(&mut self) -> Result<(), Error> {
        info!(name = self.name, "deleting worker");
        let pod_api = kube::Api::<Pod>::default_namespaced(self.kube.clone());
        // Get pod logs
        if let Ok(logs) = pod_api.logs(&self.name, &LogParams::default()).await {
            self.logs = Some(logs);
        }
        // Get final pod state
        if let Ok(pod) = pod_api.get(self.name()).await {
            self.pod = pod;
        }
        self.client = None;
        let pod_api = kube::Api::<Pod>::default_namespaced(self.kube.clone());
        let svc_api = kube::Api::<Service>::default_namespaced(self.kube.clone());
        let route_api = kube::Api::<Route>::default_namespaced(self.kube.clone());
        let _ = pod_api.delete(&self.name, &DeleteParams::default()).await;
        let _ = route_api.delete(&self.name, &DeleteParams::default()).await;
        let _ = svc_api.delete(&self.name, &DeleteParams::default()).await;
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn state(&self) -> &WorkerState {
        &self.state
    }

    pub fn job(&self) -> &Job {
        &self.job
    }

    pub fn model_config(&self) -> &ModelConfig {
        &self.job.model_config
    }

    pub fn cases(&self) -> &[TestCase] {
        &self.job.cases
    }

    pub fn results(&self) -> &[(TestCase, Result<TgisResponse, Status>)] {
        &self.job.results
    }

    pub fn pod(&self) -> &Pod {
        &self.pod
    }

    pub fn logs(&self) -> Option<&String> {
        self.logs.as_ref()
    }

    pub fn create_start(&self) -> Option<Instant> {
        self.create_start
    }

    pub fn scheduled_duration(&self) -> Option<Duration> {
        self.scheduled_duration
    }

    pub fn ready_duration(&self) -> Option<Duration> {
        self.ready_duration
    }
}

fn worker_pod(name: &str, job: &Job, config: &Config) -> Pod {
    let envs = worker_envs(&job.model_name, &job.model_config, config);
    let resources = &job.model_config.resources;
    let request_cpu = resources.request_cpu.to_string();
    let limit_cpu = resources.limit_cpu.to_string();
    let request_memory_gb = format!("{}G", resources.request_memory_gb);
    let limit_memory_gb = format!("{}G", resources.limit_memory_gb);
    let mut volume_mounts: Vec<VolumeMount> = config
        .worker
        .pvcs
        .iter()
        .map(|pvc| VolumeMount {
            name: pvc.name.clone(),
            mount_path: pvc.mount_path.clone(),
            read_only: Some(pvc.read_only),
            ..Default::default()
        })
        .collect();
    volume_mounts.push(VolumeMount {
        name: "tls-certs".to_string(),
        mount_path: "/mnt/certs".to_string(),
        read_only: Some(true),
        ..Default::default()
    });
    volume_mounts.push(VolumeMount {
        name: "home".to_string(),
        mount_path: "/home/tgis".to_string(),
        read_only: Some(false),
        ..Default::default()
    });
    volume_mounts.push(VolumeMount {
        name: "vllm".to_string(),
        mount_path: "/home/vllm".to_string(),
        read_only: Some(false),
        ..Default::default()
    });
    volume_mounts.push(VolumeMount {
        name: "tmp".to_string(),
        mount_path: "/tmp".to_string(),
        read_only: Some(false),
        ..Default::default()
    });
    volume_mounts.push(VolumeMount {
        name: "shm".to_string(),
        mount_path: "/dev/shm".to_string(),
        read_only: Some(false),
        ..Default::default()
    });
    let mut volumes: Vec<Volume> = config
        .worker
        .pvcs
        .iter()
        .map(|pvc| Volume {
            name: pvc.name.clone(),
            persistent_volume_claim: Some(PersistentVolumeClaimVolumeSource {
                claim_name: pvc.name.clone(),
                read_only: Some(pvc.read_only),
            }),
            ..Default::default()
        })
        .collect();
    volumes.push(Volume {
        name: "tls-certs".to_string(),
        secret: Some(SecretVolumeSource {
            secret_name: Some(config.worker.tls_secret.clone()),
            ..Default::default()
        }),
        ..Default::default()
    });
    volumes.push(Volume {
        name: "home".to_string(),
        empty_dir: Some(EmptyDirVolumeSource {
            medium: Some("".to_string()),
            size_limit: None,
        }),
        ..Default::default()
    });
    volumes.push(Volume {
        name: "vllm".to_string(),
        empty_dir: Some(EmptyDirVolumeSource {
            medium: Some("".to_string()),
            size_limit: None,
        }),
        ..Default::default()
    });
    volumes.push(Volume {
        name: "tmp".to_string(),
        empty_dir: Some(EmptyDirVolumeSource {
            medium: Some("".to_string()),
            size_limit: None,
        }),
        ..Default::default()
    });
    volumes.push(Volume {
        name: "shm".to_string(),
        empty_dir: Some(EmptyDirVolumeSource {
            medium: Some("".to_string()),
            size_limit: Some(Quantity("1Gi".to_string())),
        }),
        ..Default::default()
    });
    let is_vllm = config.worker.image_uri.contains("vllm");
    let node_selector: BTreeMap<String, String> = if job.model_config.flash_attention || is_vllm {
        // Schedule to node with A100
        BTreeMap::from([(
            "nvidia.com/gpu.product".into(),
            "NVIDIA-A100-SXM4-80GB".into(),
        )])
    } else {
        // Schedule to any feasible node
        BTreeMap::default()
    };
    let gpu = match &resources.gpu {
        Some(spec) => match spec {
            GpuSpec::Gpu(n) => Some(("nvidia.com/gpu".to_string(), Quantity(n.to_string()))),
            GpuSpec::Mig(mig) => match mig {
                Mig::Mig1g10gb(n) => Some((
                    "nvidia.com/mig-1g.10gb".to_string(),
                    Quantity(n.to_string()),
                )),
                Mig::Mig2g20gb(n) => Some((
                    "nvidia.com/mig-2g.20gb".to_string(),
                    Quantity(n.to_string()),
                )),
                Mig::Mig3g40gb(n) => Some((
                    "nvidia.com/mig-3g.40gb".to_string(),
                    Quantity(n.to_string()),
                )),
            },
        },
        None => None,
    };
    let resources = if let Some(gpu) = gpu {
        ResourceRequirements {
            requests: Some(BTreeMap::from([
                ("cpu".into(), Quantity(request_cpu)),
                ("memory".into(), Quantity(request_memory_gb)),
                (gpu.0.clone(), gpu.1.clone()),
            ])),
            limits: Some(BTreeMap::from([
                ("cpu".into(), Quantity(limit_cpu)),
                ("memory".into(), Quantity(limit_memory_gb)),
                (gpu.0.clone(), gpu.1.clone()),
            ])),
        }
    } else {
        ResourceRequirements {
            requests: Some(BTreeMap::from([
                ("cpu".into(), Quantity(request_cpu)),
                ("memory".into(), Quantity(request_memory_gb)),
            ])),
            limits: Some(BTreeMap::from([
                ("cpu".into(), Quantity(limit_cpu)),
                ("memory".into(), Quantity(limit_memory_gb)),
            ])),
        }
    };
    let gpu_util_str = job.model_config.gpu_memory_utilization.to_string();
    let args = if is_vllm {
        Some([
            "--ssl-keyfile",
            "/mnt/certs/tls.key",
            "--ssl-certfile",
            "/mnt/certs/tls.crt",
            "--gpu-memory-utilization",
            gpu_util_str.as_str(),
        ])
    } else {
        None
    };
    let scheme = if is_vllm { "HTTPS" } else { "HTTP" };

    let image_pull_secrets = if config.worker.image_pull_secret.is_some() {
        json!([
            {"name": config.worker.image_pull_secret },
        ])
    } else {
        json!([])
    };

    serde_json::from_value(json!({
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "labels": {
                "app.kubernetes.io/name": name,
                "app.kubernetes.io/part-of": "text-generation-tester",
                "app.kubernetes.io/managed-by": "text-generation-tester",
                "text-generation-tester/run-id": job.run_id.to_string()
            }
        },
        "spec": {
            "serviceAccountName": config.cluster.service_account_name,
            "restartPolicy": "Never",
            "imagePullSecrets": image_pull_secrets,
            "nodeSelector": node_selector,
            "containers": [{
                "name": "text-generation-server",
                "image": config.worker.image_uri,
                "imagePullPolicy": "Always",
                "ports": [
                    { "containerPort": job.model_config.grpc_port }
                ],
                "env": envs,
                "resources": resources,
                "args": args,
                "securityContext": {
                    "allowPrivilegeEscalation": false,
                    "privileged": false,
                    "runAsNonRoot": true,
                    "readOnlyRootFilesystem": true,
                    "seccompProfile": {
                        "type": "RuntimeDefault"
                    },
                    "capabilities": {
                        "drop": [
                            "ALL"
                        ]
                    }
                },
                "startupProbe": {
                    "httpGet": {
                        "port": job.model_config.port,
                        "path": "/health",
                        "scheme": scheme,
                    },
                    "failureThreshold": 24,
                    "periodSeconds": 30,
                },
                "readinessProbe": {
                    "httpGet": {
                        "port": job.model_config.port,
                        "path": "/health",
                        "scheme": scheme,
                    },
                    //"initialDelaySeconds": 20,
                    "periodSeconds": 30,
                    "timeoutSeconds": 5,
                },
                "livenessProbe": {
                    "httpGet": {
                        "port": job.model_config.port,
                        "path": "/health",
                        "scheme": scheme,
                    },
                    "periodSeconds": 100,
                    "timeoutSeconds": 5,
                },
                "terminationGracePeriodSeconds": 60,
                "volumeMounts": volume_mounts,
            }],
            "volumes": volumes,
        }
    }))
    .unwrap()
}

fn worker_service(name: &str, job: &Job) -> Service {
    serde_json::from_value(json!({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": name,
            "labels": {
                "text-generation-tester/run-id": job.run_id.to_string(),
                "app.kubernetes.io/part-of": "text-generation-tester",
                "app.kubernetes.io/managed-by": "text-generation-tester"
            }
        },
        "spec": {
            "selector": {
                "app.kubernetes.io/name": name
            },
            "ports": [
                { "protocol": "TCP", "port": job.model_config.grpc_port, "targetPort": job.model_config.grpc_port }
            ]
        }
    }))
    .unwrap()
}

fn worker_route(name: &str, job: &Job) -> Route {
    serde_json::from_value(json!({
            "apiVersion": "route.openshift.io/v1",
            "kind": "Route",
            "metadata": {
                "name": name,
                "labels": {
                    "text-generation-tester/run-id": job.run_id.to_string(),
                    "app.kubernetes.io/part-of": "text-generation-tester",
                    "app.kubernetes.io/managed-by": "text-generation-tester"
                },
                "annotations": {
                    "ingress.kubernetes.io/allow-http": "false",
                    "ingress.kubernetes.io/ssl-redirect": "false",
                    "ingress.operator.openshift.io/default-enable-http2": "true",
                    "route.openshift.io/termination": "passthrough"
                }
            },
            "spec": {
                "to": {
                    "kind": "Service",
                    "name": name,
                    "weight": 100,
                },
                "port": {
                    "targetPort": job.model_config.grpc_port,
                },
                "tls": {
                    "termination": "passthrough"
                },
                "wildcardPolicy": "None"
            }
    }))
    .unwrap()
}

fn worker_envs(model_name: &str, model_config: &ModelConfig, config: &Config) -> Vec<EnvVar> {
    let mut envs = vec![
        ("MODEL_NAME", model_name.to_string()),
        (
            "DEPLOYMENT_FRAMEWORK",
            model_config.deployment_framework.to_string(),
        ),
        ("DTYPE_STR", model_config.dtype.to_string()),
        ("NUM_SHARD", model_config.num_shard.to_string()),
        (
            "MAX_CONCURRENT_REQUESTS",
            model_config.max_concurrent_requests.to_string(),
        ),
        (
            "MAX_SEQUENCE_LENGTH",
            model_config.max_sequence_length.to_string(),
        ),
        ("MAX_BATCH_SIZE", model_config.max_batch_size.to_string()),
        ("MAX_NEW_TOKENS", model_config.max_new_tokens.to_string()),
        ("FLASH_ATTENTION", model_config.flash_attention.to_string()),
        ("PT2_COMPILE", model_config.pt2_compile.to_string()),
        ("PORT", model_config.port.to_string()),
        ("GRPC_PORT", model_config.grpc_port.to_string()),
        ("NCCL_ASYNC_ERROR_HANDLING", "1".to_string()),
        ("SAFETENSORS_FAST_GPU", "1".to_string()),
        ("PYTHONUNBUFFERED", "1".to_string()),
        (
            "TRANSFORMERS_CACHE",
            config.worker.transformers_cache_path.clone(),
        ),
        (
            "HF_HUB_CACHE",
            config.worker.transformers_cache_path.clone(),
        ),
        ("PREFIX_STORE_PATH", config.worker.prefix_store_path.clone()),
        ("ADAPTER_CACHE", config.worker.adapter_cache.clone()),
        ("TLS_CERT_PATH", "/mnt/certs/tls.crt".to_string()),
        ("TLS_KEY_PATH", "/mnt/certs/tls.key".to_string()),
        (
            "CUDA_PROCESS_MEMORY_FRACTION",
            model_config.cuda_process_memory_fraction.to_string(),
        ),
        ("NUMBA_CACHE_DIR", "/tmp".to_string()),
        ("VLLM_CONFIG_ROOT", "/tmp/".to_string()),
    ];
    if model_config.enable_lora {
        envs.push(("ENABLE_LORA", "True".to_string()));
    }
    if let Some(revision) = &model_config.revision {
        envs.push(("REVISION", revision.clone()));
    }
    if let Some(max_batch_weight) = model_config.max_batch_weight {
        envs.push(("MAX_BATCH_WEIGHT", max_batch_weight.to_string()));
    }
    if let Some(max_prefill_weight) = model_config.max_prefill_weight {
        envs.push(("MAX_PREFILL_WEIGHT", max_prefill_weight.to_string()));
    }
    if let Some(quantize) = &model_config.quantize {
        envs.push(("QUANTIZE", quantize.to_string()));
    }
    if let Some(estimate_memory) = model_config.estimate_memory {
        if !estimate_memory {
            envs.push(("ESTIMATE_MEMORY", "off".to_string()))
        }
    }
    if let Some(load_format) = model_config.load_format.clone() {
        envs.push(("LOAD_FORMAT", load_format))
    }
    if let Some(model_loader_extra_config) = model_config.model_loader_extra_config.clone() {
        envs.push(("MODEL_LOADER_EXTRA_CONFIG", model_loader_extra_config))
    }
    if model_config.output_special_tokens {
        envs.push(("OUTPUT_SPECIAL_TOKENS", "true".to_string()))
    }
    if model_config.is_cpu_only() {
        envs.push(("CUDA_VISIBLE_DEVICES", "-1".to_string()))
    }
    if matches!(
        model_config.deployment_framework,
        DeploymentFramework::HfOptimumOrt
    ) {
        envs.push(("CUDA_PAD_TO_MULT_OF_8", "false".to_string()));
        let merge_onnx_graphs = model_config.merge_onnx_graphs.unwrap_or_default();
        envs.push(("MERGE_ONNX_GRAPHS", merge_onnx_graphs.to_string()));
    }
    envs.into_iter()
        .map(|(key, value)| EnvVar {
            name: key.to_string(),
            value: Some(value),
            value_from: None,
        })
        .collect()
}
