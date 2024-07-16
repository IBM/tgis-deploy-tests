use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use chrono::{DateTime, Utc};
use futures::future::try_join_all;
use itertools::Itertools;
use object_store::ObjectStore;
use rand::seq::SliceRandom;
use tokio::sync::{mpsc, Semaphore};
use tokio_util::sync::CancellationToken;
use tonic::transport::Certificate;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{
    evaluate::{EvaluationStrategy, Evaluator},
    utils::*,
    worker::{self, Job, RemoteWorker},
    Config, SingleGenerationRequest, TestCase, TestCaseResult, TgisRequest,
};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("worker error: {0}")]
    Worker(#[from] crate::worker::Error),
    #[error("evaluation error: {0}")]
    Evaluation(#[from] crate::evaluate::Error),
    #[error("run error: {0}")]
    Run(String),
    #[error("save error: {0}")]
    Save(String),
    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Orchestrates running test workloads.
pub struct Runner {
    kube: kube::Client,
    object_store: Arc<dyn ObjectStore>,
    config: Arc<Config>,
}

impl Runner {
    pub fn new(
        config: Arc<Config>,
        kube: kube::Client,
        object_store: Arc<dyn ObjectStore>,
    ) -> Self {
        Self {
            kube,
            config,
            object_store,
        }
    }

    pub async fn run(&mut self) -> Result<Run, Error> {
        let mut run = Run::new(self.config.clone());
        let run_id = run.id;
        info!(%run_id, "starting run");
        let shutdown = CancellationToken::new();

        run.plan = create_plan(&self.config, run_id)?;
        println!("{}", run_plan_table(&run));

        let workers: Vec<RemoteWorker> = run
            .plan
            .iter()
            .map(|job| {
                RemoteWorker::new(
                    self.kube.clone(),
                    self.config.clone(),
                    job.clone(),
                    shutdown.clone(),
                )
            })
            .collect();

        let n_workers = workers.len();
        let n_tests = workers.iter().map(|w| w.cases().len()).sum::<usize>();
        let max_parallel_workers = self.config.worker.max_parallel_workers;

        let image = self.config.worker.image_uri.split('/').last().unwrap();
        info!(
            %run_id,
            %n_workers,
            %image,
            "creating workers"
        );

        if n_workers > max_parallel_workers {
            info!("limiting parallel workers to {max_parallel_workers}");
        }
        let n_pt2_compile = self
            .config
            .tests
            .iter()
            .map(|t| t.configs.iter().filter(|&c| c.pt2_compile).count())
            .sum::<usize>();
        if n_pt2_compile > 0 {
            warn!("{n_pt2_compile} configs have pt2_compile enabled, so this may take awhile")
        }
        let semaphore = Arc::new(Semaphore::new(max_parallel_workers));
        let mut tasks = Vec::with_capacity(n_workers);
        let (workers_tx, mut workers_rx) = mpsc::channel(n_workers);

        for mut worker in workers {
            let name = worker.name().to_string();
            debug!(%name, "acquiring worker permit");
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let workers_tx = workers_tx.clone();
            let shutdown = shutdown.clone();

            // Spawn worker tasks
            tasks.push(tokio::spawn(async move {
                // Create worker
                if let Err(error) = worker.create().await {
                    match error {
                        worker::Error::Cancelled => {
                            debug!(%name, "creating cancelled");
                            worker.delete().await?;
                            let _ = workers_tx.send(worker).await;
                            return Err(error);
                        }
                        _ => {
                            error!(%name, "creating failed, cancelling run");
                            shutdown.cancel();
                            worker.delete().await?;
                            let _ = workers_tx.send(worker).await;
                            return Err(error);
                        }
                    }
                }

                // Connect to worker
                tokio::time::sleep(Duration::from_secs(5)).await;
                if let Err(error) = worker.connect().await {
                    match error {
                        worker::Error::Cancelled => {
                            debug!(%name, "connecting cancelled");
                            worker.delete().await?;
                            let _ = workers_tx.send(worker).await;
                            return Err(error);
                        }
                        _ => {
                            error!(%name, "connecting failed, cancelling run");
                            shutdown.cancel();
                            worker.delete().await?;
                            let _ = workers_tx.send(worker).await;
                            return Err(error);
                        }
                    }
                }

                // Process
                if let Err(error) = worker.process().await {
                    match error {
                        worker::Error::Cancelled => {
                            debug!(%name, "processing cancelled");
                            worker.delete().await?;
                            let _ = workers_tx.send(worker).await;
                            return Err(error);
                        }
                        _ => {
                            error!(%name, "processing failed, cancelling run");
                            shutdown.cancel();
                            worker.delete().await?;
                            let _ = workers_tx.send(worker).await;
                            return Err(error);
                        }
                    }
                }

                // Delete worker
                worker.delete().await?;

                let _ = workers_tx.send(worker).await;

                debug!(%name, "dropping worker permit");
                drop(permit);

                Ok::<(), worker::Error>(())
            }));
        }

        drop(workers_tx);

        match try_join_all(tasks).await {
            Ok(tasks) if tasks.iter().all(|t| t.is_ok()) => {
                info!("worker jobs completed");
                let mut workers = Vec::with_capacity(n_workers);
                while let Some(w) = workers_rx.recv().await {
                    workers.push(w);
                }
                run.workers = workers;

                let strategy: Box<dyn EvaluationStrategy> =
                    self.config.runner.evaluation_strategy.clone().into();
                info!(strategy = strategy.name(), "evaluating results");
                let mut results: Vec<TestCaseResult> = Vec::with_capacity(n_tests);
                for worker in &run.workers {
                    for (case, result) in worker.results() {
                        results.push(Evaluator::evaluate(
                            strategy.as_ref(),
                            case,
                            result.clone(),
                        )?);
                    }
                }
                run.results = results;

                let end_time = Utc::now();
                let duration_secs = (end_time - run.start_time).num_seconds();
                info!(%duration_secs, "run completed");
                run.state = RunState::Completed;
                run.end_time = Some(end_time);
                run.save(&self.object_store).await?;

                println!("{}", run_completed_table(&run));
                Ok(run)
            }
            _ => {
                let mut workers = Vec::with_capacity(n_workers);
                while let Some(w) = workers_rx.recv().await {
                    workers.push(w);
                }
                run.workers = workers;

                let end_time = Utc::now();
                let duration_secs = (end_time - run.start_time).num_seconds();
                error!(%duration_secs, "run failed");
                run.state = RunState::Failed;
                run.end_time = Some(end_time);
                run.save(&self.object_store).await?;

                println!("{}", run_failed_table(&run));
                Err(Error::Run("run failed".to_string()))
            }
        }
    }
}

fn create_plan(config: &Config, run_id: Uuid) -> Result<Vec<Job>, Error> {
    let ca_cert = if let Some(ca_cert_path) = &config.runner.ca_cert_path {
        Some(Certificate::from_pem(std::fs::read_to_string(
            ca_cert_path,
        )?))
    } else {
        None
    };
    config
        .tests
        .iter()
        .flat_map(|test| {
            let cases = test.load_cases().unwrap(); // TODO: handle error
            test.configs
                .iter()
                .map(|model_config| {
                    let model_name = test.model_name.clone();
                    let model_config = Arc::new(model_config.clone());
                    let ca_cert = ca_cert.clone();
                    let mut cases: Box<dyn Iterator<Item = TestCase>> =
                        Box::new(cases.iter().cloned().map(|mut case| {
                            case.model_name = Some(model_name.clone());
                            case.model_config = Some(model_config.clone());
                            case
                        }));
                    if model_config.num_shard > 1 {
                        cases = Box::new(cases.filter(|t| !t.single_shard_only))
                    };
                    if model_config.flash_attention {
                        cases = Box::new(cases.filter(|t| t.request.prefix_id().is_none()))
                    };
                    let mut cases: Vec<TestCase> = cases.collect();

                    // Generate streaming cases
                    if config.runner.generate_stream {
                        let mut stream_cases = Vec::with_capacity(cases.len());
                        for case in &cases {
                            if let TgisRequest::Unary(req) = &case.request {
                                if case.expected_response.is_some() && req.len() == 1 {
                                    let mut stream_case = case.clone();
                                    stream_case.name = format!("{} [stream]", stream_case.name);
                                    stream_case.request = TgisRequest::Streaming(
                                        SingleGenerationRequest::try_from(req.clone()).unwrap(),
                                    );
                                    // We set this to the expected unary response defined in the case
                                    // as it would be tricky to "generate" the expected stream response.
                                    // During evaluation, we "merge" the streaming response to
                                    // to compare it to this.
                                    stream_case.expected_response = case.expected_response.clone();
                                    stream_cases.push(stream_case);
                                }
                            }
                        }
                        cases.append(&mut stream_cases);
                    }
                    cases.shuffle(&mut rand::thread_rng());

                    Ok(Job {
                        run_id,
                        model_name,
                        model_config,
                        cases,
                        ca_cert,
                        results: Vec::new(),
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[derive(Debug, Clone)]
pub enum RunState {
    Running,
    Completed,
    Failed,
}

pub struct Run {
    pub id: Uuid,
    pub config: Arc<Config>,
    pub state: RunState,
    pub plan: Vec<Job>,
    pub start: Instant,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub workers: Vec<RemoteWorker>,
    pub results: Vec<TestCaseResult>,
}

impl Run {
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            state: RunState::Running,
            plan: Vec::new(),
            start: Instant::now(),
            start_time: Utc::now(),
            end_time: None,
            workers: Vec::new(),
            results: Vec::new(),
        }
    }

    pub fn workers(&self) -> &[RemoteWorker] {
        &self.workers
    }

    pub fn results(&self) -> &[TestCaseResult] {
        &self.results
    }

    pub fn duration(&self) -> Option<Duration> {
        self.end_time
            .map(|t| (t - self.start_time).to_std().unwrap())
    }

    pub fn duration_secs(&self) -> Option<i64> {
        self.end_time.map(|t| (t - self.start_time).num_seconds())
    }

    pub async fn save(&self, object_store: &Arc<dyn ObjectStore>) -> Result<(), Error> {
        let date = self.start_time.date_naive();
        for worker in &self.workers {
            // Save pod logs
            if let Some(logs) = worker.logs() {
                let path = object_store::path::Path::from(format!(
                    "runs/{date}/{}/logs/{}.log",
                    self.id,
                    worker.name()
                ));
                object_store
                    .put(&path, logs.clone().into())
                    .await
                    .map_err(|e| Error::Save(e.to_string()))?;
            }
            // Save pod info
            let mut pod = worker.pod().clone();
            pod.metadata.annotations = None;
            pod.metadata.managed_fields = None;
            let pod_json = serde_json::to_string_pretty(&pod).unwrap();
            let path = object_store::path::Path::from(format!(
                "runs/{date}/{}/logs/{}.json",
                self.id,
                worker.name()
            ));
            object_store
                .put(&path, pod_json.into())
                .await
                .map_err(|e| Error::Save(e.to_string()))?;
        }
        // Save test results
        let results_to_save: Box<dyn Iterator<Item = &TestCaseResult>> =
            if self.config.runner.save_passed_test_results {
                // Save all (useful for validation/debugging)
                Box::new(self.results.iter())
            } else {
                // Save failed only (default)
                Box::new(self.results.iter().filter(|&r| !r.passed()))
            };
        for (key, group) in &results_to_save.group_by(|&r| r.group()) {
            let path = object_store::path::Path::from(format!(
                "runs/{date}/{}/results/{}.yaml",
                self.id, key
            ));
            let mut results = group.collect::<Vec<_>>();
            results.sort_by_key(|&r| &r.name);
            let results_yaml = serde_yaml::to_string(&results)?;
            object_store
                .put(&path, results_yaml.into())
                .await
                .map_err(|e| Error::Save(e.to_string()))?;
        }
        Ok(())
    }
}
