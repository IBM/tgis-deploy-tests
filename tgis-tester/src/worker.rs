pub mod remote;
use k8s_openapi::api::core::v1::Pod;
use kube::runtime::wait::Condition;
pub use remote::RemoteWorker;

use std::sync::Arc;

use tonic::transport::Certificate;
use tracing::error;
use uuid::Uuid;

use crate::{BatchedGenerationResponse, ModelConfig, ModelName, Status, TestCase, TgisResponse};

/// Worker errors.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("connect failed: {0}")]
    Connect(String),
    #[error("connect failed, max retries exceeded")]
    ConnectMaxRetriesExceeded,
    #[error("create pod failed: {0}")]
    CreatePodFailed(String),
    #[error("create service failed: {0}")]
    CreateServiceFailed(String),
    #[error("create route failed: {0}")]
    CreateRouteFailed(String),
    #[error("delete pod failed: {0}")]
    DeletePodFailed(String),
    #[error("delete service failed: {0}")]
    DeleteServiceFailed(String),
    #[error("delete route failed: {0}")]
    DeleteRouteFailed(String),
    #[error("not connected")]
    NotConnected,
    #[error(transparent)]
    Status(#[from] tonic::Status),
    #[error("not running")]
    NotRunning,
    #[error("cancelled")]
    Cancelled,
    #[error("timeout waiting for pod to be scheduled")]
    PodScheduledTimeout,
    #[error("timeout waiting for pod to be ready")]
    PodReadyTimeout,
    #[error("pod failed during startup")]
    PodFailedDuringStartup,
    #[error("pod evicted: {0}")]
    PodEvicted(String),
    #[error("logs")]
    Logs(String),
}

#[derive(Debug, Clone)]
pub struct Job {
    pub run_id: Uuid,
    pub model_name: ModelName,
    pub model_config: Arc<ModelConfig>,
    pub cases: Vec<TestCase>,
    pub results: Vec<(TestCase, Result<TgisResponse, Status>)>,
    pub ca_cert: Option<Certificate>,
}

/// Worker state.
#[derive(Debug, Clone)]
pub enum WorkerState {
    New,
    Pending,
    Running,
    Failed(FailedReason),
    Completed,
    Cancelled,
}

impl std::fmt::Display for WorkerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkerState::New => write!(f, "new"),
            WorkerState::Pending => write!(f, "pending"),
            WorkerState::Running => write!(f, "🟡 running"),
            WorkerState::Failed(reason) => write!(f, "🔴 failed ({reason})"),
            WorkerState::Completed => write!(f, "🟢 completed"),
            WorkerState::Cancelled => write!(f, "⚪ cancelled"),
        }
    }
}

/// Worker failed reason.
#[derive(Debug, Clone)]
pub enum FailedReason {
    PodFailed,
    PodEvicted(String),
    PodScheduledTimeout,
    PodReadyTimeout,
    ConnectMaxRetriesExceeded,
    CreatePodFailed,
    CreateServiceFailed,
    CreateRouteFailed,
}

impl std::fmt::Display for FailedReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FailedReason::PodFailed => write!(f, "pod failed"),
            FailedReason::PodEvicted(message) => write!(f, "pod evicted: {message}"),
            FailedReason::PodScheduledTimeout => write!(f, "pod scheduled timeout"),
            FailedReason::PodReadyTimeout => write!(f, "pod ready timeout"),
            FailedReason::ConnectMaxRetriesExceeded => write!(f, "connect max retries exceeded"),
            FailedReason::CreatePodFailed => write!(f, "create pod failed"),
            FailedReason::CreateServiceFailed => write!(f, "create service failed"),
            FailedReason::CreateRouteFailed => write!(f, "create route failed"),
        }
    }
}

pub fn is_pod_ready() -> impl Condition<Pod> {
    |obj: Option<&Pod>| {
        if let Some(pod) = &obj {
            if let Some(status) = &pod.status {
                if let Some(conds) = &status.conditions {
                    if let Some(ready) = conds.iter().find(|&c| c.type_ == "Ready") {
                        return ready.status == "True";
                    }
                }
            }
        }
        false
    }
}

pub fn is_pod_scheduled() -> impl Condition<Pod> {
    |obj: Option<&Pod>| {
        if let Some(pod) = &obj {
            if let Some(status) = &pod.status {
                if let Some(conds) = &status.conditions {
                    if let Some(ready) = conds.iter().find(|&c| c.type_ == "PodScheduled") {
                        return ready.status == "True";
                    }
                }
            }
        }
        false
    }
}

pub fn is_pod_failed() -> impl Condition<Pod> {
    |obj: Option<&Pod>| {
        if let Some(pod) = &obj {
            if let Some(status) = &pod.status {
                if let Some(phase) = &status.phase {
                    return phase == "Failed";
                }
            }
        }
        false
    }
}

pub fn is_pod_evicted() -> impl Condition<Pod> {
    |obj: Option<&Pod>| {
        if let Some(pod) = &obj {
            if let Some(status) = &pod.status {
                if let Some(reason) = &status.reason {
                    return reason == "Evicted";
                }
            }
        }
        false
    }
}
