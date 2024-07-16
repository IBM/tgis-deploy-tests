use std::{collections::HashSet, path::PathBuf};

use secrecy::Secret;
use serde::{Deserialize, Serialize};

use crate::{evaluate, Error, ModelName, TestCase, TestCaseReader};

const DEFAULT_NUM_SHARD: usize = 1;
const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 96;
const DEFAULT_MAX_SEQUENCE_LENGTH: usize = 2048;
const DEFAULT_MAX_NEW_TOKENS: usize = 1024;
const DEFAULT_MAX_BATCH_SIZE: usize = 8;
const DEFAULT_MAX_WAITING_TOKENS: usize = 10;
const DEFAULT_MASTER_ADDR: &str = "localhost";
const DEFAULT_MASTER_PORT: u16 = 29500;
const DEFAULT_PORT: u16 = 5001;
const DEFAULT_GRPC_PORT: u16 = 8001;
const DEFAULT_CUDA_PROCESS_MEMORY_FRACTION: f32 = 1.0;
const DEFAULT_REQUEST_CPU: u16 = 2;
const DEFAULT_REQUEST_MEMORY_GB: u16 = 4;
const DEFAULT_LIMIT_CPU: u16 = 4;
const DEFAULT_LIMIT_MEMORY_GB: u16 = 96;
const DEFAULT_GPU_MEMORY_UTILIZATION: f32 = 0.85;
const DEFAULT_ENABLE_LORA: bool = false;

/// Test run configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub cluster: ClusterConfig,
    pub worker: WorkerConfig,
    pub runner: RunnerConfig,
    pub tests: Vec<TestSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClusterConfig {
    pub api_endpoint: String,
    pub namespace: String,
    pub service_account_name: String,
    #[serde(default = "service_account_token")]
    pub service_account_token: Option<Secret<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkerConfig {
    pub image_uri: String,
    pub image_pull_secret: Option<String>,
    pub tls_secret: String,
    pub ready_timeout_secs: u64,
    pub scheduled_timeout_secs: u64,
    pub max_parallel_workers: usize,
    pub max_connection_retries: usize,
    pub ingress_hostname: String,
    pub pvcs: Vec<Pvc>,
    pub transformers_cache_path: String,
    #[serde(default)]
    pub prefix_store_path: String,
    #[serde(default)]
    pub adapter_cache: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RunnerConfig {
    pub ca_cert_path: Option<PathBuf>,
    pub object_store_secret: String,
    pub object_store_bucket: String, // TODO: add bucket/endpoint/region to secret?
    pub object_store_endpoint: String,
    pub object_store_region: String,
    #[serde(default)]
    pub validate_models: bool,
    #[serde(default)]
    pub generate_stream: bool,
    #[serde(default)]
    pub evaluation_strategy: EvaluationStrategyConfig,
    #[serde(default)]
    pub save_passed_test_results: bool,
}

impl Config {
    pub fn load(reader: impl std::io::Read) -> Result<Self, Error> {
        serde_yaml::from_reader(reader).map_err(|e| Error::InvalidConfig(e.to_string()))
    }

    pub fn validate(&self) -> Result<(), Error> {
        for test in self.tests.iter() {
            test.validate()?
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationStrategyConfig {
    ApproxEquality,
    ExactTextMatch,
    EmbeddingsSimilarity,
}

impl Default for EvaluationStrategyConfig {
    fn default() -> Self {
        Self::ApproxEquality
    }
}

impl From<EvaluationStrategyConfig> for Box<dyn evaluate::EvaluationStrategy> {
    fn from(value: EvaluationStrategyConfig) -> Self {
        use EvaluationStrategyConfig::*;
        match value {
            ApproxEquality => Box::<evaluate::ApproxEquality>::default(),
            ExactTextMatch => Box::<evaluate::ExactTextMatch>::default(),
            EmbeddingsSimilarity => Box::<evaluate::EmbeddingsSimilarity>::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Pvc {
    pub name: String,
    pub mount_path: String,
    #[serde(default)]
    pub read_only: bool,
}

/// Model test specification.
#[derive(Debug, Clone, Deserialize)]
pub struct TestSpec {
    /// Model name
    #[serde(rename = "model")]
    pub model_name: ModelName,
    /// Model configurations to run
    pub configs: Vec<ModelConfig>,
    /// Paths to test cases to run for each configuration
    pub cases: Vec<PathBuf>,
}

impl TestSpec {
    pub fn validate(&self) -> Result<(), Error> {
        for config in &self.configs {
            config
                .validate()
                .map_err(|e| Error::Validation(format!("{}: {e}", config.name())))?
        }
        Ok(())
    }

    pub fn load_cases(&self) -> Result<Vec<TestCase>, Error> {
        let cases: Result<Vec<_>, _> = self
            .cases
            .iter()
            .map(|path| TestCaseReader::open(path)?.read())
            .collect();
        match cases {
            Ok(cases) => Ok(cases.into_iter().flatten().collect()),
            Err(error) => Err(error),
        }
    }
}

/// TGIS model configuration and test options.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub revision: Option<String>,
    #[serde(default = "default_deployment_framework")]
    pub deployment_framework: DeploymentFramework,
    #[serde(default, rename = "dtype_str")]
    pub dtype: DataType,
    #[serde(default = "default_num_shard")]
    pub num_shard: usize,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_grpc_port")]
    pub grpc_port: u16,
    #[serde(default = "default_master_addr")]
    pub master_addr: String,
    #[serde(default = "default_master_port")]
    pub master_port: u16,
    #[serde(default = "default_cuda_process_memory_fraction")]
    pub cuda_process_memory_fraction: f32,
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "default_max_waiting_tokens")]
    pub max_waiting_tokens: usize,
    pub max_batch_weight: Option<usize>,
    pub max_prefill_weight: Option<usize>,
    #[serde(default)]
    pub output_special_tokens: bool,
    #[serde(default)]
    pub flash_attention: bool,
    #[serde(default)]
    pub pt2_compile: bool,
    pub quantize: Option<Quantize>,
    #[serde(default = "default_resources")]
    pub resources: ResourceSpec,
    // vLLM only
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f32,
    #[serde(default = "default_enable_lora")]
    pub enable_lora: bool,
    pub load_format: Option<String>,
    pub model_loader_extra_config: Option<String>,
    // onnx runtime (hf_optimum_ort) only
    pub merge_onnx_graphs: Option<bool>,
    pub estimate_memory: Option<bool>,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), Error> {
        if self.flash_attention {
            let valid_gpu = self
                .resources
                .gpu
                .as_ref()
                .is_some_and(|gpu| gpu.count() >= 1);
            if !valid_gpu {
                return Err(Error::Validation(
                    "flash attention requires `n_gpu` >= 1".into(),
                ));
            }
            let valid_dtype = matches!(self.dtype, DataType::Float16 | DataType::BFloat16);
            if !valid_dtype {
                return Err(Error::Validation(
                    "flash attention requires `float16` or `bfloat16` dtype".into(),
                ));
            }
            let valid_framework =
                matches!(self.deployment_framework, DeploymentFramework::TgisNative);
            if !valid_framework {
                return Err(Error::Validation(
                    "flash attention requires `tgis_native` deployment framework".into(),
                ));
            }
        }
        if self.pt2_compile {
            let valid_framework = matches!(
                self.deployment_framework,
                DeploymentFramework::HfTransformers
            );
            if !valid_framework {
                return Err(Error::Validation(
                    "pt2_compile requires `hf_transformers` deployment framework".into(),
                ));
            }
        }
        if self.quantize.is_some() {
            let valid_framework =
                matches!(self.deployment_framework, DeploymentFramework::TgisNative);
            if !valid_framework {
                return Err(Error::Validation(
                    "quantize requires `tgis_native` deployment framework".into(),
                ));
            }
        }
        if self.is_cpu_only() {
            let valid_dtype = matches!(self.dtype, DataType::Float32);
            if !valid_dtype {
                return Err(Error::Validation(
                    "cpu only requires `float32` dtype".into(),
                ));
            }
        }
        if self.is_sharded() {
            // MIG not currently supported
            let valid_gpu = self.resources.gpu.as_ref().is_some_and(|gpu| {
                matches!(gpu, GpuSpec::Gpu(_)) && gpu.count() == self.num_shard as u16
            });
            if !valid_gpu {
                return Err(Error::Validation(
                    "sharded requires non-MIG GPU and `num_shard` to be equal to `n_gpu`".into(),
                ));
            }
        }
        let unique_ports = [self.port, self.grpc_port, self.master_port]
            .into_iter()
            .collect::<HashSet<_>>();
        let valid_ports = unique_ports.len() == 3;
        if !valid_ports {
            return Err(Error::Validation("ports specified must be unique".into()));
        }
        let valid_cuda_process_memory_fraction = self.cuda_process_memory_fraction <= 1.0;
        if !valid_cuda_process_memory_fraction {
            return Err(Error::Validation(
                "`cuda_process_memory_fraction` must be <= 1.0".into(),
            ));
        }
        Ok(())
    }

    pub fn dtype_str(&self) -> &str {
        self.dtype.as_str()
    }

    pub fn name(&self) -> String {
        let mut name = format!(
            "{}|{}",
            self.deployment_framework.as_str(),
            self.dtype.short_name()
        );
        if self.merge_onnx_graphs.is_some() {
            name.push_str("|merge");
        }
        if self.is_sharded() {
            name.push_str(&format!("|sharded({})", self.num_shard));
        }
        if self.is_cpu_only() {
            name.push_str("|cpu");
        }
        if self.pt2_compile {
            name.push_str("|pt2c")
        }
        if self.flash_attention {
            name.push_str("|flash")
        }
        if let Some(quantize) = &self.quantize {
            match quantize {
                Quantize::Gptq => name.push_str("|gptq"),
            }
        }
        name
    }

    pub fn is_sharded(&self) -> bool {
        self.num_shard > 1
    }

    pub fn is_cpu_only(&self) -> bool {
        self.resources.gpu.is_none()
            | self
                .resources
                .gpu
                .as_ref()
                .is_some_and(|gpu| gpu.count() == 0)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub enum DataType {
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "float32")]
    Float32,
    #[serde(rename = "bfloat16")]
    BFloat16,
}

impl DataType {
    pub fn as_str(&self) -> &str {
        use DataType::*;
        match *self {
            Float16 => "float16",
            Float32 => "float32",
            BFloat16 => "bfloat16",
        }
    }

    pub fn short_name(&self) -> &str {
        use DataType::*;
        match *self {
            Float16 => "f16",
            Float32 => "f32",
            BFloat16 => "bf16",
        }
    }
}

impl Default for DataType {
    fn default() -> Self {
        Self::Float16
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::fmt::Debug for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentFramework {
    HfTransformers,
    HfAccelerate,
    HfOptimumBt,
    HfOptimumOrt,
    TgisNative,
    IbmFms,
    GptMegatron,
}

impl DeploymentFramework {
    pub fn as_str(&self) -> &str {
        use DeploymentFramework::*;
        match *self {
            HfTransformers => "hf_transformers",
            HfAccelerate => "hf_accelerate",
            HfOptimumBt => "hf_optimum_bt",
            HfOptimumOrt => "hf_optimum_ort",
            TgisNative => "tgis_native",
            IbmFms => "ibm_fms",
            GptMegatron => "gpt_megatron",
        }
    }

    pub fn short_name(&self) -> &str {
        use DeploymentFramework::*;
        match *self {
            HfTransformers => "t",
            HfAccelerate => "a",
            HfOptimumBt => "ob",
            HfOptimumOrt => "or",
            TgisNative => "n",
            IbmFms => "f",
            GptMegatron => "m",
        }
    }
}

impl std::fmt::Display for DeploymentFramework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::fmt::Debug for DeploymentFramework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub enum Quantize {
    #[serde(rename = "gptq")]
    Gptq,
}

impl Quantize {
    pub fn as_str(&self) -> &str {
        use Quantize::*;
        match *self {
            Gptq => "gptq",
        }
    }
}

impl std::fmt::Display for Quantize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::fmt::Debug for Quantize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Test worker pod resource request/limits specification.
#[derive(Debug, Clone, Deserialize)]
pub struct ResourceSpec {
    #[serde(default = "default_request_cpu")]
    pub request_cpu: u16,
    #[serde(default = "default_request_memory_gb")]
    pub request_memory_gb: u16,
    #[serde(default = "default_limit_cpu")]
    pub limit_cpu: u16,
    #[serde(default = "default_limit_memory_gb")]
    pub limit_memory_gb: u16,
    #[serde(default = "default_gpu_spec")]
    pub gpu: Option<GpuSpec>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum GpuSpec {
    #[serde(rename = "gpu")]
    Gpu(u16),
    #[serde(rename = "mig")]
    Mig(Mig),
}

impl Default for GpuSpec {
    fn default() -> Self {
        Self::Gpu(1)
    }
}

impl GpuSpec {
    pub fn count(&self) -> u16 {
        match self {
            GpuSpec::Gpu(n) => *n,
            GpuSpec::Mig(mig) => match mig {
                Mig::Mig1g10gb(n) | Mig::Mig2g20gb(n) | Mig::Mig3g40gb(n) => *n,
            },
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub enum Mig {
    #[serde(rename = "mig_1g_10gb")]
    Mig1g10gb(u16),
    #[serde(rename = "mig_2g_20gb")]
    Mig2g20gb(u16),
    #[serde(rename = "mig_3g_40gb")]
    Mig3g40gb(u16),
}

impl Default for ResourceSpec {
    fn default() -> Self {
        Self {
            request_cpu: DEFAULT_REQUEST_CPU,
            request_memory_gb: DEFAULT_REQUEST_MEMORY_GB,
            limit_cpu: DEFAULT_LIMIT_CPU,
            limit_memory_gb: DEFAULT_LIMIT_MEMORY_GB,
            gpu: Some(GpuSpec::default()),
        }
    }
}

fn default_resources() -> ResourceSpec {
    ResourceSpec::default()
}

fn default_gpu_spec() -> Option<GpuSpec> {
    Some(GpuSpec::default())
}

fn default_request_cpu() -> u16 {
    DEFAULT_REQUEST_CPU
}

fn default_request_memory_gb() -> u16 {
    DEFAULT_REQUEST_MEMORY_GB
}

fn default_limit_cpu() -> u16 {
    DEFAULT_LIMIT_CPU
}

fn default_limit_memory_gb() -> u16 {
    DEFAULT_LIMIT_MEMORY_GB
}

fn default_max_concurrent_requests() -> usize {
    DEFAULT_MAX_CONCURRENT_REQUESTS
}

fn default_max_sequence_length() -> usize {
    DEFAULT_MAX_SEQUENCE_LENGTH
}

fn default_max_new_tokens() -> usize {
    DEFAULT_MAX_NEW_TOKENS
}

fn default_max_waiting_tokens() -> usize {
    DEFAULT_MAX_WAITING_TOKENS
}

fn default_max_batch_size() -> usize {
    DEFAULT_MAX_BATCH_SIZE
}

fn default_master_addr() -> String {
    DEFAULT_MASTER_ADDR.to_string()
}

fn default_master_port() -> u16 {
    DEFAULT_MASTER_PORT
}

fn default_num_shard() -> usize {
    DEFAULT_NUM_SHARD
}

fn default_port() -> u16 {
    DEFAULT_PORT
}

fn default_grpc_port() -> u16 {
    DEFAULT_GRPC_PORT
}

fn default_cuda_process_memory_fraction() -> f32 {
    DEFAULT_CUDA_PROCESS_MEMORY_FRACTION
}

fn default_deployment_framework() -> DeploymentFramework {
    DeploymentFramework::TgisNative
}

fn default_gpu_memory_utilization() -> f32 {
    DEFAULT_GPU_MEMORY_UTILIZATION
}

fn default_enable_lora() -> bool {
    DEFAULT_ENABLE_LORA
}

fn service_account_token() -> Option<Secret<String>> {
    std::env::var("TESTER_SERVICE_ACCOUNT_TOKEN")
        .ok()
        .map(Secret::new)
}
