pub mod config;
pub use config::{Config, DeploymentFramework, ModelConfig, ResourceSpec, TestSpec};
pub mod runner;
use evaluate::EvaluationResult;
pub use runner::{Run, Runner};
pub mod evaluate;
pub mod route;
pub mod utils;
pub mod worker;
pub mod pb {
    //! This module contains protobuf generated types.
    tonic::include_proto!("fmaas");
}

use std::{fs::File, io::BufReader, path::Path, sync::Arc};

use serde::{Deserialize, Serialize};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Runner(#[from] crate::runner::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),
    #[error(transparent)]
    Kube(#[from] kube::Error),
    #[error(transparent)]
    ObjectStore(#[from] object_store::Error),
    #[error("{0}")]
    InvalidConfig(String),
    #[error("{0}")]
    InvalidTestCase(String),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    TestFailed(String),
    #[error("{0}")]
    Validation(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TgisRequest {
    Unary(BatchedGenerationRequest),
    Streaming(SingleGenerationRequest),
}

impl TgisRequest {
    pub fn prefix_id(&self) -> Option<&str> {
        match self {
            TgisRequest::Unary(req) => req.prefix_id.as_deref(),
            TgisRequest::Streaming(req) => req.prefix_id.as_deref(),
        }
    }

    pub fn params(&self) -> Option<&Parameters> {
        match self {
            TgisRequest::Unary(req) => req.params.as_ref(),
            TgisRequest::Streaming(req) => req.params.as_ref(),
        }
    }
}

impl From<BatchedGenerationRequest> for TgisRequest {
    fn from(value: BatchedGenerationRequest) -> Self {
        TgisRequest::Unary(value)
    }
}

impl From<SingleGenerationRequest> for TgisRequest {
    fn from(value: SingleGenerationRequest) -> Self {
        TgisRequest::Streaming(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TgisResponse {
    Unary(BatchedGenerationResponse),
    Streaming(Vec<GenerationResponse>),
}

impl TgisResponse {
    pub fn inner(&self) -> &[GenerationResponse] {
        match self {
            TgisResponse::Unary(resp) => &resp.responses,
            TgisResponse::Streaming(resp) => resp,
        }
    }

    pub fn into_inner(self) -> Vec<GenerationResponse> {
        match self {
            TgisResponse::Unary(resp) => resp.responses,
            TgisResponse::Streaming(resp) => resp,
        }
    }
}

impl From<BatchedGenerationResponse> for TgisResponse {
    fn from(value: BatchedGenerationResponse) -> Self {
        Self::Unary(value)
    }
}

impl From<Vec<GenerationResponse>> for TgisResponse {
    fn from(value: Vec<GenerationResponse>) -> Self {
        Self::Streaming(value)
    }
}

/// Represents a single test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    #[serde(skip)]
    pub model_name: Option<ModelName>,
    #[serde(skip)]
    pub model_config: Option<Arc<ModelConfig>>,
    pub name: String,
    #[serde(default)]
    pub single_shard_only: bool,
    #[serde(default)]
    pub skip_check: bool,
    pub request: TgisRequest,
    #[serde(rename = "response")]
    pub expected_response: Option<TgisResponse>,
    #[serde(rename = "error")]
    pub expected_error: Option<Status>,
}

impl TestCase {
    pub fn expected_result(&self) -> Result<TgisResponse, Status> {
        match (&self.expected_error, &self.expected_response) {
            (None, Some(expected_response)) => Ok(expected_response.clone()),
            (Some(expected_error), None) => Err(expected_error.clone()),
            _ => unimplemented!(),
        }
    }
}

/// Reads test case files.
pub struct TestCaseReader {
    reader: BufReader<File>,
}

impl TestCaseReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let reader = BufReader::new(File::open(path)?);
        Ok(Self { reader })
    }

    pub fn read(self) -> Result<Vec<TestCase>, Error> {
        let cases: Vec<TestCase> = serde_yaml::from_reader(self.reader)?;
        Ok(cases)
    }
}

/// Represents a single test case result.
#[derive(Debug, Clone, Serialize)]
pub struct TestCaseResult {
    #[serde(skip)]
    pub model_name: ModelName,
    #[serde(skip)]
    pub model_config: Arc<ModelConfig>,
    pub name: String,
    pub skip_check: bool,
    pub passed: bool,
    pub request: TgisRequest,
    pub expected_result: Result<TgisResponse, Status>,
    pub result: Result<TgisResponse, Status>,
    pub evaluation_result: Option<EvaluationResult>,
}

impl TestCaseResult {
    pub fn passed(&self) -> bool {
        self.passed
    }

    pub fn group(&self) -> String {
        format!(
            "{}_{}_{}",
            &self.model_name.short_name(),
            &self.model_config.deployment_framework.short_name(),
            &self.model_config.dtype.short_name(),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelName(String);

impl ModelName {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn short_name(&self) -> String {
        let name = self.0.as_str();
        let name = if name.contains('/') {
            let parts: Vec<&str> = name.split('/').collect();
            parts[parts.len() - 1]
        } else {
            name
        };
        let name = name.replace('.', "");
        let mut name = {
            let parts: Vec<&str> = name.split(['-', '_']).collect();
            parts
                .iter()
                .enumerate()
                .map(|(i, &part)| {
                    if i == 0 {
                        utils::truncate(part, 5)
                    } else {
                        utils::truncate(part, 4)
                    }
                })
                .collect::<String>()
        };
        name.truncate(11);
        name.to_lowercase()
    }

    pub fn pretty_name(&self) -> String {
        let name = self.0.as_str();
        let name = if name.contains('/') {
            let parts: Vec<&str> = name.split('/').collect();
            parts[parts.len() - 1]
        } else {
            name
        };
        name.replace(['-'], "_").to_lowercase()
    }
}

impl std::ops::Deref for ModelName {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Display for ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Status {
    pub code: Code,
    pub message: String,
}

impl From<tonic::Status> for Status {
    fn from(value: tonic::Status) -> Self {
        Self {
            code: value.code().into(),
            message: value.message().to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Code {
    Ok = 0,
    Cancelled = 1,
    Unknown = 2,
    InvalidArgument = 3,
    DeadlineExceeded = 4,
    NotFound = 5,
    AlreadyExists = 6,
    PermissionDenied = 7,
    ResourceExhausted = 8,
    FailedPrecondition = 9,
    Aborted = 10,
    OutOfRange = 11,
    Unimplemented = 12,
    Internal = 13,
    Unavailable = 14,
    DataLoss = 15,
    Unauthenticated = 16,
}

impl From<tonic::Code> for Code {
    fn from(value: tonic::Code) -> Self {
        use tonic::Code::*;
        match value {
            Ok => Code::Ok,
            Cancelled => Code::Cancelled,
            Unknown => Code::Unknown,
            InvalidArgument => Code::InvalidArgument,
            DeadlineExceeded => Code::DeadlineExceeded,
            NotFound => Code::NotFound,
            AlreadyExists => Code::AlreadyExists,
            PermissionDenied => Code::PermissionDenied,
            ResourceExhausted => Code::ResourceExhausted,
            FailedPrecondition => Code::FailedPrecondition,
            Aborted => Code::Aborted,
            OutOfRange => Code::OutOfRange,
            Unimplemented => Code::Unimplemented,
            Internal => Code::Internal,
            Unavailable => Code::Unavailable,
            DataLoss => Code::DataLoss,
            Unauthenticated => Code::Unauthenticated,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchedGenerationRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Parameters>,
    pub requests: Vec<GenerationRequest>,
}

impl BatchedGenerationRequest {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }
}

impl From<BatchedGenerationRequest> for pb::BatchedGenerationRequest {
    fn from(value: BatchedGenerationRequest) -> Self {
        Self {
            model_id: "".to_string(), // TBD: how is this used?
            prefix_id: value.prefix_id,
            adapter_id: value.adapter_id,
            requests: value.requests.into_iter().map(|v| v.into()).collect(),
            params: value.params.map(|v| v.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub text: String,
}

impl From<GenerationRequest> for pb::GenerationRequest {
    fn from(value: GenerationRequest) -> Self {
        Self { text: value.text }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SingleGenerationRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Parameters>,
    pub request: GenerationRequest,
}

impl From<SingleGenerationRequest> for pb::SingleGenerationRequest {
    fn from(value: SingleGenerationRequest) -> Self {
        Self {
            model_id: "".to_string(),
            prefix_id: value.prefix_id,
            adapter_id: value.adapter_id,
            request: Some(value.request.into()),
            params: value.params.map(|v| v.into()),
        }
    }
}

impl TryFrom<BatchedGenerationRequest> for SingleGenerationRequest {
    type Error = Error;
    fn try_from(value: BatchedGenerationRequest) -> Result<Self, Self::Error> {
        if value.len() == 1 {
            Ok(Self {
                prefix_id: value.prefix_id,
                adapter_id: value.adapter_id,
                request: value.requests[0].clone(),
                params: value.params.clone(),
            })
        } else {
            Err(Error::Invalid("must contain 1 request".into()))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchedGenerationResponse {
    pub responses: Vec<GenerationResponse>,
}

impl BatchedGenerationResponse {
    pub fn len(&self) -> usize {
        self.responses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &GenerationResponse> {
        self.responses.iter()
    }
}

impl From<pb::BatchedGenerationResponse> for BatchedGenerationResponse {
    fn from(value: pb::BatchedGenerationResponse) -> Self {
        Self {
            responses: value.responses.into_iter().map(|v| v.into()).collect(),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationResponse {
    pub input_token_count: u32,
    #[serde(default)]
    pub generated_token_count: u32,
    #[serde(default)]
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    #[serde(default)]
    pub stop_sequence: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<TokenInfo>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<Vec<TokenInfo>>,
}

impl From<pb::GenerationResponse> for GenerationResponse {
    fn from(value: pb::GenerationResponse) -> Self {
        let seed = if value.seed == 0 {
            None
        } else {
            Some(value.seed)
        };
        let tokens = if value.tokens.is_empty() {
            None
        } else {
            Some(value.tokens.into_iter().map(|v| v.into()).collect())
        };
        let input_tokens = if value.input_tokens.is_empty() {
            None
        } else {
            Some(value.input_tokens.into_iter().map(|v| v.into()).collect())
        };
        let stop_reason = if value.stop_reason == 0 {
            None
        } else {
            Some(StopReason::from_i32(value.stop_reason).unwrap())
        };
        Self {
            input_token_count: value.input_token_count,
            generated_token_count: value.generated_token_count,
            text: value.text,
            stop_reason,
            stop_sequence: value.stop_sequence,
            seed,
            tokens,
            input_tokens,
        }
    }
}

impl From<GenerationResponse> for BatchedGenerationResponse {
    fn from(value: GenerationResponse) -> Self {
        BatchedGenerationResponse {
            responses: vec![value],
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Parameters {
    #[serde(default = "default_decoding_method")]
    pub method: DecodingMethod,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingParameters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopping: Option<StoppingCriteria>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<ResponseOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoding: Option<DecodingParameters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate_input_tokens: Option<u32>,
}

impl From<Parameters> for pb::Parameters {
    fn from(value: Parameters) -> Self {
        Self {
            method: value.method.value(),
            sampling: value.sampling.map(|v| v.into()),
            stopping: value.stopping.map(|v| v.into()),
            response: value.response.map(|v| v.into()),
            decoding: value.decoding.map(|v| v.into()),
            truncate_input_tokens: value.truncate_input_tokens.unwrap_or_default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DecodingMethod {
    Greedy = 0,
    Sample = 1,
}

impl DecodingMethod {
    pub fn value(&self) -> i32 {
        use DecodingMethod::*;
        match self {
            Greedy => 0,
            Sample => 1,
        }
    }
}

impl Default for DecodingMethod {
    fn default() -> Self {
        Self::Greedy
    }
}

impl From<DecodingMethod> for pb::DecodingMethod {
    fn from(value: DecodingMethod) -> Self {
        use DecodingMethod::*;
        match value {
            Greedy => pb::DecodingMethod::Greedy,
            Sample => pb::DecodingMethod::Sample,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl From<SamplingParameters> for pb::SamplingParameters {
    fn from(value: SamplingParameters) -> Self {
        pb::SamplingParameters {
            temperature: value.temperature.unwrap_or_default(),
            top_k: value.top_k.unwrap_or_default(),
            top_p: value.top_p.unwrap_or_default(),
            typical_p: value.typical_p.unwrap_or_default(),
            seed: value.seed,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoppingCriteria {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_limit_millis: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_stop_sequence: Option<bool>,
}

impl From<StoppingCriteria> for pb::StoppingCriteria {
    fn from(value: StoppingCriteria) -> Self {
        Self {
            max_new_tokens: value.max_new_tokens.unwrap_or_default(),
            min_new_tokens: value.min_new_tokens.unwrap_or_default(),
            time_limit_millis: value.time_limit_millis.unwrap_or_default(),
            stop_sequences: value.stop_sequences.unwrap_or_default(),
            include_stop_sequence: value.include_stop_sequence,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_text: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_tokens: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ranks: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n_tokens: Option<u32>,
}

impl From<ResponseOptions> for pb::ResponseOptions {
    fn from(value: ResponseOptions) -> Self {
        Self {
            input_text: value.input_text.unwrap_or_default(),
            generated_tokens: value.generated_tokens.unwrap_or_default(),
            input_tokens: value.input_tokens.unwrap_or_default(),
            token_logprobs: value.token_logprobs.unwrap_or_default(),
            token_ranks: value.token_ranks.unwrap_or_default(),
            top_n_tokens: value.top_n_tokens.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodingParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub length_penalty: Option<LengthPenalty>,
    // TODO: guided oneof
}

impl From<DecodingParameters> for pb::DecodingParameters {
    fn from(value: DecodingParameters) -> Self {
        Self {
            repetition_penalty: value.repetition_penalty.unwrap_or_default(),
            length_penalty: value.length_penalty.map(|v| v.into()),
            guided: None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LengthPenalty {
    pub start_index: u32,
    pub decay_factor: f32,
}

impl From<LengthPenalty> for pb::decoding_parameters::LengthPenalty {
    fn from(value: LengthPenalty) -> Self {
        Self {
            start_index: value.start_index,
            decay_factor: value.decay_factor,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenInfo {
    pub text: String,
    #[serde(
        default,
        deserialize_with = "deserialize_maybe_nan",
        skip_serializing_if = "Option::is_none"
    )]
    pub logprob: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_tokens: Option<Vec<TopToken>>,
}

impl From<pb::TokenInfo> for TokenInfo {
    fn from(value: pb::TokenInfo) -> Self {
        let logprob = if value.logprob.is_nan() || value.logprob == 0.0 {
            None
        } else {
            Some(value.logprob)
        };
        let top_tokens = if value.top_tokens.is_empty() {
            None
        } else {
            Some(value.top_tokens.into_iter().map(|v| v.into()).collect())
        };
        let rank = if value.rank == 0 {
            None
        } else {
            Some(value.rank)
        };
        Self {
            text: value.text,
            logprob,
            rank,
            top_tokens,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopToken {
    pub text: String,
    #[serde(
        default,
        deserialize_with = "deserialize_maybe_nan",
        skip_serializing_if = "Option::is_none"
    )]
    pub logprob: Option<f32>,
}

impl From<pb::token_info::TopToken> for TopToken {
    fn from(value: pb::token_info::TopToken) -> Self {
        let logprob = if value.logprob.is_nan() || value.logprob == 0.0 {
            None
        } else {
            Some(value.logprob)
        };
        Self {
            text: value.text,
            logprob,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputToken {
    pub logprob: f64,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StopReason {
    NotFinished,
    MaxTokens,
    EosToken,
    Cancelled,
    TimeLimit,
    StopSequence,
    TokenLimit,
    Error,
}

impl StopReason {
    pub fn from_i32(value: i32) -> Result<Self, Error> {
        match value {
            0 => Ok(Self::NotFinished),
            1 => Ok(Self::MaxTokens),
            2 => Ok(Self::EosToken),
            3 => Ok(Self::Cancelled),
            4 => Ok(Self::TimeLimit),
            5 => Ok(Self::StopSequence),
            6 => Ok(Self::TokenLimit),
            7 => Ok(Self::Error),
            _ => unimplemented!(),
        }
    }
}

impl From<pb::StopReason> for StopReason {
    fn from(value: pb::StopReason) -> Self {
        use pb::StopReason::*;
        match value {
            NotFinished => Self::NotFinished,
            MaxTokens => Self::MaxTokens,
            EosToken => Self::EosToken,
            Cancelled => Self::Cancelled,
            TimeLimit => Self::TimeLimit,
            StopSequence => Self::StopSequence,
            TokenLimit => Self::TokenLimit,
            Error => Self::Error,
        }
    }
}

fn default_decoding_method() -> DecodingMethod {
    DecodingMethod::Greedy
}

use serde::de::Deserializer;
fn deserialize_maybe_nan<'de, D, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum MaybeNA<U> {
        Value(Option<U>),
        NaNString(String),
    }
    let value: MaybeNA<T> = Deserialize::deserialize(deserializer)?;
    match value {
        MaybeNA::Value(value) => Ok(value),
        MaybeNA::NaNString(value) => {
            if value == "NaN" {
                Ok(None) // or Ok(Some(f32::NAN))
            } else {
                Err(serde::de::Error::custom("unexpected string"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ModelName;

    #[test]
    fn test_model_short_name() {
        assert_eq!(
            ModelName::new("hf-tiny-model-private/tiny-random-GPTNeoXForCausalLM").short_name(),
            "tinyrandgpt"
        );
        assert_eq!(
            ModelName::new("yujiepan/falcon-tiny-random").short_name(),
            "falcotinyra"
        );
        assert_eq!(
            ModelName::new("integration_tests/models/llama-2-7b-gptq").short_name(),
            "llama27bgpt"
        );
        assert_eq!(
            ModelName::new("bigcode/tiny_starcoder_py").short_name(),
            "tinystarpy"
        );
        assert_eq!(
            ModelName::new("google/flan-t5-small").short_name(),
            "flant5smal"
        );
        assert_eq!(
            ModelName::new("bigscience/bloom-560m").short_name(),
            "bloom560m"
        );
        assert_eq!(
            ModelName::new("/inter_ckpts/granite-8b-2lang-instruct-v1/granite-8b-jp-galpha-0111")
                .short_name(),
            "grani8bjpga"
        );
    }
}
