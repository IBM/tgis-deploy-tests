mod approx_equality;
pub use approx_equality::ApproxEquality;
mod exact_text_match;
pub use exact_text_match::ExactTextMatch;
mod embeddings_similarity;
pub use embeddings_similarity::EmbeddingsSimilarity;
mod metrics;
pub use metrics::Metric;
mod multiple;
pub use multiple::Multiple;
mod streaming;
use streaming::{merge_streaming_response, valid_streaming_response};

use serde::Serialize;

use crate::{GenerationResponse, Status, TestCase, TestCaseResult, TgisResponse};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("tokenizers error: {0}")]
    Tokenizers(#[from] tokenizers::Error),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("hf hub error: {0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid error: {0}")]
    Invalid(String),
}

/// An interface to implement evaluation strategies.
pub trait EvaluationStrategy {
    fn name(&self) -> &'static str;
    fn evaluate(
        &self,
        expected: &[GenerationResponse],
        actual: &[GenerationResponse],
    ) -> Result<EvaluationResult, Error>;
}

/// Evaluates test cases using an [`EvaluationStrategy`].
pub struct Evaluator;

impl Evaluator {
    pub fn evaluate(
        strategy: &dyn EvaluationStrategy,
        case: &TestCase,
        result: Result<TgisResponse, Status>,
    ) -> Result<TestCaseResult, Error> {
        let expected_result = case.expected_result();
        let (passed, evaluation_result) = match (&expected_result, &result) {
            // Success case
            (Ok(expected_response), Ok(response)) => match (expected_response, response) {
                // Unary response: evaluate
                (TgisResponse::Unary(expected), TgisResponse::Unary(actual)) => {
                    if case.skip_check {
                        (true, None)
                    } else {
                        let evaluation_result =
                            strategy.evaluate(&expected.responses, &actual.responses)?;
                        (evaluation_result.passed(), Some(evaluation_result))
                    }
                }
                // Streaming response: validate stream, merge, and evaluate
                (TgisResponse::Unary(expected), TgisResponse::Streaming(actual)) => {
                    if case.skip_check {
                        (true, None)
                    } else if valid_streaming_response(case, actual) {
                        // Hacky workaround for expected input_text ending with \n\n for encoder-decoder models
                        let includes_input_text = case
                            .request
                            .params()
                            .map(|p| {
                                p.response
                                    .as_ref()
                                    .is_some_and(|r| r.input_text == Some(true))
                            })
                            .unwrap_or_default();
                        let expected =
                            if includes_input_text && expected.responses[0].text.contains("\n\n") {
                                let mut expected = expected.clone();
                                expected.responses[0].text =
                                    expected.responses[0].text.replace("\n\n", "");
                                expected
                            } else {
                                expected.clone()
                            };
                        let actual_merged = merge_streaming_response(actual);
                        let evaluation_result =
                            strategy.evaluate(&expected.responses, &[actual_merged])?;
                        (evaluation_result.passed(), Some(evaluation_result))
                    } else {
                        (
                            false,
                            Some(EvaluationResult {
                                strategy: "streaming".to_string(),
                                passed: false,
                                metrics: None,
                            }),
                        )
                    }
                }
                _ => unimplemented!(),
            },
            // Error case
            (Err(expected_error), Err(error)) => (expected_error == error, None),
            _ => (false, None),
        };
        Ok(TestCaseResult {
            model_name: case.model_name.as_ref().unwrap().clone(),
            model_config: case.model_config.as_ref().unwrap().clone(),
            name: case.name.clone(),
            skip_check: case.skip_check,
            passed,
            request: case.request.clone(),
            expected_result,
            result: result.clone(),
            evaluation_result,
        })
    }
}

/// The result of an applied [`EvaluationStrategy`].
#[derive(Debug, Clone, Serialize)]
pub struct EvaluationResult {
    pub strategy: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<(String, f32)>>,
}

impl EvaluationResult {
    pub fn new(strategy: &str, passed: bool, metrics: Option<Vec<(String, f32)>>) -> Self {
        Self {
            strategy: strategy.to_string(),
            passed,
            metrics,
        }
    }

    pub fn passed(&self) -> bool {
        self.passed
    }

    pub fn metrics(&self) -> Option<&[(String, f32)]> {
        self.metrics.as_deref()
    }
}
