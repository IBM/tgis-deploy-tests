use crate::GenerationResponse;

use super::{Error, EvaluationResult, EvaluationStrategy};

/// Evaluates true equality on generated text strings, disregarding other items.
#[derive(Default)]
pub struct ExactTextMatch {}

impl ExactTextMatch {
    pub fn new(&self) -> Self {
        Self::default()
    }
}

impl EvaluationStrategy for ExactTextMatch {
    fn name(&self) -> &'static str {
        "exact_text_match"
    }

    fn evaluate(
        &self,
        expected: &[GenerationResponse],
        actual: &[GenerationResponse],
    ) -> Result<EvaluationResult, Error> {
        let passed = actual
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| a.text == b.text);
        Ok(EvaluationResult::new(self.name(), passed, None))
    }
}

#[cfg(test)]
mod tests {
    use crate::{GenerationResponse, StopReason};

    use super::*;

    #[test]
    fn exact_text_should_pass() {
        let expected = vec![GenerationResponse {
            input_token_count: 10,
            generated_token_count: 2,
            text: "positive".to_string(),
            stop_reason: Some(StopReason::EosToken),
            stop_sequence: Default::default(),
            seed: None,
            tokens: None,
            input_tokens: None,
        }];
        let actual = &expected;
        let strategy = ExactTextMatch::default();
        let evaluation_result = strategy.evaluate(&expected, actual).unwrap();
        assert!(evaluation_result.passed())
    }

    #[test]
    fn different_text_should_fail() {
        let expected = vec![GenerationResponse {
            input_token_count: 10,
            generated_token_count: 2,
            text: "positive".to_string(),
            stop_reason: Some(StopReason::EosToken),
            stop_sequence: Default::default(),
            seed: None,
            tokens: None,
            input_tokens: None,
        }];
        let actual = vec![GenerationResponse {
            input_token_count: 10,
            generated_token_count: 2,
            text: "negative".to_string(),
            stop_reason: Some(StopReason::EosToken),
            stop_sequence: Default::default(),
            seed: None,
            tokens: None,
            input_tokens: None,
        }];
        let strategy = ExactTextMatch::default();
        let evaluation_result = strategy.evaluate(&expected, &actual).unwrap();
        assert!(!evaluation_result.passed())
    }
}
