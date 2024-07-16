use approx::{AbsDiffEq, RelativeEq};

use crate::{GenerationResponse, TokenInfo, BatchedGenerationResponse, TopToken};

use super::{EvaluationResult, EvaluationStrategy, Error};

const DEFAULT_REL_TOLERANCE: f32 = 5.0;
const DEFAULT_ABS_TOLERANCE: f32 = 5.0;

/// Evaluates approximate equality on floats using both absolute difference and
/// relative based comparisons and true equality on other types.
pub struct ApproxEquality {
    abs_tolerance: f32,
    rel_tolerance: f32,
}

impl ApproxEquality {
    pub fn new(abs_tolerance: f32, rel_tolerance: f32) -> Self {
        Self {
            abs_tolerance,
            rel_tolerance,
        }
    }
}

impl Default for ApproxEquality {
    fn default() -> Self {
        Self {
            abs_tolerance: DEFAULT_ABS_TOLERANCE,
            rel_tolerance: DEFAULT_REL_TOLERANCE,
        }
    }
}

impl EvaluationStrategy for ApproxEquality {
    fn name(&self) -> &'static str {
        "approx_equality"
    }

    fn evaluate(
        &self,
        expected: &[GenerationResponse],
        actual: &[GenerationResponse],
    ) -> Result<EvaluationResult, Error> {
        let passed = approx::relative_eq!(
            expected,
            actual,
            epsilon = self.abs_tolerance,
            max_relative = self.rel_tolerance,
        );
        Ok(EvaluationResult::new(self.name(), passed, None))
    }
}

impl AbsDiffEq for BatchedGenerationResponse {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.responses.abs_diff_eq(&other.responses, epsilon)
    }
}

impl RelativeEq for BatchedGenerationResponse {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.responses
            .relative_eq(&other.responses, epsilon, max_relative)
    }
}

impl AbsDiffEq for GenerationResponse {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let tokens = match (&self.tokens, &other.tokens) {
            (Some(a), Some(b)) => a.abs_diff_eq(b, epsilon),
            (None, None) => true,
            _ => false,
        };
        let input_tokens = match (&self.input_tokens, &other.input_tokens) {
            (Some(a), Some(b)) => a.abs_diff_eq(b, epsilon),
            (None, None) => true,
            _ => false,
        };
        self.input_token_count == other.input_token_count
            && self.generated_token_count == other.generated_token_count
            && self.text == other.text
            && self.stop_reason == other.stop_reason
            && self.stop_sequence == other.stop_sequence
            && self.seed == other.seed
            && tokens
            && input_tokens
    }
}

impl RelativeEq for GenerationResponse {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        let tokens = match (&self.tokens, &other.tokens) {
            (Some(a), Some(b)) => a.relative_eq(b, epsilon, max_relative),
            (None, None) => true,
            _ => false,
        };
        let input_tokens = match (&self.input_tokens, &other.input_tokens) {
            (Some(a), Some(b)) => a.relative_eq(b, epsilon, max_relative),
            (None, None) => true,
            _ => false,
        };
        self.input_token_count == other.input_token_count
            && self.generated_token_count == other.generated_token_count
            && self.text == other.text
            && self.stop_reason == other.stop_reason
            && self.stop_sequence == other.stop_sequence
            && self.seed == other.seed
            && tokens
            && input_tokens
    }
}

impl AbsDiffEq for TokenInfo {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let logprob = match (self.logprob, other.logprob) {
            (Some(a), Some(b)) => a.abs_diff_eq(&b, epsilon),
            (None, None) => true,
            _ => false,
        };
        let top_tokens = match (&self.top_tokens, &other.top_tokens) {
            (Some(a), Some(b)) => a.abs_diff_eq(b, epsilon),
            (None, None) => true,
            _ => false,
        };
        self.text == other.text 
        && logprob 
        //&& self.rank == other.rank 
        && top_tokens
    }
}

impl RelativeEq for TokenInfo {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        let logprob = match (self.logprob, other.logprob) {
            (Some(a), Some(b)) => a.relative_eq(&b, epsilon, max_relative),
            (None, None) => true,
            _ => false,
        };
        let top_tokens = match (&self.top_tokens, &other.top_tokens) {
            (Some(a), Some(b)) => a.relative_eq(b, epsilon, max_relative),
            (None, None) => true,
            _ => false,
        };
        self.text == other.text 
        && logprob 
        //&& self.rank == other.rank 
        && top_tokens
    }
}

impl AbsDiffEq for TopToken {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let logprob = match (self.logprob, other.logprob) {
            (Some(a), Some(b)) => a.abs_diff_eq(&b, epsilon),
            (None, None) => true,
            _ => false,
        };
        self.text == other.text && logprob
    }
}

impl RelativeEq for TopToken {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        let logprob = match (self.logprob, other.logprob) {
            (Some(a), Some(b)) => a.relative_eq(&b, epsilon, max_relative),
            (None, None) => true,
            _ => false,
        };
        self.text == other.text && logprob
    }
}


#[cfg(test)]
mod tests {
    use crate::StopReason;

    use super::*;

    #[test]
    fn exact_text_exact_logprobs_should_pass() {
        let expected = vec![GenerationResponse {
            input_token_count: 10,
            generated_token_count: 2,
            text: " positive".to_string(),
            stop_reason: Some(StopReason::EosToken),
            stop_sequence: Default::default(),
            seed: None,
            tokens: Some(vec![TokenInfo {
                text: "\u{0120}positive".to_string(),
                logprob: Some(-0.05955762),
                rank: Some(1),
                top_tokens: Some(vec![
                    TopToken {
                        text: "\u{0120}positive".to_string(),
                        logprob: Some(-0.05955762),
                    },
                    TopToken {
                        text: "\u{0120}negative".to_string(),
                        logprob: Some(-3.8135555),
                    },
                ]),
            }]),
            input_tokens: None,
        }];
        let actual = &expected;
        let strategy = ApproxEquality::default();
        let evaluation_result = strategy.evaluate(&expected, actual).unwrap();
        assert!(evaluation_result.passed())
    }

    #[test]
    fn exact_text_approx_logprobs_should_pass() {
        let expected = vec![GenerationResponse {
            input_token_count: 10,
            generated_token_count: 2,
            text: " positive".to_string(),
            stop_reason: Some(StopReason::EosToken),
            stop_sequence: Default::default(),
            seed: None,
            tokens: Some(vec![TokenInfo {
                text: "\u{0120}positive".to_string(),
                logprob: Some(-0.05955762),
                rank: Some(1),
                top_tokens: Some(vec![
                    TopToken {
                        text: "\u{0120}positive".to_string(),
                        logprob: Some(-0.05955762),
                    },
                    TopToken {
                        text: "\u{0120}negative".to_string(),
                        logprob: Some(-3.8135555),
                    },
                ]),
            }]),
            input_tokens: None,
        }];
        let actual = vec![GenerationResponse {
            input_token_count: 10,
            generated_token_count: 2,
            text: " positive".to_string(),
            stop_reason: Some(StopReason::EosToken),
            stop_sequence: Default::default(),
            seed: None,
            tokens: Some(vec![TokenInfo {
                text: "\u{0120}positive".to_string(),
                logprob: Some(-0.05955762),
                rank: Some(1),
                top_tokens: Some(vec![
                    TopToken {
                        text: "\u{0120}positive".to_string(),
                        logprob: Some(-0.05955762),
                    },
                    TopToken {
                        text: "\u{0120}negative".to_string(),
                        logprob: Some(-3.814),
                    },
                ]),
            }]),
            input_tokens: None,
        }];
        let strategy = ApproxEquality::default();
        let evaluation_result = strategy.evaluate(&expected, &actual).unwrap();
        assert!(evaluation_result.passed())
    }

    #[test]
    fn exact_text_not_approx_logprobs_should_fail() {
        let expected = vec![GenerationResponse {
            input_token_count: 10,
            generated_token_count: 2,
            text: " positive".to_string(),
            stop_reason: Some(StopReason::EosToken),
            stop_sequence: Default::default(),
            seed: None,
            tokens: Some(vec![TokenInfo {
                text: "\u{0120}positive".to_string(),
                logprob: Some(-0.05955762),
                rank: Some(1),
                top_tokens: Some(vec![
                    TopToken {
                        text: "\u{0120}positive".to_string(),
                        logprob: Some(-0.05955762),
                    },
                    TopToken {
                        text: "\u{0120}negative".to_string(),
                        logprob: Some(-3.8135555),
                    },
                ]),
            }]),
            input_tokens: None,
        }];
        let actual = vec![GenerationResponse {
            input_token_count: 10,
            generated_token_count: 2,
            text: " negative".to_string(),
            stop_reason: Some(StopReason::EosToken),
            stop_sequence: Default::default(),
            seed: None,
            tokens: Some(vec![TokenInfo {
                text: "\u{0120}positive".to_string(),
                logprob: Some(-0.05955762),
                rank: Some(1),
                top_tokens: Some(vec![
                    TopToken {
                        text: "\u{0120}positive".to_string(),
                        logprob: Some(-0.05955762),
                    },
                    TopToken {
                        text: "\u{0120}negative".to_string(),
                        logprob: Some(-3.8135555),
                    },
                ]),
            }]),
            input_tokens: None,
        }];
        let strategy = ApproxEquality::default();
        let evaluation_result = strategy.evaluate(&expected, &actual).unwrap();
        assert!(!evaluation_result.passed())
    }
}