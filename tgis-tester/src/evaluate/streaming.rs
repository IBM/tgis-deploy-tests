use itertools::Itertools;

use crate::{GenerationResponse, TestCase, TgisRequest, TgisResponse};

/// Validates the structure and contents of a GenerateStream response.
pub fn valid_streaming_response(case: &TestCase, resp: &[GenerationResponse]) -> bool {
    let req = &case.request;
    if matches!(req, TgisRequest::Unary(_)) || resp.len() < 2 {
        return false;
    }
    let expected_resp = match &case.expected_response {
        Some(TgisResponse::Unary(resp)) if resp.responses.first().is_some() => &resp.responses[0],
        _ => return false,
    };
    let params = req.params();
    let first = resp.first().unwrap();
    let second = &resp[1];
    let last = resp.last().unwrap();
    let input_text = params
        .map(|p| {
            p.response
                .as_ref()
                .is_some_and(|r| r.input_text == Some(true))
        })
        .unwrap_or_default();
    let input_tokens = params
        .map(|p| {
            p.response
                .as_ref()
                .is_some_and(|r| r.input_tokens == Some(true))
        })
        .unwrap_or_default();
    let expected_seed = params
        .map(|p| p.sampling.as_ref().map(|s| s.seed).unwrap_or_default())
        .unwrap_or_default();
    let expected_stop_reason = expected_resp.stop_reason.unwrap();
    let expected_generated_token_count = expected_resp.generated_token_count;

    // The first message only includes input_token_count
    // and if specified, input text
    let valid_first = first.input_token_count > 0
        && first.generated_token_count == u32::default()
        && first.stop_reason.is_none()
        && if input_text {
            !first.text.is_empty()
        } else {
            true
        };
    // If specified, the second message only includes input_tokens
    let valid_second = if input_tokens {
        second.generated_token_count == u32::default()
            && second.input_tokens.as_ref().is_some_and(|v| !v.is_empty())
            && second.text.is_empty()
            && second.stop_reason.is_none()
    } else {
        true
    };
    // The last message includes the expected stop_reason and generated_token_count
    // and if specified, seed
    let valid_last = last.stop_reason.is_some_and(|v| v == expected_stop_reason)
        && last.generated_token_count == expected_generated_token_count
        && if let Some(seed) = expected_seed {
            last.seed == Some(seed)
        } else {
            true
        };

    // generated_token_count is monotonically increasing
    let valid_generated_token_counts = resp
        .iter()
        .map(|r| r.generated_token_count)
        .tuple_windows()
        .all(|(a, b)| a <= b);

    valid_first && valid_second && valid_last && valid_generated_token_counts
}

/// Merge GenerateStream response messages to a single [`GenerationResponse`] for evaluation.
pub fn merge_streaming_response(resp: &[GenerationResponse]) -> GenerationResponse {
    let first = resp.first().unwrap();
    let second = &resp[1];
    let last = resp.last().unwrap();
    let input_token_count = first.input_token_count;
    let generated_token_count = last.generated_token_count;
    let text = resp
        .iter()
        .map(|r| r.text.clone())
        .collect::<Vec<_>>()
        .join("");
    let input_tokens = second.input_tokens.clone();
    let stop_reason = last.stop_reason;
    let stop_sequence = last.stop_sequence.clone();
    let seed = resp.last().unwrap().seed;
    let tokens = resp
        .iter()
        .flat_map(|r| r.tokens.clone())
        .flatten()
        .collect::<Vec<_>>();
    let tokens = if tokens.is_empty() {
        None
    } else {
        Some(tokens)
    };
    GenerationResponse {
        input_token_count,
        generated_token_count,
        text,
        stop_reason,
        stop_sequence,
        seed,
        input_tokens,
        tokens,
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        BatchedGenerationRequest, BatchedGenerationResponse, DecodingMethod, GenerationRequest,
        ModelName, Parameters, ResponseOptions, SamplingParameters, SingleGenerationRequest,
        StopReason, StoppingCriteria, TokenInfo,
    };

    use super::*;

    fn build_test_case(
        req: SingleGenerationRequest,
        expected_resp: BatchedGenerationResponse,
    ) -> TestCase {
        TestCase {
            model_name: Some(ModelName::new("test")),
            model_config: None,
            name: "test".to_string(),
            single_shard_only: false,
            skip_check: false,
            request: req.into(),
            expected_response: Some(expected_resp.into()),
            expected_error: None,
        }
    }

    #[test]
    fn test_valid_sequence_should_pass() {
        let req = SingleGenerationRequest {
            prefix_id: None,
            adapter_id: None,
            params: Some(Parameters {
                stopping: Some(StoppingCriteria {
                    max_new_tokens: Some(2),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            request: GenerationRequest {
                text: "One upon a time,".to_string(),
            },
        };
        let expected_resp = GenerationResponse {
            input_token_count: 6,
            generated_token_count: 2,
            text: " there wa".to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            ..Default::default()
        };
        let case = build_test_case(req, expected_resp.into());
        let resp = &[
            GenerationResponse {
                input_token_count: 6,
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 1,
                text: " ther".to_string(),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 2,
                text: "e wa".to_string(),
                stop_reason: Some(StopReason::MaxTokens),
                ..Default::default()
            },
        ];
        assert!(valid_streaming_response(&case, resp))
    }

    #[test]
    fn test_valid_generated_token_count_should_pass() {
        let req = SingleGenerationRequest {
            prefix_id: None,
            adapter_id: None,
            params: None,
            request: GenerationRequest {
                text: "One upon a time,".to_string(),
            },
        };
        let expected_resp = GenerationResponse {
            input_token_count: 6,
            generated_token_count: 3,
            text: " there wa".to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            ..Default::default()
        };
        let case = build_test_case(req, expected_resp.into());
        let resp = &[
            GenerationResponse {
                generated_token_count: 0,
                input_token_count: 6,
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 2,
                text: " ther".to_string(),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 3,
                text: "e wa".to_string(),
                stop_reason: Some(StopReason::MaxTokens),
                ..Default::default()
            },
        ];
        assert!(valid_streaming_response(&case, resp))
    }

    #[test]
    fn test_invalid_sequence_should_fail() {
        let req = SingleGenerationRequest {
            prefix_id: None,
            adapter_id: None,
            params: Some(Parameters {
                stopping: Some(StoppingCriteria {
                    max_new_tokens: Some(2),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            request: GenerationRequest {
                text: "One upon a time,".to_string(),
            },
        };
        let expected_resp = GenerationResponse {
            input_token_count: 6,
            generated_token_count: 2,
            text: " there wa".to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            ..Default::default()
        };
        let case = build_test_case(req, expected_resp.into());
        let resp = &[
            GenerationResponse {
                generated_token_count: 1,
                text: " ther".to_string(),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 2,
                text: "e wa".to_string(),
                stop_reason: Some(StopReason::MaxTokens),
                ..Default::default()
            },
            GenerationResponse {
                input_token_count: 6,
                ..Default::default()
            },
        ];
        assert!(!valid_streaming_response(&case, resp))
    }

    #[test]
    fn test_invalid_first_message_should_fail() {
        let req = SingleGenerationRequest {
            prefix_id: None,
            adapter_id: None,
            params: Some(Parameters {
                stopping: Some(StoppingCriteria {
                    max_new_tokens: Some(2),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            request: GenerationRequest {
                text: "One upon a time,".to_string(),
            },
        };
        let expected_resp = GenerationResponse {
            input_token_count: 6,
            generated_token_count: 2,
            text: " there wa".to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            ..Default::default()
        };
        let case = build_test_case(req, expected_resp.into());
        let resp = &[
            GenerationResponse {
                input_token_count: 6,
                generated_token_count: 2,
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 1,
                text: " ther".to_string(),
                ..Default::default()
            },
        ];
        assert!(!valid_streaming_response(&case, resp))
    }

    #[test]
    fn test_empty_resp_should_fail() {
        let req = SingleGenerationRequest {
            prefix_id: None,
            adapter_id: None,
            params: Some(Parameters {
                stopping: Some(StoppingCriteria {
                    max_new_tokens: Some(2),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            request: GenerationRequest {
                text: "One upon a time,".to_string(),
            },
        };
        let expected_resp = GenerationResponse {
            input_token_count: 6,
            generated_token_count: 2,
            text: " there wa".to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            ..Default::default()
        };
        let case = build_test_case(req, expected_resp.into());
        let resp = vec![];
        assert!(!valid_streaming_response(&case, &resp))
    }

    #[test]
    fn test_invalid_last_message_should_fail() {
        let req = SingleGenerationRequest {
            prefix_id: None,
            adapter_id: None,
            params: Some(Parameters {
                stopping: Some(StoppingCriteria {
                    max_new_tokens: Some(2),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            request: GenerationRequest {
                text: "One upon a time,".to_string(),
            },
        };
        let expected_resp = GenerationResponse {
            input_token_count: 6,
            generated_token_count: 2,
            text: " there wa".to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            ..Default::default()
        };
        let case = build_test_case(req, expected_resp.into());
        let resp = &[
            GenerationResponse {
                input_token_count: 6,
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 1,
                text: " ther".to_string(),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 2,
                text: "e wa".to_string(),
                ..Default::default()
            },
        ];
        assert!(!valid_streaming_response(&case, resp))
    }

    #[test]
    fn test_invalid_req_should_fail() {
        let req = BatchedGenerationRequest {
            prefix_id: None,
            adapter_id: None,
            params: None,
            requests: vec![],
        };
        let expected_resp = GenerationResponse {
            input_token_count: 6,
            generated_token_count: 2,
            text: " there wa".to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            ..Default::default()
        };
        let case = TestCase {
            model_name: Some(ModelName::new("test")),
            model_config: None,
            name: "test".to_string(),
            single_shard_only: false,
            skip_check: false,
            request: req.into(),
            expected_response: Some(
                BatchedGenerationResponse {
                    responses: vec![expected_resp],
                }
                .into(),
            ),
            expected_error: None,
        };
        let resp = vec![];
        assert!(!valid_streaming_response(&case, &resp))
    }

    #[test]
    fn test_valid_input_tokens_should_pass() {
        let req = SingleGenerationRequest {
            prefix_id: None,
            adapter_id: None,
            params: Some(Parameters {
                response: Some(ResponseOptions {
                    input_tokens: Some(true),
                    ..Default::default()
                }),
                stopping: Some(StoppingCriteria {
                    max_new_tokens: Some(2),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            request: GenerationRequest {
                text: "One upon a time,".to_string(),
            },
        };
        let expected_resp = GenerationResponse {
            input_token_count: 6,
            generated_token_count: 2,
            text: " there wa".to_string(),
            input_tokens: Some(vec![
                TokenInfo {
                    text: "<s>".to_string(),
                    logprob: None,
                    rank: None,
                    top_tokens: None,
                },
                TokenInfo {
                    text: "_Once".to_string(),
                    logprob: None,
                    rank: None,
                    top_tokens: None,
                },
            ]),
            stop_reason: Some(StopReason::MaxTokens),
            ..Default::default()
        };
        let case = build_test_case(req, expected_resp.into());
        let resp = &[
            GenerationResponse {
                input_token_count: 6,
                ..Default::default()
            },
            GenerationResponse {
                input_tokens: Some(vec![
                    TokenInfo {
                        text: "<s>".to_string(),
                        logprob: None,
                        rank: None,
                        top_tokens: None,
                    },
                    TokenInfo {
                        text: "_Once".to_string(),
                        logprob: None,
                        rank: None,
                        top_tokens: None,
                    },
                ]),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 1,
                text: " ther".to_string(),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 2,
                text: "e wa".to_string(),
                stop_reason: Some(StopReason::MaxTokens),
                ..Default::default()
            },
        ];
        assert!(valid_streaming_response(&case, resp))
    }

    #[test]
    fn test_seed() {
        let req = SingleGenerationRequest {
            prefix_id: None,
            adapter_id: None,
            params: Some(Parameters {
                method: DecodingMethod::Sample,
                sampling: Some(SamplingParameters {
                    seed: Some(1337),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            request: GenerationRequest {
                text: "One upon a time,".to_string(),
            },
        };
        let expected_resp = GenerationResponse {
            input_token_count: 6,
            generated_token_count: 2,
            text: " there wa".to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            seed: Some(1337),
            ..Default::default()
        };
        let case = build_test_case(req, expected_resp.into());
        let resp = &[
            GenerationResponse {
                input_token_count: 6,
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 1,
                text: " ther".to_string(),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 2,
                text: "e wa".to_string(),
                seed: Some(1337),
                stop_reason: Some(StopReason::MaxTokens),
                ..Default::default()
            },
        ];
        assert!(valid_streaming_response(&case, resp));

        let bad_resp = &[
            GenerationResponse {
                input_token_count: 6,
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 1,
                text: " ther".to_string(),
                seed: Some(1337),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 2,
                text: "e wa".to_string(),
                stop_reason: Some(StopReason::MaxTokens),
                ..Default::default()
            },
        ];
        assert!(!valid_streaming_response(&case, bad_resp))
    }

    #[test]
    fn test_merge_streaming_response() {
        let expected = &[GenerationResponse {
            input_token_count: 8,
            generated_token_count: 5,
            text: "�mentantressr".to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            ..Default::default()
        }];
        let resp = &[
            GenerationResponse {
                input_token_count: 8,
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 2,
                text: "�men".to_string(),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 3,
                text: "tan".to_string(),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 4,
                text: "tres".to_string(),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 5,
                text: "sr".to_string(),
                stop_reason: Some(StopReason::MaxTokens),
                ..Default::default()
            },
        ];
        let resp_merged = merge_streaming_response(resp);
        assert_eq!(expected[0], resp_merged)
    }

    #[test]
    fn test_merge_streaming_response_with_tokens() {
        let expected = &[GenerationResponse {
            input_token_count: 6,
            generated_token_count: 10,
            text: " there was a little girl who loved to read.".to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            tokens: Some(vec![
                TokenInfo {
                    text: "▁there".to_string(),
                    ..Default::default()
                },
                TokenInfo {
                    text: "▁was".to_string(),
                    ..Default::default()
                },
                TokenInfo {
                    text: "▁a".to_string(),
                    ..Default::default()
                },
                TokenInfo {
                    text: "▁little".to_string(),
                    ..Default::default()
                },
                TokenInfo {
                    text: "▁girl".to_string(),
                    ..Default::default()
                },
                TokenInfo {
                    text: "▁who".to_string(),
                    ..Default::default()
                },
                TokenInfo {
                    text: "▁loved".to_string(),
                    ..Default::default()
                },
                TokenInfo {
                    text: "▁to".to_string(),
                    ..Default::default()
                },
                TokenInfo {
                    text: "▁read".to_string(),
                    ..Default::default()
                },
                TokenInfo {
                    text: ".".to_string(),
                    ..Default::default()
                },
            ]),
            ..Default::default()
        }];
        let resp = &[
            GenerationResponse {
                input_token_count: 6,
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 1,
                text: " ther".to_string(),
                tokens: Some(vec![TokenInfo {
                    text: "▁there".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 2,
                text: "e wa".to_string(),
                tokens: Some(vec![TokenInfo {
                    text: "▁was".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 3,
                text: "s ".to_string(),
                tokens: Some(vec![TokenInfo {
                    text: "▁a".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 4,
                text: "a littl".to_string(),
                tokens: Some(vec![TokenInfo {
                    text: "▁little".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 5,
                text: "e gir".to_string(),
                tokens: Some(vec![TokenInfo {
                    text: "▁girl".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 6,
                text: "l wh".to_string(),
                tokens: Some(vec![TokenInfo {
                    text: "▁who".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 7,
                text: "o love".to_string(),
                tokens: Some(vec![TokenInfo {
                    text: "▁loved".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 8,
                text: "d t".to_string(),
                tokens: Some(vec![TokenInfo {
                    text: "▁to".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 9,
                text: "o rea".to_string(),
                tokens: Some(vec![TokenInfo {
                    text: "▁read".to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 10,
                text: "d.".to_string(),
                tokens: Some(vec![TokenInfo {
                    text: ".".to_string(),
                    ..Default::default()
                }]),
                stop_reason: Some(StopReason::MaxTokens),
                ..Default::default()
            },
        ];
        let resp_merged = merge_streaming_response(resp);
        assert_eq!(expected[0], resp_merged)
    }

    #[test]
    fn test_merge_streaming_response_with_stop_sequence() {
        let expected = &[GenerationResponse {
            input_token_count: 6,
            generated_token_count: 4,
            text: " there was a little".to_string(),
            stop_reason: Some(StopReason::StopSequence),
            stop_sequence: "little".to_string(),
            ..Default::default()
        }];
        let resp = &[
            GenerationResponse {
                input_token_count: 6,
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 2,
                text: " ther".to_string(),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 3,
                text: "e wa".to_string(),
                ..Default::default()
            },
            GenerationResponse {
                generated_token_count: 4,
                text: "s a little".to_string(),
                stop_reason: Some(StopReason::StopSequence),
                stop_sequence: "little".to_string(),
                ..Default::default()
            },
        ];
        let resp_merged = merge_streaming_response(resp);
        assert_eq!(expected[0], resp_merged)
    }
}
