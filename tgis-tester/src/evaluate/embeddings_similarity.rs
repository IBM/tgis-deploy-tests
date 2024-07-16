use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert;
use tokenizers::Tokenizer;

use crate::{
    evaluate::metrics::{CosineSimilarity, Metric},
    GenerationResponse,
};

use super::{Error, EvaluationResult, EvaluationStrategy};

const DEFAULT_MIN_SIMILARITY_SCORE: f32 = 0.90;
const DEFAULT_MODEL_NAME: &str = "sentence-transformers/all-MiniLM-L6-v2";

pub struct EmbeddingsSimilarity {
    model: bert::BertModel,
    tokenizer: Tokenizer,
    min_similarity_score: f32,
}

impl EmbeddingsSimilarity {
    pub fn new(model_name: &str, min_similarity_score: f32) -> Result<Self, Error> {
        let hub_api = hf_hub::api::sync::Api::new()?;
        let repo = hub_api.model(model_name.to_string());
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("pytorch_model.bin")?;
        let config: bert::Config = {
            let config_string = std::fs::read_to_string(config_path)?;
            serde_json::from_str(&config_string).unwrap()
        };
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)?;
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }
        let vb = VarBuilder::from_pth(&weights_path, bert::DTYPE, &Device::Cpu)?;
        let model = bert::BertModel::load(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            min_similarity_score,
        })
    }
}

impl Default for EmbeddingsSimilarity {
    fn default() -> Self {
        Self::new(DEFAULT_MODEL_NAME, DEFAULT_MIN_SIMILARITY_SCORE).unwrap()
    }
}

impl EvaluationStrategy for EmbeddingsSimilarity {
    fn name(&self) -> &'static str {
        "embeddings_similarity"
    }

    fn evaluate(
        &self,
        expected: &[GenerationResponse],
        actual: &[GenerationResponse],
    ) -> Result<EvaluationResult, Error> {
        let expected_text = expected[0].text.to_string();
        let actual_text = actual[0].text.to_string();

        let token_ids = self
            .tokenizer
            .encode_batch(vec![expected_text, actual_text], true)?
            .iter()
            .map(|e| Ok(Tensor::new(e.get_ids(), &Device::Cpu)?))
            .collect::<Result<Vec<_>, Error>>()?;
        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let embeds = self.model.forward(&token_ids, &token_type_ids)?;
        let embeds = mean_pooling(&embeds)?;
        let embeds = norm_l2(&embeds)?;

        let embeds = embeds.to_vec2::<f32>()?;
        let expected_embed = embeds[0].as_slice();
        let actual_embed = embeds[1].as_slice();
        let similarity_score =
            CosineSimilarity::compute(expected_embed.into(), actual_embed.into())?;
        let passed = similarity_score >= self.min_similarity_score;

        Ok(EvaluationResult::new(
            self.name(),
            passed,
            Some(vec![("similarity_score".into(), similarity_score)]),
        ))
    }
}

fn norm_l2(v: &Tensor) -> Result<Tensor, Error> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

fn mean_pooling(v: &Tensor) -> Result<Tensor, Error> {
    let (_n_sentence, n_tokens, _hidden_size) = v.dims3()?;
    let v = (v.sum(1)? / (n_tokens as f64))?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use crate::GenerationResponse;

    use super::*;

    #[test]
    fn exact_text_should_pass() {
        let expected = vec![GenerationResponse {
            text: "some generated text".to_string(),
            ..Default::default()
        }];
        let actual = &expected;
        let strategy = EmbeddingsSimilarity::default();
        let evaluation_result = strategy.evaluate(&expected, actual).unwrap();
        let similarity_score = evaluation_result.metrics().unwrap()[0].1;
        assert!(evaluation_result.passed() && similarity_score > 0.99)
    }

    #[test]
    fn similar_text_should_pass() {
        let expected = vec![GenerationResponse {
            text: "some generated text".to_string(),
            ..Default::default()
        }];
        let actual = vec![GenerationResponse {
            text: "some generated text.".to_string(),
            stop_sequence: Default::default(),
            ..Default::default()
        }];
        let strategy = EmbeddingsSimilarity::default();
        let evaluation_result = strategy.evaluate(&expected, &actual).unwrap();
        let similarity_score = evaluation_result.metrics().unwrap()[0].1;
        assert!(evaluation_result.passed() && similarity_score > 0.90)
    }

    #[test]
    fn dissimilar_text_should_fail() {
        let expected = vec![GenerationResponse {
            text: "some generated text".to_string(),
            ..Default::default()
        }];
        let actual = vec![GenerationResponse {
            text: "hello world".to_string(),
            stop_sequence: Default::default(),
            ..Default::default()
        }];
        let strategy = EmbeddingsSimilarity::default();
        let evaluation_result = strategy.evaluate(&expected, &actual).unwrap();
        let similarity_score = evaluation_result.metrics().unwrap()[0].1;
        assert!(!evaluation_result.passed() && similarity_score < 0.50)
    }
}
