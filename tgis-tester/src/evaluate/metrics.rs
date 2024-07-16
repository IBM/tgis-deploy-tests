mod cosine_similarity;
pub use cosine_similarity::CosineSimilarity;

use super::Error;

pub trait Metric {
    fn compute(a: Value, b: Value) -> Result<f32, Error>;
}

#[derive(Debug, Clone)]
pub enum Value {
    Text(String),
    Tokens(Vec<u32>),
    Embedding(Vec<f32>),
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::Text(value)
    }
}

impl From<Vec<u32>> for Value {
    fn from(value: Vec<u32>) -> Self {
        Value::Tokens(value)
    }
}

impl From<&[u32]> for Value {
    fn from(value: &[u32]) -> Self {
        Value::Tokens(value.to_vec())
    }
}

impl From<Vec<f32>> for Value {
    fn from(value: Vec<f32>) -> Self {
        Value::Embedding(value)
    }
}

impl From<&[f32]> for Value {
    fn from(value: &[f32]) -> Self {
        Value::Embedding(value.to_vec())
    }
}

// impl TryFrom<Value> for ndarray::Array1<f32> {
//     type Error = Error;

//     fn try_from(value: Value) -> Result<Self, Self::Error> {
//         match value {
//             Value::Embedding(tokens) => Ok(ndarray::Array1::<f32>::from_vec(tokens)),
//             _ => Err(Error::Invalid("must be `Value::Embedding(_)`".into())),
//         }
//     }
// }
