use super::{Error, Metric, Value};

pub struct CosineSimilarity;

impl Metric for CosineSimilarity {
    fn compute(a: Value, b: Value) -> Result<f32, Error> {
        use Value::*;
        match (a, b) {
            (Embedding(a), Embedding(b)) => {
                let dot_p = dot(&a, &b);
                let a_norm = norm_l2(&a);
                let b_norm = norm_l2(&b);
                Ok(dot_p / (a_norm * b_norm))
            },
            _ => Err(Error::Invalid("must be `Value::Embedding(_)`".into()))
        }
    }
}

// Utils to avoid ndarray and linear algebra deps just for this.

fn norm_l2(vs: &[f32]) -> f32 {
    vs.iter().map(|v| v.powi(2)).sum::<f32>().sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
