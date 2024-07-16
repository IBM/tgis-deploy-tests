use crate::GenerationResponse;

use super::{Error, EvaluationResult, EvaluationStrategy};

pub struct Multiple {
    pub strategies: Vec<Box<dyn EvaluationStrategy>>,
}

impl Multiple {
    pub fn new(strategies: Vec<Box<dyn EvaluationStrategy>>) -> Self {
        Self { strategies }
    }
}

impl EvaluationStrategy for Multiple {
    fn name(&self) -> &'static str {
        "multiple"
    }

    fn evaluate(
        &self,
        expected: &[GenerationResponse],
        actual: &[GenerationResponse],
    ) -> Result<EvaluationResult, Error> {
        let results = self
            .strategies
            .iter()
            .map(|strategy| strategy.evaluate(expected, actual))
            .collect::<Result<Vec<_>, Error>>()?;
        let passed = results.iter().all(|r| r.passed);
        Ok(EvaluationResult::new(self.name(), passed, None))
    }
}

#[cfg(test)]
mod tests {}
