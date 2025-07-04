//! Memory retrieval strategies and similarity metrics

use ndarray::Array2;

#[derive(Debug, Clone, Copy)]
pub enum RetrievalStrategy {
    TopK,
    Threshold(f32),
    Adaptive,
}

#[derive(Debug, Clone, Copy)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

impl SimilarityMetric {
    pub fn compute(&self, a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        match self {
            SimilarityMetric::Cosine => {
                let dot = (a * b).sum();
                let norm_a = (a * a).sum().sqrt();
                let norm_b = (b * b).sum().sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    0.0
                } else {
                    dot / (norm_a * norm_b)
                }
            }
            SimilarityMetric::Euclidean => {
                let diff = a - b;
                -((&diff * &diff).sum().sqrt())
            }
            SimilarityMetric::DotProduct => {
                (a * b).sum()
            }
            SimilarityMetric::Manhattan => {
                -(a - b).mapv(f32::abs).sum()
            }
        }
    }
}