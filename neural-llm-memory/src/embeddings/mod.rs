//! Semantic embeddings module for generating high-quality text embeddings
//! using pre-trained transformer models from HuggingFace.

mod service;
mod cache;
mod config;
mod simple_service;

pub use service::{EmbeddingService, EmbeddingError};
pub use cache::EmbeddingCache;
pub use config::EmbeddingConfig;
pub use simple_service::SimpleEmbeddingService;

use candle_core::{Device, Result as CandleResult};

/// Initialize the default device for model inference
pub fn get_device() -> CandleResult<Device> {
    // Try to use CUDA if available, otherwise fall back to CPU
    if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)
    } else {
        Ok(Device::Cpu)
    }
}

/// Normalize a vector to unit length for cosine similarity
pub fn normalize_vector(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

/// Calculate cosine similarity between two normalized vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_vector() {
        let mut vec = vec![3.0, 4.0, 0.0];
        normalize_vector(&mut vec);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }
}