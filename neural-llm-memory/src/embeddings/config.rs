//! Configuration for the embedding service

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for the embedding service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model identifier from HuggingFace (e.g., "sentence-transformers/all-mpnet-base-v2")
    pub model_id: String,
    
    /// Device to use for inference ("cuda", "cpu", or "auto")
    pub device: String,
    
    /// Whether to cache embeddings to disk
    pub cache_enabled: bool,
    
    /// Path to the cache directory
    pub cache_dir: Option<PathBuf>,
    
    /// Maximum cache size in MB
    pub cache_size_mb: usize,
    
    /// Batch size for processing multiple texts
    pub batch_size: usize,
    
    /// Whether to fallback to hash-based embeddings on error
    pub fallback_enabled: bool,
    
    /// Model cache directory (for downloaded models)
    pub model_cache_dir: Option<PathBuf>,
    
    /// Maximum sequence length for the model
    pub max_sequence_length: usize,
    
    /// Whether to use half precision (f16) for faster inference
    pub use_half_precision: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/all-mpnet-base-v2".to_string(),
            device: "auto".to_string(),
            cache_enabled: true,
            cache_dir: None,
            cache_size_mb: 2048, // 2GB default cache
            batch_size: 32,
            fallback_enabled: true,
            model_cache_dir: None,
            max_sequence_length: 512,
            use_half_precision: false,
        }
    }
}

impl EmbeddingConfig {
    /// Create a configuration for CPU-only inference
    pub fn cpu_only() -> Self {
        Self {
            device: "cpu".to_string(),
            use_half_precision: false,
            ..Default::default()
        }
    }
    
    /// Create a configuration for GPU inference with optimizations
    pub fn gpu_optimized() -> Self {
        Self {
            device: "cuda".to_string(),
            use_half_precision: true,
            batch_size: 64,
            ..Default::default()
        }
    }
    
    /// Create a configuration for a specific model
    pub fn with_model(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Default::default()
        }
    }
}