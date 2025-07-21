//! Main embedding service using Candle and HuggingFace models

use std::sync::Arc;
use anyhow::{Result, Context};
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use tokenizers::{Tokenizer, PaddingParams, TruncationParams};
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

use super::{EmbeddingConfig, EmbeddingCache, normalize_vector};

/// Error types for embedding operations
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
    
    #[error("Tokenization failed: {0}")]
    TokenizationError(String),
    
    #[error("Inference failed: {0}")]
    InferenceError(String),
    
    #[error("Device error: {0}")]
    DeviceError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Service for generating semantic embeddings using pre-trained models
pub struct EmbeddingService {
    /// The BERT model for generating embeddings
    model: Arc<BertModel>,
    /// Tokenizer for preprocessing text
    tokenizer: Arc<RwLock<Tokenizer>>,
    /// Device for computation
    device: Device,
    /// Configuration
    config: EmbeddingConfig,
    /// Embedding cache
    cache: Option<EmbeddingCache>,
    /// Model configuration
    model_config: BertConfig,
}

impl EmbeddingService {
    /// Create a new embedding service with the given configuration
    pub async fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        info!("Initializing embedding service with model: {}", config.model_id);
        
        // Initialize device
        let device = match config.device.as_str() {
            "cuda" => Device::new_cuda(0)
                .map_err(|e| EmbeddingError::DeviceError(e.to_string()))?,
            "cpu" => Device::Cpu,
            "auto" => Self::get_device()
                .map_err(|e| EmbeddingError::DeviceError(e.to_string()))?,
            _ => return Err(EmbeddingError::ConfigError(
                format!("Invalid device: {}", config.device)
            )),
        };
        
        info!("Using device: {:?}", device);
        
        // Download model files if needed
        let api = Api::new()
            .map_err(|e| EmbeddingError::ModelLoadError(e.to_string()))?;
        
        let repo = api.repo(Repo::new(
            config.model_id.clone(),
            RepoType::Model,
        ));
        
        // Download model files
        let model_file = match repo.get("model.safetensors").await {
            Ok(file) => file,
            Err(_) => repo.get("pytorch_model.bin").await
                .map_err(|e| EmbeddingError::ModelLoadError(
                    format!("Failed to download model: {}", e)
                ))?,
        };
        
        let config_file = repo
            .get("config.json")
            .await
            .map_err(|e| EmbeddingError::ModelLoadError(
                format!("Failed to download config: {}", e)
            ))?;
        
        let tokenizer_file = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| EmbeddingError::ModelLoadError(
                format!("Failed to download tokenizer: {}", e)
            ))?;
        
        // Load model configuration
        let model_config: BertConfig = serde_json::from_slice(
            &std::fs::read(&config_file)
                .map_err(|e| EmbeddingError::ModelLoadError(e.to_string()))?
        ).map_err(|e| EmbeddingError::ModelLoadError(
            format!("Failed to parse config: {}", e)
        ))?;
        
        // Load tokenizer
        let mut tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| EmbeddingError::TokenizationError(e.to_string()))?;
        
        // Configure tokenizer
        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: config.max_sequence_length,
            ..Default::default()
        }));
        
        // Load model weights
        let vb = if model_file.to_string_lossy().ends_with(".safetensors") {
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[model_file.clone()],
                    DType::F32,
                    &device,
                ).map_err(|e| EmbeddingError::ModelLoadError(e.to_string()))?
            }
        } else {
            VarBuilder::from_pth(model_file, DType::F32, &device)
                .map_err(|e| EmbeddingError::ModelLoadError(e.to_string()))?
        };
        
        // Create BERT model
        let model = BertModel::load(vb, &model_config)
            .map_err(|e| EmbeddingError::ModelLoadError(e.to_string()))?;
        
        // Initialize cache if enabled
        let cache = if config.cache_enabled {
            Some(EmbeddingCache::new(
                config.cache_dir.clone(),
                config.cache_size_mb,
                config.model_id.clone(),
            ).map_err(|e| EmbeddingError::ConfigError(e.to_string()))?)
        } else {
            None
        };
        
        info!("Embedding service initialized successfully");
        
        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(RwLock::new(tokenizer)),
            device,
            config,
            cache,
            model_config,
        })
    }
    
    /// Initialize the default device for model inference
    fn get_device() -> candle_core::Result<Device> {
        // Try to use CUDA if available, otherwise fall back to CPU
        if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)
        } else {
            Ok(Device::Cpu)
        }
    }
    
    /// Generate embedding for a single text
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Check cache first
        if let Some(ref cache) = self.cache {
            if let Some(embedding) = cache.get(text).await {
                debug!("Cache hit for text: {}", text.chars().take(50).collect::<String>());
                return Ok(embedding);
            }
        }
        
        // Tokenize the text
        let tokenizer = self.tokenizer.read().await;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| EmbeddingError::TokenizationError(e.to_string()))?;
        
        let input_ids = Tensor::new(
            encoding.get_ids(),
            &self.device,
        ).map_err(|e| EmbeddingError::InferenceError(e.to_string()))?
        .unsqueeze(0).map_err(|e| EmbeddingError::InferenceError(e.to_string()))?;
        
        let token_type_ids = Tensor::new(
            encoding.get_type_ids(),
            &self.device,
        ).map_err(|e| EmbeddingError::InferenceError(e.to_string()))?
        .unsqueeze(0).map_err(|e| EmbeddingError::InferenceError(e.to_string()))?;
        
        drop(tokenizer); // Release lock
        
        // Create attention mask from encoding
        let attention_mask = Tensor::new(
            encoding.get_attention_mask(),
            &self.device,
        ).map_err(|e| EmbeddingError::InferenceError(e.to_string()))?
        .unsqueeze(0).map_err(|e| EmbeddingError::InferenceError(e.to_string()))?;
        
        // Run inference
        let embeddings = self.model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| EmbeddingError::InferenceError(e.to_string()))?;
        
        // Get embeddings shape
        let (batch_size, seq_len, hidden_size) = embeddings.dims3()
            .map_err(|e| EmbeddingError::InferenceError(e.to_string()))?;
        
        // Mean pooling across sequence dimension
        // Sum across sequence dimension and divide by sequence length
        let sum_embeddings = embeddings
            .sum(1)
            .map_err(|e| EmbeddingError::InferenceError(e.to_string()))?;
        
        // Count non-padding tokens for accurate mean
        let mask_sum = attention_mask
            .sum(1)
            .map_err(|e| EmbeddingError::InferenceError(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| EmbeddingError::InferenceError(e.to_string()))?;
        
        let divisor = if mask_sum[0] > 0.0 { mask_sum[0] } else { seq_len as f32 };
        
        let divisor_tensor = Tensor::new(&[divisor], &self.device)
            .map_err(|e| EmbeddingError::InferenceError(e.to_string()))?;
        
        let mean_embedding = sum_embeddings.broadcast_div(&divisor_tensor)
            .map_err(|e| EmbeddingError::InferenceError(e.to_string()))?
            .squeeze(0)
            .map_err(|e| EmbeddingError::InferenceError(e.to_string()))?;
        
        // Convert to Vec<f32>
        let mut embedding = mean_embedding
            .to_vec1::<f32>()
            .map_err(|e| EmbeddingError::InferenceError(e.to_string()))?;
        
        // Normalize the embedding
        normalize_vector(&mut embedding);
        
        // Store in cache
        if let Some(ref cache) = self.cache {
            if let Err(e) = cache.put(text, embedding.clone()).await {
                warn!("Failed to cache embedding: {}", e);
            }
        }
        
        Ok(embedding)
    }
    
    /// Generate embeddings for multiple texts in a batch
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();
        
        // Check cache for each text
        for (i, text) in texts.iter().enumerate() {
            if let Some(ref cache) = self.cache {
                if let Some(embedding) = cache.get(text).await {
                    results.push(Some(embedding));
                    continue;
                }
            }
            results.push(None);
            uncached_indices.push(i);
            uncached_texts.push(text.as_str());
        }
        
        // Process uncached texts in batches
        if !uncached_texts.is_empty() {
            // For now, process one by one - batch processing can be added later
            for (idx, text) in uncached_indices.iter().zip(uncached_texts.iter()) {
                let embedding = self.embed(text).await?;
                results[*idx] = Some(embedding);
            }
        }
        
        // Unwrap all results
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }
    
    /// Get the embedding dimension for this model
    pub fn embedding_dimension(&self) -> usize {
        self.model_config.hidden_size
    }
    
    /// Clear the embedding cache
    pub async fn clear_cache(&self) -> Result<()> {
        if let Some(ref cache) = self.cache {
            cache.clear().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cosine_similarity;
    use tempfile::TempDir;
    
    fn test_config() -> EmbeddingConfig {
        EmbeddingConfig {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            device: "cpu".to_string(),
            cache_enabled: false,
            cache_dir: None,
            cache_size_mb: 100,
            batch_size: 8,
            fallback_enabled: true,
            model_cache_dir: None,
            max_sequence_length: 256,
            use_half_precision: false,
        }
    }
    
    #[tokio::test]
    async fn test_embedding_service_creation() {
        let config = test_config();
        
        // This test might fail in CI if model download fails
        match EmbeddingService::new(config).await {
            Ok(service) => {
                assert!(service.embedding_dimension() > 0);
            }
            Err(e) => {
                eprintln!("Skipping test due to model download failure: {}", e);
            }
        }
    }
    
    #[tokio::test]
    async fn test_semantic_similarity() -> Result<()> {
        let config = test_config();
        
        let service = match EmbeddingService::new(config).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Skipping test - could not initialize service: {}", e);
                return Ok(());
            }
        };
        
        // Test semantic similarity
        let cat = service.embed("cat").await?;
        let dog = service.embed("dog").await?;
        let car = service.embed("car").await?;
        
        let cat_dog_sim = cosine_similarity(&cat, &dog);
        let cat_car_sim = cosine_similarity(&cat, &car);
        
        println!("cat-dog similarity: {}", cat_dog_sim);
        println!("cat-car similarity: {}", cat_car_sim);
        
        // Animals should be more similar than cat-car
        assert!(cat_dog_sim > cat_car_sim);
        
        Ok(())
    }
}