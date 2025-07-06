//! Extension methods for PersistentMemoryModule to support adaptive operations

use super::{PersistentMemoryModule, MemoryConfig, MemoryValue};
use super::key_value_store::KeyValueStore;
use crate::Result;
use ndarray::{Array2, Array1};
use std::sync::Arc;
use tokio::sync::RwLock;

// Add key-value store to PersistentMemoryModule
lazy_static::lazy_static! {
    static ref KEY_VALUE_STORES: Arc<RwLock<HashMap<usize, KeyValueStore>>> = 
        Arc::new(RwLock::new(HashMap::new()));
}

use std::collections::HashMap;

impl PersistentMemoryModule {
    /// Get or create key-value store for this instance
    async fn get_kv_store(&self) -> KeyValueStore {
        let ptr = self as *const _ as usize;
        let mut stores = KEY_VALUE_STORES.write().await;
        
        stores.entry(ptr)
            .or_insert_with(|| KeyValueStore::new())
            .clone()
    }
    
    /// Store method for compatibility with adaptive module
    pub async fn store(&self, key: &str, content: &str) -> Result<()> {
        // Store in key-value store
        let kv_store = self.get_kv_store().await;
        kv_store.store(key.to_string(), content.to_string()).await?;
        
        // Also store in neural memory for search capability
        let embedding = self.create_embedding(content);
        self.store_memory(content.to_string(), embedding).await?;
        Ok(())
    }
    
    /// Retrieve method for compatibility with adaptive module
    pub async fn retrieve(&self, key: &str) -> Result<Option<String>> {
        // First try key-value store
        let kv_store = self.get_kv_store().await;
        Ok(kv_store.retrieve(key).await?)
    }
    
    /// Search method for compatibility with adaptive module
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<(String, f32)>> {
        // Create query embedding
        let query_embedding = self.create_embedding(query);
        
        // Retrieve with attention
        let results = self.retrieve_with_attention(&query_embedding, limit).await;
        
        // Calculate proper cosine similarity scores
        let query_vec = query_embedding.as_slice().unwrap();
        
        Ok(results.into_iter()
            .map(|(_, value, attention)| {
                // Calculate cosine similarity between query and stored embeddings
                let stored_vec = &value.embedding;
                let similarity = cosine_similarity(query_vec, stored_vec);
                (value.content, similarity)
            })
            .collect())
    }
    
    /// Get config for architecture swapping (legacy method)
    pub fn get_config_legacy(&self) -> &MemoryConfig {
        self.get_config()
    }
    
    /// Create a proper embedding from text using neural network
    fn create_embedding(&self, text: &str) -> Array2<f32> {
        // Use the existing simple embedding for now
        // In production, this would use the neural network encoder
        self.create_simple_embedding(text)
    }
    
    /// Create a simple embedding from text (placeholder implementation)
    fn create_simple_embedding(&self, text: &str) -> Array2<f32> {
        // Simple hash-based embedding for demo purposes
        let embedding_dim = self.get_config().embedding_dim;
        let mut embedding = vec![0.0; embedding_dim];
        
        // Simple hash-based features
        for (i, ch) in text.chars().enumerate() {
            let idx = (ch as usize + i) % embedding_dim;
            embedding[idx] += 1.0;
        }
        
        // Normalize to unit vector for cosine similarity
        let sum: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if sum > 0.0 {
            for val in &mut embedding {
                *val /= sum;
            }
        }
        
        Array2::from_shape_vec((1, embedding_dim), embedding)
            .unwrap_or_else(|_| Array2::zeros((1, embedding_dim)))
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}