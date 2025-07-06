//! Extension methods for PersistentMemoryModule to support adaptive operations

use super::{PersistentMemoryModule, MemoryConfig};
use crate::Result;
use ndarray::Array2;

impl PersistentMemoryModule {
    /// Store method for compatibility with adaptive module
    pub async fn store(&self, key: &str, content: &str) -> Result<()> {
        // Create a simple embedding based on key+content
        let text = format!("{} {}", key, content);
        let embedding = self.create_simple_embedding(&text);
        
        self.store_memory(content.to_string(), embedding).await?;
        Ok(())
    }
    
    /// Retrieve method for compatibility with adaptive module
    pub async fn retrieve(&self, key: &str) -> Result<Option<String>> {
        // Create query embedding from key
        let query_embedding = self.create_simple_embedding(key);
        
        // Search for most similar
        let results = self.retrieve_with_attention(&query_embedding, 1).await;
        
        if let Some((_, value, _)) = results.first() {
            Ok(Some(value.content.clone()))
        } else {
            Ok(None)
        }
    }
    
    /// Search method for compatibility with adaptive module
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<(String, f32)>> {
        // Create query embedding
        let query_embedding = self.create_simple_embedding(query);
        
        // Retrieve with attention
        let results = self.retrieve_with_attention(&query_embedding, limit).await;
        
        // Convert to expected format
        Ok(results.into_iter()
            .map(|(key, value, _)| {
                // Calculate simple similarity score
                let score = 1.0 / (1.0 + key.id.len() as f32);
                (value.content, score)
            })
            .collect())
    }
    
    /// Get config for architecture swapping (legacy method)
    pub fn get_config_legacy(&self) -> &MemoryConfig {
        self.get_config()
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
        
        // Normalize
        let sum: f32 = embedding.iter().sum();
        if sum > 0.0 {
            for val in &mut embedding {
                *val /= sum;
            }
        }
        
        Array2::from_shape_vec((1, embedding_dim), embedding)
            .unwrap_or_else(|_| Array2::zeros((1, embedding_dim)))
    }
}