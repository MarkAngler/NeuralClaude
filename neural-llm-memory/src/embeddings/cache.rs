//! Caching system for embeddings to avoid recomputation

use std::path::PathBuf;
use std::sync::Arc;
use dashmap::DashMap;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use cacache;

/// A cache entry containing embedding data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    /// The embedding vector
    embedding: Vec<f32>,
    /// Timestamp when the entry was created
    timestamp: u64,
    /// Model ID used to generate this embedding
    model_id: String,
}

/// Embedding cache that supports both in-memory and disk caching
pub struct EmbeddingCache {
    /// In-memory LRU cache for fast access
    memory_cache: Arc<DashMap<String, Vec<f32>>>,
    /// Disk cache directory
    disk_cache_dir: Option<PathBuf>,
    /// Maximum cache size in bytes
    max_size: usize,
    /// Current model ID for cache invalidation
    model_id: String,
}

impl EmbeddingCache {
    /// Create a new embedding cache
    pub fn new(
        disk_cache_dir: Option<PathBuf>,
        max_size_mb: usize,
        model_id: String,
    ) -> Result<Self> {
        // Create disk cache directory if specified
        if let Some(ref dir) = disk_cache_dir {
            std::fs::create_dir_all(dir)?;
        }
        
        Ok(Self {
            memory_cache: Arc::new(DashMap::new()),
            disk_cache_dir,
            max_size: max_size_mb * 1024 * 1024,
            model_id,
        })
    }
    
    /// Generate a cache key from text
    fn cache_key(&self, text: &str) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        self.model_id.hash(&mut hasher);
        format!("emb_{:x}", hasher.finish())
    }
    
    /// Get an embedding from the cache
    pub async fn get(&self, text: &str) -> Option<Vec<f32>> {
        let key = self.cache_key(text);
        
        // Check memory cache first
        if let Some(entry) = self.memory_cache.get(&key) {
            return Some(entry.clone());
        }
        
        // Check disk cache if available
        if let Some(ref cache_dir) = self.disk_cache_dir {
            if let Ok(data) = cacache::read(cache_dir, &key).await {
                if let Ok(entry) = bincode::deserialize::<CacheEntry>(&data) {
                    // Verify model ID matches
                    if entry.model_id == self.model_id {
                        // Store in memory cache for faster access
                        self.memory_cache.insert(key.clone(), entry.embedding.clone());
                        return Some(entry.embedding);
                    }
                }
            }
        }
        
        None
    }
    
    /// Store an embedding in the cache
    pub async fn put(&self, text: &str, embedding: Vec<f32>) -> Result<()> {
        let key = self.cache_key(text);
        
        // Store in memory cache
        self.memory_cache.insert(key.clone(), embedding.clone());
        
        // Store in disk cache if available
        if let Some(ref cache_dir) = self.disk_cache_dir {
            let entry = CacheEntry {
                embedding,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs(),
                model_id: self.model_id.clone(),
            };
            
            let data = bincode::serialize(&entry)?;
            cacache::write(cache_dir, &key, data).await?;
            
            // Optionally clean up old entries if cache is too large
            self.cleanup_if_needed(cache_dir).await?;
        }
        
        Ok(())
    }
    
    /// Clean up old cache entries if cache exceeds max size
    async fn cleanup_if_needed(&self, _cache_dir: &PathBuf) -> Result<()> {
        // Note: cacache API has changed - for now just return Ok
        // TODO: Implement proper cleanup with updated cacache API
        Ok(())
    }
    
    /// Clear all cached embeddings
    pub async fn clear(&self) -> Result<()> {
        self.memory_cache.clear();
        
        if let Some(ref cache_dir) = self.disk_cache_dir {
            cacache::clear(cache_dir).await?;
        }
        
        Ok(())
    }
    
    /// Get cache statistics
    pub async fn stats(&self) -> Result<CacheStats> {
        let memory_entries = self.memory_cache.len();
        // Note: cacache API has changed - returning basic stats for now
        let disk_entries = 0;
        let disk_size = 0;
        
        Ok(CacheStats {
            memory_entries,
            disk_entries,
            disk_size_bytes: disk_size,
        })
    }
}

/// Statistics about the cache
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of entries in memory cache
    pub memory_entries: usize,
    /// Number of entries in disk cache
    pub disk_entries: usize,
    /// Total size of disk cache in bytes
    pub disk_size_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_memory_cache() -> Result<()> {
        let cache = EmbeddingCache::new(None, 100, "test-model".to_string())?;
        
        let text = "Hello, world!";
        let embedding = vec![0.1, 0.2, 0.3];
        
        // Store and retrieve
        cache.put(text, embedding.clone()).await?;
        let retrieved = cache.get(text).await;
        
        assert_eq!(retrieved, Some(embedding));
        Ok(())
    }
    
    #[tokio::test]
    async fn test_disk_cache() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let cache = EmbeddingCache::new(
            Some(temp_dir.path().to_path_buf()),
            100,
            "test-model".to_string()
        )?;
        
        let text = "Hello, world!";
        let embedding = vec![0.1, 0.2, 0.3];
        
        // Store
        cache.put(text, embedding.clone()).await?;
        
        // Clear memory cache to force disk lookup
        cache.memory_cache.clear();
        
        // Retrieve from disk
        let retrieved = cache.get(text).await;
        assert_eq!(retrieved, Some(embedding));
        
        Ok(())
    }
}