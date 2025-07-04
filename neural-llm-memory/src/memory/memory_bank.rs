//! Memory bank implementation for storing and managing memories

use crate::memory::{MemoryKey, MemoryValue, MemoryOperations, MemoryMetadata};
use dashmap::DashMap;
use lru::LruCache;
use parking_lot::Mutex;
use ndarray::Array2;
use std::sync::Arc;
use std::num::NonZeroUsize;

/// Entry in the memory bank
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub key: MemoryKey,
    pub value: MemoryValue,
    pub embedding_array: Array2<f32>,
}

/// High-performance memory bank with concurrent access
pub struct MemoryBank {
    // Primary storage using DashMap for concurrent access
    storage: Arc<DashMap<MemoryKey, MemoryEntry>>,
    
    // LRU cache for frequently accessed memories
    lru_cache: Arc<Mutex<LruCache<MemoryKey, MemoryEntry>>>,
    
    // Configuration
    max_memories: usize,
    cache_size: usize,
    
    // Statistics
    total_accesses: Arc<Mutex<u64>>,
    cache_hits: Arc<Mutex<u64>>,
}

impl MemoryBank {
    pub fn new(max_memories: usize, cache_size: usize) -> Self {
        let cache_size_nonzero = NonZeroUsize::new(cache_size).unwrap_or(NonZeroUsize::new(100).unwrap());
        
        Self {
            storage: Arc::new(DashMap::with_capacity(max_memories)),
            lru_cache: Arc::new(Mutex::new(LruCache::new(cache_size_nonzero))),
            max_memories,
            cache_size,
            total_accesses: Arc::new(Mutex::new(0)),
            cache_hits: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Convert vector to ndarray
    fn vec_to_array(vec: &[f32]) -> Array2<f32> {
        Array2::from_shape_vec((1, vec.len()), vec.to_vec())
            .expect("Failed to convert vector to array")
    }
    
    /// Compute similarity between two embeddings
    fn compute_similarity(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        // Cosine similarity
        let dot_product = (a * b).sum();
        let norm_a = (a * a).sum().sqrt();
        let norm_b = (b * b).sum().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
    
    /// Apply memory decay based on time and access patterns
    fn apply_decay(&self, metadata: &mut MemoryMetadata) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let time_since_access = (current_time - metadata.last_accessed) as f32;
        let decay_amount = time_since_access * (1.0 - metadata.decay_factor) / 3600.0; // Hourly decay
        
        metadata.importance = (metadata.importance - decay_amount).max(0.1);
    }
    
    /// Evict least important memory when at capacity
    fn evict_if_needed(&self) {
        if self.storage.len() >= self.max_memories {
            // Find memory with lowest importance score
            let mut min_importance = f32::MAX;
            let mut evict_key = None;
            
            for entry in self.storage.iter() {
                let importance = entry.value().value.metadata.importance;
                if importance < min_importance {
                    min_importance = importance;
                    evict_key = Some(entry.key().clone());
                }
            }
            
            if let Some(key) = evict_key {
                self.storage.remove(&key);
                
                // Also remove from cache
                let mut cache = self.lru_cache.lock();
                cache.pop(&key);
            }
        }
    }
    
    pub fn get_statistics(&self) -> (u64, u64, f32) {
        let total = *self.total_accesses.lock();
        let hits = *self.cache_hits.lock();
        let hit_rate = if total > 0 {
            hits as f32 / total as f32
        } else {
            0.0
        };
        
        (total, hits, hit_rate)
    }
}

impl MemoryOperations for MemoryBank {
    fn store(&mut self, key: MemoryKey, mut value: MemoryValue) -> crate::Result<()> {
        self.evict_if_needed();
        
        // Initialize metadata
        value.metadata.last_accessed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let embedding_array = Self::vec_to_array(&value.embedding);
        
        let entry = MemoryEntry {
            key: key.clone(),
            value: value.clone(),
            embedding_array,
        };
        
        // Store in main storage
        self.storage.insert(key.clone(), entry.clone());
        
        // Add to cache
        let mut cache = self.lru_cache.lock();
        cache.put(key, entry);
        
        Ok(())
    }
    
    fn retrieve(&mut self, key: &MemoryKey) -> crate::Result<Option<MemoryValue>> {
        *self.total_accesses.lock() += 1;
        
        // Check cache first
        {
            let mut cache = self.lru_cache.lock();
            if let Some(entry) = cache.get_mut(key) {
                *self.cache_hits.lock() += 1;
                
                // Update access metadata
                entry.value.metadata.access_count += 1;
                entry.value.metadata.last_accessed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                
                return Ok(Some(entry.value.clone()));
            }
        }
        
        // Not in cache, check main storage
        if let Some(mut entry) = self.storage.get_mut(key) {
            // Update access metadata
            entry.value.metadata.access_count += 1;
            entry.value.metadata.last_accessed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            // Apply decay
            self.apply_decay(&mut entry.value.metadata);
            
            let value = entry.value.clone();
            
            // Add to cache
            let mut cache = self.lru_cache.lock();
            cache.put(key.clone(), entry.clone());
            
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }
    
    fn search(&self, query_embedding: &Array2<f32>, k: usize) -> Vec<(MemoryKey, MemoryValue, f32)> {
        use rayon::prelude::*;
        
        // Compute similarities in parallel
        let mut results: Vec<_> = self.storage
            .iter()
            .par_bridge()
            .map(|entry| {
                let similarity = Self::compute_similarity(query_embedding, &entry.embedding_array);
                let weighted_score = similarity * entry.value.metadata.importance;
                (entry.key().clone(), entry.value.clone(), weighted_score)
            })
            .collect();
        
        // Sort by similarity score (descending)
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top k results
        results.truncate(k);
        results
    }
    
    fn update(&mut self, key: &MemoryKey, update_fn: impl FnOnce(&mut MemoryValue)) -> crate::Result<()> {
        if let Some(mut entry) = self.storage.get_mut(key) {
            update_fn(&mut entry.value);
            
            // Update embedding array if embedding changed
            entry.embedding_array = Self::vec_to_array(&entry.value.embedding);
            
            // Update cache if present
            let mut cache = self.lru_cache.lock();
            if let Some(cached_entry) = cache.get_mut(key) {
                cached_entry.value = entry.value.clone();
                cached_entry.embedding_array = entry.embedding_array.clone();
            }
            
            Ok(())
        } else {
            Err(crate::MemoryFrameworkError {
                message: format!("Memory key not found: {:?}", key),
            })
        }
    }
    
    fn delete(&mut self, key: &MemoryKey) -> crate::Result<bool> {
        let removed = self.storage.remove(key).is_some();
        
        if removed {
            let mut cache = self.lru_cache.lock();
            cache.pop(key);
        }
        
        Ok(removed)
    }
    
    fn clear(&mut self) {
        self.storage.clear();
        let mut cache = self.lru_cache.lock();
        cache.clear();
        *self.total_accesses.lock() = 0;
        *self.cache_hits.lock() = 0;
    }
    
    fn size(&self) -> usize {
        self.storage.len()
    }
}

impl MemoryBank {
    /// Get statistics about the memory bank
    pub fn get_stats(&self) -> (usize, u64, u64, f64) {
        let total_accesses = *self.total_accesses.lock();
        let cache_hits = *self.cache_hits.lock();
        let hit_rate = if total_accesses > 0 {
            cache_hits as f64 / total_accesses as f64
        } else {
            0.0
        };
        
        (self.storage.len(), total_accesses, cache_hits, hit_rate)
    }
    
    /// Get all memories for persistence
    pub fn get_all_memories(&self) -> Vec<(MemoryKey, MemoryValue)> {
        self.storage
            .iter()
            .map(|entry| {
                let key = entry.key().clone();
                let memory_entry = &entry.value();
                let value = MemoryValue {
                    embedding: memory_entry.embedding_array.as_slice().unwrap().to_vec(),
                    content: memory_entry.value.content.clone(),
                    metadata: memory_entry.value.metadata.clone(),
                };
                (key, value)
            })
            .collect()
    }
}