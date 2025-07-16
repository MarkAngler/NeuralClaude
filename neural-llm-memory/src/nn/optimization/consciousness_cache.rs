//! Consciousness Context Cache
//! 
//! High-performance LRU cache for consciousness contexts with minimal lock contention.
//! Implements thread-safe caching with configurable size and TTL.

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use ndarray::Array1;

/// Configuration for consciousness cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries to cache
    pub max_entries: usize,
    /// Time-to-live for cached entries
    pub ttl_seconds: u64,
    /// Whether to use SIMD optimizations
    pub use_simd: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl_seconds: 60,
            use_simd: true,
        }
    }
}

/// Cached consciousness context entry
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached context data
    context: ConsciousnessContext,
    /// When this entry was created
    created_at: Instant,
    /// Last access time for LRU tracking
    last_accessed: Instant,
    /// Access count for statistics
    access_count: u64,
}

/// Consciousness context data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessContext {
    /// Context identifier
    pub id: String,
    /// Semantic embedding
    pub embedding: Array1<f32>,
    /// Attention weights
    pub attention_weights: Vec<f32>,
    /// Metadata about the context
    pub metadata: HashMap<String, String>,
    /// Pre-computed attention scores (for performance)
    pub cached_scores: Option<Vec<f32>>,
}

/// Thread-safe LRU cache for consciousness contexts
pub struct ConsciousnessCache {
    /// Internal storage with read-write lock
    storage: Arc<RwLock<CacheStorage>>,
    /// Cache configuration
    config: CacheConfig,
    /// Performance metrics
    metrics: Arc<RwLock<CacheMetrics>>,
}

/// Internal cache storage
struct CacheStorage {
    /// Main storage map
    entries: HashMap<String, CacheEntry>,
    /// LRU tracking - stores keys in order of recent use
    lru_order: Vec<String>,
}

/// Cache performance metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Total number of cache hits
    pub hits: u64,
    /// Total number of cache misses
    pub misses: u64,
    /// Total number of evictions
    pub evictions: u64,
    /// Average access time in microseconds
    pub avg_access_time_us: f64,
    /// Total number of entries currently cached
    pub current_entries: usize,
}

impl ConsciousnessCache {
    /// Create new cache with default configuration
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create new cache with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            storage: Arc::new(RwLock::new(CacheStorage {
                entries: HashMap::with_capacity(config.max_entries),
                lru_order: Vec::with_capacity(config.max_entries),
            })),
            config,
            metrics: Arc::new(RwLock::new(CacheMetrics::default())),
        }
    }

    /// Get a context from cache
    pub fn get(&self, key: &str) -> Option<ConsciousnessContext> {
        let start = Instant::now();
        
        // Use write lock for all operations
        let mut storage = self.storage.write().unwrap();
        
        let result = if let Some(entry) = storage.entries.get(key) {
            // Check TTL
            if entry.created_at.elapsed() > Duration::from_secs(self.config.ttl_seconds) {
                // Entry expired, remove it
                storage.entries.remove(key);
                storage.lru_order.retain(|k| k != key);
                None
            } else {
                // Clone the context before updating
                let context = entry.context.clone();
                
                // Update access time and count
                if let Some(entry) = storage.entries.get_mut(key) {
                    entry.last_accessed = Instant::now();
                    entry.access_count += 1;
                }
                
                // Move to front of LRU order
                storage.lru_order.retain(|k| k != key);
                storage.lru_order.insert(0, key.to_string());
                
                Some(context)
            }
        } else {
            None
        };
        
        // Release the lock early
        drop(storage);

        // Update metrics
        let elapsed = start.elapsed().as_micros() as f64;
        let mut metrics = self.metrics.write().unwrap();
        
        if result.is_some() {
            metrics.hits += 1;
        } else {
            metrics.misses += 1;
        }
        
        // Update average access time
        let total_accesses = metrics.hits + metrics.misses;
        metrics.avg_access_time_us = 
            (metrics.avg_access_time_us * (total_accesses - 1) as f64 + elapsed) / total_accesses as f64;
        
        result
    }

    /// Insert a context into cache
    pub fn insert(&self, key: String, context: ConsciousnessContext) {
        let mut storage = self.storage.write().unwrap();
        let mut metrics = self.metrics.write().unwrap();
        
        // Check if we need to evict entries
        if storage.entries.len() >= self.config.max_entries && !storage.entries.contains_key(&key) {
            // Evict least recently used entry
            if let Some(lru_key) = storage.lru_order.pop() {
                storage.entries.remove(&lru_key);
                metrics.evictions += 1;
            }
        }
        
        // Insert new entry
        let now = Instant::now();
        storage.entries.insert(key.clone(), CacheEntry {
            context,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        });
        
        // Update LRU order
        storage.lru_order.retain(|k| k != &key);
        storage.lru_order.insert(0, key);
        
        // Update metrics
        metrics.current_entries = storage.entries.len();
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        let mut storage = self.storage.write().unwrap();
        storage.entries.clear();
        storage.lru_order.clear();
        
        let mut metrics = self.metrics.write().unwrap();
        metrics.current_entries = 0;
    }

    /// Get current cache metrics
    pub fn metrics(&self) -> CacheMetrics {
        let metrics = self.metrics.read().unwrap();
        metrics.clone()
    }

    /// Remove expired entries
    pub fn cleanup_expired(&self) {
        let mut storage = self.storage.write().unwrap();
        let ttl = Duration::from_secs(self.config.ttl_seconds);
        
        let expired_keys: Vec<String> = storage.entries
            .iter()
            .filter(|(_, entry)| entry.created_at.elapsed() > ttl)
            .map(|(key, _)| key.clone())
            .collect();
        
        for key in expired_keys {
            storage.entries.remove(&key);
            storage.lru_order.retain(|k| k != &key);
        }
        
        let mut metrics = self.metrics.write().unwrap();
        metrics.current_entries = storage.entries.len();
    }

    /// Precompute attention scores for cached contexts using SIMD if available
    pub fn precompute_attention_scores(&self, query_embedding: &Array1<f32>) {
        if !self.config.use_simd {
            return;
        }

        let mut storage = self.storage.write().unwrap();
        
        for (_, entry) in storage.entries.iter_mut() {
            if entry.context.cached_scores.is_none() {
                let scores = compute_attention_scores_simd(
                    query_embedding,
                    &entry.context.embedding,
                    &entry.context.attention_weights,
                );
                entry.context.cached_scores = Some(scores);
            }
        }
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let metrics = self.metrics.read().unwrap();
        let total = metrics.hits + metrics.misses;
        if total == 0 {
            0.0
        } else {
            metrics.hits as f64 / total as f64
        }
    }

    /// Get cache efficiency score (combines hit rate and access time)
    pub fn efficiency_score(&self) -> f64 {
        let hit_rate = self.hit_rate();
        let metrics = self.metrics.read().unwrap();
        
        // Normalize access time (assume 100us is good, 1000us is bad)
        let time_score = 1.0 - (metrics.avg_access_time_us - 100.0).max(0.0) / 900.0;
        
        // Combine hit rate and time score
        hit_rate * 0.7 + time_score.max(0.0).min(1.0) * 0.3
    }
}

/// Compute attention scores using SIMD operations where available
fn compute_attention_scores_simd(
    query: &Array1<f32>,
    key: &Array1<f32>,
    weights: &[f32],
) -> Vec<f32> {
    // For now, use standard operations
    // SIMD implementation would go here when std::simd is stable
    compute_attention_scores_scalar(query, key, weights)
}

/// Fallback scalar implementation for attention score computation
fn compute_attention_scores_scalar(
    query: &Array1<f32>,
    key: &Array1<f32>,
    weights: &[f32],
) -> Vec<f32> {
    let dim = query.len().min(key.len());
    let mut scores = Vec::with_capacity(weights.len());
    
    // Compute dot product
    let mut dot_product = 0.0;
    for i in 0..dim {
        dot_product += query[i] * key[i];
    }
    
    // Apply weights and scaling
    let scale = (dim as f32).sqrt();
    for weight in weights {
        scores.push((dot_product / scale) * weight);
    }
    
    // Softmax normalization
    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp: f32 = exp_scores.iter().sum();
    
    exp_scores.iter().map(|&s| s / sum_exp).collect()
}

// SIMD implementations for when portable_simd is stable
#[cfg(feature = "portable_simd")]
mod simd_ops {
    use std::simd::{f32x8, SimdFloat};
    use ndarray::Array1;

    /// SIMD-accelerated dot product
    pub fn dot_product_simd(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let len = a.len().min(b.len());
        let chunks = len / 8;
        let remainder = len % 8;
        
        let mut sum = f32x8::splat(0.0);
        
        // Process 8 elements at a time
        for i in 0..chunks {
            let idx = i * 8;
            let a_chunk = f32x8::from_slice(&a.as_slice().unwrap()[idx..idx + 8]);
            let b_chunk = f32x8::from_slice(&b.as_slice().unwrap()[idx..idx + 8]);
            sum += a_chunk * b_chunk;
        }
        
        // Sum the SIMD vector
        let mut result = sum.reduce_sum();
        
        // Handle remainder
        let start = chunks * 8;
        for i in 0..remainder {
            result += a[start + i] * b[start + i];
        }
        
        result
    }

    /// SIMD-accelerated softmax
    pub fn softmax_simd(scores: &mut [f32]) {
        // Find max using SIMD
        let max = scores.chunks(8)
            .map(|chunk| {
                if chunk.len() == 8 {
                    let v = f32x8::from_slice(chunk);
                    v.reduce_max()
                } else {
                    chunk.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
                }
            })
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        
        // Subtract max and exp using SIMD
        let mut sum = 0.0;
        for chunk in scores.chunks_mut(8) {
            if chunk.len() == 8 {
                let mut v = f32x8::from_slice(chunk);
                v -= f32x8::splat(max);
                v = v.exp();
                sum += v.reduce_sum();
                v.copy_to_slice(chunk);
            } else {
                for val in chunk {
                    *val = (*val - max).exp();
                    sum += *val;
                }
            }
        }
        
        // Normalize using SIMD
        let inv_sum = f32x8::splat(1.0 / sum);
        for chunk in scores.chunks_mut(8) {
            if chunk.len() == 8 {
                let mut v = f32x8::from_slice(chunk);
                v *= inv_sum;
                v.copy_to_slice(chunk);
            } else {
                for val in chunk {
                    *val /= sum;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_cache_basic_operations() {
        let cache = ConsciousnessCache::new();
        
        let context = ConsciousnessContext {
            id: "test1".to_string(),
            embedding: Array1::from_vec(vec![0.1; 768]),
            attention_weights: vec![0.5; 12],
            metadata: HashMap::new(),
            cached_scores: None,
        };
        
        // Insert and retrieve
        cache.insert("key1".to_string(), context.clone());
        let retrieved = cache.get("key1").unwrap();
        assert_eq!(retrieved.id, "test1");
        
        // Check metrics
        let metrics = cache.metrics();
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.current_entries, 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut config = CacheConfig::default();
        config.max_entries = 3;
        let cache = ConsciousnessCache::with_config(config);
        
        // Insert entries up to capacity
        for i in 0..4 {
            let context = ConsciousnessContext {
                id: format!("test{}", i),
                embedding: Array1::from_vec(vec![0.1; 768]),
                attention_weights: vec![0.5; 12],
                metadata: HashMap::new(),
                cached_scores: None,
            };
            cache.insert(format!("key{}", i), context);
        }
        
        // First entry should be evicted
        assert!(cache.get("key0").is_none());
        assert!(cache.get("key1").is_some());
        assert!(cache.get("key2").is_some());
        assert!(cache.get("key3").is_some());
        
        let metrics = cache.metrics();
        assert_eq!(metrics.evictions, 1);
        assert_eq!(metrics.current_entries, 3);
    }

    #[test]
    fn test_cache_ttl() {
        let mut config = CacheConfig::default();
        config.ttl_seconds = 0; // Immediate expiration
        let cache = ConsciousnessCache::with_config(config);
        
        let context = ConsciousnessContext {
            id: "test1".to_string(),
            embedding: Array1::from_vec(vec![0.1; 768]),
            attention_weights: vec![0.5; 12],
            metadata: HashMap::new(),
            cached_scores: None,
        };
        
        cache.insert("key1".to_string(), context);
        
        // Sleep to ensure expiration
        std::thread::sleep(Duration::from_millis(10));
        
        // Should be expired
        assert!(cache.get("key1").is_none());
    }

    #[test]
    fn test_attention_score_computation() {
        let query = Array1::from_vec(vec![0.5; 768]);
        let key = Array1::from_vec(vec![0.3; 768]);
        let weights = vec![1.0, 0.8, 0.6];
        
        let scores = compute_attention_scores_scalar(&query, &key, &weights);
        
        assert_eq!(scores.len(), 3);
        assert!((scores.iter().sum::<f32>() - 1.0).abs() < 1e-6); // Should sum to 1
    }

    #[test]
    fn test_cache_efficiency() {
        let cache = ConsciousnessCache::new();
        
        let context = ConsciousnessContext {
            id: "test1".to_string(),
            embedding: Array1::from_vec(vec![0.1; 768]),
            attention_weights: vec![0.5; 12],
            metadata: HashMap::new(),
            cached_scores: None,
        };
        
        // Multiple hits
        cache.insert("key1".to_string(), context);
        for _ in 0..10 {
            cache.get("key1");
        }
        
        let hit_rate = cache.hit_rate();
        assert!(hit_rate > 0.9); // Should have high hit rate
        
        let efficiency = cache.efficiency_score();
        assert!(efficiency > 0.5); // Should have reasonable efficiency
    }
}