use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum OperationType {
    Store,
    Retrieve,
    Search,
    Update,
    Delete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetrics {
    pub operation_id: String,
    pub operation: OperationType,
    pub input_size: usize,
    pub response_time_ms: f32,
    pub memory_delta_bytes: i64,
    pub similarity_scores: Vec<f32>,
    pub cache_hit: bool,
    pub timestamp: DateTime<Utc>,
    pub feedback_score: Option<f32>,
}

#[derive(Debug)]
pub struct UsageCollector {
    metrics: RwLock<VecDeque<UsageMetrics>>,
    max_samples: usize,
    
    // Atomic counters for lock-free stats
    total_operations: AtomicUsize,
    total_cache_hits: AtomicUsize,
    operation_counts: Arc<RwLock<HashMap<OperationType, usize>>>,
    
    // Aggregated stats (updated periodically)
    avg_response_time: RwLock<f32>,
    avg_memory_usage: RwLock<f32>,
    cache_hit_rate: RwLock<f32>,
}

impl UsageCollector {
    pub fn new(max_samples: usize) -> Self {
        Self {
            metrics: RwLock::new(VecDeque::with_capacity(max_samples)),
            max_samples,
            total_operations: AtomicUsize::new(0),
            total_cache_hits: AtomicUsize::new(0),
            operation_counts: Arc::new(RwLock::new(HashMap::new())),
            avg_response_time: RwLock::new(0.0),
            avg_memory_usage: RwLock::new(0.0),
            cache_hit_rate: RwLock::new(0.0),
        }
    }
    
    pub async fn record_metric(&self, metric: UsageMetrics) {
        // Update atomic counters
        self.total_operations.fetch_add(1, Ordering::Relaxed);
        if metric.cache_hit {
            self.total_cache_hits.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update operation counts
        {
            let mut counts = self.operation_counts.write().await;
            *counts.entry(metric.operation.clone()).or_insert(0) += 1;
        }
        
        // Add to metrics queue
        {
            let mut metrics = self.metrics.write().await;
            if metrics.len() >= self.max_samples {
                metrics.pop_front();
            }
            metrics.push_back(metric);
        }
        
        // Update aggregated stats periodically
        if self.total_operations.load(Ordering::Relaxed) % 100 == 0 {
            self.update_aggregated_stats().await;
        }
    }
    
    async fn update_aggregated_stats(&self) {
        let metrics = self.metrics.read().await;
        
        if metrics.is_empty() {
            return;
        }
        
        // Calculate average response time
        let avg_response = metrics.iter()
            .map(|m| m.response_time_ms)
            .sum::<f32>() / metrics.len() as f32;
        *self.avg_response_time.write().await = avg_response;
        
        // Calculate average memory usage
        let avg_memory = metrics.iter()
            .map(|m| m.memory_delta_bytes.abs() as f32)
            .sum::<f32>() / metrics.len() as f32;
        *self.avg_memory_usage.write().await = avg_memory;
        
        // Calculate cache hit rate
        let total_ops = self.total_operations.load(Ordering::Relaxed) as f32;
        let cache_hits = self.total_cache_hits.load(Ordering::Relaxed) as f32;
        *self.cache_hit_rate.write().await = if total_ops > 0.0 {
            cache_hits / total_ops
        } else {
            0.0
        };
    }
    
    pub async fn get_stats(&self) -> UsageStats {
        UsageStats {
            total_operations: self.total_operations.load(Ordering::Relaxed),
            avg_response_time_ms: *self.avg_response_time.read().await,
            avg_memory_usage_bytes: *self.avg_memory_usage.read().await,
            cache_hit_rate: *self.cache_hit_rate.read().await,
            operation_counts: self.operation_counts.read().await.clone(),
        }
    }
    
    pub async fn get_recent_metrics(&self, count: usize) -> Vec<UsageMetrics> {
        let metrics = self.metrics.read().await;
        metrics.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
    
    pub async fn get_training_data(&self) -> TrainingCorpus {
        let metrics = self.metrics.read().await;
        
        // Group metrics by operation type for balanced training
        let mut by_operation: HashMap<OperationType, Vec<UsageMetrics>> = HashMap::new();
        
        for metric in metrics.iter() {
            by_operation.entry(metric.operation.clone())
                .or_insert_with(Vec::new)
                .push(metric.clone());
        }
        
        TrainingCorpus {
            metrics: metrics.iter().cloned().collect(),
            by_operation,
            stats: UsageStats {
                total_operations: self.total_operations.load(Ordering::Relaxed),
                avg_response_time_ms: *self.avg_response_time.read().await,
                avg_memory_usage_bytes: *self.avg_memory_usage.read().await,
                cache_hit_rate: *self.cache_hit_rate.read().await,
                operation_counts: self.operation_counts.read().await.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    pub total_operations: usize,
    pub avg_response_time_ms: f32,
    pub avg_memory_usage_bytes: f32,
    pub cache_hit_rate: f32,
    pub operation_counts: HashMap<OperationType, usize>,
}

#[derive(Debug, Clone)]
pub struct TrainingCorpus {
    pub metrics: Vec<UsageMetrics>,
    pub by_operation: HashMap<OperationType, Vec<UsageMetrics>>,
    pub stats: UsageStats,
}