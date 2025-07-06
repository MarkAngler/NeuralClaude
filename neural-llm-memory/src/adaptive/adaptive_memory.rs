use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, Mutex};
use serde_json::Value;
use uuid::Uuid;
use std::collections::HashMap;

use crate::memory::{PersistentMemoryModule, PersistentConfig, MemoryConfig};
use crate::adaptive::{
    UsageCollector, UsageMetrics, OperationType,
    BackgroundEvolver, EvolvedArchitecture,
    AdaptiveConfig,
    feedback::{OperationFeedback, calculate_fitness_with_feedback},
};

pub struct AdaptiveMemoryModule {
    // Current active memory module
    active_memory: Arc<RwLock<PersistentMemoryModule>>,
    
    // Self-optimizing components
    usage_collector: Arc<UsageCollector>,
    evolver: Arc<Mutex<BackgroundEvolver>>,
    
    // Configuration
    config: Arc<RwLock<AdaptiveConfig>>,
    
    // Operation counter for triggers
    operation_count: Arc<RwLock<usize>>,
    
    // Feedback tracking
    pending_operations: Arc<RwLock<HashMap<String, UsageMetrics>>>,
}

impl AdaptiveMemoryModule {
    pub async fn new(base_config: MemoryConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let persistent_config = PersistentConfig {
            memory_config: base_config,
            storage_path: std::path::PathBuf::from("./adaptive_memory_data"),
            auto_save_interval: 60,
            save_on_write: false,
        };
        let memory = PersistentMemoryModule::new(persistent_config).await?;
        let adaptive_config = AdaptiveConfig::default();
        
        Ok(Self {
            active_memory: Arc::new(RwLock::new(memory)),
            usage_collector: Arc::new(UsageCollector::new(10000)), // Keep last 10k operations
            evolver: Arc::new(Mutex::new(BackgroundEvolver::new())),
            config: Arc::new(RwLock::new(adaptive_config)),
            operation_count: Arc::new(RwLock::new(0)),
            pending_operations: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    pub async fn with_config(base_config: MemoryConfig, adaptive_config: AdaptiveConfig) -> Result<Self, Box<dyn std::error::Error>> {
        adaptive_config.validate()?;
        let persistent_config = PersistentConfig {
            memory_config: base_config,
            storage_path: std::path::PathBuf::from("./adaptive_memory_data"),
            auto_save_interval: 60,
            save_on_write: false,
        };
        let memory = PersistentMemoryModule::new(persistent_config).await?;
        
        Ok(Self {
            active_memory: Arc::new(RwLock::new(memory)),
            usage_collector: Arc::new(UsageCollector::new(10000)),
            evolver: Arc::new(Mutex::new(BackgroundEvolver::new())),
            config: Arc::new(RwLock::new(adaptive_config)),
            operation_count: Arc::new(RwLock::new(0)),
            pending_operations: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    // Core memory operations with metrics collection
    
    pub async fn store(&self, key: &str, content: &str) -> Result<String, Box<dyn std::error::Error>> {
        let operation_id = self.generate_operation_id();
        let start = Instant::now();
        let initial_memory = self.get_memory_usage().await;
        
        // Execute operation
        let result = {
            let memory = self.active_memory.read().await;
            memory.store(key, content).await
        };
        
        // Record metrics
        let response_time = start.elapsed().as_millis() as f32;
        let final_memory = self.get_memory_usage().await;
        
        self.record_operation(
            operation_id.clone(),
            OperationType::Store,
            content.len(),
            response_time,
            final_memory as i64 - initial_memory as i64,
            vec![],
            false,
        ).await;
        
        Ok(result.map(|_| operation_id)?)
    }
    
    pub async fn retrieve(&self, key: &str) -> Result<(Option<String>, String), Box<dyn std::error::Error>> {
        let operation_id = self.generate_operation_id();
        let start = Instant::now();
        
        // Execute operation
        let result = {
            let memory = self.active_memory.read().await;
            memory.retrieve(key).await
        };
        
        // Record metrics
        let response_time = start.elapsed().as_millis() as f32;
        let cache_hit = result.as_ref().map(|r| r.is_some()).unwrap_or(false);
        
        self.record_operation(
            operation_id.clone(),
            OperationType::Retrieve,
            key.len(),
            response_time,
            0,
            vec![],
            cache_hit,
        ).await;
        
        Ok(result.map(|content| (content, operation_id))?)
    }
    
    pub async fn search(&self, query: &str, limit: usize) -> Result<(Vec<(String, f32)>, String), Box<dyn std::error::Error>> {
        let operation_id = self.generate_operation_id();
        let start = Instant::now();
        
        // Execute operation
        let result = {
            let memory = self.active_memory.read().await;
            memory.search(query, limit).await
        };
        
        // Record metrics
        let response_time = start.elapsed().as_millis() as f32;
        let similarity_scores = result.as_ref()
            .map(|r| r.iter().map(|(_, score)| *score).collect())
            .unwrap_or_default();
        
        self.record_operation(
            operation_id.clone(),
            OperationType::Search,
            query.len(),
            response_time,
            0,
            similarity_scores,
            false,
        ).await;
        
        Ok(result.map(|results| (results, operation_id))?)
    }
    
    pub async fn get_stats(&self) -> Result<Value, Box<dyn std::error::Error>> {
        let memory = self.active_memory.read().await;
        let usage_stats = self.usage_collector.get_stats().await;
        let evolution_status = self.evolver.lock().await.get_status().await;
        
        let (size, total, hits, rate) = memory.get_stats().await;
        let memory_stats = serde_json::json!({
            "total_memories": size,
            "total_operations": total,
            "cache_hits": hits,
            "cache_hit_rate": rate
        });
        
        Ok(serde_json::json!({
            "memory_stats": memory_stats,
            "usage_stats": usage_stats,
            "evolution_status": evolution_status,
            "adaptive_enabled": self.config.read().await.enabled,
        }))
    }
    
    // Adaptive-specific operations
    
    pub async fn get_adaptive_status(&self, verbose: bool) -> Result<Value, Box<dyn std::error::Error>> {
        let evolver = self.evolver.lock().await;
        let status = evolver.get_status().await;
        let config = self.config.read().await;
        
        let mut result = serde_json::json!({
            "enabled": config.enabled,
            "evolution": status,
            "operation_count": *self.operation_count.read().await,
            "next_trigger": {
                "operations": config.evolution_after_operations - (*self.operation_count.read().await % config.evolution_after_operations),
                "hours": config.evolution_interval_hours,
            }
        });
        
        if verbose {
            let usage_stats = self.usage_collector.get_stats().await;
            result["usage_stats"] = serde_json::to_value(usage_stats)?;
            result["recent_metrics"] = serde_json::to_value(
                self.usage_collector.get_recent_metrics(10).await
            )?;
        }
        
        Ok(result)
    }
    
    pub async fn trigger_evolution(&self, generations: Option<usize>, force: bool) -> Result<Value, Box<dyn std::error::Error>> {
        let mut config = self.config.write().await;
        
        // Check if we have enough training data
        let training_data = self.usage_collector.get_training_data().await;
        if !force && training_data.metrics.len() < config.min_training_samples {
            return Ok(serde_json::json!({
                "status": "insufficient_data",
                "message": format!("Need {} samples, have {}", config.min_training_samples, training_data.metrics.len()),
            }));
        }
        
        // Override generations if specified
        if let Some(gens) = generations {
            config.generations = gens;
        }
        
        // Get current memory config
        let current_config = {
            let memory = self.active_memory.read().await;
            memory.get_config().clone()
        };
        
        // Start evolution
        let mut evolver = self.evolver.lock().await;
        evolver.start_evolution(config.clone(), training_data, current_config).await?;
        
        Ok(serde_json::json!({
            "status": "started",
            "generations": config.generations,
            "message": "Evolution started in background",
        }))
    }
    
    pub async fn get_insights(&self) -> Result<Value, Box<dyn std::error::Error>> {
        let usage_stats = self.usage_collector.get_stats().await;
        let evolver = self.evolver.lock().await;
        let last_evolution = evolver.get_last_evolution_time().await;
        
        let mut insights = vec![];
        
        // Performance insights
        if usage_stats.avg_response_time_ms > 15.0 {
            insights.push("Response times are above optimal (>15ms). Consider evolution for speed optimization.".to_string());
        }
        
        if usage_stats.cache_hit_rate < 0.3 {
            insights.push("Low cache hit rate. Consider adjusting memory architecture for better caching.".to_string());
        }
        
        // Usage pattern insights
        let dominant_op = usage_stats.operation_counts.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(op, _)| op);
        
        if let Some(op) = dominant_op {
            insights.push(format!("Dominant operation: {:?}. Architecture can be optimized for this pattern.", op));
        }
        
        Ok(serde_json::json!({
            "insights": insights,
            "usage_patterns": {
                "dominant_operation": dominant_op,
                "operation_distribution": usage_stats.operation_counts,
            },
            "performance": {
                "avg_response_ms": usage_stats.avg_response_time_ms,
                "cache_hit_rate": usage_stats.cache_hit_rate,
            },
            "last_evolution": last_evolution,
        }))
    }
    
    pub async fn update_config(&self, objectives: Option<std::collections::HashMap<String, f32>>, enabled: Option<bool>) -> Result<Value, Box<dyn std::error::Error>> {
        let mut config = self.config.write().await;
        
        if let Some(objs) = objectives {
            config.objectives = objs;
            // Normalize weights
            let sum: f32 = config.objectives.values().sum();
            if sum > 0.0 {
                for value in config.objectives.values_mut() {
                    *value /= sum;
                }
            }
        }
        
        if let Some(enable) = enabled {
            config.enabled = enable;
        }
        
        config.validate()?;
        
        Ok(serde_json::json!({
            "updated": true,
            "config": {
                "enabled": config.enabled,
                "objectives": config.objectives,
            }
        }))
    }
    
    pub async fn provide_feedback(&self, feedback: OperationFeedback) -> Result<Value, Box<dyn std::error::Error>> {
        let operation_id = feedback.operation_id.clone();
        
        // Find the operation in pending operations
        let mut pending = self.pending_operations.write().await;
        if let Some(mut metric) = pending.remove(&operation_id) {
            // Store values before updating metric
            let response_time = metric.response_time_ms;
            let memory_delta = metric.memory_delta_bytes;
            
            // Update the metric with feedback score
            metric.feedback_score = feedback.score;
            
            // Re-record the metric with feedback
            self.usage_collector.record_metric(metric).await;
            
            // Calculate new fitness based on feedback
            let fitness = calculate_fitness_with_feedback(
                response_time,
                memory_delta,
                feedback.score,
            );
            
            // Store feedback for future learning
            // TODO: Store feedback in a persistent feedback store
            
            Ok(serde_json::json!({
                "status": "feedback_received",
                "operation_id": operation_id,
                "feedback_type": feedback.feedback_type,
                "score": feedback.score,
                "fitness": fitness,
                "context": feedback.context,
                "usage_context": feedback.usage_context,
            }))
        } else {
            Err(format!("Operation ID {} not found in pending operations", operation_id).into())
        }
    }
    
    // Internal helper methods
    
    fn generate_operation_id(&self) -> String {
        format!("op_{}", Uuid::new_v4().to_string().replace('-', "")[..8].to_string())
    }
    
    async fn record_operation(
        &self,
        operation_id: String,
        operation: OperationType,
        input_size: usize,
        response_time_ms: f32,
        memory_delta: i64,
        similarity_scores: Vec<f32>,
        cache_hit: bool,
    ) {
        // Record metric
        let metric = UsageMetrics {
            operation_id: operation_id.clone(),
            operation,
            input_size,
            response_time_ms,
            memory_delta_bytes: memory_delta,
            similarity_scores,
            cache_hit,
            timestamp: chrono::Utc::now(),
            feedback_score: None,
        };
        
        // Store in pending operations for feedback
        self.pending_operations.write().await.insert(operation_id, metric.clone());
        
        self.usage_collector.record_metric(metric).await;
        
        // Increment operation count
        {
            let mut count = self.operation_count.write().await;
            *count += 1;
        }
        
        // Check for evolution triggers
        self.check_evolution_triggers().await;
    }
    
    async fn check_evolution_triggers(&self) {
        let config = self.config.read().await;
        let count = *self.operation_count.read().await;
        let evolver = self.evolver.lock().await;
        let last_evolution = evolver.get_last_evolution_time().await;
        
        if config.should_trigger_evolution(count, last_evolution) {
            // Check if evolution is already running
            let status = evolver.get_status().await;
            if !status.is_running {
                // Reset operation count
                *self.operation_count.write().await = 0;
                
                // Trigger evolution in background
                tokio::spawn({
                    let adaptive = self.clone();
                    async move {
                        if let Err(e) = adaptive.trigger_evolution(None, false).await {
                            eprintln!("Auto-evolution failed: {}", e);
                        }
                    }
                });
            }
        }
        
        // Check for new evolved architectures
        if let Some(evolved) = evolver.try_receive_architecture().await {
            tokio::spawn({
                let adaptive = self.clone();
                async move {
                    if let Err(e) = adaptive.swap_architecture(evolved).await {
                        eprintln!("Architecture swap failed: {}", e);
                    }
                }
            });
        }
    }
    
    async fn swap_architecture(&self, new_arch: EvolvedArchitecture) -> Result<(), Box<dyn std::error::Error>> {
        println!("Swapping to evolved architecture: generation {}, fitness {}", 
                 new_arch.generation, new_arch.fitness_score);
        
        // Create new memory module with evolved architecture
        let persistent_config = PersistentConfig {
            memory_config: new_arch.memory_config,
            storage_path: std::path::PathBuf::from("./adaptive_memory_data"),
            auto_save_interval: 60,
            save_on_write: false,
        };
        let new_module = PersistentMemoryModule::new(persistent_config).await?;
        
        // Transfer existing memories (simplified - in production would be more sophisticated)
        // For now, we'll just swap the module
        let mut active = self.active_memory.write().await;
        *active = new_module;
        
        println!("Architecture swap complete: {}", new_arch.architecture_summary);
        
        Ok(())
    }
    
    async fn get_memory_usage(&self) -> usize {
        // Simplified memory usage calculation
        // In production, would use actual memory profiling
        let memory = self.active_memory.read().await;
        let (size, _, _, _) = memory.get_stats().await;
        size * 1024 // Rough estimate
    }
}

impl Clone for AdaptiveMemoryModule {
    fn clone(&self) -> Self {
        Self {
            active_memory: Arc::clone(&self.active_memory),
            usage_collector: Arc::clone(&self.usage_collector),
            evolver: Arc::clone(&self.evolver),
            config: Arc::clone(&self.config),
            operation_count: Arc::clone(&self.operation_count),
            pending_operations: Arc::clone(&self.pending_operations),
        }
    }
}