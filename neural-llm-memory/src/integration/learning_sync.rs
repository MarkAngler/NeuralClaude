//! Synchronization between neural pattern learning and memory systems

use crate::integration::{
    LearnedPattern, PatternContext, PerformanceMetrics,
    PatternMemory, IntegrationConfig,
};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use chrono::Utc;

/// Events for cross-system synchronization
#[derive(Debug, Clone)]
pub enum SyncEvent {
    PatternLearned {
        pattern: LearnedPattern,
        source: String,
    },
    PatternUpdated {
        pattern_id: String,
        metrics: PerformanceMetrics,
    },
    PatternRequested {
        context: PatternContext,
        response_channel: mpsc::Sender<Vec<LearnedPattern>>,
    },
    MemoryConsolidation {
        pattern_ids: Vec<String>,
    },
}

/// Synchronization manager for neural patterns and memory
pub struct LearningSyncManager<M: PatternMemory> {
    pattern_memory: Arc<RwLock<M>>,
    config: IntegrationConfig,
    event_sender: mpsc::Sender<SyncEvent>,
    event_receiver: mpsc::Receiver<SyncEvent>,
}

impl<M: PatternMemory + Send + Sync + 'static> LearningSyncManager<M> {
    pub fn new(pattern_memory: M, config: IntegrationConfig) -> Self {
        let (event_sender, event_receiver) = mpsc::channel(100);
        
        Self {
            pattern_memory: Arc::new(RwLock::new(pattern_memory)),
            config,
            event_sender,
            event_receiver,
        }
    }
    
    /// Get event sender for other systems
    pub fn get_sender(&self) -> mpsc::Sender<SyncEvent> {
        self.event_sender.clone()
    }
    
    /// Start the synchronization loop
    pub async fn start(mut self) {
        loop {
            match self.event_receiver.recv().await {
                Some(event) => {
                    self.handle_event(event).await;
                }
                None => {
                    // Channel closed, exit
                    break;
                }
            }
        }
    }
    
    async fn handle_event(&self, event: SyncEvent) {
        match event {
            SyncEvent::PatternLearned { pattern, source } => {
                self.handle_pattern_learned(pattern, source).await;
            }
            SyncEvent::PatternUpdated { pattern_id, metrics } => {
                self.handle_pattern_updated(pattern_id, metrics).await;
            }
            SyncEvent::PatternRequested { context, response_channel } => {
                self.handle_pattern_request(context, response_channel).await;
            }
            SyncEvent::MemoryConsolidation { pattern_ids } => {
                self.handle_consolidation(pattern_ids).await;
            }
        }
    }
    
    async fn handle_pattern_learned(&self, pattern: LearnedPattern, source: String) {
        // Check if pattern meets storage threshold
        if pattern.confidence >= self.config.auto_store_threshold {
            let mut memory = self.pattern_memory.write().await;
            
            match memory.store_pattern(pattern.clone()) {
                Ok(pattern_id) => {
                    println!("✅ Stored pattern {} from {} with confidence {:.2}", 
                        pattern_id, source, pattern.confidence);
                }
                Err(e) => {
                    eprintln!("❌ Failed to store pattern: {}", e);
                }
            }
        }
    }
    
    async fn handle_pattern_updated(&self, pattern_id: String, metrics: PerformanceMetrics) {
        let mut memory = self.pattern_memory.write().await;
        memory.update_pattern_metrics(&pattern_id, &metrics);
    }
    
    async fn handle_pattern_request(
        &self,
        context: PatternContext,
        response_channel: mpsc::Sender<Vec<LearnedPattern>>
    ) {
        let memory = self.pattern_memory.read().await;
        let matches = memory.search_patterns(&context, 5);
        
        let patterns: Vec<LearnedPattern> = matches.into_iter()
            .map(|m| m.pattern)
            .collect();
        
        let _ = response_channel.send(patterns).await;
    }
    
    async fn handle_consolidation(&self, pattern_ids: Vec<String>) {
        let mut memory = self.pattern_memory.write().await;
        
        // Retrieve all patterns
        let patterns: Vec<LearnedPattern> = pattern_ids.iter()
            .filter_map(|id| memory.retrieve_pattern(id))
            .collect();
        
        if patterns.len() < 2 {
            return; // Need at least 2 patterns to consolidate
        }
        
        // Create consolidated pattern
        let consolidated = self.consolidate_patterns(patterns);
        
        // Store consolidated pattern
        match memory.store_pattern(consolidated) {
            Ok(id) => {
                println!("✅ Created consolidated pattern: {}", id);
            }
            Err(e) => {
                eprintln!("❌ Failed to store consolidated pattern: {}", e);
            }
        }
    }
    
    fn consolidate_patterns(&self, patterns: Vec<LearnedPattern>) -> LearnedPattern {
        // Average performance metrics
        let mut avg_metrics = PerformanceMetrics {
            success_rate: 0.0,
            average_time_ms: 0.0,
            token_efficiency: 0.0,
            error_rate: 0.0,
            adaptability_score: 0.0,
            usage_count: 0,
        };
        
        for pattern in &patterns {
            avg_metrics.success_rate += pattern.performance.success_rate;
            avg_metrics.average_time_ms += pattern.performance.average_time_ms;
            avg_metrics.token_efficiency += pattern.performance.token_efficiency;
            avg_metrics.error_rate += pattern.performance.error_rate;
            avg_metrics.adaptability_score += pattern.performance.adaptability_score;
            avg_metrics.usage_count += pattern.performance.usage_count;
        }
        
        let count = patterns.len() as f32;
        avg_metrics.success_rate /= count;
        avg_metrics.average_time_ms /= count as f64;
        avg_metrics.token_efficiency /= count;
        avg_metrics.error_rate /= count;
        avg_metrics.adaptability_score /= count;
        
        // Use most common pattern type
        let pattern_type = patterns[0].pattern_type.clone(); // Simplified
        
        // Merge contexts
        let mut all_file_types = Vec::new();
        let mut all_domains = Vec::new();
        
        for pattern in &patterns {
            all_file_types.extend(pattern.context.file_types.clone());
            all_domains.push(pattern.context.domain.clone());
        }
        
        all_file_types.sort();
        all_file_types.dedup();
        
        LearnedPattern {
            id: format!("consolidated_{}", uuid::Uuid::new_v4()),
            pattern_type,
            context: PatternContext {
                task_type: "consolidated".to_string(),
                file_types: all_file_types,
                complexity: patterns[0].context.complexity.clone(),
                domain: all_domains.join("+"),
                success_criteria: vec![],
                environmental_factors: HashMap::new(),
            },
            performance: avg_metrics,
            neural_weights: vec![], // Would average weights in real implementation
            timestamp: Utc::now(),
            confidence: patterns.iter().map(|p| p.confidence).sum::<f32>() / count,
        }
    }
}

use std::collections::HashMap;