//! Dream Consolidation - Background processor for insight generation and memory reorganization
//! 
//! This module implements a dream-like consolidation process that runs during idle cycles
//! to extract insights from memory patterns, reorganize temporal memories, and generate
//! new pattern nodes from recent activity.

use crate::graph::{
    ConsciousGraph, ConsciousNode, NodeType, NodeId, EdgeType, ConsciousEdge,
    PatternExtractor, ExtractedPattern, GraphStorage, GraphOperations,
    core::PatternNode,
};
use anyhow::{Result, anyhow};
use parking_lot::RwLock;
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tokio::task::JoinHandle;
use chrono::{Utc, Duration as ChronoDuration};
use std::collections::HashMap;
use uuid::Uuid;
use serde::{Serialize, Deserialize};

/// Configuration for dream consolidation
#[derive(Debug, Clone)]
pub struct DreamConfig {
    /// How often to run consolidation (seconds)
    pub consolidation_interval: u64,
    /// Minimum confidence for insights
    pub insight_confidence_threshold: f32,
    /// Hours of history to analyze
    pub analysis_window_hours: i64,
    /// Maximum insights per cycle
    pub max_insights_per_cycle: usize,
    /// Enable temporal reorganization
    pub enable_temporal_reorg: bool,
    /// Activity threshold (below this is considered idle)
    pub idle_activity_threshold: f32,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            consolidation_interval: 300, // 5 minutes
            insight_confidence_threshold: 0.7,
            analysis_window_hours: 24,
            max_insights_per_cycle: 10,
            enable_temporal_reorg: true,
            idle_activity_threshold: 0.3,
        }
    }
}

/// Insight generated during consolidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamInsight {
    pub id: String,
    pub insight_type: InsightType,
    pub description: String,
    pub confidence: f32,
    pub source_patterns: Vec<String>,
    pub related_memories: Vec<NodeId>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: chrono::DateTime<Utc>,
}

/// Types of insights that can be generated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    PatternRecognition,
    TemporalConnection,
    ConceptualSynthesis,
    EmergentProperty,
    MemoryConsolidation,
    CognitiveReorganization,
}

/// Statistics for consolidation cycles
#[derive(Debug, Default, Clone)]
pub struct ConsolidationStats {
    pub cycles_completed: u64,
    pub insights_generated: u64,
    pub memories_reorganized: u64,
    pub patterns_extracted: u64,
    pub last_cycle_duration_ms: u64,
    pub last_cycle_time: Option<chrono::DateTime<Utc>>,
}

/// Dream consolidation processor
pub struct DreamConsolidation {
    /// Reference to the conscious graph
    graph: Arc<ConsciousGraph>,
    /// Pattern extractor
    pattern_extractor: Arc<PatternExtractor>,
    /// Configuration
    config: DreamConfig,
    /// Runtime statistics
    stats: Arc<RwLock<ConsolidationStats>>,
    /// Background task handle
    task_handle: Option<JoinHandle<()>>,
    /// Flag to stop processing
    shutdown_flag: Arc<RwLock<bool>>,
}

impl DreamConsolidation {
    /// Create a new DreamConsolidation processor
    pub fn new(graph: Arc<ConsciousGraph>, config: DreamConfig) -> Self {
        Self {
            graph,
            pattern_extractor: Arc::new(PatternExtractor::new()),
            config,
            stats: Arc::new(RwLock::new(ConsolidationStats::default())),
            task_handle: None,
            shutdown_flag: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Start the background consolidation process
    pub fn start(&mut self) -> Result<()> {
        if self.task_handle.is_some() {
            return Err(anyhow!("Dream consolidation already running"));
        }
        
        let graph = Arc::clone(&self.graph);
        let pattern_extractor = Arc::clone(&self.pattern_extractor);
        let config = self.config.clone();
        let stats = Arc::clone(&self.stats);
        let shutdown_flag = Arc::clone(&self.shutdown_flag);
        
        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.consolidation_interval));
            
            loop {
                interval.tick().await;
                
                // Check shutdown flag
                if *shutdown_flag.read() {
                    break;
                }
                
                // Check if system is idle
                if !Self::is_system_idle(&graph, config.idle_activity_threshold) {
                    continue;
                }
                
                // Run consolidation cycle
                let start_time = std::time::Instant::now();
                
                match Self::run_consolidation_cycle(
                    &graph,
                    &pattern_extractor,
                    &config,
                ).await {
                    Ok(insights) => {
                        let mut stats_guard = stats.write();
                        stats_guard.cycles_completed += 1;
                        stats_guard.insights_generated += insights.len() as u64;
                        stats_guard.last_cycle_duration_ms = start_time.elapsed().as_millis() as u64;
                        stats_guard.last_cycle_time = Some(Utc::now());
                    }
                    Err(e) => {
                        eprintln!("Dream consolidation error: {}", e);
                    }
                }
            }
        });
        
        self.task_handle = Some(handle);
        Ok(())
    }
    
    /// Stop the background consolidation process
    pub fn stop(&mut self) -> Result<()> {
        *self.shutdown_flag.write() = true;
        
        if let Some(handle) = self.task_handle.take() {
            // Note: In a real implementation, we'd use handle.abort() or similar
            // For now, the task will stop on next interval tick
        }
        
        Ok(())
    }
    
    /// Get consolidation statistics
    pub fn get_stats(&self) -> ConsolidationStats {
        self.stats.read().clone()
    }
    
    /// Check if the system is idle enough for consolidation
    fn is_system_idle(graph: &ConsciousGraph, threshold: f32) -> bool {
        // Check recent activity level
        let stats = graph.get_stats();
        
        // Simple heuristic: check if recent query rate is low
        // In a real implementation, this would check various activity metrics
        true // For now, always consider idle
    }
    
    /// Run a single consolidation cycle
    async fn run_consolidation_cycle(
        graph: &ConsciousGraph,
        pattern_extractor: &PatternExtractor,
        config: &DreamConfig,
    ) -> Result<Vec<DreamInsight>> {
        let mut insights = Vec::new();
        
        // Phase 1: Extract patterns from recent memories
        let patterns = pattern_extractor.extract_temporal_patterns_by_hours(
            config.analysis_window_hours
        )?;
        
        // Phase 2: Generate insights from patterns
        for pattern in patterns.iter().take(config.max_insights_per_cycle) {
            if pattern.confidence >= config.insight_confidence_threshold {
                if let Some(insight) = Self::generate_insight_from_pattern(pattern, graph).await? {
                    insights.push(insight);
                }
            }
        }
        
        // Phase 3: Temporal memory reorganization
        if config.enable_temporal_reorg {
            Self::reorganize_temporal_memories(graph, &patterns).await?;
        }
        
        // Phase 4: Store insights as pattern nodes
        for insight in &insights {
            Self::store_insight_as_node(insight, graph)?;
        }
        
        Ok(insights)
    }
    
    /// Generate an insight from a pattern
    async fn generate_insight_from_pattern(
        pattern: &ExtractedPattern,
        graph: &ConsciousGraph,
    ) -> Result<Option<DreamInsight>> {
        // Analyze pattern properties
        let insight_type = match pattern.pattern_type.as_str() {
            "temporal" => InsightType::TemporalConnection,
            "semantic" => InsightType::ConceptualSynthesis,
            "structural" => InsightType::PatternRecognition,
            "behavioral" => InsightType::EmergentProperty,
            _ => InsightType::MemoryConsolidation,
        };
        
        // Generate insight description
        let description = format!(
            "Discovered {} pattern: {} (frequency: {}, confidence: {:.2})",
            pattern.pattern_type,
            pattern.description,
            pattern.frequency,
            pattern.confidence
        );
        
        // Create insight
        let insight = DreamInsight {
            id: Uuid::new_v4().to_string(),
            insight_type,
            description,
            confidence: pattern.confidence,
            source_patterns: vec![pattern.pattern_type.clone()],
            related_memories: pattern.examples.clone(),
            metadata: pattern.properties.clone(),
            created_at: Utc::now(),
        };
        
        Ok(Some(insight))
    }
    
    /// Reorganize temporal memories based on patterns
    async fn reorganize_temporal_memories(
        graph: &ConsciousGraph,
        patterns: &[ExtractedPattern],
    ) -> Result<()> {
        // Find temporal patterns
        let temporal_patterns: Vec<_> = patterns.iter()
            .filter(|p| p.pattern_type == "temporal")
            .collect();
        
        // For each temporal pattern, strengthen connections
        for pattern in temporal_patterns {
            for i in 0..pattern.examples.len().saturating_sub(1) {
                let source = &pattern.examples[i];
                let target = &pattern.examples[i + 1];
                
                // Create or strengthen temporal edge
                let edge = ConsciousEdge::new(
                    source.clone(),
                    target.clone(),
                    EdgeType::Temporal { delta_ms: 0 }, // Would calculate actual time delta
                );
                
                graph.add_edge(edge)?;
            }
        }
        
        Ok(())
    }
    
    /// Store an insight as a pattern node in the graph
    fn store_insight_as_node(insight: &DreamInsight, graph: &ConsciousGraph) -> Result<()> {
        let pattern_node = NodeType::Pattern(PatternNode {
            id: insight.id.clone(),
            pattern_type: format!("{:?}", insight.insight_type),
            description: insight.description.clone(),
            frequency: 1, // Initial frequency
            confidence: insight.confidence,
            examples: insight.related_memories.clone(),
        });
        
        // Add the pattern node to the graph
        let node_id = graph.add_node(pattern_node)?;
        
        // Connect to related memories
        for memory_id in &insight.related_memories {
            let edge = ConsciousEdge::new(
                node_id.clone(),
                memory_id.clone(),
                EdgeType::Derived,
            );
            graph.add_edge(edge)?;
        }
        
        Ok(())
    }
    
    /// Manually trigger a consolidation cycle
    pub async fn trigger_consolidation(&self) -> Result<Vec<DreamInsight>> {
        Self::run_consolidation_cycle(
            &self.graph,
            &self.pattern_extractor,
            &self.config,
        ).await
    }
}

/// Integration helper for ConsciousGraph
/// Note: In production, ConsciousGraph would need to be wrapped in Arc from the beginning
pub fn create_dream_consolidation(
    graph: Arc<ConsciousGraph>, 
    config: DreamConfig
) -> Result<Arc<RwLock<DreamConsolidation>>> {
    let mut processor = DreamConsolidation::new(graph, config);
    processor.start()?;
    Ok(Arc::new(RwLock::new(processor)))
}

