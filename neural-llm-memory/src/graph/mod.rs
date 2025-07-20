//! Conscious Knowledge Graph module for NeuralClaude
//! 
//! This module implements a graph-based memory system with consciousness attributes,
//! emotional states, and advanced cognitive capabilities.

pub mod core;
pub mod storage;
pub mod index;
pub mod inference;
pub mod patterns;
pub mod algorithms;
pub mod compatibility;
pub mod conscious_graph;
pub mod dream_consolidation;
pub mod cross_modal;
pub mod temporal_tracker;

// Re-export key types
pub use self::core::{ConsciousNode, ConsciousEdge, NodeType, EdgeType, NodeId, EdgeId};
pub use self::storage::{GraphStorage, GraphSnapshot};
pub use self::index::{GraphIndices, HnswIndex};
pub use self::inference::RelationshipInference;
pub use self::patterns::{PatternExtractor, ExtractedPattern};
pub use self::algorithms::{GraphAlgorithms, TraversalOptions};
pub use self::compatibility::{HybridMemoryBank, MigrationState};
pub use self::conscious_graph::{ConsciousGraph, ConsciousGraphConfig, GraphStats};
pub use self::dream_consolidation::{DreamConsolidation, DreamConfig, DreamInsight, InsightType, ConsolidationStats, ConsolidationResult};
pub use self::cross_modal::{
    CrossModalBridge, CrossModalConfig, MemoryModality, CrossModalConnection,
    CrossModalStats, BridgeStrength, FeatureExtractor
};
pub use self::temporal_tracker::{TemporalTracker, AccessSequence, CooccurrencePattern, TemporalStats};

use anyhow::Result;

/// Trait for graph operations
pub trait GraphOperations {
    /// Add a node to the graph
    fn add_node(&self, node: NodeType) -> Result<NodeId>;
    
    /// Add an edge between nodes
    fn add_edge(&self, edge: ConsciousEdge) -> Result<EdgeId>;
    
    /// Query the graph starting from a node
    fn query_graph(&self, start: &NodeId, depth: usize) -> Result<GraphQueryResult>;
    
    /// Find patterns in the graph
    fn extract_patterns(&self, context: &str) -> Result<Vec<ExtractedPattern>>;
}

/// Result of a graph query
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphQueryResult {
    pub nodes: Vec<ConsciousNode>,
    pub edges: Vec<ConsciousEdge>,
    pub paths: Vec<GraphPath>,
    pub stats: QueryStats,
}

/// A path through the graph
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphPath {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeType>,
    pub total_weight: f32,
}

/// Query statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryStats {
    pub total_traversed: usize,
    pub execution_time_ms: u64,
    pub consciousness_activations: usize,
}