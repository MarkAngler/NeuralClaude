//! Pattern extraction and analysis for the knowledge graph

use crate::graph::core::{NodeId, ConsciousNode, NodeType, PatternType};
use crate::graph::storage::GraphStorage;
use crate::graph::ConsciousGraph;
use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use chrono::{Utc, Duration};

/// Pattern extractor for discovering patterns in the graph
pub struct PatternExtractor {
    pub min_frequency: usize,
    pub confidence_threshold: f32,
    pub pattern_types: Vec<PatternTypeConfig>,
    graph: Option<Arc<ConsciousGraph>>,
}

/// Configuration for pattern types
#[derive(Debug, Clone)]
pub enum PatternTypeConfig {
    Structural,   // Graph structure patterns
    Temporal,     // Time-based patterns
    Causal,       // Cause-effect patterns
    Semantic,     // Content similarity patterns
    Behavioral,   // Access/usage patterns
    Cognitive,    // Thinking style patterns
}

/// An extracted pattern from the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedPattern {
    pub pattern_type: String,
    pub description: String,
    pub frequency: usize,
    pub confidence: f32,
    pub examples: Vec<NodeId>,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Result of subgraph analysis
#[derive(Debug, Clone)]
pub struct SubgraphPattern {
    pub nodes: Vec<NodeId>,
    pub structure: String,
    pub frequency: usize,
    pub instances: Vec<SubgraphInstance>,
}

/// An instance of a subgraph pattern
#[derive(Debug, Clone)]
pub struct SubgraphInstance {
    pub nodes: Vec<NodeId>,
}

impl PatternExtractor {
    pub fn new() -> Self {
        Self {
            min_frequency: 2,
            confidence_threshold: 0.6,
            pattern_types: vec![
                PatternTypeConfig::Temporal,
                PatternTypeConfig::Semantic,
                PatternTypeConfig::Structural,
            ],
            graph: None,
        }
    }
    
    pub fn with_graph(graph: Arc<ConsciousGraph>) -> Self {
        Self {
            min_frequency: 2,
            confidence_threshold: 0.6,
            pattern_types: vec![
                PatternTypeConfig::Temporal,
                PatternTypeConfig::Semantic,
                PatternTypeConfig::Structural,
            ],
            graph: Some(graph),
        }
    }
    
    pub fn extract_temporal_patterns_by_hours(&self, hours: i64) -> Result<Vec<ExtractedPattern>> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| anyhow!("Graph not initialized"))?;
        
        let since = Utc::now() - Duration::hours(hours);
        let recent_nodes = graph.storage.get_nodes_since(since)?;
        
        // Group by temporal proximity
        let temporal_groups = self.group_by_temporal_proximity(recent_nodes);
        
        // Create patterns with actual node IDs
        let mut patterns = Vec::new();
        for (group_key, nodes) in temporal_groups {
            if nodes.len() >= self.min_frequency {
                patterns.push(ExtractedPattern {
                    pattern_type: "temporal".to_string(),
                    description: format!("Temporal cluster: {}", group_key),
                    frequency: nodes.len(),
                    confidence: self.calculate_temporal_confidence(&nodes),
                    examples: nodes.iter().map(|(id, _)| id.clone()).collect(),
                    properties: self.extract_temporal_properties(&nodes),
                });
            }
        }
        
        Ok(patterns)
    }
    
    /// Group nodes by temporal proximity
    fn group_by_temporal_proximity(&self, nodes: Vec<(NodeId, ConsciousNode)>) -> HashMap<String, Vec<(NodeId, ConsciousNode)>> {
        let mut groups: HashMap<String, Vec<(NodeId, ConsciousNode)>> = HashMap::new();
        
        // Group nodes that were created within 5 minute windows
        for (node_id, node) in nodes {
            let time_bucket = node.created_at.timestamp() / 300; // 5-minute buckets
            let group_key = format!("bucket_{}", time_bucket);
            groups.entry(group_key)
                .or_insert_with(Vec::new)
                .push((node_id, node));
        }
        
        groups
    }
    
    /// Calculate temporal confidence based on node characteristics
    fn calculate_temporal_confidence(&self, nodes: &[(NodeId, ConsciousNode)]) -> f32 {
        if nodes.is_empty() {
            return 0.0;
        }
        
        // Calculate confidence based on:
        // 1. Number of nodes
        // 2. Time consistency (how close in time they are)
        // 3. Type consistency
        
        let mut time_deltas = Vec::new();
        for window in nodes.windows(2) {
            let delta = window[1].1.created_at.timestamp() - window[0].1.created_at.timestamp();
            time_deltas.push(delta.abs());
        }
        
        let avg_delta = if !time_deltas.is_empty() {
            time_deltas.iter().sum::<i64>() as f32 / time_deltas.len() as f32
        } else {
            0.0
        };
        
        // Lower time delta = higher confidence
        let time_confidence = 1.0 / (1.0 + avg_delta / 300.0); // Normalize by 5 minutes
        
        // Node count confidence
        let count_confidence = (nodes.len() as f32 / 10.0).min(1.0);
        
        // Combined confidence
        (time_confidence * 0.7 + count_confidence * 0.3).min(1.0)
    }
    
    /// Extract properties from temporal node groups
    fn extract_temporal_properties(&self, nodes: &[(NodeId, ConsciousNode)]) -> HashMap<String, serde_json::Value> {
        let mut properties = HashMap::new();
        
        if nodes.is_empty() {
            return properties;
        }
        
        // Time span
        let first_time = nodes.first().unwrap().1.created_at;
        let last_time = nodes.last().unwrap().1.created_at;
        let span_seconds = (last_time - first_time).num_seconds();
        
        properties.insert("time_span_seconds".to_string(), 
                         serde_json::Value::Number(serde_json::Number::from(span_seconds)));
        
        // Node types distribution
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for (_, node) in nodes {
            let type_name = format!("{:?}", node.node_type);
            *type_counts.entry(type_name).or_insert(0) += 1;
        }
        
        properties.insert("node_types".to_string(), 
                         serde_json::to_value(type_counts).unwrap_or(serde_json::Value::Null));
        
        // Average consciousness level
        let avg_consciousness: f32 = nodes.iter()
            .map(|(_, node)| node.awareness.level)
            .sum::<f32>() / nodes.len() as f32;
        
        properties.insert("avg_consciousness".to_string(),
                         serde_json::Value::Number(serde_json::Number::from_f64(avg_consciousness as f64).unwrap()));
        
        properties
    }
    
    pub fn extract_contextual_patterns(&self, context: &str) -> Result<Vec<ExtractedPattern>> {
        // Return empty patterns for now - in production this would analyze contextual patterns
        let patterns = Vec::new();
        
        // Note: This is a simplified implementation. In production, this would:
        // 1. Search for nodes matching the context
        // 2. Analyze relationships between context-relevant nodes
        // 3. Identify common patterns in how context appears
        // 4. Calculate confidence based on pattern strength
        
        Ok(patterns)
    }
    
    /// Extract semantic patterns from recent nodes
    pub fn extract_semantic_patterns_from_recent(&self, hours: i64) -> Result<Vec<ExtractedPattern>> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| anyhow!("Graph not initialized"))?;
        
        let since = Utc::now() - Duration::hours(hours);
        let recent_nodes = graph.storage.get_nodes_since(since)?;
        
        // Cluster by semantic similarity using embeddings
        let clusters = self.cluster_by_similarity(recent_nodes)?;
        
        // Create patterns from clusters
        let mut patterns = Vec::new();
        for (cluster_id, node_ids) in clusters {
            if node_ids.len() >= self.min_frequency {
                patterns.push(ExtractedPattern {
                    pattern_type: "semantic".to_string(),
                    description: format!("Semantic cluster {}", cluster_id),
                    frequency: node_ids.len(),
                    confidence: 0.85,
                    examples: node_ids,
                    properties: HashMap::new(),
                });
            }
        }
        
        Ok(patterns)
    }
    
    /// Cluster nodes by embedding similarity
    fn cluster_by_similarity(&self, nodes: Vec<(NodeId, ConsciousNode)>) -> Result<HashMap<usize, Vec<NodeId>>> {
        if nodes.is_empty() {
            return Ok(HashMap::new());
        }
        
        let mut clusters: HashMap<usize, Vec<NodeId>> = HashMap::new();
        let similarity_threshold = 0.8;
        let mut cluster_id = 0;
        
        // Simple clustering: for each node, find similar nodes
        for i in 0..nodes.len() {
            let (ref node_id_i, ref node_i) = nodes[i];
            
            // Skip if already assigned to a cluster
            let already_clustered = clusters.values()
                .any(|cluster| cluster.contains(node_id_i));
            if already_clustered {
                continue;
            }
            
            // Start new cluster
            let mut current_cluster = vec![node_id_i.clone()];
            
            // Find similar nodes
            for j in i+1..nodes.len() {
                let (ref node_id_j, ref node_j) = nodes[j];
                
                // Skip if already in a cluster
                let already_clustered_j = clusters.values()
                    .any(|cluster| cluster.contains(node_id_j));
                if already_clustered_j {
                    continue;
                }
                
                // Calculate cosine similarity
                let similarity = self.cosine_similarity(&node_i.embeddings, &node_j.embeddings);
                if similarity > similarity_threshold {
                    current_cluster.push(node_id_j.clone());
                }
            }
            
            // Add cluster if it has multiple nodes
            if current_cluster.len() >= self.min_frequency {
                clusters.insert(cluster_id, current_cluster);
                cluster_id += 1;
            }
        }
        
        Ok(clusters)
    }
    
    /// Calculate cosine similarity between two embedding vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter()
            .zip(b.iter())
            .map(|(x, y)| x * y)
            .sum();
        
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        (dot_product / (norm_a * norm_b)).min(1.0).max(-1.0)
    }
}

impl Default for PatternExtractor {
    fn default() -> Self {
        Self {
            min_frequency: 3,
            confidence_threshold: 0.7,
            pattern_types: vec![
                PatternTypeConfig::Structural,
                PatternTypeConfig::Temporal,
                PatternTypeConfig::Semantic,
            ],
            graph: None,
        }
    }
}

impl PatternExtractor {
    /// Extract patterns from the graph based on context
    pub async fn extract_patterns(
        &self,
        context: &str,
        graph: &GraphStorage,
    ) -> Result<Vec<ExtractedPattern>> {
        let candidate_nodes = self.find_context_nodes(context, graph)?;
        let mut patterns = Vec::new();
        
        for pattern_type in &self.pattern_types {
            let extracted = match pattern_type {
                PatternTypeConfig::Structural => {
                    self.extract_structural_patterns(&candidate_nodes, graph)?
                }
                PatternTypeConfig::Temporal => {
                    self.extract_temporal_patterns(&candidate_nodes, graph)?
                }
                PatternTypeConfig::Causal => {
                    self.extract_causal_patterns(&candidate_nodes, graph)?
                }
                PatternTypeConfig::Semantic => {
                    self.extract_semantic_patterns(&candidate_nodes, graph)?
                }
                PatternTypeConfig::Behavioral => {
                    self.extract_behavioral_patterns(&candidate_nodes, graph)?
                }
                PatternTypeConfig::Cognitive => {
                    self.extract_cognitive_patterns(&candidate_nodes, graph)?
                }
            };
            
            patterns.extend(extracted);
        }
        
        // Filter by frequency and confidence
        patterns.retain(|p| {
            p.frequency >= self.min_frequency && 
            p.confidence >= self.confidence_threshold
        });
        
        // Sort by confidence
        patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        Ok(patterns)
    }
    
    /// Find nodes relevant to the context
    fn find_context_nodes(&self, context: &str, graph: &GraphStorage) -> Result<Vec<NodeId>> {
        // In a real implementation, this would use the search functionality
        // For now, return a placeholder
        Ok(vec![])
    }
    
    /// Extract structural patterns (common subgraphs)
    fn extract_structural_patterns(
        &self,
        nodes: &[NodeId],
        graph: &GraphStorage,
    ) -> Result<Vec<ExtractedPattern>> {
        let mut patterns = Vec::new();
        
        // Look for common graph structures
        let subgraphs = self.find_frequent_subgraphs(nodes, graph)?;
        
        for subgraph in subgraphs {
            if subgraph.frequency >= self.min_frequency {
                let pattern = ExtractedPattern {
                    pattern_type: "structural".to_string(),
                    description: format!("Frequent {} structure", subgraph.structure),
                    frequency: subgraph.frequency,
                    confidence: self.calculate_structural_confidence(&subgraph),
                    examples: subgraph.instances.iter()
                        .flat_map(|i| i.nodes.clone())
                        .take(5)
                        .collect(),
                    properties: HashMap::new(),
                };
                
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    /// Extract temporal patterns
    fn extract_temporal_patterns(
        &self,
        nodes: &[NodeId],
        graph: &GraphStorage,
    ) -> Result<Vec<ExtractedPattern>> {
        let mut patterns = Vec::new();
        
        // Analyze temporal sequences
        let sequences = self.find_temporal_sequences(nodes, graph)?;
        
        for (sequence_type, instances) in sequences {
            if instances.len() >= self.min_frequency {
                let pattern = ExtractedPattern {
                    pattern_type: "temporal".to_string(),
                    description: format!("Temporal sequence: {}", sequence_type),
                    frequency: instances.len(),
                    confidence: self.calculate_temporal_confidence_legacy(&instances),
                    examples: instances.into_iter().take(5).collect(),
                    properties: HashMap::new(),
                };
                
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    /// Extract causal patterns
    fn extract_causal_patterns(
        &self,
        nodes: &[NodeId],
        graph: &GraphStorage,
    ) -> Result<Vec<ExtractedPattern>> {
        let mut patterns = Vec::new();
        
        // Look for cause-effect relationships
        for node_id in nodes {
            let edges = graph.get_edges_for_node(node_id)?;
            
            let causal_edges: Vec<_> = edges.iter()
                .filter(|e| matches!(e.edge_type, crate::graph::core::EdgeType::CausedBy))
                .collect();
            
            if causal_edges.len() >= self.min_frequency {
                let pattern = ExtractedPattern {
                    pattern_type: "causal".to_string(),
                    description: "Causal chain pattern".to_string(),
                    frequency: causal_edges.len(),
                    confidence: 0.8,
                    examples: causal_edges.iter()
                        .map(|e| e.target.clone())
                        .take(5)
                        .collect(),
                    properties: HashMap::new(),
                };
                
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    /// Extract semantic patterns
    fn extract_semantic_patterns(
        &self,
        nodes: &[NodeId],
        graph: &GraphStorage,
    ) -> Result<Vec<ExtractedPattern>> {
        let mut patterns = Vec::new();
        
        // Group nodes by semantic similarity
        let semantic_groups = self.group_by_semantics(nodes, graph)?;
        
        for (group_name, group_nodes) in semantic_groups {
            if group_nodes.len() >= self.min_frequency {
                let pattern = ExtractedPattern {
                    pattern_type: "semantic".to_string(),
                    description: format!("Semantic cluster: {}", group_name),
                    frequency: group_nodes.len(),
                    confidence: 0.75,
                    examples: group_nodes.into_iter().take(5).collect(),
                    properties: HashMap::new(),
                };
                
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    /// Extract behavioral patterns
    fn extract_behavioral_patterns(
        &self,
        nodes: &[NodeId],
        graph: &GraphStorage,
    ) -> Result<Vec<ExtractedPattern>> {
        let mut patterns = Vec::new();
        
        // Analyze access patterns
        let mut access_sequences = HashMap::new();
        
        for node_id in nodes {
            let node = graph.get_node(node_id)?;
            
            // Group by access frequency patterns
            let access_pattern = self.categorize_access_pattern(&node);
            access_sequences.entry(access_pattern.clone())
                .or_insert_with(Vec::new)
                .push(node_id.clone());
        }
        
        for (pattern_name, nodes) in access_sequences {
            if nodes.len() >= self.min_frequency {
                let pattern = ExtractedPattern {
                    pattern_type: "behavioral".to_string(),
                    description: format!("Access pattern: {}", pattern_name),
                    frequency: nodes.len(),
                    confidence: 0.85,
                    examples: nodes.into_iter().take(5).collect(),
                    properties: HashMap::new(),
                };
                
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    /// Extract cognitive patterns
    fn extract_cognitive_patterns(
        &self,
        nodes: &[NodeId],
        graph: &GraphStorage,
    ) -> Result<Vec<ExtractedPattern>> {
        let mut patterns = Vec::new();
        
        // Group by cognitive patterns
        let mut cognitive_groups: HashMap<String, Vec<NodeId>> = HashMap::new();
        
        for node_id in nodes {
            let node = graph.get_node(node_id)?;
            let pattern_key = format!("{:?}-{:?}", 
                node.cognitive_metadata.pattern,
                node.cognitive_metadata.thinking_style
            );
            
            cognitive_groups.entry(pattern_key)
                .or_insert_with(Vec::new)
                .push(node_id.clone());
        }
        
        for (pattern_key, nodes) in cognitive_groups {
            if nodes.len() >= self.min_frequency {
                let pattern = ExtractedPattern {
                    pattern_type: "cognitive".to_string(),
                    description: format!("Cognitive pattern: {}", pattern_key),
                    frequency: nodes.len(),
                    confidence: 0.9,
                    examples: nodes.into_iter().take(5).collect(),
                    properties: HashMap::new(),
                };
                
                patterns.push(pattern);
            }
        }
        
        Ok(patterns)
    }
    
    /// Find frequent subgraphs
    fn find_frequent_subgraphs(
        &self,
        nodes: &[NodeId],
        graph: &GraphStorage,
    ) -> Result<Vec<SubgraphPattern>> {
        // Simplified implementation
        // In production, use algorithms like gSpan or FSG
        Ok(vec![])
    }
    
    /// Find temporal sequences
    fn find_temporal_sequences(
        &self,
        nodes: &[NodeId],
        graph: &GraphStorage,
    ) -> Result<HashMap<String, Vec<NodeId>>> {
        // Simplified implementation
        Ok(HashMap::new())
    }
    
    /// Group nodes by semantic similarity
    fn group_by_semantics(
        &self,
        nodes: &[NodeId],
        graph: &GraphStorage,
    ) -> Result<HashMap<String, Vec<NodeId>>> {
        // Simplified implementation
        // In production, use clustering algorithms
        Ok(HashMap::new())
    }
    
    /// Categorize access pattern
    fn categorize_access_pattern(&self, node: &ConsciousNode) -> String {
        match node.learning_state.access_frequency {
            f if f > 0.8 => "high_frequency".to_string(),
            f if f > 0.5 => "medium_frequency".to_string(),
            f if f > 0.2 => "low_frequency".to_string(),
            _ => "rare".to_string(),
        }
    }
    
    /// Calculate confidence for structural patterns
    fn calculate_structural_confidence(&self, subgraph: &SubgraphPattern) -> f32 {
        // Simple confidence based on frequency and consistency
        let base_confidence = (subgraph.frequency as f32) / 10.0;
        base_confidence.min(1.0)
    }
    
    /// Calculate confidence for temporal patterns (for legacy method)
    fn calculate_temporal_confidence_legacy(&self, instances: &[NodeId]) -> f32 {
        // Simple confidence based on instance count
        let base_confidence = (instances.len() as f32) / 10.0;
        base_confidence.min(1.0)
    }
}

/// Pattern mining algorithms
pub struct PatternMiner {
    pub min_support: f32,
    pub max_pattern_size: usize,
}

impl PatternMiner {
    /// Mine association rules from the graph
    pub fn mine_association_rules(
        &self,
        graph: &GraphStorage,
    ) -> Result<Vec<AssociationRule>> {
        // Simplified implementation
        Ok(vec![])
    }
    
    /// Mine sequential patterns
    pub fn mine_sequential_patterns(
        &self,
        graph: &GraphStorage,
    ) -> Result<Vec<SequentialPattern>> {
        // Simplified implementation
        Ok(vec![])
    }
}

/// Association rule
#[derive(Debug, Clone)]
pub struct AssociationRule {
    pub antecedent: Vec<NodeId>,
    pub consequent: Vec<NodeId>,
    pub support: f32,
    pub confidence: f32,
}

/// Sequential pattern
#[derive(Debug, Clone)]
pub struct SequentialPattern {
    pub sequence: Vec<NodeId>,
    pub support: f32,
    pub instances: Vec<Vec<NodeId>>,
}

