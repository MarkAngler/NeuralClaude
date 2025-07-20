//! Graph algorithms for traversal and analysis

use crate::graph::core::{NodeId, EdgeType, ConsciousNode, ConsciousEdge};
use crate::graph::storage::GraphStorage;
use crate::graph::{GraphPath, GraphQueryResult, QueryStats};
use anyhow::{Result, anyhow};
use petgraph::algo::{dijkstra, all_simple_paths};
use petgraph::visit::{EdgeRef, Bfs, Dfs};
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

/// Collection of graph algorithms
pub struct GraphAlgorithms;

/// Options for graph traversal
#[derive(Debug, Clone)]
pub struct TraversalOptions {
    pub max_depth: usize,
    pub max_nodes: usize,
    pub edge_filter: Option<Vec<String>>,
    pub min_edge_strength: Option<f32>,
    pub follow_emotional: bool,
    pub consciousness_threshold: Option<f32>,
}

impl Default for TraversalOptions {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_nodes: 50,
            edge_filter: None,
            min_edge_strength: None,
            follow_emotional: false,
            consciousness_threshold: None,
        }
    }
}

impl GraphAlgorithms {
    /// Breadth-first search from a starting node
    pub fn bfs(
        storage: &GraphStorage,
        start: &NodeId,
        options: TraversalOptions,
    ) -> Result<GraphQueryResult> {
        let start_time = Instant::now();
        let mut visited_nodes = Vec::new();
        let mut visited_edges = Vec::new();
        let mut visited_set = HashSet::new();
        let mut consciousness_activations = 0;
        
        // Get starting node index
        let start_idx = storage.node_index
            .get(start)
            .ok_or_else(|| anyhow!("Start node {} not found", start))?
            .clone();
        
        let graph = storage.graph.read();
        let mut queue = VecDeque::new();
        queue.push_back((start_idx, 0));
        visited_set.insert(start_idx);
        
        while let Some((current_idx, depth)) = queue.pop_front() {
            if depth >= options.max_depth || visited_nodes.len() >= options.max_nodes {
                break;
            }
            
            // Get node
            if let Some(node) = graph.node_weight(current_idx) {
                // Check consciousness threshold
                if let Some(threshold) = options.consciousness_threshold {
                    if node.awareness.level >= threshold {
                        consciousness_activations += 1;
                    }
                }
                
                visited_nodes.push(node.clone());
                
                // Explore edges
                for edge_ref in graph.edges(current_idx) {
                    let target_idx = edge_ref.target();
                    let edge = edge_ref.weight();
                    
                    // Apply filters
                    if !Self::should_follow_edge(edge, &options) {
                        continue;
                    }
                    
                    if !visited_set.contains(&target_idx) {
                        visited_set.insert(target_idx);
                        queue.push_back((target_idx, depth + 1));
                        visited_edges.push(edge.clone());
                    }
                }
            }
        }
        
        let stats = QueryStats {
            total_traversed: visited_nodes.len(),
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            consciousness_activations,
        };
        
        Ok(GraphQueryResult {
            nodes: visited_nodes,
            edges: visited_edges,
            paths: vec![], // BFS doesn't return specific paths
            stats,
        })
    }
    
    /// Depth-first search from a starting node
    pub fn dfs(
        storage: &GraphStorage,
        start: &NodeId,
        options: TraversalOptions,
    ) -> Result<GraphQueryResult> {
        let start_time = Instant::now();
        let mut visited_nodes = Vec::new();
        let mut visited_edges = Vec::new();
        let mut visited_set = HashSet::new();
        let mut consciousness_activations = 0;
        
        // Get starting node index
        let start_idx = storage.node_index
            .get(start)
            .ok_or_else(|| anyhow!("Start node {} not found", start))?
            .clone();
        
        let graph = storage.graph.read();
        let mut stack = vec![(start_idx, 0)];
        
        while let Some((current_idx, depth)) = stack.pop() {
            if depth >= options.max_depth || visited_nodes.len() >= options.max_nodes {
                break;
            }
            
            if visited_set.contains(&current_idx) {
                continue;
            }
            
            visited_set.insert(current_idx);
            
            // Get node
            if let Some(node) = graph.node_weight(current_idx) {
                // Check consciousness threshold
                if let Some(threshold) = options.consciousness_threshold {
                    if node.awareness.level >= threshold {
                        consciousness_activations += 1;
                    }
                }
                
                visited_nodes.push(node.clone());
                
                // Explore edges in reverse order for DFS
                let mut edges: Vec<_> = graph.edges(current_idx).collect();
                edges.reverse();
                
                for edge_ref in edges {
                    let target_idx = edge_ref.target();
                    let edge = edge_ref.weight();
                    
                    // Apply filters
                    if !Self::should_follow_edge(edge, &options) {
                        continue;
                    }
                    
                    if !visited_set.contains(&target_idx) {
                        stack.push((target_idx, depth + 1));
                        visited_edges.push(edge.clone());
                    }
                }
            }
        }
        
        let stats = QueryStats {
            total_traversed: visited_nodes.len(),
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            consciousness_activations,
        };
        
        Ok(GraphQueryResult {
            nodes: visited_nodes,
            edges: visited_edges,
            paths: vec![],
            stats,
        })
    }
    
    /// Find all paths between two nodes
    pub fn find_paths(
        storage: &GraphStorage,
        source: &NodeId,
        target: &NodeId,
        max_length: usize,
    ) -> Result<Vec<GraphPath>> {
        let graph = storage.graph.read();
        
        let source_idx = storage.node_index
            .get(source)
            .ok_or_else(|| anyhow!("Source node {} not found", source))?;
        let target_idx = storage.node_index
            .get(target)
            .ok_or_else(|| anyhow!("Target node {} not found", target))?;
        
        // Find all simple paths
        let paths: Vec<Vec<_>> = all_simple_paths(
            &*graph,
            *source_idx,
            *target_idx,
            0,
            Some(max_length),
        ).collect();
        
        // Convert to GraphPath
        let mut graph_paths = Vec::new();
        
        for path in paths {
            let mut nodes = Vec::new();
            let mut edges = Vec::new();
            let mut total_weight = 1.0;
            
            // Convert node indices to IDs
            for &idx in &path {
                if let Some(node) = graph.node_weight(idx) {
                    nodes.push(node.id_string());
                }
            }
            
            // Get edges
            for window in path.windows(2) {
                if let Some(edge_idx) = graph.find_edge(window[0], window[1]) {
                    if let Some(edge) = graph.edge_weight(edge_idx) {
                        edges.push(edge.edge_type.clone());
                        
                        // Calculate combined weight
                        if let EdgeType::Related { weight } = &edge.edge_type {
                            total_weight *= weight;
                        }
                        total_weight *= edge.strength.combined;
                    }
                }
            }
            
            graph_paths.push(GraphPath {
                nodes,
                edges,
                total_weight,
            });
        }
        
        // Sort by total weight (descending)
        graph_paths.sort_by(|a, b| b.total_weight.partial_cmp(&a.total_weight).unwrap());
        
        Ok(graph_paths)
    }
    
    /// Consciousness-aware traversal
    pub fn consciousness_traversal(
        storage: &GraphStorage,
        start: &NodeId,
        activation_threshold: f32,
        max_depth: usize,
    ) -> Result<GraphQueryResult> {
        let start_time = Instant::now();
        let mut visited_nodes = Vec::new();
        let mut visited_edges = Vec::new();
        let mut consciousness_activations = 0;
        
        // Priority queue based on consciousness level
        let mut queue: Vec<(NodeId, f32, usize)> = vec![(start.clone(), 1.0, 0)];
        let mut visited = HashSet::new();
        
        while let Some((node_id, activation, depth)) = queue.pop() {
            if depth >= max_depth || visited.contains(&node_id) {
                continue;
            }
            
            visited.insert(node_id.clone());
            
            // Get node
            let node = storage.get_node(&node_id)?;
            
            // Check if node's consciousness exceeds threshold
            if node.awareness.level * activation >= activation_threshold {
                consciousness_activations += 1;
                visited_nodes.push(node.clone());
                
                // Get edges and propagate activation
                let edges = storage.get_edges_for_node(&node_id)?;
                
                for edge in edges {
                    if visited.contains(&edge.target) {
                        continue;
                    }
                    
                    // Calculate propagated activation
                    let propagated_activation = activation 
                        * edge.strength.combined 
                        * edge.consciousness.emotional_resonance
                        * (1.0 - if edge.consciousness.inhibitory { 0.5 } else { 0.0 });
                    
                    if propagated_activation >= activation_threshold {
                        queue.push((edge.target.clone(), propagated_activation, depth + 1));
                        visited_edges.push(edge);
                    }
                }
                
                // Sort queue by activation level (highest first)
                queue.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
        }
        
        let stats = QueryStats {
            total_traversed: visited_nodes.len(),
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            consciousness_activations,
        };
        
        Ok(GraphQueryResult {
            nodes: visited_nodes,
            edges: visited_edges,
            paths: vec![],
            stats,
        })
    }
    
    /// Check if an edge should be followed based on options
    fn should_follow_edge(edge: &ConsciousEdge, options: &TraversalOptions) -> bool {
        // Check edge strength
        if let Some(min_strength) = options.min_edge_strength {
            if edge.strength.combined < min_strength {
                return false;
            }
        }
        
        // Check edge type filter
        if let Some(ref filter) = options.edge_filter {
            let edge_type_name = match &edge.edge_type {
                EdgeType::Related { .. } => "related",
                EdgeType::PartOf => "part_of",
                EdgeType::CausedBy => "caused_by",
                EdgeType::Temporal { .. } => "temporal",
                EdgeType::Derived => "derived",
                EdgeType::Association { .. } => "association",
            };
            
            if !filter.contains(&edge_type_name.to_string()) {
                return false;
            }
        }
        
        // Check emotional following
        if options.follow_emotional && edge.consciousness.emotional_resonance < 0.5 {
            return false;
        }
        
        true
    }
}

/// Clustering algorithms for graph analysis
pub enum ClusteringAlgorithm {
    Louvain,
    LabelPropagation,
    ConnectedComponents,
}

/// A cluster of nodes
#[derive(Debug, Clone)]
pub struct GraphCluster {
    pub id: String,
    pub nodes: Vec<NodeId>,
    pub size: usize,
    pub density: f32,
    pub description: String,
}

