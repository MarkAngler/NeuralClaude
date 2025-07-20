//! Enhanced search with graph traversal capabilities

use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};
use crate::{
    adaptive::AdaptiveMemoryModule,
    graph::{ConsciousGraph, GraphOperations, TraversalOptions, ConsciousNode, EdgeType},
};

/// Result from enhanced search including graph traversal
#[derive(Debug, Clone)]
pub struct EnhancedSearchResult {
    pub query: String,
    pub operation_id: String,
    pub matches: Vec<MatchResult>,
}

/// Individual match result with relationship information
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub key: String,
    pub content: String,
    pub score: f32,
    pub match_type: MatchType,
    pub relationship: Option<RelationshipInfo>,
    pub distance: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum MatchType {
    Direct,
    Related,
}

#[derive(Debug, Clone)]
pub struct RelationshipInfo {
    pub edge_type: String,
    pub path_from: String,
    pub description: String,
}

/// Perform enhanced search with optional graph traversal
pub async fn enhanced_search(
    module: &AdaptiveMemoryModule,
    graph: &ConsciousGraph,
    query: &str,
    limit: usize,
    include_related: bool,
    traversal_depth: usize,
    max_related: usize,
    follow_types: Option<Vec<String>>,
) -> Result<EnhancedSearchResult> {
    // First, perform regular embedding-based search
    let (direct_results, operation_id) = module.search(query, limit).await
        .map_err(|e| anyhow!("Search failed: {}", e))?;
    
    let mut all_matches = Vec::new();
    let mut visited_keys = HashSet::new();
    
    // Add direct matches
    for (key, score) in direct_results {
        visited_keys.insert(key.clone());
        
        // Retrieve full content
        let content = if let Ok((Some(mem), _)) = module.retrieve(&key).await {
            mem
        } else {
            key.clone()
        };
        
        all_matches.push(MatchResult {
            key: key.clone(),
            content,
            score,
            match_type: MatchType::Direct,
            relationship: None,
            distance: Some(0),
        });
    }
    
    // If graph traversal is requested
    if include_related && !all_matches.is_empty() {
        let mut related_count = 0;
        
        // Use top direct matches as starting points for traversal
        for direct_match in &all_matches.clone() {
            if related_count >= max_related {
                break;
            }
            
            // Configure traversal options
            let options = TraversalOptions {
                max_depth: traversal_depth,
                max_nodes: max_related - related_count,
                edge_filter: follow_types.clone(),
                follow_emotional: true,
                consciousness_threshold: Some(0.3),
                ..Default::default()
            };
            
            // Perform graph traversal from this node
            if let Ok(graph_result) = graph.query_graph(&direct_match.key, traversal_depth) {
                for (idx, node) in graph_result.nodes.iter().enumerate() {
                    // Skip if we've already added this key
                    if visited_keys.contains(&node.key) {
                        continue;
                    }
                    
                    visited_keys.insert(node.key.clone());
                    related_count += 1;
                    
                    // Determine relationship based on edges
                    let relationship = if idx < graph_result.edges.len() {
                        let edge = &graph_result.edges[idx];
                        Some(RelationshipInfo {
                            edge_type: format!("{:?}", edge.edge_type),
                            path_from: direct_match.key.clone(),
                            description: describe_edge_type(&edge.edge_type),
                        })
                    } else {
                        None
                    };
                    
                    all_matches.push(MatchResult {
                        key: node.key.clone(),
                        content: node.content.clone(),
                        score: calculate_related_score(direct_match.score, idx + 1),
                        match_type: MatchType::Related,
                        relationship,
                        distance: Some(idx + 1),
                    });
                    
                    if related_count >= max_related {
                        break;
                    }
                }
            }
        }
    }
    
    Ok(EnhancedSearchResult {
        query: query.to_string(),
        operation_id,
        matches: all_matches,
    })
}

/// Calculate score for related nodes based on distance
fn calculate_related_score(base_score: f32, distance: usize) -> f32 {
    // Decay score based on graph distance
    base_score * (0.8_f32).powi(distance as i32)
}

/// Generate human-readable description for edge types
fn describe_edge_type(edge_type: &EdgeType) -> String {
    match edge_type {
        EdgeType::Related { weight } => format!("Related (weight: {:.2})", weight),
        EdgeType::PartOf => "Part of".to_string(),
        EdgeType::CausedBy => "Caused by".to_string(),
        EdgeType::Temporal { delta_ms } => format!("Temporal connection ({} ms apart)", delta_ms),
        EdgeType::Derived => "Derived insight".to_string(),
        EdgeType::Association { strength } => format!("Associated (strength: {:.2})", strength),
        EdgeType::Semantic => "Semantic relationship".to_string(),
    }
}

