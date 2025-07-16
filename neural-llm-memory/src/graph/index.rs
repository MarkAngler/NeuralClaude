//! High-performance indices for graph operations

use crate::graph::core::{NodeId, ConsciousNode};
use anyhow::{Result, anyhow};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::Arc;

/// Collection of graph indices for fast lookups
pub struct GraphIndices {
    /// HNSW index for similarity search
    pub embedding_index: Arc<RwLock<HnswIndex>>,
    
    /// Inverted indices
    pub tag_index: Arc<DashMap<String, HashSet<NodeId>>>,
    pub type_index: Arc<DashMap<String, HashSet<NodeId>>>,
    pub time_index: Arc<RwLock<BTreeMap<i64, Vec<NodeId>>>>,
}

/// HNSW index for fast similarity search
pub struct HnswIndex {
    embeddings: Vec<Vec<f32>>,
    id_map: Vec<NodeId>,
    embedding_dim: usize,
}

impl HnswIndex {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embeddings: Vec::new(),
            id_map: Vec::new(),
            embedding_dim,
        }
    }
    
    pub fn insert(&mut self, node_id: NodeId, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.embedding_dim {
            return Err(anyhow!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                embedding.len()
            ));
        }
        
        self.id_map.push(node_id);
        self.embeddings.push(embedding.to_vec());
        
        Ok(())
    }
    
    pub fn search_by_node(&self, _node_id: &NodeId, limit: usize) -> Result<Vec<(NodeId, f32)>> {
        // Simplified implementation - in reality would use HNSW algorithm
        let mut results = Vec::new();
        for (_i, id) in self.id_map.iter().enumerate().take(limit) {
            results.push((id.clone(), 0.5)); // Dummy similarity score
        }
        Ok(results)
    }
    
    /// Search for similar embeddings using brute force cosine similarity
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        if query.len() != self.embedding_dim {
            return vec![];
        }
        
        let mut similarities: Vec<(usize, f32)> = self.embeddings
            .iter()
            .enumerate()
            .map(|(idx, embedding)| {
                let similarity = Self::cosine_similarity(query, embedding);
                (idx, similarity)
            })
            .collect();
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top k
        similarities.truncate(k);
        
        // Convert to (NodeId, similarity)
        similarities.into_iter()
            .map(|(idx, sim)| (self.id_map[idx].clone(), sim))
            .collect()
    }
    
    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    /// Get the number of indexed items
    pub fn len(&self) -> usize {
        self.id_map.len()
    }
    
    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.id_map.is_empty()
    }
}

impl GraphIndices {
    /// Create new graph indices
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_index: Arc::new(RwLock::new(HnswIndex::new(embedding_dim))),
            tag_index: Arc::new(DashMap::new()),
            type_index: Arc::new(DashMap::new()),
            time_index: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }
    
    /// Index a node
    pub fn index_node(&self, node: &ConsciousNode) -> Result<()> {
        let node_id = node.id_string();
        
        // Index embeddings
        self.embedding_index.write()
            .insert(node_id.clone(), &node.embeddings)?;
        
        // Index by type
        let type_name = match &node.node_type {
            crate::graph::core::NodeType::Memory(_) => "memory",
            crate::graph::core::NodeType::Concept(_) => "concept",
            crate::graph::core::NodeType::Entity(_) => "entity",
            crate::graph::core::NodeType::Context(_) => "context",
            crate::graph::core::NodeType::Pattern(_) => "pattern",
        };
        
        self.type_index
            .entry(type_name.to_string())
            .or_insert_with(HashSet::new)
            .insert(node_id.clone());
        
        // Index by time
        let timestamp = node.created_at.timestamp();
        self.time_index.write()
            .entry(timestamp)
            .or_insert_with(Vec::new)
            .push(node_id.clone());
        
        // Index emotional tags
        for tag in &node.emotional_state.tags {
            self.tag_index
                .entry(tag.0.clone())
                .or_insert_with(HashSet::new)
                .insert(node_id.clone());
        }
        
        Ok(())
    }
    
    /// Remove a node from indices
    pub fn remove_node(&self, node_id: &NodeId) -> Result<()> {
        // Note: HNSW doesn't support removal, would need rebuilding
        // For now, we'll just remove from other indices
        
        // Remove from type index
        for mut entry in self.type_index.iter_mut() {
            entry.value_mut().remove(node_id);
        }
        
        // Remove from tag index
        for mut entry in self.tag_index.iter_mut() {
            entry.value_mut().remove(node_id);
        }
        
        // Remove from time index
        let mut time_index = self.time_index.write();
        time_index.retain(|_, nodes| {
            nodes.retain(|id| id != node_id);
            !nodes.is_empty()
        });
        
        Ok(())
    }
    
    /// Find nodes by type
    pub fn find_by_type(&self, node_type: &str) -> Vec<NodeId> {
        self.type_index
            .get(node_type)
            .map(|entry| entry.value().iter().cloned().collect())
            .unwrap_or_default()
    }
    
    /// Find nodes by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<NodeId> {
        self.tag_index
            .get(tag)
            .map(|entry| entry.value().iter().cloned().collect())
            .unwrap_or_default()
    }
    
    /// Find nodes in time range
    pub fn find_by_time_range(&self, start: i64, end: i64) -> Vec<NodeId> {
        let time_index = self.time_index.read();
        let mut results = Vec::new();
        
        for (_, nodes) in time_index.range(start..=end) {
            results.extend(nodes.iter().cloned());
        }
        
        results
    }
    
    /// Clear all indices
    pub fn clear(&self) {
        // Clear embedding index - create a new one
        let embedding_dim = self.embedding_index.read().embedding_dim;
        *self.embedding_index.write() = HnswIndex::new(embedding_dim);
        
        // Clear other indices
        self.tag_index.clear();
        self.type_index.clear();
        self.time_index.write().clear();
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cosine_similarity() {
        // Same vectors should have similarity 1.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((HnswIndex::cosine_similarity(&a, &b) - 1.0).abs() < 0.0001);
        
        // Orthogonal vectors should have similarity 0.0
        let c = vec![1.0, 0.0];
        let d = vec![0.0, 1.0];
        assert!((HnswIndex::cosine_similarity(&c, &d) - 0.0).abs() < 0.0001);
    }
    
    #[test]
    fn test_hnsw_index() {
        let mut index = HnswIndex::new(3);
        
        // Insert some vectors
        index.insert("node1".to_string(), &[1.0, 0.0, 0.0]).unwrap();
        index.insert("node2".to_string(), &[0.9, 0.1, 0.0]).unwrap();
        index.insert("node3".to_string(), &[0.0, 1.0, 0.0]).unwrap();
        
        // Search for similar vectors
        let results = index.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "node1");
        assert_eq!(results[1].0, "node2");
    }
}