//! Compatibility layer for backward compatibility with key-value storage

use crate::memory::{MemoryBank, MemoryOperations, MemoryKey, MemoryValue, MemoryMetadata, MemoryConfig};
use crate::graph::{GraphStorage, GraphIndices, ConsciousNode, NodeType, EdgeType, ConsciousGraphConfig};
use crate::graph::core::{MemoryNode, ConsciousEdge, NodeId};
use crate::graph::inference::RelationshipInference;
use anyhow::{Result, anyhow};
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::Mutex;
use ndarray::{Array1, Array2};
use chrono::Utc;

/// Hybrid memory bank that maintains both KV and graph storage
pub struct HybridMemoryBank {
    /// Existing KV storage (for compatibility)
    kv_bank: Arc<RwLock<MemoryBank>>,
    
    /// Graph storage
    graph: Arc<GraphStorage>,
    
    /// Graph indices
    indices: Arc<GraphIndices>,
    
    /// Inference engine
    inference: Arc<RelationshipInference>,
    
    /// Configuration
    config: HybridMemoryConfig,
    
    /// Migration state
    migration_state: Arc<Mutex<MigrationState>>,
}

/// Configuration for hybrid memory
#[derive(Debug, Clone)]
pub struct HybridMemoryConfig {
    pub graph_enabled: bool,
    pub auto_infer_relationships: bool,
    pub migration_batch_size: usize,
    pub embedding_dim: usize,
}

impl Default for HybridMemoryConfig {
    fn default() -> Self {
        Self {
            graph_enabled: true,
            auto_infer_relationships: true,
            migration_batch_size: 100,
            embedding_dim: 768,
        }
    }
}

/// State of KV to graph migration
#[derive(Debug, Clone)]
pub struct MigrationState {
    pub total_keys: usize,
    pub migrated_keys: usize,
    pub in_progress: bool,
    pub last_error: Option<String>,
}

/// Statistics for the hybrid memory bank
#[derive(Debug, Clone)]
pub struct HybridStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub kv_count: usize,
    pub graph_enabled: bool,
    pub total_memories: usize,
}

impl HybridMemoryBank {
    /// Create a new hybrid memory bank from components
    pub fn from_components(
        kv_bank: MemoryBank,
        storage_path: std::path::PathBuf,
        config: HybridMemoryConfig,
    ) -> Result<Self> {
        let graph = Arc::new(GraphStorage::new(storage_path)?);
        let indices = Arc::new(GraphIndices::new(config.embedding_dim));
        let inference = Arc::new(RelationshipInference::default());
        
        Ok(Self {
            kv_bank: Arc::new(RwLock::new(kv_bank)),
            graph,
            indices,
            inference,
            config,
            migration_state: Arc::new(Mutex::new(MigrationState {
                total_keys: 0,
                migrated_keys: 0,
                in_progress: false,
                last_error: None,
            })),
        })
    }
    
    /// Store memory in both KV and graph
    async fn store_dual(&self, key: MemoryKey, value: MemoryValue) -> Result<String> {
        // 1. Store in KV (existing behavior)
        self.kv_bank.write().store(key.clone(), value.clone())?;
        
        // 2. Create graph node if enabled
        if self.config.graph_enabled {
            let node = self.create_conscious_node(key.clone(), value)?;
            let node_id = self.graph.add_node(node.clone())?;
            
            // 3. Index the node
            self.indices.index_node(&node)?;
            
            // 4. Auto-discover relationships if enabled
            if self.config.auto_infer_relationships {
                tokio::spawn({
                    let inference = self.inference.clone();
                    let graph = self.graph.clone();
                    let indices = self.indices.clone();
                    let node_id = node_id.clone();
                    
                    async move {
                        if let Err(e) = inference.infer_for_new_node(&node_id, &graph, &indices).await {
                            eprintln!("Failed to infer relationships: {}", e);
                        }
                    }
                });
            }
            
            Ok(node_id)
        } else {
            Ok(key.id.clone())
        }
    }
    
    /// Search using graph-enhanced retrieval
    async fn search_enhanced(
        &self, 
        query_embedding: &Array2<f32>, 
        k: usize
    ) -> Vec<(MemoryKey, MemoryValue, f32)> {
        if !self.config.graph_enabled {
            // Fall back to KV search
            return self.kv_bank.read().search(query_embedding, k);
        }
        
        let embedding_slice = query_embedding.as_slice().unwrap();
        
        // 1. Find similar nodes using HNSW
        let similar = self.indices.embedding_index.read()
            .search(embedding_slice, k * 2); // Get more candidates
        
        // 2. Expand with graph context
        let expanded = self.expand_with_graph_context(similar, k).await;
        
        // 3. Convert to legacy format
        let mut results = Vec::new();
        for (node_id, score) in expanded {
            if let Ok(node) = self.graph.get_node(&node_id) {
                if let Some((key, value)) = self.node_to_memory(&node) {
                    results.push((key, value, score));
                }
            }
        }
        
        results
    }
    
    /// Create a ConsciousNode from memory
    fn create_conscious_node(&self, key: MemoryKey, value: MemoryValue) -> Result<ConsciousNode> {
        let now = Utc::now();
        
        // Use embeddings directly (they're already Vec<f32>)
        let embeddings = value.embedding.clone();
        
        // Create memory node
        let memory_node = MemoryNode {
            id: uuid::Uuid::new_v4().to_string(),
            key: key.id.clone(),
            value: value.content.clone(),
            embedding: embeddings.clone(),
            created_at: now,
            accessed_at: now,
            access_count: 0,
        };
        
        // Create conscious node
        let node = ConsciousNode::new(
            key.id.clone(),
            value.content,
            embeddings,
            NodeType::Memory(memory_node),
        );
        
        Ok(node)
    }
    
    /// Convert a ConsciousNode back to memory format
    fn node_to_memory(&self, node: &ConsciousNode) -> Option<(MemoryKey, MemoryValue)> {
        match &node.node_type {
            NodeType::Memory(mem_node) => {
                // Create a MemoryKey from the stored key string
                let memory_key = MemoryKey {
                    id: mem_node.key.clone(),
                    timestamp: node.created_at.timestamp() as u64,
                    context_hash: 0, // Default value, could be computed from content
                };
                
                // Create metadata from node properties
                let metadata = MemoryMetadata {
                    importance: node.awareness.level,
                    access_count: mem_node.access_count as u32,
                    last_accessed: mem_node.accessed_at.timestamp() as u64,
                    decay_factor: 0.9, // Default decay factor
                    tags: node.emotional_state.tags.iter().map(|tag| tag.0.clone()).collect(),
                };
                
                let value = MemoryValue {
                    content: mem_node.value.clone(),
                    embedding: mem_node.embedding.clone(),
                    metadata,
                };
                
                Some((memory_key, value))
            }
            _ => None,
        }
    }
    
    /// Expand search results with graph context
    async fn expand_with_graph_context(
        &self,
        initial: Vec<(String, f32)>,
        k: usize,
    ) -> Vec<(String, f32)> {
        let mut expanded = Vec::new();
        let mut seen = std::collections::HashSet::new();
        
        // Add initial results
        for (node_id, score) in initial {
            if seen.insert(node_id.clone()) {
                expanded.push((node_id.clone(), score));
            }
            
            // Get connected nodes
            if let Ok(edges) = self.graph.get_edges_for_node(&node_id) {
                for edge in edges {
                    if seen.insert(edge.target.clone()) {
                        // Decay score based on edge strength
                        let connected_score = score * edge.strength.combined * 0.8;
                        expanded.push((edge.target, connected_score));
                    }
                }
            }
        }
        
        // Sort by score and limit
        expanded.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        expanded.truncate(k);
        
        expanded
    }
    
    /// Migrate existing KV data to graph
    pub async fn migrate_to_graph(&self) -> Result<()> {
        let mut state = self.migration_state.lock().await;
        if state.in_progress {
            return Err(anyhow!("Migration already in progress"));
        }
        
        state.in_progress = true;
        state.last_error = None;
        
        // Get all keys from KV store
        let all_memories = self.kv_bank.read().get_all_memories();
        let keys: Vec<_> = all_memories.iter().map(|(k, _)| k.clone()).collect();
        
        state.total_keys = keys.len();
        state.migrated_keys = 0;
        
        // Process in batches
        for chunk in keys.chunks(self.config.migration_batch_size) {
            for key in chunk {
                match self.migrate_single_key(key).await {
                    Ok(_) => {
                        state.migrated_keys += 1;
                    }
                    Err(e) => {
                        state.last_error = Some(e.to_string());
                        eprintln!("Failed to migrate key {:?}: {}", key, e);
                    }
                }
            }
            
            // Save snapshot periodically
            if state.migrated_keys % 1000 == 0 {
                self.graph.save_snapshot().await?;
            }
        }
        
        state.in_progress = false;
        Ok(())
    }
    
    /// Migrate a single key
    async fn migrate_single_key(&self, key: &MemoryKey) -> Result<()> {
        let value = self.kv_bank.write().retrieve(key)
            .map_err(|e| anyhow!("Failed to retrieve key: {}", e))?
            .ok_or_else(|| anyhow!("Key not found"))?;
        
        let node = self.create_conscious_node(key.clone(), value.clone())?;
        let node_id = self.graph.add_node(node.clone())?;
        self.indices.index_node(&node)?;
        
        // Infer relationships
        if self.config.auto_infer_relationships {
            self.inference.infer_for_new_node(&node_id, &self.graph, &self.indices).await?;
        }
        
        Ok(())
    }
    
    /// Enable graph features dynamically
    pub fn enable_graph_features(&mut self) {
        self.config.graph_enabled = true;
        self.config.auto_infer_relationships = true;
    }
    
    /// Disable graph features dynamically
    pub fn disable_graph_features(&mut self) {
        self.config.graph_enabled = false;
        self.config.auto_infer_relationships = false;
    }
    
    /// Check if graph features are enabled
    pub fn is_graph_enabled(&self) -> bool {
        self.config.graph_enabled
    }
    
    /// Get migration progress
    pub async fn get_migration_progress(&self) -> MigrationState {
        self.migration_state.lock().await.clone()
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: HybridMemoryConfig) {
        self.config = config;
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> &HybridMemoryConfig {
        &self.config
    }
    
    /// Get memories by modality (placeholder for testing)
    pub async fn get_memories_by_modality(&self, modality: crate::graph::cross_modal::MemoryModality) -> Result<Vec<MemoryValue>> {
        // For now, return empty vector - this would need proper implementation
        // to filter memories by their registered modality
        Ok(Vec::new())
    }
    
    /// Get node ID for a key
    pub async fn get_node_id(&self, key: &String) -> Result<NodeId> {
        self.graph.key_to_node.get(key)
            .map(|r| r.clone())
            .ok_or_else(|| anyhow!("Node not found for key: {}", key))
    }
    
    /// Create a new hybrid memory bank with default paths
    pub async fn new(memory_config: MemoryConfig, graph_config: ConsciousGraphConfig) -> Result<Self> {
        let kv_bank = MemoryBank::new(10000, 100); // max memories, cache size
        let storage_path = std::path::PathBuf::from("./test_graph_storage");
        let config = HybridMemoryConfig::default();
        
        let graph = Arc::new(GraphStorage::new(storage_path)?);
        let indices = Arc::new(GraphIndices::new(config.embedding_dim));
        let inference = Arc::new(RelationshipInference::default());
        
        Ok(Self {
            kv_bank: Arc::new(RwLock::new(kv_bank)),
            graph,
            indices,
            inference,
            config,
            migration_state: Arc::new(Mutex::new(MigrationState {
                total_keys: 0,
                migrated_keys: 0,
                in_progress: false,
                last_error: None,
            })),
        })
    }
    
    /// Async store method for testing
    pub async fn store(&self, key: String, value: MemoryValue) -> Result<()> {
        let memory_key = MemoryKey {
            id: key.clone(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            context_hash: 0,
        };
        self.store_dual(memory_key, value).await?;
        Ok(())
    }
    
    /// Async retrieve method for testing
    pub async fn retrieve(&self, key: &String) -> Result<MemoryValue> {
        let memory_key = MemoryKey {
            id: key.clone(),
            timestamp: 0, // Not used for retrieval
            context_hash: 0,
        };
        let mut bank = self.kv_bank.write();
        bank.retrieve(&memory_key)?
            .ok_or_else(|| anyhow!("Key not found"))
    }
    
    /// Async search method for testing
    pub async fn search(&self, query: &Array1<f32>, k: usize) -> Result<Vec<(String, f32)>> {
        // Convert Array1 to Array2 for compatibility
        let query_2d = query.clone().insert_axis(ndarray::Axis(0));
        let results = self.search_enhanced(&query_2d, k).await;
        
        Ok(results.into_iter()
            .map(|(key, _value, score)| (key.id, score))
            .collect())
    }
    
    /// Get the underlying graph storage
    pub async fn get_graph(&self) -> Arc<GraphStorage> {
        self.graph.clone()
    }
    
    /// Get stats about the hybrid memory bank
    pub async fn get_stats(&self) -> Result<HybridStats> {
        let node_count = self.graph.node_count();
        let edge_count = self.graph.edge_count();
        let (kv_count, _, _, _) = self.kv_bank.read().get_stats();
        
        Ok(HybridStats {
            node_count,
            edge_count,
            kv_count,
            graph_enabled: self.config.graph_enabled,
            total_memories: node_count.max(kv_count),
        })
    }
}

impl MemoryOperations for HybridMemoryBank {
    fn store(&mut self, key: MemoryKey, value: MemoryValue) -> crate::Result<()> {
        // Use tokio runtime for async operation
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            // We're already in an async runtime, use block_in_place
            tokio::task::block_in_place(|| {
                handle.block_on(self.store_dual(key, value))
            })
        } else {
            // Create a new runtime
            let runtime = tokio::runtime::Runtime::new()
                .map_err(|e| crate::MemoryFrameworkError { message: e.to_string() })?;
            runtime.block_on(self.store_dual(key, value))
        }
        .map_err(|e| crate::MemoryFrameworkError { message: e.to_string() })?;
        Ok(())
    }
    
    fn retrieve(&mut self, key: &MemoryKey) -> crate::Result<Option<MemoryValue>> {
        // First try KV store
        self.kv_bank.write().retrieve(key)
    }
    
    fn search(&self, query_embedding: &Array2<f32>, k: usize) -> Vec<(MemoryKey, MemoryValue, f32)> {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            tokio::task::block_in_place(|| {
                handle.block_on(self.search_enhanced(query_embedding, k))
            })
        } else {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(self.search_enhanced(query_embedding, k))
        }
    }
    
    fn update(&mut self, key: &MemoryKey, update_fn: impl FnOnce(&mut MemoryValue)) -> crate::Result<()> {
        self.kv_bank.write().update(key, update_fn)
    }
    
    fn delete(&mut self, key: &MemoryKey) -> crate::Result<bool> {
        // Delete from KV
        let deleted = self.kv_bank.write().delete(key)?;
        
        // Also delete from graph if present
        if self.config.graph_enabled {
            if let Some(node_id) = self.graph.key_to_node.get(&key.id) {
                let _ = self.graph.remove_node(&node_id);
                let _ = self.indices.remove_node(&node_id);
            }
        }
        
        Ok(deleted)
    }
    
    fn clear(&mut self) {
        self.kv_bank.write().clear();
        // TODO: Clear graph as well
    }
    
    fn size(&self) -> usize {
        self.kv_bank.read().size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_hybrid_memory_bank() {
        let kv_bank = MemoryBank::new(1000, 100);
        let mut hybrid = HybridMemoryBank::new(
            kv_bank,
            std::path::PathBuf::from("test_hybrid.bin"),
            HybridMemoryConfig::default(),
        ).unwrap();
        
        // Test dual storage
        let key = MemoryKey {
            id: "test_key".to_string(),
            timestamp: 0,
            context_hash: 0,
        };
        
        let value = MemoryValue {
            content: "test content".to_string(),
            embedding: vec![0.1; 768],
            metadata: MemoryMetadata {
                importance: 1.0,
                access_count: 0,
                last_accessed: 0,
                decay_factor: 0.9,
                tags: vec![],
            },
        };
        
        let result = hybrid.store_dual(key, value).await.unwrap();
        assert!(!result.is_empty());
    }
    
    #[test]
    fn test_clear_and_update() {
        use crate::memory::MemoryOperations;
        
        let kv_bank = MemoryBank::new(1000, 100);
        let mut hybrid = HybridMemoryBank::new(
            kv_bank,
            std::path::PathBuf::from("test_clear_update.bin"),
            HybridMemoryConfig::default(),
        ).unwrap();
        
        // Store some data
        let key1 = MemoryKey {
            id: "key1".to_string(),
            timestamp: 1,
            context_hash: 0,
        };
        
        let value1 = MemoryValue {
            content: "content1".to_string(),
            embedding: vec![0.1; 768],
            metadata: MemoryMetadata {
                importance: 1.0,
                access_count: 0,
                last_accessed: 0,
                decay_factor: 0.9,
                tags: vec!["test".to_string()],
            },
        };
        
        hybrid.store(key1.clone(), value1.clone()).unwrap();
        assert_eq!(hybrid.size(), 1);
        
        // Test update
        hybrid.update(&key1, |v| {
            v.content = "updated content".to_string();
            v.metadata.tags.push("updated".to_string());
        }).unwrap();
        
        let retrieved = hybrid.retrieve(&key1).unwrap().unwrap();
        assert_eq!(retrieved.content, "updated content");
        assert!(retrieved.metadata.tags.contains(&"updated".to_string()));
        
        // Test clear
        hybrid.clear();
        assert_eq!(hybrid.size(), 0);
    }
    
    #[test]
    fn test_config_helpers() {
        let kv_bank = MemoryBank::new(1000, 100);
        let mut hybrid = HybridMemoryBank::new(
            kv_bank,
            std::path::PathBuf::from("test_config.bin"),
            HybridMemoryConfig::default(),
        ).unwrap();
        
        // Test initial state
        assert!(hybrid.is_graph_enabled());
        
        // Test disable
        hybrid.disable_graph_features();
        assert!(!hybrid.is_graph_enabled());
        assert!(!hybrid.config.auto_infer_relationships);
        
        // Test enable
        hybrid.enable_graph_features();
        assert!(hybrid.is_graph_enabled());
        assert!(hybrid.config.auto_infer_relationships);
        
        // Test config update
        let mut new_config = hybrid.get_config().clone();
        new_config.migration_batch_size = 200;
        hybrid.update_config(new_config);
        assert_eq!(hybrid.get_config().migration_batch_size, 200);
    }
}