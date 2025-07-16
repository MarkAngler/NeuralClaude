//! Graph storage backend using petgraph

use crate::graph::core::{ConsciousNode, ConsciousEdge, NodeType, EdgeType, NodeId, EdgeId};
use anyhow::{Result, anyhow};
use dashmap::DashMap;
use parking_lot::RwLock;
use petgraph::graph::{Graph, NodeIndex, EdgeIndex};
use petgraph::Directed;
use serde::{Serialize, Deserialize};
use serde_json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::io::{BufWriter, BufReader, Write, BufRead};
use std::fs::{File, OpenOptions};
use tokio::fs as async_fs;

/// Main graph storage structure
pub struct GraphStorage {
    /// Core graph structure
    pub graph: Arc<RwLock<Graph<ConsciousNode, ConsciousEdge, Directed>>>,
    
    /// Fast lookups
    pub node_index: Arc<DashMap<NodeId, NodeIndex>>,
    pub edge_index: Arc<DashMap<EdgeId, EdgeIndex>>,
    pub key_to_node: Arc<DashMap<String, NodeId>>,
    
    /// Persistence
    storage_path: PathBuf,
    wal_log: Arc<RwLock<WalLog>>,
}

/// Write-ahead log for persistence
pub struct WalLog {
    entries: Vec<WalOp>,
    file_path: PathBuf,
    writer: Option<BufWriter<File>>,
    checkpoint_threshold: usize,
    current_size: usize,
}

/// Operations for write-ahead log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOp {
    AddNode { id: NodeId, node: ConsciousNode },
    AddEdge { edge: ConsciousEdge },
    RemoveNode { id: NodeId },
    RemoveEdge { id: EdgeId },
    UpdateNode { id: NodeId, node: ConsciousNode },
}

/// Snapshot of the graph for persistence
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub version: u32,
    pub timestamp: i64,
    pub nodes: Vec<ConsciousNode>,
    pub edges: Vec<ConsciousEdge>,
    pub indices: IndexSnapshot,
}

/// Snapshot of indices
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexSnapshot {
    pub node_to_index: HashMap<NodeId, usize>,
    pub edge_to_index: HashMap<EdgeId, usize>,
    pub key_to_node: HashMap<String, NodeId>,
}

impl WalLog {
    pub fn new(file_path: PathBuf) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Open file in append mode
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)?;
        
        let writer = BufWriter::new(file);
        
        Ok(Self {
            entries: Vec::new(),
            file_path,
            writer: Some(writer),
            checkpoint_threshold: 1000, // Checkpoint after 1000 operations
            current_size: 0,
        })
    }
    
    /// Load existing WAL entries from disk
    pub fn load_from_disk(&mut self) -> Result<Vec<WalOp>> {
        if !self.file_path.exists() {
            return Ok(Vec::new());
        }
        
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);
        let mut operations = Vec::new();
        
        // Read each line as a serialized WalOp
        use std::io::BufRead;
        for line in reader.lines() {
            let line = line?;
            if !line.is_empty() {
                if let Ok(op) = serde_json::from_str::<WalOp>(&line) {
                    operations.push(op);
                }
            }
        }
        
        self.current_size = operations.len();
        Ok(operations)
    }
    
    pub fn log_operation(&mut self, op: WalOp) -> Result<()> {
        // Add to in-memory buffer
        self.entries.push(op.clone());
        
        // Write to disk
        if let Some(writer) = &mut self.writer {
            let serialized = serde_json::to_string(&op)?;
            writeln!(writer, "{}", serialized)?;
            writer.flush()?;
        }
        
        self.current_size += 1;
        
        // Check if we need to trigger a checkpoint
        if self.current_size >= self.checkpoint_threshold {
            // Note: Checkpoint should be triggered by the parent storage
            // We just track the size here
        }
        
        Ok(())
    }
    
    /// Clear the log and reset the file
    fn clear(&mut self) -> Result<()> {
        self.entries.clear();
        self.current_size = 0;
        
        // Close current writer
        self.writer = None;
        
        // Truncate the file
        if self.file_path.exists() {
            std::fs::remove_file(&self.file_path)?;
        }
        
        // Reopen for new writes
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)?;
        
        self.writer = Some(BufWriter::new(file));
        
        Ok(())
    }
    
    /// Sync all pending writes to disk
    pub fn sync(&mut self) -> Result<()> {
        if let Some(writer) = &mut self.writer {
            writer.flush()?;
            writer.get_mut().sync_all()?;
        }
        Ok(())
    }
    
    /// Check if checkpoint is needed
    pub fn needs_checkpoint(&self) -> bool {
        self.current_size >= self.checkpoint_threshold
    }
    
    /// Get current WAL size
    pub fn size(&self) -> usize {
        self.current_size
    }
}

impl Drop for WalLog {
    fn drop(&mut self) {
        // Ensure all pending writes are flushed to disk
        if let Err(e) = self.sync() {
            eprintln!("Error syncing WAL on drop: {}", e);
        }
    }
}

impl GraphStorage {
    /// Create a new graph storage
    pub fn new(storage_path: PathBuf) -> Result<Self> {
        let wal_path = storage_path.with_extension("wal");
        
        let mut storage = Self {
            graph: Arc::new(RwLock::new(Graph::new())),
            node_index: Arc::new(DashMap::new()),
            edge_index: Arc::new(DashMap::new()),
            key_to_node: Arc::new(DashMap::new()),
            storage_path,
            wal_log: Arc::new(RwLock::new(WalLog::new(wal_path)?)),
        };
        
        // Recover from WAL if it exists
        storage.recover_from_wal()?;
        
        Ok(storage)
    }
    
    /// Recover operations from WAL
    fn recover_from_wal(&mut self) -> Result<()> {
        let operations = self.wal_log.write().load_from_disk()?;
        
        if operations.is_empty() {
            return Ok(());
        }
        
        println!("Recovering {} operations from WAL", operations.len());
        
        // Replay operations
        for op in operations {
            match op {
                WalOp::AddNode { id, node } => {
                    self.replay_add_node(id, node)?;
                }
                WalOp::AddEdge { edge } => {
                    self.replay_add_edge(edge)?;
                }
                WalOp::RemoveNode { id } => {
                    self.replay_remove_node(&id)?;
                }
                WalOp::RemoveEdge { id } => {
                    self.replay_remove_edge(&id)?;
                }
                WalOp::UpdateNode { id, node } => {
                    self.replay_update_node(&id, node)?;
                }
            }
        }
        
        println!("WAL recovery complete");
        Ok(())
    }
    
    // Replay methods that don't log to WAL (to avoid double logging)
    fn replay_add_node(&self, node_id: NodeId, node: ConsciousNode) -> Result<()> {
        let mut graph = self.graph.write();
        let idx = graph.add_node(node.clone());
        drop(graph);
        
        self.node_index.insert(node_id.clone(), idx);
        self.key_to_node.insert(node.key.clone(), node_id);
        
        Ok(())
    }
    
    fn replay_add_edge(&self, edge: ConsciousEdge) -> Result<()> {
        if let (Some(source_idx), Some(target_idx)) = (
            self.node_index.get(&edge.source).map(|v| *v),
            self.node_index.get(&edge.target).map(|v| *v)
        ) {
            let mut graph = self.graph.write();
            let edge_idx = graph.add_edge(source_idx, target_idx, edge.clone());
            self.edge_index.insert(edge.id_string(), edge_idx);
        }
        Ok(())
    }
    
    fn replay_remove_node(&self, node_id: &NodeId) -> Result<()> {
        if let Some(idx) = self.node_index.get(node_id).map(|v| *v) {
            self.graph.write().remove_node(idx);
            self.node_index.remove(node_id);
        }
        Ok(())
    }
    
    fn replay_remove_edge(&self, edge_id: &EdgeId) -> Result<()> {
        if let Some(idx) = self.edge_index.get(edge_id).map(|v| *v) {
            self.graph.write().remove_edge(idx);
            self.edge_index.remove(edge_id);
        }
        Ok(())
    }
    
    fn replay_update_node(&self, node_id: &NodeId, updated_node: ConsciousNode) -> Result<()> {
        if let Some(idx) = self.node_index.get(node_id).map(|v| *v) {
            let mut graph = self.graph.write();
            if let Some(node) = graph.node_weight_mut(idx) {
                *node = updated_node.clone();
                
                // Update key mapping if changed
                if let Some(old_key) = self.key_to_node.iter()
                    .find(|entry| entry.value() == node_id)
                    .map(|entry| entry.key().clone()) {
                    if old_key != updated_node.key {
                        self.key_to_node.remove(&old_key);
                        self.key_to_node.insert(updated_node.key.clone(), node_id.clone());
                    }
                }
            }
        }
        Ok(())
    }
    
    /// Load from a snapshot file
    pub async fn load_from_snapshot(storage_path: PathBuf) -> Result<Self> {
        // Try to load the snapshot first
        let snapshot = if storage_path.exists() {
            let data = tokio::fs::read(&storage_path).await?;
            let decompressed = zstd::decode_all(&data[..])?;
            Some(bincode::deserialize::<GraphSnapshot>(&decompressed)?)
        } else {
            None
        };
        
        let mut storage = Self::new(storage_path)?;
        
        // If we have a snapshot, restore from it
        if let Some(snapshot) = snapshot {
            storage.restore_from_snapshot(snapshot)?;
        }
        
        Ok(storage)
    }
    
    /// Add a node to the graph
    pub fn add_node(&self, node: ConsciousNode) -> Result<NodeId> {
        let node_id = node.id_string();
        
        // Add to graph
        let mut graph = self.graph.write();
        let idx = graph.add_node(node.clone());
        drop(graph);
        
        // Update indices
        self.node_index.insert(node_id.clone(), idx);
        self.key_to_node.insert(node.key.clone(), node_id.clone());
        
        // Log operation
        let op = WalOp::AddNode { id: node_id.clone(), node: node };
        {
            let mut wal = self.wal_log.write();
            wal.log_operation(op)?;
            
            // Check if we need to checkpoint
            if wal.needs_checkpoint() {
                drop(wal); // Release lock before checkpointing
                self.trigger_checkpoint()?;
            }
        }
        
        Ok(node_id)
    }
    
    /// Add an edge to the graph
    pub fn add_edge(&self, edge: ConsciousEdge) -> Result<EdgeId> {
        let source_idx = self.node_index.get(&edge.source)
            .ok_or_else(|| anyhow!("Source node {} not found", edge.source))?
            .clone();
        let target_idx = self.node_index.get(&edge.target)
            .ok_or_else(|| anyhow!("Target node {} not found", edge.target))?
            .clone();
        
        let edge_id = edge.id_string();
        
        // Add to graph
        let mut graph = self.graph.write();
        let edge_idx = graph.add_edge(source_idx, target_idx, edge.clone());
        
        // Update index
        self.edge_index.insert(edge_id.clone(), edge_idx);
        
        // Log to WAL
        {
            let mut wal = self.wal_log.write();
            wal.log_operation(WalOp::AddEdge { edge })?;
            
            // Check if we need to checkpoint
            if wal.needs_checkpoint() {
                drop(wal);
                self.trigger_checkpoint()?;
            }
        }
        
        Ok(edge_id)
    }
    
    /// Get a node by ID
    pub fn get_node(&self, node_id: &NodeId) -> Result<ConsciousNode> {
        let idx = self.node_index.get(node_id)
            .ok_or_else(|| anyhow!("Node {} not found", node_id))?;
        
        let graph = self.graph.read();
        graph.node_weight(*idx)
            .cloned()
            .ok_or_else(|| anyhow!("Node weight not found"))
    }
    
    /// Get an edge by ID
    pub fn get_edge(&self, edge_id: &EdgeId) -> Result<ConsciousEdge> {
        let idx = self.edge_index.get(edge_id)
            .ok_or_else(|| anyhow!("Edge {} not found", edge_id))?;
        
        let graph = self.graph.read();
        graph.edge_weight(*idx)
            .cloned()
            .ok_or_else(|| anyhow!("Edge weight not found"))
    }
    
    /// Get all edges for a node
    pub fn get_edges_for_node(&self, node_id: &NodeId) -> Result<Vec<ConsciousEdge>> {
        let idx = self.node_index.get(node_id)
            .ok_or_else(|| anyhow!("Node {} not found", node_id))?;
        
        let graph = self.graph.read();
        let mut edges = Vec::new();
        
        // Outgoing edges
        for edge_ref in graph.edges(*idx) {
            edges.push(edge_ref.weight().clone());
        }
        
        Ok(edges)
    }
    
    /// Remove a node and its edges
    pub fn remove_node(&self, node_id: &NodeId) -> Result<()> {
        let idx = self.node_index.get(node_id)
            .ok_or_else(|| anyhow!("Node {} not found", node_id))?
            .clone();
        
        // Remove from graph
        let mut graph = self.graph.write();
        graph.remove_node(idx);
        
        // Update indices
        self.node_index.remove(node_id);
        // Note: key_to_node cleanup would need the node data
        
        // Log to WAL
        {
            let mut wal = self.wal_log.write();
            wal.log_operation(WalOp::RemoveNode { 
                id: node_id.clone() 
            })?;
            
            // Check if we need to checkpoint
            if wal.needs_checkpoint() {
                drop(wal);
                self.trigger_checkpoint()?;
            }
        }
        
        Ok(())
    }
    
    /// Trigger a checkpoint by saving a snapshot and clearing WAL
    fn trigger_checkpoint(&self) -> Result<()> {
        // Use blocking I/O since we're in a sync context
        let rt = tokio::runtime::Handle::try_current();
        if let Ok(handle) = rt {
            handle.block_on(self.save_snapshot())?;
        } else {
            // Fallback to synchronous save
            self.save_snapshot_sync()?;
        }
        Ok(())
    }
    
    /// Synchronous version of save_snapshot for when async runtime isn't available
    fn save_snapshot_sync(&self) -> Result<()> {
        let graph = self.graph.read();
        
        // Collect nodes
        let nodes: Vec<ConsciousNode> = graph.node_weights()
            .cloned()
            .collect();
        
        // Collect edges
        let edges: Vec<ConsciousEdge> = graph.edge_weights()
            .cloned()
            .collect();
        
        // Create index snapshot
        let indices = self.create_index_snapshot();
        
        let snapshot = GraphSnapshot {
            version: 1,
            timestamp: Utc::now().timestamp(),
            nodes,
            edges,
            indices,
        };
        
        // Serialize and compress
        let encoded = bincode::serialize(&snapshot)?;
        let compressed = zstd::encode_all(&encoded[..], 3)?;
        
        // Write to file
        std::fs::write(&self.storage_path, compressed)?;
        
        // Clear WAL after successful snapshot
        self.wal_log.write().clear()?;
        
        Ok(())
    }
    
    /// Save a snapshot of the graph
    pub async fn save_snapshot(&self) -> Result<()> {
        let graph = self.graph.read();
        
        // Collect nodes
        let nodes: Vec<ConsciousNode> = graph.node_weights()
            .cloned()
            .collect();
        
        // Collect edges
        let edges: Vec<ConsciousEdge> = graph.edge_weights()
            .cloned()
            .collect();
        
        // Create index snapshot
        let indices = self.create_index_snapshot();
        
        let snapshot = GraphSnapshot {
            version: 1,
            timestamp: Utc::now().timestamp(),
            nodes,
            edges,
            indices,
        };
        
        // Serialize and compress
        let encoded = bincode::serialize(&snapshot)?;
        let compressed = zstd::encode_all(&encoded[..], 3)?;
        
        // Write to file
        tokio::fs::write(&self.storage_path, compressed).await?;
        
        // Clear WAL after successful snapshot
        self.wal_log.write().clear()?;
        
        Ok(())
    }
    
    /// Create a snapshot of indices
    fn create_index_snapshot(&self) -> IndexSnapshot {
        let mut node_to_index = HashMap::new();
        let mut key_to_node = HashMap::new();
        let mut edge_to_index = HashMap::new();
        
        // Convert DashMap to HashMap for serialization
        for entry in self.node_index.iter() {
            node_to_index.insert(entry.key().clone(), entry.value().index());
        }
        
        for entry in self.key_to_node.iter() {
            key_to_node.insert(entry.key().clone(), entry.value().clone());
        }
        
        for entry in self.edge_index.iter() {
            edge_to_index.insert(entry.key().clone(), entry.value().index());
        }
        
        IndexSnapshot {
            node_to_index,
            edge_to_index,
            key_to_node,
        }
    }
    
    /// Restore from a snapshot
    fn restore_from_snapshot(&mut self, snapshot: GraphSnapshot) -> Result<()> {
        let mut graph = Graph::new();
        let mut node_idx_map = HashMap::new();
        
        // Restore nodes
        for node in snapshot.nodes {
            let node_id = node.id_string();
            let idx = graph.add_node(node.clone());
            node_idx_map.insert(node_id.clone(), idx);
            self.node_index.insert(node_id.clone(), idx);
            self.key_to_node.insert(node.key.clone(), node_id);
        }
        
        // Restore edges
        for edge in snapshot.edges {
            let source_idx = node_idx_map.get(&edge.source)
                .ok_or_else(|| anyhow!("Source node {} not found in snapshot", edge.source))?;
            let target_idx = node_idx_map.get(&edge.target)
                .ok_or_else(|| anyhow!("Target node {} not found in snapshot", edge.target))?;
            
            let edge_idx = graph.add_edge(*source_idx, *target_idx, edge.clone());
            self.edge_index.insert(edge.id_string(), edge_idx);
        }
        
        *self.graph.write() = graph;
        
        Ok(())
    }
    
    /// Get total node count
    pub fn node_count(&self) -> usize {
        self.graph.read().node_count()
    }
    
    /// Get total edge count
    pub fn edge_count(&self) -> usize {
        self.graph.read().edge_count()
    }
    
    /// Flush WAL to disk
    pub fn flush_wal(&self) -> Result<()> {
        self.wal_log.write().sync()
    }
    
    /// Get WAL size
    pub fn wal_size(&self) -> usize {
        self.wal_log.read().size()
    }
    
    /// Clear all nodes and edges from the graph
    pub fn clear(&self) -> Result<()> {
        // Clear the graph
        self.graph.write().clear();
        
        // Clear all indices
        self.node_index.clear();
        self.edge_index.clear();
        self.key_to_node.clear();
        
        // Clear WAL
        self.wal_log.write().clear()?;
        
        Ok(())
    }
    
    /// Update an existing node
    pub fn update_node(&self, node_id: &NodeId, updated_node: ConsciousNode) -> Result<()> {
        // Find the node index
        let idx = self.node_index.get(node_id)
            .ok_or_else(|| anyhow!("Node not found: {}", node_id))?
            .clone();
        
        // Update the node in the graph
        let mut graph = self.graph.write();
        *graph.node_weight_mut(idx)
            .ok_or_else(|| anyhow!("Node index not found in graph"))? = updated_node.clone();
        
        // Update key mapping if the key changed
        if let Some(old_key) = self.key_to_node.iter()
            .find(|entry| entry.value() == node_id)
            .map(|entry| entry.key().clone()) {
            if old_key != updated_node.key {
                self.key_to_node.remove(&old_key);
                self.key_to_node.insert(updated_node.key.clone(), node_id.clone());
            }
        }
        
        // Log to WAL
        {
            let mut wal = self.wal_log.write();
            wal.log_operation(WalOp::UpdateNode {
                id: node_id.clone(),
                node: updated_node,
            })?;
            
            // Check if we need to checkpoint
            if wal.needs_checkpoint() {
                drop(wal);
                self.trigger_checkpoint()?;
            }
        }
        
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_graph_storage_basic() {
        let storage = GraphStorage::new(PathBuf::from("test_graph.bin")).unwrap();
        
        // Create a test node
        let node = ConsciousNode::new(
            "test_key".to_string(),
            "test content".to_string(),
            vec![0.1; 768],
            NodeType::Memory(crate::graph::core::MemoryNode {
                id: "test_id".to_string(),
                key: "test_key".to_string(),
                value: "test_value".to_string(),
                embedding: vec![0.1; 768],
                created_at: Utc::now(),
                accessed_at: Utc::now(),
                access_count: 0,
            })
        );
        
        // Add node
        let node_id = storage.add_node(node).unwrap();
        assert_eq!(storage.node_count(), 1);
        
        // Retrieve node
        let retrieved = storage.get_node(&node_id).unwrap();
        assert_eq!(retrieved.key, "test_key");
    }
}