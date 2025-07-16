//! Memory module for LLM memory framework

pub mod memory_bank;
pub mod memory_module;
pub mod retrieval;
pub mod storage;
pub mod persistent_memory_module;
pub mod persistent_memory_module_ext;
pub mod key_value_store;
pub mod episodic_memory;
pub mod hybrid_memory_bank;

pub use memory_bank::{MemoryBank, MemoryEntry};
pub use memory_module::{MemoryModule, MemoryConfig};
pub use retrieval::{RetrievalStrategy, SimilarityMetric};
pub use storage::{StorageBackend, InMemoryStorage};
pub use persistent_memory_module::{PersistentMemoryModule, PersistentMemoryBuilder, PersistentConfig};
pub use episodic_memory::{Episode, EpisodicBank, EpisodicStats};
pub use hybrid_memory_bank::{HybridMemoryBank, HybridMemoryConfig, ConsciousnessContext, EmotionalContext};

use ndarray::Array2;
use serde::{Serialize, Deserialize};

/// Memory key for indexing memories
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryKey {
    pub id: String,
    pub timestamp: u64,
    pub context_hash: u64,
}

impl MemoryKey {
    pub fn new(id: String, context: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        context.hash(&mut hasher);
        
        Self {
            id,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            context_hash: hasher.finish(),
        }
    }
    
    /// Create a MemoryKey with just an ID (context hash will be based on ID)
    pub fn from_id(id: &str) -> Self {
        Self::new(id.to_string(), id)
    }
}

/// Memory value containing embeddings and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryValue {
    pub embedding: Vec<f32>,
    pub content: String,
    pub metadata: MemoryMetadata,
}

/// Metadata associated with memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub importance: f32,
    pub access_count: u32,
    pub last_accessed: u64,
    pub decay_factor: f32,
    pub tags: Vec<String>,
}

impl Default for MemoryMetadata {
    fn default() -> Self {
        Self {
            importance: 1.0,
            access_count: 0,
            last_accessed: 0,
            decay_factor: 0.95,
            tags: Vec::new(),
        }
    }
}

/// Interface for memory operations
pub trait MemoryOperations {
    fn store(&mut self, key: MemoryKey, value: MemoryValue) -> crate::Result<()>;
    fn retrieve(&mut self, key: &MemoryKey) -> crate::Result<Option<MemoryValue>>;
    fn search(&self, query_embedding: &Array2<f32>, k: usize) -> Vec<(MemoryKey, MemoryValue, f32)>;
    fn update(&mut self, key: &MemoryKey, update_fn: impl FnOnce(&mut MemoryValue)) -> crate::Result<()>;
    fn delete(&mut self, key: &MemoryKey) -> crate::Result<bool>;
    fn clear(&mut self);
    fn size(&self) -> usize;
}