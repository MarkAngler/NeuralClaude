//! Storage backends for persisting memories

use crate::memory::{MemoryKey, MemoryValue};
use crate::Result;
use async_trait::async_trait;

pub mod file_storage;

pub use file_storage::FileStorage;

/// Trait for storage backends
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Save a memory to storage
    async fn save(&self, key: &MemoryKey, value: &MemoryValue) -> Result<()>;
    
    /// Load a memory from storage
    async fn load(&self, key: &MemoryKey) -> Result<Option<MemoryValue>>;
    
    /// Delete a memory from storage
    async fn delete(&self, key: &MemoryKey) -> Result<()>;
    
    /// List all stored memory keys
    async fn list_keys(&self) -> Result<Vec<MemoryKey>>;
    
    /// Load all memories from storage
    async fn load_all(&self) -> Result<Vec<(MemoryKey, MemoryValue)>>;
    
    /// Save multiple memories at once (batch operation)
    async fn save_batch(&self, memories: &[(MemoryKey, MemoryValue)]) -> Result<()>;
    
    /// Clear all stored memories
    async fn clear(&self) -> Result<()>;
    
    /// Get storage statistics
    async fn stats(&self) -> Result<StorageStats>;
}

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    pub total_memories: usize,
    pub total_size_bytes: u64,
    pub last_save: Option<u64>,
    pub last_load: Option<u64>,
}