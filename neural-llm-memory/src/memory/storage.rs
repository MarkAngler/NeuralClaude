//! Storage backends for memory persistence

use crate::memory::{MemoryKey, MemoryValue};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

pub trait StorageBackend: Send + Sync {
    fn save(&mut self, key: &MemoryKey, value: &MemoryValue) -> std::io::Result<()>;
    fn load(&self, key: &MemoryKey) -> std::io::Result<Option<MemoryValue>>;
    fn delete(&mut self, key: &MemoryKey) -> std::io::Result<bool>;
    fn list_keys(&self) -> Vec<MemoryKey>;
}

/// In-memory storage backend
pub struct InMemoryStorage {
    data: Arc<RwLock<HashMap<MemoryKey, MemoryValue>>>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl StorageBackend for InMemoryStorage {
    fn save(&mut self, key: &MemoryKey, value: &MemoryValue) -> std::io::Result<()> {
        self.data.write().insert(key.clone(), value.clone());
        Ok(())
    }
    
    fn load(&self, key: &MemoryKey) -> std::io::Result<Option<MemoryValue>> {
        Ok(self.data.read().get(key).cloned())
    }
    
    fn delete(&mut self, key: &MemoryKey) -> std::io::Result<bool> {
        Ok(self.data.write().remove(key).is_some())
    }
    
    fn list_keys(&self) -> Vec<MemoryKey> {
        self.data.read().keys().cloned().collect()
    }
}