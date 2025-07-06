//! Key-value store extension for persistent memory module

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::Result;

/// Key-value store for direct key-based access
#[derive(Debug, Clone)]
pub struct KeyValueStore {
    store: Arc<RwLock<HashMap<String, String>>>,
}

impl KeyValueStore {
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Store a value with a key
    pub async fn store(&self, key: String, value: String) -> Result<()> {
        let mut store = self.store.write().await;
        store.insert(key, value);
        Ok(())
    }
    
    /// Retrieve a value by key
    pub async fn retrieve(&self, key: &str) -> Result<Option<String>> {
        let store = self.store.read().await;
        Ok(store.get(key).cloned())
    }
    
    /// Remove a value by key
    pub async fn remove(&self, key: &str) -> Result<bool> {
        let mut store = self.store.write().await;
        Ok(store.remove(key).is_some())
    }
    
    /// Get all keys
    pub async fn keys(&self) -> Vec<String> {
        let store = self.store.read().await;
        store.keys().cloned().collect()
    }
    
    /// Get number of entries
    pub async fn len(&self) -> usize {
        let store = self.store.read().await;
        store.len()
    }
    
    /// Clear all entries
    pub async fn clear(&self) {
        let mut store = self.store.write().await;
        store.clear();
    }
}