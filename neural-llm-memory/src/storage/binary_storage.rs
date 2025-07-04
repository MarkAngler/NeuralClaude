//! Binary storage backend for perfect precision

use super::{StorageBackend, StorageStats};
use crate::memory::{MemoryKey, MemoryValue};
use crate::{Result, MemoryFrameworkError};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::sync::RwLock;
use std::sync::Arc;
use std::collections::HashMap;

/// Binary storage using bincode for perfect float precision
pub struct BinaryStorage {
    base_path: PathBuf,
    memories_file: PathBuf,
    index: Arc<RwLock<HashMap<String, MemoryKey>>>,
    last_save: Arc<RwLock<Option<u64>>>,
    last_load: Arc<RwLock<Option<u64>>>,
}

impl BinaryStorage {
    pub async fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        
        fs::create_dir_all(&base_path).await
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to create storage directory: {}", e),
            })?;
        
        let memories_file = base_path.join("memories.bin");
        
        let storage = Self {
            base_path,
            memories_file,
            index: Arc::new(RwLock::new(HashMap::new())),
            last_save: Arc::new(RwLock::new(None)),
            last_load: Arc::new(RwLock::new(None)),
        };
        
        storage.load_index().await?;
        Ok(storage)
    }
    
    async fn load_index(&self) -> Result<()> {
        if !self.memories_file.exists() {
            return Ok(());
        }
        
        let contents = fs::read(&self.memories_file).await
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to read memories file: {}", e),
            })?;
        
        let memories: Vec<(MemoryKey, MemoryValue)> = bincode::deserialize(&contents)
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to deserialize memories: {}", e),
            })?;
        
        let mut index = self.index.write().await;
        index.clear();
        
        for (key, _) in &memories {
            index.insert(key.id.clone(), key.clone());
        }
        
        *self.last_load.write().await = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        
        Ok(())
    }
}

#[async_trait]
impl StorageBackend for BinaryStorage {
    async fn save(&self, key: &MemoryKey, value: &MemoryValue) -> Result<()> {
        let mut memories = self.load_all().await?;
        
        let exists = memories.iter().position(|(k, _)| k.id == key.id);
        if let Some(pos) = exists {
            memories[pos] = (key.clone(), value.clone());
        } else {
            memories.push((key.clone(), value.clone()));
        }
        
        self.save_batch(&memories).await?;
        Ok(())
    }
    
    async fn load(&self, key: &MemoryKey) -> Result<Option<MemoryValue>> {
        let memories = self.load_all().await?;
        Ok(memories
            .into_iter()
            .find(|(k, _)| k.id == key.id)
            .map(|(_, v)| v))
    }
    
    async fn delete(&self, key: &MemoryKey) -> Result<()> {
        let mut memories = self.load_all().await?;
        memories.retain(|(k, _)| k.id != key.id);
        self.save_batch(&memories).await?;
        Ok(())
    }
    
    async fn list_keys(&self) -> Result<Vec<MemoryKey>> {
        let index = self.index.read().await;
        Ok(index.values().cloned().collect())
    }
    
    async fn load_all(&self) -> Result<Vec<(MemoryKey, MemoryValue)>> {
        if !self.memories_file.exists() {
            return Ok(Vec::new());
        }
        
        let contents = fs::read(&self.memories_file).await
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to read memories file: {}", e),
            })?;
        
        bincode::deserialize(&contents)
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to deserialize memories: {}", e),
            })
    }
    
    async fn save_batch(&self, memories: &[(MemoryKey, MemoryValue)]) -> Result<()> {
        let binary = bincode::serialize(memories)
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to serialize memories: {}", e),
            })?;
        
        fs::write(&self.memories_file, binary).await
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to write memories file: {}", e),
            })?;
        
        let mut index = self.index.write().await;
        index.clear();
        for (key, _) in memories {
            index.insert(key.id.clone(), key.clone());
        }
        
        *self.last_save.write().await = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        
        Ok(())
    }
    
    async fn clear(&self) -> Result<()> {
        if self.memories_file.exists() {
            fs::remove_file(&self.memories_file).await
                .map_err(|e| MemoryFrameworkError {
                    message: format!("Failed to delete memories file: {}", e),
                })?;
        }
        
        self.index.write().await.clear();
        Ok(())
    }
    
    async fn stats(&self) -> Result<StorageStats> {
        let metadata = if self.memories_file.exists() {
            fs::metadata(&self.memories_file).await
                .map_err(|e| MemoryFrameworkError {
                    message: format!("Failed to get file metadata: {}", e),
                })?
        } else {
            return Ok(StorageStats::default());
        };
        
        Ok(StorageStats {
            total_memories: self.index.read().await.len(),
            total_size_bytes: metadata.len(),
            last_save: *self.last_save.read().await,
            last_load: *self.last_load.read().await,
        })
    }
}