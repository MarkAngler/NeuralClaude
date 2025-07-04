//! File-based storage backend for memories

use super::{StorageBackend, StorageStats};
use crate::memory::{MemoryKey, MemoryValue};
use crate::{Result, MemoryFrameworkError};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use tokio::fs;
use tokio::sync::RwLock;
use std::sync::Arc;

/// Memory storage format for serialization
#[derive(Serialize, Deserialize)]
struct StoredMemory {
    key: MemoryKey,
    value: MemoryValue,
}

/// File-based storage implementation
pub struct FileStorage {
    base_path: PathBuf,
    memories_file: PathBuf,
    /// In-memory index for faster lookups
    index: Arc<RwLock<HashMap<String, MemoryKey>>>,
    /// Track last save time
    last_save: Arc<RwLock<Option<u64>>>,
    /// Track last load time
    last_load: Arc<RwLock<Option<u64>>>,
}

impl FileStorage {
    /// Create a new file storage backend
    pub async fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        
        // Create directory if it doesn't exist
        fs::create_dir_all(&base_path).await
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to create storage directory: {}", e),
            })?;
        
        let memories_file = base_path.join("memories.json");
        
        let storage = Self {
            base_path,
            memories_file,
            index: Arc::new(RwLock::new(HashMap::new())),
            last_save: Arc::new(RwLock::new(None)),
            last_load: Arc::new(RwLock::new(None)),
        };
        
        // Load existing memories
        storage.load_index().await?;
        
        Ok(storage)
    }
    
    /// Load the memory index from disk
    async fn load_index(&self) -> Result<()> {
        if !self.memories_file.exists() {
            return Ok(());
        }
        
        let contents = fs::read_to_string(&self.memories_file).await
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to read memories file: {}", e),
            })?;
        
        let memories: Vec<StoredMemory> = serde_json::from_str(&contents)
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to parse memories file: {}", e),
            })?;
        
        let mut index = self.index.write().await;
        index.clear();
        
        for memory in memories {
            index.insert(memory.key.id.clone(), memory.key);
        }
        
        *self.last_load.write().await = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
        
        Ok(())
    }
    
    /// Get current timestamp
    fn timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

#[async_trait]
impl StorageBackend for FileStorage {
    async fn save(&self, key: &MemoryKey, value: &MemoryValue) -> Result<()> {
        // Load current memories
        let mut memories = self.load_all().await?;
        
        // Update or insert the memory
        let exists = memories.iter().position(|(k, _)| k.id == key.id);
        if let Some(pos) = exists {
            memories[pos] = (key.clone(), value.clone());
        } else {
            memories.push((key.clone(), value.clone()));
        }
        
        // Save all memories
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
        
        let contents = fs::read_to_string(&self.memories_file).await
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to read memories file: {}", e),
            })?;
        
        let memories: Vec<StoredMemory> = serde_json::from_str(&contents)
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to parse memories file: {}", e),
            })?;
        
        Ok(memories
            .into_iter()
            .map(|m| (m.key, m.value))
            .collect())
    }
    
    async fn save_batch(&self, memories: &[(MemoryKey, MemoryValue)]) -> Result<()> {
        let stored_memories: Vec<StoredMemory> = memories
            .iter()
            .map(|(k, v)| StoredMemory {
                key: k.clone(),
                value: v.clone(),
            })
            .collect();
        
        let json = serde_json::to_string_pretty(&stored_memories)
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to serialize memories: {}", e),
            })?;
        
        fs::write(&self.memories_file, json).await
            .map_err(|e| MemoryFrameworkError {
                message: format!("Failed to write memories file: {}", e),
            })?;
        
        // Update index
        let mut index = self.index.write().await;
        index.clear();
        for (key, _) in memories {
            index.insert(key.id.clone(), key.clone());
        }
        
        *self.last_save.write().await = Some(Self::timestamp());
        
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