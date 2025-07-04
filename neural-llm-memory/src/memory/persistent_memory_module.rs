//! Persistent memory module that combines memory operations with storage

use super::{MemoryModule, MemoryConfig, MemoryKey, MemoryValue};
use crate::storage::{StorageBackend, FileStorage};
use crate::Result;
use ndarray::Array2;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tokio::time::{interval, Duration};
use std::path::PathBuf;

/// Configuration for persistent memory
#[derive(Debug, Clone)]
pub struct PersistentConfig {
    /// Base memory configuration
    pub memory_config: MemoryConfig,
    /// Storage directory path
    pub storage_path: PathBuf,
    /// Auto-save interval in seconds (0 = disabled)
    pub auto_save_interval: u64,
    /// Save on every write (can impact performance)
    pub save_on_write: bool,
}

impl Default for PersistentConfig {
    fn default() -> Self {
        Self {
            memory_config: MemoryConfig::default(),
            storage_path: PathBuf::from("./neural_memory_storage"),
            auto_save_interval: 300, // 5 minutes
            save_on_write: false,
        }
    }
}

/// Persistent memory module with automatic storage
pub struct PersistentMemoryModule {
    /// Inner memory module
    inner: Arc<Mutex<MemoryModule>>,
    /// Storage backend
    storage: Arc<dyn StorageBackend>,
    /// Configuration
    config: PersistentConfig,
    /// Track if we need to save
    dirty: Arc<RwLock<bool>>,
    /// Auto-save handle
    auto_save_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl PersistentMemoryModule {
    /// Create a new persistent memory module
    pub async fn new(config: PersistentConfig) -> Result<Self> {
        // Create storage backend
        let storage = Arc::new(FileStorage::new(&config.storage_path).await?);
        
        // Create inner memory module
        let inner = Arc::new(Mutex::new(MemoryModule::new(config.memory_config.clone())));
        
        // Load existing memories from storage
        let memories = storage.load_all().await?;
        
        // Populate memory module with loaded memories
        if !memories.is_empty() {
            let mut inner_guard = inner.lock().await;
            for (_key, value) in memories {
                // Convert embedding vector back to Array2
                let embedding = Array2::from_shape_vec(
                    (1, value.embedding.len()),
                    value.embedding.clone()
                ).map_err(|e| crate::MemoryFrameworkError {
                    message: format!("Failed to restore embedding: {}", e),
                })?;
                
                // Store in memory module
                inner_guard.store_memory(value.content, embedding)?;
            }
        }
        
        let module = Self {
            inner: inner.clone(),
            storage,
            config: config.clone(),
            dirty: Arc::new(RwLock::new(false)),
            auto_save_handle: Arc::new(Mutex::new(None)),
        };
        
        // Start auto-save task if enabled
        if config.auto_save_interval > 0 {
            module.start_auto_save().await;
        }
        
        Ok(module)
    }
    
    /// Start the auto-save background task
    async fn start_auto_save(&self) {
        let interval_secs = self.config.auto_save_interval;
        let inner = self.inner.clone();
        let storage = self.storage.clone();
        let dirty = self.dirty.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(interval_secs));
            
            loop {
                interval.tick().await;
                
                // Check if we need to save
                let needs_save = {
                    let mut dirty_guard = dirty.write().await;
                    let was_dirty = *dirty_guard;
                    *dirty_guard = false;
                    was_dirty
                };
                
                if needs_save {
                    // Get all memories from the inner module
                    let memories = {
                        let module = inner.lock().await;
                        module.get_all_memories()
                    };
                    
                    // Save to storage
                    if let Err(e) = storage.save_batch(&memories).await {
                        eprintln!("Auto-save failed: {}", e);
                    }
                }
            }
        });
        
        *self.auto_save_handle.lock().await = Some(handle);
    }
    
    /// Store a memory with persistence
    pub async fn store_memory(&self, content: String, embedding: Array2<f32>) -> Result<MemoryKey> {
        // Store in inner module
        let key = {
            let mut inner = self.inner.lock().await;
            inner.store_memory(content.clone(), embedding.clone())?
        };
        
        // Mark as dirty
        *self.dirty.write().await = true;
        
        // Save immediately if configured
        if self.config.save_on_write {
            let value = MemoryValue {
                embedding: embedding.as_slice().unwrap().to_vec(),
                content,
                metadata: Default::default(),
            };
            
            self.storage.save(&key, &value).await?;
        }
        
        Ok(key)
    }
    
    /// Retrieve with attention
    pub async fn retrieve_with_attention(
        &self,
        query_embedding: &Array2<f32>,
        k: usize
    ) -> Vec<(MemoryKey, MemoryValue, Array2<f32>)> {
        let inner = self.inner.lock().await;
        inner.retrieve_with_attention(query_embedding, k)
    }
    
    /// Get memory statistics
    pub async fn get_stats(&self) -> (usize, u64, u64, f64) {
        let inner = self.inner.lock().await;
        let (size, total, hits, rate) = inner.get_stats();
        (size, total, hits, rate as f64)
    }
    
    /// Save all memories to storage
    pub async fn save_all(&self) -> Result<()> {
        let memories = {
            let inner = self.inner.lock().await;
            inner.get_all_memories()
        };
        
        self.storage.save_batch(&memories).await?;
        *self.dirty.write().await = false;
        Ok(())
    }
    
    /// Shutdown and save
    pub async fn shutdown(&self) -> Result<()> {
        // Stop auto-save task
        if let Some(handle) = self.auto_save_handle.lock().await.take() {
            handle.abort();
        }
        
        // Final save
        self.save_all().await?;
        Ok(())
    }
}

/// Builder for persistent memory module
pub struct PersistentMemoryBuilder {
    config: PersistentConfig,
}

impl PersistentMemoryBuilder {
    pub fn new() -> Self {
        Self {
            config: PersistentConfig::default(),
        }
    }
    
    pub fn storage_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.config.storage_path = path.into();
        self
    }
    
    pub fn auto_save_interval(mut self, seconds: u64) -> Self {
        self.config.auto_save_interval = seconds;
        self
    }
    
    pub fn save_on_write(mut self, enabled: bool) -> Self {
        self.config.save_on_write = enabled;
        self
    }
    
    pub fn memory_config(mut self, config: MemoryConfig) -> Self {
        self.config.memory_config = config;
        self
    }
    
    pub async fn build(self) -> Result<PersistentMemoryModule> {
        PersistentMemoryModule::new(self.config).await
    }
}