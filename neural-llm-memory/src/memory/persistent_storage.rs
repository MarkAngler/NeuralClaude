//! Persistent storage backend that wraps JsonStorage

use super::{StorageBackend, MemoryKey, MemoryValue};
use crate::storage::{JsonStorage, StorageConfig, PersistentStorage as PersistentStorageTrait};
use parking_lot::{RwLock, Mutex};
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::thread;

/// A storage backend that combines in-memory caching with persistent JSON storage
pub struct PersistentStorageBackend {
    /// In-memory cache for fast access
    cache: Arc<RwLock<HashMap<MemoryKey, MemoryValue>>>,
    
    /// Persistent storage implementation
    persistent: Arc<Mutex<JsonStorage>>,
    
    /// Configuration
    config: StorageConfig,
    
    /// Track modifications for periodic saves
    dirty: Arc<RwLock<bool>>,
    
    /// Last save time
    last_save: Arc<RwLock<Instant>>,
    
    /// Background save thread handle
    save_thread: Option<thread::JoinHandle<()>>,
    
    /// Flag to stop background thread
    stop_flag: Arc<RwLock<bool>>,
}

impl PersistentStorageBackend {
    pub fn new(config: StorageConfig) -> std::io::Result<Self> {
        let mut json_storage = JsonStorage::new(config.clone());
        json_storage.init()?;
        
        // Load existing data into cache
        let existing_data = json_storage.load_all()?;
        let cache: HashMap<MemoryKey, MemoryValue> = existing_data.into_iter().collect();
        
        let backend = Self {
            cache: Arc::new(RwLock::new(cache)),
            persistent: Arc::new(Mutex::new(json_storage)),
            config,
            dirty: Arc::new(RwLock::new(false)),
            last_save: Arc::new(RwLock::new(Instant::now())),
            save_thread: None,
            stop_flag: Arc::new(RwLock::new(false)),
        };
        
        Ok(backend)
    }
    
    /// Start background save thread
    pub fn start_background_save(&mut self) {
        if self.save_thread.is_some() {
            return;
        }
        
        let cache = self.cache.clone();
        let persistent = self.persistent.clone();
        let dirty = self.dirty.clone();
        let last_save = self.last_save.clone();
        let stop_flag = self.stop_flag.clone();
        let save_interval = Duration::from_secs(self.config.save_interval_secs);
        
        let handle = thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(1));
                
                if *stop_flag.read() {
                    break;
                }
                
                let should_save = {
                    let is_dirty = *dirty.read();
                    let time_elapsed = last_save.read().elapsed() >= save_interval;
                    is_dirty && time_elapsed
                };
                
                if should_save {
                    // Perform save
                    let memories: Vec<(MemoryKey, MemoryValue)> = {
                        cache.read().iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect()
                    };
                    
                    if let Some(storage) = persistent.try_lock() {
                        if let Err(e) = storage.save_all(&memories) {
                            eprintln!("Background save failed: {}", e);
                        } else {
                            *dirty.write() = false;
                            *last_save.write() = Instant::now();
                        }
                    }
                }
            }
        });
        
        self.save_thread = Some(handle);
    }
    
    /// Stop background save thread
    pub fn stop_background_save(&mut self) {
        *self.stop_flag.write() = true;
        
        if let Some(handle) = self.save_thread.take() {
            let _ = handle.join();
        }
    }
    
    /// Force a save to disk
    pub fn force_save(&self) -> std::io::Result<()> {
        let memories: Vec<(MemoryKey, MemoryValue)> = {
            self.cache.read().iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };
        
        self.persistent.lock().save_all(&memories)?;
        *self.dirty.write() = false;
        *self.last_save.write() = Instant::now();
        
        Ok(())
    }
    
    /// Create a backup
    pub fn backup(&self) -> std::io::Result<String> {
        self.persistent.lock().backup()
    }
    
    /// Restore from backup
    pub fn restore(&mut self, backup_path: &std::path::Path) -> std::io::Result<()> {
        self.persistent.lock().restore(backup_path)?;
        
        // Reload cache from restored data
        let restored_data = self.persistent.lock().load_all()?;
        let mut cache = self.cache.write();
        cache.clear();
        cache.extend(restored_data);
        
        *self.dirty.write() = false;
        
        Ok(())
    }
}

impl StorageBackend for PersistentStorageBackend {
    fn save(&mut self, key: &MemoryKey, value: &MemoryValue) -> std::io::Result<()> {
        self.cache.write().insert(key.clone(), value.clone());
        *self.dirty.write() = true;
        
        if self.config.auto_save {
            self.persistent.lock().save_one(key, value)?;
        }
        
        Ok(())
    }
    
    fn load(&self, key: &MemoryKey) -> std::io::Result<Option<MemoryValue>> {
        Ok(self.cache.read().get(key).cloned())
    }
    
    fn delete(&mut self, key: &MemoryKey) -> std::io::Result<bool> {
        let removed = self.cache.write().remove(key).is_some();
        
        if removed {
            *self.dirty.write() = true;
            
            if self.config.auto_save {
                self.persistent.lock().delete_one(key)?;
            }
        }
        
        Ok(removed)
    }
    
    fn list_keys(&self) -> Vec<MemoryKey> {
        self.cache.read().keys().cloned().collect()
    }
}

impl Drop for PersistentStorageBackend {
    fn drop(&mut self) {
        // Stop background thread
        self.stop_background_save();
        
        // Save any pending changes
        if *self.dirty.read() {
            let _ = self.force_save();
        }
    }
}