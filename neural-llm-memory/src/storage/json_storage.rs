//! JSON-based persistent storage implementation

use super::{PersistentStorage, StorageConfig};
use crate::memory::{MemoryKey, MemoryValue};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use chrono::{DateTime, Utc};

/// Container for serializing memories
#[derive(Serialize, Deserialize)]
struct MemoryContainer {
    version: String,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    memories: Vec<SerializedMemory>,
}

/// Serializable memory entry
#[derive(Serialize, Deserialize)]
struct SerializedMemory {
    key: MemoryKey,
    value: MemoryValue,
}

/// JSON-based storage backend with support for concurrent access
pub struct JsonStorage {
    config: StorageConfig,
    storage_path: PathBuf,
    wal_path: PathBuf,
    write_lock: Arc<RwLock<()>>,
}

impl JsonStorage {
    pub fn new(config: StorageConfig) -> Self {
        let storage_path = config.base_path.join("memories.json");
        let wal_path = config.base_path.join("memories.wal");
        
        Self {
            config,
            storage_path,
            wal_path,
            write_lock: Arc::new(RwLock::new(())),
        }
    }
    
    /// Write data to a file atomically
    fn atomic_write<T: Serialize>(&self, path: &Path, data: &T) -> io::Result<()> {
        let temp_path = path.with_extension("tmp");
        
        // Write to temporary file
        {
            let file = File::create(&temp_path)?;
            let writer = BufWriter::new(file);
            
            if self.config.compress {
                // TODO: Add compression support with flate2 or similar
                serde_json::to_writer_pretty(writer, data)?;
            } else {
                serde_json::to_writer_pretty(writer, data)?;
            }
        }
        
        // Atomically rename temp file to final path
        fs::rename(&temp_path, path)?;
        
        Ok(())
    }
    
    /// Append to write-ahead log
    fn append_to_wal(&self, operation: WalOperation) -> io::Result<()> {
        if !self.config.enable_wal {
            return Ok(());
        }
        
        let _guard = self.write_lock.write();
        
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.wal_path)?;
        
        let entry = WalEntry {
            timestamp: Utc::now(),
            operation,
        };
        
        writeln!(file, "{}", serde_json::to_string(&entry)?)?;
        file.sync_all()?;
        
        Ok(())
    }
    
    /// Replay WAL entries
    fn replay_wal(&self, memories: &mut HashMap<MemoryKey, MemoryValue>) -> io::Result<()> {
        if !self.wal_path.exists() {
            return Ok(());
        }
        
        let file = File::open(&self.wal_path)?;
        let reader = BufReader::new(file);
        
        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            
            let entry: WalEntry = serde_json::from_str(&line)?;
            match entry.operation {
                WalOperation::Insert { key, value } => {
                    memories.insert(key, value);
                }
                WalOperation::Delete { key } => {
                    memories.remove(&key);
                }
            }
        }
        
        Ok(())
    }
    
    /// Create a backup of the current storage
    fn create_backup_internal(&self) -> io::Result<String> {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let backup_name = format!("backup_{}.json", timestamp);
        let backup_path = self.config.base_path.join("backups").join(&backup_name);
        
        // Create backups directory if it doesn't exist
        fs::create_dir_all(backup_path.parent().unwrap())?;
        
        // Copy current storage file to backup
        if self.storage_path.exists() {
            fs::copy(&self.storage_path, &backup_path)?;
        }
        
        // Clean up old backups
        self.cleanup_old_backups()?;
        
        Ok(backup_path.to_string_lossy().to_string())
    }
    
    /// Remove old backups beyond the configured limit
    fn cleanup_old_backups(&self) -> io::Result<()> {
        let backup_dir = self.config.base_path.join("backups");
        if !backup_dir.exists() {
            return Ok(());
        }
        
        let mut backups: Vec<_> = fs::read_dir(&backup_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path().extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "json")
                    .unwrap_or(false)
            })
            .collect();
        
        // Sort by modification time (oldest first)
        backups.sort_by_key(|entry| {
            entry.metadata()
                .and_then(|m| m.modified())
                .ok()
        });
        
        // Remove old backups if we exceed the limit
        while backups.len() > self.config.max_backups {
            if let Some(old_backup) = backups.first() {
                fs::remove_file(old_backup.path())?;
                backups.remove(0);
            }
        }
        
        Ok(())
    }
}

impl PersistentStorage for JsonStorage {
    fn init(&mut self) -> io::Result<()> {
        // Create storage directory if it doesn't exist
        fs::create_dir_all(&self.config.base_path)?;
        
        // Create backups directory
        fs::create_dir_all(self.config.base_path.join("backups"))?;
        
        Ok(())
    }
    
    fn save_all(&self, memories: &[(MemoryKey, MemoryValue)]) -> io::Result<()> {
        let _guard = self.write_lock.write();
        
        let container = MemoryContainer {
            version: "1.0".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            memories: memories.iter()
                .map(|(key, value)| SerializedMemory {
                    key: key.clone(),
                    value: value.clone(),
                })
                .collect(),
        };
        
        // Create backup before saving
        if self.storage_path.exists() {
            self.create_backup_internal()?;
        }
        
        // Save to file
        self.atomic_write(&self.storage_path, &container)?;
        
        // Clear WAL after successful save
        if self.config.enable_wal && self.wal_path.exists() {
            fs::remove_file(&self.wal_path)?;
        }
        
        Ok(())
    }
    
    fn load_all(&self) -> io::Result<Vec<(MemoryKey, MemoryValue)>> {
        let mut memories: HashMap<MemoryKey, MemoryValue> = HashMap::new();
        
        // Load from main storage file if it exists
        if self.storage_path.exists() {
            let file = File::open(&self.storage_path)?;
            let reader = BufReader::new(file);
            
            let container: MemoryContainer = serde_json::from_reader(reader)?;
            
            // Convert to HashMap for WAL replay
            memories = container.memories
                .into_iter()
                .map(|m| (m.key, m.value))
                .collect();
        }
        
        // Replay WAL if it exists (even if main storage doesn't exist)
        if self.config.enable_wal {
            self.replay_wal(&mut memories)?;
        }
        
        Ok(memories.into_iter().collect())
    }
    
    fn save_one(&self, key: &MemoryKey, value: &MemoryValue) -> io::Result<()> {
        // For single saves, we use the WAL if enabled
        if self.config.enable_wal {
            self.append_to_wal(WalOperation::Insert {
                key: key.clone(),
                value: value.clone(),
            })
        } else {
            // If WAL is disabled, we need to load all, update, and save all
            let mut memories = self.load_all()?
                .into_iter()
                .collect::<HashMap<_, _>>();
            
            memories.insert(key.clone(), value.clone());
            
            let memories_vec: Vec<_> = memories.into_iter().collect();
            self.save_all(&memories_vec)
        }
    }
    
    fn delete_one(&self, key: &MemoryKey) -> io::Result<()> {
        if self.config.enable_wal {
            self.append_to_wal(WalOperation::Delete {
                key: key.clone(),
            })
        } else {
            // If WAL is disabled, we need to load all, delete, and save all
            let mut memories = self.load_all()?
                .into_iter()
                .collect::<HashMap<_, _>>();
            
            memories.remove(key);
            
            let memories_vec: Vec<_> = memories.into_iter().collect();
            self.save_all(&memories_vec)
        }
    }
    
    fn exists(&self) -> bool {
        self.storage_path.exists()
    }
    
    fn location(&self) -> &Path {
        &self.storage_path
    }
    
    fn backup(&self) -> io::Result<String> {
        self.create_backup_internal()
    }
    
    fn restore(&self, backup_path: &Path) -> io::Result<()> {
        if !backup_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "Backup file not found",
            ));
        }
        
        let _guard = self.write_lock.write();
        
        // Create a backup of current state before restore
        if self.storage_path.exists() {
            let pre_restore_backup = self.storage_path.with_extension("pre_restore");
            fs::copy(&self.storage_path, &pre_restore_backup)?;
        }
        
        // Copy backup to main storage location
        fs::copy(backup_path, &self.storage_path)?;
        
        // Clear WAL as it's no longer valid
        if self.wal_path.exists() {
            fs::remove_file(&self.wal_path)?;
        }
        
        Ok(())
    }
}

/// Write-ahead log operations
#[derive(Serialize, Deserialize)]
enum WalOperation {
    Insert { key: MemoryKey, value: MemoryValue },
    Delete { key: MemoryKey },
}

/// Write-ahead log entry
#[derive(Serialize, Deserialize)]
struct WalEntry {
    timestamp: DateTime<Utc>,
    operation: WalOperation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_json_storage_basic() {
        let temp_dir = TempDir::new().unwrap();
        let config = StorageConfig::new(temp_dir.path());
        let mut storage = JsonStorage::new(config);
        
        storage.init().unwrap();
        
        // Test save and load
        let key = MemoryKey::new("test_id".to_string(), "test context");
        let value = MemoryValue {
            embedding: vec![1.0, 2.0, 3.0],
            content: "test content".to_string(),
            metadata: Default::default(),
        };
        
        storage.save_one(&key, &value).unwrap();
        
        let loaded = storage.load_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0.id, "test_id");
        assert_eq!(loaded[0].1.content, "test content");
    }
}