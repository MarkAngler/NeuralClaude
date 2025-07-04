//! Configuration for persistent storage

use serde::{Serialize, Deserialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Base directory for storage files
    pub base_path: PathBuf,
    
    /// Whether to automatically save on each write
    pub auto_save: bool,
    
    /// Interval for periodic saves (in seconds)
    pub save_interval_secs: u64,
    
    /// Maximum number of backups to keep
    pub max_backups: usize,
    
    /// Whether to compress storage files
    pub compress: bool,
    
    /// Whether to enable write-ahead logging for crash recovery
    pub enable_wal: bool,
    
    /// Maximum file size before rotation (in MB)
    pub max_file_size_mb: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("./neural_memory_storage"),
            auto_save: false,
            save_interval_secs: 300, // 5 minutes
            max_backups: 5,
            compress: false,
            enable_wal: true,
            max_file_size_mb: 100,
        }
    }
}

impl StorageConfig {
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
            ..Default::default()
        }
    }
    
    pub fn with_auto_save(mut self, enabled: bool) -> Self {
        self.auto_save = enabled;
        self
    }
    
    pub fn with_save_interval(mut self, seconds: u64) -> Self {
        self.save_interval_secs = seconds;
        self
    }
    
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compress = enabled;
        self
    }
    
    pub fn with_wal(mut self, enabled: bool) -> Self {
        self.enable_wal = enabled;
        self
    }
}