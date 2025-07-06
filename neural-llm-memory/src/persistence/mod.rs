//! Persistence module for saving and loading neural networks
//! 
//! This module provides functionality to serialize and deserialize neural networks,
//! including their architecture, weights, and optimization state.

use std::path::Path;
use std::fs::{File, create_dir_all};
use std::io::{Read, Write, BufReader, BufWriter};
use serde::{Serialize, Deserialize};
use bincode;
use anyhow::{Result, Context};

pub mod network_state;
pub mod layer_state;
pub mod format;

pub use network_state::{
    NetworkState, NetworkMetadata, NetworkStateBuilder,
    TrainingStateSnapshot, EvolutionHistory, MutationRecord
};
pub use layer_state::{LayerState, LayerConfig};
pub use format::{PersistenceFormat, FormatVersion};

/// Main persistence handler for neural networks
pub struct NetworkPersistence;

impl NetworkPersistence {
    /// Save a network to a file
    pub fn save<P: AsRef<Path>>(
        path: P,
        state: &NetworkState,
        format: PersistenceFormat,
    ) -> Result<()> {
        let path = path.as_ref();
        
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            create_dir_all(parent)
                .context("Failed to create parent directory")?;
        }
        
        match format {
            PersistenceFormat::Binary => Self::save_binary(path, state),
            PersistenceFormat::Json => Self::save_json(path, state),
            PersistenceFormat::Compressed => Self::save_compressed(path, state),
        }
    }
    
    /// Load a network from a file
    pub fn load<P: AsRef<Path>>(
        path: P,
        format: PersistenceFormat,
    ) -> Result<NetworkState> {
        let path = path.as_ref();
        
        if !path.exists() {
            anyhow::bail!("Network file does not exist: {:?}", path);
        }
        
        match format {
            PersistenceFormat::Binary => Self::load_binary(path),
            PersistenceFormat::Json => Self::load_json(path),
            PersistenceFormat::Compressed => Self::load_compressed(path),
        }
    }
    
    /// Save network state as binary (fastest, smallest)
    fn save_binary(path: &Path, state: &NetworkState) -> Result<()> {
        let file = File::create(path)
            .context("Failed to create file")?;
        let writer = BufWriter::new(file);
        
        bincode::serialize_into(writer, state)
            .context("Failed to serialize network state")?;
        
        Ok(())
    }
    
    /// Load network state from binary
    fn load_binary(path: &Path) -> Result<NetworkState> {
        let file = File::open(path)
            .context("Failed to open file")?;
        let reader = BufReader::new(file);
        
        let state = bincode::deserialize_from(reader)
            .context("Failed to deserialize network state")?;
        
        Ok(state)
    }
    
    /// Save network state as JSON (human-readable, larger)
    fn save_json(path: &Path, state: &NetworkState) -> Result<()> {
        let json = serde_json::to_string_pretty(state)
            .context("Failed to serialize to JSON")?;
        
        let mut file = File::create(path)
            .context("Failed to create file")?;
        file.write_all(json.as_bytes())
            .context("Failed to write JSON")?;
        
        Ok(())
    }
    
    /// Load network state from JSON
    fn load_json(path: &Path) -> Result<NetworkState> {
        let mut file = File::open(path)
            .context("Failed to open file")?;
        let mut json = String::new();
        file.read_to_string(&mut json)
            .context("Failed to read file")?;
        
        let state = serde_json::from_str(&json)
            .context("Failed to deserialize from JSON")?;
        
        Ok(state)
    }
    
    /// Save compressed binary (smallest size)
    fn save_compressed(path: &Path, state: &NetworkState) -> Result<()> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        
        let file = File::create(path)
            .context("Failed to create file")?;
        let encoder = GzEncoder::new(file, Compression::default());
        let writer = BufWriter::new(encoder);
        
        bincode::serialize_into(writer, state)
            .context("Failed to serialize compressed state")?;
        
        Ok(())
    }
    
    /// Load compressed binary
    fn load_compressed(path: &Path) -> Result<NetworkState> {
        use flate2::read::GzDecoder;
        
        let file = File::open(path)
            .context("Failed to open file")?;
        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);
        
        let state = bincode::deserialize_from(reader)
            .context("Failed to deserialize compressed state")?;
        
        Ok(state)
    }
    
    /// Get file size in bytes
    pub fn get_file_size<P: AsRef<Path>>(path: P) -> Result<u64> {
        let metadata = std::fs::metadata(path)?;
        Ok(metadata.len())
    }
    
    /// Validate a saved network file
    pub fn validate<P: AsRef<Path>>(
        path: P,
        format: PersistenceFormat,
    ) -> Result<bool> {
        // Try to load and check basic properties
        match Self::load(&path, format) {
            Ok(state) => {
                // Basic validation checks
                if state.layers.is_empty() {
                    return Ok(false);
                }
                if state.metadata.version != FormatVersion::current() {
                    eprintln!("Warning: Version mismatch");
                }
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }
}

/// Checkpoint manager for training
pub struct CheckpointManager {
    base_path: String,
    max_checkpoints: usize,
    checkpoints: Vec<CheckpointInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub path: String,
    pub epoch: usize,
    pub loss: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl CheckpointManager {
    pub fn new(base_path: impl Into<String>, max_checkpoints: usize) -> Self {
        Self {
            base_path: base_path.into(),
            max_checkpoints,
            checkpoints: Vec::new(),
        }
    }
    
    /// Save a checkpoint
    pub fn save_checkpoint(
        &mut self,
        state: &NetworkState,
        epoch: usize,
        loss: f32,
    ) -> Result<()> {
        let timestamp = chrono::Utc::now();
        let filename = format!(
            "checkpoint_epoch_{}_loss_{:.4}_{}.bin",
            epoch,
            loss,
            timestamp.timestamp()
        );
        
        let path = Path::new(&self.base_path).join(&filename);
        
        // Save the checkpoint
        NetworkPersistence::save(&path, state, PersistenceFormat::Compressed)?;
        
        // Track checkpoint info
        self.checkpoints.push(CheckpointInfo {
            path: path.to_string_lossy().to_string(),
            epoch,
            loss,
            timestamp,
        });
        
        // Clean up old checkpoints if needed
        self.cleanup_old_checkpoints()?;
        
        Ok(())
    }
    
    /// Load the best checkpoint (lowest loss)
    pub fn load_best_checkpoint(&self) -> Result<NetworkState> {
        let best = self.checkpoints
            .iter()
            .min_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap())
            .context("No checkpoints available")?;
        
        NetworkPersistence::load(&best.path, PersistenceFormat::Compressed)
    }
    
    /// Load the latest checkpoint
    pub fn load_latest_checkpoint(&self) -> Result<NetworkState> {
        let latest = self.checkpoints
            .iter()
            .max_by_key(|c| c.timestamp)
            .context("No checkpoints available")?;
        
        NetworkPersistence::load(&latest.path, PersistenceFormat::Compressed)
    }
    
    /// Clean up old checkpoints, keeping only the best N
    fn cleanup_old_checkpoints(&mut self) -> Result<()> {
        if self.checkpoints.len() <= self.max_checkpoints {
            return Ok(());
        }
        
        // Sort by loss (best first)
        self.checkpoints.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap());
        
        // Remove worst checkpoints
        while self.checkpoints.len() > self.max_checkpoints {
            if let Some(checkpoint) = self.checkpoints.pop() {
                // Delete the file
                if let Err(e) = std::fs::remove_file(&checkpoint.path) {
                    eprintln!("Failed to remove checkpoint {}: {}", checkpoint.path, e);
                }
            }
        }
        
        Ok(())
    }
    
    /// List all checkpoints
    pub fn list_checkpoints(&self) -> &[CheckpointInfo] {
        &self.checkpoints
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::self_optimizing::{SelfOptimizingConfig, SelfOptimizingNetwork};
    use tempfile::TempDir;
    
    #[test]
    fn test_save_load_binary() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("network.bin");
        
        // Create a simple network state
        let config = SelfOptimizingConfig::default();
        let network = SelfOptimizingNetwork::new_with_io_sizes(config, 10, 5);
        
        // Convert to state (would need to implement conversion)
        // let state = NetworkState::from_network(&network);
        
        // Test would continue here once conversion is implemented
    }
}