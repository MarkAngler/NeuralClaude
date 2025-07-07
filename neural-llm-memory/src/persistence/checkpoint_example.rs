// Example implementation of checkpoint saving for NeuralClaude
// This demonstrates how to save/load neural network weights

use std::path::{Path, PathBuf};
use std::fs;
use tokio::fs as async_fs;
use serde::{Serialize, Deserialize};
use anyhow::Result;

use crate::self_optimizing::SelfOptimizingNetwork;
use crate::persistence::{NetworkState, NetworkPersistence, PersistenceFormat};

/// Manages neural network checkpoints
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    max_checkpoints: usize,
}

impl CheckpointManager {
    pub fn new(checkpoint_dir: impl AsRef<Path>) -> Result<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();
        
        // Ensure directory exists
        fs::create_dir_all(&checkpoint_dir)?;
        
        Ok(Self {
            checkpoint_dir,
            max_checkpoints: 10,
        })
    }
    
    /// Save a checkpoint of the current network state
    pub async fn save_checkpoint(
        &self,
        network: &SelfOptimizingNetwork,
        tag: &str,
    ) -> Result<PathBuf> {
        // Extract network state
        let network_state = network.to_network_state()
            .map_err(|e| anyhow::anyhow!("Failed to extract network state: {}", e))?;
        
        // Generate checkpoint filename
        let timestamp = chrono::Utc::now().timestamp();
        let filename = format!("checkpoint_{}_{}.bin", tag, timestamp);
        let checkpoint_path = self.checkpoint_dir.join(&filename);
        
        // Save atomically (write to temp file, then rename)
        let temp_path = self.checkpoint_dir.join(format!(".{}.tmp", filename));
        
        // Save in binary format for efficiency
        NetworkPersistence::save(&temp_path, &network_state, PersistenceFormat::Binary)?;
        
        // Atomic rename
        async_fs::rename(&temp_path, &checkpoint_path).await?;
        
        // Update symlink for latest checkpoint
        let latest_link = self.checkpoint_dir.join(format!("checkpoint_{}_latest.bin", tag));
        if latest_link.exists() {
            async_fs::remove_file(&latest_link).await?;
        }
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            symlink(&filename, &latest_link)?;
        }
        
        // Cleanup old checkpoints
        self.cleanup_old_checkpoints(tag).await?;
        
        println!("✅ Saved checkpoint: {}", checkpoint_path.display());
        
        Ok(checkpoint_path)
    }
    
    /// Load the latest checkpoint
    pub async fn load_latest_checkpoint(&self, tag: &str) -> Result<NetworkState> {
        let latest_link = self.checkpoint_dir.join(format!("checkpoint_{}_latest.bin", tag));
        
        let checkpoint_path = if latest_link.exists() {
            latest_link
        } else {
            // Find most recent checkpoint file
            self.find_latest_checkpoint(tag).await?
        };
        
        println!("📂 Loading checkpoint: {}", checkpoint_path.display());
        
        let state = NetworkPersistence::load(&checkpoint_path)?;
        
        Ok(state)
    }
    
    /// Find the most recent checkpoint file
    async fn find_latest_checkpoint(&self, tag: &str) -> Result<PathBuf> {
        let mut entries = async_fs::read_dir(&self.checkpoint_dir).await?;
        let mut checkpoints = Vec::new();
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with(&format!("checkpoint_{}_", tag)) && name.ends_with(".bin") {
                    checkpoints.push(path);
                }
            }
        }
        
        checkpoints.sort_by_key(|p| {
            fs::metadata(p)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        
        checkpoints.into_iter().last()
            .ok_or_else(|| anyhow::anyhow!("No checkpoints found for tag: {}", tag))
    }
    
    /// Remove old checkpoints, keeping only the most recent N
    async fn cleanup_old_checkpoints(&self, tag: &str) -> Result<()> {
        let mut entries = async_fs::read_dir(&self.checkpoint_dir).await?;
        let mut checkpoints = Vec::new();
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with(&format!("checkpoint_{}_", tag)) 
                    && name.ends_with(".bin") 
                    && !name.contains("latest") {
                    checkpoints.push(path);
                }
            }
        }
        
        // Sort by modification time (oldest first)
        checkpoints.sort_by_key(|p| {
            fs::metadata(p)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        
        // Remove oldest checkpoints if we exceed the limit
        while checkpoints.len() > self.max_checkpoints {
            if let Some(old_checkpoint) = checkpoints.remove(0) {
                async_fs::remove_file(&old_checkpoint).await?;
                println!("🗑️  Removed old checkpoint: {}", old_checkpoint.display());
            }
        }
        
        Ok(())
    }
}

/// Integration with AdaptiveMemoryModule
impl AdaptiveMemoryModule {
    /// Save current network state
    pub async fn save_network_checkpoint(&self) -> Result<()> {
        let checkpoint_mgr = CheckpointManager::new("./adaptive_memory_data/network_checkpoints")?;
        
        // Get current evolution status
        let evolution_status = self.get_evolution_status().await?;
        let fitness = evolution_status.best_fitness;
        let generation = evolution_status.current_generation;
        
        // Save with descriptive tag
        let tag = format!("gen{}_fit{:.3}", generation, fitness);
        checkpoint_mgr.save_checkpoint(&self.network, &tag).await?;
        
        // Also save as "best" if this is the best fitness so far
        if self.is_best_fitness(fitness).await {
            checkpoint_mgr.save_checkpoint(&self.network, "best").await?;
        }
        
        Ok(())
    }
    
    /// Restore network from checkpoint
    pub async fn restore_from_checkpoint(&mut self) -> Result<bool> {
        let checkpoint_mgr = CheckpointManager::new("./adaptive_memory_data/network_checkpoints")?;
        
        // Try to load best checkpoint first
        match checkpoint_mgr.load_latest_checkpoint("best").await {
            Ok(state) => {
                self.network = SelfOptimizingNetwork::from_network_state(state)?;
                println!("✅ Restored from best checkpoint");
                return Ok(true);
            }
            Err(e) => {
                println!("⚠️  No 'best' checkpoint found: {}", e);
            }
        }
        
        // Fall back to latest checkpoint
        match checkpoint_mgr.load_latest_checkpoint("gen").await {
            Ok(state) => {
                self.network = SelfOptimizingNetwork::from_network_state(state)?;
                println!("✅ Restored from latest checkpoint");
                Ok(true)
            }
            Err(e) => {
                println!("⚠️  No checkpoints found: {}", e);
                Ok(false)
            }
        }
    }
}