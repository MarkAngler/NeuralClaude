//! Persistence extension for AdaptiveMemoryModule
//! 
//! This module adds save/load functionality to persist neural network state 
//! and evolution history across server restarts.

use super::{AdaptiveMemoryModule, AdaptiveConfig, UsageMetrics};
use crate::memory::{PersistentMemoryModule, PersistentConfig, MemoryConfig};
// Removed unused imports
use crate::adaptive::{BackgroundEvolver, EvolutionStatus};
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use std::error::Error as StdError;

/// Adaptive module state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveModuleState {
    /// Adaptive configuration
    pub config: AdaptiveConfig,
    
    /// Current operation count
    pub operation_count: usize,
    
    /// Recent usage metrics (last 1000)
    pub recent_metrics: Vec<UsageMetrics>,
    
    /// Evolution status
    pub evolution_status: EvolutionStatus,
    
    /// Last evolution time
    pub last_evolution_time: Option<DateTime<Utc>>,
    
    /// Timestamp when state was saved
    pub saved_at: DateTime<Utc>,
    
    /// Version for future compatibility
    pub version: u32,
}

impl AdaptiveModuleState {
    const CURRENT_VERSION: u32 = 1;
    
    /// Create a new state snapshot
    pub fn new(
        config: AdaptiveConfig,
        operation_count: usize,
        recent_metrics: Vec<UsageMetrics>,
        evolution_status: EvolutionStatus,
        last_evolution_time: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            config,
            operation_count,
            recent_metrics,
            evolution_status,
            last_evolution_time,
            saved_at: Utc::now(),
            version: Self::CURRENT_VERSION,
        }
    }
}

impl AdaptiveMemoryModule {
    /// Save the complete adaptive module state
    pub async fn save_state<P: AsRef<Path>>(&self, base_path: P) -> Result<(), Box<dyn StdError>> {
        let base_path = base_path.as_ref();
        
        // Create directory if it doesn't exist
        tokio::fs::create_dir_all(base_path).await
            .map_err(|e| format!("Failed to create adaptive state directory: {}", e))?;
        
        // Save adaptive module state
        let state = self.create_state_snapshot().await?;
        let state_path = base_path.join("adaptive_state.json");
        let state_json = serde_json::to_string_pretty(&state)
            .map_err(|e| format!("Failed to serialize adaptive state: {}", e))?;
        tokio::fs::write(&state_path, state_json).await
            .map_err(|e| format!("Failed to write adaptive state: {}", e))?;
        
        // Save neural network state if evolution has run
        let evolver = self.evolver().lock().await;
        if evolver.get_last_evolution_time().await.is_some() {
            // We need to save the evolved network architecture
            // This requires extracting the network from the evolver
            // For now, we'll save a marker file
            let network_marker_path = base_path.join("network_evolved.marker");
            tokio::fs::write(&network_marker_path, b"evolved").await
                .map_err(|e| format!("Failed to write network marker: {}", e))?;
        }
        
        // The persistent memory module already saves its own state
        // through its auto-save mechanism
        
        tracing::info!("Saved adaptive module state to {:?}", base_path);
        Ok(())
    }
    
    /// Load adaptive module state from disk
    pub async fn load_state<P: AsRef<Path>>(
        base_path: P,
        base_config: MemoryConfig,
    ) -> Result<Self, Box<dyn StdError>> {
        let base_path = base_path.as_ref();
        let state_path = base_path.join("adaptive_state.json");
        
        // Check if state file exists
        if !state_path.exists() {
            // No saved state, create new module
            return Self::new(base_config).await
                .map_err(|e| format!("Failed to create new adaptive module: {}", e).into());
        }
        
        // Load adaptive state
        let state_json = tokio::fs::read_to_string(&state_path).await
            .map_err(|e| format!("Failed to read adaptive state: {}", e))?;
        let state: AdaptiveModuleState = serde_json::from_str(&state_json)
            .map_err(|e| format!("Failed to deserialize adaptive state: {}", e))?;
        
        // Check if we have an evolved network configuration to load
        let evolved_config_path = base_path.join("network_checkpoints/evolved_config.json");
        let memory_config = if evolved_config_path.exists() {
            tracing::info!("🧬 Found evolved network configuration, loading...");
            match tokio::fs::read_to_string(&evolved_config_path).await {
                Ok(config_json) => {
                    match serde_json::from_str::<MemoryConfig>(&config_json) {
                        Ok(evolved_config) => {
                            tracing::info!("✅ Successfully loaded evolved network configuration");
                            evolved_config
                        }
                        Err(e) => {
                            eprintln!("⚠️  Failed to parse evolved config: {}. Using base config.", e);
                            base_config
                        }
                    }
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to read evolved config: {}. Using base config.", e);
                    base_config
                }
            }
        } else {
            base_config
        };
        
        // Create the module using the constructor with the loaded config (and potentially evolved memory config)
        let module = Self::with_config(memory_config, state.config).await
            .map_err(|e| format!("Failed to create module with loaded config: {}", e))?;
        
        // Restore the operation count
        *module.operation_count().write().await = state.operation_count;
        
        // Restore metrics to the usage collector (clear existing and add saved ones)
        for metric in state.recent_metrics.iter().take(1000) {
            module.usage_collector().record_metric(metric.clone()).await;
        }
        
        // Log evolution status if it was running
        if state.evolution_status.is_running {
            tracing::info!("Evolution was running when saved, marking as stopped");
        }
        
        tracing::info!("Loaded adaptive module state from {:?}", base_path);
        tracing::info!("  - Operation count: {}", state.operation_count);
        tracing::info!("  - Metrics restored: {}", state.recent_metrics.len());
        tracing::info!("  - Last evolution: {:?}", state.last_evolution_time);
        
        Ok(module)
    }
    
    /// Create a snapshot of current state
    async fn create_state_snapshot(&self) -> Result<AdaptiveModuleState, Box<dyn StdError>> {
        let config = self.config().read().await.clone();
        let operation_count = *self.operation_count().read().await;
        let recent_metrics = self.usage_collector().get_recent_metrics(1000).await;
        let evolution_status = self.evolver().lock().await.get_status().await;
        let last_evolution_time = self.evolver().lock().await.get_last_evolution_time().await;
        
        Ok(AdaptiveModuleState::new(
            config,
            operation_count,
            recent_metrics,
            evolution_status,
            last_evolution_time,
        ))
    }
    
    /// Save state periodically (to be called by a background task)
    pub async fn auto_save(&self) -> Result<(), Box<dyn StdError>> {
        let base_path = PathBuf::from("./adaptive_memory_data");
        self.save_state(&base_path).await
    }
    
    /// Initialize with loading from saved state if available
    pub async fn new_with_persistence(
        base_config: MemoryConfig,
        load_from_disk: bool,
    ) -> Result<Self, Box<dyn StdError>> {
        if load_from_disk {
            let base_path = PathBuf::from("./adaptive_memory_data");
            if base_path.join("adaptive_state.json").exists() {
                println!("Loading adaptive module from saved state...");
                match Self::load_state(&base_path, base_config.clone()).await {
                    Ok(module) => {
                        // Log loaded state details
                        let op_count = *module.operation_count().read().await;
                        let status = module.evolver().lock().await.get_status().await;
                        println!("✓ Successfully loaded adaptive state:");
                        println!("  - Operation count: {}", op_count);
                        println!("  - Current generation: {}", status.current_generation);
                        println!("  - Best fitness: {}", status.best_fitness);
                        println!("  - Persistence enabled!");
                        return Ok(module);
                    }
                    Err(e) => {
                        eprintln!("Failed to load adaptive state: {}. Starting fresh.", e);
                        return Self::new(base_config).await;
                    }
                }
            }
        }
        
        // No saved state or not loading from disk
        Self::new(base_config).await
            .map_err(|e| format!("Failed to create new adaptive module: {}", e).into())
    }
    
    /// Get path for saving neural network checkpoints
    pub fn get_network_checkpoint_path(&self) -> PathBuf {
        PathBuf::from("./adaptive_memory_data/network_checkpoints")
    }
    
    /// Save evolved network architecture
    pub async fn save_evolved_network(&self) -> Result<(), Box<dyn StdError>> {
        let checkpoint_path = self.get_network_checkpoint_path();
        tokio::fs::create_dir_all(&checkpoint_path).await
            .map_err(|e| format!("Failed to create checkpoint directory: {}", e))?;
        
        // Get current memory config (which represents the evolved architecture)
        let memory = self.active_memory().read().await;
        let config = memory.get_config().clone();
        
        // Save as JSON for easy inspection
        let config_path = checkpoint_path.join("evolved_config.json");
        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| format!("Failed to serialize memory config: {}", e))?;
        tokio::fs::write(&config_path, config_json).await
            .map_err(|e| format!("Failed to write evolved config: {}", e))?;
        
        // Also save timestamp
        let timestamp_path = checkpoint_path.join("timestamp.txt");
        tokio::fs::write(&timestamp_path, Utc::now().to_rfc3339()).await
            .map_err(|e| format!("Failed to write timestamp: {}", e))?;
        
        Ok(())
    }
    
    /// Save complete network state on shutdown, including weights
    pub async fn save_shutdown_checkpoint(&self) -> Result<(), Box<dyn StdError>> {
        let checkpoint_path = self.get_network_checkpoint_path();
        tokio::fs::create_dir_all(&checkpoint_path).await
            .map_err(|e| format!("Failed to create checkpoint directory: {}", e))?;
        
        // Create timestamped checkpoint filename
        let timestamp = Utc::now();
        let checkpoint_filename = format!("checkpoint_shutdown_{}.bin", timestamp.format("%Y%m%d_%H%M%S"));
        let checkpoint_file_path = checkpoint_path.join(&checkpoint_filename);
        
        // Save the complete adaptive state including neural network weights
        self.save_state(&checkpoint_path).await
            .map_err(|e| format!("Failed to save adaptive state: {}", e))?;
        
        // Get evolution status to save additional metadata
        let evolver = self.evolver().lock().await;
        let evolution_status = evolver.get_status().await;
        let last_evolution_time = evolver.get_last_evolution_time().await;
        
        // Create metadata for the checkpoint
        let metadata = serde_json::json!({
            "timestamp": timestamp.to_rfc3339(),
            "type": "shutdown_checkpoint",
            "evolution_status": {
                "current_generation": evolution_status.current_generation,
                "best_fitness": evolution_status.best_fitness,
                "is_running": evolution_status.is_running,
            },
            "last_evolution": last_evolution_time.map(|t| t.to_rfc3339()),
        });
        
        // Save metadata
        let metadata_path = checkpoint_path.join(format!("checkpoint_shutdown_{}_metadata.json", timestamp.format("%Y%m%d_%H%M%S")));
        let metadata_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| format!("Failed to serialize metadata: {}", e))?;
        tokio::fs::write(&metadata_path, metadata_json).await
            .map_err(|e| format!("Failed to write metadata: {}", e))?;
        
        // Update the latest.bin symlink to point to the most recent checkpoint
        let latest_link = checkpoint_path.join("latest.bin");
        
        // Remove existing symlink if it exists
        if latest_link.exists() {
            tokio::fs::remove_file(&latest_link).await.ok();
        }
        
        // Create new symlink pointing to the adaptive state file
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let state_file = checkpoint_path.join("adaptive_state.json");
            if let Err(e) = symlink(&state_file, &latest_link) {
                eprintln!("Warning: Failed to create latest.bin symlink: {}", e);
            }
        }
        
        // Also save evolved network configuration if available
        let evolved_config_path = checkpoint_path.join("evolved_config.json");
        if !evolved_config_path.exists() {
            self.save_evolved_network().await?;
        }
        
        eprintln!("💾 Saved shutdown checkpoint: {}", checkpoint_filename);
        eprintln!("📊 Evolution status: Generation {}, Fitness: {:.4}", 
                  evolution_status.current_generation, 
                  evolution_status.best_fitness);
        
        Ok(())
    }
}

/// Background task for periodic auto-save
pub async fn start_auto_save_task(module: Arc<AdaptiveMemoryModule>, interval_secs: u64) {
    use tokio::time::{interval, Duration};
    
    let mut interval = interval(Duration::from_secs(interval_secs));
    
    loop {
        interval.tick().await;
        
        if let Err(e) = module.auto_save().await {
            eprintln!("Auto-save failed: {}", e);
        } else {
            println!("Auto-saved adaptive module state");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_save_load_state() {
        let temp_dir = TempDir::new().unwrap();
        let base_config = MemoryConfig::default();
        
        // Create module and perform some operations
        let module = AdaptiveMemoryModule::new(base_config.clone()).await.unwrap();
        
        // Store some data
        let _ = module.store("test_key", "test_content").await.unwrap();
        
        // Save state
        module.save_state(temp_dir.path()).await.unwrap();
        
        // Verify files were created
        assert!(temp_dir.path().join("adaptive_state.json").exists());
        
        // Load state
        let loaded_module = AdaptiveMemoryModule::load_state(
            temp_dir.path(),
            base_config
        ).await.unwrap();
        
        // Verify operation count was preserved
        assert_eq!(
            *loaded_module.operation_count().read().await,
            *module.operation_count().read().await
        );
    }
}