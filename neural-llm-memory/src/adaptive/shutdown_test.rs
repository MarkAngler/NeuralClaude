#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::MemoryConfig;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use tokio::fs;
    
    #[tokio::test]
    async fn test_shutdown_checkpoint() -> Result<(), Box<dyn std::error::Error>> {
        // Create a temporary directory for test
        let _temp_dir = TempDir::new()?;
        
        // Create adaptive module
        let config = MemoryConfig::default();
        let module = AdaptiveMemoryModule::new(config).await?;
        
        // Perform some operations to have data to save
        module.store("test_key1", "test_content1").await?;
        module.store("test_key2", "test_content2").await?;
        
        // Test save_shutdown_checkpoint method
        module.save_shutdown_checkpoint().await?;
        
        // Verify checkpoint files were created
        let checkpoint_path = PathBuf::from("./adaptive_memory_data/network_checkpoints");
        assert!(checkpoint_path.exists(), "Checkpoint directory should exist");
        
        // Check for adaptive state file
        let state_file = checkpoint_path.join("adaptive_state.json");
        assert!(state_file.exists(), "Adaptive state file should exist");
        
        // Check for evolved config
        let evolved_config = checkpoint_path.join("evolved_config.json");
        assert!(evolved_config.exists(), "Evolved config should exist");
        
        // Check for metadata files (should have at least one)
        let entries = fs::read_dir(&checkpoint_path).await?;
        let mut metadata_found = false;
        let mut entries = entries;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.contains("checkpoint_shutdown_") && name.ends_with("_metadata.json") {
                    metadata_found = true;
                    
                    // Verify metadata content
                    let metadata_content = fs::read_to_string(&path).await?;
                    let metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;
                    
                    assert!(metadata.get("timestamp").is_some());
                    assert_eq!(metadata.get("type").and_then(|v| v.as_str()), Some("shutdown_checkpoint"));
                    assert!(metadata.get("evolution_status").is_some());
                }
            }
        }
        
        assert!(metadata_found, "Should have found at least one metadata file");
        
        // Check for latest.bin symlink on Unix systems
        #[cfg(unix)]
        {
            let latest_link = checkpoint_path.join("latest.bin");
            if latest_link.exists() {
                assert!(latest_link.symlink_metadata()?.file_type().is_symlink());
            }
        }
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_shutdown_checkpoint_creates_timestamp() -> Result<(), Box<dyn std::error::Error>> {
        let _temp_dir = TempDir::new()?;
        
        let config = MemoryConfig::default();
        let module = AdaptiveMemoryModule::new(config).await?;
        
        // Save checkpoint twice with a delay of at least 1 second to ensure different timestamps
        module.save_shutdown_checkpoint().await?;
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        module.save_shutdown_checkpoint().await?;
        
        // Check that we have multiple timestamped checkpoints
        let checkpoint_path = PathBuf::from("./adaptive_memory_data/network_checkpoints");
        let entries = fs::read_dir(&checkpoint_path).await?;
        
        let mut checkpoint_count = 0;
        let mut entries = entries;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("checkpoint_shutdown_") && name.ends_with("_metadata.json") {
                    checkpoint_count += 1;
                }
            }
        }
        
        assert!(checkpoint_count >= 2, "Should have at least 2 timestamped checkpoints");
        
        Ok(())
    }
}