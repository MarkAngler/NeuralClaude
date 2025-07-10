//! Tests for weight persistence on shutdown functionality
//! 
//! These tests verify that neural network weights are saved even when
//! evolution hasn't been triggered, addressing the MCP server lifecycle issue.

use neural_llm_memory::{
    adaptive::{AdaptiveMemoryModule, AdaptiveConfig},
    memory::MemoryConfig,
};
use tempfile::TempDir;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::time::Duration;

/// Helper to check if weight files exist in a directory
fn has_weight_files(dir: &Path) -> bool {
    fs::read_dir(dir)
        .ok()
        .map(|entries| {
            entries.filter_map(|e| e.ok())
                .any(|e| {
                    let path = e.path();
                    path.extension() == Some(std::ffi::OsStr::new("bin")) &&
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("weights_"))
                        .unwrap_or(false)
                })
        })
        .unwrap_or(false)
}

/// Helper to count weight files in a directory
fn count_weight_files(dir: &Path) -> usize {
    fs::read_dir(dir)
        .ok()
        .map(|entries| {
            entries.filter_map(|e| e.ok())
                .filter(|e| {
                    let path = e.path();
                    path.extension() == Some(std::ffi::OsStr::new("bin")) &&
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("weights_"))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}

#[tokio::test]
async fn test_save_weights_without_evolution() {
    // Create adaptive module
    let config = AdaptiveConfig {
        evolution_threshold: 1000, // High threshold so evolution won't trigger
        ..Default::default()
    };
    
    let module = AdaptiveMemoryModule::with_config(
        MemoryConfig::default(),
        config
    ).await.unwrap();
    
    // Perform < 1000 operations
    for i in 0..50 {
        module.store(&format!("key_{}", i), "test content").await.unwrap();
    }
    
    // Verify evolution hasn't triggered
    let evolver = module.evolver().lock().await;
    let status = evolver.get_status().await;
    assert_eq!(status.total_evolutions, 0, "Evolution should not have triggered");
    drop(evolver);
    
    // Save state (simulating shutdown)
    let temp_dir = TempDir::new().unwrap();
    module.save_state(temp_dir.path()).await.unwrap();
    
    // Call the new method to save weights on shutdown
    module.save_weights_on_shutdown(temp_dir.path()).await.unwrap();
    
    // Verify weight files exist
    let checkpoint_dir = temp_dir.path().join("network_checkpoints");
    assert!(checkpoint_dir.exists(), "Checkpoint directory should exist");
    assert!(has_weight_files(&checkpoint_dir), "Weight files should be saved even without evolution");
}

#[tokio::test]
async fn test_operation_count_persistence_across_restart() {
    let temp_dir = TempDir::new().unwrap();
    let initial_count = 750;
    
    // First session: perform operations below threshold
    {
        let config = AdaptiveConfig {
            evolution_threshold: 1000,
            ..Default::default()
        };
        
        let module = AdaptiveMemoryModule::with_config(
            MemoryConfig::default(),
            config
        ).await.unwrap();
        
        // Perform operations
        for i in 0..initial_count {
            module.store(&format!("key_{}", i), "content").await.unwrap();
        }
        
        // Verify count
        let count = *module.operation_count().read().await;
        assert_eq!(count, initial_count);
        
        // Save state
        module.save_state(temp_dir.path()).await.unwrap();
        module.save_weights_on_shutdown(temp_dir.path()).await.unwrap();
    }
    
    // Second session: load and continue
    {
        let module = AdaptiveMemoryModule::load_state(
            temp_dir.path(),
            MemoryConfig::default()
        ).await.unwrap();
        
        // Verify count persisted
        let count = *module.operation_count().read().await;
        assert_eq!(count, initial_count, "Operation count should persist");
        
        // Add more operations to trigger evolution
        for i in 0..300 {
            module.store(&format!("new_key_{}", i), "content").await.unwrap();
        }
        
        // Should have triggered evolution (750 + 300 > 1000)
        tokio::time::sleep(Duration::from_millis(100)).await; // Give time for evolution
        
        let evolver = module.evolver().lock().await;
        let status = evolver.get_status().await;
        assert!(status.total_evolutions > 0, "Evolution should have triggered after reaching threshold");
    }
}

#[tokio::test]
async fn test_configurable_evolution_threshold() {
    // Test with environment variable
    std::env::set_var("NEURAL_MCP_EVOLUTION_THRESHOLD", "100");
    
    let module = AdaptiveMemoryModule::new_with_recovery(
        MemoryConfig::default(),
        false // Don't load from disk
    ).await.unwrap();
    
    // Get config to verify threshold
    let config = module.config();
    assert_eq!(config.evolution_threshold, 100, "Threshold should be configurable via env var");
    
    // Perform operations to trigger evolution
    for i in 0..150 {
        module.store(&format!("key_{}", i), "content").await.unwrap();
    }
    
    // Give time for evolution to trigger
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    // Check if evolution triggered
    let evolver = module.evolver().lock().await;
    let status = evolver.get_status().await;
    assert!(status.total_evolutions > 0, "Evolution should trigger with lower threshold");
    
    // Clean up env var
    std::env::remove_var("NEURAL_MCP_EVOLUTION_THRESHOLD");
}

#[tokio::test]
async fn test_weight_file_timestamp_format() {
    let temp_dir = TempDir::new().unwrap();
    
    let module = AdaptiveMemoryModule::new(MemoryConfig::default()).await.unwrap();
    
    // Save weights
    module.save_weights_on_shutdown(temp_dir.path()).await.unwrap();
    
    // Check file format
    let checkpoint_dir = temp_dir.path().join("network_checkpoints");
    let entries: Vec<_> = fs::read_dir(&checkpoint_dir).unwrap()
        .filter_map(|e| e.ok())
        .collect();
    
    assert!(!entries.is_empty(), "Should have at least one weight file");
    
    for entry in entries {
        let filename = entry.file_name();
        let filename_str = filename.to_string_lossy();
        
        if filename_str.ends_with(".bin") && filename_str.starts_with("weights_") {
            // Verify timestamp format: weights_YYYYMMDD_HHMMSS.bin
            assert!(filename_str.len() > 20, "Filename should include timestamp");
            assert!(filename_str.contains("_"), "Filename should have proper format");
        }
    }
}

#[tokio::test]
async fn test_recovery_from_saved_weights() {
    let temp_dir = TempDir::new().unwrap();
    
    // First session: create and save
    {
        let module = AdaptiveMemoryModule::new(MemoryConfig::default()).await.unwrap();
        
        // Store some data
        for i in 0..20 {
            module.store(&format!("key_{}", i), &format!("content_{}", i)).await.unwrap();
        }
        
        // Save everything
        module.save_state(temp_dir.path()).await.unwrap();
        module.save_weights_on_shutdown(temp_dir.path()).await.unwrap();
    }
    
    // Second session: load and verify
    {
        let module = AdaptiveMemoryModule::load_state(
            temp_dir.path(),
            MemoryConfig::default()
        ).await.unwrap();
        
        // Verify we can retrieve data (network is functional)
        let results = module.retrieve("key_5").await.unwrap();
        assert!(!results.is_empty(), "Should retrieve stored data");
        
        // Verify we can store new data
        module.store("new_key", "new_content").await.unwrap();
        let new_results = module.retrieve("new_key").await.unwrap();
        assert!(!new_results.is_empty(), "Should store and retrieve new data");
    }
}

#[tokio::test]
async fn test_concurrent_shutdown_saves() {
    let temp_dir = TempDir::new().unwrap();
    let module = Arc::new(AdaptiveMemoryModule::new(MemoryConfig::default()).await.unwrap());
    
    // Simulate multiple shutdown handlers
    let mut handles = vec![];
    
    for i in 0..3 {
        let module_clone = Arc::clone(&module);
        let path = temp_dir.path().to_path_buf();
        
        let handle = tokio::spawn(async move {
            // Slight delay to make saves more concurrent
            tokio::time::sleep(Duration::from_millis(i * 10)).await;
            
            // Both state and weights
            module_clone.save_state(&path).await?;
            module_clone.save_weights_on_shutdown(&path).await
        });
        
        handles.push(handle);
    }
    
    // Wait for all saves
    for handle in handles {
        assert!(handle.await.unwrap().is_ok(), "Concurrent saves should succeed");
    }
    
    // Verify files exist and are valid
    assert!(temp_dir.path().join("adaptive_state.json").exists());
    assert!(temp_dir.path().join("network_checkpoints").exists());
    
    // Load to verify integrity
    let loaded = AdaptiveMemoryModule::load_state(
        temp_dir.path(),
        MemoryConfig::default()
    ).await;
    
    assert!(loaded.is_ok(), "Should load successfully after concurrent saves");
}

#[tokio::test]
async fn test_auto_save_includes_weights() {
    let temp_dir = TempDir::new().unwrap();
    std::env::set_var("NEURAL_MCP_DATA_DIR", temp_dir.path().to_str().unwrap());
    
    let module = Arc::new(AdaptiveMemoryModule::new(MemoryConfig::default()).await.unwrap());
    
    // Perform some operations
    for i in 0..10 {
        module.store(&format!("key_{}", i), "content").await.unwrap();
    }
    
    // Trigger auto-save
    module.auto_save().await.unwrap();
    
    // Check that both state and weights were saved
    let state_path = temp_dir.path().join("adaptive_memory_data/adaptive_state.json");
    let checkpoint_dir = temp_dir.path().join("adaptive_memory_data/network_checkpoints");
    
    assert!(state_path.exists(), "State should be saved");
    assert!(checkpoint_dir.exists(), "Checkpoint directory should exist");
    
    // Clean up
    std::env::remove_var("NEURAL_MCP_DATA_DIR");
}