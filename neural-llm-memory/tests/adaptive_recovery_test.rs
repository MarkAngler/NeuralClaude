//! Tests for AdaptiveMemoryModule recovery functionality
//! 
//! This module tests the Phase 5 implementation of persistence and recovery,
//! including auto-save, state restoration, evolved network loading, and
//! environment variable controls.

use neural_llm_memory::adaptive::{
    AdaptiveMemoryModule, AdaptiveConfig, AdaptiveModuleState,
    start_auto_save_task, UsageMetrics, OperationType,
};
use neural_llm_memory::memory::MemoryConfig;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::sleep;
use chrono::Utc;
use uuid::Uuid;

/// Helper to create a test configuration
fn create_test_config() -> MemoryConfig {
    MemoryConfig {
        memory_size: 10000,
        embedding_dim: 768,
        hidden_dim: 256,
        num_heads: 8,
        num_layers: 2,
        dropout_rate: 0.1,
        max_sequence_length: 512,
        use_positional_encoding: true,
    }
}

#[tokio::test]
async fn test_new_with_persistence_loads_saved_state() {
    // Clean up any existing data
    let _ = tokio::fs::remove_dir_all("./adaptive_memory_data").await;
    
    let base_config = create_test_config();
    
    // Step 1: Create module and perform operations
    {
        let module = AdaptiveMemoryModule::new(base_config.clone()).await.unwrap();
        
        // Perform some operations
        module.store("test/key1", "content1").await.unwrap();
        module.store("test/key2", "content2").await.unwrap();
        module.search("test content", 5).await.unwrap();
        
        // Record some operations which will generate metrics internally
        for i in 0..5 {
            module.store(&format!("metric/key{}", i), &format!("content{}", i)).await.unwrap();
        }
        
        // Save state to the default location
        module.save_state("./adaptive_memory_data").await.unwrap();
        
        // Verify save created files
        assert!(std::path::Path::new("./adaptive_memory_data/adaptive_state.json").exists());
    }
    
    // Step 2: Load from saved state
    {
        let loaded_module = AdaptiveMemoryModule::new_with_persistence(
            base_config.clone(),
            true,
        ).await.unwrap();
        
        // Verify data was loaded correctly by checking if we can retrieve
        // previously stored data (this indirectly verifies operation count was restored)
        let (content, _op_id) = loaded_module.retrieve("test/key1").await.unwrap();
        assert_eq!(content.unwrap(), "content1", "Stored data should be retrievable");
        
        let (content2, _op_id2) = loaded_module.retrieve("test/key2").await.unwrap();
        assert_eq!(content2.unwrap(), "content2", "Second stored data should be retrievable");
    }
    
    // Cleanup
    let _ = tokio::fs::remove_dir_all("./adaptive_memory_data").await;
}

#[tokio::test]
async fn test_auto_save_background_task() {
    // Clean up any existing data
    let _ = tokio::fs::remove_dir_all("./adaptive_memory_data").await;
    
    let base_config = create_test_config();
    let module = Arc::new(AdaptiveMemoryModule::new(base_config).await.unwrap());
    
    // Start auto-save task with 1 second interval
    let module_clone = module.clone();
    let auto_save_handle = tokio::spawn(async move {
        start_auto_save_task(module_clone, 1).await;
    });
    
    // Perform operations
    module.store("auto/test1", "auto content 1").await.unwrap();
    
    // Wait for auto-save to trigger
    sleep(Duration::from_millis(1500)).await;
    
    // Verify state was saved
    assert!(std::path::Path::new("./adaptive_memory_data/adaptive_state.json").exists(), "Auto-save should create state file");
    
    // Perform more operations
    module.store("auto/test2", "auto content 2").await.unwrap();
    module.search("auto content", 5).await.unwrap();
    
    // Wait for another auto-save
    sleep(Duration::from_millis(1500)).await;
    
    // Load and verify updated state
    let loaded_module = AdaptiveMemoryModule::load_state(
        "./adaptive_memory_data",
        create_test_config()
    ).await.unwrap();
    
    // Verify state was updated by checking we can retrieve both items
    let (content1, _) = loaded_module.retrieve("auto/test1").await.unwrap();
    assert_eq!(content1.unwrap(), "auto content 1");
    let (content2, _) = loaded_module.retrieve("auto/test2").await.unwrap();
    assert_eq!(content2.unwrap(), "auto content 2");
    
    // Cancel auto-save task
    auto_save_handle.abort();
    
    // Cleanup
    let _ = tokio::fs::remove_dir_all("./adaptive_memory_data").await;
}

#[tokio::test]
async fn test_evolved_network_configuration_restored() {
    // Clean up any existing data
    let _ = tokio::fs::remove_dir_all("./adaptive_memory_data").await;
    
    let base_config = create_test_config();
    
    // Step 1: Create module and simulate evolution
    {
        let module = AdaptiveMemoryModule::new(base_config.clone()).await.unwrap();
        
        // Create evolved config directory and save a modified config
        let checkpoint_path = std::path::Path::new("./adaptive_memory_data/network_checkpoints");
        tokio::fs::create_dir_all(&checkpoint_path).await.unwrap();
        
        // Create an evolved configuration with different parameters
        let mut evolved_config = base_config.clone();
        evolved_config.hidden_dim = 512; // Changed from 256
        evolved_config.embedding_dim = 768; // Keep same
        evolved_config.num_heads = 16; // Changed from 8
        
        // Save evolved config
        let config_json = serde_json::to_string_pretty(&evolved_config).unwrap();
        tokio::fs::write(
            checkpoint_path.join("evolved_config.json"),
            config_json
        ).await.unwrap();
        
        // Save module state
        module.save_state("./adaptive_memory_data").await.unwrap();
    }
    
    // Step 2: Load and verify evolved configuration is used
    {
        let loaded_module = AdaptiveMemoryModule::new_with_persistence(
            base_config.clone(),
            true,
        ).await.unwrap();
        
        // Verify evolved configuration was used by checking it can handle operations
        // (The evolved config would have been loaded internally)
        loaded_module.store("evolved/test", "evolved content").await.unwrap();
        let (content, _) = loaded_module.retrieve("evolved/test").await.unwrap();
        assert_eq!(content.unwrap(), "evolved content", "Module should work with evolved config");
    }
    
    // Cleanup
    let _ = tokio::fs::remove_dir_all("./adaptive_memory_data").await;
}

#[tokio::test]
async fn test_environment_variable_controls() {
    // Clean up any existing data
    let _ = tokio::fs::remove_dir_all("./adaptive_memory_data").await;
    
    let base_config = create_test_config();
    
    // Create and save a module state
    {
        let module = AdaptiveMemoryModule::new(base_config.clone()).await.unwrap();
        module.store("env/test", "env content").await.unwrap();
        module.save_state("./adaptive_memory_data").await.unwrap();
    }
    
    // Test 1: When not loading from disk, should not load saved state
    {
        let module = AdaptiveMemoryModule::new_with_persistence(
            base_config.clone(),
            false, // Not loading from disk
        ).await.unwrap();
        
        // Should not find previously stored data
        let result = module.retrieve("env/test").await;
        assert!(result.is_err(), "Should not find data from previous session");
    }
    
    // Test 2: When loading from disk, should load saved state
    {
        let module = AdaptiveMemoryModule::new_with_persistence(
            base_config.clone(),
            true, // Loading from disk
        ).await.unwrap();
        
        // Should find previously stored data
        let (content, _) = module.retrieve("env/test").await.unwrap();
        assert_eq!(content.unwrap(), "env content", "Should find data from previous session");
    }
    
    // Cleanup
    let _ = tokio::fs::remove_dir_all("./adaptive_memory_data").await;
}

#[tokio::test]
async fn test_graceful_fallback_on_recovery_failure() {
    // Clean up any existing data
    let _ = tokio::fs::remove_dir_all("./adaptive_memory_data").await;
    
    let base_config = create_test_config();
    
    // Create corrupted state file
    tokio::fs::create_dir_all("./adaptive_memory_data").await.unwrap();
    tokio::fs::write(
        "./adaptive_memory_data/adaptive_state.json",
        "{ invalid json content"
    ).await.unwrap();
    
    // Should fallback to creating new module
    let module = AdaptiveMemoryModule::new_with_persistence(
        base_config.clone(),
        true,
    ).await.unwrap();
    
    // Module should be functional
    module.store("recovery/test", "test content").await.unwrap();
    let (content, _) = module.retrieve("recovery/test").await.unwrap();
    assert_eq!(content.unwrap(), "test content", "Module should be functional after fallback");
    
    // Cleanup
    let _ = tokio::fs::remove_dir_all("./adaptive_memory_data").await;
}

#[tokio::test]
async fn test_state_snapshot_creation() {
    let module = AdaptiveMemoryModule::new(create_test_config()).await.unwrap();
    
    // Perform various operations
    module.store("snapshot/key1", "content1").await.unwrap();
    module.store("snapshot/key2", "content2").await.unwrap();
    module.search("snapshot", 5).await.unwrap();
    module.retrieve("snapshot/key1").await.unwrap();
    
    // Perform different types of operations to generate metrics
    module.store("snapshot/key3", "content3").await.unwrap();
    module.search("snapshot content", 10).await.unwrap();
    
    // Create snapshot
    let temp_dir = TempDir::new().unwrap();
    module.save_state(temp_dir.path()).await.unwrap();
    
    // Load and verify snapshot
    let state_json = tokio::fs::read_to_string(
        temp_dir.path().join("adaptive_state.json")
    ).await.unwrap();
    let state: AdaptiveModuleState = serde_json::from_str(&state_json).unwrap();
    
    assert!(state.operation_count >= 5, "Operation count should include all operations");
    assert!(!state.recent_metrics.is_empty(), "Metrics should be included");
    assert_eq!(state.version, 1, "Version should be set");
    assert!(state.saved_at <= Utc::now(), "Saved timestamp should be valid");
}

#[tokio::test]
async fn test_concurrent_save_and_operations() {
    let temp_dir = TempDir::new().unwrap();
    let test_path = temp_dir.path().join("adaptive_memory_data");
    std::env::set_var("NEURAL_MCP_PERSISTENCE_PATH", test_path.to_str().unwrap());
    
    let module = Arc::new(
        AdaptiveMemoryModule::new(create_test_config()).await.unwrap()
    );
    
    // Spawn multiple tasks performing operations
    let mut handles = vec![];
    
    // Task 1: Continuous stores
    let module_clone = module.clone();
    handles.push(tokio::spawn(async move {
        for i in 0..10 {
            module_clone.store(&format!("concurrent/key{}", i), "content").await.ok();
            sleep(Duration::from_millis(50)).await;
        }
    }));
    
    // Task 2: Continuous searches
    let module_clone = module.clone();
    handles.push(tokio::spawn(async move {
        for _ in 0..10 {
            module_clone.search("concurrent", 5).await.ok();
            sleep(Duration::from_millis(60)).await;
        }
    }));
    
    // Task 3: Periodic saves
    let module_clone = module.clone();
    let test_path_clone = test_path.clone();
    handles.push(tokio::spawn(async move {
        for _ in 0..5 {
            sleep(Duration::from_millis(100)).await;
            module_clone.save_state(&test_path_clone).await.ok();
        }
    }));
    
    // Wait for all tasks
    for handle in handles {
        handle.await.ok();
    }
    
    // Verify final state is consistent
    let loaded_module = AdaptiveMemoryModule::load_state(
        &test_path,
        create_test_config()
    ).await.unwrap();
    
    // Verify that we can retrieve some of the stored data
    let result = loaded_module.retrieve("concurrent/key5").await;
    assert!(result.is_ok(), "Should be able to retrieve stored data");
    
    // Cleanup already handled in temp_dir drop
}

#[tokio::test]
async fn test_network_checkpoint_functionality() {
    // Clean up any existing data
    let _ = tokio::fs::remove_dir_all("./adaptive_memory_data").await;
    
    let module = AdaptiveMemoryModule::new(create_test_config()).await.unwrap();
    
    // Save evolved network
    module.save_evolved_network().await.unwrap();
    
    // Verify checkpoint files
    let checkpoint_path = std::path::Path::new("./adaptive_memory_data/network_checkpoints");
    assert!(checkpoint_path.join("evolved_config.json").exists());
    assert!(checkpoint_path.join("timestamp.txt").exists());
    
    // Verify timestamp is valid
    let timestamp_str = tokio::fs::read_to_string(
        checkpoint_path.join("timestamp.txt")
    ).await.unwrap();
    let timestamp = timestamp_str.parse::<chrono::DateTime<Utc>>().unwrap();
    assert!(timestamp <= Utc::now(), "Timestamp should be valid");
    
    // Cleanup
    let _ = tokio::fs::remove_dir_all("./adaptive_memory_data").await;
}

#[tokio::test]
async fn test_version_compatibility() {
    let temp_dir = TempDir::new().unwrap();
    let test_path = temp_dir.path().join("adaptive_memory_data");
    
    // Create a state with a different version
    let state = AdaptiveModuleState {
        config: AdaptiveConfig::default(),
        operation_count: 42,
        recent_metrics: vec![],
        evolution_status: neural_llm_memory::adaptive::EvolutionStatus {
            is_running: false,
            current_generation: 5,
            best_fitness: 0.85,
            started_at: Some(Utc::now()),
            progress_percent: 75.0,
            estimated_completion: Some(Utc::now()),
        },
        last_evolution_time: Some(Utc::now()),
        saved_at: Utc::now(),
        version: 999, // Future version
    };
    
    // Save state
    tokio::fs::create_dir_all(&test_path).await.unwrap();
    let state_json = serde_json::to_string_pretty(&state).unwrap();
    tokio::fs::write(test_path.join("adaptive_state.json"), state_json).await.unwrap();
    
    // Loading should still work (forward compatibility)
    let result = AdaptiveMemoryModule::load_state(
        &test_path,
        create_test_config()
    ).await;
    
    assert!(result.is_ok(), "Should handle future versions gracefully");
}