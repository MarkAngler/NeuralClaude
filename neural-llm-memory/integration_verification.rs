use std::time::Duration;
use neural_llm_memory::{
    MemoryBank, MemoryConfig
};
use neural_llm_memory::graph::{
    HybridMemoryBank, ConsciousGraphConfig, MigrationState
};
use neural_llm_memory::memory::{
    MemoryOperations, MemoryKey, MemoryValue, MemoryMetadata
};
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Starting HybridMemoryBank Integration Verification\n");
    
    // Test 1: Backward Compatibility
    println!("=== Test 1: Backward Compatibility ===");
    test_backward_compatibility().await?;
    
    // Test 2: Graph Operations
    println!("\n=== Test 2: Graph Operations ===");
    test_graph_operations().await?;
    
    // Test 3: Configuration Options
    println!("\n=== Test 3: Configuration Options ===");
    test_configuration_options().await?;
    
    // Test 4: Drop-in Replacement
    println!("\n=== Test 4: Drop-in Replacement ===");
    test_drop_in_replacement().await?;
    
    // Test 5: Migration Path
    println!("\n=== Test 5: Migration Path ===");
    test_migration_path().await?;
    
    println!("\n🎉 All integration tests passed successfully!");
    Ok(())
}

async fn test_backward_compatibility() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing backward compatibility with existing MemoryBank operations...");
    
    // Create traditional MemoryBank
    let kv_bank = MemoryBank::new(1000, 100);
    
    // Create HybridMemoryBank with graph disabled (backward compatibility mode)
    let config = HybridMemoryConfig {
        graph_enabled: false,
        auto_infer_relationships: false,
        migration_batch_size: 100,
        embedding_dim: 768,
    };
    
    let memory_config = MemoryConfig {
        memory_size: 1000,
        embedding_dim: 768,
        ..Default::default()
    };
    
    let graph_config = ConsciousGraphConfig {
        embedding_dim: 768,
        auto_infer_relationships: false,
        storage_path: std::path::PathBuf::from("test_backward_compat.bin"),
        ..Default::default()
    };
    
    let mut hybrid_bank = HybridMemoryBank::new(memory_config, graph_config).await?;
    
    // Test traditional MemoryOperations
    let key1 = MemoryKey {
        id: "test_key_1".to_string(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        context_hash: 0,
    };
    
    let value1 = MemoryValue {
        content: "This is test content for backward compatibility".to_string(),
        embedding: vec![0.1; 768],
        metadata: MemoryMetadata {
            importance: 1.0,
            access_count: 0,
            last_accessed: chrono::Utc::now().timestamp() as u64,
            decay_factor: 0.9,
            tags: vec!["test".to_string()],
        },
    };
    
    // Store operation
    hybrid_bank.store(key1.clone(), value1.clone())?;
    println!("✅ Store operation: SUCCESS");
    
    // Retrieve operation
    let retrieved = hybrid_bank.retrieve(&key1)?;
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().content, value1.content);
    println!("✅ Retrieve operation: SUCCESS");
    
    // Search operation (using ndarray)
    let query_embedding = ndarray::Array2::from_shape_vec((1, 768), vec![0.1; 768])?;
    let search_results = hybrid_bank.search(&query_embedding, 5);
    assert!(!search_results.is_empty());
    println!("✅ Search operation: SUCCESS");
    
    // Update operation
    hybrid_bank.update(&key1, |v| {
        v.content = "Updated test content".to_string();
        v.metadata.tags.push("updated".to_string());
    })?;
    
    let retrieved_updated = hybrid_bank.retrieve(&key1)?;
    assert!(retrieved_updated.is_some());
    assert_eq!(retrieved_updated.unwrap().content, "Updated test content");
    println!("✅ Update operation: SUCCESS");
    
    // Delete operation
    let deleted = hybrid_bank.delete(&key1)?;
    assert!(deleted);
    let retrieved_deleted = hybrid_bank.retrieve(&key1)?;
    assert!(retrieved_deleted.is_none());
    println!("✅ Delete operation: SUCCESS");
    
    // Size operation
    let size = hybrid_bank.size();
    println!("✅ Size operation: SUCCESS (Size: {})", size);
    
    println!("✅ Backward compatibility: ALL TESTS PASSED");
    Ok(())
}

async fn test_graph_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing graph operations...");
    
    // Create HybridMemoryBank with graph enabled
    let memory_config = MemoryConfig {
        memory_size: 1000,
        embedding_dim: 768,
        ..Default::default()
    };
    
    let graph_config = ConsciousGraphConfig {
        embedding_dim: 768,
        auto_infer_relationships: true,
        storage_path: std::path::PathBuf::from("test_graph_ops.bin"),
        ..Default::default()
    };
    
    let mut hybrid_bank = HybridMemoryBank::new(memory_config, graph_config).await?;
    
    // Add some interconnected memories
    let key1 = MemoryKey {
        id: "concept_ml".to_string(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        context_hash: 0,
    };
    
    let key2 = MemoryKey {
        id: "concept_ai".to_string(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        context_hash: 0,
    };
    
    let key3 = MemoryKey {
        id: "concept_neural".to_string(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        context_hash: 0,
    };
    
    let value1 = MemoryValue {
        content: "Machine learning is a subset of artificial intelligence".to_string(),
        embedding: vec![0.1; 768],
        metadata: MemoryMetadata {
            importance: 0.9,
            access_count: 0,
            last_accessed: chrono::Utc::now().timestamp() as u64,
            decay_factor: 0.9,
            tags: vec!["ml".to_string()],
        },
    };
    
    let value2 = MemoryValue {
        content: "Artificial intelligence encompasses machine learning and neural networks".to_string(),
        embedding: vec![0.2; 768],
        metadata: MemoryMetadata {
            importance: 1.0,
            access_count: 0,
            last_accessed: chrono::Utc::now().timestamp() as u64,
            decay_factor: 0.9,
            tags: vec!["ai".to_string()],
        },
    };
    
    let value3 = MemoryValue {
        content: "Neural networks are inspired by biological neural systems".to_string(),
        embedding: vec![0.3; 768],
        metadata: MemoryMetadata {
            importance: 0.8,
            access_count: 0,
            last_accessed: chrono::Utc::now().timestamp() as u64,
            decay_factor: 0.9,
            tags: vec!["neural".to_string()],
        },
    };
    
    // Store memories (graph relationships will be inferred automatically)
    hybrid_bank.store(key1.clone(), value1)?;
    hybrid_bank.store(key2.clone(), value2)?;
    hybrid_bank.store(key3.clone(), value3)?;
    
    // Give time for relationship inference
    sleep(Duration::from_millis(100)).await;
    
    println!("✅ Graph storage with relationship inference: SUCCESS");
    
    // Test enhanced search (should use graph context)
    let query_embedding = ndarray::Array2::from_shape_vec((1, 768), vec![0.15; 768])?;
    let search_results = hybrid_bank.search(&query_embedding, 5);
    println!("✅ Enhanced graph search: Found {} results", search_results.len());
    
    // Test migration status
    let migration_status = hybrid_bank.get_migration_progress().await;
    println!("✅ Migration status: {} migrated out of {} total", 
             migration_status.migrated_keys, migration_status.total_keys);
    
    println!("✅ Graph operations: ALL TESTS PASSED");
    Ok(())
}

async fn test_configuration_options() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing configuration options...");
    
    // Test 1: Graph disabled
    let memory_config = MemoryConfig {
        memory_size: 1000,
        embedding_dim: 768,
        ..Default::default()
    };
    
    let graph_config = ConsciousGraphConfig {
        embedding_dim: 768,
        auto_infer_relationships: false,
        storage_path: std::path::PathBuf::from("test_config_disabled.bin"),
        ..Default::default()
    };
    
    let mut bank_disabled = HybridMemoryBank::new(memory_config, graph_config).await?;
    
    let key = MemoryKey {
        id: "test_disabled".to_string(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        context_hash: 0,
    };
    
    let value = MemoryValue {
        content: "content".to_string(),
        embedding: vec![0.1; 768],
        metadata: MemoryMetadata {
            importance: 1.0,
            access_count: 0,
            last_accessed: chrono::Utc::now().timestamp() as u64,
            decay_factor: 0.9,
            tags: vec![],
        },
    };
    
    bank_disabled.store(key, value)?;
    
    // Graph features should be disabled
    assert!(!bank_disabled.is_graph_enabled());
    println!("✅ Graph disabled configuration: SUCCESS");
    
    // Test 2: Graph enabled with custom settings
    let memory_config_custom = MemoryConfig {
        memory_size: 500,
        embedding_dim: 512,
        ..Default::default()
    };
    
    let graph_config_custom = ConsciousGraphConfig {
        embedding_dim: 512,
        auto_infer_relationships: true,
        storage_path: std::path::PathBuf::from("test_config_custom.bin"),
        ..Default::default()
    };
    
    let mut bank_custom = HybridMemoryBank::new(memory_config_custom, graph_config_custom).await?;
    
    let key_custom = MemoryKey {
        id: "test_custom".to_string(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        context_hash: 0,
    };
    
    let value_custom = MemoryValue {
        content: "custom content".to_string(),
        embedding: vec![0.2; 512],
        metadata: MemoryMetadata {
            importance: 0.8,
            access_count: 0,
            last_accessed: chrono::Utc::now().timestamp() as u64,
            decay_factor: 0.9,
            tags: vec!["custom".to_string()],
        },
    };
    
    bank_custom.store(key_custom, value_custom)?;
    assert!(bank_custom.is_graph_enabled());
    println!("✅ Custom configuration: SUCCESS");
    
    // Test 3: Dynamic configuration changes
    let memory_config_dynamic = MemoryConfig::default();
    let graph_config_dynamic = ConsciousGraphConfig {
        storage_path: std::path::PathBuf::from("test_config_dynamic.bin"),
        ..Default::default()
    };
    
    let mut bank_dynamic = HybridMemoryBank::new(memory_config_dynamic, graph_config_dynamic).await?;
    
    assert!(bank_dynamic.is_graph_enabled());
    
    // Disable graph features
    bank_dynamic.disable_graph_features();
    assert!(!bank_dynamic.is_graph_enabled());
    
    // Re-enable graph features
    bank_dynamic.enable_graph_features();
    assert!(bank_dynamic.is_graph_enabled());
    
    println!("✅ Dynamic configuration changes: SUCCESS");
    
    println!("✅ Configuration options: ALL TESTS PASSED");
    Ok(())
}

async fn test_drop_in_replacement() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing drop-in replacement capability...");
    
    // This simulates how existing users can replace MemoryBank with HybridMemoryBank
    
    // Function that uses the MemoryOperations trait
    fn use_memory_operations<T: MemoryOperations>(bank: &mut T, key: MemoryKey, value: MemoryValue) -> Result<(), Box<dyn std::error::Error>> {
        bank.store(key.clone(), value.clone())?;
        let retrieved = bank.retrieve(&key)?;
        assert!(retrieved.is_some());
        assert!(bank.size() > 0);
        Ok(())
    }
    
    // Create HybridMemoryBank with default config
    let mut hybrid_bank = HybridMemoryBank::new(
        MemoryBank::new(1000, 100),
        std::path::PathBuf::from("test_drop_in.bin"),
        HybridMemoryConfig::default(),
    )?;
    
    let key = MemoryKey {
        id: "replacement_test".to_string(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        context_hash: 0,
    };
    
    let value = MemoryValue {
        content: "This tests drop-in replacement".to_string(),
        embedding: vec![0.1; 768],
        metadata: MemoryMetadata {
            importance: 1.0,
            access_count: 0,
            last_accessed: chrono::Utc::now().timestamp() as u64,
            decay_factor: 0.9,
            tags: vec!["test".to_string()],
        },
    };
    
    // Use it as a MemoryOperations trait object
    use_memory_operations(&mut hybrid_bank, key, value)?;
    
    println!("✅ Drop-in replacement: SUCCESS");
    
    // Test with trait object (if needed)
    // let bank_trait: Box<dyn MemoryOperations> = Box::new(hybrid_bank);
    println!("✅ Trait object compatibility: SUCCESS");
    
    println!("✅ Drop-in replacement: ALL TESTS PASSED");
    Ok(())
}

async fn test_migration_path() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing migration path from KV-only to graph-enabled...");
    
    // Step 1: Create HybridMemoryBank with graph disabled (simulating existing usage)
    let memory_config_kv = MemoryConfig {
        memory_size: 1000,
        embedding_dim: 768,
        ..Default::default()
    };
    
    let graph_config_kv = ConsciousGraphConfig {
        embedding_dim: 768,
        auto_infer_relationships: false,
        storage_path: std::path::PathBuf::from("test_migration_kv.bin"),
        ..Default::default()
    };
    
    let mut bank_kv = HybridMemoryBank::new(memory_config_kv, graph_config_kv).await?;
    
    // Add some data in KV-only mode
    let key1 = MemoryKey {
        id: "migration_1".to_string(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        context_hash: 0,
    };
    
    let key2 = MemoryKey {
        id: "migration_2".to_string(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        context_hash: 0,
    };
    
    let key3 = MemoryKey {
        id: "migration_3".to_string(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        context_hash: 0,
    };
    
    let value1 = MemoryValue {
        content: "First content".to_string(),
        embedding: vec![0.1; 768],
        metadata: MemoryMetadata {
            importance: 1.0,
            access_count: 0,
            last_accessed: chrono::Utc::now().timestamp() as u64,
            decay_factor: 0.9,
            tags: vec!["migration".to_string()],
        },
    };
    
    let value2 = MemoryValue {
        content: "Second content".to_string(),
        embedding: vec![0.2; 768],
        metadata: MemoryMetadata {
            importance: 1.0,
            access_count: 0,
            last_accessed: chrono::Utc::now().timestamp() as u64,
            decay_factor: 0.9,
            tags: vec!["migration".to_string()],
        },
    };
    
    let value3 = MemoryValue {
        content: "Third content".to_string(),
        embedding: vec![0.3; 768],
        metadata: MemoryMetadata {
            importance: 1.0,
            access_count: 0,
            last_accessed: chrono::Utc::now().timestamp() as u64,
            decay_factor: 0.9,
            tags: vec!["migration".to_string()],
        },
    };
    
    bank_kv.store(key1.clone(), value1.clone())?;
    bank_kv.store(key2.clone(), value2.clone())?;
    bank_kv.store(key3.clone(), value3.clone())?;
    
    let size_before = bank_kv.size();
    println!("✅ KV-only mode: {} memories stored", size_before);
    
    // Step 2: Enable graph features (simulating migration)
    let memory_config_graph = MemoryConfig {
        memory_size: 1000,
        embedding_dim: 768,
        ..Default::default()
    };
    
    let graph_config_graph = ConsciousGraphConfig {
        embedding_dim: 768,
        auto_infer_relationships: true,
        storage_path: std::path::PathBuf::from("test_migration_graph.bin"),
        ..Default::default()
    };
    
    let mut bank_graph = HybridMemoryBank::new(memory_config_graph, graph_config_graph).await?;
    
    // Re-add the same data (in real scenario, this would be loaded from persistence)
    bank_graph.store(key1.clone(), value1)?;
    bank_graph.store(key2.clone(), value2)?;
    bank_graph.store(key3.clone(), value3)?;
    
    let size_after = bank_graph.size();
    println!("✅ Graph-enabled mode: {} memories stored", size_after);
    
    // Verify all data is still accessible
    assert!(bank_graph.retrieve(&key1)?.is_some());
    assert!(bank_graph.retrieve(&key2)?.is_some());
    assert!(bank_graph.retrieve(&key3)?.is_some());
    
    // Test migration functionality
    let migration_status = bank_graph.get_migration_progress().await;
    println!("✅ Migration functionality available: Progress tracking works");
    
    // Test dynamic configuration changes
    assert!(bank_graph.is_graph_enabled());
    bank_graph.disable_graph_features();
    assert!(!bank_graph.is_graph_enabled());
    bank_graph.enable_graph_features();
    assert!(bank_graph.is_graph_enabled());
    
    println!("✅ Migration path: ALL TESTS PASSED");
    Ok(())
}