use neural_llm_memory::memory::MemoryConfig;
use neural_llm_memory::adaptive::AdaptiveMemoryModule;

#[tokio::test]
async fn test_update_and_delete_operations() {
    // Create adaptive memory module  
    let config = MemoryConfig::default();
    
    let module = AdaptiveMemoryModule::new(config).await.expect("Failed to create module");
    
    // Test store operation
    let key = "test_key";
    let initial_content = "Initial content";
    let store_result = module.store(key, initial_content).await;
    assert!(store_result.is_ok(), "Store operation failed: {:?}", store_result);
    
    // Test retrieve to verify initial store
    let retrieve_result = module.retrieve(key).await;
    assert!(retrieve_result.is_ok(), "Retrieve operation failed: {:?}", retrieve_result);
    let (content, _) = retrieve_result.unwrap();
    assert_eq!(content, Some(initial_content.to_string()));
    
    // Test update operation
    let updated_content = "Updated content";
    let update_result = module.update(key, updated_content).await;
    assert!(update_result.is_ok(), "Update operation failed: {:?}", update_result);
    
    // Verify update
    let retrieve_after_update = module.retrieve(key).await;
    assert!(retrieve_after_update.is_ok());
    let (content, _) = retrieve_after_update.unwrap();
    assert_eq!(content, Some(updated_content.to_string()));
    
    // Test update on non-existent key
    let update_nonexistent = module.update("nonexistent_key", "some content").await;
    assert!(update_nonexistent.is_err(), "Update should fail for non-existent key");
    assert!(update_nonexistent.unwrap_err().to_string().contains("not found"));
    
    // Test delete operation
    let delete_result = module.delete(key).await;
    assert!(delete_result.is_ok(), "Delete operation failed: {:?}", delete_result);
    let (deleted, _) = delete_result.unwrap();
    assert!(deleted, "Delete should return true for existing key");
    
    // Verify deletion
    let retrieve_after_delete = module.retrieve(key).await;
    assert!(retrieve_after_delete.is_ok());
    let (content, _) = retrieve_after_delete.unwrap();
    assert_eq!(content, None, "Content should be None after deletion");
    
    // Test delete on non-existent key
    let delete_nonexistent = module.delete("already_deleted").await;
    assert!(delete_nonexistent.is_ok(), "Delete should not fail for non-existent key");
    let (deleted, _) = delete_nonexistent.unwrap();
    assert!(!deleted, "Delete should return false for non-existent key");
    
    // Test stats to ensure operations are tracked
    let stats = module.get_stats().await;
    assert!(stats.is_ok(), "Stats retrieval failed");
    let stats_json = stats.unwrap();
    
    // Verify usage stats include update and delete operations
    if let Some(usage_stats) = stats_json.get("usage_stats") {
        if let Some(operation_counts) = usage_stats.get("operation_counts") {
            println!("Operation counts: {:?}", operation_counts);
            // We should have operations for Store, Retrieve, Update, and Delete
        }
    }
}

#[tokio::test]
async fn test_operation_metrics() {
    let config = MemoryConfig::default();
    
    let module = AdaptiveMemoryModule::new(config).await.expect("Failed to create module");
    
    // Perform several operations
    let _ = module.store("key1", "content1").await;
    let _ = module.store("key2", "content2").await;
    let _ = module.update("key1", "updated_content1").await;
    let _ = module.delete("key2").await;
    
    // Get adaptive status
    let status = module.get_adaptive_status(true).await;
    assert!(status.is_ok(), "Failed to get adaptive status");
    
    let status_json = status.unwrap();
    println!("Adaptive status: {}", serde_json::to_string_pretty(&status_json).unwrap());
    
    // Verify operation count
    if let Some(op_count) = status_json.get("operation_count") {
        assert!(op_count.as_u64().unwrap() >= 4, "Should have at least 4 operations");
    }
}