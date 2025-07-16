//! Integration tests for persistent storage

use neural_llm_memory::{
    memory::{MemoryKey, MemoryValue, MemoryMetadata},
    storage::{FileStorage, StorageBackend},
};
use tempfile::TempDir;

#[tokio::test]
async fn test_file_storage_basic_operations() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create and initialize storage
    let storage = FileStorage::new(temp_dir.path()).await.unwrap();
    
    // Create test data
    let key1 = MemoryKey::new("memory1".to_string(), "context1");
    let value1 = MemoryValue {
        embedding: vec![1.0, 2.0, 3.0],
        content: "First memory".to_string(),
        metadata: MemoryMetadata::default(),
    };
    
    let key2 = MemoryKey::new("memory2".to_string(), "context2");
    let value2 = MemoryValue {
        embedding: vec![4.0, 5.0, 6.0],
        content: "Second memory".to_string(),
        metadata: MemoryMetadata {
            importance: 2.0,
            ..Default::default()
        },
    };
    
    // Save memories
    storage.save(&key1, &value1).await.unwrap();
    storage.save(&key2, &value2).await.unwrap();
    
    // Load and verify
    let loaded = storage.load_all().await.unwrap();
    assert_eq!(loaded.len(), 2);
    
    // Find our memories
    let found1 = loaded.iter().find(|(k, _)| k.id == "memory1");
    let found2 = loaded.iter().find(|(k, _)| k.id == "memory2");
    
    assert!(found1.is_some());
    assert!(found2.is_some());
    
    assert_eq!(found1.unwrap().1.content, "First memory");
    assert_eq!(found2.unwrap().1.content, "Second memory");
    assert_eq!(found2.unwrap().1.metadata.importance, 2.0);
}

#[tokio::test]
async fn test_file_storage_delete() {
    let temp_dir = TempDir::new().unwrap();
    let storage = FileStorage::new(temp_dir.path()).await.unwrap();
    
    // Store some data
    let key = MemoryKey::new("test_key".to_string(), "test_context");
    let value = MemoryValue {
        embedding: vec![1.0, 2.0, 3.0, 4.0],
        content: "Test content".to_string(),
        metadata: MemoryMetadata::default(),
    };
    
    storage.save(&key, &value).await.unwrap();
    
    // Verify it was saved
    let loaded = storage.load(&key).await.unwrap();
    assert!(loaded.is_some());
    assert_eq!(loaded.unwrap().content, "Test content");
    
    // Delete
    storage.delete(&key).await.unwrap();
    
    // Verify deletion
    let loaded_after_delete = storage.load(&key).await.unwrap();
    assert!(loaded_after_delete.is_none());
}

#[tokio::test]
async fn test_file_storage_batch_operations() {
    let temp_dir = TempDir::new().unwrap();
    let storage = FileStorage::new(temp_dir.path()).await.unwrap();
    
    // Create multiple memories
    let memories: Vec<(MemoryKey, MemoryValue)> = (0..5)
        .map(|i| {
            (
                MemoryKey::new(format!("batch_key_{}", i), &format!("context_{}", i)),
                MemoryValue {
                    embedding: vec![i as f32; 3],
                    content: format!("Batch memory {}", i),
                    metadata: MemoryMetadata::default(),
                },
            )
        })
        .collect();
    
    // Save all at once
    storage.save_batch(&memories).await.unwrap();
    
    // Load all and verify
    let loaded = storage.load_all().await.unwrap();
    assert_eq!(loaded.len(), 5);
    
    // Verify each memory
    for i in 0..5 {
        let found = loaded.iter().find(|(k, _)| k.id == format!("batch_key_{}", i));
        assert!(found.is_some());
        assert_eq!(found.unwrap().1.content, format!("Batch memory {}", i));
    }
}

#[tokio::test]
async fn test_file_storage_persistence_across_instances() {
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().to_path_buf();
    
    // First instance - save data
    {
        let storage = FileStorage::new(&storage_path).await.unwrap();
        
        let key = MemoryKey::new("persistent_key".to_string(), "context");
        let value = MemoryValue {
            embedding: vec![1.0, 2.0, 3.0],
            content: "Persistent content".to_string(),
            metadata: MemoryMetadata::default(),
        };
        
        storage.save(&key, &value).await.unwrap();
    }
    
    // Second instance - load data
    {
        let storage = FileStorage::new(&storage_path).await.unwrap();
        
        let loaded = storage.load_all().await.unwrap();
        assert_eq!(loaded.len(), 1);
        
        let (loaded_key, loaded_value) = &loaded[0];
        assert_eq!(loaded_key.id, "persistent_key");
        assert_eq!(loaded_value.content, "Persistent content");
    }
}

#[tokio::test]
async fn test_file_storage_stats() {
    let temp_dir = TempDir::new().unwrap();
    let storage = FileStorage::new(temp_dir.path()).await.unwrap();
    
    // Get initial stats
    let initial_stats = storage.stats().await.unwrap();
    assert_eq!(initial_stats.total_memories, 0);
    
    // Add some memories
    for i in 0..3 {
        let key = MemoryKey::new(format!("stats_key_{}", i), "context");
        let value = MemoryValue {
            embedding: vec![i as f32; 10],
            content: format!("Stats test memory {}", i),
            metadata: MemoryMetadata::default(),
        };
        storage.save(&key, &value).await.unwrap();
    }
    
    // Get updated stats
    let updated_stats = storage.stats().await.unwrap();
    assert_eq!(updated_stats.total_memories, 3);
    assert!(updated_stats.total_size_bytes > 0);
    assert!(updated_stats.last_save.is_some());
}

#[tokio::test]
async fn test_file_storage_clear() {
    let temp_dir = TempDir::new().unwrap();
    let storage = FileStorage::new(temp_dir.path()).await.unwrap();
    
    // Add some memories
    for i in 0..5 {
        let key = MemoryKey::new(format!("clear_key_{}", i), "context");
        let value = MemoryValue {
            embedding: vec![i as f32; 3],
            content: format!("Clear test memory {}", i),
            metadata: MemoryMetadata::default(),
        };
        storage.save(&key, &value).await.unwrap();
    }
    
    // Verify they were saved
    let loaded = storage.load_all().await.unwrap();
    assert_eq!(loaded.len(), 5);
    
    // Clear all
    storage.clear().await.unwrap();
    
    // Verify all were cleared
    let loaded_after_clear = storage.load_all().await.unwrap();
    assert_eq!(loaded_after_clear.len(), 0);
}