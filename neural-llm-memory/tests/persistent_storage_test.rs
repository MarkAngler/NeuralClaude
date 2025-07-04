//! Integration tests for persistent storage

use neural_llm_memory::{
    memory::{MemoryKey, MemoryValue, MemoryMetadata, PersistentStorageBackend, StorageBackend},
    storage::{JsonStorage, StorageConfig, PersistentStorage},
};
use tempfile::TempDir;
use std::thread;
use std::time::Duration;

#[test]
fn test_json_storage_persistence() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig::new(temp_dir.path());
    
    // Create and initialize storage
    let mut storage = JsonStorage::new(config);
    storage.init().unwrap();
    
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
    storage.save_one(&key1, &value1).unwrap();
    storage.save_one(&key2, &value2).unwrap();
    
    // Load and verify
    let loaded = storage.load_all().unwrap();
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

#[test]
fn test_persistent_storage_backend() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig::new(temp_dir.path())
        .with_auto_save(true);
    
    let mut storage = PersistentStorageBackend::new(config).unwrap();
    
    // Store some data
    let key = MemoryKey::new("test_key".to_string(), "test_context");
    let value = MemoryValue {
        embedding: vec![1.0, 2.0, 3.0, 4.0],
        content: "Test content".to_string(),
        metadata: MemoryMetadata::default(),
    };
    
    storage.save(&key, &value).unwrap();
    
    // Retrieve and verify
    let loaded = storage.load(&key).unwrap();
    assert!(loaded.is_some());
    assert_eq!(loaded.unwrap().content, "Test content");
    
    // List keys
    let keys = storage.list_keys();
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].id, "test_key");
    
    // Delete
    let deleted = storage.delete(&key).unwrap();
    assert!(deleted);
    
    // Verify deletion
    let loaded_after_delete = storage.load(&key).unwrap();
    assert!(loaded_after_delete.is_none());
}

#[test]
fn test_backup_and_restore() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig::new(temp_dir.path());
    
    let mut storage = JsonStorage::new(config);
    storage.init().unwrap();
    
    // Add some memories
    let memories = vec![
        (
            MemoryKey::new("mem1".to_string(), "ctx1"),
            MemoryValue {
                embedding: vec![1.0, 2.0],
                content: "Memory 1".to_string(),
                metadata: Default::default(),
            },
        ),
        (
            MemoryKey::new("mem2".to_string(), "ctx2"),
            MemoryValue {
                embedding: vec![3.0, 4.0],
                content: "Memory 2".to_string(),
                metadata: Default::default(),
            },
        ),
    ];
    
    storage.save_all(&memories).unwrap();
    
    // Create backup
    let backup_path = storage.backup().unwrap();
    assert!(std::path::Path::new(&backup_path).exists());
    
    // Modify data
    storage.delete_one(&memories[0].0).unwrap();
    
    // Restore from backup
    storage.restore(std::path::Path::new(&backup_path)).unwrap();
    
    // Verify restoration
    let restored = storage.load_all().unwrap();
    assert_eq!(restored.len(), 2);
}

#[test]
fn test_periodic_save() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig::new(temp_dir.path())
        .with_auto_save(false)
        .with_save_interval(1); // 1 second for testing
    
    let mut storage = PersistentStorageBackend::new(config).unwrap();
    storage.start_background_save();
    
    // Add data
    let key = MemoryKey::new("periodic_test".to_string(), "context");
    let value = MemoryValue {
        embedding: vec![1.0],
        content: "Periodic save test".to_string(),
        metadata: Default::default(),
    };
    
    storage.save(&key, &value).unwrap();
    
    // Wait for periodic save
    thread::sleep(Duration::from_secs(2));
    
    // Create new instance to verify persistence
    let storage2 = PersistentStorageBackend::new(
        StorageConfig::new(temp_dir.path())
    ).unwrap();
    
    let loaded = storage2.load(&key).unwrap();
    assert!(loaded.is_some());
    assert_eq!(loaded.unwrap().content, "Periodic save test");
    
    // Clean up
    storage.stop_background_save();
}

#[test]
fn test_wal_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let config = StorageConfig::new(temp_dir.path())
        .with_wal(true);
    
    let mut storage = JsonStorage::new(config.clone());
    storage.init().unwrap();
    
    // Save initial data
    let key1 = MemoryKey::new("wal1".to_string(), "ctx1");
    let value1 = MemoryValue {
        embedding: vec![1.0],
        content: "WAL test 1".to_string(),
        metadata: Default::default(),
    };
    
    storage.save_all(&[(key1.clone(), value1.clone())]).unwrap();
    
    // Add more data via WAL
    let key2 = MemoryKey::new("wal2".to_string(), "ctx2");
    let value2 = MemoryValue {
        embedding: vec![2.0],
        content: "WAL test 2".to_string(),
        metadata: Default::default(),
    };
    
    storage.save_one(&key2, &value2).unwrap();
    
    // Delete via WAL
    storage.delete_one(&key1).unwrap();
    
    // Create new instance (simulating crash recovery)
    let storage2 = JsonStorage::new(config);
    let recovered = storage2.load_all().unwrap();
    
    // Should have only key2 after WAL replay
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].0.id, "wal2");
    assert_eq!(recovered[0].1.content, "WAL test 2");
}