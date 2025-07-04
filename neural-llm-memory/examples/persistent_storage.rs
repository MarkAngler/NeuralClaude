//! Example demonstrating persistent storage for neural memory system

use neural_llm_memory::{
    memory::{MemoryModule, MemoryConfig, PersistentStorageBackend},
    storage::StorageConfig,
};
use ndarray::Array2;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Neural Memory Framework - Persistent Storage Example\n");
    
    // Configure storage
    let storage_config = StorageConfig::new("./memory_storage")
        .with_auto_save(false) // We'll use periodic saves instead
        .with_save_interval(30) // Save every 30 seconds
        .with_compression(false)
        .with_wal(true); // Enable write-ahead logging for crash recovery
    
    // Create persistent storage backend
    let mut storage = PersistentStorageBackend::new(storage_config)?;
    
    // Start background save thread
    storage.start_background_save();
    
    // Configure memory module
    let config = MemoryConfig {
        memory_size: 1000,
        embedding_dim: 384,
        hidden_dim: 768,
        num_heads: 6,
        num_layers: 4,
        dropout_rate: 0.1,
        max_sequence_length: 256,
        use_positional_encoding: true,
    };
    
    let mut memory_module = MemoryModule::new(config);
    
    println!("Storing memories...");
    
    // Store some memories
    let memories = vec![
        ("The capital of France is Paris", vec![0.1; 384]),
        ("Machine learning is a subset of AI", vec![0.2; 384]),
        ("Rust is a systems programming language", vec![0.3; 384]),
        ("Neural networks are inspired by the brain", vec![0.4; 384]),
    ];
    
    for (content, embedding) in memories {
        let embedding_array = Array2::from_shape_vec((1, 384), embedding)?;
        let key = memory_module.store_memory(content.to_string(), embedding_array)?;
        println!("Stored: {} with key: {}", content, key.id);
    }
    
    // Get memory statistics
    let (size, total_accesses, cache_hits, hit_rate) = memory_module.get_stats();
    println!("\nMemory Statistics:");
    println!("  Total memories: {}", size);
    println!("  Total accesses: {}", total_accesses);
    println!("  Cache hits: {}", cache_hits);
    println!("  Cache hit rate: {:.2}%", hit_rate * 100.0);
    
    // Force save to disk
    println!("\nForcing save to disk...");
    storage.force_save()?;
    
    // Create a backup
    println!("Creating backup...");
    let backup_path = storage.backup()?;
    println!("Backup created at: {}", backup_path);
    
    // Simulate retrieval
    println!("\nRetrieving memories...");
    let query = Array2::from_elem((1, 384), 0.25);
    let results = memory_module.retrieve_with_attention(&query, 3);
    
    for (i, (key, value, _score)) in results.iter().enumerate() {
        println!("  {}. {} (id: {})", i + 1, value.content, key.id);
    }
    
    // Demonstrate restoration from backup
    println!("\nSimulating system restart...");
    println!("Loading from persistent storage...");
    
    // Create new storage instance (simulating restart)
    let storage_config2 = StorageConfig::new("./memory_storage");
    let storage2 = PersistentStorageBackend::new(storage_config2)?;
    
    // Check that memories persist
    let keys = storage2.list_keys();
    println!("Found {} memories in persistent storage", keys.len());
    
    for key in keys.iter().take(3) {
        if let Ok(Some(value)) = storage2.load(key) {
            println!("  Loaded: {}", value.content);
        }
    }
    
    // Clean up (optional)
    drop(storage);
    
    println!("\nPersistent storage example completed!");
    
    Ok(())
}