//! Comprehensive demo of persistent memory functionality

use neural_llm_memory::memory::{PersistentMemoryBuilder, PersistentMemoryModule};
use ndarray::Array2;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Neural Memory Framework - Persistent Memory Demo ===\n");
    
    let storage_path = "./demo_memory_storage";
    
    // Check if we have existing memories
    if Path::new(storage_path).exists() {
        println!("Found existing memory storage. Loading...");
        demo_load_existing(storage_path)?;
    } else {
        println!("No existing storage found. Creating new memory system...");
        demo_create_new(storage_path)?;
    }
    
    Ok(())
}

fn demo_create_new(storage_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Build a new persistent memory module with custom configuration
    let mut memory = PersistentMemoryBuilder::new()
        .with_storage_path(storage_path)
        .with_memory_size(5000)
        .with_embedding_dim(768)  // Must be divisible by num_heads (12)
        .with_auto_save(false)  // Use periodic saves
        .with_save_interval(60) // Save every minute
        .with_compression(false)
        .with_wal(true)        // Enable write-ahead logging
        .build()?;
    
    println!("Created new memory system with:");
    println!("  - Storage path: {}", storage_path);
    println!("  - Memory size: 5000");
    println!("  - Embedding dimension: 768");
    println!("  - Periodic saves every 60 seconds");
    println!("  - Write-ahead logging enabled\n");
    
    // Store various types of memories
    let memories = vec![
        // Facts
        ("The Earth orbits around the Sun", generate_embedding(768, 0.1)),
        ("Water freezes at 0 degrees Celsius", generate_embedding(768, 0.2)),
        ("The speed of light is 299,792,458 m/s", generate_embedding(768, 0.3)),
        
        // Personal information (simulated)
        ("User prefers dark mode in applications", generate_embedding(768, 0.4)),
        ("User's favorite programming language is Rust", generate_embedding(768, 0.5)),
        
        // Context from conversations
        ("In our last conversation, we discussed neural networks", generate_embedding(768, 0.6)),
        ("The user is interested in machine learning applications", generate_embedding(768, 0.7)),
        
        // Task-related memories
        ("TODO: Research quantum computing basics", generate_embedding(768, 0.8)),
        ("Completed: Implemented persistent storage for memory system", generate_embedding(768, 0.9)),
    ];
    
    println!("Storing {} memories...", memories.len());
    
    let mut stored_keys = Vec::new();
    for (i, (content, embedding)) in memories.into_iter().enumerate() {
        let embedding_array = Array2::from_shape_vec((1, 768), embedding)?;
        let key = memory.store_memory(content.to_string(), embedding_array)?;
        stored_keys.push(key.clone());
        println!("  [{}] Stored: {} (id: {})", i + 1, content, &key.id[..8]);
    }
    
    // Force save to ensure everything is persisted
    println!("\nSaving to disk...");
    memory.save_to_disk()?;
    
    // Create a backup
    println!("Creating backup...");
    let backup_path = memory.backup()?;
    println!("Backup saved to: {}", backup_path);
    
    // Demonstrate retrieval
    println!("\nTesting retrieval with semantic search...");
    let query = "Tell me about programming languages";
    let query_embedding = Array2::from_shape_vec((1, 768), generate_embedding(768, 0.55))?;
    
    let results = memory.retrieve_with_attention(&query_embedding, 3);
    println!("Query: '{}'\nTop 3 results:", query);
    for (i, (key, value, _score)) in results.iter().enumerate() {
        println!("  {}. {} (importance: {:.2})", 
                 i + 1, 
                 value.content, 
                 value.metadata.importance);
    }
    
    // Update a memory
    println!("\nUpdating a memory...");
    if let Some(key_to_update) = stored_keys.get(4) {
        let new_content = "User's favorite programming languages are Rust and Python".to_string();
        let new_embedding = Array2::from_shape_vec((1, 768), generate_embedding(768, 0.52))?;
        memory.update_memory(key_to_update, new_content.clone(), new_embedding)?;
        println!("Updated memory: {}", new_content);
    }
    
    // Get statistics
    let (size, total_accesses, cache_hits, hit_rate) = memory.get_stats();
    println!("\nMemory Statistics:");
    println!("  Total memories: {}", size);
    println!("  Total accesses: {}", total_accesses);
    println!("  Cache hits: {}", cache_hits);
    println!("  Cache hit rate: {:.2}%", hit_rate * 100.0);
    
    // Final save
    memory.save_to_disk()?;
    println!("\nMemory system created and saved successfully!");
    
    Ok(())
}

fn demo_load_existing(storage_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Load existing persistent memory
    let mut memory = PersistentMemoryModule::load(storage_path)?;
    
    let (size, _, _, _) = memory.get_stats();
    println!("Loaded {} memories from storage\n", size);
    
    // Test retrieval on loaded memories
    println!("Testing retrieval on loaded memories...");
    let query = "What do we know about science and physics?";
    let query_embedding = Array2::from_shape_vec((1, 768), generate_embedding(768, 0.25))?;
    
    let results = memory.retrieve_with_attention(&query_embedding, 5);
    println!("Query: '{}'\nTop {} results:", query, results.len());
    for (i, (key, value, _score)) in results.iter().enumerate() {
        println!("  {}. {} (accessed {} times)", 
                 i + 1, 
                 value.content,
                 value.metadata.access_count);
    }
    
    // Add a new memory to existing system
    println!("\nAdding new memory to existing system...");
    let new_memory = "Quantum entanglement is a phenomenon in quantum physics";
    let embedding = Array2::from_shape_vec((1, 768), generate_embedding(768, 0.35))?;
    let key = memory.store_memory(new_memory.to_string(), embedding)?;
    println!("Added: {} (id: {})", new_memory, &key.id[..8]);
    
    // Save changes
    memory.save_to_disk()?;
    
    // Demonstrate backup and restore
    println!("\nCreating backup of current state...");
    let backup_path = memory.backup()?;
    println!("Backup created: {}", backup_path);
    
    // Get final statistics
    let (size, total_accesses, cache_hits, hit_rate) = memory.get_stats();
    println!("\nFinal Memory Statistics:");
    println!("  Total memories: {}", size);
    println!("  Total accesses: {}", total_accesses);
    println!("  Cache hits: {}", cache_hits);
    println!("  Cache hit rate: {:.2}%", hit_rate * 100.0);
    
    Ok(())
}

/// Generate a simple embedding vector for demonstration
fn generate_embedding(dim: usize, seed: f32) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let angle = i as f32 * seed;
            (angle.sin() + angle.cos()) * 0.5
        })
        .collect()
}