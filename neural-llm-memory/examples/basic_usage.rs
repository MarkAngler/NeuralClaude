//! Basic usage example of the neural LLM memory framework

use neural_llm_memory::{
    MemoryModule, MemoryConfig, FrameworkConfig,
    nn::{NetworkBuilder, ActivationFunction},
};
use ndarray::Array2;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Neural LLM Memory Framework - Basic Usage Example\n");
    
    // Initialize configuration
    let config = MemoryConfig {
        memory_size: 1000,
        embedding_dim: 128,
        hidden_dim: 256,
        num_heads: 8,
        num_layers: 4,
        dropout_rate: 0.1,
        max_sequence_length: 100,
        use_positional_encoding: true,
    };
    
    println!("Initializing memory module with config:");
    println!("  - Memory size: {}", config.memory_size);
    println!("  - Embedding dim: {}", config.embedding_dim);
    println!("  - Hidden dim: {}", config.hidden_dim);
    println!("  - Attention heads: {}\n", config.num_heads);
    
    // Create memory module
    let mut memory_module = MemoryModule::new(config.clone());
    
    // Generate some example memories
    println!("Storing example memories...");
    let mut rng = rand::thread_rng();
    let mut stored_keys = Vec::new();
    
    for i in 0..10 {
        // Create random embedding
        let embedding = Array2::from_shape_fn((1, config.embedding_dim), |_| {
            rng.gen_range(-1.0..1.0)
        });
        
        // Store memory
        let content = format!("Memory entry {}: Important information about topic {}", i, i % 3);
        let key = memory_module.store_memory(content.clone(), embedding)?;
        stored_keys.push(key.clone());
        
        println!("  - Stored: {} (key: {})", content, &key.id[..8]);
    }
    
    println!("\nMemory storage complete!");
    
    // Test retrieval with attention
    println!("\nTesting memory retrieval with attention mechanism...");
    
    // Create a query embedding
    let query_embedding = Array2::from_shape_fn((1, config.embedding_dim), |_| {
        rng.gen_range(-1.0..1.0)
    });
    
    // Retrieve top 3 most relevant memories
    let retrieved = memory_module.retrieve_with_attention(&query_embedding, 3);
    
    println!("\nTop 3 retrieved memories:");
    for (i, (key, value, score)) in retrieved.iter().enumerate() {
        println!("  {}. {} (score: {:.4})", 
            i + 1, 
            &value.content[..50.min(value.content.len())], 
            score[[0, 0]]
        );
    }
    
    // Test memory consolidation
    println!("\nTesting memory consolidation...");
    let keys_to_consolidate = &stored_keys[0..3];
    let consolidated_key = memory_module.consolidate_memories(keys_to_consolidate)?;
    println!("  - Created consolidated memory with key: {}", &consolidated_key.id[..8]);
    
    // Get statistics
    let (size, total_accesses, cache_hits, hit_rate) = memory_module.get_stats();
    println!("\nMemory statistics:");
    println!("  - Total memories: {}", size);
    println!("  - Total accesses: {}", total_accesses);
    println!("  - Cache hits: {}", cache_hits);
    println!("  - Cache hit rate: {:.2}%", hit_rate * 100.0);
    
    // Demonstrate neural network usage
    println!("\n\nDemonstrating standalone neural network...");
    
    let mut nn = NetworkBuilder::new()
        .add_linear(128, 256, ActivationFunction::ReLU, true)
        .add_dropout(0.1)
        .add_linear(256, 256, ActivationFunction::ReLU, true)
        .add_dropout(0.1)
        .add_linear(256, 128, ActivationFunction::Identity, true)
        .build(0.001);
    
    // Forward pass
    let input = Array2::from_shape_fn((4, 128), |_| rng.gen_range(-1.0..1.0));
    let output = nn.predict(&input);
    
    println!("  - Input shape: {:?}", input.shape());
    println!("  - Output shape: {:?}", output.shape());
    println!("  - Output mean: {:.4}", output.mean().unwrap());
    println!("  - Output std: {:.4}", output.std(0.0));
    
    println!("\nExample completed successfully!");
    
    Ok(())
}