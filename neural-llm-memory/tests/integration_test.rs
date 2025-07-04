//! Integration tests for the neural LLM memory framework

use neural_llm_memory::{
    MemoryModule, MemoryConfig,
    nn::{NetworkBuilder, ActivationFunction},
};
use ndarray::Array2;

#[test]
fn test_memory_storage_and_retrieval() {
    let config = MemoryConfig {
        memory_size: 100,
        embedding_dim: 64,
        hidden_dim: 128,
        num_heads: 4,
        num_layers: 2,
        dropout_rate: 0.1,
        max_sequence_length: 50,
        use_positional_encoding: true,
    };
    
    let mut memory_module = MemoryModule::new(config.clone());
    
    // Store a memory
    let embedding = Array2::ones((1, config.embedding_dim));
    let key = memory_module.store_memory(
        "Test memory content".to_string(),
        embedding.clone()
    ).unwrap();
    
    // Retrieve memories
    let query = Array2::ones((1, config.embedding_dim));
    let retrieved = memory_module.retrieve_with_attention(&query, 1);
    
    assert_eq!(retrieved.len(), 1);
    assert_eq!(retrieved[0].1.content, "Test memory content");
}

#[test]
fn test_neural_network_forward_pass() {
    let nn = NetworkBuilder::new()
        .add_linear(64, 128, ActivationFunction::ReLU, true)
        .add_dropout(0.1)
        .add_linear(128, 64, ActivationFunction::Identity, true)
        .build(0.001);
    
    let input = Array2::zeros((2, 64));
    let output = nn.predict(&input);
    
    assert_eq!(output.shape(), &[2, 64]);
}

#[test]
fn test_memory_consolidation() {
    let config = MemoryConfig {
        memory_size: 100,
        embedding_dim: 32,
        hidden_dim: 64,
        num_heads: 4,
        num_layers: 2,
        dropout_rate: 0.0,
        max_sequence_length: 50,
        use_positional_encoding: false,
    };
    
    let mut memory_module = MemoryModule::new(config.clone());
    
    // Store multiple memories
    let mut keys = Vec::new();
    for i in 0..3 {
        let embedding = Array2::ones((1, config.embedding_dim)) * (i as f32 + 1.0);
        let key = memory_module.store_memory(
            format!("Memory {}", i),
            embedding
        ).unwrap();
        keys.push(key);
    }
    
    // Consolidate memories
    let consolidated_key = memory_module.consolidate_memories(&keys).unwrap();
    
    // Check consolidated memory exists
    let (size, _, _, _) = memory_module.get_stats();
    assert_eq!(size, 4); // 3 original + 1 consolidated
}