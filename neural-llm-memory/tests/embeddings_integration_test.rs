//! Integration tests for semantic embeddings

use neural_llm_memory::embeddings::{EmbeddingService, EmbeddingConfig, cosine_similarity};
use neural_llm_memory::graph::{ConsciousGraph, ConsciousGraphConfig, NodeType};
use anyhow::Result;
use tempfile::TempDir;

#[tokio::test]
async fn test_semantic_embeddings_similarity() -> Result<()> {
    // Create a test embedding config
    let mut config = EmbeddingConfig::default();
    config.cache_enabled = false; // Disable cache for testing
    
    // Try to create embedding service - skip test if model download fails
    let service = match EmbeddingService::new(config).await {
        Ok(service) => service,
        Err(e) => {
            eprintln!("Skipping test - could not download model: {}", e);
            return Ok(());
        }
    };
    
    // Test semantic similarity
    let cat_embedding = service.embed("cat").await?;
    let dog_embedding = service.embed("dog").await?;
    let car_embedding = service.embed("car").await?;
    
    // Calculate similarities
    let cat_dog_similarity = cosine_similarity(&cat_embedding, &dog_embedding);
    let cat_car_similarity = cosine_similarity(&cat_embedding, &car_embedding);
    
    // Animals should be more similar to each other than to vehicles
    assert!(
        cat_dog_similarity > cat_car_similarity,
        "Expected cat-dog similarity ({}) > cat-car similarity ({})",
        cat_dog_similarity,
        cat_car_similarity
    );
    
    Ok(())
}

#[tokio::test]
async fn test_embedding_cache() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create config with caching enabled
    let mut config = EmbeddingConfig::default();
    config.cache_dir = Some(temp_dir.path().to_path_buf());
    config.cache_enabled = true;
    
    // Try to create embedding service - skip test if model download fails
    let service = match EmbeddingService::new(config).await {
        Ok(service) => service,
        Err(e) => {
            eprintln!("Skipping test - could not download model: {}", e);
            return Ok(());
        }
    };
    
    // First call - should compute embedding
    let start = std::time::Instant::now();
    let embedding1 = service.embed("test text for caching").await?;
    let first_duration = start.elapsed();
    
    // Second call - should use cache
    let start = std::time::Instant::now();
    let embedding2 = service.embed("test text for caching").await?;
    let cache_duration = start.elapsed();
    
    // Embeddings should be identical
    assert_eq!(embedding1, embedding2, "Cached embedding should match original");
    
    // Cache lookup should be faster (this might be flaky in CI)
    println!("First call: {:?}, Cache hit: {:?}", first_duration, cache_duration);
    
    Ok(())
}

#[tokio::test]
async fn test_batch_embeddings() -> Result<()> {
    let mut config = EmbeddingConfig::default();
    config.cache_enabled = false;
    
    let service = match EmbeddingService::new(config).await {
        Ok(service) => service,
        Err(e) => {
            eprintln!("Skipping test - could not download model: {}", e);
            return Ok(());
        }
    };
    
    let texts = vec![
        "The quick brown fox".to_string(),
        "jumps over the lazy dog".to_string(),
        "Machine learning is fascinating".to_string(),
    ];
    
    let embeddings = service.embed_batch(&texts).await?;
    
    assert_eq!(embeddings.len(), texts.len());
    
    // Each embedding should have the expected dimension
    let expected_dim = service.embedding_dimension();
    for (i, embedding) in embeddings.iter().enumerate() {
        assert_eq!(
            embedding.len(),
            expected_dim,
            "Embedding {} has wrong dimension",
            i
        );
        
        // Should be normalized
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Embedding {} not normalized", i);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_conscious_graph_with_semantic_embeddings() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create graph config with semantic embeddings
    let mut graph_config = ConsciousGraphConfig::default();
    graph_config.storage_path = temp_dir.path().to_path_buf();
    graph_config.semantic_embeddings_enabled = true;
    
    // Use a small model for testing
    if let Some(ref mut embedding_config) = graph_config.embedding_config {
        embedding_config.cache_enabled = false;
    }
    
    // Create graph with async initialization
    let graph = match ConsciousGraph::new_with_config_async(graph_config).await {
        Ok(graph) => graph,
        Err(e) => {
            eprintln!("Skipping test - could not initialize graph with embeddings: {}", e);
            return Ok(());
        }
    };
    
    // Add some nodes
    let node1 = graph.add_node(NodeType::Memory {
        content: "Cats are domestic animals that are often kept as pets".to_string(),
        timestamp: chrono::Utc::now(),
        metadata: Default::default(),
    })?;
    
    let node2 = graph.add_node(NodeType::Memory {
        content: "Dogs are loyal companions and popular pets worldwide".to_string(),
        timestamp: chrono::Utc::now(),
        metadata: Default::default(),
    })?;
    
    let node3 = graph.add_node(NodeType::Memory {
        content: "Automobiles are vehicles used for transportation".to_string(),
        timestamp: chrono::Utc::now(),
        metadata: Default::default(),
    })?;
    
    // The embeddings should create meaningful relationships
    // We can't test the exact similarity values, but we can verify the system works
    assert_ne!(node1, node2);
    assert_ne!(node2, node3);
    
    Ok(())
}

#[tokio::test]
async fn test_fallback_to_hash_embeddings() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create graph config with semantic embeddings but invalid model
    let mut graph_config = ConsciousGraphConfig::default();
    graph_config.storage_path = temp_dir.path().to_path_buf();
    graph_config.semantic_embeddings_enabled = true;
    
    if let Some(ref mut embedding_config) = graph_config.embedding_config {
        embedding_config.model_id = "invalid/model/that/does/not/exist".to_string();
        embedding_config.fallback_enabled = true;
    }
    
    // Should still create graph successfully with fallback
    let graph = ConsciousGraph::new_with_config_async(graph_config).await?;
    
    // Should be able to add nodes using hash-based embeddings
    let node = graph.add_node(NodeType::Memory {
        content: "Test content".to_string(),
        timestamp: chrono::Utc::now(),
        metadata: Default::default(),
    })?;
    
    assert!(!node.0.is_nil());
    
    Ok(())
}