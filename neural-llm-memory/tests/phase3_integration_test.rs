//! Phase 3 Integration Tests
//! 
//! Comprehensive integration tests for:
//! - CrossModalBridge with HybridMemoryBank
//! - DreamConsolidation background processing
//! - Consciousness-weighted attention
//! - Performance benchmarks

use neural_llm_memory::{
    graph::{
        HybridMemoryBank, CrossModalBridge, DreamConsolidation,
        CrossModalConfig, DreamConfig, MemoryModality,
        ConsciousGraph, ConsciousGraphConfig, GraphStorage,
        ConsciousNode, NodeType, EdgeType,
    },
    consciousness::{ConsciousnessCore, ConsciousnessConfig, ConsciousnessLevel},
    emotional::{EmotionalState, EmotionalIntensity},
    memory::{MemoryConfig, MemoryValue, MemoryKey, MemoryBank, MemoryOperations},
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use ndarray::{Array1, Array2};
use parking_lot::RwLock;
use uuid::Uuid;

/// Test cross-modal bridge with hybrid memory bank integration
#[tokio::test]
async fn test_cross_modal_hybrid_integration() {
    // Create hybrid memory bank
    let memory_config = MemoryConfig::default();
    let graph_config = ConsciousGraphConfig::default();
    let hybrid_bank = Arc::new(HybridMemoryBank::new(memory_config, graph_config).await.unwrap());
    
    // Create cross-modal bridge
    let modal_config = CrossModalConfig {
        modalities: vec![
            MemoryModality::Semantic,
            MemoryModality::Episodic,
            MemoryModality::Emotional,
        ],
        embedding_dim: 768,
        translation_layers: 2,
        enable_learning: true,
        bridge_learning_rate: 0.01,
    };
    let bridge = Arc::new(RwLock::new(
        CrossModalBridge::new(modal_config).unwrap()
    ));
    
    // Create test memories in different modalities
    let semantic_key = "test_semantic_fact";
    let semantic_value = MemoryValue {
        embedding: vec![1.0; 768],
        content: "Semantic fact content".to_string(),
        metadata: Default::default(),
    };
    
    let episodic_key = "test_episodic_event";
    let episodic_value = MemoryValue {
        embedding: vec![0.5; 768],
        content: "Episodic event content".to_string(),
        metadata: Default::default(),
    };
    
    // Store memories
    hybrid_bank.store(semantic_key.to_string(), semantic_value.clone()).await.unwrap();
    hybrid_bank.store(episodic_key.to_string(), episodic_value.clone()).await.unwrap();
    
    // Create cross-modal connections
    let semantic_node_id = hybrid_bank.get_node_id(&semantic_key.to_string()).await.unwrap();
    let episodic_node_id = hybrid_bank.get_node_id(&episodic_key.to_string()).await.unwrap();
    
    // Bridge memories across modalities
    let connection = bridge.write().connect_modalities(
        semantic_node_id.clone(),
        MemoryModality::Semantic,
        episodic_node_id.clone(),
        MemoryModality::Episodic,
        0.8
    ).unwrap();
    
    // Verify connection was created
    assert!(connection.strength.base_strength > 0.7);
    assert_eq!(connection.source_modality, MemoryModality::Semantic);
    assert_eq!(connection.target_modality, MemoryModality::Episodic);
    
    // Test cross-modal query
    let query_embedding = Array1::from_vec(vec![0.75; 768]);
    let results = bridge.read().cross_modal_query(
        &query_embedding,
        MemoryModality::Semantic,
        &[MemoryModality::Episodic, MemoryModality::Emotional],
        5
    ).unwrap();
    
    assert!(!results.is_empty());
    assert!(results[0].similarity > 0.5);
}

/// Test dream consolidation background processing
#[tokio::test]
async fn test_dream_consolidation_processing() {
    // Create conscious graph with config to ensure proper setup
    let config = ConsciousGraphConfig {
        embedding_dim: 768,
        auto_infer_relationships: true,
        consciousness_threshold: 0.5,
        max_nodes: 10000,
        persistence_enabled: true,
        storage_path: std::path::PathBuf::from("./test_dream_consolidation"),
    };
    let graph = Arc::new(ConsciousGraph::new_with_config(config).unwrap());
    
    // Create dream consolidation
    let dream_config = DreamConfig {
        consolidation_interval: 1, // 1 second for testing
        insight_confidence_threshold: 0.6,
        analysis_window_hours: 1,
        max_insights_per_cycle: 5,
        enable_temporal_reorg: true,
        idle_activity_threshold: 0.3,
    };
    
    let dream = Arc::new(DreamConsolidation::new(
        Arc::clone(&graph),
        dream_config
    ));
    
    // Start consolidation
    dream.start_consolidation().await.unwrap();
    
    // Wait for at least one consolidation cycle
    sleep(Duration::from_secs(2)).await;
    
    // Get consolidation stats
    let stats = dream.get_stats().await;
    assert!(stats.total_consolidations > 0);
    assert!(stats.last_consolidation.is_some());
    
    // Check for generated insights
    let insights = dream.get_recent_insights(10).await.unwrap();
    assert!(!insights.is_empty(), "Should have generated some insights");
    
    // Verify insight quality
    for insight in insights {
        assert!(insight.confidence >= 0.6);
        assert!(!insight.description.is_empty());
        assert!(!insight.related_memories.is_empty());
    }
    
    // Stop consolidation
    dream.stop_consolidation().await.unwrap();
}

/// Test consciousness-weighted attention in memory operations
#[tokio::test]
async fn test_consciousness_weighted_attention() {
    // Create consciousness core
    let consciousness_config = ConsciousnessConfig::default();
    let consciousness = Arc::new(RwLock::new(
        ConsciousnessCore::new(consciousness_config).unwrap()
    ));
    
    // Create hybrid memory bank
    let memory_config = MemoryConfig::default();
    let graph_config = ConsciousGraphConfig::default();
    let hybrid_bank = Arc::new(HybridMemoryBank::new(memory_config, graph_config).await.unwrap());
    
    // Test memories with different consciousness levels
    let test_cases = vec![
        ("high_conscious_memory", ConsciousnessLevel::HighlyConscious, 0.9),
        ("normal_conscious_memory", ConsciousnessLevel::Conscious, 0.7),
        ("low_conscious_memory", ConsciousnessLevel::Subconscious, 0.3),
    ];
    
    for (key, level, expected_weight) in test_cases {
        // Set consciousness level
        consciousness.write().unwrap().set_consciousness_level(level);
        
        // Create memory with consciousness-influenced embedding
        let base_embedding = Array1::from_vec(vec![0.5; 768]);
        let consciousness_weight = consciousness.read().unwrap().get_attention_weight();
        let weighted_embedding = &base_embedding * consciousness_weight;
        
        let value = MemoryValue {
            embedding: weighted_embedding.to_vec(),
            content: format!("Memory content for {}", key),
            metadata: Default::default(),
        };
        
        // Store memory
        hybrid_bank.store(key.to_string(), value).await.unwrap();
        
        // Verify consciousness influence
        let retrieved = hybrid_bank.retrieve(&key.to_string()).await.unwrap();
        let mean_activation: f32 = retrieved.embedding.iter().sum::<f32>() / retrieved.embedding.len() as f32;
        
        // Higher consciousness should result in stronger embeddings
        assert!(
            (mean_activation - (0.5 * expected_weight)).abs() < 0.1,
            "Consciousness level {} should influence embedding strength",
            level
        );
    }
    
    // Test consciousness-aware search
    let query = Array1::from_vec(vec![0.6; 768]);
    consciousness.write().unwrap().set_consciousness_level(ConsciousnessLevel::HighlyConscious);
    
    let results = hybrid_bank.search(&query, 5).await.unwrap();
    
    // Highly conscious queries should prefer highly conscious memories
    assert!(!results.is_empty());
    assert!(results[0].0.contains("high_conscious_memory"));
}

/// Performance benchmark: verify <10% latency increase
#[tokio::test]
async fn test_performance_latency_benchmark() {
    // Create baseline memory bank
    let mut baseline_bank = neural_llm_memory::memory::MemoryBank::new(10000, 100);
    
    // Create enhanced system with all Phase 3 features
    let memory_config = MemoryConfig::default();
    let graph_config = ConsciousGraphConfig::default();
    let hybrid_bank = Arc::new(HybridMemoryBank::new(memory_config, graph_config).await.unwrap());
    
    let modal_config = CrossModalConfig::default();
    let bridge = Arc::new(RwLock::new(
        CrossModalBridge::new(modal_config).unwrap()
    ));
    
    // Prepare test data
    let num_operations = 1000;
    let test_data: Vec<(String, MemoryValue)> = (0..num_operations)
        .map(|i| {
            let key = format!("perf_test_{}", i);
            let value = MemoryValue {
                embedding: vec![i as f32 / 1000.0; 768],
                content: format!("Performance test content {}", i),
                metadata: Default::default(),
            };
            (key, value)
        })
        .collect();
    
    // Benchmark baseline operations
    let baseline_start = Instant::now();
    for (key, value) in &test_data {
        let memory_key = MemoryKey {
            id: key.clone(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            context_hash: 0,
        };
        baseline_bank.store(memory_key, value.clone()).unwrap();
    }
    let baseline_duration = baseline_start.elapsed();
    
    // Benchmark enhanced operations
    let enhanced_start = Instant::now();
    for (key, value) in &test_data {
        hybrid_bank.store(key.clone(), value.clone()).await.unwrap();
    }
    let enhanced_duration = enhanced_start.elapsed();
    
    // Calculate latency increase
    let latency_increase = (enhanced_duration.as_millis() as f64 - baseline_duration.as_millis() as f64) 
        / baseline_duration.as_millis() as f64;
    
    println!("Baseline duration: {:?}", baseline_duration);
    println!("Enhanced duration: {:?}", enhanced_duration);
    println!("Latency increase: {:.2}%", latency_increase * 100.0);
    
    // Verify <10% latency increase
    assert!(
        latency_increase < 0.10,
        "Latency increase {:.2}% exceeds 10% threshold",
        latency_increase * 100.0
    );
}

/// Test cross-modal queries with emotional memory consolidation
#[tokio::test]
async fn test_emotional_cross_modal_consolidation() {
    // Create system components
    let memory_config = MemoryConfig::default();
    let graph_config = ConsciousGraphConfig::default();
    let hybrid_bank = Arc::new(HybridMemoryBank::new(memory_config, graph_config).await.unwrap());
    
    let modal_config = CrossModalConfig::default();
    let bridge = Arc::new(RwLock::new(
        CrossModalBridge::new(modal_config).unwrap()
    ));
    
    // Create emotional memories
    let emotional_memories = vec![
        ("happy_moment", EmotionalState::Joy, EmotionalIntensity::High),
        ("sad_event", EmotionalState::Sadness, EmotionalIntensity::Medium),
        ("exciting_discovery", EmotionalState::Excitement, EmotionalIntensity::VeryHigh),
    ];
    
    for (key, state, intensity) in emotional_memories {
        let mut embedding = Array1::from_vec(vec![0.5; 768]);
        
        // Modify embedding based on emotional state
        let intensity_factor = match intensity {
            EmotionalIntensity::Low => 0.3,
            EmotionalIntensity::Medium => 0.5,
            EmotionalIntensity::High => 0.7,
            EmotionalIntensity::VeryHigh => 0.9,
        };
        embedding *= intensity_factor;
        
        let value = MemoryValue {
            embedding: embedding.to_vec(),
            content: format!("Emotional memory: {}", key),
            metadata: Default::default(),
        };
        
        hybrid_bank.store(key.to_string(), value).await.unwrap();
    }
    
    // Create cross-modal connections between emotional and semantic memories
    let semantic_key = "learned_fact";
    let semantic_value = MemoryValue {
        embedding: vec![0.6; 768],
        content: "A learned fact".to_string(),
        metadata: Default::default(),
    };
    hybrid_bank.store(semantic_key.to_string(), semantic_value).await.unwrap();
    
    // Connect emotional memories to semantic memory
    let semantic_id = hybrid_bank.get_node_id(&semantic_key.to_string()).await.unwrap();
    let happy_id = hybrid_bank.get_node_id(&"happy_moment".to_string()).await.unwrap();
    
    bridge.write().connect_modalities(
        happy_id,
        MemoryModality::Emotional,
        semantic_id,
        MemoryModality::Semantic,
        0.75
    ).unwrap();
    
    // Test emotional influence on cross-modal queries
    let emotional_query = Array1::from_vec(vec![0.7; 768]);
    let results = bridge.read().cross_modal_query(
        &emotional_query,
        MemoryModality::Emotional,
        &[MemoryModality::Semantic],
        5
    ).unwrap();
    
    assert!(!results.is_empty());
    assert!(results.iter().any(|r| r.node_id.to_string() == semantic_id.to_string()));
}

/// Test thread safety and concurrent access
#[tokio::test]
async fn test_concurrent_access_safety() {
    let memory_config = MemoryConfig::default();
    let graph_config = ConsciousGraphConfig::default();
    let hybrid_bank = Arc::new(HybridMemoryBank::new(memory_config, graph_config).await.unwrap());
    
    let dream_config = DreamConfig {
        consolidation_interval: 1,
        ..Default::default()
    };
    // Create a separate ConsciousGraph for DreamConsolidation
    let graph_for_dream = Arc::new(ConsciousGraph::new().unwrap());
    let dream = Arc::new(DreamConsolidation::new(graph_for_dream, dream_config));
    
    // Start background consolidation
    dream.start_consolidation().await.unwrap();
    
    // Spawn multiple concurrent tasks
    let mut handles = vec![];
    
    // Writers
    for i in 0..5 {
        let bank = Arc::clone(&hybrid_bank);
        let handle = tokio::spawn(async move {
            for j in 0..100 {
                let key = format!("concurrent_write_{}_{}", i, j);
                let value = MemoryValue {
                    embedding: vec![i as f32; 768],
                    content: format!("Concurrent write content {} {}", i, j),
                    metadata: Default::default(),
                };
                bank.store(key, value).await.unwrap();
            }
        });
        handles.push(handle);
    }
    
    // Readers
    for i in 0..5 {
        let bank = Arc::clone(&hybrid_bank);
        let handle = tokio::spawn(async move {
            for j in 0..100 {
                let query = Array1::from_vec(vec![i as f32; 768]);
                let _ = bank.search(&query, 5).await;
            }
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Stop consolidation
    dream.stop_consolidation().await.unwrap();
    
    // Verify data integrity
    let stats = hybrid_bank.get_stats().await.unwrap();
    assert!(stats.node_count >= 500); // At least 500 nodes created
    assert!(stats.graph_enabled);
}

/// Test complete workflow integration
#[tokio::test]
async fn test_complete_phase3_workflow() {
    // Initialize all components
    let memory_config = MemoryConfig::default();
    let graph_config = ConsciousGraphConfig::default();
    let hybrid_bank = Arc::new(HybridMemoryBank::new(memory_config, graph_config).await.unwrap());
    
    let modal_config = CrossModalConfig::default();
    let bridge = Arc::new(RwLock::new(CrossModalBridge::new(modal_config).unwrap()));
    
    let consciousness_config = ConsciousnessConfig::default();
    let consciousness = Arc::new(RwLock::new(ConsciousnessCore::new(consciousness_config).unwrap()));
    
    let dream_config = DreamConfig::default();
    // Create a new ConsciousGraph for DreamConsolidation
    let graph_for_dream = Arc::new(ConsciousGraph::new().unwrap());
    let dream = Arc::new(DreamConsolidation::new(graph_for_dream, dream_config));
    
    // Step 1: Store diverse memories
    let memories = vec![
        ("fact_1", MemoryModality::Semantic, Array1::from_vec(vec![0.8; 768])),
        ("experience_1", MemoryModality::Episodic, Array1::from_vec(vec![0.6; 768])),
        ("feeling_1", MemoryModality::Emotional, Array1::from_vec(vec![0.9; 768])),
        ("skill_1", MemoryModality::Procedural, Array1::from_vec(vec![0.7; 768])),
    ];
    
    for (key, modality, embedding) in memories {
        let value = MemoryValue {
            embedding: embedding.to_vec(),
            content: format!("Content for {} - {:?}", key, modality),
            metadata: Default::default(),
        };
        hybrid_bank.store(key.to_string(), value).await.unwrap();
    }
    
    // Step 2: Create cross-modal connections
    let node_ids: Vec<_> = vec!["fact_1", "experience_1", "feeling_1", "skill_1"]
        .iter()
        .map(|k| hybrid_bank.get_node_id(&k.to_string()))
        .collect::<Vec<_>>();
    
    for id_future in node_ids.windows(2) {
        if let (Ok(id1), Ok(id2)) = (id_future[0].await, id_future[1].await) {
            bridge.write().connect_modalities(
                id1,
                MemoryModality::Semantic,
                id2,
                MemoryModality::Episodic,
                0.7
            ).unwrap();
        }
    }
    
    // Step 3: Start dream consolidation
    dream.start_consolidation().await.unwrap();
    sleep(Duration::from_secs(2)).await;
    
    // Step 4: Perform consciousness-weighted query
    consciousness.write().unwrap().set_consciousness_level(ConsciousnessLevel::HighlyConscious);
    let query = Array1::from_vec(vec![0.75; 768]);
    let results = hybrid_bank.search(&query, 10).await.unwrap();
    
    // Step 5: Check cross-modal insights
    let insights = dream.get_recent_insights(5).await.unwrap();
    
    // Verify complete integration
    assert!(!results.is_empty(), "Should find memories");
    assert!(!insights.is_empty(), "Should generate insights");
    assert_eq!(bridge.read().get_stats().total_connections, 3, "Should have cross-modal connections");
    
    // Stop consolidation
    dream.stop_consolidation().await.unwrap();
}
/// Test AttentionFusion mechanism for multi-modal queries
#[tokio::test]
async fn test_attention_fusion_multi_modal() {
    use neural_llm_memory::attention::{AttentionFusion, AttentionFusionConfig};
    use std::collections::HashMap;
    use ndarray::Array3;
    
    // Create cross-modal bridge
    let modal_config = CrossModalConfig::default();
    let bridge = Arc::new(CrossModalBridge::new(modal_config).unwrap());
    
    // Create attention fusion system
    let fusion_config = AttentionFusionConfig {
        feature_dimension: 768,
        num_heads: 8,
        dropout_rate: 0.1,
        coherence_threshold: 0.5,
        max_modalities: 8,
        adaptive_weighting: true,
        target_latency_ms: 5.0,
    };
    let fusion = Arc::new(AttentionFusion::new(fusion_config, bridge.clone()));
    
    // Create test query
    let query = Array2::from_shape_fn((1, 768), |_| rand::random::<f32>());
    
    // Create memory contents for different modalities
    let mut memory_contents = HashMap::new();
    
    // Semantic memories
    memory_contents.insert(
        MemoryModality::Semantic,
        Array3::from_shape_fn((10, 1, 768), |_| rand::random::<f32>())
    );
    
    // Episodic memories
    memory_contents.insert(
        MemoryModality::Episodic,
        Array3::from_shape_fn((8, 1, 768), |_| rand::random::<f32>())
    );
    
    // Emotional memories
    memory_contents.insert(
        MemoryModality::Emotional,
        Array3::from_shape_fn((5, 1, 768), |_| rand::random::<f32>())
    );
    
    // Test 1: Single modality fusion
    let start_time = Instant::now();
    let single_result = fusion.fuse_attention(
        &query,
        memory_contents.clone(),
        vec![MemoryModality::Semantic],
    ).unwrap();
    let single_latency = start_time.elapsed();
    
    assert!(single_result.coherence_score == 1.0, "Single modality should have perfect coherence");
    assert!(single_latency.as_millis() < 5, "Should meet performance target");
    
    // Test 2: Multi-modal fusion
    let start_time = Instant::now();
    let multi_result = fusion.fuse_attention(
        &query,
        memory_contents.clone(),
        vec![MemoryModality::Semantic, MemoryModality::Episodic, MemoryModality::Emotional],
    ).unwrap();
    let multi_latency = start_time.elapsed();
    
    assert!(multi_result.attention_weights.len() == 3, "Should have attention weights for all modalities");
    assert!(multi_result.modality_contributions.len() == 3, "Should have contributions from all modalities");
    assert!(multi_latency.as_millis() < 5, "Should meet performance target");
    
    // Test 3: Cache effectiveness
    let start_time = Instant::now();
    let cached_result = fusion.fuse_attention(
        &query,
        memory_contents.clone(),
        vec![MemoryModality::Semantic, MemoryModality::Episodic, MemoryModality::Emotional],
    ).unwrap();
    let cached_latency = start_time.elapsed();
    
    assert!(cached_latency < multi_latency / 2, "Cached query should be significantly faster");
    
    // Test 4: Coherence monitoring
    // Create less coherent memories
    let mut incoherent_memories = HashMap::new();
    incoherent_memories.insert(
        MemoryModality::Semantic,
        Array3::from_shape_fn((10, 1, 768), |_| rand::random::<f32>() * 2.0 - 1.0)
    );
    incoherent_memories.insert(
        MemoryModality::Emotional,
        Array3::from_shape_fn((10, 1, 768), |_| rand::random::<f32>() * 0.1)
    );
    
    let incoherent_result = fusion.fuse_attention(
        &query,
        incoherent_memories,
        vec![MemoryModality::Semantic, MemoryModality::Emotional],
    ).unwrap();
    
    assert!(incoherent_result.coherence_score < 0.5, "Incoherent memories should have low coherence");
    
    // Test 5: Performance statistics
    let stats = fusion.get_stats();
    assert!(stats.total_fusions >= 4, "Should have performed at least 4 fusions");
    assert!(stats.avg_latency_ms < 5.0, "Average latency should meet target");
    assert!(stats.cache_hit_rate > 0.0, "Should have cache hits");
    
    println!("AttentionFusion Statistics:");
    println!("  Total fusions: {}", stats.total_fusions);
    println!("  Avg latency: {:.2}ms", stats.avg_latency_ms);
    println!("  Cache hit rate: {:.2}%", stats.cache_hit_rate * 100.0);
    println!("  Overall coherence: {:.2}", stats.overall_coherence);
    println!("  Successful fusion rate: {:.2}%", stats.successful_fusion_rate * 100.0);
}

/// Test AttentionFusion integration with HybridMemoryBank
#[tokio::test]
async fn test_fusion_hybrid_memory_integration() {
    use neural_llm_memory::attention::{AttentionFusion, AttentionFusionConfig};
    use std::collections::HashMap;
    use ndarray::Array3;
    
    // Create hybrid memory bank
    let memory_config = MemoryConfig::default();
    let graph_config = ConsciousGraphConfig::default();
    let hybrid_bank = Arc::new(HybridMemoryBank::new(memory_config, graph_config).await.unwrap());
    
    // Create cross-modal bridge
    let modal_config = CrossModalConfig::default();
    let bridge = Arc::new(CrossModalBridge::new(modal_config).unwrap());
    
    // Create attention fusion
    let fusion_config = AttentionFusionConfig::default();
    let fusion = Arc::new(AttentionFusion::new(fusion_config, bridge.clone()));
    
    // Store memories with different modalities
    let memories = vec![
        ("semantic_fact", MemoryModality::Semantic, vec![0.5; 768]),
        ("personal_event", MemoryModality::Episodic, vec![0.7; 768]),
        ("emotional_response", MemoryModality::Emotional, vec![0.9; 768]),
        ("learned_skill", MemoryModality::Procedural, vec![0.6; 768]),
    ];
    
    for (key, modality, embedding) in memories {
        let value = MemoryValue {
            embeddings: Array1::from_vec(embedding),
            metadata: Default::default(),
        };
        hybrid_bank.store(key.to_string(), value).await.unwrap();
        
        // Register modality with bridge
        if let Ok(node_id) = hybrid_bank.get_node_id(&key.to_string()).await {
            bridge.register_node(node_id, modality).unwrap();
        }
    }
    
    // Prepare memory contents for fusion
    let mut memory_contents = HashMap::new();
    
    // Retrieve memories by modality
    for modality in &[MemoryModality::Semantic, MemoryModality::Episodic, MemoryModality::Emotional] {
        let memories = hybrid_bank.get_memories_by_modality(*modality).await.unwrap();
        if !memories.is_empty() {
            let memory_array = Array3::from_shape_fn(
                (memories.len(), 1, 768),
                |(i, _, k)| memories[i].embeddings[k]
            );
            memory_contents.insert(*modality, memory_array);
        }
    }
    
    // Perform multi-modal query with fusion
    let query = Array2::from_shape_fn((1, 768), |_| 0.65);
    
    let fusion_result = fusion.multi_modal_query(
        &query,
        memory_contents,
        vec![MemoryModality::Semantic, MemoryModality::Episodic, MemoryModality::Emotional],
    ).unwrap();
    
    // Verify fusion results
    assert!(fusion_result.coherence_score > 0.7, "Should have high coherence for similar embeddings");
    assert!(fusion_result.latency_ms < 5.0, "Should meet performance target");
    
    // Check modality contributions
    let total_contribution: f32 = fusion_result.modality_contributions.values().sum();
    assert!((total_contribution - 1.0).abs() < 0.01, "Contributions should sum to 1.0");
    
    println!("Fusion with HybridMemoryBank:");
    println!("  Coherence: {:.2}", fusion_result.coherence_score);
    println!("  Latency: {:.2}ms", fusion_result.latency_ms);
    println!("  Modality contributions:");
    for (modality, contribution) in &fusion_result.modality_contributions {
        println!("    {:?}: {:.2}%", modality, contribution * 100.0);
    }
}
