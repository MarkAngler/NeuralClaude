//! Simplified Phase 3 Integration Tests
//! 
//! Tests the core functionality of Phase 3 components without complex dependencies

use neural_llm_memory::{
    memory::{MemoryConfig, MemoryValue, MemoryKey, MemoryBank, MemoryOperations},
    consciousness::{ConsciousnessCore, ConsciousnessConfig, ConsciousnessLevel},
    emotional::{EmotionalState, EmotionalIntensity},
};
use std::time::{Duration, Instant};
use ndarray::Array1;

/// Test consciousness-weighted memory operations
#[test]
fn test_consciousness_memory_integration() {
    // Create consciousness core
    let consciousness_config = ConsciousnessConfig::default();
    let mut consciousness = ConsciousnessCore::new(consciousness_config).unwrap();
    
    // Create memory bank
    let mut memory_bank = MemoryBank::new(1000, 100);
    
    // Test storing memories with different consciousness levels
    let test_cases = vec![
        ("high_conscious", ConsciousnessLevel::HighlyConscious, 0.9),
        ("normal_conscious", ConsciousnessLevel::Conscious, 0.7),
        ("low_conscious", ConsciousnessLevel::Subconscious, 0.3),
    ];
    
    for (key_prefix, level, expected_weight) in test_cases {
        // Set consciousness level
        consciousness.set_consciousness_level(level);
        let weight = consciousness.get_attention_weight();
        
        // Create weighted embedding
        let base_embedding = vec![0.5; 768];
        let weighted_embedding: Vec<f32> = base_embedding.iter()
            .map(|&v| v * weight)
            .collect();
        
        let memory_key = MemoryKey {
            id: format!("{}_memory", key_prefix),
            timestamp: chrono::Utc::now().timestamp() as u64,
            context_hash: 0,
        };
        
        let memory_value = MemoryValue {
            embedding: weighted_embedding.clone(),
            content: format!("Memory content at {} consciousness", key_prefix),
            metadata: Default::default(),
        };
        
        // Store memory
        memory_bank.store(memory_key.clone(), memory_value).unwrap();
        
        // Verify storage
        let retrieved = memory_bank.retrieve(&memory_key).unwrap().unwrap();
        let mean_value: f32 = retrieved.embedding.iter().sum::<f32>() / retrieved.embedding.len() as f32;
        
        // Check consciousness influence
        assert!(
            (mean_value - (0.5 * expected_weight)).abs() < 0.1,
            "Consciousness level {} should affect embedding strength",
            level
        );
    }
}

/// Test emotional state integration
#[test]
fn test_emotional_memory_integration() {
    let mut memory_bank = MemoryBank::new(1000, 100);
    
    // Test memories with emotional states
    let emotional_memories = vec![
        ("joy_memory", EmotionalState::Joy, EmotionalIntensity::High),
        ("sad_memory", EmotionalState::Sadness, EmotionalIntensity::Medium),
        ("excited_memory", EmotionalState::Excitement, EmotionalIntensity::VeryHigh),
    ];
    
    for (key_name, state, intensity) in emotional_memories {
        // Create embedding influenced by emotional intensity
        let base_value = 0.5;
        let intensity_factor = match intensity {
            EmotionalIntensity::Low => 0.3,
            EmotionalIntensity::Medium => 0.5,
            EmotionalIntensity::High => 0.7,
            EmotionalIntensity::VeryHigh => 0.9,
        };
        
        let emotional_embedding = vec![base_value * intensity_factor; 768];
        
        let memory_key = MemoryKey {
            id: key_name.to_string(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            context_hash: 0,
        };
        
        let memory_value = MemoryValue {
            embedding: emotional_embedding,
            content: format!("Emotional memory: {:?} at {:?} intensity", state, intensity),
            metadata: Default::default(),
        };
        
        memory_bank.store(memory_key, memory_value).unwrap();
    }
    
    // Verify emotional memories were stored
    let (count, _, _, _) = memory_bank.get_stats();
    assert_eq!(count, 3, "Should have stored 3 emotional memories");
}

/// Performance benchmark for memory operations
#[test]
fn test_memory_performance_benchmark() {
    let mut baseline_bank = MemoryBank::new(10000, 100);
    let num_operations = 1000;
    
    // Generate test data
    let test_data: Vec<(MemoryKey, MemoryValue)> = (0..num_operations)
        .map(|i| {
            let key = MemoryKey {
                id: format!("perf_test_{}", i),
                timestamp: chrono::Utc::now().timestamp() as u64,
                context_hash: i as u64,
            };
            let value = MemoryValue {
                embedding: vec![i as f32 / 1000.0; 768],
                content: format!("Performance test content {}", i),
                metadata: Default::default(),
            };
            (key, value)
        })
        .collect();
    
    // Benchmark store operations
    let store_start = Instant::now();
    for (key, value) in &test_data {
        baseline_bank.store(key.clone(), value.clone()).unwrap();
    }
    let store_duration = store_start.elapsed();
    
    // Benchmark retrieve operations
    let retrieve_start = Instant::now();
    for (key, _) in &test_data[..100] { // Test first 100
        baseline_bank.retrieve(key).unwrap();
    }
    let retrieve_duration = retrieve_start.elapsed();
    
    // Benchmark search operations
    let search_start = Instant::now();
    let query = Array1::from_vec(vec![0.5; 768]).insert_axis(ndarray::Axis(0));
    for _ in 0..10 {
        let _results = baseline_bank.search(&query, 10);
    }
    let search_duration = search_start.elapsed();
    
    println!("Performance Results:");
    println!("Store {} items: {:?}", num_operations, store_duration);
    println!("Retrieve 100 items: {:?}", retrieve_duration);
    println!("Search 10 times: {:?}", search_duration);
    
    // Verify reasonable performance
    assert!(store_duration < Duration::from_secs(5), "Store operations should complete within 5 seconds");
    assert!(retrieve_duration < Duration::from_millis(100), "Retrieve operations should be fast");
    assert!(search_duration < Duration::from_secs(1), "Search operations should complete within 1 second");
}

/// Test concurrent access safety
#[test]
fn test_concurrent_memory_access() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let memory_bank = Arc::new(Mutex::new(MemoryBank::new(10000, 100)));
    let mut handles = vec![];
    
    // Spawn writer threads
    for i in 0..5 {
        let bank = Arc::clone(&memory_bank);
        let handle = thread::spawn(move || {
            for j in 0..100 {
                let key = MemoryKey {
                    id: format!("concurrent_{}_{}", i, j),
                    timestamp: chrono::Utc::now().timestamp() as u64,
                    context_hash: (i * 100 + j) as u64,
                };
                let value = MemoryValue {
                    embedding: vec![i as f32; 768],
                    content: format!("Concurrent content {} {}", i, j),
                    metadata: Default::default(),
                };
                
                bank.lock().unwrap().store(key, value).unwrap();
            }
        });
        handles.push(handle);
    }
    
    // Spawn reader threads
    for i in 0..5 {
        let bank = Arc::clone(&memory_bank);
        let handle = thread::spawn(move || {
            let query = Array1::from_vec(vec![i as f32; 768]).insert_axis(ndarray::Axis(0));
            for _ in 0..50 {
                let _results = bank.lock().unwrap().search(&query, 5);
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify data integrity
    let (count, _, _, _) = memory_bank.lock().unwrap().get_stats();
    assert_eq!(count, 500, "Should have exactly 500 entries from concurrent writes");
}

/// Test memory metadata and decay
#[test]
fn test_memory_metadata_operations() {
    let mut memory_bank = MemoryBank::new(100, 50);
    
    // Create memory with metadata
    let key = MemoryKey {
        id: "metadata_test".to_string(),
        timestamp: chrono::Utc::now().timestamp() as u64,
        context_hash: 42,
    };
    
    let mut metadata = neural_llm_memory::memory::MemoryMetadata::default();
    metadata.importance = 0.9;
    metadata.tags = vec!["important".to_string(), "test".to_string()];
    
    let value = MemoryValue {
        embedding: vec![0.7; 768],
        content: "Important memory with metadata".to_string(),
        metadata,
    };
    
    // Store and retrieve
    memory_bank.store(key.clone(), value.clone()).unwrap();
    let retrieved = memory_bank.retrieve(&key).unwrap().unwrap();
    
    // Verify metadata preserved
    assert_eq!(retrieved.metadata.importance, 0.9);
    assert_eq!(retrieved.metadata.tags.len(), 2);
    assert!(retrieved.metadata.tags.contains(&"important".to_string()));
    
    // Test update operation
    memory_bank.update(&key, |mem_val| {
        mem_val.metadata.access_count += 1;
        mem_val.metadata.importance *= 0.95; // Decay
    }).unwrap();
    
    let updated = memory_bank.retrieve(&key).unwrap().unwrap();
    assert_eq!(updated.metadata.access_count, 1);
    assert!((updated.metadata.importance - 0.855).abs() < 0.001);
}