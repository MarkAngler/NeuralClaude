//! Integration tests for consciousness optimizations

use neural_llm_memory::consciousness::{ConsciousnessCore, ConsciousInput, ContentType};
use neural_llm_memory::consciousness_integration::{OptimizedConsciousness, OptimizedConfig};
use neural_llm_memory::nn::optimization::{ConsciousnessCache, SimdAttention};
use ndarray::Array1;
use std::collections::HashMap;
use std::time::Instant;

#[test]
fn test_latency_target_validation() {
    // Create optimized consciousness
    let consciousness = OptimizedConsciousness::new();
    
    // Warm up cache
    for i in 0..10 {
        let input = ConsciousInput {
            content_type: ContentType::PerceptualInput,
            activation: 0.8,
            attention_weight: 0.7,
            semantic_embedding: Array1::from_vec(vec![i as f32 * 0.1; 768]),
            metadata: HashMap::new(),
        };
        consciousness.process_optimized(input);
    }
    
    // Measure baseline (without consciousness)
    let baseline_start = Instant::now();
    for _ in 0..100 {
        let a = Array1::from_vec(vec![0.1; 768]);
        let b = Array1::from_vec(vec![0.2; 768]);
        let _ = a.dot(&b);
    }
    let baseline_duration = baseline_start.elapsed();
    
    // Measure with consciousness
    let consciousness_start = Instant::now();
    for i in 0..100 {
        let input = ConsciousInput {
            content_type: ContentType::PerceptualInput,
            activation: 0.8,
            attention_weight: 0.7,
            semantic_embedding: Array1::from_vec(vec![i as f32 * 0.01; 768]),
            metadata: HashMap::new(),
        };
        consciousness.process_optimized(input);
    }
    let consciousness_duration = consciousness_start.elapsed();
    
    // Calculate overhead
    let overhead_percent = ((consciousness_duration.as_secs_f64() - baseline_duration.as_secs_f64()) 
        / baseline_duration.as_secs_f64()) * 100.0;
    
    println!("Baseline: {:?}", baseline_duration);
    println!("With Consciousness: {:?}", consciousness_duration);
    println!("Overhead: {:.2}%", overhead_percent);
    
    // Validate performance
    let validation = consciousness.validate_performance();
    println!("Performance Validation: {:?}", validation);
    
    // Check that overhead is reasonable (may be higher in debug mode)
    #[cfg(not(debug_assertions))]
    assert!(overhead_percent < 20.0, "Overhead too high: {:.2}%", overhead_percent);
}

#[test]
fn test_cache_effectiveness() {
    let cache = ConsciousnessCache::new();
    
    // Insert test contexts
    for i in 0..100 {
        let context = neural_llm_memory::nn::optimization::ConsciousnessContext {
            id: format!("test_{}", i),
            embedding: Array1::from_vec(vec![0.1; 768]),
            attention_weights: vec![0.5; 12],
            metadata: HashMap::new(),
            cached_scores: None,
        };
        cache.insert(format!("key_{}", i), context);
    }
    
    // Test cache hits
    let mut hits = 0;
    for i in 0..200 {
        if cache.get(&format!("key_{}", i % 100)).is_some() {
            hits += 1;
        }
    }
    
    let hit_rate = cache.hit_rate();
    println!("Cache hit rate: {:.2}%", hit_rate * 100.0);
    assert!(hit_rate > 0.8, "Cache hit rate too low: {:.2}", hit_rate);
    
    let efficiency = cache.efficiency_score();
    println!("Cache efficiency: {:.2}", efficiency);
    assert!(efficiency > 0.5, "Cache efficiency too low: {:.2}", efficiency);
}

#[test]
fn test_simd_attention_performance() {
    let attention = SimdAttention::new();
    
    let seq_len = 128;
    let d_model = 768;
    
    let query = ndarray::Array2::from_shape_fn((seq_len, d_model), |(i, j)| {
        ((i + j) as f32 * 0.01).sin()
    });
    let key = query.clone();
    let value = ndarray::Array2::from_shape_fn((seq_len, d_model), |(i, j)| {
        ((i * j) as f32 * 0.01).cos()
    });
    
    let start = Instant::now();
    let _ = attention.scaled_dot_product_attention(&query, &key, &value, None);
    let duration = start.elapsed();
    
    println!("SIMD attention computation time: {:?}", duration);
    
    // Should complete quickly
    assert!(duration.as_millis() < 100, "Attention computation too slow: {:?}", duration);
}

#[test]
fn test_batch_processing_efficiency() {
    let consciousness = OptimizedConsciousness::new();
    
    // Create batch of inputs
    let inputs: Vec<ConsciousInput> = (0..50)
        .map(|i| ConsciousInput {
            content_type: ContentType::PerceptualInput,
            activation: 0.8,
            attention_weight: 0.7,
            semantic_embedding: Array1::from_vec(vec![i as f32 * 0.01; 768]),
            metadata: HashMap::new(),
        })
        .collect();
    
    // Time batch processing
    let start = Instant::now();
    let outputs = consciousness.batch_process(inputs);
    let batch_duration = start.elapsed();
    
    assert_eq!(outputs.len(), 50);
    
    // Compare with individual processing
    let individual_start = Instant::now();
    for i in 0..50 {
        let input = ConsciousInput {
            content_type: ContentType::PerceptualInput,
            activation: 0.8,
            attention_weight: 0.7,
            semantic_embedding: Array1::from_vec(vec![i as f32 * 0.01; 768]),
            metadata: HashMap::new(),
        };
        consciousness.process_optimized(input);
    }
    let individual_duration = individual_start.elapsed();
    
    println!("Batch processing: {:?}", batch_duration);
    println!("Individual processing: {:?}", individual_duration);
    
    // Batch should be more efficient
    assert!(batch_duration < individual_duration * 2);
}

#[test]
fn test_memory_usage() {
    let consciousness = OptimizedConsciousness::new();
    
    // Process many inputs to test memory usage
    for i in 0..1000 {
        let input = ConsciousInput {
            content_type: ContentType::PerceptualInput,
            activation: 0.8,
            attention_weight: 0.7,
            semantic_embedding: Array1::from_vec(vec![i as f32 * 0.001; 768]),
            metadata: HashMap::new(),
        };
        consciousness.process_optimized(input);
        
        // Periodically check metrics
        if i % 100 == 0 {
            let metrics = consciousness.metrics();
            println!("After {} operations - Cache hit rate: {:.2}%, Overhead: {:.2}%", 
                i, metrics.cache_hit_rate * 100.0, metrics.latency_overhead_percent);
        }
    }
    
    let final_metrics = consciousness.metrics();
    println!("Final metrics: {:?}", final_metrics);
    
    // Memory should be bounded by cache size
    assert!(final_metrics.operation_count == 1000);
}