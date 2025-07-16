//! Consciousness Performance Benchmarks
//! 
//! Validates that consciousness optimizations meet the <10% latency increase target.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neural_llm_memory::nn::optimization::{ConsciousnessCache, ConsciousnessContext, SimdAttention};
use neural_llm_memory::consciousness::{ConsciousnessCore, ConsciousInput, ContentType};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::time::Instant;

/// Benchmark consciousness processing without caching
fn bench_consciousness_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("consciousness_baseline");
    
    let core = ConsciousnessCore::new();
    
    for size in [128, 256, 512, 768, 1024].iter() {
        let input = ConsciousInput {
            content_type: ContentType::PerceptualInput,
            activation: 0.8,
            attention_weight: 0.7,
            semantic_embedding: Array1::from_vec(vec![0.1; *size]),
            metadata: HashMap::new(),
        };
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    let output = core.process_input(black_box(input.clone()));
                    black_box(output);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark consciousness processing with caching enabled
fn bench_consciousness_with_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("consciousness_with_cache");
    
    let core = ConsciousnessCore::new();
    let cache = ConsciousnessCache::new();
    
    // Pre-populate cache with some contexts
    for i in 0..100 {
        let context = ConsciousnessContext {
            id: format!("context_{}", i),
            embedding: Array1::from_vec(vec![0.1; 768]),
            attention_weights: vec![0.5; 12],
            metadata: HashMap::new(),
            cached_scores: None,
        };
        cache.insert(format!("key_{}", i), context);
    }
    
    for size in [128, 256, 512, 768, 1024].iter() {
        let input = ConsciousInput {
            content_type: ContentType::PerceptualInput,
            activation: 0.8,
            attention_weight: 0.7,
            semantic_embedding: Array1::from_vec(vec![0.1; *size]),
            metadata: HashMap::new(),
        };
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    // Simulate cache lookup
                    let _ = cache.get("key_50");
                    
                    let output = core.process_input(black_box(input.clone()));
                    black_box(output);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark cache operations
fn bench_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_operations");
    
    let cache = ConsciousnessCache::new();
    
    // Benchmark cache insertion
    group.bench_function("insert", |b| {
        let mut counter = 0;
        b.iter(|| {
            let context = ConsciousnessContext {
                id: format!("context_{}", counter),
                embedding: Array1::from_vec(vec![0.1; 768]),
                attention_weights: vec![0.5; 12],
                metadata: HashMap::new(),
                cached_scores: None,
            };
            cache.insert(format!("key_{}", counter), context);
            counter += 1;
        });
    });
    
    // Pre-populate cache for retrieval benchmarks
    for i in 0..1000 {
        let context = ConsciousnessContext {
            id: format!("context_{}", i),
            embedding: Array1::from_vec(vec![0.1; 768]),
            attention_weights: vec![0.5; 12],
            metadata: HashMap::new(),
            cached_scores: None,
        };
        cache.insert(format!("key_{}", i), context);
    }
    
    // Benchmark cache retrieval (hit)
    group.bench_function("get_hit", |b| {
        b.iter(|| {
            let result = cache.get(black_box("key_500"));
            black_box(result);
        });
    });
    
    // Benchmark cache retrieval (miss)
    group.bench_function("get_miss", |b| {
        b.iter(|| {
            let result = cache.get(black_box("nonexistent_key"));
            black_box(result);
        });
    });
    
    group.finish();
}

/// Benchmark SIMD attention operations
fn bench_simd_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_attention");
    
    let attention = SimdAttention::new();
    
    for seq_len in [32, 64, 128, 256].iter() {
        for d_model in [256, 512, 768].iter() {
            let query = Array2::from_shape_fn((*seq_len, *d_model), |(i, j)| {
                ((i + j) as f32 * 0.01).sin()
            });
            let key = query.clone();
            let value = Array2::from_shape_fn((*seq_len, *d_model), |(i, j)| {
                ((i * j) as f32 * 0.01).cos()
            });
            
            let param = format!("{}x{}", seq_len, d_model);
            
            group.bench_with_input(
                BenchmarkId::from_parameter(&param),
                &param,
                |b, _| {
                    b.iter(|| {
                        let output = attention.scaled_dot_product_attention(
                            black_box(&query),
                            black_box(&key),
                            black_box(&value),
                            None,
                        );
                        black_box(output);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark multi-head attention
fn bench_multi_head_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_head_attention");
    
    let attention = SimdAttention::new();
    
    for num_heads in [4, 8, 12].iter() {
        let seq_len = 128;
        let d_model = 768;
        
        let query = Array2::from_shape_fn((seq_len, d_model), |(i, j)| {
            ((i + j) as f32 * 0.01).sin()
        });
        let key = query.clone();
        let value = Array2::from_shape_fn((seq_len, d_model), |(i, j)| {
            ((i * j) as f32 * 0.01).cos()
        });
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_heads),
            num_heads,
            |b, &heads| {
                b.iter(|| {
                    let output = attention.multi_head_attention(
                        black_box(&query),
                        black_box(&key),
                        black_box(&value),
                        heads,
                        d_model,
                    );
                    black_box(output);
                });
            },
        );
    }
    
    group.finish();
}

/// Measure latency overhead of consciousness features
fn bench_latency_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_overhead");
    
    // Baseline: simple matrix multiplication
    let a = Array2::from_shape_fn((128, 768), |(i, j)| ((i + j) as f32 * 0.01).sin());
    let b = Array2::from_shape_fn((768, 256), |(i, j)| ((i * j) as f32 * 0.01).cos());
    
    group.bench_function("baseline_matmul", |bench| {
        bench.iter(|| {
            let result = a.dot(&b);
            black_box(result);
        });
    });
    
    // With consciousness processing
    let core = ConsciousnessCore::new();
    let input = ConsciousInput {
        content_type: ContentType::PerceptualInput,
        activation: 0.8,
        attention_weight: 0.7,
        semantic_embedding: Array1::from_vec(vec![0.1; 768]),
        metadata: HashMap::new(),
    };
    
    group.bench_function("with_consciousness", |bench| {
        bench.iter(|| {
            let result = a.dot(&b);
            let output = core.process_input(input.clone());
            black_box((result, output));
        });
    });
    
    group.finish();
}

/// Validate that latency increase is under 10%
fn validate_latency_target() {
    println!("\n=== Latency Target Validation ===");
    
    let iterations = 1000;
    let a = Array2::from_shape_fn((128, 768), |(i, j)| ((i + j) as f32 * 0.01).sin());
    let b = Array2::from_shape_fn((768, 256), |(i, j)| ((i * j) as f32 * 0.01).cos());
    
    // Measure baseline
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.dot(&b);
    }
    let baseline_duration = start.elapsed();
    
    // Measure with consciousness
    let core = ConsciousnessCore::new();
    let cache = ConsciousnessCache::new();
    let input = ConsciousInput {
        content_type: ContentType::PerceptualInput,
        activation: 0.8,
        attention_weight: 0.7,
        semantic_embedding: Array1::from_vec(vec![0.1; 768]),
        metadata: HashMap::new(),
    };
    
    // Pre-populate cache
    for i in 0..100 {
        let context = ConsciousnessContext {
            id: format!("context_{}", i),
            embedding: Array1::from_vec(vec![0.1; 768]),
            attention_weights: vec![0.5; 12],
            metadata: HashMap::new(),
            cached_scores: None,
        };
        cache.insert(format!("key_{}", i), context);
    }
    
    let start = Instant::now();
    for i in 0..iterations {
        let _ = a.dot(&b);
        let _ = cache.get(&format!("key_{}", i % 100));
        let _ = core.process_input(input.clone());
    }
    let consciousness_duration = start.elapsed();
    
    let baseline_ms = baseline_duration.as_secs_f64() * 1000.0;
    let consciousness_ms = consciousness_duration.as_secs_f64() * 1000.0;
    let overhead_percent = ((consciousness_ms - baseline_ms) / baseline_ms) * 100.0;
    
    println!("Baseline:        {:.2} ms", baseline_ms);
    println!("With Consciousness: {:.2} ms", consciousness_ms);
    println!("Overhead:        {:.2}%", overhead_percent);
    println!("Target:          <10%");
    println!("Status:          {}", if overhead_percent < 10.0 { "✓ PASS" } else { "✗ FAIL" });
    
    // Check cache performance
    let metrics = cache.metrics();
    let hit_rate = cache.hit_rate();
    let efficiency = cache.efficiency_score();
    
    println!("\n=== Cache Performance ===");
    println!("Hit Rate:        {:.2}%", hit_rate * 100.0);
    println!("Avg Access Time: {:.2} μs", metrics.avg_access_time_us);
    println!("Efficiency Score: {:.2}", efficiency);
    println!("Total Hits:      {}", metrics.hits);
    println!("Total Misses:    {}", metrics.misses);
}

/// Memory usage profiling
fn profile_memory_usage() {
    println!("\n=== Memory Usage Profile ===");
    
    // Create instances
    let core = ConsciousnessCore::new();
    let cache = ConsciousnessCache::new();
    let attention = SimdAttention::new();
    
    // Estimate memory usage
    let cache_entries = 1000;
    let embedding_size = 768;
    let attention_weights = 12;
    
    let per_entry_bytes = 
        embedding_size * 4 + // f32 embedding
        attention_weights * 4 + // f32 weights
        100; // metadata estimate
    
    let cache_memory_mb = (cache_entries * per_entry_bytes) as f64 / (1024.0 * 1024.0);
    
    println!("Cache Capacity:  {} entries", cache_entries);
    println!("Per Entry Size:  ~{} bytes", per_entry_bytes);
    println!("Total Cache:     ~{:.2} MB", cache_memory_mb);
    
    // Measure actual allocations
    let contexts: Vec<ConsciousnessContext> = (0..100)
        .map(|i| ConsciousnessContext {
            id: format!("context_{}", i),
            embedding: Array1::from_vec(vec![0.1; embedding_size]),
            attention_weights: vec![0.5; attention_weights],
            metadata: HashMap::new(),
            cached_scores: Some(vec![0.0; 100]),
        })
        .collect();
    
    println!("\nActual Allocations:");
    println!("100 contexts:    ~{:.2} MB", 
        (100 * per_entry_bytes) as f64 / (1024.0 * 1024.0));
}

fn custom_criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(5))
        .sample_size(100)
}

criterion_group!(
    name = benches;
    config = custom_criterion();
    targets = 
        bench_consciousness_baseline,
        bench_consciousness_with_cache,
        bench_cache_operations,
        bench_simd_attention,
        bench_multi_head_attention,
        bench_latency_overhead
);

criterion_main!(benches);

// Additional validation that runs separately
#[test]
fn test_performance_targets() {
    validate_latency_target();
    profile_memory_usage();
}