use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neural_llm_memory::{
    attention::{AttentionFusion, AttentionFusionConfig},
    graph::cross_modal::{CrossModalBridge, CrossModalConfig, MemoryModality},
};
use std::sync::Arc;
use std::collections::HashMap;
use ndarray::{Array2, Array3};

fn bench_single_modality_fusion(c: &mut Criterion) {
    // Setup
    let cross_modal_config = CrossModalConfig::default();
    let bridge = Arc::new(CrossModalBridge::new(cross_modal_config));
    
    let fusion_config = AttentionFusionConfig::default();
    let fusion = Arc::new(AttentionFusion::new(fusion_config, bridge));
    
    let query = Array2::from_shape_fn((1, 768), |_| rand::random::<f32>());
    let mut memory_contents = HashMap::new();
    memory_contents.insert(
        MemoryModality::Semantic,
        Array3::from_shape_fn((10, 1, 768), |_| rand::random::<f32>())
    );
    
    c.bench_function("single_modality_fusion", |b| {
        b.iter(|| {
            let result = fusion.fuse_attention(
                &query,
                black_box(memory_contents.clone()),
                vec![MemoryModality::Semantic],
            );
            black_box(result);
        })
    });
}

fn bench_multi_modal_fusion(c: &mut Criterion) {
    // Setup
    let cross_modal_config = CrossModalConfig::default();
    let bridge = Arc::new(CrossModalBridge::new(cross_modal_config));
    
    let fusion_config = AttentionFusionConfig::default();
    let fusion = Arc::new(AttentionFusion::new(fusion_config, bridge));
    
    let query = Array2::from_shape_fn((1, 768), |_| rand::random::<f32>());
    
    let mut group = c.benchmark_group("multi_modal_fusion");
    
    for num_modalities in [2, 3, 4, 5].iter() {
        let mut memory_contents = HashMap::new();
        let modalities = vec![
            MemoryModality::Semantic,
            MemoryModality::Episodic,
            MemoryModality::Emotional,
            MemoryModality::Procedural,
            MemoryModality::Contextual,
        ];
        
        let active_modalities: Vec<_> = modalities.iter().take(*num_modalities).cloned().collect();
        
        for modality in &active_modalities {
            memory_contents.insert(
                *modality,
                Array3::from_shape_fn((10, 1, 768), |_| rand::random::<f32>())
            );
        }
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_modalities),
            num_modalities,
            |b, _| {
                b.iter(|| {
                    let result = fusion.fuse_attention(
                        &query,
                        black_box(memory_contents.clone()),
                        active_modalities.clone(),
                    );
                    black_box(result);
                })
            },
        );
    }
    group.finish();
}

fn bench_coherence_computation(c: &mut Criterion) {
    // Setup
    let cross_modal_config = CrossModalConfig::default();
    let bridge = Arc::new(CrossModalBridge::new(cross_modal_config));
    
    let fusion_config = AttentionFusionConfig::default();
    let fusion = Arc::new(AttentionFusion::new(fusion_config, bridge));
    
    let query = Array2::from_shape_fn((1, 768), |_| rand::random::<f32>());
    
    let mut group = c.benchmark_group("coherence_computation");
    
    // Test with coherent vs incoherent memories
    for (name, factor) in [("coherent", 1.0), ("incoherent", 10.0)].iter() {
        let mut memory_contents = HashMap::new();
        
        memory_contents.insert(
            MemoryModality::Semantic,
            Array3::from_shape_fn((10, 1, 768), |_| rand::random::<f32>() * factor)
        );
        memory_contents.insert(
            MemoryModality::Emotional,
            Array3::from_shape_fn((10, 1, 768), |_| rand::random::<f32>() * factor)
        );
        
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            name,
            |b, _| {
                b.iter(|| {
                    let result = fusion.fuse_attention(
                        &query,
                        black_box(memory_contents.clone()),
                        vec![MemoryModality::Semantic, MemoryModality::Emotional],
                    );
                    black_box(result);
                })
            },
        );
    }
    group.finish();
}

fn bench_cache_performance(c: &mut Criterion) {
    // Setup
    let cross_modal_config = CrossModalConfig::default();
    let bridge = Arc::new(CrossModalBridge::new(cross_modal_config));
    
    let fusion_config = AttentionFusionConfig::default();
    let fusion = Arc::new(AttentionFusion::new(fusion_config, bridge));
    
    let query = Array2::from_shape_fn((1, 768), |_| 0.5); // Fixed query for cache hits
    let mut memory_contents = HashMap::new();
    memory_contents.insert(
        MemoryModality::Semantic,
        Array3::from_shape_fn((10, 1, 768), |_| 0.5) // Fixed content for cache hits
    );
    
    // Prime the cache
    let _ = fusion.fuse_attention(
        &query,
        memory_contents.clone(),
        vec![MemoryModality::Semantic],
    );
    
    c.bench_function("cached_fusion", |b| {
        b.iter(|| {
            let result = fusion.fuse_attention(
                &query,
                black_box(memory_contents.clone()),
                vec![MemoryModality::Semantic],
            );
            black_box(result);
        })
    });
}

criterion_group!(
    benches,
    bench_single_modality_fusion,
    bench_multi_modal_fusion,
    bench_coherence_computation,
    bench_cache_performance
);
criterion_main!(benches);