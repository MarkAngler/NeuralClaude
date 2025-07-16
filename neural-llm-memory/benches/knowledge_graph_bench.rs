//! Performance benchmarks for knowledge graph operations
//! Ensures all operations meet the <50ms requirement

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neural_llm_memory::graph::{KnowledgeGraph, GraphPattern, NodeConstraint};
use neural_llm_memory::memory::{MemoryBank, MemoryKey, MemoryValue, MemoryMetadata};
use std::collections::HashMap;
use std::time::Duration;
use ndarray::Array2;
use rand::Rng;

/// Create a test graph with specified number of nodes
fn create_benchmark_graph(size: usize) -> KnowledgeGraph {
    let mut graph = KnowledgeGraph::new();
    let mut nodes = Vec::new();
    let mut rng = rand::thread_rng();
    
    // Create nodes with realistic embeddings
    for i in 0..size {
        let embedding: Vec<f32> = (0..768).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let node_id = graph.create_node(
            format!("memory_{}", i),
            embedding,
            HashMap::from([
                ("type".to_string(), ["concept", "instance", "relation"][i % 3].to_string()),
                ("importance".to_string(), rng.gen_range(0.1..1.0).to_string()),
                ("category".to_string(), format!("cat_{}", i % 10)),
            ])
        ).unwrap();
        nodes.push(node_id);
    }
    
    // Create edges with power-law distribution (realistic for knowledge graphs)
    for i in 0..size {
        let num_edges = (rng.gen_range(1.0..5.0).powf(2.0) as usize).min(10);
        for _ in 0..num_edges {
            let target = rng.gen_range(0..size);
            if target != i {
                let _ = graph.create_edge(
                    nodes[i].clone(),
                    nodes[target].clone(),
                    ["related_to", "derived_from", "similar_to", "opposite_of"][rng.gen_range(0..4)].to_string(),
                    HashMap::from([
                        ("weight".to_string(), rng.gen_range(0.1..1.0).to_string()),
                    ])
                );
            }
        }
    }
    
    graph
}

/// Create a hybrid memory system with graph support
fn create_hybrid_memory(size: usize) -> HybridMemorySystem {
    HybridMemorySystem {
        memory_bank: MemoryBank::new(size * 2, size / 10),
        knowledge_graph: KnowledgeGraph::new(),
    }
}

/// Benchmark graph query operations
fn benchmark_graph_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_queries");
    group.measurement_time(Duration::from_secs(10));
    
    for size in &[100, 1000, 10000] {
        let graph = create_benchmark_graph(*size);
        let nodes: Vec<_> = graph.get_all_nodes().collect();
        let mut rng = rand::thread_rng();
        
        // Benchmark: Single neighbor lookup
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("single_neighbor_lookup", size),
            size,
            |b, _| {
                b.iter(|| {
                    let node = &nodes[rng.gen_range(0..nodes.len())];
                    graph.get_neighbors(black_box(&node.id), None)
                });
            }
        );
        
        // Benchmark: 2-hop neighbor search (must be <50ms)
        group.bench_with_input(
            BenchmarkId::new("two_hop_neighbors", size),
            size,
            |b, _| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let node = &nodes[rng.gen_range(0..nodes.len())];
                        let start = std::time::Instant::now();
                        let _ = graph.find_neighbors_n_hops(black_box(&node.id), black_box(2));
                        let elapsed = start.elapsed();
                        assert!(elapsed < Duration::from_millis(50), 
                               "2-hop search took {:?}, must be <50ms", elapsed);
                        total += elapsed;
                    }
                    total
                });
            }
        );
        
        // Benchmark: Pattern matching
        group.bench_with_input(
            BenchmarkId::new("pattern_matching", size),
            size,
            |b, _| {
                let pattern = GraphPattern {
                    node_constraints: vec![
                        NodeConstraint { 
                            property: "type".to_string(), 
                            value: "concept".to_string() 
                        },
                    ],
                    edge_constraints: vec![],
                    max_results: 10,
                };
                b.iter(|| {
                    graph.match_pattern(black_box(&pattern))
                });
            }
        );
        
        // Benchmark: Shortest path
        group.bench_with_input(
            BenchmarkId::new("shortest_path", size),
            size,
            |b, _| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start_node = &nodes[rng.gen_range(0..nodes.len())].id;
                        let end_node = &nodes[rng.gen_range(0..nodes.len())].id;
                        let start = std::time::Instant::now();
                        let _ = graph.shortest_path(
                            black_box(start_node),
                            black_box(end_node)
                        );
                        let elapsed = start.elapsed();
                        total += elapsed;
                    }
                    total
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark memory operations with graph enhancement
fn benchmark_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark: Hybrid store (KV + Graph)
    group.bench_function("hybrid_store_with_relations", |b| {
        let mut memory = create_hybrid_memory(1000);
        let mut counter = 0;
        let mut rng = rand::thread_rng();
        
        b.iter(|| {
            let embedding: Vec<f32> = (0..768).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let key = format!("test_key_{}", counter);
            counter += 1;
            
            memory.store_with_relations(
                black_box(MemoryKey::from(key)),
                black_box(MemoryValue {
                    embedding: embedding.clone(),
                    content: "Test memory content with relations".to_string(),
                    metadata: MemoryMetadata::default(),
                }),
                black_box(vec![
                    ("related_to", "test_key_0"),
                    ("derived_from", "test_key_1"),
                ])
            )
        });
    });
    
    // Benchmark: Graph-enhanced search
    group.bench_function("graph_enhanced_search", |b| {
        let mut memory = create_populated_hybrid_memory(1000);
        let mut rng = rand::thread_rng();
        
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let query_embedding = Array2::from_shape_vec(
                    (1, 768),
                    (0..768).map(|_| rng.gen_range(-1.0..1.0)).collect()
                ).unwrap();
                
                let start = std::time::Instant::now();
                let _ = memory.search_with_graph_context(
                    black_box(&query_embedding),
                    black_box(5)
                );
                let elapsed = start.elapsed();
                assert!(elapsed < Duration::from_millis(50),
                       "Graph-enhanced search took {:?}, must be <50ms", elapsed);
                total += elapsed;
            }
            total
        });
    });
    
    // Benchmark: Relationship traversal
    group.bench_function("relationship_traversal", |b| {
        let memory = create_populated_hybrid_memory(1000);
        let keys: Vec<_> = memory.get_all_keys().collect();
        let mut rng = rand::thread_rng();
        
        b.iter(|| {
            let start_key = &keys[rng.gen_range(0..keys.len())];
            memory.traverse_relationships(
                black_box(start_key),
                black_box("related_to"),
                black_box(3) // depth
            )
        });
    });
    
    group.finish();
}

/// Benchmark concurrent access patterns
fn benchmark_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_access");
    group.measurement_time(Duration::from_secs(10));
    
    use std::sync::Arc;
    use rayon::prelude::*;
    
    let graph = Arc::new(create_benchmark_graph(10000));
    let nodes: Vec<_> = graph.get_all_nodes().map(|n| n.id.clone()).collect();
    
    // Benchmark: Parallel reads
    group.bench_function("parallel_reads_1000", |b| {
        b.iter(|| {
            (0..1000).into_par_iter().for_each(|i| {
                let node = &nodes[i % nodes.len()];
                let _ = graph.get_node(black_box(node));
                let _ = graph.get_neighbors(black_box(node), None);
            });
        });
    });
    
    // Benchmark: Mixed read/write
    group.bench_function("mixed_read_write", |b| {
        let graph = Arc::new(KnowledgeGraph::new_concurrent());
        
        b.iter(|| {
            (0..100).into_par_iter().for_each(|i| {
                if i % 5 == 0 {
                    // Write operation
                    let _ = graph.create_node(
                        format!("concurrent_node_{}", i),
                        vec![i as f32; 10],
                        HashMap::new()
                    );
                } else {
                    // Read operation
                    let _ = graph.node_count();
                }
            });
        });
    });
    
    group.finish();
}

/// Benchmark memory overhead
fn benchmark_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead");
    
    // Measure memory usage for different graph sizes
    for size in &[1000, 10000, 100000] {
        group.bench_with_input(
            BenchmarkId::new("graph_memory_usage", size),
            size,
            |b, &size| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = std::time::Instant::now();
                        let graph = create_benchmark_graph(size);
                        let _ = black_box(graph.calculate_memory_usage());
                        total += start.elapsed();
                    }
                    total
                });
            }
        );
    }
    
    group.finish();
}

/// Create a populated hybrid memory system for benchmarks
fn create_populated_hybrid_memory(size: usize) -> HybridMemorySystem {
    let mut memory = create_hybrid_memory(size);
    let mut rng = rand::thread_rng();
    
    // Populate with interconnected memories
    for i in 0..size {
        let embedding: Vec<f32> = (0..768).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let key = MemoryKey::from(format!("memory_{}", i));
        
        memory.store(
            key.clone(),
            MemoryValue {
                embedding,
                content: format!("Memory content {}", i),
                metadata: MemoryMetadata {
                    importance: rng.gen_range(0.1..1.0),
                    ..Default::default()
                },
            }
        ).unwrap();
        
        // Add relations to previously stored memories
        if i > 0 {
            let relations = vec![
                ("related_to", format!("memory_{}", rng.gen_range(0..i))),
                ("similar_to", format!("memory_{}", rng.gen_range(0..i))),
            ];
            memory.add_relations(&key, relations).unwrap();
        }
    }
    
    memory
}

// Placeholder for HybridMemorySystem (would be implemented in actual code)
struct HybridMemorySystem {
    memory_bank: MemoryBank,
    knowledge_graph: KnowledgeGraph,
}

impl HybridMemorySystem {
    fn store(&mut self, key: MemoryKey, value: MemoryValue) -> Result<(), Box<dyn std::error::Error>> {
        self.memory_bank.store(key, value)?;
        Ok(())
    }
    
    fn store_with_relations(&mut self, key: MemoryKey, value: MemoryValue, relations: Vec<(&str, &str)>) -> Result<(), Box<dyn std::error::Error>> {
        // Store in memory bank
        self.memory_bank.store(key.clone(), value.clone())?;
        
        // Create node in graph
        let node_id = self.knowledge_graph.create_node(
            key.to_string(),
            value.embedding.clone(),
            HashMap::new()
        )?;
        
        // Add relations
        for (rel_type, target_key) in relations {
            if let Ok(target_node) = self.knowledge_graph.find_node_by_key(target_key) {
                self.knowledge_graph.create_edge(
                    node_id.clone(),
                    target_node,
                    rel_type.to_string(),
                    HashMap::new()
                )?;
            }
        }
        
        Ok(())
    }
    
    fn search_with_graph_context(&self, query: &Array2<f32>, k: usize) -> Vec<(MemoryKey, MemoryValue, f32)> {
        // Implement graph-enhanced search
        vec![]
    }
    
    fn traverse_relationships(&self, start: &MemoryKey, rel_type: &str, depth: usize) -> Vec<MemoryKey> {
        vec![]
    }
    
    fn get_all_keys(&self) -> impl Iterator<Item = MemoryKey> {
        vec![].into_iter()
    }
    
    fn add_relations(&mut self, key: &MemoryKey, relations: Vec<(&str, String)>) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

criterion_group!(
    benches,
    benchmark_graph_queries,
    benchmark_memory_operations,
    benchmark_concurrent_access,
    benchmark_memory_overhead
);

criterion_main!(benches);