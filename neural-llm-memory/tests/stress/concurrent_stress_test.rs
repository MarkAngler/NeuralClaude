//! Stress tests for concurrent access patterns and edge cases
//! Tests system behavior under extreme load and failure conditions

use neural_llm_memory::{
    graph::{KnowledgeGraph, NodeId},
    memory::{MemoryBank, MemoryKey, MemoryValue, MemoryMetadata, MemoryOperations},
};
use std::sync::{Arc, Barrier, atomic::{AtomicUsize, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use rand::Rng;
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_thundering_herd_graph_creation() {
        // Test: Many threads trying to create same nodes simultaneously
        let graph = Arc::new(KnowledgeGraph::new_concurrent());
        let barrier = Arc::new(Barrier::new(100));
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];
        
        for i in 0..100 {
            let graph_clone = Arc::clone(&graph);
            let barrier_clone = Arc::clone(&barrier);
            let success_clone = Arc::clone(&success_count);
            
            let handle = thread::spawn(move || {
                // Wait for all threads to be ready
                barrier_clone.wait();
                
                // All threads try to create same node
                let result = graph_clone.create_node(
                    "shared_node".to_string(),
                    vec![0.1; 768],
                    HashMap::new()
                );
                
                if result.is_ok() {
                    success_clone.fetch_add(1, Ordering::Relaxed);
                }
                
                // Also create unique nodes
                for j in 0..10 {
                    let _ = graph_clone.create_node(
                        format!("thread_{}_node_{}", i, j),
                        vec![i as f32, j as f32],
                        HashMap::new()
                    );
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify: Only one thread should succeed with shared node
        assert_eq!(success_count.load(Ordering::Relaxed), 1);
        
        // Verify: All unique nodes created
        assert_eq!(graph.node_count(), 1001); // 1 shared + 1000 unique
    }
    
    #[test]
    fn test_cascading_updates_stress() {
        // Test: Updates that trigger cascading effects
        let graph = Arc::new(KnowledgeGraph::new_concurrent());
        let update_count = Arc::new(AtomicUsize::new(0));
        
        // Create interconnected graph
        let mut nodes = vec![];
        for i in 0..100 {
            let node = graph.create_node(
                format!("cascade_{}", i),
                vec![i as f32; 10],
                HashMap::from([("value".to_string(), i.to_string())])
            ).unwrap();
            nodes.push(node);
        }
        
        // Create dense connections
        for i in 0..100 {
            for j in (i+1)..std::cmp::min(i+5, 100) {
                graph.create_edge(
                    nodes[i].clone(),
                    nodes[j].clone(),
                    "triggers".to_string(),
                    HashMap::new()
                ).unwrap();
            }
        }
        
        // Spawn threads doing cascading updates
        let mut handles = vec![];
        for t in 0..10 {
            let graph_clone = Arc::clone(&graph);
            let update_clone = Arc::clone(&update_count);
            let nodes_clone = nodes.clone();
            
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let start_idx = t * 10;
                    
                    // Update a node and cascade to neighbors
                    if let Ok(neighbors) = graph_clone.get_neighbors(&nodes_clone[start_idx], Some("triggers")) {
                        for neighbor in neighbors {
                            graph_clone.update_node_property(
                                &neighbor,
                                "updated_by",
                                &format!("thread_{}", t)
                            ).ok();
                            update_clone.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify: System remained consistent
        assert!(update_count.load(Ordering::Relaxed) > 0);
        assert_eq!(graph.node_count(), 100); // No nodes lost
    }
    
    #[test]
    fn test_memory_pressure_with_graph_operations() {
        // Test: System behavior near memory limits
        let mut memory_bank = MemoryBank::new(100, 10); // Small limits
        let graph = KnowledgeGraph::new();
        
        // Fill memory to capacity
        for i in 0..100 {
            let key = MemoryKey::from(format!("mem_{}", i));
            let value = MemoryValue {
                embedding: vec![i as f32; 768],
                content: format!("Content {}", i),
                metadata: MemoryMetadata {
                    importance: 1.0 - (i as f32 / 100.0), // Decreasing importance
                    ..Default::default()
                },
            };
            memory_bank.store(key, value).unwrap();
        }
        
        // Verify at capacity
        assert_eq!(memory_bank.size(), 100);
        
        // Add more memories - should trigger eviction
        for i in 100..200 {
            let key = MemoryKey::from(format!("mem_{}", i));
            let value = MemoryValue {
                embedding: vec![i as f32; 768],
                content: format!("Content {}", i),
                metadata: MemoryMetadata {
                    importance: 0.9, // High importance
                    ..Default::default()
                },
            };
            memory_bank.store(key, value).unwrap();
        }
        
        // Verify: Still at capacity
        assert_eq!(memory_bank.size(), 100);
        
        // Verify: Low importance memories were evicted
        assert!(memory_bank.retrieve(&MemoryKey::from("mem_99")).unwrap().is_none());
        assert!(memory_bank.retrieve(&MemoryKey::from("mem_150")).unwrap().is_some());
    }
    
    #[test]
    fn test_rapid_growth_shrink_cycles() {
        // Test: Rapid expansion and contraction
        let graph = Arc::new(KnowledgeGraph::new_concurrent());
        let mut handles = vec![];
        
        for cycle in 0..5 {
            // Growth phase
            let graph_clone = Arc::clone(&graph);
            let handle = thread::spawn(move || {
                let mut local_nodes = vec![];
                
                // Rapidly create nodes
                for i in 0..1000 {
                    if let Ok(node) = graph_clone.create_node(
                        format!("cycle_{}_node_{}", cycle, i),
                        vec![cycle as f32, i as f32],
                        HashMap::new()
                    ) {
                        local_nodes.push(node);
                    }
                }
                
                // Create edges
                for i in 0..local_nodes.len()-1 {
                    let _ = graph_clone.create_edge(
                        local_nodes[i].clone(),
                        local_nodes[i+1].clone(),
                        "next".to_string(),
                        HashMap::new()
                    );
                }
                
                // Shrink phase - delete half
                for i in (0..local_nodes.len()).step_by(2) {
                    let _ = graph_clone.delete_node(&local_nodes[i]);
                }
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify: Graph structure remains valid
        assert!(graph.validate_consistency().is_ok());
    }
    
    #[test]
    fn test_deadlock_prevention() {
        // Test: Operations that could cause deadlocks
        let graph = Arc::new(KnowledgeGraph::new_concurrent());
        let memory = Arc::new(MemoryBank::new(1000, 100));
        let deadlock_detected = Arc::new(AtomicUsize::new(0));
        
        // Create circular dependencies
        let n1 = graph.create_node("deadlock_1".to_string(), vec![0.1; 10], HashMap::new()).unwrap();
        let n2 = graph.create_node("deadlock_2".to_string(), vec![0.2; 10], HashMap::new()).unwrap();
        let n3 = graph.create_node("deadlock_3".to_string(), vec![0.3; 10], HashMap::new()).unwrap();
        
        graph.create_edge(n1.clone(), n2.clone(), "locks".to_string(), HashMap::new()).unwrap();
        graph.create_edge(n2.clone(), n3.clone(), "locks".to_string(), HashMap::new()).unwrap();
        graph.create_edge(n3.clone(), n1.clone(), "locks".to_string(), HashMap::new()).unwrap();
        
        // Spawn threads that could deadlock
        let mut handles = vec![];
        for i in 0..3 {
            let graph_clone = Arc::clone(&graph);
            let deadlock_clone = Arc::clone(&deadlock_detected);
            let nodes = vec![n1.clone(), n2.clone(), n3.clone()];
            
            let handle = thread::spawn(move || {
                let start = Instant::now();
                let timeout = Duration::from_secs(5);
                
                // Try operations that traverse the cycle
                for _ in 0..100 {
                    let idx = i % 3;
                    let next_idx = (i + 1) % 3;
                    
                    // Operation that could deadlock
                    let result = graph_clone.atomic_update(
                        &nodes[idx],
                        &nodes[next_idx],
                        |n1, n2| {
                            // Simulate complex operation
                            thread::sleep(Duration::from_micros(100));
                        }
                    );
                    
                    if start.elapsed() > timeout {
                        deadlock_clone.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                }
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify: No deadlocks detected
        assert_eq!(deadlock_detected.load(Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_performance_under_extreme_load() {
        // Test: Maintain <50ms requirement under extreme conditions
        let graph = Arc::new(KnowledgeGraph::new_concurrent());
        let barrier = Arc::new(Barrier::new(50));
        let mut handles = vec![];
        let violations = Arc::new(AtomicUsize::new(0));
        
        // Create large graph
        for i in 0..10000 {
            graph.create_node(
                format!("load_test_{}", i),
                vec![i as f32 / 10000.0; 768],
                HashMap::new()
            ).unwrap();
        }
        
        // Spawn many concurrent readers
        for t in 0..50 {
            let graph_clone = Arc::clone(&graph);
            let barrier_clone = Arc::clone(&barrier);
            let violations_clone = Arc::clone(&violations);
            
            let handle = thread::spawn(move || {
                let mut rng = rand::thread_rng();
                barrier_clone.wait();
                
                // Perform many operations
                for _ in 0..1000 {
                    let node_id = NodeId::from(format!("load_test_{}", rng.gen_range(0..10000)));
                    
                    let start = Instant::now();
                    let _ = graph_clone.find_neighbors_n_hops(&node_id, 2);
                    let elapsed = start.elapsed();
                    
                    if elapsed > Duration::from_millis(50) {
                        violations_clone.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify: Less than 1% violations acceptable under extreme load
        let total_ops = 50 * 1000;
        let violation_count = violations.load(Ordering::Relaxed);
        let violation_rate = violation_count as f64 / total_ops as f64;
        
        assert!(violation_rate < 0.01, 
               "Too many performance violations: {} out of {} ({:.2}%)", 
               violation_count, total_ops, violation_rate * 100.0);
    }
    
    #[test]
    fn test_recovery_from_partial_corruption() {
        // Test: System can recover from partial data corruption
        let mut graph = KnowledgeGraph::new();
        
        // Create valid structure
        let mut nodes = vec![];
        for i in 0..100 {
            let node = graph.create_node(
                format!("node_{}", i),
                vec![i as f32; 10],
                HashMap::new()
            ).unwrap();
            nodes.push(node);
        }
        
        // Create edges
        for i in 0..99 {
            graph.create_edge(
                nodes[i].clone(),
                nodes[i+1].clone(),
                "next".to_string(),
                HashMap::new()
            ).unwrap();
        }
        
        // Simulate corruption by forcefully removing some internal data
        graph.simulate_corruption(10); // Remove 10% of internal structures
        
        // Test: Validation detects corruption
        assert!(graph.validate_consistency().is_err());
        
        // Test: Recovery process
        let recovered = graph.recover_from_corruption().unwrap();
        
        // Verify: Most data recovered
        assert!(recovered.node_count() >= 80); // At least 80% recovered
        assert!(recovered.validate_consistency().is_ok());
    }
}