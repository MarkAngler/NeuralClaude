//! Unit tests for core graph operations
//! Tests fundamental graph functionality in isolation

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{KnowledgeGraph, Node, Edge, NodeId, EdgeId, GraphError};
    use crate::memory::{MemoryKey, MemoryValue, MemoryMetadata};
    
    #[test]
    fn test_node_creation_and_retrieval() {
        let mut graph = KnowledgeGraph::new();
        
        // Test: Create node with properties
        let node_id = graph.create_node(
            "test_memory_1".to_string(),
            vec![0.1, 0.2, 0.3, 0.4], // embedding
            HashMap::from([
                ("type".to_string(), "concept".to_string()),
                ("importance".to_string(), "0.8".to_string()),
            ])
        ).unwrap();
        
        // Verify: Node exists and properties are correct
        let node = graph.get_node(&node_id).unwrap();
        assert_eq!(node.memory_key, "test_memory_1");
        assert_eq!(node.properties.get("type").unwrap(), "concept");
        assert_eq!(node.embedding.len(), 4);
        
        // Edge case: Try to create duplicate node
        let duplicate_result = graph.create_node(
            "test_memory_1".to_string(),
            vec![0.1, 0.2, 0.3, 0.4],
            HashMap::new()
        );
        assert!(duplicate_result.is_err());
    }
    
    #[test]
    fn test_edge_creation_and_validation() {
        let mut graph = KnowledgeGraph::new();
        
        // Create two nodes
        let node1 = graph.create_node(
            "node1".to_string(),
            vec![0.1; 10],
            HashMap::new()
        ).unwrap();
        
        let node2 = graph.create_node(
            "node2".to_string(),
            vec![0.2; 10],
            HashMap::new()
        ).unwrap();
        
        // Test: Create edge between nodes
        let edge_id = graph.create_edge(
            node1.clone(),
            node2.clone(),
            "related_to".to_string(),
            HashMap::from([
                ("weight".to_string(), "0.9".to_string()),
                ("bidirectional".to_string(), "true".to_string()),
            ])
        ).unwrap();
        
        // Verify: Edge exists with correct properties
        let edge = graph.get_edge(&edge_id).unwrap();
        assert_eq!(edge.from_node, node1);
        assert_eq!(edge.to_node, node2);
        assert_eq!(edge.relation_type, "related_to");
        
        // Test: Bidirectional edge access
        let neighbors = graph.get_neighbors(&node2, Some("related_to")).unwrap();
        assert!(neighbors.contains(&node1));
        
        // Edge case: Self-loop
        let self_loop = graph.create_edge(
            node1.clone(),
            node1.clone(),
            "self_reference".to_string(),
            HashMap::new()
        );
        assert!(self_loop.is_ok());
        
        // Edge case: Multiple edges between same nodes
        let second_edge = graph.create_edge(
            node1.clone(),
            node2.clone(),
            "another_relation".to_string(),
            HashMap::new()
        );
        assert!(second_edge.is_ok());
    }
    
    #[test]
    fn test_graph_traversal_algorithms() {
        let mut graph = create_test_graph(100);
        let start_node = NodeId::from("node_0");
        
        // Test: BFS traversal
        let start = Instant::now();
        let bfs_result = graph.breadth_first_search(&start_node, 3).unwrap();
        let bfs_time = start.elapsed();
        
        assert!(bfs_time < Duration::from_millis(50), "BFS took {:?}, expected <50ms", bfs_time);
        assert!(bfs_result.len() > 0);
        assert!(bfs_result.len() <= 100); // Should not exceed total nodes
        
        // Test: DFS traversal
        let start = Instant::now();
        let dfs_result = graph.depth_first_search(&start_node, 3).unwrap();
        let dfs_time = start.elapsed();
        
        assert!(dfs_time < Duration::from_millis(50), "DFS took {:?}, expected <50ms", dfs_time);
        
        // Test: Shortest path
        let target_node = NodeId::from("node_50");
        let start = Instant::now();
        let path = graph.shortest_path(&start_node, &target_node).unwrap();
        let path_time = start.elapsed();
        
        assert!(path_time < Duration::from_millis(50), "Shortest path took {:?}, expected <50ms", path_time);
        assert!(path.len() >= 2); // At least start and end
        assert_eq!(path[0], start_node);
        assert_eq!(path[path.len() - 1], target_node);
    }
    
    #[test]
    fn test_subgraph_extraction() {
        let mut graph = create_test_graph(50);
        let center_node = NodeId::from("node_25");
        
        // Test: Extract neighborhood subgraph
        let subgraph = graph.extract_subgraph(&center_node, 2).unwrap();
        
        // Verify: Subgraph contains center and neighbors
        assert!(subgraph.contains_node(&center_node));
        assert!(subgraph.node_count() > 1);
        assert!(subgraph.node_count() < graph.node_count());
        
        // Test: Connected components
        let components = graph.find_connected_components().unwrap();
        assert!(components.len() >= 1);
        
        // Edge case: Single node subgraph
        let isolated_node = graph.create_node(
            "isolated".to_string(),
            vec![0.0; 10],
            HashMap::new()
        ).unwrap();
        
        let single_subgraph = graph.extract_subgraph(&isolated_node, 1).unwrap();
        assert_eq!(single_subgraph.node_count(), 1);
    }
    
    #[test]
    fn test_pattern_matching() {
        let mut graph = KnowledgeGraph::new();
        
        // Create a specific pattern
        let n1 = graph.create_node("concept_1".to_string(), vec![0.1; 10], 
            HashMap::from([("type".to_string(), "concept".to_string())])).unwrap();
        let n2 = graph.create_node("concept_2".to_string(), vec![0.2; 10],
            HashMap::from([("type".to_string(), "concept".to_string())])).unwrap();
        let n3 = graph.create_node("instance_1".to_string(), vec![0.3; 10],
            HashMap::from([("type".to_string(), "instance".to_string())])).unwrap();
        
        graph.create_edge(n1.clone(), n2.clone(), "related_to".to_string(), HashMap::new()).unwrap();
        graph.create_edge(n2.clone(), n3.clone(), "has_instance".to_string(), HashMap::new()).unwrap();
        
        // Test: Match pattern
        let pattern = GraphPattern {
            node_constraints: vec![
                NodeConstraint { property: "type".to_string(), value: "concept".to_string() },
                NodeConstraint { property: "type".to_string(), value: "instance".to_string() },
            ],
            edge_constraints: vec![
                EdgeConstraint { relation_type: Some("has_instance".to_string()) },
            ],
        };
        
        let matches = graph.match_pattern(&pattern).unwrap();
        assert!(!matches.is_empty());
        assert!(matches[0].contains(&n2));
        assert!(matches[0].contains(&n3));
    }
    
    #[test]
    fn test_graph_metrics() {
        let graph = create_test_graph(100);
        
        // Test: Basic metrics
        assert_eq!(graph.node_count(), 100);
        assert!(graph.edge_count() > 0);
        
        // Test: Degree centrality
        let centrality = graph.calculate_degree_centrality().unwrap();
        assert_eq!(centrality.len(), 100);
        
        // Test: Clustering coefficient
        let clustering = graph.calculate_clustering_coefficient().unwrap();
        assert!(clustering >= 0.0 && clustering <= 1.0);
        
        // Test: Average path length
        let avg_path = graph.calculate_average_path_length(10).unwrap(); // Sample 10 pairs
        assert!(avg_path > 0.0);
    }
    
    #[test]
    fn test_concurrent_graph_modifications() {
        use std::sync::Arc;
        use std::thread;
        
        let graph = Arc::new(KnowledgeGraph::new_concurrent());
        let mut handles = vec![];
        
        // Spawn multiple threads doing modifications
        for i in 0..10 {
            let graph_clone = Arc::clone(&graph);
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let node_key = format!("thread_{}_node_{}", i, j);
                    graph_clone.create_node(
                        node_key,
                        vec![i as f32, j as f32],
                        HashMap::new()
                    ).unwrap();
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify: All nodes created
        assert_eq!(graph.node_count(), 100);
    }
    
    #[test]
    fn test_graph_persistence() {
        let mut graph = KnowledgeGraph::new();
        
        // Create some structure
        let n1 = graph.create_node("persist_1".to_string(), vec![0.1; 5], HashMap::new()).unwrap();
        let n2 = graph.create_node("persist_2".to_string(), vec![0.2; 5], HashMap::new()).unwrap();
        graph.create_edge(n1, n2, "persisted_edge".to_string(), HashMap::new()).unwrap();
        
        // Test: Serialize to JSON
        let serialized = graph.to_json().unwrap();
        assert!(!serialized.is_empty());
        
        // Test: Deserialize
        let restored = KnowledgeGraph::from_json(&serialized).unwrap();
        assert_eq!(restored.node_count(), 2);
        assert_eq!(restored.edge_count(), 1);
    }
    
    // Helper function to create test graphs
    fn create_test_graph(size: usize) -> KnowledgeGraph {
        let mut graph = KnowledgeGraph::new();
        let mut nodes = Vec::new();
        
        // Create nodes
        for i in 0..size {
            let node_id = graph.create_node(
                format!("node_{}", i),
                vec![i as f32 / size as f32; 10],
                HashMap::from([
                    ("index".to_string(), i.to_string()),
                    ("type".to_string(), if i % 2 == 0 { "even" } else { "odd" }.to_string()),
                ])
            ).unwrap();
            nodes.push(node_id);
        }
        
        // Create edges (simple ring + some random connections)
        for i in 0..size {
            let next = (i + 1) % size;
            graph.create_edge(
                nodes[i].clone(),
                nodes[next].clone(),
                "next".to_string(),
                HashMap::new()
            ).unwrap();
            
            // Add some random connections
            if i % 3 == 0 && i + 5 < size {
                graph.create_edge(
                    nodes[i].clone(),
                    nodes[i + 5].clone(),
                    "skip".to_string(),
                    HashMap::new()
                ).unwrap();
            }
        }
        
        graph
    }
}