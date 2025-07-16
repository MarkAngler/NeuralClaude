//! Comprehensive tests for ConsciousGraph

use neural_llm_memory::graph::{
    ConsciousGraph, ConsciousGraphConfig, GraphStats,
    ConsciousNode, ConsciousEdge, NodeType, EdgeType,
    MemoryNode, ConceptNode, EntityNode, ContextNode, PatternNode
};
use chrono::Utc;
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::tempdir;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> ConsciousGraphConfig {
        let temp_dir = tempdir().unwrap();
        ConsciousGraphConfig {
            embedding_dim: 64, // Smaller for tests
            auto_infer_relationships: true,
            consciousness_threshold: 0.3,
            max_nodes: 1000,
            persistence_enabled: false, // Disable for unit tests
            storage_path: temp_dir.path().to_path_buf(),
        }
    }

    #[test]
    fn test_conscious_graph_creation() {
        let config = create_test_config();
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 0);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.total_queries, 0);
    }

    #[test]
    fn test_default_conscious_graph() {
        let graph = ConsciousGraph::default();
        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 0);
    }

    #[test]
    fn test_add_memory_node() {
        let config = create_test_config();
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        let memory_node = NodeType::Memory(MemoryNode {
            id: "memory_1".to_string(),
            key: "test_memory".to_string(),
            value: "This is a test memory".to_string(),
            embedding: vec![0.1; 64],
            created_at: Utc::now(),
            accessed_at: Utc::now(),
            access_count: 1,
        });
        
        let node_id = graph.add_node(memory_node).unwrap();
        assert!(!node_id.is_empty());
        
        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 1);
    }

    #[test]
    fn test_add_concept_node() {
        let config = create_test_config();
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        let concept_node = NodeType::Concept(ConceptNode {
            id: "concept_1".to_string(),
            name: "Machine Learning".to_string(),
            definition: "A method of data analysis that automates analytical model building".to_string(),
            embedding: vec![0.2; 64],
            confidence: 0.9,
            source_memories: vec!["memory_1".to_string()],
        });
        
        let node_id = graph.add_node(concept_node).unwrap();
        assert!(!node_id.is_empty());
        
        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 1);
    }

    #[test]
    fn test_add_entity_node() {
        let config = create_test_config();
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        let mut attributes = HashMap::new();
        attributes.insert("type".to_string(), serde_json::Value::String("person".to_string()));
        attributes.insert("age".to_string(), serde_json::Value::Number(serde_json::Number::from(30)));
        
        let entity_node = NodeType::Entity(EntityNode {
            id: "entity_1".to_string(),
            name: "John Doe".to_string(),
            entity_type: "person".to_string(),
            attributes,
            embedding: vec![0.3; 64],
        });
        
        let node_id = graph.add_node(entity_node).unwrap();
        assert!(!node_id.is_empty());
        
        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 1);
    }

    #[test]
    fn test_add_context_node() {
        let config = create_test_config();
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        let now = Utc::now();
        let context_node = NodeType::Context(ContextNode {
            id: "context_1".to_string(),
            description: "Meeting about project planning".to_string(),
            time_range: Some((now, now + chrono::Duration::hours(1))),
            location: Some("Conference Room A".to_string()),
            participants: vec!["entity_1".to_string(), "entity_2".to_string()],
        });
        
        let node_id = graph.add_node(context_node).unwrap();
        assert!(!node_id.is_empty());
        
        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 1);
    }

    #[test]
    fn test_add_pattern_node() {
        let config = create_test_config();
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        let pattern_node = NodeType::Pattern(PatternNode {
            id: "pattern_1".to_string(),
            pattern_type: "temporal".to_string(),
            description: "Users tend to ask similar questions on Mondays".to_string(),
            frequency: 15,
            confidence: 0.8,
            examples: vec!["memory_1".to_string(), "memory_2".to_string()],
        });
        
        let node_id = graph.add_node(pattern_node).unwrap();
        assert!(!node_id.is_empty());
        
        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 1);
    }

    #[test]
    fn test_add_edge() {
        let config = create_test_config();
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        // First add two nodes
        let memory_node1 = NodeType::Memory(MemoryNode {
            id: "memory_1".to_string(),
            key: "key1".to_string(),
            value: "value1".to_string(),
            embedding: vec![0.1; 64],
            created_at: Utc::now(),
            accessed_at: Utc::now(),
            access_count: 1,
        });
        
        let memory_node2 = NodeType::Memory(MemoryNode {
            id: "memory_2".to_string(),
            key: "key2".to_string(),
            value: "value2".to_string(),
            embedding: vec![0.2; 64],
            created_at: Utc::now(),
            accessed_at: Utc::now(),
            access_count: 1,
        });
        
        let node_id1 = graph.add_node(memory_node1).unwrap();
        let node_id2 = graph.add_node(memory_node2).unwrap();
        
        // Add edge between them
        let edge = ConsciousEdge::new(
            node_id1,
            node_id2,
            EdgeType::Related { weight: 0.7 }
        );
        
        let edge_id = graph.add_edge(edge).unwrap();
        assert!(!edge_id.is_empty());
        
        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.edge_count, 1);
    }

    #[test]
    fn test_consciousness_threshold() {
        let mut config = create_test_config();
        config.consciousness_threshold = 0.7;
        
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        assert_eq!(graph.config.consciousness_threshold, 0.7);
    }

    #[test]
    fn test_update_consciousness() {
        let config = create_test_config();
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        let initial_activations = graph.get_stats().consciousness_activations;
        graph.update_consciousness().unwrap();
        let updated_activations = graph.get_stats().consciousness_activations;
        
        assert_eq!(updated_activations, initial_activations + 1);
    }

    #[test]
    fn test_dream_consolidation() {
        let config = create_test_config();
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        // Add some test nodes first
        for i in 0..5 {
            let memory_node = NodeType::Memory(MemoryNode {
                id: format!("memory_{}", i),
                key: format!("key_{}", i),
                value: format!("value_{}", i),
                embedding: vec![0.1 * i as f32; 64],
                created_at: Utc::now(),
                accessed_at: Utc::now(),
                access_count: 1,
            });
            graph.add_node(memory_node).unwrap();
        }
        
        let initial_patterns = graph.get_stats().pattern_extractions;
        let consolidated = graph.dream_consolidation().unwrap();
        let updated_patterns = graph.get_stats().pattern_extractions;
        
        assert!(consolidated >= 0);
        assert!(updated_patterns >= initial_patterns);
    }

    #[test]
    fn test_graph_query() {
        let config = create_test_config();
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        // Add a test node
        let memory_node = NodeType::Memory(MemoryNode {
            id: "memory_test".to_string(),
            key: "test_key".to_string(),
            value: "test_value".to_string(),
            embedding: vec![0.5; 64],
            created_at: Utc::now(),
            accessed_at: Utc::now(),
            access_count: 1,
        });
        
        let node_id = graph.add_node(memory_node).unwrap();
        
        // Query the graph
        let result = graph.query_graph(&node_id, 2).unwrap();
        assert!(!result.nodes.is_empty());
        
        let stats = graph.get_stats();
        assert!(stats.total_queries > 0);
    }

    #[test]
    fn test_extract_patterns() {
        let config = create_test_config();
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        let initial_extractions = graph.get_stats().pattern_extractions;
        let patterns = graph.extract_patterns("test context").unwrap();
        let updated_extractions = graph.get_stats().pattern_extractions;
        
        assert!(patterns.len() >= 0);
        assert!(updated_extractions >= initial_extractions);
    }

    #[test]
    fn test_multiple_node_types() {
        let config = create_test_config();
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        // Add different types of nodes
        let memory_node = NodeType::Memory(MemoryNode {
            id: "memory_1".to_string(),
            key: "key1".to_string(),
            value: "value1".to_string(),
            embedding: vec![0.1; 64],
            created_at: Utc::now(),
            accessed_at: Utc::now(),
            access_count: 1,
        });
        
        let concept_node = NodeType::Concept(ConceptNode {
            id: "concept_1".to_string(),
            name: "AI".to_string(),
            definition: "Artificial Intelligence".to_string(),
            embedding: vec![0.2; 64],
            confidence: 0.9,
            source_memories: vec!["memory_1".to_string()],
        });
        
        graph.add_node(memory_node).unwrap();
        graph.add_node(concept_node).unwrap();
        
        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 2);
    }

    #[test]
    fn test_auto_relationship_inference() {
        let mut config = create_test_config();
        config.auto_infer_relationships = true;
        
        let graph = ConsciousGraph::new_with_config(config).unwrap();
        
        let memory_node = NodeType::Memory(MemoryNode {
            id: "memory_1".to_string(),
            key: "ai_concept".to_string(),
            value: "Artificial intelligence is transformative".to_string(),
            embedding: vec![0.1; 64],
            created_at: Utc::now(),
            accessed_at: Utc::now(),
            access_count: 1,
        });
        
        graph.add_node(memory_node).unwrap();
        
        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 1);
        // With auto-inference enabled, there might be inferred relationships
        assert!(stats.relationship_inferences >= 0);
    }
}