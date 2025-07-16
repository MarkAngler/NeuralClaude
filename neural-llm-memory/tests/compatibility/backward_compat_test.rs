//! Backward compatibility tests for knowledge graph integration
//! Ensures existing APIs and functionality remain intact

use neural_llm_memory::{
    MemoryModule, MemoryConfig, MemoryBank, MemoryOperations,
    memory::{MemoryKey, MemoryValue, MemoryMetadata},
    adaptive::AdaptiveMemoryModule,
    mcp::{MCPServer, MCPRequest, MCPResponse},
};
use ndarray::Array2;
use serde_json::json;
use std::path::Path;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_existing_memory_module_api() {
        // Test: Original MemoryModule API still works
        let config = MemoryConfig {
            memory_size: 100,
            embedding_dim: 64,
            hidden_dim: 128,
            num_heads: 4,
            num_layers: 2,
            dropout_rate: 0.1,
            max_sequence_length: 50,
            use_positional_encoding: true,
        };
        
        let mut memory_module = MemoryModule::new(config.clone());
        
        // Test: Store operation unchanged
        let embedding = Array2::ones((1, config.embedding_dim));
        let key = memory_module.store_memory(
            "Legacy memory content".to_string(),
            embedding.clone()
        ).unwrap();
        
        // Test: Retrieve operation unchanged
        let query = Array2::ones((1, config.embedding_dim));
        let retrieved = memory_module.retrieve_with_attention(&query, 1);
        
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].1.content, "Legacy memory content");
        
        // Test: Stats API unchanged
        let (size, accesses, hits, rate) = memory_module.get_stats();
        assert_eq!(size, 1);
        assert!(accesses > 0);
    }
    
    #[test]
    fn test_memory_bank_compatibility() {
        // Test: MemoryBank trait implementations unchanged
        let mut bank = MemoryBank::new(1000, 100);
        
        // Test: Store operation
        let key = MemoryKey::from("test_key");
        let value = MemoryValue {
            embedding: vec![0.1; 768],
            content: "Test content".to_string(),
            metadata: MemoryMetadata::default(),
        };
        
        bank.store(key.clone(), value.clone()).unwrap();
        
        // Test: Retrieve operation
        let retrieved = bank.retrieve(&key).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test content");
        
        // Test: Search operation
        let query = Array2::from_shape_vec((1, 768), vec![0.1; 768]).unwrap();
        let results = bank.search(&query, 5);
        assert!(!results.is_empty());
        
        // Test: Update operation
        bank.update(&key, |v| {
            v.content = "Updated content".to_string();
        }).unwrap();
        
        // Test: Delete operation
        let deleted = bank.delete(&key).unwrap();
        assert!(deleted);
    }
    
    #[test]
    fn test_adaptive_module_compatibility() {
        // Test: AdaptiveMemoryModule API unchanged
        let mut adaptive = AdaptiveMemoryModule::new(Default::default());
        
        // Test: Store operation
        adaptive.store(
            "adaptive_key".to_string(),
            vec![0.2; 768],
            "Adaptive content".to_string()
        ).unwrap();
        
        // Test: Retrieve operation
        let retrieved = adaptive.retrieve("adaptive_key").unwrap();
        assert!(retrieved.is_some());
        
        // Test: Search operation
        let results = adaptive.search(vec![0.2; 768], 3).unwrap();
        assert!(!results.is_empty());
        
        // Test: Evolution still works
        adaptive.evolve(5).unwrap();
        
        // Test: Status API unchanged
        let status = adaptive.status(false).unwrap();
        assert!(status.contains("Generation"));
    }
    
    #[test]
    fn test_mcp_protocol_compatibility() {
        // Test: MCP server still accepts old protocol
        let server = MCPServer::new();
        
        // Test: Store via MCP
        let store_request = json!({
            "method": "store_memory",
            "params": {
                "key": "mcp_test",
                "content": "MCP test content"
            }
        });
        
        let response = server.handle_request(store_request).unwrap();
        assert!(response["success"].as_bool().unwrap());
        
        // Test: Retrieve via MCP
        let retrieve_request = json!({
            "method": "retrieve_memory",
            "params": {
                "key": "mcp_test"
            }
        });
        
        let response = server.handle_request(retrieve_request).unwrap();
        assert!(response["result"].is_object());
        
        // Test: Search via MCP
        let search_request = json!({
            "method": "search_memory",
            "params": {
                "query": "test",
                "limit": 5
            }
        });
        
        let response = server.handle_request(search_request).unwrap();
        assert!(response["result"].is_array());
    }
    
    #[test]
    fn test_data_migration_compatibility() {
        // Test: Can load old format data
        let old_format_data = json!({
            "memories": {
                "old_key_1": {
                    "embedding": vec![0.1; 768],
                    "content": "Old format content",
                    "metadata": {
                        "importance": 0.8,
                        "last_accessed": 1234567890,
                        "access_count": 5
                    }
                }
            }
        });
        
        // Test: Migration function works
        let migrated = migrate_to_graph_format(old_format_data).unwrap();
        assert!(migrated["memories"].is_object());
        assert!(migrated["graph"].is_object());
        
        // Test: Can still access migrated data
        let memory_module = MemoryModule::from_json(&migrated).unwrap();
        let old_key = MemoryKey::from("old_key_1");
        let retrieved = memory_module.retrieve_by_key(&old_key).unwrap();
        assert!(retrieved.is_some());
    }
    
    #[test]
    fn test_persistence_format_compatibility() {
        use tempfile::TempDir;
        
        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test_data");
        
        // Create memory module with old API
        let mut memory = MemoryModule::new(MemoryConfig::default());
        memory.store_memory(
            "Persistent content".to_string(),
            Array2::ones((1, 64))
        ).unwrap();
        
        // Save using old method
        memory.save_to_disk(&data_path).unwrap();
        
        // Load using potentially new implementation
        let loaded = MemoryModule::load_from_disk(&data_path).unwrap();
        
        // Verify data integrity
        let (size, _, _, _) = loaded.get_stats();
        assert_eq!(size, 1);
    }
    
    #[test]
    fn test_api_versioning() {
        // Test: Version detection works
        let request_v1 = json!({
            "version": "1.0",
            "method": "store_memory",
            "params": {
                "key": "v1_key",
                "content": "V1 content"
            }
        });
        
        let request_v2 = json!({
            "version": "2.0",
            "method": "store_memory",
            "params": {
                "key": "v2_key",
                "content": "V2 content",
                "relations": [
                    {"type": "related_to", "target": "other_key"}
                ]
            }
        });
        
        let server = MCPServer::new();
        
        // Both versions should work
        let response_v1 = server.handle_versioned_request(request_v1).unwrap();
        assert!(response_v1["success"].as_bool().unwrap());
        
        let response_v2 = server.handle_versioned_request(request_v2).unwrap();
        assert!(response_v2["success"].as_bool().unwrap());
    }
    
    #[test]
    fn test_evolution_algorithm_compatibility() {
        // Test: Evolution still works with graph-enhanced memory
        let mut adaptive = AdaptiveMemoryModule::new(Default::default());
        
        // Store memories with potential graph structure
        for i in 0..10 {
            adaptive.store(
                format!("evo_key_{}", i),
                vec![i as f32 / 10.0; 768],
                format!("Evolution test content {}", i)
            ).unwrap();
        }
        
        // Get baseline fitness
        let initial_fitness = adaptive.get_current_fitness().unwrap();
        
        // Run evolution
        adaptive.evolve(10).unwrap();
        
        // Verify evolution improved fitness
        let final_fitness = adaptive.get_current_fitness().unwrap();
        assert!(final_fitness >= initial_fitness);
        
        // Verify memories still accessible
        let retrieved = adaptive.retrieve("evo_key_5").unwrap();
        assert!(retrieved.is_some());
    }
    
    // Helper function for data migration
    fn migrate_to_graph_format(old_data: serde_json::Value) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut new_data = json!({
            "memories": old_data["memories"].clone(),
            "graph": {
                "nodes": {},
                "edges": []
            }
        });
        
        // Add graph nodes for each memory
        if let Some(memories) = old_data["memories"].as_object() {
            for (key, _) in memories {
                new_data["graph"]["nodes"][key] = json!({
                    "id": key,
                    "type": "memory"
                });
            }
        }
        
        Ok(new_data)
    }
}