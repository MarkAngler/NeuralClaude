//! Test enhanced search functionality

use neural_llm_memory::{
    adaptive::{AdaptiveMemoryModule, enhanced_search},
    graph::{ConsciousGraph, ConsciousGraphConfig},
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Testing enhanced search with graph traversal...\n");
    
    // Initialize modules
    let module = AdaptiveMemoryModule::new().await?;
    let graph = Arc::new(ConsciousGraph::new_with_config(ConsciousGraphConfig::default())?);
    
    // Store some test memories
    println!("Storing test memories...");
    module.store("project/auth/design", "JWT authentication design document").await?;
    module.store("project/auth/implementation", "JWT implementation with bcrypt").await?;
    module.store("project/auth/testing", "Authentication test suite").await?;
    
    // Test regular search
    println!("\n1. Regular search (no graph traversal):");
    let (results, _) = module.search("JWT authentication", 3).await?;
    for (key, score) in results {
        println!("  - {} (score: {:.3})", key, score);
    }
    
    // Test enhanced search with graph traversal
    println!("\n2. Enhanced search (with graph traversal):");
    let enhanced_result = enhanced_search(
        &module,
        &graph,
        "JWT authentication",
        3,      // limit
        true,   // include_related
        2,      // traversal_depth
        10,     // max_related
        None,   // follow all edge types
    ).await?;
    
    println!("Query: {}", enhanced_result.query);
    println!("Operation ID: {}", enhanced_result.operation_id);
    println!("Matches:");
    
    for (i, match_result) in enhanced_result.matches.iter().enumerate() {
        println!("\n  {}. {} (score: {:.3})", i + 1, match_result.key, match_result.score);
        println!("     Type: {:?}", match_result.match_type);
        if let Some(rel) = &match_result.relationship {
            println!("     Relationship: {}", rel.description);
            println!("     From: {}", rel.path_from);
        }
        println!("     Distance: {:?}", match_result.distance);
    }
    
    Ok(())
}