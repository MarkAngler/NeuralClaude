use neural_llm_memory::memory::MemoryConfig;
use neural_llm_memory::adaptive::AdaptiveMemoryModule;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Memory Update/Delete Demo");
    println!("===================================\n");
    
    // Create adaptive memory module with default config
    let config = MemoryConfig::default();
    let module = AdaptiveMemoryModule::new(config).await?;
    
    // 1. Store some initial memories
    println!("📝 Storing initial memories...");
    let memories = vec![
        ("user_preferences", "Theme: dark, Language: English, Notifications: enabled"),
        ("project_context", "Working on neural memory system with adaptive learning"),
        ("session_data", "Last login: 2025-01-10, Session duration: 2 hours"),
    ];
    
    for (key, content) in &memories {
        let operation_id = module.store(key, content).await?;
        println!("  ✅ Stored '{}' (operation: {})", key, operation_id);
    }
    
    // 2. Retrieve and display current state
    println!("\n📖 Current memory state:");
    for (key, _) in &memories {
        match module.retrieve(key).await? {
            (Some(content), _) => println!("  - {}: {}", key, content),
            (None, _) => println!("  - {}: [not found]", key),
        }
    }
    
    // 3. Update some memories
    println!("\n✏️  Updating memories...");
    let updates = vec![
        ("user_preferences", "Theme: light, Language: Spanish, Notifications: disabled"),
        ("session_data", "Last login: 2025-01-10, Session duration: 5 hours, Active: true"),
    ];
    
    for (key, new_content) in updates {
        match module.update(key, new_content).await {
            Ok(operation_id) => println!("  ✅ Updated '{}' (operation: {})", key, operation_id),
            Err(e) => println!("  ❌ Failed to update '{}': {}", key, e),
        }
    }
    
    // 4. Try to update non-existent key
    println!("\n🔍 Testing update on non-existent key:");
    match module.update("non_existent_key", "some content").await {
        Ok(_) => println!("  ✅ Unexpectedly succeeded"),
        Err(e) => println!("  ✅ Expected error: {}", e),
    }
    
    // 5. Show updated state
    println!("\n📖 Memory state after updates:");
    for (key, _) in &memories {
        match module.retrieve(key).await? {
            (Some(content), _) => println!("  - {}: {}", key, content),
            (None, _) => println!("  - {}: [not found]", key),
        }
    }
    
    // 6. Delete some memories
    println!("\n🗑️  Deleting memories...");
    let to_delete = vec!["session_data", "non_existent_key"];
    
    for key in to_delete {
        match module.delete(key).await? {
            (true, operation_id) => println!("  ✅ Deleted '{}' (operation: {})", key, operation_id),
            (false, operation_id) => println!("  ⚠️  Key '{}' not found (operation: {})", key, operation_id),
        }
    }
    
    // 7. Final state
    println!("\n📖 Final memory state:");
    let all_keys = vec!["user_preferences", "project_context", "session_data"];
    for key in all_keys {
        match module.retrieve(key).await? {
            (Some(content), _) => println!("  - {}: {}", key, content),
            (None, _) => println!("  - {}: [deleted]", key),
        }
    }
    
    // 8. Show statistics
    println!("\n📊 Memory Statistics:");
    let stats = module.get_stats().await?;
    if let Some(usage_stats) = stats.get("usage_stats") {
        if let Some(operation_counts) = usage_stats.get("operation_counts") {
            println!("  Operation counts: {}", 
                serde_json::to_string_pretty(operation_counts)?);
        }
        if let Some(total_ops) = usage_stats.get("total_operations") {
            println!("  Total operations: {}", total_ops);
        }
    }
    
    println!("\n✨ Demo completed successfully!");
    Ok(())
}