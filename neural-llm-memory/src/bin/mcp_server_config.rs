// Configuration override for MCP server
use neural_llm_memory::adaptive::AdaptiveConfig;
use std::collections::HashMap;

pub fn get_adaptive_config() -> AdaptiveConfig {
    let mut objectives = HashMap::new();
    objectives.insert("response_time".to_string(), 0.4);
    objectives.insert("memory_efficiency".to_string(), 0.3);
    objectives.insert("accuracy".to_string(), 0.3);
    
    AdaptiveConfig {
        enabled: true,
        evolution_interval_hours: 24,
        evolution_after_operations: 100,  // Changed from default 1000
        min_training_samples: 50,         // Changed from default 500
        objectives,
        target_response_ms: 10.0,
        max_memory_mb: 512,
        min_accuracy: 0.85,
        population_size: 20,              // Changed from default 50 for faster evolution
        generations: 10,                  // Changed from default 20 for faster evolution
        mutation_rate: 0.05,
        crossover_rate: 0.8,
        evolution_thread_priority: neural_llm_memory::adaptive::ThreadPriority::Normal,
        max_evolution_time_minutes: 30,
    }
}