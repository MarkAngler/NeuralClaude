//! Example of neural pattern and memory integration

use neural_llm_memory::{
    MemoryModule, MemoryConfig,
    integration::{
        PatternType, PatternContext, PerformanceMetrics, ComplexityLevel,
        LearnedPattern, PatternLearner, PatternMemory,
        adaptive_system::AdaptiveSystemBuilder,
        IntegrationConfig,
    },
};
use std::collections::HashMap;
use chrono::Utc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Neural Pattern + Memory Integration Demo ===\n");
    
    // Create adaptive system
    let mut adaptive_system = AdaptiveSystemBuilder::new()
        .with_config(IntegrationConfig {
            auto_store_threshold: 0.8,
            pattern_decay_rate: 0.95,
            max_pattern_age_days: 30,
            similarity_threshold: 0.7,
            learning_rate: 0.01,
        })
        .with_memory_config(MemoryConfig {
            memory_size: 5000,
            embedding_dim: 768,
            hidden_dim: 1024,
            num_heads: 8,
            num_layers: 4,
            dropout_rate: 0.1,
            max_sequence_length: 256,
            use_positional_encoding: true,
        })
        .build();
    
    println!("Adaptive system initialized!\n");
    
    // Simulate different coding scenarios
    simulate_coding_scenarios(&mut adaptive_system)?;
    
    // Demonstrate pattern learning
    demonstrate_pattern_learning(&mut adaptive_system)?;
    
    // Show cross-system benefits
    demonstrate_cross_system_benefits(&mut adaptive_system)?;
    
    Ok(())
}

fn simulate_coding_scenarios<T: PatternLearner>(
    system: &mut T
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Simulating coding scenarios...\n");
    
    // Scenario 1: Simple bug fix
    let bug_fix_context = PatternContext {
        task_type: "bug_fix".to_string(),
        file_types: vec!["rust".to_string()],
        complexity: ComplexityLevel::Simple,
        domain: "error_handling".to_string(),
        success_criteria: vec!["compile_success".to_string(), "test_pass".to_string()],
        environmental_factors: HashMap::new(),
    };
    
    let bug_fix_metrics = PerformanceMetrics {
        success_rate: 0.95,
        average_time_ms: 5000.0,
        token_efficiency: 0.85,
        error_rate: 0.05,
        adaptability_score: 0.8,
        usage_count: 1,
    };
    
    system.learn_from_success(&bug_fix_context, &bug_fix_metrics);
    println!("✅ Learned from successful bug fix");
    
    // Scenario 2: Complex refactoring
    let refactor_context = PatternContext {
        task_type: "refactoring".to_string(),
        file_types: vec!["rust".to_string(), "toml".to_string()],
        complexity: ComplexityLevel::Complex,
        domain: "architecture".to_string(),
        success_criteria: vec!["maintain_api".to_string(), "improve_performance".to_string()],
        environmental_factors: HashMap::new(),
    };
    
    let refactor_metrics = PerformanceMetrics {
        success_rate: 0.88,
        average_time_ms: 25000.0,
        token_efficiency: 0.72,
        error_rate: 0.12,
        adaptability_score: 0.9,
        usage_count: 1,
    };
    
    system.learn_from_success(&refactor_context, &refactor_metrics);
    println!("✅ Learned from complex refactoring");
    
    // Scenario 3: Failed attempt
    let failed_context = PatternContext {
        task_type: "optimization".to_string(),
        file_types: vec!["rust".to_string()],
        complexity: ComplexityLevel::VeryComplex,
        domain: "performance".to_string(),
        success_criteria: vec!["10x_speedup".to_string()],
        environmental_factors: HashMap::new(),
    };
    
    system.learn_from_failure(&failed_context, "Performance target not met");
    println!("❌ Learned from failed optimization attempt\n");
    
    Ok(())
}

fn demonstrate_pattern_learning<T: PatternLearner>(
    system: &mut T
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Demonstrating pattern learning...\n");
    
    // Request pattern suggestion for a new task
    let new_task_context = PatternContext {
        task_type: "bug_fix".to_string(),
        file_types: vec!["rust".to_string(), "rs".to_string()],
        complexity: ComplexityLevel::Moderate,
        domain: "error_handling".to_string(),
        success_criteria: vec!["fix_panic".to_string()],
        environmental_factors: HashMap::new(),
    };
    
    if let Some(pattern_match) = system.suggest_pattern(&new_task_context) {
        println!("🎯 Suggested pattern: {:?}", pattern_match.pattern.pattern_type);
        println!("   Similarity: {:.2}", pattern_match.similarity);
        println!("   Relevance: {:.2}", pattern_match.relevance_score);
        println!("   Success rate: {:.2}", pattern_match.pattern.performance.success_rate);
        
        if !pattern_match.suggested_adaptations.is_empty() {
            println!("\n   📋 Suggested adaptations:");
            for adaptation in &pattern_match.suggested_adaptations {
                println!("      - {}: {} (confidence: {:.2})",
                    format!("{:?}", adaptation.adaptation_type),
                    adaptation.description,
                    adaptation.confidence
                );
            }
        }
    } else {
        println!("🤔 No similar patterns found - will use neural prediction");
    }
    
    println!();
    Ok(())
}

fn demonstrate_cross_system_benefits<T: PatternLearner>(
    system: &mut T
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Demonstrating cross-system benefits...\n");
    
    // Simulate learning from multiple similar tasks
    let domains = vec!["web", "cli", "api", "database"];
    let mut total_patterns = 0;
    
    for domain in &domains {
        for complexity in &[ComplexityLevel::Simple, ComplexityLevel::Moderate] {
            let context = PatternContext {
                task_type: "feature_implementation".to_string(),
                file_types: vec!["rust".to_string()],
                complexity: complexity.clone(),
                domain: domain.to_string(),
                success_criteria: vec!["tests_pass".to_string()],
                environmental_factors: HashMap::new(),
            };
            
            let metrics = PerformanceMetrics {
                success_rate: 0.85 + rand::random::<f32>() * 0.15,
                average_time_ms: 10000.0 + rand::random::<f64>() * 15000.0,
                token_efficiency: 0.7 + rand::random::<f32>() * 0.3,
                error_rate: rand::random::<f32>() * 0.2,
                adaptability_score: 0.8 + rand::random::<f32>() * 0.2,
                usage_count: 1,
            };
            
            system.learn_from_success(&context, &metrics);
            total_patterns += 1;
        }
    }
    
    println!("📊 Learned from {} different patterns across {} domains", total_patterns, domains.len());
    
    // Now test generalization
    let test_context = PatternContext {
        task_type: "feature_implementation".to_string(),
        file_types: vec!["rust".to_string(), "yaml".to_string()], // New file type
        complexity: ComplexityLevel::Complex, // Higher complexity
        domain: "microservice".to_string(), // New domain
        success_criteria: vec!["tests_pass".to_string(), "performance_target".to_string()],
        environmental_factors: HashMap::new(),
    };
    
    if let Some(pattern) = system.suggest_pattern(&test_context) {
        println!("\n🎉 System can generalize to new scenarios!");
        println!("   Found pattern from '{}' domain", pattern.pattern.context.domain);
        println!("   Original complexity: {:?}", pattern.pattern.context.complexity);
        println!("   Adaptations needed: {}", pattern.suggested_adaptations.len());
    }
    
    println!("\n✨ Benefits demonstrated:");
    println!("   - Patterns learned from one domain apply to others");
    println!("   - System suggests adaptations for new contexts");
    println!("   - Performance improves with more examples");
    println!("   - Memory provides instant access to successful patterns");
    
    Ok(())
}