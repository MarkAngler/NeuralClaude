//! Demonstration of self-optimizing neural networks

use neural_llm_memory::self_optimizing::{SelfOptimizingNetwork, SelfOptimizingConfig};
use ndarray::Array2;
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Self-Optimizing Neural Network Demo ===\n");
    
    // Configure self-optimization
    let mut config = SelfOptimizingConfig::default();
    config.population_size = 20;
    config.mutation_rate = 0.15;
    config.max_layers = 6;
    config.layer_size_range = (32, 512);
    
    // Set optimization objectives
    config.objectives.clear();
    config.objectives.insert("accuracy".to_string(), 0.5);
    config.objectives.insert("speed".to_string(), 0.3);
    config.objectives.insert("memory".to_string(), 0.2);
    
    println!("Configuration:");
    println!("- Population size: {}", config.population_size);
    println!("- Max layers: {}", config.max_layers);
    println!("- Layer size range: {:?}", config.layer_size_range);
    println!("- Objectives: {:?}\n", config.objectives);
    
    // Create self-optimizing network with input size of 10 and output size of 4
    let mut network = SelfOptimizingNetwork::new_with_io_sizes(config, 10, 4);
    
    // Generate synthetic training data (XOR-like problem)
    let (train_data, val_data) = generate_synthetic_data(1000, 100);
    
    println!("Dataset:");
    println!("- Training samples: {}", train_data.len());
    println!("- Validation samples: {}", val_data.len());
    println!("- Input dimensions: 10");
    println!("- Output dimensions: 4\n");
    
    // Train with automatic optimization
    println!("Starting adaptive training...\n");
    
    for epoch in 0..50 {
        // Train for one epoch
        let start = std::time::Instant::now();
        
        // The network will automatically evolve its architecture
        network.train_adaptive(&train_data, &val_data, 1)?;
        
        let elapsed = start.elapsed();
        
        // Get current performance
        if epoch % 10 == 0 {
            let insights = network.get_insights();
            println!("Epoch {} completed in {:.2?}", epoch, elapsed);
            println!("Generation: {}", insights.generation_stats.current_generation);
            println!("Best fitness: {:.4}", insights.generation_stats.best_fitness);
            println!("Population diversity: {:.4}", insights.generation_stats.diversity_score);
            
            if !insights.winning_patterns.is_empty() {
                println!("Winning patterns:");
                for pattern in &insights.winning_patterns {
                    println!("  - {}", pattern);
                }
            }
            
            if !insights.recommendations.is_empty() {
                println!("Recommendations:");
                for rec in &insights.recommendations {
                    println!("  - {}", rec);
                }
            }
            
            println!();
        }
    }
    
    // Get the best discovered architecture
    let best_arch = network.get_best_architecture();
    println!("\n=== Best Architecture Found ===");
    println!("Layers: {}", best_arch.layers.len());
    println!("Parameters: {}", best_arch.num_parameters);
    println!("Learning rate: {:.6}", best_arch.hyperparameters.learning_rate);
    println!("Batch size: {}", best_arch.hyperparameters.batch_size);
    
    println!("\nLayer details:");
    for (i, layer) in best_arch.layers.iter().enumerate() {
        println!("  Layer {}: size={}, activation={:?}, dropout={:?}",
            i, layer.size, layer.activation, layer.dropout_rate);
    }
    
    println!("\nFitness scores:");
    for (objective, score) in &best_arch.fitness_scores {
        println!("  {}: {:.4}", objective, score);
    }
    
    println!("\nHardware metrics:");
    println!("  Inference time: {:.2} ms", best_arch.hardware_metrics.inference_time_ms);
    println!("  Memory usage: {:.2} MB", best_arch.hardware_metrics.memory_usage_mb);
    
    // Export the best network
    network.export_best("best_evolved_network.nn")?;
    println!("\nBest network exported to 'best_evolved_network.nn'");
    
    // Demonstrate different optimization scenarios
    println!("\n=== Testing Different Scenarios ===\n");
    
    // Scenario 1: Optimize for speed
    test_speed_optimization()?;
    
    // Scenario 2: Optimize for memory efficiency
    test_memory_optimization()?;
    
    // Scenario 3: Multi-objective optimization
    test_pareto_optimization()?;
    
    println!("\n=== Demo Complete ===");
    Ok(())
}

/// Generate synthetic data for testing
fn generate_synthetic_data(
    train_size: usize,
    val_size: usize,
) -> (Vec<(Array2<f32>, Array2<f32>)>, Vec<(Array2<f32>, Array2<f32>)>) {
    let mut rng = thread_rng();
    
    let mut generate_batch = |size: usize| -> Vec<(Array2<f32>, Array2<f32>)> {
        (0..size)
            .map(|_| {
                // Random input
                let input: Vec<f32> = (0..10).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let input = Array2::from_shape_vec((1, 10), input).unwrap();
                
                // Create a non-linear target based on input
                let sum: f32 = input.sum();
                let mut target = vec![0.0; 4];
                let class = ((sum + 2.0) / 4.0 * 3.0).round() as usize;
                if class < 4 {
                    target[class] = 1.0;
                } else {
                    target[3] = 1.0;
                }
                let target = Array2::from_shape_vec((1, 4), target).unwrap();
                
                (input, target)
            })
            .collect()
    };
    
    (generate_batch(train_size), generate_batch(val_size))
}

/// Test optimization for inference speed
fn test_speed_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("Scenario 1: Speed Optimization");
    
    let mut config = SelfOptimizingConfig::default();
    config.objectives.clear();
    config.objectives.insert("speed".to_string(), 0.8);
    config.objectives.insert("accuracy".to_string(), 0.2);
    config.target_inference_ms = 5.0;
    config.population_size = 10;
    
    let mut network = SelfOptimizingNetwork::new_with_io_sizes(config, 10, 4);
    let (train_data, val_data) = generate_synthetic_data(100, 20);
    
    network.train_adaptive(&train_data[..50].to_vec(), &val_data, 10)?;
    
    let best = network.get_best_architecture();
    println!("  Best inference time: {:.2} ms", best.hardware_metrics.inference_time_ms);
    println!("  Layer count: {}", best.layers.len());
    println!("  Speed score: {:.4}\n", best.fitness_scores.get("speed").unwrap_or(&0.0));
    
    Ok(())
}

/// Test optimization for memory efficiency
fn test_memory_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("Scenario 2: Memory Optimization");
    
    let mut config = SelfOptimizingConfig::default();
    config.objectives.clear();
    config.objectives.insert("memory".to_string(), 0.8);
    config.objectives.insert("accuracy".to_string(), 0.2);
    config.max_memory_mb = 10;
    config.population_size = 10;
    
    let mut network = SelfOptimizingNetwork::new_with_io_sizes(config, 10, 4);
    let (train_data, val_data) = generate_synthetic_data(100, 20);
    
    network.train_adaptive(&train_data[..50].to_vec(), &val_data, 10)?;
    
    let best = network.get_best_architecture();
    println!("  Best memory usage: {:.2} MB", best.hardware_metrics.memory_usage_mb);
    println!("  Parameters: {}", best.num_parameters);
    println!("  Memory score: {:.4}\n", best.fitness_scores.get("memory").unwrap_or(&0.0));
    
    Ok(())
}

/// Test Pareto-optimal solutions
fn test_pareto_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("Scenario 3: Multi-Objective Pareto Optimization");
    
    let mut config = SelfOptimizingConfig::default();
    config.objectives.clear();
    config.objectives.insert("accuracy".to_string(), 0.33);
    config.objectives.insert("speed".to_string(), 0.33);
    config.objectives.insert("memory".to_string(), 0.34);
    config.population_size = 15;
    
    let mut network = SelfOptimizingNetwork::new_with_io_sizes(config, 10, 4);
    let (train_data, val_data) = generate_synthetic_data(100, 20);
    
    network.train_adaptive(&train_data[..50].to_vec(), &val_data, 10)?;
    
    let insights = network.get_insights();
    println!("  Final generation: {}", insights.generation_stats.current_generation);
    println!("  Population diversity: {:.4}", insights.generation_stats.diversity_score);
    
    if let Some(best) = &insights.best_scores.iter().next() {
        println!("  Best combined fitness: {:.4}", best.1);
    }
    
    println!();
    Ok(())
}