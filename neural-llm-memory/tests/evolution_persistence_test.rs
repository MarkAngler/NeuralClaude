//! Test evolution state persistence

use neural_llm_memory::self_optimizing::{
    SelfOptimizingNetwork, SelfOptimizingConfig, ArchitectureGenome,
};
use neural_llm_memory::persistence::PersistenceFormat;
use ndarray::Array2;
use tempfile::TempDir;

#[test]
fn test_evolution_history_persistence() {
    // Create a self-optimizing network
    let mut config = SelfOptimizingConfig::default();
    config.population_size = 10;
    config.mutation_rate = 0.5; // High mutation rate for testing
    
    let mut network = SelfOptimizingNetwork::new_with_io_sizes(config, 10, 5);
    
    // Create some dummy training data
    let training_data: Vec<(Array2<f32>, Array2<f32>)> = (0..5)
        .map(|_| {
            let input = Array2::zeros((1, 10));
            let target = Array2::zeros((1, 5));
            (input, target)
        })
        .collect();
    
    // Evolve for a few generations to build up history
    for gen in 0..3 {
        println!("Evolving generation {}", gen);
        network.evolve_architecture(&training_data).unwrap();
    }
    
    // Save the network state
    let temp_dir = TempDir::new().unwrap();
    let save_path = temp_dir.path().join("evolved_network.bin");
    
    network.save(&save_path, PersistenceFormat::Binary).unwrap();
    
    // Load the network back
    let loaded_network = SelfOptimizingNetwork::load(&save_path, PersistenceFormat::Binary).unwrap();
    
    // Verify evolution history was preserved
    assert_eq!(loaded_network.generation, network.generation);
    
    // Get the network state to inspect evolution history
    let original_state = network.to_network_state().unwrap();
    let loaded_state = loaded_network.to_network_state().unwrap();
    
    // Check that evolution history is populated
    assert!(!original_state.evolution_history.best_genomes.is_empty());
    assert!(!original_state.evolution_history.fitness_history.is_empty());
    
    // Verify loaded state matches
    assert_eq!(
        original_state.evolution_history.generation,
        loaded_state.evolution_history.generation
    );
    
    assert_eq!(
        original_state.evolution_history.best_genomes.len(),
        loaded_state.evolution_history.best_genomes.len()
    );
    
    assert_eq!(
        original_state.evolution_history.fitness_history.len(),
        loaded_state.evolution_history.fitness_history.len()
    );
    
    // Check that mutation history is tracked
    let best_genome = &original_state.evolution_history.best_genomes.last().unwrap().1;
    if !best_genome.mutation_history.is_empty() {
        println!("Mutation history found:");
        for mutation in &best_genome.mutation_history {
            println!("  - {}: {}", mutation.mutation_type, mutation.description);
        }
    }
    
    // Verify successful mutations are tracked
    if !original_state.evolution_history.successful_mutations.is_empty() {
        println!("Successful mutations found:");
        for mutation in &original_state.evolution_history.successful_mutations {
            println!("  - Gen {}: {} (improvement: {:.4})", 
                mutation.generation, 
                mutation.mutation_type,
                mutation.fitness_improvement
            );
        }
    }
    
    println!("Evolution persistence test passed!");
}

#[test]
fn test_mutation_tracking() {
    let config = SelfOptimizingConfig::default();
    let mut genome = ArchitectureGenome::random(&config);
    
    // Clear any initial mutation history
    genome.mutation_history.clear();
    
    // Apply mutations
    genome.mutate(1.0); // 100% mutation rate to ensure mutations happen
    
    // Check that mutations were tracked
    assert!(!genome.mutation_history.is_empty(), "No mutations were tracked");
    
    println!("Tracked {} mutations:", genome.mutation_history.len());
    for mutation in &genome.mutation_history {
        println!("  - {}: {}", mutation.mutation_type, mutation.description);
    }
}