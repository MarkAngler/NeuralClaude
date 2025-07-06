//! Demonstration of neural network persistence functionality

use neural_llm_memory::self_optimizing::{SelfOptimizingConfig, SelfOptimizingNetwork};
use neural_llm_memory::persistence::{PersistenceFormat, CheckpointManager, NetworkPersistence};
use ndarray::{Array2, array};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Neural Network Persistence Demo");
    println!("===============================\n");
    
    // Create a self-optimizing network
    let config = SelfOptimizingConfig {
        enabled: true,
        population_size: 20,
        min_layers: 2,
        max_layers: 5,
        ..Default::default()
    };
    
    println!("Creating self-optimizing network...");
    let mut network = SelfOptimizingNetwork::new_with_io_sizes(config, 10, 5);
    
    // Create some dummy training data
    let training_data: Vec<(Array2<f32>, Array2<f32>)> = (0..10)
        .map(|i| {
            let input = Array2::from_shape_fn((1, 10), |_| rand::random::<f32>());
            let target = Array2::from_shape_fn((1, 5), |_| i as f32 / 10.0);
            (input, target)
        })
        .collect();
    
    // Train for a few epochs
    println!("\nTraining network for 5 epochs...");
    for epoch in 0..5 {
        let mut epoch_loss = 0.0;
        for (input, target) in &training_data {
            let output = network.best_network.read().unwrap().forward(input, true);
            let loss = ((output - target) * (output - target)).mean().unwrap();
            epoch_loss += loss;
        }
        epoch_loss /= training_data.len() as f32;
        println!("Epoch {}: loss = {:.4}", epoch, epoch_loss);
    }
    
    // Save the network in different formats
    println!("\nSaving network in different formats...");
    
    // Binary format
    let binary_path = "network_demo.bin";
    network.save(binary_path, PersistenceFormat::Binary)?;
    let binary_size = NetworkPersistence::get_file_size(binary_path)?;
    println!("Binary format: {} bytes", binary_size);
    
    // JSON format
    let json_path = "network_demo.json";
    network.save(json_path, PersistenceFormat::Json)?;
    let json_size = NetworkPersistence::get_file_size(json_path)?;
    println!("JSON format: {} bytes", json_size);
    
    // Compressed format
    let compressed_path = "network_demo.gz";
    network.save(compressed_path, PersistenceFormat::Compressed)?;
    let compressed_size = NetworkPersistence::get_file_size(compressed_path)?;
    println!("Compressed format: {} bytes", compressed_size);
    
    // Calculate compression ratios
    println!("\nCompression ratios:");
    println!("JSON/Binary: {:.2}x", json_size as f32 / binary_size as f32);
    println!("Binary/Compressed: {:.2}x", binary_size as f32 / compressed_size as f32);
    
    // Load the network back
    println!("\nLoading network from compressed format...");
    let loaded_network = SelfOptimizingNetwork::load(compressed_path, PersistenceFormat::Compressed)?;
    println!("Successfully loaded network with generation: {}", loaded_network.generation);
    
    // Demonstrate checkpoint management
    println!("\nDemonstrating checkpoint management...");
    let mut checkpoint_manager = CheckpointManager::new("checkpoints", 3);
    
    // Save some checkpoints
    for i in 0..5 {
        let loss = 1.0 / (i as f32 + 1.0);
        network.save_checkpoint(&mut checkpoint_manager, i, loss)?;
        println!("Saved checkpoint {} with loss {:.4}", i, loss);
    }
    
    // List checkpoints
    println!("\nAvailable checkpoints (should keep best 3):");
    for checkpoint in checkpoint_manager.list_checkpoints() {
        println!("  Epoch {}: loss = {:.4}", checkpoint.epoch, checkpoint.loss);
    }
    
    // Load best checkpoint
    println!("\nLoading best checkpoint...");
    let best_state = checkpoint_manager.load_best_checkpoint()?;
    println!("Loaded checkpoint from generation: {}", best_state.evolution_history.generation);
    
    // Validate saved files
    println!("\nValidating saved files...");
    for (path, format) in &[
        (binary_path, PersistenceFormat::Binary),
        (json_path, PersistenceFormat::Json),
        (compressed_path, PersistenceFormat::Compressed),
    ] {
        let valid = NetworkPersistence::validate(path, *format)?;
        println!("{}: {}", path, if valid { "VALID" } else { "INVALID" });
    }
    
    // Clean up demo files
    println!("\nCleaning up demo files...");
    std::fs::remove_file(binary_path).ok();
    std::fs::remove_file(json_path).ok();
    std::fs::remove_file(compressed_path).ok();
    std::fs::remove_dir_all("checkpoints").ok();
    
    println!("\nDemo completed successfully!");
    
    Ok(())
}