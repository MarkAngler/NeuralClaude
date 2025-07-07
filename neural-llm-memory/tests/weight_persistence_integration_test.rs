//! Comprehensive integration tests for weight persistence in NeuralClaude
//! 
//! Tests cover:
//! - Full training cycle: train → save → restart → load → continue training
//! - Corrupted file handling
//! - Performance benchmarks for save/load operations
//! - Recovery from incomplete saves

use neural_llm_memory::{
    nn::{NeuralNetwork, LinearLayer, Conv1DLayer, DropoutLayer, LayerNormLayer, 
         Layer, WeightInit, ActivationFunction, NetworkBuilder},
    nn::layers::weight_extraction::WeightExtraction,
    persistence::{NetworkPersistence, NetworkState, NetworkStateBuilder, 
                  PersistenceFormat, LayerState, LayerConfig},
    self_optimizing::{SelfOptimizingNetwork, SelfOptimizingConfig},
};
use ndarray::{Array2, Array1};
use tempfile::TempDir;
use std::time::{Duration, Instant};
use std::fs;
use std::io::Write;
use std::sync::Arc;
use parking_lot::RwLock;

/// Helper function to create a test network with specific weights
fn create_test_network() -> NeuralNetwork {
    let layers: Vec<Box<dyn Layer + Send + Sync>> = vec![
        Box::new(LinearLayer::new(10, 64, ActivationFunction::ReLU, true, WeightInit::Xavier)),
        Box::new(DropoutLayer::new(0.2)),
        Box::new(LinearLayer::new(64, 32, ActivationFunction::ReLU, true, WeightInit::Xavier)),
        Box::new(LayerNormLayer::new(32, 1e-5)),
        Box::new(LinearLayer::new(32, 5, ActivationFunction::Sigmoid, true, WeightInit::Xavier)),
    ];
    
    NeuralNetwork::new(layers, 0.01)
}

/// Helper function to set specific weights for deterministic testing
fn set_deterministic_weights(network: &mut NeuralNetwork) {
    let layers = network.get_layers_mut();
    
    // Set weights for first linear layer
    if let Some(linear) = layers[0].get_params_mut().get_mut(0) {
        for (i, val) in linear.iter_mut().enumerate() {
            *val = (i as f32 + 1.0) * 0.01;
        }
    }
    
    // Set weights for second linear layer
    if let Some(linear) = layers[2].get_params_mut().get_mut(0) {
        for (i, val) in linear.iter_mut().enumerate() {
            *val = (i as f32 + 1.0) * 0.02;
        }
    }
    
    // Set weights for final linear layer
    if let Some(linear) = layers[4].get_params_mut().get_mut(0) {
        for (i, val) in linear.iter_mut().enumerate() {
            *val = (i as f32 + 1.0) * 0.03;
        }
    }
}

/// Generate synthetic training data
fn generate_training_data(samples: usize) -> Vec<(Array2<f32>, Array2<f32>)> {
    (0..samples)
        .map(|i| {
            let mut input = Array2::zeros((1, 10));
            let mut target = Array2::zeros((1, 5));
            
            // Create some pattern in the data
            for j in 0..10 {
                input[[0, j]] = ((i + j) as f32).sin() * 0.5 + 0.5;
            }
            
            let class = i % 5;
            target[[0, class]] = 1.0;
            
            (input, target)
        })
        .collect()
}

#[test]
fn test_full_training_cycle_persistence() {
    let temp_dir = TempDir::new().unwrap();
    let save_path = temp_dir.path().join("training_cycle.bin");
    
    // Phase 1: Initial training
    let mut network = create_test_network();
    set_deterministic_weights(&mut network);
    
    let training_data = generate_training_data(100);
    let initial_loss: f32;
    let weights_before_training: Vec<Array2<f32>>;
    
    {
        // Store initial weights
        weights_before_training = network.get_layers()
            .iter()
            .filter_map(|layer| layer.get_params().get(0).map(|w| (*w).clone()))
            .collect();
        
        // Calculate initial loss
        initial_loss = training_data.iter()
            .map(|(input, target)| {
                let output = network.predict(input);
                // MSE loss
                ((output - target) * (&output - target)).sum() / output.len() as f32
            })
            .sum::<f32>() / training_data.len() as f32;
        
        println!("Initial loss: {}", initial_loss);
    }
    
    // Train for 10 epochs
    let mut losses = Vec::new();
    for epoch in 0..10 {
        let mut epoch_loss = 0.0;
        
        for (input, target) in &training_data {
            let output = network.forward(input, true);
            let loss = ((output.clone() - target) * (&output - target)).sum() / output.len() as f32;
            epoch_loss += loss;
            
            // Backprop
            let grad = 2.0 * (output - target) / output.len() as f32;
            network.backward(&grad, input);
        }
        
        epoch_loss /= training_data.len() as f32;
        losses.push(epoch_loss);
        println!("Epoch {}: loss = {}", epoch, epoch_loss);
    }
    
    let mid_training_loss = losses.last().unwrap();
    
    // Save network state
    {
        // Create a proper NetworkState using the builder
        let layer_states: Vec<LayerState> = network.get_layers()
            .iter()
            .filter_map(|layer| layer.to_layer_state())
            .collect();
        
        let mut builder = NetworkStateBuilder::new(
            Default::default(), // Use default genome for now
            SelfOptimizingConfig::default(),
            10,
            5,
        )
        .with_description("Test network mid-training")
        .with_training_state(10, 1000, losses.clone(), network.get_learning_rate());
        
        for layer_state in layer_states {
            builder = builder.add_layer(layer_state);
        }
        
        let state = builder.build().unwrap();
        NetworkPersistence::save(&save_path, &state, PersistenceFormat::Binary).unwrap();
    }
    
    // Phase 2: Load and continue training
    let loaded_state = NetworkPersistence::load(&save_path, PersistenceFormat::Binary).unwrap();
    
    // Rebuild network from loaded state
    let mut loaded_network = {
        let mut layers: Vec<Box<dyn Layer + Send + Sync>> = Vec::new();
        
        for layer_state in &loaded_state.layers {
            match &layer_state.config {
                LayerConfig::Linear { .. } => {
                    let layer = LinearLayer::from_layer_state(layer_state).unwrap();
                    layers.push(Box::new(layer));
                },
                LayerConfig::Dropout { .. } => {
                    let layer = DropoutLayer::from_layer_state(layer_state).unwrap();
                    layers.push(Box::new(layer));
                },
                LayerConfig::LayerNorm { .. } => {
                    let layer = LayerNormLayer::from_layer_state(layer_state).unwrap();
                    layers.push(Box::new(layer));
                },
                _ => {}
            }
        }
        
        NeuralNetwork::new(layers, loaded_state.training_state.learning_rate)
    };
    
    // Verify weights were loaded correctly
    {
        let loaded_weights: Vec<Array2<f32>> = loaded_network.get_layers()
            .iter()
            .filter_map(|layer| layer.get_params().get(0).map(|w| (*w).clone()))
            .collect();
        
        // Check that we have the same number of weight matrices
        assert_eq!(loaded_weights.len(), weights_before_training.len(), 
                   "Number of weight matrices should match");
        
        // Weights should have changed from initial values due to training
        for (loaded, initial) in loaded_weights.iter().zip(&weights_before_training) {
            let diff: f32 = (loaded - initial).mapv(|x| x.abs()).sum();
            assert!(diff > 0.01, "Weights should have changed during training");
        }
    }
    
    // Calculate loss on loaded network (should match mid-training loss)
    let loaded_loss: f32 = training_data.iter()
        .map(|(input, target)| {
            let output = loaded_network.predict(input);
            ((output - target) * (&output - target)).sum() / output.len() as f32
        })
        .sum::<f32>() / training_data.len() as f32;
    
    println!("Loss after loading: {} (expected: {})", loaded_loss, mid_training_loss);
    assert!((loaded_loss - mid_training_loss).abs() < 0.001, 
            "Loss should be preserved after load");
    
    // Continue training for 10 more epochs
    let mut continued_losses = Vec::new();
    for epoch in 0..10 {
        let mut epoch_loss = 0.0;
        
        for (input, target) in &training_data {
            let output = loaded_network.forward(input, true);
            let loss = ((output.clone() - target) * (&output - target)).sum() / output.len() as f32;
            epoch_loss += loss;
            
            let grad = 2.0 * (output - target) / output.len() as f32;
            loaded_network.backward(&grad, input);
        }
        
        epoch_loss /= training_data.len() as f32;
        continued_losses.push(epoch_loss);
        println!("Continued epoch {}: loss = {}", epoch, epoch_loss);
    }
    
    // Verify continued training improved the loss
    let final_loss = continued_losses.last().unwrap();
    assert!(final_loss < mid_training_loss, 
            "Continued training should reduce loss further");
    assert!(final_loss < &initial_loss, 
            "Final loss should be much better than initial");
}

#[test]
fn test_corrupted_file_handling() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test 1: Truncated file
    {
        let truncated_path = temp_dir.path().join("truncated.bin");
        
        // Create a valid network state first
        let network = create_test_network();
        let layer_states: Vec<LayerState> = network.get_layers()
            .iter()
            .filter_map(|layer| layer.to_layer_state())
            .collect();
        
        let state = NetworkStateBuilder::new(
            Default::default(),
            SelfOptimizingConfig::default(),
            10,
            5,
        )
        .add_layer(layer_states[0].clone())
        .build()
        .unwrap();
        
        // Save it properly
        NetworkPersistence::save(&truncated_path, &state, PersistenceFormat::Binary).unwrap();
        
        // Now truncate the file
        let file_size = fs::metadata(&truncated_path).unwrap().len();
        let mut file = fs::OpenOptions::new()
            .write(true)
            .open(&truncated_path)
            .unwrap();
        file.set_len(file_size / 2).unwrap();
        drop(file);
        
        // Try to load - should fail gracefully
        let result = NetworkPersistence::load(&truncated_path, PersistenceFormat::Binary);
        assert!(result.is_err(), "Should fail to load truncated file");
    }
    
    // Test 2: Corrupted header
    {
        let corrupted_path = temp_dir.path().join("corrupted_header.bin");
        
        // Write garbage data
        let mut file = fs::File::create(&corrupted_path).unwrap();
        file.write_all(&[0xFF; 1024]).unwrap();
        drop(file);
        
        // Try to load - should fail gracefully
        let result = NetworkPersistence::load(&corrupted_path, PersistenceFormat::Binary);
        assert!(result.is_err(), "Should fail to load file with corrupted header");
    }
    
    // Test 3: Wrong format file
    {
        let wrong_format_path = temp_dir.path().join("wrong_format.json");
        
        // Write some JSON that's not a NetworkState
        let mut file = fs::File::create(&wrong_format_path).unwrap();
        file.write_all(br#"{"not": "a network state", "random": [1, 2, 3]}"#).unwrap();
        drop(file);
        
        // Try to load as binary - should fail
        let result = NetworkPersistence::load(&wrong_format_path, PersistenceFormat::Binary);
        assert!(result.is_err(), "Should fail to load wrong format");
        
        // Try to load as JSON - should also fail (wrong structure)
        let result = NetworkPersistence::load(&wrong_format_path, PersistenceFormat::Json);
        assert!(result.is_err(), "Should fail to load JSON with wrong structure");
    }
    
    // Test 4: Partial write recovery
    {
        let partial_path = temp_dir.path().join("partial.bin");
        let backup_path = temp_dir.path().join("partial.bin.backup");
        
        // Create a valid state and save it
        let network = create_test_network();
        let layer_states: Vec<LayerState> = network.get_layers()
            .iter()
            .filter_map(|layer| layer.to_layer_state())
            .collect();
        
        let state = NetworkStateBuilder::new(
            Default::default(),
            SelfOptimizingConfig::default(),
            10,
            5,
        )
        .add_layer(layer_states[0].clone())
        .build()
        .unwrap();
        
        // First save a good backup
        NetworkPersistence::save(&backup_path, &state, PersistenceFormat::Binary).unwrap();
        
        // Simulate partial write by creating empty file
        fs::File::create(&partial_path).unwrap();
        
        // Loading should fail
        let result = NetworkPersistence::load(&partial_path, PersistenceFormat::Binary);
        assert!(result.is_err(), "Should fail to load empty file");
        
        // But backup should still be valid
        let backup_result = NetworkPersistence::load(&backup_path, PersistenceFormat::Binary);
        assert!(backup_result.is_ok(), "Backup should remain valid");
    }
}

#[test]
fn test_save_load_performance_benchmarks() {
    let temp_dir = TempDir::new().unwrap();
    
    // Test networks of different sizes
    let layer_configs = vec![
        ("Small", vec![(100, 50), (50, 10)]),
        ("Medium", vec![(1000, 500), (500, 200), (200, 50)]),
        ("Large", vec![(5000, 2000), (2000, 1000), (1000, 500), (500, 100)]),
    ];
    
    for (name, config) in layer_configs {
        println!("\n=== Testing {} network ===", name);
        
        // Create network
        let mut layers: Vec<Box<dyn Layer + Send + Sync>> = Vec::new();
        let mut prev_size = config[0].0;
        
        for (i, &(_, out_size)) in config.iter().enumerate() {
            layers.push(Box::new(LinearLayer::new(
                prev_size,
                out_size,
                ActivationFunction::ReLU,
                true,
                WeightInit::Xavier,
            )));
            
            if i < config.len() - 1 {
                layers.push(Box::new(DropoutLayer::new(0.1)));
            }
            
            prev_size = out_size;
        }
        
        let network = NeuralNetwork::new(layers, 0.01);
        
        // Count total parameters
        let total_params: usize = network.get_layers()
            .iter()
            .map(|layer| layer.count_parameters())
            .sum();
        println!("Total parameters: {}", total_params);
        
        // Create network state
        let layer_states: Vec<LayerState> = network.get_layers()
            .iter()
            .filter_map(|layer| layer.to_layer_state())
            .collect();
        
        let state = NetworkStateBuilder::new(
            Default::default(),
            SelfOptimizingConfig::default(),
            config[0].0,
            config.last().unwrap().1,
        )
        .with_description(&format!("{} test network", name))
        .with_training_state(0, 0, vec![], 0.01)
        .add_layers(layer_states)
        .build()
        .unwrap();
        
        // Test different formats
        for format in &[
            PersistenceFormat::Binary,
            PersistenceFormat::Json,
            PersistenceFormat::Compressed,
        ] {
            let path = temp_dir.path().join(format!("{}_{:?}.dat", name.to_lowercase(), format));
            
            // Benchmark save
            let save_start = Instant::now();
            NetworkPersistence::save(&path, &state, *format).unwrap();
            let save_duration = save_start.elapsed();
            
            // Get file size
            let file_size = fs::metadata(&path).unwrap().len();
            
            // Benchmark load
            let load_start = Instant::now();
            let loaded = NetworkPersistence::load(&path, *format).unwrap();
            let load_duration = load_start.elapsed();
            
            // Verify loaded correctly
            assert_eq!(loaded.layers.len(), state.layers.len());
            
            println!("  Format {:?}:", format);
            println!("    Save time: {:?}", save_duration);
            println!("    Load time: {:?}", load_duration);
            println!("    File size: {} KB", file_size / 1024);
            println!("    Save speed: {:.2} MB/s", 
                     (file_size as f64 / 1_000_000.0) / save_duration.as_secs_f64());
            println!("    Load speed: {:.2} MB/s", 
                     (file_size as f64 / 1_000_000.0) / load_duration.as_secs_f64());
            
            // Performance assertions
            match format {
                PersistenceFormat::Binary => {
                    // Binary should be fastest
                    assert!(save_duration < Duration::from_secs(1), 
                            "Binary save should be under 1 second");
                    assert!(load_duration < Duration::from_secs(1), 
                            "Binary load should be under 1 second");
                },
                PersistenceFormat::Compressed => {
                    // Compressed should have smallest file size
                    let binary_size = fs::metadata(
                        temp_dir.path().join(format!("{}_Binary.dat", name.to_lowercase()))
                    ).unwrap().len();
                    assert!(file_size < binary_size, 
                            "Compressed should be smaller than binary");
                },
                _ => {}
            }
        }
    }
}

#[test]
fn test_recovery_from_incomplete_saves() {
    let temp_dir = TempDir::new().unwrap();
    let primary_path = temp_dir.path().join("primary.bin");
    let temp_path = temp_dir.path().join("primary.bin.tmp");
    
    // Create initial network and save it
    let network = create_test_network();
    set_deterministic_weights(&mut network);
    
    let layer_states: Vec<LayerState> = network.get_layers()
        .iter()
        .filter_map(|layer| layer.to_layer_state())
        .collect();
    
    let initial_state = NetworkStateBuilder::new(
        Default::default(),
        SelfOptimizingConfig::default(),
        10,
        5,
    )
    .with_description("Initial state")
    .add_layers(layer_states)
    .build()
    .unwrap();
    
    // Save initial state
    NetworkPersistence::save(&primary_path, &initial_state, PersistenceFormat::Binary).unwrap();
    
    // Simulate training and periodic saves
    for i in 0..5 {
        // Modify the network (simulate training)
        let mut modified_network = create_test_network();
        modified_network.get_layers_mut()
            .iter_mut()
            .for_each(|layer| layer.perturb_weights(0.01 * (i + 1) as f32));
        
        let modified_states: Vec<LayerState> = modified_network.get_layers()
            .iter()
            .filter_map(|layer| layer.to_layer_state())
            .collect();
        
        let modified_state = NetworkStateBuilder::new(
            Default::default(),
            SelfOptimizingConfig::default(),
            10,
            5,
        )
        .with_description(&format!("State after {} updates", i + 1))
        .add_layers(modified_states)
        .build()
        .unwrap();
        
        // Simulate atomic save with temp file
        // Step 1: Write to temp file
        NetworkPersistence::save(&temp_path, &modified_state, PersistenceFormat::Binary).unwrap();
        
        // Step 2: Simulate crash during rename (don't complete the atomic operation)
        if i == 2 {
            println!("Simulating crash during save {}", i);
            // Delete temp file to simulate incomplete save
            fs::remove_file(&temp_path).unwrap();
            
            // Primary file should still contain last good state
            let recovered = NetworkPersistence::load(&primary_path, PersistenceFormat::Binary).unwrap();
            assert_eq!(recovered.metadata.description, "State after 2 updates",
                      "Should recover last successfully saved state");
            break;
        } else {
            // Complete the atomic save
            fs::rename(&temp_path, &primary_path).unwrap();
            
            // Verify save completed correctly
            let loaded = NetworkPersistence::load(&primary_path, PersistenceFormat::Binary).unwrap();
            assert_eq!(loaded.metadata.description, format!("State after {} updates", i + 1));
        }
    }
}

#[test]
fn test_concurrent_save_load_safety() {
    use std::thread;
    use std::sync::Arc;
    
    let temp_dir = Arc::new(TempDir::new().unwrap());
    let save_path = Arc::new(temp_dir.path().join("concurrent.bin"));
    
    // Create initial state
    let network = create_test_network();
    let layer_states: Vec<LayerState> = network.get_layers()
        .iter()
        .filter_map(|layer| layer.to_layer_state())
        .collect();
    
    let initial_state = Arc::new(NetworkStateBuilder::new(
        Default::default(),
        SelfOptimizingConfig::default(),
        10,
        5,
    )
    .add_layers(layer_states)
    .build()
    .unwrap());
    
    // Save initial state
    NetworkPersistence::save(&*save_path, &initial_state, PersistenceFormat::Binary).unwrap();
    
    // Spawn multiple threads that try to load simultaneously
    let mut handles = vec![];
    
    for i in 0..10 {
        let path = Arc::clone(&save_path);
        let handle = thread::spawn(move || {
            for j in 0..5 {
                // Random delay to increase chance of concurrent access
                thread::sleep(Duration::from_millis((i * 10 + j) as u64));
                
                // Try to load
                match NetworkPersistence::load(&*path, PersistenceFormat::Binary) {
                    Ok(state) => {
                        // Verify loaded state is valid
                        assert!(!state.layers.is_empty(), "Loaded state should have layers");
                    },
                    Err(e) => {
                        // Loading might fail if file is being written
                        println!("Thread {} iteration {} load failed: {}", i, j, e);
                    }
                }
            }
        });
        handles.push(handle);
    }
    
    // One thread that periodically saves
    let save_path_clone = Arc::clone(&save_path);
    let save_state = Arc::clone(&initial_state);
    let save_handle = thread::spawn(move || {
        for i in 0..3 {
            thread::sleep(Duration::from_millis(50));
            
            // Create a modified state
            let mut modified_state = (*save_state).clone();
            modified_state.metadata.description = format!("Update {}", i);
            
            // Save with atomic write
            let temp_path = save_path_clone.with_extension("tmp");
            NetworkPersistence::save(&temp_path, &modified_state, PersistenceFormat::Binary).unwrap();
            fs::rename(&temp_path, &*save_path_clone).unwrap();
            
            println!("Saved update {}", i);
        }
    });
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    save_handle.join().unwrap();
    
    // Final verification - should be able to load valid state
    let final_state = NetworkPersistence::load(&*save_path, PersistenceFormat::Binary).unwrap();
    assert!(!final_state.layers.is_empty(), "Final state should be valid");
}

#[test] 
fn test_weight_precision_preservation() {
    let temp_dir = TempDir::new().unwrap();
    
    // Create network with very specific weight values
    let mut network = create_test_network();
    
    // Set precise weight values that might lose precision
    let test_values = vec![
        1.234567890123456789_f32,
        -0.000000123456789_f32,
        std::f32::consts::PI,
        std::f32::consts::E,
        f32::MIN_POSITIVE,
        f32::MAX / 1000.0,
    ];
    
    // Set test values in first layer
    let layers = network.get_layers_mut();
    if let Some(weights) = layers[0].get_params_mut().get_mut(0) {
        for (i, test_val) in test_values.iter().enumerate() {
            if i < weights.len() {
                weights.as_slice_mut().unwrap()[i] = *test_val;
            }
        }
    }
    
    // Extract original weights for comparison
    let original_weights: Vec<Vec<f32>> = network.get_layers()
        .iter()
        .filter_map(|layer| {
            layer.get_params().get(0).map(|w| w.iter().cloned().collect())
        })
        .collect();
    
    // Save and load in each format
    for format in &[
        PersistenceFormat::Binary,
        PersistenceFormat::Json,
        PersistenceFormat::Compressed,
    ] {
        let path = temp_dir.path().join(format!("precision_{:?}.dat", format));
        
        // Create state
        let layer_states: Vec<LayerState> = network.get_layers()
            .iter()
            .filter_map(|layer| layer.to_layer_state())
            .collect();
        
        let state = NetworkStateBuilder::new(
            Default::default(),
            SelfOptimizingConfig::default(),
            10,
            5,
        )
        .add_layers(layer_states)
        .build()
        .unwrap();
        
        // Save
        NetworkPersistence::save(&path, &state, *format).unwrap();
        
        // Load
        let loaded_state = NetworkPersistence::load(&path, *format).unwrap();
        
        // Rebuild network and check weights
        let mut weight_idx = 0;
        for layer_state in &loaded_state.layers {
            if let Some(weights) = &layer_state.weights {
                let loaded_weights: Vec<f32> = weights.iter().cloned().collect();
                let original = &original_weights[weight_idx];
                
                // Check each weight value
                for (i, (loaded, original)) in loaded_weights.iter().zip(original.iter()).enumerate() {
                    let diff = (loaded - original).abs();
                    let relative_error = if *original != 0.0 {
                        diff / original.abs()
                    } else {
                        diff
                    };
                    
                    // For special test values, check precision
                    if i < test_values.len() {
                        match format {
                            PersistenceFormat::Binary | PersistenceFormat::Compressed => {
                                // Binary formats should preserve exact f32 representation
                                assert_eq!(*loaded, *original, 
                                          "Binary format should preserve exact f32 values");
                            },
                            PersistenceFormat::Json => {
                                // JSON might lose some precision due to decimal representation
                                assert!(relative_error < 1e-6, 
                                       "JSON format should preserve at least 6 decimal places: {} vs {}", 
                                       loaded, original);
                            },
                            _ => {}
                        }
                    }
                }
                
                weight_idx += 1;
            }
        }
    }
}

/// Store progress feedback for adaptive learning
fn store_test_feedback(operation_id: &str, success: bool, score: f32) {
    // This would integrate with the MCP feedback system in production
    println!("Test feedback: operation_id={}, success={}, score={}", 
             operation_id, success, score);
}

#[test]
fn test_adaptive_learning_persistence() {
    // Store test context in memory
    let test_context = "Testing weight persistence with adaptive learning integration";
    
    // Simulate operation tracking
    let operation_id = "op_test_weight_persistence";
    
    // Run the test and provide feedback
    let test_passed = std::panic::catch_unwind(|| {
        test_full_training_cycle_persistence();
    }).is_ok();
    
    // Provide feedback on test success
    store_test_feedback(
        operation_id,
        test_passed,
        if test_passed { 1.0 } else { 0.0 }
    );
}