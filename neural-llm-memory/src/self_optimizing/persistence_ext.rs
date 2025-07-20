//! Persistence extension for self-optimizing networks

use super::SelfOptimizingNetwork;
use crate::persistence::{
    NetworkPersistence, NetworkState, NetworkStateBuilder, LayerState,
    PersistenceFormat, CheckpointManager, EvolutionHistory,
};
use crate::nn::{NeuralNetwork, NetworkBuilder, Layer, LinearLayer, Conv1DLayer, 
                DropoutLayer, LayerNormLayer};
use crate::nn::layer::EmbeddingLayer;
use std::path::Path;
use anyhow::Result;

impl SelfOptimizingNetwork {
    /// Save the network to a file
    pub fn save<P: AsRef<Path>>(
        &self,
        path: P,
        format: PersistenceFormat,
    ) -> Result<()> {
        let state = self.to_network_state()?;
        NetworkPersistence::save(path, &state, format)?;
        Ok(())
    }
    
    /// Load a network from a file
    pub fn load<P: AsRef<Path>>(
        path: P,
        format: PersistenceFormat,
    ) -> Result<Self> {
        let state = NetworkPersistence::load(path, format)?;
        Self::from_network_state(state)
    }
    
    /// Save a checkpoint during training
    pub fn save_checkpoint(
        &self,
        checkpoint_manager: &mut CheckpointManager,
        epoch: usize,
        loss: f32,
    ) -> Result<()> {
        let state = self.to_network_state()?;
        checkpoint_manager.save_checkpoint(&state, epoch, loss)?;
        Ok(())
    }
    
    /// Convert the network to a serializable state
    pub fn to_network_state(&self) -> Result<NetworkState> {
        // Get the current best network
        let network = self.best_network.read().unwrap();
        
        // Extract layer states
        let mut layer_states: Vec<LayerState> = Vec::new();
        
        // We need to iterate through the layers and convert them to LayerState
        // Now we can use the to_layer_state() method that we added to the Layer trait
        for layer in network.get_layers() {
            if let Some(layer_state) = layer.to_layer_state() {
                layer_states.push(layer_state);
            } else {
                eprintln!("Failed to extract layer state for a layer");
            }
        }
        
        // Get the current genome
        let genome = self.evolution_controller.get_best_genome()
            .unwrap_or_else(|| self.get_best_architecture());
        
        // Get the internal evolution history
        let internal_history = self.evolution_controller.get_evolution_history();
        let hall_of_fame = self.evolution_controller.get_hall_of_fame();
        
        // Convert generation records to GenerationStats for fitness history
        let fitness_history: Vec<crate::self_optimizing::GenerationStats> = internal_history.generations.iter()
            .map(|record| crate::self_optimizing::GenerationStats {
                current_generation: record.generation,
                best_fitness: record.best_fitness,
                average_fitness: record.average_fitness,
                diversity_score: record.diversity_score,
                convergence_rate: if record.generation > 0 {
                    (record.best_fitness - record.worst_fitness) / record.generation as f32
                } else {
                    0.0
                },
            })
            .collect();
        
        // Build best genomes list from hall of fame
        let mut best_genomes = Vec::new();
        for (i, genome) in hall_of_fame.iter().enumerate() {
            // Estimate generation based on position in hall of fame
            // This is approximate since we don't track exact generation for each
            let estimated_gen = self.generation.saturating_sub(hall_of_fame.len() - i - 1);
            best_genomes.push((estimated_gen, genome.clone()));
        }
        
        // If we don't have the current genome in hall of fame, add it
        if !hall_of_fame.iter().any(|g| g.id == genome.id) {
            best_genomes.push((self.generation, genome.clone()));
        }
        
        // Convert mutation history from genomes to MutationRecord
        let mut successful_mutations = Vec::new();
        
        // Check the current best genome and any in hall of fame for mutations
        for (gen, genome) in &best_genomes {
            for mutation in &genome.mutation_history {
                // Calculate fitness improvement by checking mutation description
                // If it contains improvement info, parse it; otherwise estimate
                let fitness_improvement = if mutation.description.contains("improved fitness by") {
                    // Try to parse the improvement from the description
                    if let Some(start) = mutation.description.find("improved fitness by ") {
                        let start = start + "improved fitness by ".len();
                        if let Some(end) = mutation.description[start..].find(')') {
                            mutation.description[start..start+end]
                                .parse::<f32>()
                                .unwrap_or(0.0)
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                } else {
                    // Estimate improvement as 10% if we don't have exact data
                    let current_fitness: f32 = genome.fitness_scores.values().sum();
                    current_fitness * 0.1
                };
                
                successful_mutations.push(crate::persistence::MutationRecord {
                    generation: *gen,
                    mutation_type: mutation.mutation_type.clone(),
                    fitness_improvement,
                    description: mutation.description.clone(),
                });
            }
        }
        
        // Build evolution history
        let evolution_history = EvolutionHistory {
            generation: self.generation,
            best_genomes,
            fitness_history,
            successful_mutations,
        };
        
        // Build the network state
        let mut builder = NetworkStateBuilder::new(
            genome,
            self.config.clone(),
            self.input_size,
            self.evolution_controller.output_size,
        )
        .with_description(&format!(
            "Self-optimizing network, generation {}",
            self.generation
        ))
        .with_training_state(
            self.generation,
            0, // Would need to track global steps
            self.get_loss_history(),
            network.get_learning_rate(),
        )
        .with_evolution_history(evolution_history)
        .with_insights(self.get_insights());
        
        // Add all the layer states
        for layer_state in layer_states {
            builder = builder.add_layer(layer_state);
        }
        
        let state = builder.build()
            .map_err(|e| anyhow::anyhow!("Failed to build network state: {}", e))?;
        
        Ok(state)
    }
    
    /// Create a network from a saved state
    pub fn from_network_state(state: NetworkState) -> Result<Self> {
        // Create a new network with the saved configuration
        let mut network = Self::new_with_io_sizes(
            state.config.clone(),
            state.metadata.input_size,
            state.metadata.output_size,
        );
        
        // If we have layer states, use them to rebuild the network
        if !state.layers.is_empty() {
            // Get learning rate from training state (it's not optional in NetworkState)
            let learning_rate = state.training_state.learning_rate;
            
            // Use NetworkBuilderExt to create network from layer states
            let new_network = NetworkBuilder::from_layer_states(state.layers, learning_rate)?;
            
            *network.best_network.write().unwrap() = new_network;
        } else {
            // Fall back to building from genome if no layer states
            if let Some((gen, genome)) = state.evolution_history.best_genomes.last() {
                network.generation = *gen;
                
                // Build network from genome
                let new_network = genome.to_network_with_io_sizes(
                    state.metadata.input_size,
                    state.metadata.output_size,
                ).map_err(|e| anyhow::anyhow!("Failed to build network from genome: {}", e))?;
                
                *network.best_network.write().unwrap() = new_network;
            }
        }
        
        // Restore training state
        network.generation = state.evolution_history.generation;
        
        // Restore learning rate
        network.best_network.write().unwrap().set_learning_rate(state.training_state.learning_rate);
        
        // Note: We can't fully restore the evolution controller state
        // without more architectural changes
        
        Ok(network)
    }
    
    /// Export the network in a portable format
    pub fn export<P: AsRef<Path>>(
        &self,
        path: P,
        include_evolution_history: bool,
    ) -> Result<()> {
        let mut state = self.to_network_state()?;
        
        if !include_evolution_history {
            // Clear evolution history for smaller file size
            state.evolution_history = EvolutionHistory {
                generation: self.generation,
                best_genomes: vec![],
                fitness_history: vec![],
                successful_mutations: vec![],
            };
        }
        
        // Use compressed format for exports
        NetworkPersistence::save(path, &state, PersistenceFormat::Compressed)?;
        Ok(())
    }
    
    /// Get the loss history from the training state
    fn get_loss_history(&self) -> Vec<f32> {
        // Access the training state from the best network
        let network = self.best_network.read().unwrap();
        // In the current implementation, we don't have direct access to loss history
        // This would require architectural changes to track loss history
        Vec::new()
    }
}

/// Extension trait for NetworkBuilder to support loading
pub trait NetworkBuilderExt {
    /// Build a network from layer states
    fn from_layer_states(layers: Vec<LayerState>, learning_rate: f32) -> Result<NeuralNetwork>;
}

impl NetworkBuilderExt for NetworkBuilder {
    fn from_layer_states(layers: Vec<LayerState>, learning_rate: f32) -> Result<NeuralNetwork> {
        let mut network_layers: Vec<Box<dyn Layer + Send + Sync>> = Vec::new();
        
        for layer_state in layers {
            // Convert LayerState to concrete layer type based on the config
            let layer: Box<dyn Layer + Send + Sync> = match &layer_state.config {
                crate::persistence::LayerConfig::Linear { .. } => {
                    let linear_layer = LinearLayer::from_layer_state(&layer_state)
                        .map_err(|e| anyhow::anyhow!("Failed to create LinearLayer: {}", e))?;
                    Box::new(linear_layer)
                },
                crate::persistence::LayerConfig::Conv1D { .. } => {
                    let conv_layer = Conv1DLayer::from_layer_state(&layer_state)
                        .map_err(|e| anyhow::anyhow!("Failed to create Conv1DLayer: {}", e))?;
                    Box::new(conv_layer)
                },
                crate::persistence::LayerConfig::Dropout { .. } => {
                    let dropout_layer = DropoutLayer::from_layer_state(&layer_state)
                        .map_err(|e| anyhow::anyhow!("Failed to create DropoutLayer: {}", e))?;
                    Box::new(dropout_layer)
                },
                crate::persistence::LayerConfig::LayerNorm { .. } => {
                    let norm_layer = LayerNormLayer::from_layer_state(&layer_state)
                        .map_err(|e| anyhow::anyhow!("Failed to create LayerNormLayer: {}", e))?;
                    Box::new(norm_layer)
                },
                crate::persistence::LayerConfig::Embedding { .. } => {
                    // Skip embedding layers as they don't implement the Layer trait
                    // and are used differently (not in the main forward/backward pass)
                    continue;
                },
            };
            
            network_layers.push(layer);
        }
        
        // Create the neural network with the reconstructed layers
        let network = NeuralNetwork::new(network_layers, learning_rate);
        Ok(network)
    }
}

