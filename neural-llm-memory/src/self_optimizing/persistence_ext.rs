//! Persistence extension for self-optimizing networks

use super::SelfOptimizingNetwork;
use crate::persistence::{
    NetworkPersistence, NetworkState, NetworkStateBuilder, LayerState,
    PersistenceFormat, CheckpointManager, EvolutionHistory,
};
use crate::nn::NeuralNetwork;
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
        // This is a simplified version - in a real implementation, we'd need
        // to properly extract layers from the network
        for layer in network.get_layers() {
            // Try to downcast to specific layer types
            // This is a limitation of the current design - we'd need to modify
            // the Layer trait to support serialization
            
            // For now, we'll create placeholder states
            // In a real implementation, we'd need to add serialization support
            // to the Layer trait or use an enum-based approach
            
            // Skip for now - would need architectural changes
        }
        
        // Get the current genome
        let genome = self.evolution_controller.get_best_genome()
            .unwrap_or_else(|| self.get_best_architecture());
        
        // Build evolution history
        let evolution_history = EvolutionHistory {
            generation: self.generation,
            best_genomes: vec![(self.generation, genome.clone())],
            fitness_history: Vec::new(),
            successful_mutations: Vec::new(),
        };
        
        // Build the network state
        let state = NetworkStateBuilder::new(
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
        .with_insights(self.get_insights())
        .build()
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
        
        // Restore the genome
        if let Some((gen, genome)) = state.evolution_history.best_genomes.last() {
            network.generation = *gen;
            
            // Build network from genome
            let new_network = genome.to_network_with_io_sizes(
                state.metadata.input_size,
                state.metadata.output_size,
            ).map_err(|e| anyhow::anyhow!("Failed to build network from genome: {}", e))?;
            
            *network.best_network.write().unwrap() = new_network;
        }
        
        // Restore training state
        network.generation = state.evolution_history.generation;
        
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_save_load_network() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("network.bin");
        
        let config = SelfOptimizingConfig::default();
        let network = SelfOptimizingNetwork::new_with_io_sizes(config, 10, 5);
        
        // Save the network
        network.save(&path, PersistenceFormat::Binary).unwrap();
        
        // Check file exists
        assert!(path.exists());
        
        // Load the network
        let loaded = SelfOptimizingNetwork::load(&path, PersistenceFormat::Binary).unwrap();
        
        // Verify basic properties
        assert_eq!(loaded.input_size, network.input_size);
        assert_eq!(loaded.generation, network.generation);
    }
}