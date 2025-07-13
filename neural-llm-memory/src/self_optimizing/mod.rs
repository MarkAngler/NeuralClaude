//! Self-optimizing neural network module
//! 
//! This module implements automatic neural architecture search and optimization
//! for continuously improving network performance.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};
use crate::nn::{NeuralNetwork, NetworkBuilder, ActivationFunction};
use ndarray::Array2;

pub mod genome;
pub mod evolution;
pub mod evaluator;
pub mod optimizer;
pub mod network_ext;
pub mod optimizer_ext;
pub mod persistence_ext;

pub use genome::{ArchitectureGenome, LayerGene, ConnectionGene, MutationInfo};
pub use evolution::{EvolutionController, EvolutionConfig, EvolutionHistory, GenerationRecord};
pub use evaluator::{FitnessEvaluator, FitnessScore};
pub use optimizer::{OnlineOptimizer, AdaptationStrategy};

/// Configuration for self-optimizing networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfOptimizingConfig {
    /// Enable self-optimization
    pub enabled: bool,
    
    /// Maximum number of layers
    pub max_layers: usize,
    
    /// Minimum number of layers
    pub min_layers: usize,
    
    /// Layer size range
    pub layer_size_range: (usize, usize),
    
    /// Population size for evolution
    pub population_size: usize,
    
    /// Mutation rate
    pub mutation_rate: f32,
    
    /// Crossover rate
    pub crossover_rate: f32,
    
    /// Number of elite architectures to preserve
    pub elite_size: usize,
    
    /// Optimization objectives with weights
    pub objectives: HashMap<String, f32>,
    
    /// Hardware constraints
    pub max_memory_mb: usize,
    pub target_inference_ms: f32,
    
    /// Fitness threshold for early stopping
    pub fitness_threshold: f32,
    
    /// Adaptation interval for online learning
    pub adaptation_interval: usize,
}

impl Default for SelfOptimizingConfig {
    fn default() -> Self {
        let mut objectives = HashMap::new();
        objectives.insert("accuracy".to_string(), 0.4);
        objectives.insert("speed".to_string(), 0.3);
        objectives.insert("memory".to_string(), 0.2);
        objectives.insert("energy".to_string(), 0.1);
        
        Self {
            enabled: true,
            max_layers: 10,
            min_layers: 2,
            layer_size_range: (64, 2048),
            population_size: 50,
            mutation_rate: 0.1,
            crossover_rate: 0.9,
            elite_size: 5,
            objectives,
            max_memory_mb: 1024,
            target_inference_ms: 10.0,
            fitness_threshold: 0.8,
            adaptation_interval: 100,
        }
    }
}

/// Self-optimizing neural network that evolves its architecture
pub struct SelfOptimizingNetwork {
    /// Current best network
    best_network: Arc<RwLock<NeuralNetwork>>,
    
    /// Evolution controller
    evolution_controller: EvolutionController,
    
    /// Online optimizer for runtime adaptation
    online_optimizer: OnlineOptimizer,
    
    /// Configuration
    config: SelfOptimizingConfig,
    
    /// Training generation counter
    pub generation: usize,
    
    /// Input size for the network
    input_size: usize,
}

impl SelfOptimizingNetwork {
    /// Create a new self-optimizing network
    pub fn new(config: SelfOptimizingConfig) -> Self {
        Self::new_with_input_size(config, 768) // Default input size
    }
    
    /// Create a new self-optimizing network with specified input size
    pub fn new_with_input_size(config: SelfOptimizingConfig, input_size: usize) -> Self {
        Self::new_with_io_sizes(config, input_size, 128) // Default output size
    }
    
    /// Create a new self-optimizing network with specified input and output sizes
    pub fn new_with_io_sizes(config: SelfOptimizingConfig, input_size: usize, output_size: usize) -> Self {
        // Initialize with a simple default network
        let default_network = NetworkBuilder::new()
            .add_linear(input_size, 512, ActivationFunction::GELU, true)
            .add_dropout(0.1)
            .add_linear(512, 256, ActivationFunction::ReLU, true)
            .add_linear(256, output_size, ActivationFunction::ReLU, true)
            .build(0.001);
        
        let best_network = Arc::new(RwLock::new(default_network));
        
        let mut evolution_controller = EvolutionController::new(
            config.clone(),
            best_network.clone(),
        );
        evolution_controller.set_input_size(input_size);
        evolution_controller.set_output_size(output_size);
        
        let online_optimizer = OnlineOptimizer::new(
            best_network.clone(),
            AdaptationStrategy::default(),
        );
        
        Self {
            best_network,
            evolution_controller,
            online_optimizer,
            config,
            generation: 0,
            input_size,
        }
    }
    
    /// Train with automatic architecture optimization
    pub fn train_adaptive(
        &mut self,
        training_data: &[(Array2<f32>, Array2<f32>)],
        validation_data: &[(Array2<f32>, Array2<f32>)],
        epochs: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for epoch in 0..epochs {
            // Evolve architecture every N epochs
            if epoch % 10 == 0 && self.config.enabled {
                self.evolve_architecture(validation_data)?;
            }
            
            // Train current best network
            self.train_epoch(training_data)?;
            
            // Apply online optimizations during training
            if self.config.enabled {
                self.online_optimizer.adapt_during_epoch(epoch);
            }
            
            // Validate and log progress
            let val_loss = self.validate(validation_data)?;
            println!("Epoch {}: Validation loss: {:.4}", epoch, val_loss);
        }
        
        Ok(())
    }
    
    /// Evolve architecture for one generation
    pub fn evolve_architecture(
        &mut self,
        validation_data: &[(Array2<f32>, Array2<f32>)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("Generation {}: Evolving architecture...", self.generation);
        
        // Run evolution for one generation
        self.evolution_controller.evolve_generation(validation_data)?;
        
        // Update best network if a better one was found
        if let Some(best_genome) = self.evolution_controller.get_best_genome() {
            let new_network = best_genome.to_network_with_io_sizes(
                self.evolution_controller.input_size,
                self.evolution_controller.output_size
            )?;
            *self.best_network.write().unwrap() = new_network;
            
            println!("Found better architecture: {:?}", best_genome.get_summary());
        }
        
        self.generation += 1;
        Ok(())
    }
    
    /// Train for one epoch
    fn train_epoch(
        &self,
        training_data: &[(Array2<f32>, Array2<f32>)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut network = self.best_network.write().unwrap();
        
        for (input, target) in training_data {
            // Use train_step which handles forward, backward, and weight update
            let loss_fn = |output: &Array2<f32>, target: &Array2<f32>| {
                // MSE loss
                let diff = output - target;
                let loss = diff.iter().map(|x| x * x).sum::<f32>() / diff.len() as f32;
                (loss, diff.clone())
            };
            
            network.train_step(input, target, loss_fn);
        }
        
        Ok(())
    }
    
    /// Validate on validation set
    fn validate(
        &self,
        validation_data: &[(Array2<f32>, Array2<f32>)],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let network = self.best_network.read().unwrap();
        let mut total_loss = 0.0;
        
        for (input, target) in validation_data {
            let output = network.forward(input, false); // Not training
            let diff = &output - target;
            let loss = diff.iter().map(|x| x * x).sum::<f32>() / diff.len() as f32;
            total_loss += loss;
        }
        
        Ok(total_loss / validation_data.len() as f32)
    }
    
    /// Get the best discovered architecture
    pub fn get_best_architecture(&self) -> ArchitectureGenome {
        self.evolution_controller.get_best_genome()
            .expect("No best genome found")
    }
    
    /// Export the best network
    pub fn export_best(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let _network = self.best_network.read().unwrap();
        // TODO: Implement network serialization
        println!("Exporting network to {}", path);
        Ok(())
    }
    
    /// Get optimization insights
    pub fn get_insights(&self) -> OptimizationInsights {
        self.evolution_controller.get_insights()
    }
    
    /// Set the input size for network evolution
    pub fn set_input_size(&mut self, size: usize) {
        self.evolution_controller.set_input_size(size);
    }
    
    /// Set the output size for network evolution
    pub fn set_output_size(&mut self, size: usize) {
        self.evolution_controller.set_output_size(size);
    }
}

/// Insights from the optimization process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationInsights {
    /// Most successful architecture patterns
    pub winning_patterns: Vec<String>,
    
    /// Failed attempts and reasons
    pub failed_attempts: Vec<(String, String)>,
    
    /// Current best fitness scores
    pub best_scores: HashMap<String, f32>,
    
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    
    /// Generation statistics
    pub generation_stats: GenerationStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub current_generation: usize,
    pub best_fitness: f32,
    pub average_fitness: f32,
    pub diversity_score: f32,
    pub convergence_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_self_optimizing_network_creation() {
        let config = SelfOptimizingConfig::default();
        let network = SelfOptimizingNetwork::new(config);
        
        assert_eq!(network.generation, 0);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = SelfOptimizingConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SelfOptimizingConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.population_size, deserialized.population_size);
    }
}