//! Architecture genome representation for neural architecture search

use std::collections::HashMap;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use crate::nn::{NeuralNetwork, NetworkBuilder, ActivationFunction};
use rand::prelude::*;

/// Represents a complete neural network architecture as a genome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureGenome {
    /// Unique identifier
    pub id: Uuid,
    
    /// Layer definitions
    pub layers: Vec<LayerGene>,
    
    /// Connection topology
    pub connections: Vec<ConnectionGene>,
    
    /// Hyperparameters
    pub hyperparameters: HyperparameterSet,
    
    /// Performance metrics
    pub fitness_scores: HashMap<String, f32>,
    
    /// Hardware metrics
    pub hardware_metrics: HardwareMetrics,
    
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    
    /// Number of parameters
    pub num_parameters: usize,
}

/// Gene representing a single layer
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerGene {
    pub id: usize,
    pub layer_type: LayerType,
    pub size: usize,
    pub activation: ActivationFunction,
    pub dropout_rate: Option<f32>,
    pub use_batch_norm: bool,
    pub use_layer_norm: bool,
}

/// Types of layers available
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LayerType {
    Linear,
    Conv1D { kernel_size: usize, stride: usize },
    Attention { num_heads: usize },
    LSTM { hidden_size: usize },
    GRU { hidden_size: usize },
}

/// Gene representing a connection between layers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConnectionGene {
    pub from_layer: usize,
    pub to_layer: usize,
    pub connection_type: ConnectionType,
    pub enabled: bool,
    pub weight_scale: f32,
}

/// Types of connections
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConnectionType {
    Sequential,    // Standard forward connection
    Skip,          // Skip connection (residual)
    Dense,         // Dense connection (like DenseNet)
    Attention,     // Attention-based connection
}

/// Hyperparameter set for the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSet {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub weight_decay: f32,
    pub gradient_clip: Option<f32>,
    pub warmup_steps: usize,
    pub optimizer_type: OptimizerType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizerType {
    SGD { momentum: f32 },
    Adam { beta1: f32, beta2: f32, epsilon: f32 },
    AdamW { beta1: f32, beta2: f32, epsilon: f32 },
    RMSprop { alpha: f32, epsilon: f32 },
}

/// Hardware performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareMetrics {
    pub inference_time_ms: f32,
    pub memory_usage_mb: f32,
    pub flops: usize,
    pub energy_consumption: Option<f32>,
}

impl ArchitectureGenome {
    /// Create a new random genome
    pub fn random(config: &super::SelfOptimizingConfig) -> Self {
        let mut rng = thread_rng();
        let num_layers = rng.gen_range(config.min_layers..=config.max_layers);
        
        let mut layers = Vec::new();
        let mut _current_size = 768; // Default input size, will be overridden
        
        for i in 0..num_layers {
            let size = rng.gen_range(config.layer_size_range.0..=config.layer_size_range.1);
            let activation = match rng.gen_range(0..4) {
                0 => ActivationFunction::ReLU,
                1 => ActivationFunction::GELU,
                2 => ActivationFunction::SiLU,
                _ => ActivationFunction::LeakyReLU(0.01),
            };
            
            layers.push(LayerGene {
                id: i,
                layer_type: LayerType::Linear,
                size,
                activation,
                dropout_rate: if rng.gen_bool(0.5) { Some(rng.gen_range(0.1..0.5)) } else { None },
                use_batch_norm: rng.gen_bool(0.3),
                use_layer_norm: rng.gen_bool(0.3),
            });
            
            _current_size = size;
        }
        
        // Generate connections (initially sequential)
        let mut connections = Vec::new();
        for i in 0..layers.len() - 1 {
            connections.push(ConnectionGene {
                from_layer: i,
                to_layer: i + 1,
                connection_type: ConnectionType::Sequential,
                enabled: true,
                weight_scale: 1.0,
            });
        }
        
        // Add some skip connections
        if layers.len() > 2 && rng.gen_bool(0.5) {
            for i in 0..layers.len() - 2 {
                if rng.gen_bool(0.3) {
                    connections.push(ConnectionGene {
                        from_layer: i,
                        to_layer: i + 2,
                        connection_type: ConnectionType::Skip,
                        enabled: true,
                        weight_scale: 1.0,
                    });
                }
            }
        }
        
        let hyperparameters = HyperparameterSet {
            learning_rate: 10f32.powf(rng.gen_range(-5.0..-2.0)),
            batch_size: *[16, 32, 64, 128].choose(&mut rng).unwrap(),
            weight_decay: 10f32.powf(rng.gen_range(-5.0..-2.0)),
            gradient_clip: if rng.gen_bool(0.5) { Some(rng.gen_range(0.5..5.0)) } else { None },
            warmup_steps: rng.gen_range(0..1000),
            optimizer_type: match rng.gen_range(0..3) {
                0 => OptimizerType::SGD { momentum: rng.gen_range(0.8..0.99) },
                1 => OptimizerType::Adam { 
                    beta1: 0.9, 
                    beta2: 0.999, 
                    epsilon: 1e-8 
                },
                _ => OptimizerType::AdamW { 
                    beta1: 0.9, 
                    beta2: 0.999, 
                    epsilon: 1e-8 
                },
            },
        };
        
        Self {
            id: Uuid::new_v4(),
            layers,
            connections,
            hyperparameters,
            fitness_scores: HashMap::new(),
            hardware_metrics: HardwareMetrics {
                inference_time_ms: 0.0,
                memory_usage_mb: 0.0,
                flops: 0,
                energy_consumption: None,
            },
            created_at: chrono::Utc::now(),
            num_parameters: 0,
        }
    }
    
    /// Convert genome to actual neural network
    pub fn to_network(&self) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        self.to_network_with_input_size(768) // Default input size
    }
    
    /// Convert genome to actual neural network with specified input size
    pub fn to_network_with_input_size(&self, input_size: usize) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        self.to_network_with_io_sizes(input_size, 128) // Default output size
    }
    
    /// Convert genome to actual neural network with specified input and output sizes
    pub fn to_network_with_io_sizes(&self, input_size: usize, output_size: usize) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut builder = NetworkBuilder::new();
        
        // Add layers based on genome
        for (i, layer) in self.layers.iter().enumerate() {
            match &layer.layer_type {
                LayerType::Linear => {
                    let layer_input_size = if i == 0 { 
                        input_size // Use provided input size
                    } else { 
                        self.layers[i - 1].size 
                    };
                    
                    builder = builder.add_linear(
                        layer_input_size,
                        layer.size,
                        layer.activation.clone(),
                        true,
                    );
                    
                    if let Some(dropout) = layer.dropout_rate {
                        builder = builder.add_dropout(dropout);
                    }
                    
                    if layer.use_layer_norm {
                        builder = builder.add_layer_norm(layer.size);
                    }
                }
                _ => {
                    // TODO: Implement other layer types
                }
            }
        }
        
        // Add final output layer if needed
        let last_size = self.layers.last().map(|l| l.size).unwrap_or(input_size);
        if last_size != output_size {
            builder = builder.add_linear(last_size, output_size, ActivationFunction::Identity, true);
        }
        
        // Build network with hyperparameters
        let network = builder.build(self.hyperparameters.learning_rate);
        
        Ok(network)
    }
    
    /// Mutate the genome
    pub fn mutate(&mut self, mutation_rate: f32) {
        let mut rng = thread_rng();
        
        // Mutate layers
        for layer in &mut self.layers {
            if rng.gen_bool(mutation_rate as f64) {
                // Mutate layer size
                layer.size = (layer.size as f32 * rng.gen_range(0.8..1.2)) as usize;
                layer.size = layer.size.clamp(32, 4096);
            }
            
            if rng.gen_bool(mutation_rate as f64 * 0.5) {
                // Mutate activation
                layer.activation = match rng.gen_range(0..4) {
                    0 => ActivationFunction::ReLU,
                    1 => ActivationFunction::GELU,
                    2 => ActivationFunction::SiLU,
                    _ => ActivationFunction::LeakyReLU(0.01),
                };
            }
            
            if rng.gen_bool(mutation_rate as f64 * 0.3) {
                // Toggle dropout
                layer.dropout_rate = if layer.dropout_rate.is_some() {
                    None
                } else {
                    Some(rng.gen_range(0.1..0.5))
                };
            }
        }
        
        // Mutate connections
        for conn in &mut self.connections {
            if rng.gen_bool(mutation_rate as f64 * 0.2) {
                conn.enabled = !conn.enabled;
            }
        }
        
        // Mutate hyperparameters
        if rng.gen_bool(mutation_rate as f64) {
            self.hyperparameters.learning_rate *= rng.gen_range(0.5..2.0);
            self.hyperparameters.learning_rate = self.hyperparameters.learning_rate.clamp(1e-6, 1.0);
        }
    }
    
    /// Crossover two genomes to create offspring
    pub fn crossover(&self, other: &Self) -> Self {
        let mut rng = thread_rng();
        let mut offspring = self.clone();
        offspring.id = Uuid::new_v4();
        
        // Crossover layers
        for i in 0..offspring.layers.len().min(other.layers.len()) {
            if rng.gen_bool(0.5) {
                offspring.layers[i] = other.layers[i].clone();
            }
        }
        
        // Crossover connections
        for i in 0..offspring.connections.len().min(other.connections.len()) {
            if rng.gen_bool(0.5) {
                offspring.connections[i] = other.connections[i].clone();
            }
        }
        
        // Crossover hyperparameters
        if rng.gen_bool(0.5) {
            offspring.hyperparameters = other.hyperparameters.clone();
        }
        
        offspring.fitness_scores.clear();
        offspring.created_at = chrono::Utc::now();
        
        offspring
    }
    
    /// Get a summary of the architecture
    pub fn get_summary(&self) -> String {
        format!(
            "Genome {}: {} layers, {} connections, LR: {:.6}, Fitness: {:.4}",
            self.id.to_string()[..8].to_string(),
            self.layers.len(),
            self.connections.iter().filter(|c| c.enabled).count(),
            self.hyperparameters.learning_rate,
            self.fitness_scores.values().sum::<f32>()
        )
    }
    
    /// Calculate total number of parameters
    pub fn calculate_parameters(&mut self) {
        let mut params = 0;
        let mut prev_size = 768; // Input size
        
        for layer in &self.layers {
            match &layer.layer_type {
                LayerType::Linear => {
                    params += prev_size * layer.size + layer.size; // weights + bias
                    prev_size = layer.size;
                }
                _ => {
                    // TODO: Calculate for other layer types
                }
            }
        }
        
        self.num_parameters = params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_random_genome_generation() {
        let config = crate::self_optimizing::SelfOptimizingConfig::default();
        let genome = ArchitectureGenome::random(&config);
        
        assert!(genome.layers.len() >= config.min_layers);
        assert!(genome.layers.len() <= config.max_layers);
        assert!(!genome.connections.is_empty());
    }
    
    #[test]
    fn test_genome_mutation() {
        let config = crate::self_optimizing::SelfOptimizingConfig::default();
        let mut genome = ArchitectureGenome::random(&config);
        let original = genome.clone();
        
        genome.mutate(1.0); // High mutation rate for testing
        
        // Something should have changed
        assert!(
            genome.layers != original.layers || 
            genome.connections != original.connections ||
            genome.hyperparameters.learning_rate != original.hyperparameters.learning_rate
        );
    }
    
    #[test]
    fn test_genome_crossover() {
        let config = crate::self_optimizing::SelfOptimizingConfig::default();
        let parent1 = ArchitectureGenome::random(&config);
        let parent2 = ArchitectureGenome::random(&config);
        
        let offspring = parent1.crossover(&parent2);
        
        assert_ne!(offspring.id, parent1.id);
        assert_ne!(offspring.id, parent2.id);
    }
}