//! Network state representation for persistence

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::self_optimizing::{
    ArchitectureGenome, SelfOptimizingConfig, OptimizationInsights,
    GenerationStats
};
use crate::self_optimizing::genome::{HyperparameterSet, HardwareMetrics};
use super::layer_state::LayerState;
use super::format::FormatVersion;

/// Complete state of a self-optimizing neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkState {
    /// Metadata about the network
    pub metadata: NetworkMetadata,
    
    /// Layer states (weights, biases, configs)
    pub layers: Vec<LayerState>,
    
    /// Architecture genome for evolution
    pub genome: ArchitectureGenome,
    
    /// Self-optimizing configuration
    pub config: SelfOptimizingConfig,
    
    /// Training state
    pub training_state: TrainingStateSnapshot,
    
    /// Evolution history
    pub evolution_history: EvolutionHistory,
    
    /// Optimization insights
    pub insights: OptimizationInsights,
}

/// Metadata about the saved network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetadata {
    /// Format version for compatibility
    pub version: FormatVersion,
    
    /// Network identifier
    pub id: uuid::Uuid,
    
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    
    /// Last modified timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
    
    /// Description or name
    pub description: String,
    
    /// Input/output dimensions
    pub input_size: usize,
    pub output_size: usize,
    
    /// Total number of parameters
    pub total_parameters: usize,
    
    /// Additional metadata
    pub tags: HashMap<String, String>,
}

/// Snapshot of training state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStateSnapshot {
    /// Current epoch
    pub epoch: usize,
    
    /// Total steps taken
    pub global_step: usize,
    
    /// Current learning rate
    pub learning_rate: f32,
    
    /// Weight decay
    pub weight_decay: f32,
    
    /// Loss history
    pub loss_history: Vec<f32>,
    
    /// Best validation loss
    pub best_val_loss: Option<f32>,
    
    /// Learning rate schedule state
    pub lr_schedule_state: Option<LRScheduleState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRScheduleState {
    pub initial_lr: f32,
    pub current_lr: f32,
    pub warmup_steps: usize,
    pub decay_steps: usize,
}

/// Evolution history for architecture search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionHistory {
    /// Current generation
    pub generation: usize,
    
    /// Best genomes from each generation
    pub best_genomes: Vec<(usize, ArchitectureGenome)>,
    
    /// Fitness progression
    pub fitness_history: Vec<GenerationStats>,
    
    /// Successful mutations
    pub successful_mutations: Vec<MutationRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRecord {
    pub generation: usize,
    pub mutation_type: String,
    pub fitness_improvement: f32,
    pub description: String,
}

impl NetworkState {
    /// Create a new network state
    pub fn new(
        genome: ArchitectureGenome,
        config: SelfOptimizingConfig,
        input_size: usize,
        output_size: usize,
    ) -> Self {
        Self {
            metadata: NetworkMetadata {
                version: FormatVersion::current(),
                id: uuid::Uuid::new_v4(),
                created_at: chrono::Utc::now(),
                modified_at: chrono::Utc::now(),
                description: "Self-optimizing neural network".to_string(),
                input_size,
                output_size,
                total_parameters: 0,
                tags: HashMap::new(),
            },
            layers: Vec::new(),
            genome,
            config,
            training_state: TrainingStateSnapshot {
                epoch: 0,
                global_step: 0,
                learning_rate: 0.001,
                weight_decay: 0.0,
                loss_history: Vec::new(),
                best_val_loss: None,
                lr_schedule_state: None,
            },
            evolution_history: EvolutionHistory {
                generation: 0,
                best_genomes: Vec::new(),
                fitness_history: Vec::new(),
                successful_mutations: Vec::new(),
            },
            insights: OptimizationInsights {
                winning_patterns: Vec::new(),
                failed_attempts: Vec::new(),
                best_scores: HashMap::new(),
                recommendations: Vec::new(),
                generation_stats: GenerationStats {
                    current_generation: 0,
                    best_fitness: 0.0,
                    average_fitness: 0.0,
                    diversity_score: 0.0,
                    convergence_rate: 0.0,
                },
            },
        }
    }
    
    /// Update metadata
    pub fn update_metadata(&mut self) {
        self.metadata.modified_at = chrono::Utc::now();
        self.metadata.total_parameters = self.layers.iter()
            .map(|l| l.count_parameters())
            .sum();
    }
    
    /// Add a layer state
    pub fn add_layer(&mut self, layer: LayerState) {
        self.layers.push(layer);
        self.update_metadata();
    }
    
    /// Update training state
    pub fn update_training_state(
        &mut self,
        epoch: usize,
        step: usize,
        loss: f32,
        learning_rate: f32,
    ) {
        self.training_state.epoch = epoch;
        self.training_state.global_step = step;
        self.training_state.learning_rate = learning_rate;
        self.training_state.loss_history.push(loss);
        
        if let Some(best) = self.training_state.best_val_loss {
            if loss < best {
                self.training_state.best_val_loss = Some(loss);
            }
        } else {
            self.training_state.best_val_loss = Some(loss);
        }
        
        self.update_metadata();
    }
    
    /// Record evolution progress
    pub fn record_evolution(
        &mut self,
        generation: usize,
        best_genome: ArchitectureGenome,
        stats: GenerationStats,
    ) {
        self.evolution_history.generation = generation;
        self.evolution_history.best_genomes.push((generation, best_genome));
        self.evolution_history.fitness_history.push(stats);
        self.update_metadata();
    }
    
    /// Validate the state
    pub fn validate(&self) -> Result<(), String> {
        if self.layers.is_empty() {
            return Err("No layers in network".to_string());
        }
        
        // Check layer compatibility
        for i in 1..self.layers.len() {
            let prev_output = self.layers[i-1].output_size();
            let curr_input = self.layers[i].input_size();
            
            if prev_output != curr_input {
                return Err(format!(
                    "Layer size mismatch: layer {} output ({}) != layer {} input ({})",
                    i-1, prev_output, i, curr_input
                ));
            }
        }
        
        Ok(())
    }
    
    /// Get a summary of the network state
    pub fn summary(&self) -> String {
        format!(
            "Network {} - {} layers, {} parameters, generation {}, best loss: {:.4}",
            self.metadata.id.to_string().chars().take(8).collect::<String>(),
            self.layers.len(),
            self.metadata.total_parameters,
            self.evolution_history.generation,
            self.training_state.best_val_loss.unwrap_or(f32::INFINITY)
        )
    }
}

/// Builder for NetworkState from a running network
pub struct NetworkStateBuilder {
    state: NetworkState,
}

impl NetworkStateBuilder {
    pub fn new(
        genome: ArchitectureGenome,
        config: SelfOptimizingConfig,
        input_size: usize,
        output_size: usize,
    ) -> Self {
        Self {
            state: NetworkState::new(genome, config, input_size, output_size),
        }
    }
    
    pub fn with_description(mut self, desc: &str) -> Self {
        self.state.metadata.description = desc.to_string();
        self
    }
    
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.state.metadata.tags.insert(key.to_string(), value.to_string());
        self
    }
    
    pub fn add_layer(mut self, layer: LayerState) -> Self {
        self.state.add_layer(layer);
        self
    }
    
    pub fn with_training_state(
        mut self,
        epoch: usize,
        step: usize,
        loss_history: Vec<f32>,
        learning_rate: f32,
    ) -> Self {
        self.state.training_state.epoch = epoch;
        self.state.training_state.global_step = step;
        self.state.training_state.loss_history = loss_history;
        self.state.training_state.learning_rate = learning_rate;
        self
    }
    
    pub fn with_evolution_history(mut self, history: EvolutionHistory) -> Self {
        self.state.evolution_history = history;
        self
    }
    
    pub fn with_insights(mut self, insights: OptimizationInsights) -> Self {
        self.state.insights = insights;
        self
    }
    
    pub fn build(mut self) -> Result<NetworkState, String> {
        self.state.update_metadata();
        self.state.validate()?;
        Ok(self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_network_state_creation() {
        let genome = ArchitectureGenome::random(&SelfOptimizingConfig::default());
        let config = SelfOptimizingConfig::default();
        
        let state = NetworkState::new(genome, config, 768, 128);
        
        assert_eq!(state.metadata.input_size, 768);
        assert_eq!(state.metadata.output_size, 128);
        assert_eq!(state.layers.len(), 0);
    }
    
    #[test]
    fn test_network_state_builder() {
        let genome = ArchitectureGenome::random(&SelfOptimizingConfig::default());
        let config = SelfOptimizingConfig::default();
        
        let result = NetworkStateBuilder::new(genome, config, 768, 128)
            .with_description("Test network")
            .with_tag("purpose", "testing")
            .build();
        
        assert!(result.is_ok());
        let state = result.unwrap();
        assert_eq!(state.metadata.description, "Test network");
        assert_eq!(state.metadata.tags.get("purpose"), Some(&"testing".to_string()));
    }
}