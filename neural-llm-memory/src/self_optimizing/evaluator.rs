//! Fitness evaluation for neural architectures

use std::collections::HashMap;
use std::time::Instant;
use crate::nn::NeuralNetwork;
use ndarray::Array2;
use super::SelfOptimizingConfig;
use super::genome::HardwareMetrics;

/// Evaluates fitness of neural architectures
pub struct FitnessEvaluator {
    /// Weights for different objectives
    objective_weights: HashMap<String, f32>,
    
    /// Evaluation configuration
    eval_config: EvaluationConfig,
}

/// Configuration for evaluation
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Number of batches to evaluate
    pub eval_batches: usize,
    
    /// Timeout for evaluation
    pub timeout_seconds: u64,
    
    /// Memory measurement interval
    pub memory_sample_interval_ms: u64,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            eval_batches: 100,
            timeout_seconds: 60,
            memory_sample_interval_ms: 100,
        }
    }
}

/// Fitness score with multiple objectives
#[derive(Debug, Clone)]
pub struct FitnessScore {
    /// Individual objective scores
    pub scores: HashMap<String, f32>,
    
    /// Hardware metrics
    pub hardware_metrics: HardwareMetrics,
    
    /// Overall weighted fitness
    pub total_fitness: f32,
}

impl FitnessEvaluator {
    /// Create a new fitness evaluator
    pub fn new(objective_weights: HashMap<String, f32>) -> Self {
        Self {
            objective_weights,
            eval_config: EvaluationConfig::default(),
        }
    }
    
    /// Evaluate a neural network
    pub fn evaluate(
        &self,
        network: &NeuralNetwork,
        validation_data: &[(Array2<f32>, Array2<f32>)],
        config: &SelfOptimizingConfig,
    ) -> Result<FitnessScore, Box<dyn std::error::Error>> {
        let mut scores = HashMap::new();
        
        // Measure accuracy
        let accuracy = self.evaluate_accuracy(network, validation_data)?;
        scores.insert("accuracy".to_string(), accuracy);
        
        // Measure inference speed
        let (speed_score, inference_time_ms) = self.evaluate_speed(network, validation_data)?;
        scores.insert("speed".to_string(), speed_score);
        
        // Measure memory usage
        let (memory_score, memory_usage_mb) = self.evaluate_memory(network)?;
        scores.insert("memory".to_string(), memory_score);
        
        // Measure energy efficiency (simplified)
        let energy_score = self.evaluate_energy(network, inference_time_ms);
        scores.insert("energy".to_string(), energy_score);
        
        // Calculate parameter count
        let num_params = self.count_parameters(network);
        
        // Create hardware metrics
        let hardware_metrics = HardwareMetrics {
            inference_time_ms,
            memory_usage_mb,
            flops: self.estimate_flops(network, validation_data),
            energy_consumption: Some(inference_time_ms * memory_usage_mb * 0.001), // Simplified
        };
        
        // Apply constraints
        let mut constraint_penalty = 1.0;
        
        if memory_usage_mb > config.max_memory_mb as f32 {
            constraint_penalty *= 0.5; // Penalize over-memory usage
        }
        
        if inference_time_ms > config.target_inference_ms {
            constraint_penalty *= 0.7; // Penalize slow inference
        }
        
        // Calculate weighted fitness
        let mut total_fitness = 0.0;
        for (objective, score) in &scores {
            if let Some(weight) = self.objective_weights.get(objective) {
                total_fitness += score * weight * constraint_penalty;
            }
        }
        
        Ok(FitnessScore {
            scores,
            hardware_metrics,
            total_fitness,
        })
    }
    
    /// Evaluate accuracy on validation set
    fn evaluate_accuracy(
        &self,
        network: &NeuralNetwork,
        validation_data: &[(Array2<f32>, Array2<f32>)],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        if validation_data.is_empty() {
            return Ok(0.0);
        }
        
        let mut correct = 0;
        let mut total = 0;
        
        // Sample evaluation for efficiency
        let sample_size = self.eval_config.eval_batches.min(validation_data.len());
        let step = validation_data.len() / sample_size;
        
        for i in (0..validation_data.len()).step_by(step.max(1)) {
            let (input, target) = &validation_data[i];
            let output = network.forward(input, false); // Not training
            
            // Simple accuracy: check if prediction matches target
            if self.predictions_match(&output, target) {
                correct += 1;
            }
            total += 1;
            
            if total >= sample_size {
                break;
            }
        }
        
        Ok(correct as f32 / total as f32)
    }
    
    /// Evaluate inference speed
    fn evaluate_speed(
        &self,
        network: &NeuralNetwork,
        validation_data: &[(Array2<f32>, Array2<f32>)],
    ) -> Result<(f32, f32), Box<dyn std::error::Error>> {
        if validation_data.is_empty() {
            return Ok((0.0, 0.0));
        }
        
        // Warm up
        for _ in 0..5 {
            let (input, _) = &validation_data[0];
            let _ = network.forward(input, false);
        }
        
        // Time inference
        let mut total_time = 0.0;
        let num_samples = 100.min(validation_data.len());
        
        for i in 0..num_samples {
            let (input, _) = &validation_data[i % validation_data.len()];
            
            let start = Instant::now();
            let _ = network.forward(input, false);
            let elapsed = start.elapsed();
            
            total_time += elapsed.as_secs_f32() * 1000.0; // Convert to ms
        }
        
        let avg_inference_ms = total_time / num_samples as f32;
        
        // Convert to score (lower time = higher score)
        let speed_score = 1.0 / (1.0 + avg_inference_ms / 10.0); // Normalize around 10ms
        
        Ok((speed_score, avg_inference_ms))
    }
    
    /// Evaluate memory usage
    fn evaluate_memory(
        &self,
        network: &NeuralNetwork,
    ) -> Result<(f32, f32), Box<dyn std::error::Error>> {
        // Estimate memory usage based on parameter count
        let num_params = self.count_parameters(network);
        
        // Each parameter is f32 (4 bytes), plus activation memory
        let param_memory_mb = (num_params * 4) as f32 / (1024.0 * 1024.0);
        
        // Estimate activation memory (rough approximation)
        let activation_memory_mb = param_memory_mb * 0.5;
        
        let total_memory_mb = param_memory_mb + activation_memory_mb;
        
        // Convert to score (lower memory = higher score)
        let memory_score = 1.0 / (1.0 + total_memory_mb / 100.0); // Normalize around 100MB
        
        Ok((memory_score, total_memory_mb))
    }
    
    /// Evaluate energy efficiency
    fn evaluate_energy(
        &self,
        network: &NeuralNetwork,
        inference_time_ms: f32,
    ) -> f32 {
        // Simplified energy model
        let num_params = self.count_parameters(network);
        
        // Energy proportional to FLOPs and time
        let estimated_energy = (num_params as f32 * inference_time_ms) / 1e6;
        
        // Convert to score (lower energy = higher score)
        1.0 / (1.0 + estimated_energy)
    }
    
    /// Count network parameters
    fn count_parameters(&self, network: &NeuralNetwork) -> usize {
        // This is a simplified count - in real implementation,
        // we would iterate through all layers
        let layers = network.get_layers();
        let mut total = 0;
        
        for layer in layers {
            total += layer.count_parameters();
        }
        
        total
    }
    
    /// Estimate FLOPs for network
    fn estimate_flops(
        &self,
        network: &NeuralNetwork,
        validation_data: &[(Array2<f32>, Array2<f32>)],
    ) -> usize {
        if validation_data.is_empty() {
            return 0;
        }
        
        let (input, _) = &validation_data[0];
        let batch_size = input.shape()[0];
        let input_size = input.shape()[1];
        
        // Simplified FLOP estimation
        let layers = network.get_layers();
        let mut flops = 0;
        let mut current_size = input_size;
        
        for layer in layers {
            if let Some(output_size) = layer.output_size() {
                // Matrix multiply: 2 * batch * input * output
                flops += 2 * batch_size * current_size * output_size;
                current_size = output_size;
            }
        }
        
        flops
    }
    
    /// Check if predictions match targets
    fn predictions_match(&self, output: &Array2<f32>, target: &Array2<f32>) -> bool {
        // Simple comparison - in practice, this would depend on the task
        let output_flat = output.as_slice().unwrap();
        let target_flat = target.as_slice().unwrap();
        
        if output_flat.is_empty() || target_flat.is_empty() {
            return false;
        }
        
        // Find max indices
        let output_max_idx = output_flat.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
            
        let target_max_idx = target_flat.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        output_max_idx == target_max_idx
    }
}

/// Multi-objective optimization utilities
pub struct ParetoFrontier {
    /// Solutions on the Pareto frontier
    solutions: Vec<(FitnessScore, usize)>, // (score, genome_index)
}

impl ParetoFrontier {
    /// Create a new Pareto frontier
    pub fn new() -> Self {
        Self {
            solutions: Vec::new(),
        }
    }
    
    /// Update frontier with new solutions
    pub fn update(&mut self, scores: Vec<FitnessScore>) {
        self.solutions.clear();
        
        for (idx, score) in scores.iter().enumerate() {
            let mut dominated = false;
            
            // Check if this solution is dominated by any existing solution
            for (other_score, _) in &self.solutions {
                if self.dominates(other_score, score) {
                    dominated = true;
                    break;
                }
            }
            
            if !dominated {
                // Remove solutions dominated by this one
                let solutions_to_keep: Vec<_> = self.solutions.iter()
                    .filter(|(other_score, _)| !Self::dominates_static(score, other_score))
                    .cloned()
                    .collect();
                self.solutions = solutions_to_keep;
                
                self.solutions.push((score.clone(), idx));
            }
        }
    }
    
    /// Check if solution a dominates solution b
    fn dominates(&self, a: &FitnessScore, b: &FitnessScore) -> bool {
        Self::dominates_static(a, b)
    }
    
    /// Static version of dominates for use in closures
    fn dominates_static(a: &FitnessScore, b: &FitnessScore) -> bool {
        let mut at_least_one_better = false;
        
        for (objective, a_score) in &a.scores {
            if let Some(b_score) = b.scores.get(objective) {
                if a_score < b_score {
                    return false; // a is worse in this objective
                }
                if a_score > b_score {
                    at_least_one_better = true;
                }
            }
        }
        
        at_least_one_better
    }
    
    /// Get the frontier solutions
    pub fn get_frontier(&self) -> &[(FitnessScore, usize)] {
        &self.solutions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use crate::nn::{NetworkBuilder, ActivationFunction};
    
    #[test]
    fn test_fitness_evaluator() {
        let mut weights = HashMap::new();
        weights.insert("accuracy".to_string(), 0.5);
        weights.insert("speed".to_string(), 0.3);
        weights.insert("memory".to_string(), 0.2);
        
        let evaluator = FitnessEvaluator::new(weights);
        
        // Create a simple network
        let network = NetworkBuilder::new()
            .add_linear(10, 5, ActivationFunction::ReLU, true)
            .build(0.001);
        
        // Create dummy validation data
        let validation_data = vec![
            (
                Array2::from_shape_vec((1, 10), vec![1.0; 10]).unwrap(),
                Array2::from_shape_vec((1, 5), vec![0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            ),
        ];
        
        let config = SelfOptimizingConfig::default();
        let score = evaluator.evaluate(&network, &validation_data, &config).unwrap();
        
        assert!(score.total_fitness >= 0.0);
        assert!(!score.scores.is_empty());
    }
    
    #[test]
    fn test_pareto_frontier() {
        let mut frontier = ParetoFrontier::new();
        
        // Create test scores
        let mut scores = Vec::new();
        
        for i in 0..5 {
            let mut score_map = HashMap::new();
            score_map.insert("accuracy".to_string(), 0.1 * i as f32);
            score_map.insert("speed".to_string(), 0.9 - 0.1 * i as f32);
            
            scores.push(FitnessScore {
                scores: score_map,
                hardware_metrics: HardwareMetrics {
                    inference_time_ms: 10.0,
                    memory_usage_mb: 50.0,
                    flops: 1000000,
                    energy_consumption: None,
                },
                total_fitness: 0.5,
            });
        }
        
        frontier.update(scores);
        
        // All solutions should be on the frontier (trade-off between accuracy and speed)
        assert_eq!(frontier.get_frontier().len(), 5);
    }
}