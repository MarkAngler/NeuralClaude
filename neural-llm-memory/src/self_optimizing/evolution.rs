//! Evolution controller for neural architecture search

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use rand::prelude::*;
use crate::nn::NeuralNetwork;
use ndarray::Array2;
use super::{ArchitectureGenome, SelfOptimizingConfig, FitnessEvaluator, OptimizationInsights, GenerationStats};

/// Controls the evolutionary process for architecture optimization
pub struct EvolutionController {
    /// Current population of architectures
    population: Vec<ArchitectureGenome>,
    
    /// Hall of fame - best architectures ever found
    hall_of_fame: Vec<ArchitectureGenome>,
    
    /// Configuration
    config: SelfOptimizingConfig,
    
    /// Fitness evaluator
    evaluator: FitnessEvaluator,
    
    /// Reference to best network
    best_network: Arc<RwLock<NeuralNetwork>>,
    
    /// Evolution history
    history: EvolutionHistory,
    
    /// Current generation
    generation: usize,
    
    /// Input size for networks
    pub input_size: usize,
    
    /// Output size for networks
    pub output_size: usize,
}

/// Evolution configuration
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub mutation_rate: f32,
    pub crossover_rate: f32,
    pub elite_size: usize,
    pub tournament_size: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            mutation_rate: 0.1,
            crossover_rate: 0.9,
            elite_size: 5,
            tournament_size: 3,
        }
    }
}

/// History of evolution process
#[derive(Debug, Clone)]
struct EvolutionHistory {
    generations: Vec<GenerationRecord>,
    best_fitness_history: Vec<f32>,
    diversity_history: Vec<f32>,
}

#[derive(Debug, Clone)]
struct GenerationRecord {
    generation: usize,
    best_fitness: f32,
    average_fitness: f32,
    worst_fitness: f32,
    diversity_score: f32,
    elite_count: usize,
    mutation_count: usize,
    crossover_count: usize,
}

impl EvolutionController {
    /// Create a new evolution controller
    pub fn new(
        config: SelfOptimizingConfig,
        best_network: Arc<RwLock<NeuralNetwork>>,
    ) -> Self {
        // Initialize random population
        let mut population = Vec::new();
        for _ in 0..config.population_size {
            population.push(ArchitectureGenome::random(&config));
        }
        
        let evaluator = FitnessEvaluator::new(config.objectives.clone());
        
        Self {
            population,
            hall_of_fame: Vec::new(),
            config,
            evaluator,
            best_network,
            history: EvolutionHistory {
                generations: Vec::new(),
                best_fitness_history: Vec::new(),
                diversity_history: Vec::new(),
            },
            generation: 0,
            input_size: 768, // Default input size
            output_size: 128, // Default output size
        }
    }
    
    /// Set the input size for network creation
    pub fn set_input_size(&mut self, input_size: usize) {
        self.input_size = input_size;
    }
    
    /// Set the output size for network creation
    pub fn set_output_size(&mut self, output_size: usize) {
        self.output_size = output_size;
    }
    
    /// Evolve for one generation
    pub fn evolve_generation(
        &mut self,
        validation_data: &[(Array2<f32>, Array2<f32>)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Step 1: Evaluate current population
        self.evaluate_population(validation_data)?;
        
        // Step 2: Update hall of fame
        self.update_hall_of_fame();
        
        // Step 3: Record generation statistics
        self.record_generation_stats();
        
        // Step 4: Create next generation
        let mut new_population = Vec::new();
        
        // Keep elite individuals
        let elite = self.select_elite();
        new_population.extend(elite);
        
        // Generate offspring through crossover and mutation
        while new_population.len() < self.config.population_size {
            if thread_rng().gen_bool(self.config.crossover_rate as f64) {
                // Crossover
                let parent1 = self.tournament_select();
                let parent2 = self.tournament_select();
                let mut offspring = parent1.crossover(&parent2);
                
                // Apply mutation to offspring
                if thread_rng().gen_bool(self.config.mutation_rate as f64) {
                    offspring.mutate(self.config.mutation_rate);
                }
                
                new_population.push(offspring);
            } else {
                // Direct mutation
                let mut individual = self.tournament_select();
                individual.mutate(self.config.mutation_rate);
                new_population.push(individual);
            }
        }
        
        // Replace population
        self.population = new_population;
        self.generation += 1;
        
        Ok(())
    }
    
    /// Evaluate fitness of all individuals in population
    fn evaluate_population(
        &mut self,
        validation_data: &[(Array2<f32>, Array2<f32>)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for genome in &mut self.population {
            // Skip if already evaluated
            if !genome.fitness_scores.is_empty() {
                continue;
            }
            
            // Convert genome to network with proper input and output sizes
            let network = genome.to_network_with_io_sizes(self.input_size, self.output_size)?;
            
            // Evaluate fitness
            let fitness_score = self.evaluator.evaluate(
                &network,
                validation_data,
                &self.config,
            )?;
            
            genome.fitness_scores = fitness_score.scores;
            genome.hardware_metrics = fitness_score.hardware_metrics;
        }
        
        Ok(())
    }
    
    /// Select elite individuals
    fn select_elite(&self) -> Vec<ArchitectureGenome> {
        let mut sorted_pop = self.population.clone();
        sorted_pop.sort_by(|a, b| {
            let a_fitness = a.fitness_scores.values().sum::<f32>();
            let b_fitness = b.fitness_scores.values().sum::<f32>();
            b_fitness.partial_cmp(&a_fitness).unwrap()
        });
        
        sorted_pop.into_iter()
            .take(self.config.elite_size)
            .collect()
    }
    
    /// Tournament selection
    fn tournament_select(&self) -> ArchitectureGenome {
        let mut rng = thread_rng();
        let tournament_size = 3;
        
        let mut best = None;
        let mut best_fitness = 0.0;
        
        for _ in 0..tournament_size {
            let idx = rng.gen_range(0..self.population.len());
            let individual = &self.population[idx];
            let fitness = individual.fitness_scores.values().sum::<f32>();
            
            if best.is_none() || fitness > best_fitness {
                best = Some(individual.clone());
                best_fitness = fitness;
            }
        }
        
        best.unwrap()
    }
    
    /// Update hall of fame with best individuals
    fn update_hall_of_fame(&mut self) {
        // Add best individual from current generation
        if let Some(best) = self.population.iter()
            .max_by(|a, b| {
                let a_fitness = a.fitness_scores.values().sum::<f32>();
                let b_fitness = b.fitness_scores.values().sum::<f32>();
                a_fitness.partial_cmp(&b_fitness).unwrap()
            })
        {
            self.hall_of_fame.push(best.clone());
        }
        
        // Keep only top N in hall of fame
        self.hall_of_fame.sort_by(|a, b| {
            let a_fitness = a.fitness_scores.values().sum::<f32>();
            let b_fitness = b.fitness_scores.values().sum::<f32>();
            b_fitness.partial_cmp(&a_fitness).unwrap()
        });
        
        self.hall_of_fame.truncate(10);
    }
    
    /// Calculate population diversity
    fn calculate_diversity(&self) -> f32 {
        if self.population.len() < 2 {
            return 0.0;
        }
        
        let mut diversity = 0.0;
        let mut count = 0;
        
        // Compare each pair of individuals
        for i in 0..self.population.len() {
            for j in (i + 1)..self.population.len() {
                let genome1 = &self.population[i];
                let genome2 = &self.population[j];
                
                // Calculate structural difference
                let layer_diff = (genome1.layers.len() as f32 - genome2.layers.len() as f32).abs();
                let conn_diff = (genome1.connections.len() as f32 - genome2.connections.len() as f32).abs();
                
                // Calculate hyperparameter difference
                let lr_diff = (genome1.hyperparameters.learning_rate - genome2.hyperparameters.learning_rate).abs();
                
                diversity += layer_diff + conn_diff + lr_diff * 100.0;
                count += 1;
            }
        }
        
        diversity / count as f32
    }
    
    /// Record statistics for current generation
    fn record_generation_stats(&mut self) {
        let fitness_values: Vec<f32> = self.population.iter()
            .map(|g| g.fitness_scores.values().sum::<f32>())
            .collect();
        
        let best_fitness = fitness_values.iter().cloned().fold(0.0f32, f32::max);
        let worst_fitness = fitness_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let average_fitness = fitness_values.iter().sum::<f32>() / fitness_values.len() as f32;
        let diversity_score = self.calculate_diversity();
        
        self.history.best_fitness_history.push(best_fitness);
        self.history.diversity_history.push(diversity_score);
        
        let record = GenerationRecord {
            generation: self.generation,
            best_fitness,
            average_fitness,
            worst_fitness,
            diversity_score,
            elite_count: self.config.elite_size,
            mutation_count: 0, // TODO: Track actual counts
            crossover_count: 0,
        };
        
        self.history.generations.push(record);
        
        println!("Generation {}: Best: {:.4}, Avg: {:.4}, Diversity: {:.4}",
            self.generation, best_fitness, average_fitness, diversity_score);
    }
    
    /// Get the best genome
    pub fn get_best_genome(&self) -> Option<ArchitectureGenome> {
        self.hall_of_fame.first().cloned()
            .or_else(|| {
                self.population.iter()
                    .max_by(|a, b| {
                        let a_fitness = a.fitness_scores.values().sum::<f32>();
                        let b_fitness = b.fitness_scores.values().sum::<f32>();
                        a_fitness.partial_cmp(&b_fitness).unwrap()
                    })
                    .cloned()
            })
    }
    
    /// Get optimization insights
    pub fn get_insights(&self) -> OptimizationInsights {
        // Analyze winning patterns
        let mut winning_patterns = Vec::new();
        
        // Common layer counts in top performers
        if !self.hall_of_fame.is_empty() {
            let avg_layers = self.hall_of_fame.iter()
                .map(|g| g.layers.len())
                .sum::<usize>() as f32 / self.hall_of_fame.len() as f32;
            winning_patterns.push(format!("Average layer count in top performers: {:.1}", avg_layers));
            
            // Most common activation functions
            let mut activation_counts = HashMap::new();
            for genome in &self.hall_of_fame {
                for layer in &genome.layers {
                    *activation_counts.entry(format!("{:?}", layer.activation))
                        .or_insert(0) += 1;
                }
            }
            
            if let Some((best_activation, _)) = activation_counts.iter()
                .max_by_key(|(_, count)| *count)
            {
                winning_patterns.push(format!("Most successful activation: {}", best_activation));
            }
        }
        
        // Current best scores
        let mut best_scores = HashMap::new();
        if let Some(best) = self.get_best_genome() {
            best_scores = best.fitness_scores.clone();
        }
        
        // Recommendations
        let mut recommendations = Vec::new();
        
        // Check diversity
        if let Some(diversity) = self.history.diversity_history.last() {
            if *diversity < 1.0 {
                recommendations.push("Low diversity - consider increasing mutation rate".to_string());
            }
        }
        
        // Check convergence
        if self.history.best_fitness_history.len() > 10 {
            let recent = &self.history.best_fitness_history[self.history.best_fitness_history.len() - 10..];
            let improvement = recent.last().unwrap() - recent.first().unwrap();
            if improvement < 0.01 {
                recommendations.push("Slow improvement - consider adjusting search space or objectives".to_string());
            }
        }
        
        OptimizationInsights {
            winning_patterns,
            failed_attempts: Vec::new(), // TODO: Track failures
            best_scores,
            recommendations,
            generation_stats: GenerationStats {
                current_generation: self.generation,
                best_fitness: self.history.best_fitness_history.last().cloned().unwrap_or(0.0),
                average_fitness: self.history.generations.last()
                    .map(|g| g.average_fitness)
                    .unwrap_or(0.0),
                diversity_score: self.history.diversity_history.last().cloned().unwrap_or(0.0),
                convergence_rate: self.calculate_convergence_rate(),
            },
        }
    }
    
    /// Calculate convergence rate
    fn calculate_convergence_rate(&self) -> f32 {
        if self.history.best_fitness_history.len() < 2 {
            return 0.0;
        }
        
        let window = 5.min(self.history.best_fitness_history.len());
        let recent = &self.history.best_fitness_history[self.history.best_fitness_history.len() - window..];
        
        let first = recent.first().unwrap();
        let last = recent.last().unwrap();
        
        (last - first) / window as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_evolution_controller_creation() {
        let config = SelfOptimizingConfig::default();
        let network = Arc::new(RwLock::new(
            crate::nn::NetworkBuilder::new()
                .add_linear(768, 512, crate::nn::ActivationFunction::ReLU, true)
                .build(0.001)
        ));
        
        let controller = EvolutionController::new(config.clone(), network);
        
        assert_eq!(controller.population.len(), config.population_size);
        assert_eq!(controller.generation, 0);
    }
    
    #[test]
    fn test_tournament_selection() {
        let config = SelfOptimizingConfig::default();
        let network = Arc::new(RwLock::new(
            crate::nn::NetworkBuilder::new()
                .add_linear(768, 512, crate::nn::ActivationFunction::ReLU, true)
                .build(0.001)
        ));
        
        let mut controller = EvolutionController::new(config, network);
        
        // Assign some fitness scores
        for (i, genome) in controller.population.iter_mut().enumerate() {
            genome.fitness_scores.insert("test".to_string(), i as f32);
        }
        
        // Tournament selection should favor higher fitness
        let selected = controller.tournament_select();
        let fitness = selected.fitness_scores.values().sum::<f32>();
        
        assert!(fitness > 0.0);
    }
}