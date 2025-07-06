use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    // Enable/disable adaptive features
    pub enabled: bool,
    
    // Evolution triggers
    pub evolution_interval_hours: u64,
    pub evolution_after_operations: usize,
    pub min_training_samples: usize,
    
    // Optimization objectives with weights
    pub objectives: HashMap<String, f32>,
    
    // Performance thresholds
    pub target_response_ms: f32,
    pub max_memory_mb: usize,
    pub min_accuracy: f32,
    
    // Evolution parameters
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f32,
    pub crossover_rate: f32,
    
    // Background processing
    pub evolution_thread_priority: ThreadPriority,
    pub max_evolution_time_minutes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreadPriority {
    Low,
    Normal,
    High,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        let mut objectives = HashMap::new();
        objectives.insert("response_time".to_string(), 0.4);
        objectives.insert("memory_efficiency".to_string(), 0.3);
        objectives.insert("accuracy".to_string(), 0.3);
        
        Self {
            enabled: true,
            evolution_interval_hours: 24,
            evolution_after_operations: 1000,
            min_training_samples: 500,
            objectives,
            target_response_ms: 10.0,
            max_memory_mb: 512,
            min_accuracy: 0.85,
            population_size: 50,
            generations: 20,
            mutation_rate: 0.05,
            crossover_rate: 0.8,
            evolution_thread_priority: ThreadPriority::Low,
            max_evolution_time_minutes: 30,
        }
    }
}

impl AdaptiveConfig {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_objectives(mut self, objectives: HashMap<String, f32>) -> Self {
        // Normalize weights to sum to 1.0
        let sum: f32 = objectives.values().sum();
        if sum > 0.0 {
            self.objectives = objectives.into_iter()
                .map(|(k, v)| (k, v / sum))
                .collect();
        }
        self
    }
    
    pub fn validate(&self) -> Result<(), String> {
        if self.min_training_samples == 0 {
            return Err("min_training_samples must be greater than 0".to_string());
        }
        
        if self.population_size == 0 {
            return Err("population_size must be greater than 0".to_string());
        }
        
        if self.mutation_rate < 0.0 || self.mutation_rate > 1.0 {
            return Err("mutation_rate must be between 0.0 and 1.0".to_string());
        }
        
        if self.crossover_rate < 0.0 || self.crossover_rate > 1.0 {
            return Err("crossover_rate must be between 0.0 and 1.0".to_string());
        }
        
        let weight_sum: f32 = self.objectives.values().sum();
        if (weight_sum - 1.0).abs() > 0.01 {
            return Err("objective weights must sum to 1.0".to_string());
        }
        
        Ok(())
    }
    
    pub fn should_trigger_evolution(&self, operation_count: usize, last_evolution: Option<chrono::DateTime<chrono::Utc>>) -> bool {
        if !self.enabled {
            return false;
        }
        
        // Check operation count trigger
        if operation_count >= self.evolution_after_operations {
            return true;
        }
        
        // Check time-based trigger
        if let Some(last) = last_evolution {
            let hours_since = chrono::Utc::now()
                .signed_duration_since(last)
                .num_hours() as u64;
            if hours_since >= self.evolution_interval_hours {
                return true;
            }
        }
        
        false
    }
}