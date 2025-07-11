//! Strategy selection and evaluation for metacognitive thinking
//! 
//! Implements dynamic strategy selection based on task characteristics
//! and current cognitive state.

// Stub types for compilation (TODO: Move to proper modules)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingStrategy {
    pub name: String,
    pub effectiveness: f32,
}

#[derive(Debug, Clone)]
pub struct ThinkingInput {
    pub task: String,
    pub context: HashMap<String, String>,
    pub constraints: Vec<Constraint>,
    pub available_time: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct MetaCognitiveConfig {
    pub strategy_switch_threshold: f32,
    pub optimization_rate: f32,
}
use crate::metacognition::monitor::CognitiveState;
use crate::nn::{NeuralNetwork, Layer, Activation};

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use ndarray::{Array1, Array2, arr1};
use serde::{Serialize, Deserialize};

/// Strategy evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyEvaluation {
    pub recommended_strategy: ThinkingStrategy,
    pub confidence: f32,
    pub alternative_strategies: Vec<(ThinkingStrategy, f32)>,
    pub reasoning: String,
}

/// Strategy selection subsystem
pub struct StrategySelector {
    /// Available thinking strategies
    strategies: Vec<ThinkingStrategy>,
    
    /// Neural network for strategy effectiveness prediction
    effectiveness_model: Arc<RwLock<NeuralNetwork>>,
    
    /// Context encoder network
    context_encoder: Arc<RwLock<NeuralNetwork>>,
    
    /// Strategy transition probabilities
    transition_matrix: Arc<RwLock<Array2<f32>>>,
    
    /// Historical strategy performance
    performance_history: Arc<RwLock<HashMap<String, StrategyPerformance>>>,
    
    /// Configuration
    config: StrategySelectorConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySelectorConfig {
    pub min_confidence_threshold: f32,
    pub exploration_rate: f32,
    pub transition_smoothing: f32,
    pub history_window: usize,
}

impl Default for StrategySelectorConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.6,
            exploration_rate: 0.1,
            transition_smoothing: 0.9,
            history_window: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StrategyPerformance {
    pub success_count: usize,
    pub failure_count: usize,
    pub average_time: f32,
    pub average_confidence: f32,
    pub context_effectiveness: HashMap<String, f32>,
}

impl StrategySelector {
    /// Create new strategy selector
    pub fn new(config: &MetaCognitiveConfig) -> Self {
        let strategies = Self::initialize_strategies();
        let num_strategies = strategies.len();
        
        // Create effectiveness prediction model
        let effectiveness_model = NeuralNetwork::new(
            vec![
                Box::new(crate::nn::LinearLayer::new(512, 256, crate::nn::ActivationFunction::ReLU, true, crate::nn::WeightInit::Xavier)),
                Box::new(crate::nn::LinearLayer::new(256, 128, crate::nn::ActivationFunction::ReLU, true, crate::nn::WeightInit::Xavier)),
                Box::new(crate::nn::LinearLayer::new(128, num_strategies, crate::nn::ActivationFunction::Softmax, true, crate::nn::WeightInit::Xavier)),
            ],
            0.001,
        );
        
        // Create context encoder
        let context_encoder = NeuralNetwork::new(
            vec![
                Box::new(crate::nn::LinearLayer::new(512, 256, crate::nn::ActivationFunction::ReLU, true, crate::nn::WeightInit::Xavier)),
                Box::new(crate::nn::LinearLayer::new(256, 128, crate::nn::ActivationFunction::ReLU, true, crate::nn::WeightInit::Xavier)),
                Box::new(crate::nn::LinearLayer::new(128, 64, crate::nn::ActivationFunction::Tanh, true, crate::nn::WeightInit::Xavier)),
            ],
            0.001,
        );
        
        // Initialize transition matrix with uniform probabilities
        let mut transition_matrix = Array2::zeros((num_strategies, num_strategies));
        for i in 0..num_strategies {
            for j in 0..num_strategies {
                transition_matrix[[i, j]] = 1.0 / num_strategies as f32;
            }
        }
        
        Self {
            strategies,
            effectiveness_model: Arc::new(RwLock::new(effectiveness_model)),
            context_encoder: Arc::new(RwLock::new(context_encoder)),
            transition_matrix: Arc::new(RwLock::new(transition_matrix)),
            performance_history: Arc::new(RwLock::new(HashMap::new())),
            config: StrategySelectorConfig::default(),
        }
    }
    
    /// Initialize available thinking strategies
    fn initialize_strategies() -> Vec<ThinkingStrategy> {
        vec![
            ThinkingStrategy {
                name: "Convergent".to_string(),
                effectiveness: 0.8,
            },
            ThinkingStrategy {
                name: "Divergent".to_string(),
                effectiveness: 0.7,
            },
            ThinkingStrategy {
                name: "Lateral".to_string(),
                effectiveness: 0.9,
            },
            ThinkingStrategy {
                name: "Systems".to_string(),
                effectiveness: 0.8,
            },
            ThinkingStrategy {
                name: "Critical".to_string(),
                effectiveness: 0.8,
            },
            ThinkingStrategy {
                name: "Abstract".to_string(),
                effectiveness: 0.7,
            },
        ]
    }
    
    /// Select optimal strategy for current task and state
    pub fn select_strategy(
        &self,
        input: &ThinkingInput,
        state: &CognitiveState,
    ) -> ThinkingStrategy {
        // Encode context
        let context_vector = self.encode_context(input, state);
        
        // Get strategy predictions
        let predictions = self.predict_strategy_effectiveness(&context_vector);
        
        // Apply exploration
        let selected_idx = if rand::random::<f32>() < self.config.exploration_rate {
            // Random exploration
            rand::random::<usize>() % self.strategies.len()
        } else {
            // Choose best predicted strategy
            predictions.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        };
        
        // Consider transitions from current strategy
        let adjusted_idx = self.apply_transition_constraints(selected_idx, state);
        
        self.strategies[adjusted_idx].clone()
    }
    
    /// Encode task input and cognitive state into vector
    fn encode_context(&self, input: &ThinkingInput, state: &CognitiveState) -> Array1<f32> {
        let mut features = Vec::new();
        
        // Task features
        features.push(input.task.len() as f32 / 1000.0); // Normalized length
        features.push(input.constraints.len() as f32 / 10.0);
        features.push(input.available_time.as_secs() as f32 / 3600.0);
        
        // Extract task type indicators
        let task_lower = input.task.to_lowercase();
        features.push(if task_lower.contains("analyze") { 1.0 } else { 0.0 });
        features.push(if task_lower.contains("create") { 1.0 } else { 0.0 });
        features.push(if task_lower.contains("optimize") { 1.0 } else { 0.0 });
        features.push(if task_lower.contains("debug") { 1.0 } else { 0.0 });
        features.push(if task_lower.contains("design") { 1.0 } else { 0.0 });
        
        // Cognitive state features
        features.push(state.performance_metrics.average_confidence);
        features.push(state.performance_metrics.self_awareness_score);
        features.push(state.recent_decisions.len() as f32 / 10.0);
        
        // Pattern activity
        for strategy in &self.strategies {
            let pattern_active = state.active_patterns.iter()
                .any(|p| self.pattern_matches_strategy(p, strategy));
            features.push(if pattern_active { 1.0 } else { 0.0 });
        }
        
        // Context complexity
        let context_complexity = input.context.len() as f32 / 20.0;
        features.push(context_complexity.min(1.0));
        
        // Pad or truncate to expected size
        features.resize(512, 0.0);
        
        // Encode through context network
        let encoder = self.context_encoder.read().unwrap();
        let input_arr = arr1(&features).insert_axis(ndarray::Axis(0)); // Add batch dimension
        let output = encoder.forward(&input_arr, false);
        output.remove_axis(ndarray::Axis(0)) // Remove batch dimension
    }
    
    /// Predict strategy effectiveness scores
    fn predict_strategy_effectiveness(&self, context: &Array1<f32>) -> Vec<f32> {
        let model = self.effectiveness_model.read().unwrap();
        let input_arr = context.clone().insert_axis(ndarray::Axis(0)); // Add batch dimension
        let predictions = model.forward(&input_arr, false);
        predictions.remove_axis(ndarray::Axis(0)).to_vec() // Remove batch dimension and convert to vec
    }
    
    /// Apply transition constraints based on current state
    fn apply_transition_constraints(
        &self,
        selected_idx: usize,
        state: &CognitiveState,
    ) -> usize {
        // Find current strategy if any
        let current_strategy_idx = state.active_patterns.first()
            .and_then(|p| self.find_strategy_index_by_name(&p.pattern_type));
        
        if let Some(current_idx) = current_strategy_idx {
            let transition_matrix = self.transition_matrix.read().unwrap();
            let transition_probs = transition_matrix.row(current_idx);
            
            // Check if transition is allowed
            if transition_probs[selected_idx] < 0.1 {
                // Find best alternative
                transition_probs.iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != current_idx)
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(selected_idx)
            } else {
                selected_idx
            }
        } else {
            selected_idx
        }
    }
    
    /// Evaluate a strategy's effectiveness for given context
    pub fn evaluate_strategy(
        &self,
        strategy: &ThinkingStrategy,
        input: &ThinkingInput,
        state: &CognitiveState,
    ) -> StrategyEvaluation {
        let context = self.encode_context(input, state);
        let all_predictions = self.predict_strategy_effectiveness(&context);
        
        let strategy_idx = self.find_strategy_index(strategy)
            .unwrap_or(0);
        
        let confidence = all_predictions[strategy_idx];
        
        // Get alternative strategies
        let mut alternatives: Vec<(ThinkingStrategy, f32)> = self.strategies.iter()
            .enumerate()
            .filter(|(idx, _)| *idx != strategy_idx)
            .map(|(idx, strat)| (strat.clone(), all_predictions[idx]))
            .collect();
        
        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        alternatives.truncate(3);
        
        // Generate reasoning
        let reasoning = self.generate_strategy_reasoning(strategy, input, confidence);
        
        StrategyEvaluation {
            recommended_strategy: strategy.clone(),
            confidence,
            alternative_strategies: alternatives,
            reasoning,
        }
    }
    
    /// Update strategy performance based on results
    pub fn update_performance(
        &mut self,
        strategy: &ThinkingStrategy,
        context: &str,
        success: bool,
        time_taken: f32,
        confidence: f32,
    ) {
        let strategy_name = format!("{:?}", strategy);
        
        let mut history = self.performance_history.write().unwrap();
        let performance = history.entry(strategy_name).or_insert_with(|| {
            StrategyPerformance {
                success_count: 0,
                failure_count: 0,
                average_time: 0.0,
                average_confidence: 0.0,
                context_effectiveness: HashMap::new(),
            }
        });
        
        // Update counts
        if success {
            performance.success_count += 1;
        } else {
            performance.failure_count += 1;
        }
        
        // Update averages
        let total_count = (performance.success_count + performance.failure_count) as f32;
        performance.average_time = (performance.average_time * (total_count - 1.0) + time_taken) / total_count;
        performance.average_confidence = (performance.average_confidence * (total_count - 1.0) + confidence) / total_count;
        
        // Update context effectiveness
        let effectiveness = if success { 1.0 } else { 0.0 };
        performance.context_effectiveness
            .entry(context.to_string())
            .and_modify(|e| *e = (*e + effectiveness) / 2.0)
            .or_insert(effectiveness);
        
        // Update transition matrix
        self.update_transition_matrix(strategy, success);
    }
    
    /// Update transition matrix based on strategy success
    fn update_transition_matrix(&self, from_strategy: &ThinkingStrategy, success: bool) {
        let from_idx = self.find_strategy_index(from_strategy);
        
        if let Some(from_idx) = from_idx {
            let mut matrix = self.transition_matrix.write().unwrap();
            
            // Increase self-transition if successful
            if success {
                matrix[[from_idx, from_idx]] = (matrix[[from_idx, from_idx]] + 0.1).min(0.9);
            } else {
                // Decrease self-transition if failed
                matrix[[from_idx, from_idx]] = (matrix[[from_idx, from_idx]] - 0.05).max(0.1);
            }
            
            // Normalize row
            let row_sum: f32 = matrix.row(from_idx).sum();
            for j in 0..matrix.ncols() {
                matrix[[from_idx, j]] /= row_sum;
            }
        }
    }
    
    /// Find strategy index by reference
    fn find_strategy_index(&self, strategy: &ThinkingStrategy) -> Option<usize> {
        self.strategies.iter()
            .position(|s| s.name == strategy.name)
    }

    /// Find strategy index by name
    fn find_strategy_index_by_name(&self, strategy_name: &str) -> Option<usize> {
        self.strategies.iter()
            .position(|s| s.name == strategy_name)
    }
    
    /// Check if pattern matches strategy
    fn pattern_matches_strategy(
        &self,
        pattern: &crate::metacognition::monitor::CognitivePattern,
        strategy: &ThinkingStrategy,
    ) -> bool {
        pattern.pattern_type == strategy.name
    }
    
    /// Generate reasoning for strategy selection
    fn generate_strategy_reasoning(
        &self,
        strategy: &ThinkingStrategy,
        input: &ThinkingInput,
        confidence: f32,
    ) -> String {
        let task_type = self.classify_task(input);
        
        match strategy.name.as_str() {
            "Convergent" => {
                format!(
                    "Convergent thinking selected (confidence: {:.2}) for {} task. \
                     This approach will systematically narrow down possibilities to find the optimal solution.",
                    confidence, task_type
                )
            },
            "Divergent" => {
                format!(
                    "Divergent thinking selected (confidence: {:.2}) for {} task. \
                     This will explore multiple creative possibilities before converging on a solution.",
                    confidence, task_type
                )
            },
            "Lateral" => {
                format!(
                    "Lateral thinking selected (confidence: {:.2}) for {} task. \
                     This approach will seek unexpected connections and challenge assumptions.",
                    confidence, task_type
                )
            },
            "Systems" => {
                format!(
                    "Systems thinking selected (confidence: {:.2}) for {} task. \
                     This will analyze the problem holistically, considering all interconnections.",
                    confidence, task_type
                )
            },
            "Critical" => {
                format!(
                    "Critical thinking selected (confidence: {:.2}) for {} task. \
                     This approach will carefully evaluate evidence and question assumptions.",
                    confidence, task_type
                )
            },
            "Abstract" => {
                format!(
                    "Abstract thinking selected (confidence: {:.2}) for {} task. \
                     This will extract general principles and patterns from specific instances.",
                    confidence, task_type
                )
            },
            _ => format!(
                "Strategy '{}' selected (confidence: {:.2}) for {} task.",
                strategy.name, confidence, task_type
            ),
        }
    }
    
    /// Classify task type from input
    fn classify_task(&self, input: &ThinkingInput) -> &'static str {
        let task_lower = input.task.to_lowercase();
        
        if task_lower.contains("analyze") || task_lower.contains("evaluate") {
            "analytical"
        } else if task_lower.contains("create") || task_lower.contains("design") {
            "creative"
        } else if task_lower.contains("optimize") || task_lower.contains("improve") {
            "optimization"
        } else if task_lower.contains("debug") || task_lower.contains("fix") {
            "problem-solving"
        } else if task_lower.contains("understand") || task_lower.contains("explain") {
            "comprehension"
        } else {
            "general"
        }
    }
}

use std::time::Duration;
// Stub type for compilation
#[derive(Debug, Clone)]
pub struct Constraint {
    pub name: String,
    pub value: f32,
}