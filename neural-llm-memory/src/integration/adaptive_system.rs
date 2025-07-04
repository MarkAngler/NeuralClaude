//! Adaptive system that combines neural patterns with memory for continuous improvement

use crate::integration::{
    LearnedPattern, PatternType, PatternContext, PerformanceMetrics,
    PatternLearner, PatternMemory, PatternMatch, IntegrationConfig,
    ComplexityLevel, PatternAdaptation, AdaptationType,
};
use crate::memory::MemoryModule;
use crate::nn::{NeuralNetwork, NetworkBuilder, ActivationFunction};
use std::collections::HashMap;
use chrono::Utc;
use ndarray::Array2;

/// Adaptive neural system that learns and remembers patterns
pub struct AdaptiveNeuralSystem<M: PatternMemory> {
    pattern_memory: M,
    pattern_networks: HashMap<PatternType, NeuralNetwork>,
    config: IntegrationConfig,
    learning_history: Vec<LearningEvent>,
    current_context: Option<PatternContext>,
}

#[derive(Debug, Clone)]
struct LearningEvent {
    timestamp: chrono::DateTime<Utc>,
    pattern_type: PatternType,
    success: bool,
    context: PatternContext,
    metrics: Option<PerformanceMetrics>,
}

impl<M: PatternMemory> AdaptiveNeuralSystem<M> {
    pub fn new(pattern_memory: M, config: IntegrationConfig) -> Self {
        let mut pattern_networks = HashMap::new();
        
        // Initialize neural networks for each pattern type
        let pattern_types = vec![
            PatternType::Convergent,
            PatternType::Divergent,
            PatternType::Lateral,
            PatternType::Systems,
            PatternType::Critical,
            PatternType::Abstract,
        ];
        
        for pattern_type in pattern_types {
            let network = NetworkBuilder::new()
                .add_linear(768, 512, ActivationFunction::GELU, true)
                .add_dropout(0.1)
                .add_linear(512, 256, ActivationFunction::GELU, true)
                .add_layer_norm(256)
                .add_linear(256, 128, ActivationFunction::ReLU, true)
                .add_linear(128, 1, ActivationFunction::Sigmoid, true)
                .build(config.learning_rate);
            
            pattern_networks.insert(pattern_type, network);
        }
        
        Self {
            pattern_memory,
            pattern_networks,
            config,
            learning_history: Vec::new(),
            current_context: None,
        }
    }
    
    /// Set the current working context
    pub fn set_context(&mut self, context: PatternContext) {
        self.current_context = Some(context);
    }
    
    /// Get recommended pattern for current context
    pub fn recommend_pattern(&self) -> Option<(PatternType, f32)> {
        let context = self.current_context.as_ref()?;
        
        // Search for similar patterns in memory
        let similar_patterns = self.pattern_memory.search_patterns(context, 10);
        
        if !similar_patterns.is_empty() {
            // Use historical data to make recommendation
            let mut pattern_scores: HashMap<PatternType, f32> = HashMap::new();
            
            for pattern_match in &similar_patterns {
                let score = pattern_match.similarity * pattern_match.relevance_score 
                    * pattern_match.pattern.performance.success_rate;
                
                pattern_scores.entry(pattern_match.pattern.pattern_type.clone())
                    .and_modify(|s| *s += score)
                    .or_insert(score);
            }
            
            // Find best pattern
            pattern_scores.into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(pattern_type, score)| (pattern_type, score))
        } else {
            // No historical data, use neural networks to predict
            self.predict_best_pattern(context)
        }
    }
    
    /// Use neural networks to predict best pattern
    fn predict_best_pattern(&self, context: &PatternContext) -> Option<(PatternType, f32)> {
        let context_embedding = self.context_to_features(context);
        
        let mut best_pattern = None;
        let mut best_score = 0.0;
        
        for (pattern_type, network) in &self.pattern_networks {
            let score = network.predict(&context_embedding)[[0, 0]];
            if score > best_score {
                best_score = score;
                best_pattern = Some((pattern_type.clone(), score));
            }
        }
        
        best_pattern
    }
    
    /// Convert context to feature vector
    fn context_to_features(&self, context: &PatternContext) -> Array2<f32> {
        let mut features = vec![0.0f32; 768];
        
        // Encode task type
        let task_hash = self.hash_string(&context.task_type);
        for i in 0..32 {
            features[i] = ((task_hash >> i) & 1) as f32;
        }
        
        // Encode complexity
        features[32] = match context.complexity {
            ComplexityLevel::Simple => 0.25,
            ComplexityLevel::Moderate => 0.5,
            ComplexityLevel::Complex => 0.75,
            ComplexityLevel::VeryComplex => 1.0,
        };
        
        // Encode domain
        let domain_hash = self.hash_string(&context.domain);
        for i in 0..32 {
            features[64 + i] = ((domain_hash >> i) & 1) as f32;
        }
        
        // Encode file types
        for (i, file_type) in context.file_types.iter().take(16).enumerate() {
            let type_hash = self.hash_string(file_type);
            features[128 + i * 4] = ((type_hash >> 0) & 0xFF) as f32 / 255.0;
            features[128 + i * 4 + 1] = ((type_hash >> 8) & 0xFF) as f32 / 255.0;
            features[128 + i * 4 + 2] = ((type_hash >> 16) & 0xFF) as f32 / 255.0;
            features[128 + i * 4 + 3] = ((type_hash >> 24) & 0xFF) as f32 / 255.0;
        }
        
        Array2::from_shape_vec((1, 768), features).unwrap()
    }
    
    fn hash_string(&self, s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Generate pattern adaptations based on context differences
    fn generate_adaptations(
        &self,
        pattern: &LearnedPattern,
        current_context: &PatternContext,
    ) -> Vec<PatternAdaptation> {
        let mut adaptations = Vec::new();
        
        // Check complexity difference
        if pattern.context.complexity != current_context.complexity {
            adaptations.push(PatternAdaptation {
                adaptation_type: AdaptationType::ParameterAdjustment,
                description: format!(
                    "Adjust complexity from {:?} to {:?}",
                    pattern.context.complexity,
                    current_context.complexity
                ),
                confidence: 0.8,
            });
        }
        
        // Check domain difference
        if pattern.context.domain != current_context.domain {
            adaptations.push(PatternAdaptation {
                adaptation_type: AdaptationType::ContextualTweak,
                description: format!(
                    "Adapt from {} domain to {} domain",
                    pattern.context.domain,
                    current_context.domain
                ),
                confidence: 0.7,
            });
        }
        
        // Check file type differences
        let new_file_types: Vec<_> = current_context.file_types.iter()
            .filter(|ft| !pattern.context.file_types.contains(ft))
            .collect();
        
        if !new_file_types.is_empty() {
            adaptations.push(PatternAdaptation {
                adaptation_type: AdaptationType::StrategyModification,
                description: format!(
                    "Extend pattern to handle: {}",
                    new_file_types.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                ),
                confidence: 0.6,
            });
        }
        
        adaptations
    }
}

impl<M: PatternMemory> PatternLearner for AdaptiveNeuralSystem<M> {
    fn learn_from_success(&mut self, context: &PatternContext, metrics: &PerformanceMetrics) {
        // Record learning event
        let event = LearningEvent {
            timestamp: Utc::now(),
            pattern_type: self.recommend_pattern()
                .map(|(pt, _)| pt)
                .unwrap_or(PatternType::Convergent),
            success: true,
            context: context.clone(),
            metrics: Some(metrics.clone()),
        };
        
        self.learning_history.push(event.clone());
        
        // Create and store learned pattern if confidence is high enough
        if metrics.success_rate >= self.config.auto_store_threshold {
            let pattern = LearnedPattern {
                id: uuid::Uuid::new_v4().to_string(),
                pattern_type: event.pattern_type.clone(),
                context: context.clone(),
                performance: metrics.clone(),
                neural_weights: vec![], // TODO: Extract from neural network
                timestamp: Utc::now(),
                confidence: metrics.success_rate,
            };
            
            let _ = self.pattern_memory.store_pattern(pattern);
        }
        
        // Update neural network
        let features = self.context_to_features(context);
        let target = Array2::from_elem((1, 1), 1.0); // Success = 1.0
        
        if let Some(network) = self.pattern_networks.get_mut(&event.pattern_type) {
            // Simple training step
            let _ = network.train_step(&features, &target, |pred, tgt| {
                let diff = pred - tgt;
                let loss = (&diff * &diff).mean().unwrap();
                (loss, diff * 2.0 / pred.len() as f32)
            });
        }
    }
    
    fn learn_from_failure(&mut self, context: &PatternContext, _error: &str) {
        let event = LearningEvent {
            timestamp: Utc::now(),
            pattern_type: self.recommend_pattern()
                .map(|(pt, _)| pt)
                .unwrap_or(PatternType::Convergent),
            success: false,
            context: context.clone(),
            metrics: None,
        };
        
        self.learning_history.push(event.clone());
        
        // Update neural network with negative example
        let features = self.context_to_features(context);
        let target = Array2::from_elem((1, 1), 0.0); // Failure = 0.0
        
        if let Some(network) = self.pattern_networks.get_mut(&event.pattern_type) {
            let _ = network.train_step(&features, &target, |pred, tgt| {
                let diff = pred - tgt;
                let loss = (&diff * &diff).mean().unwrap();
                (loss, diff * 2.0 / pred.len() as f32)
            });
        }
    }
    
    fn suggest_pattern(&self, context: &PatternContext) -> Option<PatternMatch> {
        // Search for similar patterns
        let matches = self.pattern_memory.search_patterns(context, 5);
        
        // Find best match with adaptations
        matches.into_iter()
            .map(|mut pattern_match| {
                // Add adaptation suggestions
                pattern_match.suggested_adaptations = self.generate_adaptations(
                    &pattern_match.pattern,
                    context
                );
                pattern_match
            })
            .max_by(|a, b| {
                let score_a = a.similarity * 0.6 + a.relevance_score * 0.4;
                let score_b = b.similarity * 0.6 + b.relevance_score * 0.4;
                score_a.partial_cmp(&score_b).unwrap()
            })
    }
    
    fn update_pattern(&mut self, pattern_id: &str, metrics: &PerformanceMetrics) {
        self.pattern_memory.update_pattern_metrics(pattern_id, metrics);
    }
}

/// Builder for creating adaptive systems
pub struct AdaptiveSystemBuilder {
    config: IntegrationConfig,
    memory_config: crate::MemoryConfig,
}

impl AdaptiveSystemBuilder {
    pub fn new() -> Self {
        Self {
            config: IntegrationConfig::default(),
            memory_config: crate::MemoryConfig::default(),
        }
    }
    
    pub fn with_config(mut self, config: IntegrationConfig) -> Self {
        self.config = config;
        self
    }
    
    pub fn with_memory_config(mut self, config: crate::MemoryConfig) -> Self {
        self.memory_config = config;
        self
    }
    
    pub fn build(self) -> AdaptiveNeuralSystem<crate::integration::pattern_storage::PatternStorage> {
        let memory_module = MemoryModule::new(self.memory_config);
        let pattern_storage = crate::integration::pattern_storage::PatternStorage::new(memory_module);
        
        AdaptiveNeuralSystem::new(pattern_storage, self.config)
    }
}