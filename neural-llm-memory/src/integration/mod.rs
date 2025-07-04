//! Integration layer between neural patterns and neural memory systems

pub mod pattern_storage;
pub mod pattern_retrieval;
pub mod learning_sync;
pub mod adaptive_system;

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Represents a learned pattern from the neural pattern system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedPattern {
    pub id: String,
    pub pattern_type: PatternType,
    pub context: PatternContext,
    pub performance: PerformanceMetrics,
    pub neural_weights: Vec<f32>,
    pub timestamp: DateTime<Utc>,
    pub confidence: f32,
}

/// Types of cognitive patterns
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PatternType {
    Convergent,  // Focused problem-solving
    Divergent,   // Creative exploration
    Lateral,     // Alternative approaches
    Systems,     // Holistic thinking
    Critical,    // Analytical evaluation
    Abstract,    // High-level design
    Hybrid(Box<PatternType>, Box<PatternType>), // Combination patterns
}

/// Context in which a pattern was learned
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternContext {
    pub task_type: String,
    pub file_types: Vec<String>,
    pub complexity: ComplexityLevel,
    pub domain: String,
    pub success_criteria: Vec<String>,
    pub environmental_factors: HashMap<String, String>,
}

/// Complexity levels for pattern categorization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum ComplexityLevel {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

/// Performance metrics for a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub success_rate: f32,
    pub average_time_ms: f64,
    pub token_efficiency: f32,
    pub error_rate: f32,
    pub adaptability_score: f32,
    pub usage_count: u32,
}

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    pub auto_store_threshold: f32,  // Confidence threshold for auto-storing patterns
    pub pattern_decay_rate: f32,    // How quickly unused patterns lose importance
    pub max_pattern_age_days: u32,  // Maximum age before pattern review
    pub similarity_threshold: f32,   // Threshold for pattern matching
    pub learning_rate: f32,         // Rate of cross-system learning
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            auto_store_threshold: 0.85,
            pattern_decay_rate: 0.95,
            max_pattern_age_days: 90,
            similarity_threshold: 0.75,
            learning_rate: 0.1,
        }
    }
}

/// Result of pattern matching
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern: LearnedPattern,
    pub similarity: f32,
    pub relevance_score: f32,
    pub suggested_adaptations: Vec<PatternAdaptation>,
}

/// Suggested adaptations for a pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAdaptation {
    pub adaptation_type: AdaptationType,
    pub description: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationType {
    ParameterAdjustment,
    StrategyModification,
    ContextualTweak,
    HybridApproach,
}

/// Trait for pattern learning systems
pub trait PatternLearner {
    fn learn_from_success(&mut self, context: &PatternContext, metrics: &PerformanceMetrics);
    fn learn_from_failure(&mut self, context: &PatternContext, error: &str);
    fn suggest_pattern(&self, context: &PatternContext) -> Option<PatternMatch>;
    fn update_pattern(&mut self, pattern_id: &str, metrics: &PerformanceMetrics);
}

/// Trait for memory systems that can store patterns
pub trait PatternMemory {
    fn store_pattern(&mut self, pattern: LearnedPattern) -> Result<String, Box<dyn std::error::Error>>;
    fn retrieve_pattern(&self, pattern_id: &str) -> Option<LearnedPattern>;
    fn search_patterns(&self, context: &PatternContext, limit: usize) -> Vec<PatternMatch>;
    fn update_pattern_metrics(&mut self, pattern_id: &str, metrics: &PerformanceMetrics);
    fn prune_old_patterns(&mut self, max_age_days: u32) -> usize;
}