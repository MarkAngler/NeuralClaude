//! Feedback collection for adaptive learning

use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    /// Explicit positive feedback
    Success,
    /// Explicit negative feedback  
    Failure,
    /// Implicit positive (e.g., result was used)
    ImplicitPositive,
    /// Implicit negative (e.g., search repeated)
    ImplicitNegative,
    /// Neutral or unknown
    Neutral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationFeedback {
    /// ID of the operation this feedback relates to
    pub operation_id: String,
    /// Type of feedback
    pub feedback_type: FeedbackType,
    /// Optional score (0.0 to 1.0)
    pub score: Option<f32>,
    /// Optional context or reason
    pub context: Option<String>,
    /// How the result was used (Claude-specific)
    pub usage_context: Option<String>,
    /// When feedback was received
    pub timestamp: DateTime<Utc>,
}

impl OperationFeedback {
    pub fn success(operation_id: String) -> Self {
        Self {
            operation_id,
            feedback_type: FeedbackType::Success,
            score: Some(1.0),
            context: None,
            usage_context: None,
            timestamp: Utc::now(),
        }
    }
    
    pub fn failure(operation_id: String, reason: Option<String>) -> Self {
        Self {
            operation_id,
            feedback_type: FeedbackType::Failure,
            score: Some(0.0),
            context: reason,
            usage_context: None,
            timestamp: Utc::now(),
        }
    }
    
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = Some(score.clamp(0.0, 1.0));
        self
    }
    
    pub fn with_usage_context(mut self, usage_context: String) -> Self {
        self.usage_context = Some(usage_context);
        self
    }
    
    /// Create feedback from Claude's tool input
    pub fn from_claude(
        operation_id: String,
        success: bool,
        score: Option<f32>,
        reason: Option<String>,
        usage_context: Option<String>,
    ) -> Self {
        Self {
            operation_id,
            feedback_type: if success { FeedbackType::Success } else { FeedbackType::Failure },
            score: score.or(Some(if success { 1.0 } else { 0.0 })),
            context: reason,
            usage_context,
            timestamp: Utc::now(),
        }
    }
}

/// Feedback-aware fitness calculation
pub fn calculate_fitness_with_feedback(
    response_time_ms: f32,
    memory_usage_bytes: i64,
    feedback_score: Option<f32>,
) -> f32 {
    // Base fitness from performance metrics
    let speed_fitness = 1.0 / (1.0 + response_time_ms / 10.0);
    let memory_fitness = 1.0 / (1.0 + memory_usage_bytes as f32 / 1_000_000.0);
    
    // If we have feedback, weight it heavily
    if let Some(score) = feedback_score {
        // 70% feedback, 15% speed, 15% memory (as per FEEDBACK_INTEGRATION.md)
        0.7 * score + 0.15 * speed_fitness + 0.15 * memory_fitness
    } else {
        // No feedback: 50% speed, 50% memory
        0.5 * speed_fitness + 0.5 * memory_fitness
    }
}