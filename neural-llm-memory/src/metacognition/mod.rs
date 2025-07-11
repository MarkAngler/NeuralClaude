//! Metacognition module organization

pub mod monitor;
pub mod strategy;
// TODO: Implement these modules
// pub mod confidence;
// pub mod bias_detector;
// pub mod self_optimizer;
// pub mod introspection;

// Re-export main types (commented out until implemented)
// pub use crate::metacognition::{
//     MetaCognition,
//     MetaCognitiveMonitor,
//     ThinkingStrategy,
//     CognitivePattern,
//     ConfidenceEstimator,
//     CognitiveBias,
//     BiasType,
//     MetaCognitiveConfig,
//     ThinkingInput,
//     ThinkingOutput,
// };

pub use monitor::{CognitiveMonitor, AttentionMetrics, PatternMetrics};
pub use strategy::{StrategySelector, StrategyEvaluation};
// pub use confidence::{ConfidenceConfig, CalibrationMethod};
// pub use bias_detector::{BiasDetector, MitigationStrategy};
// pub use self_optimizer::{SelfOptimizer, ResourceAllocator, GoalHierarchyManager};
// pub use introspection::{IntrospectionAPI, CognitiveState, DecisionExplanation};