pub mod usage_collector;
pub mod adaptive_memory;
pub mod background_evolver;
pub mod adaptive_config;
pub mod feedback;

pub use usage_collector::{UsageCollector, UsageMetrics, OperationType, TrainingCorpus};
pub use adaptive_memory::AdaptiveMemoryModule;
pub use background_evolver::{BackgroundEvolver, EvolutionStatus, EvolvedArchitecture};
pub use adaptive_config::AdaptiveConfig;
pub use feedback::{OperationFeedback, FeedbackType, calculate_fitness_with_feedback};