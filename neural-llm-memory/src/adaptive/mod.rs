pub mod usage_collector;
pub mod adaptive_memory;
pub mod background_evolver;
pub mod adaptive_config;
pub mod feedback;
pub mod adaptive_memory_ext;
pub mod enhanced_search;

#[cfg(test)]
mod config_test;

#[cfg(test)]
mod shutdown_test;

pub use usage_collector::{UsageCollector, UsageMetrics, OperationType, TrainingCorpus};
pub use adaptive_memory::AdaptiveMemoryModule;
pub use background_evolver::{BackgroundEvolver, EvolutionStatus, EvolvedArchitecture};
pub use adaptive_config::AdaptiveConfig;
pub use feedback::{OperationFeedback, FeedbackType, calculate_fitness_with_feedback};
pub use adaptive_memory_ext::{AdaptiveModuleState, start_auto_save_task};
pub use enhanced_search::{enhanced_search, EnhancedSearchResult, MatchResult, MatchType};