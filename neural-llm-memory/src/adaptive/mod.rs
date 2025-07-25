pub mod usage_collector;
pub mod adaptive_memory;
pub mod background_evolver;
pub mod adaptive_config;
pub mod feedback;
pub mod adaptive_memory_ext;
pub mod enhanced_search;

// Re-export commonly used types
pub use self::usage_collector::{UsageCollector, UsageMetrics, OperationType, TrainingCorpus};
pub use self::adaptive_memory::AdaptiveMemoryModule;
pub use self::background_evolver::{BackgroundEvolver, EvolutionStatus, EvolvedArchitecture};
pub use self::adaptive_config::AdaptiveConfig;
pub use self::enhanced_search::{enhanced_search, MatchType, EnhancedSearchResult, MatchResult};

