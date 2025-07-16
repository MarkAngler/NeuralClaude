//! Neural Network Optimization Module
//! 
//! High-performance optimizations for neural network operations including
//! SIMD acceleration, caching, and specialized algorithms.

pub mod consciousness_cache;
pub mod simd_attention;

pub use consciousness_cache::{ConsciousnessCache, ConsciousnessContext, CacheConfig, CacheMetrics};
pub use simd_attention::{SimdAttention, SimdAttentionConfig};

// Re-export commonly used items
pub mod prelude {
    pub use super::consciousness_cache::{ConsciousnessCache, ConsciousnessContext, CacheConfig};
    pub use super::simd_attention::{SimdAttention, SimdAttentionConfig};
}