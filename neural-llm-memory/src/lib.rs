//! Neural LLM Memory Framework
//! 
//! A high-performance neural network framework designed specifically for
//! implementing memory mechanisms in Large Language Models using Rust.

// Removed portable_simd feature for stability

pub mod nn;
pub mod memory;
pub mod attention;
pub mod utils;
pub mod optimizers;
pub mod integration;
pub mod storage;
pub mod self_optimizing;
pub mod adaptive;
pub mod persistence;
pub mod consolidation;
pub mod metacognition;
pub mod continual_learning;
pub mod consciousness;
pub mod emotional;

pub use nn::{NeuralNetwork, Layer, Activation};
pub use memory::{MemoryBank, MemoryModule, MemoryConfig, PersistentMemoryModule, PersistentMemoryBuilder};
pub use attention::MultiHeadAttention;

use std::error::Error;
use std::fmt;

#[derive(Debug, Clone)]
pub struct MemoryFrameworkError {
    message: String,
}

impl fmt::Display for MemoryFrameworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Memory Framework Error: {}", self.message)
    }
}

impl Error for MemoryFrameworkError {}

pub type Result<T> = std::result::Result<T, MemoryFrameworkError>;

/// Configuration for the neural memory framework
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FrameworkConfig {
    pub memory_size: usize,
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub hidden_dim: usize,
    pub dropout_rate: f32,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub use_simd: bool,
    pub use_gpu: bool,
}

impl Default for FrameworkConfig {
    fn default() -> Self {
        Self {
            memory_size: 1024,
            embedding_dim: 768,
            num_heads: 12,
            hidden_dim: 3072,
            dropout_rate: 0.1,
            learning_rate: 0.001,
            batch_size: 32,
            use_simd: true,
            use_gpu: false,
        }
    }
}