//! Main memory module integrating neural networks with memory operations

use crate::memory::{MemoryBank, MemoryKey, MemoryValue, MemoryOperations};
use crate::nn::{NeuralNetwork, NetworkBuilder, ActivationFunction};
use crate::attention::MultiHeadAttention;
use ndarray::{Array2, Axis};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use parking_lot::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub memory_size: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dropout_rate: f32,
    pub max_sequence_length: usize,
    pub use_positional_encoding: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            memory_size: 10000,
            embedding_dim: 768,
            hidden_dim: 2048,
            num_heads: 12,
            num_layers: 6,
            dropout_rate: 0.1,
            max_sequence_length: 512,
            use_positional_encoding: true,
        }
    }
}

/// Main memory module combining neural processing with memory storage
pub struct MemoryModule {
    config: MemoryConfig,
    memory_bank: Arc<RwLock<MemoryBank>>,
    encoder: NeuralNetwork,
    decoder: NeuralNetwork,
    attention: MultiHeadAttention,
    positional_encoding: Option<Array2<f32>>,
}

impl MemoryModule {
    pub fn new(config: MemoryConfig) -> Self {
        // Initialize memory bank
        let memory_bank = Arc::new(RwLock::new(
            MemoryBank::new(config.memory_size, config.memory_size / 10)
        ));
        
        // Build encoder network
        let encoder = NetworkBuilder::new()
            .add_linear(config.embedding_dim, config.hidden_dim, ActivationFunction::GELU, true)
            .add_layer_norm(config.hidden_dim)
            .add_dropout(config.dropout_rate)
            .add_linear(config.hidden_dim, config.hidden_dim, ActivationFunction::GELU, true)
            .add_layer_norm(config.hidden_dim)
            .add_dropout(config.dropout_rate)
            .add_linear(config.hidden_dim, config.embedding_dim, ActivationFunction::Identity, true)
            .build(0.001);
        
        // Build decoder network
        let decoder = NetworkBuilder::new()
            .add_linear(config.embedding_dim * 2, config.hidden_dim, ActivationFunction::GELU, true)
            .add_layer_norm(config.hidden_dim)
            .add_dropout(config.dropout_rate)
            .add_linear(config.hidden_dim, config.hidden_dim, ActivationFunction::GELU, true)
            .add_layer_norm(config.hidden_dim)
            .add_dropout(config.dropout_rate)
            .add_linear(config.hidden_dim, config.embedding_dim, ActivationFunction::Identity, true)
            .build(0.001);
        
        // Initialize attention mechanism
        let attention = MultiHeadAttention::new(
            config.embedding_dim,
            config.num_heads,
            config.dropout_rate,
        );
        
        // Initialize positional encoding if enabled
        let positional_encoding = if config.use_positional_encoding {
            Some(Self::create_positional_encoding(
                config.max_sequence_length,
                config.embedding_dim,
            ))
        } else {
            None
        };
        
        Self {
            config,
            memory_bank,
            encoder,
            decoder,
            attention,
            positional_encoding,
        }
    }
    
    /// Create sinusoidal positional encoding
    fn create_positional_encoding(max_len: usize, d_model: usize) -> Array2<f32> {
        let mut encoding = Array2::zeros((max_len, d_model));
        
        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f32 / 10000_f32.powf((2 * (i / 2)) as f32 / d_model as f32);
                
                if i % 2 == 0 {
                    encoding[[pos, i]] = angle.sin();
                } else {
                    encoding[[pos, i]] = angle.cos();
                }
            }
        }
        
        encoding
    }
    
    /// Encode input into memory representation
    pub fn encode(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut encoded = input.clone();
        
        // Add positional encoding if available
        if let Some(ref pos_enc) = self.positional_encoding {
            let seq_len = encoded.shape()[0].min(pos_enc.shape()[0]);
            for i in 0..seq_len {
                encoded.row_mut(i).scaled_add(1.0, &pos_enc.row(i));
            }
        }
        
        // Process through encoder
        self.encoder.forward(&encoded, false)
    }
    
    /// Store a new memory
    pub fn store_memory(&mut self, content: String, embedding: Array2<f32>) -> crate::Result<MemoryKey> {
        // Encode the embedding
        let encoded = self.encode(&embedding);
        
        // Create memory key
        let key = MemoryKey::new(
            uuid::Uuid::new_v4().to_string(),
            &content,
        );
        
        // Create memory value
        let value = MemoryValue {
            embedding: encoded.as_slice().unwrap().to_vec(),
            content,
            metadata: Default::default(),
        };
        
        // Store in memory bank
        self.memory_bank.write().store(key.clone(), value)?;
        
        Ok(key)
    }
    
    /// Retrieve memories using attention mechanism
    pub fn retrieve_with_attention(
        &self,
        query: &Array2<f32>,
        k: usize,
    ) -> Vec<(MemoryKey, MemoryValue, Array2<f32>)> {
        // Encode query
        let encoded_query = self.encode(query);
        
        // Search for similar memories
        let results = self.memory_bank.read().search(&encoded_query, k * 2);
        
        if results.is_empty() {
            return Vec::new();
        }
        
        // Prepare memory embeddings for attention
        let memory_embeddings: Vec<Array2<f32>> = results
            .iter()
            .map(|(_, value, _)| {
                Array2::from_shape_vec((1, value.embedding.len()), value.embedding.clone())
                    .unwrap()
            })
            .collect();
        
        // Stack embeddings
        let stacked = ndarray::stack(
            Axis(0),
            &memory_embeddings.iter().map(|a| a.view()).collect::<Vec<_>>(),
        ).unwrap();
        
        // Apply attention to find most relevant memories
        let attention_weights = self.attention.compute_attention(&encoded_query, &stacked);
        
        // Combine results with attention weights
        let mut weighted_results: Vec<_> = results
            .into_iter()
            .enumerate()
            .map(|(i, (key, value, similarity))| {
                let attention_score = attention_weights[[0, i]];
                let combined_score = similarity * 0.5 + attention_score * 0.5;
                (key, value, Array2::from_elem((1, 1), combined_score))
            })
            .collect();
        
        // Sort by combined score
        weighted_results.sort_by(|a, b| {
            b.2[[0, 0]].partial_cmp(&a.2[[0, 0]]).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Return top k
        weighted_results.truncate(k);
        weighted_results
    }
    
    /// Update memory with new information
    pub fn update_memory(
        &mut self,
        key: &MemoryKey,
        new_content: String,
        new_embedding: Array2<f32>,
    ) -> crate::Result<()> {
        let encoded = self.encode(&new_embedding);
        
        self.memory_bank.write().update(key, |value| {
            value.content = new_content;
            value.embedding = encoded.as_slice().unwrap().to_vec();
            value.metadata.importance *= 1.1; // Boost importance on update
        })
    }
    
    /// Consolidate memories using neural processing
    pub fn consolidate_memories(&mut self, keys: &[MemoryKey]) -> crate::Result<MemoryKey> {
        let mut memory_bank = self.memory_bank.write();
        
        // Retrieve all memories
        let memories: Vec<_> = keys
            .iter()
            .filter_map(|key| memory_bank.retrieve(key).ok().flatten())
            .collect();
        
        if memories.is_empty() {
            return Err(crate::MemoryFrameworkError {
                message: "No memories found to consolidate".to_string(),
            });
        }
        
        // Average embeddings
        let embedding_sum = memories.iter().fold(
            Array2::zeros((1, self.config.embedding_dim)),
            |acc, mem| {
                acc + Array2::from_shape_vec((1, mem.embedding.len()), mem.embedding.clone())
                    .unwrap()
            },
        );
        
        let avg_embedding = embedding_sum / memories.len() as f32;
        
        // Concatenate content
        let consolidated_content = memories
            .iter()
            .map(|m| &m.content)
            .cloned()
            .collect::<Vec<_>>()
            .join(" | ");
        
        // Create new consolidated memory
        let key = MemoryKey::new(
            format!("consolidated_{}", uuid::Uuid::new_v4()),
            &consolidated_content,
        );
        
        let value = MemoryValue {
            embedding: avg_embedding.as_slice().unwrap().to_vec(),
            content: consolidated_content,
            metadata: Default::default(),
        };
        
        memory_bank.store(key.clone(), value)?;
        
        Ok(key)
    }
    
    /// Get memory statistics
    pub fn get_stats(&self) -> (usize, u64, u64, f32) {
        let memory_bank = self.memory_bank.read();
        let size = memory_bank.size();
        let (total_accesses, cache_hits, hit_rate) = memory_bank.get_statistics();
        
        (size, total_accesses, cache_hits, hit_rate)
    }
    
    /// Get the configuration
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }
    
    /// Get the memory bank (read-only access)
    pub fn memory_bank(&self) -> Arc<RwLock<MemoryBank>> {
        self.memory_bank.clone()
    }
    
    /// Get all memories for persistence
    pub fn get_all_memories(&self) -> Vec<(MemoryKey, MemoryValue)> {
        let bank = self.memory_bank.read();
        bank.get_all_memories()
    }
}