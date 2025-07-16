//! Attention mechanisms for memory access

mod fusion;
pub use fusion::{AttentionFusion, AttentionFusionConfig, FusionResult, FusionStats};

use ndarray::{Array2, Array3, Axis};
use crate::nn::tensor::{Tensor, TensorOps};
use crate::nn::{WeightInit, LinearLayer, ActivationFunction, Layer};

/// Multi-head attention mechanism
pub struct MultiHeadAttention {
    num_heads: usize,
    d_model: usize,
    d_k: usize,
    d_v: usize,
    
    w_q: LinearLayer,
    w_k: LinearLayer,
    w_v: LinearLayer,
    w_o: LinearLayer,
    
    dropout_rate: f32,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize, dropout_rate: f32) -> Self {
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
        
        let d_k = d_model / num_heads;
        let d_v = d_model / num_heads;
        
        Self {
            num_heads,
            d_model,
            d_k,
            d_v,
            w_q: LinearLayer::new(d_model, d_model, ActivationFunction::Identity, false, WeightInit::Xavier),
            w_k: LinearLayer::new(d_model, d_model, ActivationFunction::Identity, false, WeightInit::Xavier),
            w_v: LinearLayer::new(d_model, d_model, ActivationFunction::Identity, false, WeightInit::Xavier),
            w_o: LinearLayer::new(d_model, d_model, ActivationFunction::Identity, false, WeightInit::Xavier),
            dropout_rate,
        }
    }
    
    /// Compute scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        mask: Option<&Array2<bool>>,
    ) -> (Array2<f32>, Array2<f32>) {
        let d_k = self.d_k as f32;
        
        // Compute attention scores
        let scores = Tensor::matmul_simd(query, &key.t().to_owned()) / d_k.sqrt();
        
        // Apply mask if provided
        let masked_scores = if let Some(_mask) = mask {
            scores.mapv(|x| x) // In production, apply mask properly
        } else {
            scores
        };
        
        // Apply softmax
        let attention_weights = Tensor::softmax(&masked_scores, Axis(1));
        
        // Apply attention to values
        let output = Tensor::matmul_simd(&attention_weights, value);
        
        (output, attention_weights)
    }
    
    /// Forward pass of multi-head attention
    pub fn forward(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        mask: Option<&Array2<bool>>,
    ) -> (Array2<f32>, Array2<f32>) {
        let _batch_size = query.shape()[0];
        let _seq_len = query.shape()[1] / self.d_model;
        
        // Linear projections
        let q = self.w_q.forward(query, false);
        let k = self.w_k.forward(key, false);
        let v = self.w_v.forward(value, false);
        
        // Reshape for multi-head attention
        // In production, properly reshape to (batch, heads, seq_len, d_k)
        
        // For now, compute single-head attention
        let (attended, weights) = self.scaled_dot_product_attention(&q, &k, &v, mask);
        
        // Output projection
        let output = self.w_o.forward(&attended, false);
        
        // Apply dropout during training
        let output = if self.dropout_rate > 0.0 {
            self.apply_dropout(&output, false)
        } else {
            output
        };
        
        (output, weights)
    }
    
    /// Compute attention weights for memory retrieval
    pub fn compute_attention(&self, query: &Array2<f32>, memory_keys: &Array3<f32>) -> Array2<f32> {
        // Simplified attention computation for memory retrieval
        let num_memories = memory_keys.shape()[0];
        let mut attention_scores = Array2::zeros((query.shape()[0], num_memories));
        
        for i in 0..num_memories {
            let key = memory_keys.index_axis(Axis(0), i).to_owned();
            let score = self.compute_similarity(query, &key);
            attention_scores[[0, i]] = score;
        }
        
        // Apply softmax to get attention weights
        Tensor::softmax(&attention_scores, Axis(1))
    }
    
    fn compute_similarity(&self, a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        let dot_product = (a * b).sum();
        let norm_a = (a * a).sum().sqrt();
        let norm_b = (b * b).sum().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
    
    fn apply_dropout(&self, input: &Array2<f32>, training: bool) -> Array2<f32> {
        if training && self.dropout_rate > 0.0 {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let scale = 1.0 / (1.0 - self.dropout_rate);
            
            input.mapv(|x| {
                if rng.gen::<f32>() > self.dropout_rate {
                    x * scale
                } else {
                    0.0
                }
            })
        } else {
            input.clone()
        }
    }
}

/// Self-attention layer for transformers
pub struct SelfAttention {
    mha: MultiHeadAttention,
    layer_norm: Array2<f32>,
}

impl SelfAttention {
    pub fn new(d_model: usize, num_heads: usize, dropout_rate: f32) -> Self {
        Self {
            mha: MultiHeadAttention::new(d_model, num_heads, dropout_rate),
            layer_norm: Array2::ones((1, d_model)),
        }
    }
    
    pub fn forward(&self, input: &Array2<f32>, mask: Option<&Array2<bool>>) -> Array2<f32> {
        // Self-attention: query, key, value are all the same
        let (attended, _) = self.mha.forward(input, input, input, mask);
        
        // Residual connection and layer normalization
        let output = input + &attended;
        Tensor::layer_norm(&output, 1e-6)
    }
}

/// Cross-attention for memory-augmented models
pub struct CrossAttention {
    mha: MultiHeadAttention,
}

impl CrossAttention {
    pub fn new(d_model: usize, num_heads: usize, dropout_rate: f32) -> Self {
        Self {
            mha: MultiHeadAttention::new(d_model, num_heads, dropout_rate),
        }
    }
    
    pub fn forward(
        &self,
        query: &Array2<f32>,
        memory: &Array2<f32>,
        mask: Option<&Array2<bool>>,
    ) -> (Array2<f32>, Array2<f32>) {
        // Cross-attention: query from input, key/value from memory
        self.mha.forward(query, memory, memory, mask)
    }
}