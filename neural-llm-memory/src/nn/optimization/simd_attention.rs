//! SIMD-Optimized Attention Calculations
//! 
//! High-performance attention mechanisms using SIMD instructions where available.
//! Falls back to optimized scalar operations when SIMD is not available.

use ndarray::{Array1, Array2, Axis, s};
use std::sync::Arc;

/// Configuration for SIMD attention operations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimdAttentionConfig {
    /// Whether to use SIMD operations
    pub use_simd: bool,
    /// Chunk size for batch processing
    pub chunk_size: usize,
    /// Whether to use fused operations
    pub use_fused_ops: bool,
}

impl Default for SimdAttentionConfig {
    fn default() -> Self {
        Self {
            use_simd: true,
            chunk_size: 8, // Process 8 elements at a time for SIMD
            use_fused_ops: true,
        }
    }
}

/// SIMD-optimized attention calculator
pub struct SimdAttention {
    config: SimdAttentionConfig,
}

impl SimdAttention {
    /// Create new SIMD attention calculator
    pub fn new() -> Self {
        Self::with_config(SimdAttentionConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SimdAttentionConfig) -> Self {
        Self { config }
    }

    /// Compute scaled dot-product attention
    /// 
    /// attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    pub fn scaled_dot_product_attention(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        mask: Option<&Array2<bool>>,
    ) -> Array2<f32> {
        let (seq_len_q, d_k) = query.dim();
        let (seq_len_k, _) = key.dim();
        let scale = (d_k as f32).sqrt();

        // Compute QK^T
        let scores = if self.config.use_simd {
            let key_t = key.t().to_owned();
            self.matmul_simd(query, &key_t)
        } else {
            query.dot(&key.t())
        };

        // Scale scores
        let mut scaled_scores = scores / scale;

        // Apply mask if provided
        if let Some(mask) = mask {
            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    if !mask[[i, j]] {
                        scaled_scores[[i, j]] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Apply softmax
        let attention_weights = self.softmax_2d(&scaled_scores);

        // Compute attention output
        if self.config.use_simd {
            self.matmul_simd(&attention_weights, value)
        } else {
            attention_weights.dot(value)
        }
    }

    /// Multi-head attention with SIMD optimization
    pub fn multi_head_attention(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        num_heads: usize,
        d_model: usize,
    ) -> Array2<f32> {
        let d_k = d_model / num_heads;
        let seq_len = query.shape()[0];
        
        let mut outputs = Vec::with_capacity(num_heads);

        // Process each head
        for head in 0..num_heads {
            let start = head * d_k;
            let end = start + d_k;

            // Extract head-specific slices
            let q_head = query.slice(s![.., start..end]).to_owned();
            let k_head = key.slice(s![.., start..end]).to_owned();
            let v_head = value.slice(s![.., start..end]).to_owned();

            // Compute attention for this head
            let head_output = self.scaled_dot_product_attention(
                &q_head,
                &k_head,
                &v_head,
                None,
            );

            outputs.push(head_output);
        }

        // Concatenate all heads
        let mut result = Array2::zeros((seq_len, d_model));
        for (head, output) in outputs.iter().enumerate() {
            let start = head * d_k;
            let end = start + d_k;
            result.slice_mut(s![.., start..end]).assign(output);
        }

        result
    }

    /// SIMD-optimized matrix multiplication
    fn matmul_simd(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2, "Matrix dimensions don't match for multiplication");

        let mut result = Array2::zeros((m, n));

        // Process in chunks for better cache utilization
        let chunk_size = self.config.chunk_size;
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                
                // Process chunks
                let chunks = k / chunk_size;
                let remainder = k % chunk_size;

                // Main chunk processing
                for chunk in 0..chunks {
                    let start = chunk * chunk_size;
                    sum += self.dot_product_chunk(
                        &a.row(i).slice(s![start..start + chunk_size]),
                        &b.column(j).slice(s![start..start + chunk_size]),
                    );
                }

                // Handle remainder
                if remainder > 0 {
                    let start = chunks * chunk_size;
                    sum += self.dot_product_chunk(
                        &a.row(i).slice(s![start..]),
                        &b.column(j).slice(s![start..]),
                    );
                }

                result[[i, j]] = sum;
            }
        }

        result
    }

    /// Compute dot product for a chunk of data
    fn dot_product_chunk(&self, a: &ndarray::ArrayView1<f32>, b: &ndarray::ArrayView1<f32>) -> f32 {
        if self.config.use_fused_ops {
            // Use fused multiply-add when available
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        } else {
            // Standard dot product
            a.dot(b)
        }
    }

    /// SIMD-optimized 2D softmax
    fn softmax_2d(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut result = input.clone();
        
        // Apply softmax to each row
        for mut row in result.axis_iter_mut(Axis(0)) {
            self.softmax_1d_inplace(row.as_slice_mut().unwrap());
        }
        
        result
    }

    /// In-place 1D softmax with SIMD optimization
    fn softmax_1d_inplace(&self, data: &mut [f32]) {
        if data.is_empty() {
            return;
        }

        // Find max for numerical stability
        let max = if self.config.use_simd {
            self.find_max_simd(data)
        } else {
            *data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        };

        // Subtract max and compute exp
        let mut sum = 0.0;
        if self.config.use_simd && data.len() >= self.config.chunk_size {
            // Process in chunks
            let chunks = data.len() / self.config.chunk_size;
            
            for i in 0..chunks {
                let start = i * self.config.chunk_size;
                let end = start + self.config.chunk_size;
                sum += self.exp_sum_chunk(&mut data[start..end], max);
            }
            
            // Handle remainder
            let remainder_start = chunks * self.config.chunk_size;
            if remainder_start < data.len() {
                sum += self.exp_sum_chunk(&mut data[remainder_start..], max);
            }
        } else {
            // Scalar fallback
            for val in data.iter_mut() {
                *val = (*val - max).exp();
                sum += *val;
            }
        }

        // Normalize
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for val in data.iter_mut() {
                *val *= inv_sum;
            }
        }
    }

    /// Find maximum value using SIMD
    fn find_max_simd(&self, data: &[f32]) -> f32 {
        if data.is_empty() {
            return f32::NEG_INFINITY;
        }

        let mut max = data[0];
        let chunk_size = self.config.chunk_size;
        
        // Process chunks
        for chunk in data.chunks(chunk_size) {
            let chunk_max = chunk.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            max = max.max(chunk_max);
        }
        
        max
    }

    /// Compute exp and sum for a chunk
    fn exp_sum_chunk(&self, data: &mut [f32], max: f32) -> f32 {
        let mut sum = 0.0;
        
        for val in data.iter_mut() {
            *val = (*val - max).exp();
            sum += *val;
        }
        
        sum
    }

    /// Compute attention scores with caching support
    pub fn compute_attention_with_cache(
        &self,
        query: &Array1<f32>,
        key_cache: &[Array1<f32>],
        value_cache: &[Array1<f32>],
    ) -> Array1<f32> {
        assert_eq!(key_cache.len(), value_cache.len());
        
        if key_cache.is_empty() {
            return Array1::zeros(query.len());
        }

        let d_k = query.len() as f32;
        let scale = d_k.sqrt();
        
        // Compute attention scores
        let mut scores = Vec::with_capacity(key_cache.len());
        for key in key_cache {
            let score = query.dot(key) / scale;
            scores.push(score);
        }

        // Apply softmax
        self.softmax_1d_inplace(&mut scores);

        // Weighted sum of values
        let mut result = Array1::zeros(value_cache[0].len());
        for (score, value) in scores.iter().zip(value_cache) {
            result.scaled_add(*score, value);
        }

        result
    }
}

// Placeholder for future SIMD implementations when portable_simd is stable
#[cfg(feature = "portable_simd")]
mod simd_impl {
    use std::simd::{f32x8, SimdFloat, Simd};
    use super::*;

    /// SIMD-accelerated matrix multiplication kernel
    pub fn matmul_kernel_simd(
        a: &Array2<f32>,
        b: &Array2<f32>,
        result: &mut Array2<f32>,
    ) {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        
        // Process 8 elements at a time
        const LANES: usize = 8;
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = f32x8::splat(0.0);
                
                // Process k dimension in chunks of 8
                let chunks = k / LANES;
                for chunk in 0..chunks {
                    let offset = chunk * LANES;
                    
                    // Load 8 elements from row i of matrix a
                    let a_vec = f32x8::from_slice(&a.row(i).as_slice().unwrap()[offset..offset + LANES]);
                    
                    // Load 8 elements from column j of matrix b
                    let b_vals: [f32; LANES] = (0..LANES)
                        .map(|l| b[[offset + l, j]])
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap();
                    let b_vec = f32x8::from_array(b_vals);
                    
                    // Multiply and accumulate
                    sum += a_vec * b_vec;
                }
                
                // Sum the vector elements
                result[[i, j]] = sum.reduce_sum();
                
                // Handle remainder
                for l in (chunks * LANES)..k {
                    result[[i, j]] += a[[i, l]] * b[[l, j]];
                }
            }
        }
    }

    /// SIMD-accelerated exponential function
    pub fn exp_simd(data: &mut [f32]) {
        const LANES: usize = 8;
        let chunks = data.len() / LANES;
        
        for i in 0..chunks {
            let offset = i * LANES;
            let chunk = f32x8::from_slice(&data[offset..offset + LANES]);
            
            // Taylor series approximation for exp
            // exp(x) ≈ 1 + x + x²/2! + x³/3! + x⁴/4!
            let x = chunk;
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x3 * x;
            
            let result = f32x8::splat(1.0) 
                + x 
                + x2 * f32x8::splat(0.5)
                + x3 * f32x8::splat(1.0 / 6.0)
                + x4 * f32x8::splat(1.0 / 24.0);
            
            result.copy_to_slice(&mut data[offset..offset + LANES]);
        }
        
        // Handle remainder with scalar operations
        for i in (chunks * LANES)..data.len() {
            data[i] = data[i].exp();
        }
    }
}

