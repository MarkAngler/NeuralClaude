//! Tensor operations with optimization

use ndarray::{Array2, Axis};
use rayon::prelude::*;

pub struct Tensor;

pub trait TensorOps {
    fn matmul_simd(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;
    fn add_bias_simd(input: &Array2<f32>, bias: &Array2<f32>) -> Array2<f32>;
    fn relu_simd(input: &Array2<f32>) -> Array2<f32>;
    fn softmax(input: &Array2<f32>, axis: Axis) -> Array2<f32>;
    fn layer_norm(input: &Array2<f32>, eps: f32) -> Array2<f32>;
}

impl Tensor {
    pub fn cosine_similarity(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        let dot_product: f32 = (a * b).sum();
        let norm_a: f32 = (a * a).sum().sqrt();
        let norm_b: f32 = (b * b).sum().sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

impl TensorOps for Tensor {
    fn matmul_simd(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2, "Matrix dimensions don't match for multiplication");
        
        let mut result = Array2::zeros((m, n));
        
        // Use parallel iteration for large matrices
        if m * n > 10000 {
            result.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        
                        // Optimized dot product
                        for k_idx in 0..k {
                            sum += a[[i, k_idx]] * b[[k_idx, j]];
                        }
                        
                        row[j] = sum;
                    }
                });
        } else {
            // Sequential version for smaller matrices
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for k_idx in 0..k {
                        sum += a[[i, k_idx]] * b[[k_idx, j]];
                    }
                    result[[i, j]] = sum;
                }
            }
        }
        
        result
    }
    
    fn add_bias_simd(input: &Array2<f32>, bias: &Array2<f32>) -> Array2<f32> {
        // Broadcast bias to match input shape
        input + bias
    }
    
    fn relu_simd(input: &Array2<f32>) -> Array2<f32> {
        input.mapv(|x| x.max(0.0))
    }
    
    fn softmax(input: &Array2<f32>, axis: Axis) -> Array2<f32> {
        let mut result = input.clone();
        
        match axis {
            Axis(1) => {
                // Softmax over rows
                result.axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .for_each(|mut row| {
                        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        row.mapv_inplace(|x| (x - max_val).exp());
                        let sum: f32 = row.sum();
                        row.mapv_inplace(|x| x / sum);
                    });
            }
            Axis(0) => {
                // Softmax over columns
                for mut col in result.axis_iter_mut(Axis(1)) {
                    let max_val = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    col.mapv_inplace(|x| (x - max_val).exp());
                    let sum: f32 = col.sum();
                    col.mapv_inplace(|x| x / sum);
                }
            }
            _ => panic!("Softmax only supports axis 0 or 1"),
        }
        
        result
    }
    
    fn layer_norm(input: &Array2<f32>, eps: f32) -> Array2<f32> {
        let mut result = input.clone();
        
        result.axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                let mean = row.mean().unwrap();
                let variance = row.mapv(|x| (x - mean).powi(2)).mean().unwrap();
                let std_dev = (variance + eps).sqrt();
                
                row.mapv_inplace(|x| (x - mean) / std_dev);
            });
        
        result
    }
}