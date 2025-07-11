//! Temporal layers for sequence processing: LSTM, GRU, and Multi-Head Attention

use ndarray::Array2;
use crate::nn::{Layer, WeightInit, Gradient, ActivationFunction, Activation};
use crate::nn::tensor::{Tensor, TensorOps};
// use crate::nn::layers::weight_extraction::WeightExtraction;
use std::sync::Arc;
use parking_lot::RwLock;

/// LSTM layer for sequence processing with episodic memory capabilities
#[derive(Debug, Clone)]
pub struct LSTMLayer {
    // Input gate weights
    pub weight_ii: Array2<f32>,
    pub weight_hi: Array2<f32>,
    pub bias_i: Array2<f32>,
    
    // Forget gate weights
    pub weight_if: Array2<f32>,
    pub weight_hf: Array2<f32>,
    pub bias_f: Array2<f32>,
    
    // Cell gate weights
    pub weight_ig: Array2<f32>,
    pub weight_hg: Array2<f32>,
    pub bias_g: Array2<f32>,
    
    // Output gate weights
    pub weight_io: Array2<f32>,
    pub weight_ho: Array2<f32>,
    pub bias_o: Array2<f32>,
    
    hidden_size: usize,
    hidden_state: Arc<RwLock<Array2<f32>>>,
    cell_state: Arc<RwLock<Array2<f32>>>,
    gradient: Arc<RwLock<Gradient>>,
}

impl LSTMLayer {
    pub fn new(input_size: usize, hidden_size: usize, init: WeightInit) -> Self {
        // Initialize all gate weights
        let weight_ii = init.initialize(&[input_size, hidden_size]);
        let weight_hi = init.initialize(&[hidden_size, hidden_size]);
        let bias_i = Array2::zeros((1, hidden_size));
        
        let weight_if = init.initialize(&[input_size, hidden_size]);
        let weight_hf = init.initialize(&[hidden_size, hidden_size]);
        let bias_f = Array2::ones((1, hidden_size)); // Forget gate bias initialized to 1
        
        let weight_ig = init.initialize(&[input_size, hidden_size]);
        let weight_hg = init.initialize(&[hidden_size, hidden_size]);
        let bias_g = Array2::zeros((1, hidden_size));
        
        let weight_io = init.initialize(&[input_size, hidden_size]);
        let weight_ho = init.initialize(&[hidden_size, hidden_size]);
        let bias_o = Array2::zeros((1, hidden_size));
        
        Self {
            weight_ii, weight_hi, bias_i,
            weight_if, weight_hf, bias_f,
            weight_ig, weight_hg, bias_g,
            weight_io, weight_ho, bias_o,
            hidden_size,
            hidden_state: Arc::new(RwLock::new(Array2::zeros((1, hidden_size)))),
            cell_state: Arc::new(RwLock::new(Array2::zeros((1, hidden_size)))),
            gradient: Arc::new(RwLock::new(Gradient::new())),
        }
    }
    
    pub fn reset_state(&self, batch_size: usize) {
        *self.hidden_state.write() = Array2::zeros((batch_size, self.hidden_size));
        *self.cell_state.write() = Array2::zeros((batch_size, self.hidden_size));
    }
}

impl Layer for LSTMLayer {
    fn forward(&self, input: &Array2<f32>, _training: bool) -> Array2<f32> {
        let batch_size = input.shape()[0];
        let mut hidden = self.hidden_state.read().clone();
        let mut cell = self.cell_state.read().clone();
        
        // Ensure state dimensions match batch size
        if hidden.shape()[0] != batch_size {
            hidden = Array2::zeros((batch_size, self.hidden_size));
            cell = Array2::zeros((batch_size, self.hidden_size));
        }
        
        // Input gate
        let i_gate = {
            let xi = Tensor::matmul_simd(input, &self.weight_ii);
            let hi = Tensor::matmul_simd(&hidden, &self.weight_hi);
            let i_pre = Tensor::add_bias_simd(&(&xi + &hi), &self.bias_i);
            Activation::forward(&i_pre, ActivationFunction::Sigmoid)
        };
        
        // Forget gate
        let f_gate = {
            let xf = Tensor::matmul_simd(input, &self.weight_if);
            let hf = Tensor::matmul_simd(&hidden, &self.weight_hf);
            let f_pre = Tensor::add_bias_simd(&(&xf + &hf), &self.bias_f);
            Activation::forward(&f_pre, ActivationFunction::Sigmoid)
        };
        
        // Cell gate (candidate values)
        let g_gate = {
            let xg = Tensor::matmul_simd(input, &self.weight_ig);
            let hg = Tensor::matmul_simd(&hidden, &self.weight_hg);
            let g_pre = Tensor::add_bias_simd(&(&xg + &hg), &self.bias_g);
            Activation::forward(&g_pre, ActivationFunction::Tanh)
        };
        
        // Output gate
        let o_gate = {
            let xo = Tensor::matmul_simd(input, &self.weight_io);
            let ho = Tensor::matmul_simd(&hidden, &self.weight_ho);
            let o_pre = Tensor::add_bias_simd(&(&xo + &ho), &self.bias_o);
            Activation::forward(&o_pre, ActivationFunction::Sigmoid)
        };
        
        // Update cell state
        cell = &f_gate * &cell + &i_gate * &g_gate;
        
        // Update hidden state
        let cell_tanh = Activation::forward(&cell, ActivationFunction::Tanh);
        hidden = &o_gate * &cell_tanh;
        
        // Store states for next timestep
        *self.hidden_state.write() = hidden.clone();
        *self.cell_state.write() = cell;
        
        hidden
    }
    
    fn backward(&self, grad_output: &Array2<f32>, _input: &Array2<f32>) -> Array2<f32> {
        // LSTM backward pass is complex, storing simplified gradient for now
        let mut grad = self.gradient.write();
        grad.input = Some(grad_output.clone());
        
        // Return gradient w.r.t input (simplified)
        grad_output.clone()
    }
    
    fn update_weights(&mut self, _gradient: &Gradient, _learning_rate: f32) {
        // TODO: Implement proper LSTM weight updates
    }
    
    fn get_params(&self) -> Vec<&Array2<f32>> {
        vec![
            &self.weight_ii, &self.weight_hi, &self.bias_i,
            &self.weight_if, &self.weight_hf, &self.bias_f,
            &self.weight_ig, &self.weight_hg, &self.bias_g,
            &self.weight_io, &self.weight_ho, &self.bias_o,
        ]
    }
    
    fn get_params_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![
            &mut self.weight_ii, &mut self.weight_hi, &mut self.bias_i,
            &mut self.weight_if, &mut self.weight_hf, &mut self.bias_f,
            &mut self.weight_ig, &mut self.weight_hg, &mut self.bias_g,
            &mut self.weight_io, &mut self.weight_ho, &mut self.bias_o,
        ]
    }
    
    fn output_size(&self) -> Option<usize> {
        Some(self.hidden_size)
    }
}

/// GRU layer - simpler alternative to LSTM
#[derive(Debug, Clone)]
pub struct GRULayer {
    // Reset gate weights
    pub weight_ir: Array2<f32>,
    pub weight_hr: Array2<f32>,
    pub bias_r: Array2<f32>,
    
    // Update gate weights
    pub weight_iz: Array2<f32>,
    pub weight_hz: Array2<f32>,
    pub bias_z: Array2<f32>,
    
    // New gate weights
    pub weight_in: Array2<f32>,
    pub weight_hn: Array2<f32>,
    pub bias_n: Array2<f32>,
    
    hidden_size: usize,
    hidden_state: Arc<RwLock<Array2<f32>>>,
    gradient: Arc<RwLock<Gradient>>,
}

impl GRULayer {
    pub fn new(input_size: usize, hidden_size: usize, init: WeightInit) -> Self {
        let weight_ir = init.initialize(&[input_size, hidden_size]);
        let weight_hr = init.initialize(&[hidden_size, hidden_size]);
        let bias_r = Array2::zeros((1, hidden_size));
        
        let weight_iz = init.initialize(&[input_size, hidden_size]);
        let weight_hz = init.initialize(&[hidden_size, hidden_size]);
        let bias_z = Array2::zeros((1, hidden_size));
        
        let weight_in = init.initialize(&[input_size, hidden_size]);
        let weight_hn = init.initialize(&[hidden_size, hidden_size]);
        let bias_n = Array2::zeros((1, hidden_size));
        
        Self {
            weight_ir, weight_hr, bias_r,
            weight_iz, weight_hz, bias_z,
            weight_in, weight_hn, bias_n,
            hidden_size,
            hidden_state: Arc::new(RwLock::new(Array2::zeros((1, hidden_size)))),
            gradient: Arc::new(RwLock::new(Gradient::new())),
        }
    }
}

impl Layer for GRULayer {
    fn forward(&self, input: &Array2<f32>, _training: bool) -> Array2<f32> {
        let batch_size = input.shape()[0];
        let mut hidden = self.hidden_state.read().clone();
        
        if hidden.shape()[0] != batch_size {
            hidden = Array2::zeros((batch_size, self.hidden_size));
        }
        
        // Reset gate
        let r_gate = {
            let xr = Tensor::matmul_simd(input, &self.weight_ir);
            let hr = Tensor::matmul_simd(&hidden, &self.weight_hr);
            let r_pre = Tensor::add_bias_simd(&(&xr + &hr), &self.bias_r);
            Activation::forward(&r_pre, ActivationFunction::Sigmoid)
        };
        
        // Update gate
        let z_gate = {
            let xz = Tensor::matmul_simd(input, &self.weight_iz);
            let hz = Tensor::matmul_simd(&hidden, &self.weight_hz);
            let z_pre = Tensor::add_bias_simd(&(&xz + &hz), &self.bias_z);
            Activation::forward(&z_pre, ActivationFunction::Sigmoid)
        };
        
        // New gate
        let n_gate = {
            let xn = Tensor::matmul_simd(input, &self.weight_in);
            let hn = Tensor::matmul_simd(&(&r_gate * &hidden), &self.weight_hn);
            let n_pre = Tensor::add_bias_simd(&(&xn + &hn), &self.bias_n);
            Activation::forward(&n_pre, ActivationFunction::Tanh)
        };
        
        // Update hidden state
        hidden = &(&Array2::ones(z_gate.raw_dim()) - &z_gate) * &n_gate + &z_gate * &hidden;
        
        *self.hidden_state.write() = hidden.clone();
        hidden
    }
    
    fn backward(&self, grad_output: &Array2<f32>, _input: &Array2<f32>) -> Array2<f32> {
        // Simplified gradient storage
        let mut grad = self.gradient.write();
        grad.input = Some(grad_output.clone());
        grad_output.clone()
    }
    
    fn update_weights(&mut self, _gradient: &Gradient, _learning_rate: f32) {
        // TODO: Implement GRU weight updates
    }
    
    fn get_params(&self) -> Vec<&Array2<f32>> {
        vec![
            &self.weight_ir, &self.weight_hr, &self.bias_r,
            &self.weight_iz, &self.weight_hz, &self.bias_z,
            &self.weight_in, &self.weight_hn, &self.bias_n,
        ]
    }
    
    fn get_params_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![
            &mut self.weight_ir, &mut self.weight_hr, &mut self.bias_r,
            &mut self.weight_iz, &mut self.weight_hz, &mut self.bias_z,
            &mut self.weight_in, &mut self.weight_hn, &mut self.bias_n,
        ]
    }
    
    fn output_size(&self) -> Option<usize> {
        Some(self.hidden_size)
    }
}

/// Multi-Head Attention layer for episodic memory retrieval
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionLayer {
    pub weight_q: Array2<f32>,
    pub weight_k: Array2<f32>,
    pub weight_v: Array2<f32>,
    pub weight_o: Array2<f32>,
    
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    gradient: Arc<RwLock<Gradient>>,
}

impl MultiHeadAttentionLayer {
    pub fn new(embed_dim: usize, num_heads: usize, init: WeightInit) -> Self {
        assert!(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");
        let head_dim = embed_dim / num_heads;
        
        let weight_q = init.initialize(&[embed_dim, embed_dim]);
        let weight_k = init.initialize(&[embed_dim, embed_dim]);
        let weight_v = init.initialize(&[embed_dim, embed_dim]);
        let weight_o = init.initialize(&[embed_dim, embed_dim]);
        
        Self {
            weight_q, weight_k, weight_v, weight_o,
            embed_dim,
            num_heads,
            head_dim,
            scale: (head_dim as f32).sqrt(),
            gradient: Arc::new(RwLock::new(Gradient::new())),
        }
    }
}

impl Layer for MultiHeadAttentionLayer {
    fn forward(&self, input: &Array2<f32>, _training: bool) -> Array2<f32> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1] / self.embed_dim;
        
        // Linear projections
        let q = Tensor::matmul_simd(&input, &self.weight_q);
        let k = Tensor::matmul_simd(&input, &self.weight_k);
        let v = Tensor::matmul_simd(&input, &self.weight_v);
        
        // Compute attention scores
        // For simplicity, computing single-head attention and scaling
        let k_t = k.t().to_owned();
        let scores = Tensor::matmul_simd(&q, &k_t) / self.scale;
        
        // Softmax
        let attention_weights = Activation::forward(&scores, ActivationFunction::Softmax);
        
        // Apply attention to values
        let attended = Tensor::matmul_simd(&attention_weights, &v);
        
        // Output projection
        Tensor::matmul_simd(&attended, &self.weight_o)
    }
    
    fn backward(&self, grad_output: &Array2<f32>, _input: &Array2<f32>) -> Array2<f32> {
        // Simplified gradient storage
        let mut grad = self.gradient.write();
        grad.input = Some(grad_output.clone());
        grad_output.clone()
    }
    
    fn update_weights(&mut self, _gradient: &Gradient, _learning_rate: f32) {
        // TODO: Implement attention weight updates
    }
    
    fn get_params(&self) -> Vec<&Array2<f32>> {
        vec![&self.weight_q, &self.weight_k, &self.weight_v, &self.weight_o]
    }
    
    fn get_params_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![&mut self.weight_q, &mut self.weight_k, &mut self.weight_v, &mut self.weight_o]
    }
    
    fn output_size(&self) -> Option<usize> {
        Some(self.embed_dim)
    }
}