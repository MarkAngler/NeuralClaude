//! Neural network layer implementations

use ndarray::{Array2, Array3, Axis};
use crate::nn::{WeightInit, Gradient, ActivationFunction, Activation};
use crate::nn::tensor::{Tensor, TensorOps};
use crate::nn::layers::weight_extraction::WeightExtraction;
use crate::nn::layers::temporal::{LSTMLayer, GRULayer, MultiHeadAttentionLayer};
use std::sync::Arc;
use parking_lot::RwLock;

#[derive(Debug, Clone)]
pub enum LayerType {
    Linear(LinearLayer),
    Conv1D(Conv1DLayer),
    Dropout(DropoutLayer),
    LayerNorm(LayerNormLayer),
    Embedding(EmbeddingLayer),
    LSTM(LSTMLayer),
    GRU(GRULayer),
    MultiHeadAttention(MultiHeadAttentionLayer),
}

impl Layer for LayerType {
    fn forward(&self, input: &Array2<f32>, training: bool) -> Array2<f32> {
        match self {
            LayerType::Linear(layer) => layer.forward(input, training),
            LayerType::Conv1D(layer) => layer.forward(input, training),
            LayerType::Dropout(layer) => layer.forward(input, training),
            LayerType::LayerNorm(layer) => layer.forward(input, training),
            LayerType::Embedding(_) => panic!("Embedding layer forward not implemented for LayerType"),
            LayerType::LSTM(layer) => layer.forward(input, training),
            LayerType::GRU(layer) => layer.forward(input, training),
            LayerType::MultiHeadAttention(layer) => layer.forward(input, training),
        }
    }
    
    fn backward(&self, grad_output: &Array2<f32>, input: &Array2<f32>) -> Array2<f32> {
        match self {
            LayerType::Linear(layer) => layer.backward(grad_output, input),
            LayerType::Conv1D(layer) => layer.backward(grad_output, input),
            LayerType::Dropout(layer) => layer.backward(grad_output, input),
            LayerType::LayerNorm(layer) => layer.backward(grad_output, input),
            LayerType::Embedding(_) => panic!("Embedding layer backward not implemented for LayerType"),
            LayerType::LSTM(layer) => layer.backward(grad_output, input),
            LayerType::GRU(layer) => layer.backward(grad_output, input),
            LayerType::MultiHeadAttention(layer) => layer.backward(grad_output, input),
        }
    }
    
    fn update_weights(&mut self, gradient: &Gradient, learning_rate: f32) {
        match self {
            LayerType::Linear(layer) => layer.update_weights(gradient, learning_rate),
            LayerType::Conv1D(layer) => layer.update_weights(gradient, learning_rate),
            LayerType::Dropout(layer) => layer.update_weights(gradient, learning_rate),
            LayerType::LayerNorm(layer) => layer.update_weights(gradient, learning_rate),
            LayerType::Embedding(_) => {},
            LayerType::LSTM(layer) => layer.update_weights(gradient, learning_rate),
            LayerType::GRU(layer) => layer.update_weights(gradient, learning_rate),
            LayerType::MultiHeadAttention(layer) => layer.update_weights(gradient, learning_rate),
        }
    }
    
    fn get_params(&self) -> Vec<&Array2<f32>> {
        match self {
            LayerType::Linear(layer) => layer.get_params(),
            LayerType::Conv1D(layer) => layer.get_params(),
            LayerType::Dropout(layer) => layer.get_params(),
            LayerType::LayerNorm(layer) => layer.get_params(),
            LayerType::Embedding(_) => vec![],
            LayerType::LSTM(layer) => layer.get_params(),
            LayerType::GRU(layer) => layer.get_params(),
            LayerType::MultiHeadAttention(layer) => layer.get_params(),
        }
    }
    
    fn get_params_mut(&mut self) -> Vec<&mut Array2<f32>> {
        match self {
            LayerType::Linear(layer) => layer.get_params_mut(),
            LayerType::Conv1D(layer) => layer.get_params_mut(),
            LayerType::Dropout(layer) => layer.get_params_mut(),
            LayerType::LayerNorm(layer) => layer.get_params_mut(),
            LayerType::Embedding(_) => vec![],
            LayerType::LSTM(layer) => layer.get_params_mut(),
            LayerType::GRU(layer) => layer.get_params_mut(),
            LayerType::MultiHeadAttention(layer) => layer.get_params_mut(),
        }
    }
    
    fn to_layer_state(&self) -> Option<crate::persistence::LayerState> {
        match self {
            LayerType::Linear(layer) => Layer::to_layer_state(layer),
            LayerType::Conv1D(layer) => Layer::to_layer_state(layer),
            LayerType::Dropout(layer) => Layer::to_layer_state(layer),
            LayerType::LayerNorm(layer) => Layer::to_layer_state(layer),
            LayerType::Embedding(layer) => Some(WeightExtraction::to_layer_state(layer)),
            LayerType::LSTM(layer) => Layer::to_layer_state(layer),
            LayerType::GRU(layer) => Layer::to_layer_state(layer),
            LayerType::MultiHeadAttention(layer) => Layer::to_layer_state(layer),
        }
    }
}

pub trait Layer: Send + Sync {
    fn forward(&self, input: &Array2<f32>, training: bool) -> Array2<f32>;
    fn backward(&self, grad_output: &Array2<f32>, input: &Array2<f32>) -> Array2<f32>;
    fn update_weights(&mut self, gradient: &Gradient, learning_rate: f32);
    fn get_params(&self) -> Vec<&Array2<f32>>;
    fn get_params_mut(&mut self) -> Vec<&mut Array2<f32>>;
    
    // Methods for self-optimizing support
    fn count_parameters(&self) -> usize {
        self.get_params().iter().map(|p| p.len()).sum()
    }
    
    fn output_size(&self) -> Option<usize> {
        None // Default implementation, override in specific layers
    }
    
    fn increase_dropout(&mut self, _increase: f32) {
        // Default no-op, override in dropout layers
    }
    
    fn perturb_weights(&mut self, noise_scale: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for param in self.get_params_mut() {
            for val in param.iter_mut() {
                *val += rng.gen_range(-noise_scale..noise_scale);
            }
        }
    }
    
    // Method for persistence support
    fn to_layer_state(&self) -> Option<crate::persistence::LayerState> {
        None // Default implementation, override in specific layers
    }
    
    // Methods for continual learning support
    fn get_importance_weights(&self) -> Option<Vec<Array2<f32>>> {
        None // Default implementation, override in layers that track importance
    }
    
    fn set_importance_weights(&mut self, _weights: Vec<Array2<f32>>) {
        // Default no-op, override in layers that support importance weights
    }
    
    fn update_weights_with_ewc(
        &mut self,
        gradient: &Gradient,
        learning_rate: f32,
        importance: Option<&Vec<Array2<f32>>>,
        lambda: f32,
        prev_params: Option<&Vec<Array2<f32>>>,
    ) {
        if let (Some(imp), Some(prev)) = (importance, prev_params) {
            // EWC-aware weight update
            let params = self.get_params_mut();
            
            for (idx, param) in params.iter_mut().enumerate() {
                if let Some(ref grad) = gradient.weights {
                    if idx < imp.len() && idx < prev.len() {
                        // Standard gradient + EWC penalty gradient
                        let ewc_grad = lambda * &imp[idx] * (&**param - &prev[idx]);
                        **param = &**param - learning_rate * (grad + &ewc_grad);
                    } else {
                        // Standard update if no importance weights
                        **param = &**param - learning_rate * grad;
                    }
                }
            }
        } else {
            // Standard update without EWC
            self.update_weights(gradient, learning_rate);
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f32>,
    pub bias: Array2<f32>,
    pub activation: ActivationFunction,
    pub use_bias: bool,
    gradient: Arc<RwLock<Gradient>>,
}

impl LinearLayer {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        activation: ActivationFunction,
        use_bias: bool,
        init: WeightInit,
    ) -> Self {
        let weights = init.initialize(&[input_dim, output_dim]);
        let bias = if use_bias {
            Array2::zeros((1, output_dim))
        } else {
            Array2::zeros((1, 1))
        };
        
        Self {
            weights,
            bias,
            activation,
            use_bias,
            gradient: Arc::new(RwLock::new(Gradient::new())),
        }
    }
}

impl Layer for LinearLayer {
    fn forward(&self, input: &Array2<f32>, _training: bool) -> Array2<f32> {
        let z = Tensor::matmul_simd(input, &self.weights);
        let z_bias = if self.use_bias {
            Tensor::add_bias_simd(&z, &self.bias)
        } else {
            z
        };
        
        Activation::forward(&z_bias, self.activation)
    }
    
    fn backward(&self, grad_output: &Array2<f32>, input: &Array2<f32>) -> Array2<f32> {
        // Backprop through activation
        let z = Tensor::matmul_simd(input, &self.weights);
        let z_bias = if self.use_bias {
            Tensor::add_bias_simd(&z, &self.bias)
        } else {
            z
        };
        
        let grad_activation = Activation::backward(grad_output, &z_bias, self.activation);
        
        // Compute weight gradients
        let grad_weights = Tensor::matmul_simd(&input.t().to_owned(), &grad_activation);
        
        // Compute bias gradients
        let grad_bias = if self.use_bias {
            grad_activation.sum_axis(Axis(0)).insert_axis(Axis(0))
        } else {
            Array2::zeros((1, 1))
        };
        
        // Store gradients
        let mut grad = self.gradient.write();
        grad.weights = Some(grad_weights);
        grad.bias = Some(grad_bias);
        
        // Compute input gradients
        Tensor::matmul_simd(&grad_activation, &self.weights.t().to_owned())
    }
    
    fn update_weights(&mut self, gradient: &Gradient, learning_rate: f32) {
        if let Some(ref grad_w) = gradient.weights {
            self.weights = &self.weights - learning_rate * grad_w;
        }
        
        if self.use_bias {
            if let Some(ref grad_b) = gradient.bias {
                self.bias = &self.bias - learning_rate * grad_b;
            }
        }
    }
    
    fn get_params(&self) -> Vec<&Array2<f32>> {
        if self.use_bias {
            vec![&self.weights, &self.bias]
        } else {
            vec![&self.weights]
        }
    }
    
    fn get_params_mut(&mut self) -> Vec<&mut Array2<f32>> {
        if self.use_bias {
            vec![&mut self.weights, &mut self.bias]
        } else {
            vec![&mut self.weights]
        }
    }
    
    fn output_size(&self) -> Option<usize> {
        Some(self.weights.shape()[1])
    }
    
    fn to_layer_state(&self) -> Option<crate::persistence::LayerState> {
        Some(WeightExtraction::to_layer_state(self))
    }
}

#[derive(Debug, Clone)]
pub struct Conv1DLayer {
    pub filters: Array3<f32>,
    pub bias: Array2<f32>,
    pub stride: usize,
    pub padding: usize,
    pub activation: ActivationFunction,
    gradient: Arc<RwLock<Gradient>>,
}

impl Conv1DLayer {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        activation: ActivationFunction,
        _init: WeightInit,
    ) -> Self {
        let filters = Array3::from_shape_fn((out_channels, in_channels, kernel_size), |_| {
            rand::random::<f32>() * 0.01
        });
        let bias = Array2::zeros((1, out_channels));
        
        Self {
            filters,
            bias,
            stride,
            padding,
            activation,
            gradient: Arc::new(RwLock::new(Gradient::new())),
        }
    }
}

impl Layer for Conv1DLayer {
    fn forward(&self, input: &Array2<f32>, _training: bool) -> Array2<f32> {
        // Simplified 1D convolution for memory sequences
        // Input shape: (batch_size, sequence_length * channels)
        // For full implementation, would need proper conv1d with padding/stride
        
        let batch_size = input.shape()[0];
        let input_len = input.shape()[1];
        let in_channels = self.filters.shape()[1];
        let out_channels = self.filters.shape()[0];
        let kernel_size = self.filters.shape()[2];
        
        // Reshape input to (batch, channels, sequence)
        let seq_len = input_len / in_channels;
        
        // Simplified output without proper convolution
        // In production, implement proper conv1d
        let output_len = (seq_len + 2 * self.padding - kernel_size) / self.stride + 1;
        let output = Array2::zeros((batch_size, output_len * out_channels));
        
        // Apply activation
        Activation::forward(&output, self.activation)
    }
    
    fn backward(&self, _grad_output: &Array2<f32>, input: &Array2<f32>) -> Array2<f32> {
        // Simplified backward pass - full implementation would compute proper conv gradients
        Array2::zeros(input.dim())
    }
    
    fn update_weights(&mut self, _gradient: &Gradient, _learning_rate: f32) {
        // Update conv filters and bias based on gradients
    }
    
    fn get_params(&self) -> Vec<&Array2<f32>> {
        vec![&self.bias]
    }
    
    fn get_params_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![&mut self.bias]
    }
    
    fn to_layer_state(&self) -> Option<crate::persistence::LayerState> {
        Some(WeightExtraction::to_layer_state(self))
    }
}

#[derive(Debug, Clone)]
pub struct DropoutLayer {
    pub dropout_rate: f32,
}

impl DropoutLayer {
    pub fn new(dropout_rate: f32) -> Self {
        Self { dropout_rate }
    }
}

impl Layer for DropoutLayer {
    fn forward(&self, input: &Array2<f32>, training: bool) -> Array2<f32> {
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
    
    fn backward(&self, grad_output: &Array2<f32>, _input: &Array2<f32>) -> Array2<f32> {
        // Dropout gradient is applied element-wise
        grad_output.clone()
    }
    
    fn update_weights(&mut self, _gradient: &Gradient, _learning_rate: f32) {
        // Dropout has no weights
    }
    
    fn get_params(&self) -> Vec<&Array2<f32>> {
        vec![]
    }
    
    fn get_params_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![]
    }
    
    fn increase_dropout(&mut self, increase: f32) {
        self.dropout_rate = (self.dropout_rate + increase).min(0.9); // Cap at 90%
    }
    
    fn to_layer_state(&self) -> Option<crate::persistence::LayerState> {
        Some(WeightExtraction::to_layer_state(self))
    }
}

#[derive(Debug, Clone)]
pub struct LayerNormLayer {
    pub gamma: Array2<f32>,
    pub beta: Array2<f32>,
    pub eps: f32,
    gradient: Arc<RwLock<Gradient>>,
}

impl LayerNormLayer {
    pub fn new(normalized_shape: usize, eps: f32) -> Self {
        Self {
            gamma: Array2::ones((1, normalized_shape)),
            beta: Array2::zeros((1, normalized_shape)),
            eps,
            gradient: Arc::new(RwLock::new(Gradient::new())),
        }
    }
}

impl Layer for LayerNormLayer {
    fn forward(&self, input: &Array2<f32>, _training: bool) -> Array2<f32> {
        let normalized = Tensor::layer_norm(input, self.eps);
        &normalized * &self.gamma + &self.beta
    }
    
    fn backward(&self, grad_output: &Array2<f32>, input: &Array2<f32>) -> Array2<f32> {
        // Compute layer norm backward pass
        let normalized = Tensor::layer_norm(input, self.eps);
        
        // Gradients for gamma and beta
        let grad_gamma = (grad_output * &normalized).sum_axis(Axis(0)).insert_axis(Axis(0));
        let grad_beta = grad_output.sum_axis(Axis(0)).insert_axis(Axis(0));
        
        let mut grad = self.gradient.write();
        grad.weights = Some(grad_gamma);
        grad.bias = Some(grad_beta);
        
        // Input gradient (simplified)
        grad_output * &self.gamma
    }
    
    fn update_weights(&mut self, gradient: &Gradient, learning_rate: f32) {
        if let Some(ref grad_gamma) = gradient.weights {
            self.gamma = &self.gamma - learning_rate * grad_gamma;
        }
        if let Some(ref grad_beta) = gradient.bias {
            self.beta = &self.beta - learning_rate * grad_beta;
        }
    }
    
    fn get_params(&self) -> Vec<&Array2<f32>> {
        vec![&self.gamma, &self.beta]
    }
    
    fn get_params_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![&mut self.gamma, &mut self.beta]
    }
    
    fn to_layer_state(&self) -> Option<crate::persistence::LayerState> {
        Some(WeightExtraction::to_layer_state(self))
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingLayer {
    pub embeddings: Array2<f32>,
    pub vocab_size: usize,
    pub embedding_dim: usize,
}

impl EmbeddingLayer {
    pub fn new(vocab_size: usize, embedding_dim: usize, init: WeightInit) -> Self {
        let embeddings = init.initialize(&[vocab_size, embedding_dim]);
        
        Self {
            embeddings,
            vocab_size,
            embedding_dim,
        }
    }
    
    pub fn lookup(&self, indices: &[usize]) -> Array2<f32> {
        let batch_size = indices.len();
        let mut output = Array2::zeros((batch_size, self.embedding_dim));
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.vocab_size {
                output.row_mut(i).assign(&self.embeddings.row(idx));
            }
        }
        
        output
    }
}