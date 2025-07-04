//! Neural network architecture and builder

use crate::nn::{Layer, LinearLayer, DropoutLayer, LayerNormLayer, WeightInit, ActivationFunction, TrainingState};
use ndarray::Array2;
use std::sync::Arc;
use parking_lot::RwLock;

pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer + Send + Sync>>,
    training_state: Arc<RwLock<TrainingState>>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Box<dyn Layer + Send + Sync>>, learning_rate: f32) -> Self {
        let num_layers = layers.len();
        Self {
            layers,
            training_state: Arc::new(RwLock::new(TrainingState::new(learning_rate, num_layers))),
        }
    }
    
    pub fn forward(&self, input: &Array2<f32>, training: bool) -> Array2<f32> {
        self.layers.iter().fold(input.clone(), |acc, layer| {
            layer.forward(&acc, training)
        })
    }
    
    pub fn backward(&self, loss_gradient: &Array2<f32>, inputs: &[Array2<f32>]) -> Array2<f32> {
        let mut grad = loss_gradient.clone();
        
        // Backpropagate through layers in reverse order
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let input = if i == 0 {
                &inputs[0]
            } else {
                &inputs[i]
            };
            
            grad = layer.backward(&grad, input);
        }
        
        grad
    }
    
    pub fn update_weights(&mut self) {
        let state = self.training_state.read();
        let learning_rate = state.learning_rate;
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i < state.gradients.len() {
                let gradient = state.gradients[i].read();
                layer.update_weights(&gradient, learning_rate);
            }
        }
    }
    
    pub fn train_step(
        &mut self,
        input: &Array2<f32>,
        target: &Array2<f32>,
        loss_fn: impl Fn(&Array2<f32>, &Array2<f32>) -> (f32, Array2<f32>)
    ) -> f32 {
        // Forward pass
        let output = self.forward(input, true);
        
        // Compute loss and gradient
        let (loss, loss_grad) = loss_fn(&output, target);
        
        // Store intermediate activations for backward pass
        let mut activations = vec![input.clone()];
        let mut current = input.clone();
        
        for layer in &self.layers {
            current = layer.forward(&current, true);
            activations.push(current.clone());
        }
        
        // Backward pass
        self.backward(&loss_grad, &activations);
        
        // Update weights
        self.update_weights();
        
        // Update training state
        let mut state = self.training_state.write();
        state.update_step();
        state.record_loss(loss);
        
        loss
    }
    
    pub fn predict(&self, input: &Array2<f32>) -> Array2<f32> {
        self.forward(input, false)
    }
    
    // Methods for self-optimizing support
    pub fn get_layers(&self) -> &[Box<dyn Layer + Send + Sync>] {
        &self.layers
    }
    
    pub fn get_learning_rate(&self) -> f32 {
        self.training_state.read().learning_rate
    }
    
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.training_state.write().learning_rate = lr;
    }
    
    pub fn get_weight_decay(&self) -> f32 {
        self.training_state.read().weight_decay
    }
    
    pub fn set_weight_decay(&mut self, decay: f32) {
        self.training_state.write().weight_decay = decay;
    }
    
    pub fn increase_dropout(&mut self, increase: f32) {
        for layer in &mut self.layers {
            layer.increase_dropout(increase);
        }
    }
    
    pub fn perturb_weights(&mut self, noise_scale: f32) {
        for layer in &mut self.layers {
            layer.perturb_weights(noise_scale);
        }
    }
    
    pub fn get_current_metrics(&self) -> Option<NetworkMetrics> {
        let state = self.training_state.read();
        if state.loss_history.is_empty() {
            return None;
        }
        
        Some(NetworkMetrics {
            train_loss: state.loss_history.last().cloned().unwrap_or(0.0),
            val_loss: 0.0, // Would need to be tracked separately
            gradient_norm: state.compute_gradient_norm(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    pub train_loss: f32,
    pub val_loss: f32,
    pub gradient_norm: f32,
}

pub struct NetworkBuilder {
    layers: Vec<Box<dyn Layer + Send + Sync>>,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }
    
    pub fn add_linear(
        mut self,
        input_dim: usize,
        output_dim: usize,
        activation: ActivationFunction,
        use_bias: bool,
    ) -> Self {
        let layer = LinearLayer::new(
            input_dim,
            output_dim,
            activation,
            use_bias,
            WeightInit::Xavier,
        );
        self.layers.push(Box::new(layer));
        self
    }
    
    pub fn add_dropout(mut self, dropout_rate: f32) -> Self {
        let layer = DropoutLayer::new(dropout_rate);
        self.layers.push(Box::new(layer));
        self
    }
    
    pub fn add_layer_norm(mut self, normalized_shape: usize) -> Self {
        let layer = LayerNormLayer::new(normalized_shape, 1e-5);
        self.layers.push(Box::new(layer));
        self
    }
    
    pub fn build(self, learning_rate: f32) -> NeuralNetwork {
        NeuralNetwork::new(self.layers, learning_rate)
    }
}

// Loss functions
pub mod loss {
    use ndarray::{Array2, Axis};
    
    pub fn mse_loss(predictions: &Array2<f32>, targets: &Array2<f32>) -> (f32, Array2<f32>) {
        let diff = predictions - targets;
        let loss = (&diff * &diff).mean().unwrap();
        let grad = 2.0 * diff / predictions.len() as f32;
        (loss, grad)
    }
    
    pub fn cross_entropy_loss(predictions: &Array2<f32>, targets: &Array2<f32>) -> (f32, Array2<f32>) {
        let eps = 1e-7;
        let clipped_preds = predictions.mapv(|x| x.max(eps).min(1.0 - eps));
        
        let loss = -(targets * clipped_preds.mapv(|x| x.ln())).sum() / predictions.shape()[0] as f32;
        let grad = (predictions - targets) / predictions.shape()[0] as f32;
        
        (loss, grad)
    }
    
    pub fn cosine_similarity_loss(predictions: &Array2<f32>, targets: &Array2<f32>) -> (f32, Array2<f32>) {
        let norm_pred = predictions.mapv(|x| x * x).sum_axis(Axis(1)).mapv(|x| x.sqrt());
        let norm_target = targets.mapv(|x| x * x).sum_axis(Axis(1)).mapv(|x| x.sqrt());
        
        let dot_product = (predictions * targets).sum_axis(Axis(1));
        let cosine_sim = &dot_product / (&norm_pred * &norm_target + 1e-8);
        
        let loss = 1.0 - cosine_sim.mean().unwrap();
        
        // Gradient computation for cosine similarity
        let _batch_size = predictions.shape()[0] as f32;
        let grad = Array2::zeros(predictions.dim()); // Simplified - full gradient would be more complex
        
        (loss, grad)
    }
}