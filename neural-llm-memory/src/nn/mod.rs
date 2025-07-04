//! Core neural network components

pub mod layer;
pub mod activation;
pub mod network;
pub mod tensor;

pub use layer::{Layer, LayerType, LinearLayer, Conv1DLayer, DropoutLayer, LayerNormLayer};
pub use activation::{Activation, ActivationFunction};
pub use network::{NeuralNetwork, NetworkBuilder};
pub use tensor::{Tensor, TensorOps};

use ndarray::Array2;
use rand::Rng;
use std::sync::Arc;
use parking_lot::RwLock;

/// Initialize weights using various strategies
pub enum WeightInit {
    Xavier,
    He,
    Normal(f32),
    Uniform(f32, f32),
    Zeros,
}

impl WeightInit {
    pub fn initialize(&self, shape: &[usize]) -> Array2<f32> {
        let mut rng = rand::thread_rng();
        let _size: usize = shape.iter().product();
        
        match self {
            WeightInit::Xavier => {
                let fan_in = shape[0] as f32;
                let fan_out = shape[1] as f32;
                let scale = (2.0 / (fan_in + fan_out)).sqrt();
                
                Array2::from_shape_fn((shape[0], shape[1]), |_| {
                    rng.gen_range(-scale..scale)
                })
            }
            WeightInit::He => {
                let fan_in = shape[0] as f32;
                let scale = (2.0 / fan_in).sqrt();
                
                Array2::from_shape_fn((shape[0], shape[1]), |_| {
                    rng.gen_range(-scale..scale)
                })
            }
            WeightInit::Normal(std_dev) => {
                use rand_distr::{Normal, Distribution};
                let normal = Normal::new(0.0, *std_dev).unwrap();
                
                Array2::from_shape_fn((shape[0], shape[1]), |_| {
                    normal.sample(&mut rng)
                })
            }
            WeightInit::Uniform(min, max) => {
                Array2::from_shape_fn((shape[0], shape[1]), |_| {
                    rng.gen_range(*min..*max)
                })
            }
            WeightInit::Zeros => Array2::zeros((shape[0], shape[1])),
        }
    }
}

/// Gradient information for backpropagation
#[derive(Clone, Debug)]
pub struct Gradient {
    pub weights: Option<Array2<f32>>,
    pub bias: Option<Array2<f32>>,
    pub input: Option<Array2<f32>>,
}

impl Gradient {
    pub fn new() -> Self {
        Self {
            weights: None,
            bias: None,
            input: None,
        }
    }
}

/// Training state for neural network
pub struct TrainingState {
    pub epoch: usize,
    pub global_step: usize,
    pub learning_rate: f32,
    pub gradients: Vec<Arc<RwLock<Gradient>>>,
    pub loss_history: Vec<f32>,
}

impl TrainingState {
    pub fn new(learning_rate: f32, num_layers: usize) -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            learning_rate,
            gradients: (0..num_layers)
                .map(|_| Arc::new(RwLock::new(Gradient::new())))
                .collect(),
            loss_history: Vec::new(),
        }
    }
    
    pub fn update_step(&mut self) {
        self.global_step += 1;
    }
    
    pub fn update_epoch(&mut self) {
        self.epoch += 1;
    }
    
    pub fn record_loss(&mut self, loss: f32) {
        self.loss_history.push(loss);
    }
}