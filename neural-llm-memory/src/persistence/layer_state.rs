//! Layer state representation for persistence

use serde::{Serialize, Deserialize};
use ndarray::{Array2, Array3};
use crate::nn::{ActivationFunction, LinearLayer, DropoutLayer, LayerNormLayer};

/// Serializable state of a neural network layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerState {
    /// Layer configuration
    pub config: LayerConfig,
    
    /// Weights (if applicable)
    pub weights: Option<Array2<f32>>,
    
    /// Bias (if applicable)
    pub bias: Option<Array2<f32>>,
    
    /// Additional parameters (e.g., layer norm gamma/beta)
    pub extra_params: Option<ExtraParameters>,
}

/// Layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerConfig {
    Linear {
        input_dim: usize,
        output_dim: usize,
        activation: ActivationFunction,
        use_bias: bool,
    },
    Conv1D {
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        activation: ActivationFunction,
    },
    Dropout {
        dropout_rate: f32,
    },
    LayerNorm {
        normalized_shape: usize,
        eps: f32,
    },
    Embedding {
        vocab_size: usize,
        embedding_dim: usize,
    },
}

/// Extra parameters for special layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtraParameters {
    LayerNorm {
        gamma: Array2<f32>,
        beta: Array2<f32>,
    },
    Conv1D {
        filters: Array3<f32>,
    },
    Embedding {
        embeddings: Array2<f32>,
    },
}

impl LayerState {
    /// Create from a linear layer
    pub fn from_linear(layer: &LinearLayer) -> Self {
        Self {
            config: LayerConfig::Linear {
                input_dim: layer.weights.shape()[0],
                output_dim: layer.weights.shape()[1],
                activation: layer.activation.clone(),
                use_bias: layer.use_bias,
            },
            weights: Some(layer.weights.clone()),
            bias: if layer.use_bias {
                Some(layer.bias.clone())
            } else {
                None
            },
            extra_params: None,
        }
    }
    
    /// Create from a dropout layer
    pub fn from_dropout(layer: &DropoutLayer) -> Self {
        Self {
            config: LayerConfig::Dropout {
                dropout_rate: layer.dropout_rate,
            },
            weights: None,
            bias: None,
            extra_params: None,
        }
    }
    
    /// Create from a layer norm layer
    pub fn from_layer_norm(layer: &LayerNormLayer) -> Self {
        Self {
            config: LayerConfig::LayerNorm {
                normalized_shape: layer.gamma.shape()[1],
                eps: layer.eps,
            },
            weights: None,
            bias: None,
            extra_params: Some(ExtraParameters::LayerNorm {
                gamma: layer.gamma.clone(),
                beta: layer.beta.clone(),
            }),
        }
    }
    
    /// Convert back to a linear layer
    pub fn to_linear(&self) -> Result<LinearLayer, String> {
        match &self.config {
            LayerConfig::Linear { input_dim, output_dim, activation, use_bias } => {
                let weights = self.weights.as_ref()
                    .ok_or("Missing weights for linear layer")?;
                
                let mut layer = LinearLayer::new(
                    *input_dim,
                    *output_dim,
                    activation.clone(),
                    *use_bias,
                    crate::nn::WeightInit::Zeros, // Will be overwritten
                );
                
                layer.weights = weights.clone();
                if *use_bias {
                    layer.bias = self.bias.as_ref()
                        .ok_or("Missing bias for linear layer")?
                        .clone();
                }
                
                Ok(layer)
            }
            _ => Err("Not a linear layer config".to_string()),
        }
    }
    
    /// Convert back to a dropout layer
    pub fn to_dropout(&self) -> Result<DropoutLayer, String> {
        match &self.config {
            LayerConfig::Dropout { dropout_rate } => {
                Ok(DropoutLayer::new(*dropout_rate))
            }
            _ => Err("Not a dropout layer config".to_string()),
        }
    }
    
    /// Convert back to a layer norm layer
    pub fn to_layer_norm(&self) -> Result<LayerNormLayer, String> {
        match &self.config {
            LayerConfig::LayerNorm { normalized_shape, eps } => {
                match &self.extra_params {
                    Some(ExtraParameters::LayerNorm { gamma, beta }) => {
                        let mut layer = LayerNormLayer::new(*normalized_shape, *eps);
                        layer.gamma = gamma.clone();
                        layer.beta = beta.clone();
                        Ok(layer)
                    }
                    _ => Err("Missing layer norm parameters".to_string()),
                }
            }
            _ => Err("Not a layer norm config".to_string()),
        }
    }
    
    /// Get the output size of this layer
    pub fn output_size(&self) -> usize {
        match &self.config {
            LayerConfig::Linear { output_dim, .. } => *output_dim,
            LayerConfig::Conv1D { out_channels, .. } => *out_channels,
            LayerConfig::Dropout { .. } => 0, // Preserves input size
            LayerConfig::LayerNorm { normalized_shape, .. } => *normalized_shape,
            LayerConfig::Embedding { embedding_dim, .. } => *embedding_dim,
        }
    }
    
    /// Get the input size of this layer
    pub fn input_size(&self) -> usize {
        match &self.config {
            LayerConfig::Linear { input_dim, .. } => *input_dim,
            LayerConfig::Conv1D { in_channels, .. } => *in_channels,
            LayerConfig::Dropout { .. } => 0, // Preserves input size
            LayerConfig::LayerNorm { normalized_shape, .. } => *normalized_shape,
            LayerConfig::Embedding { vocab_size, .. } => *vocab_size,
        }
    }
    
    /// Count parameters in this layer
    pub fn count_parameters(&self) -> usize {
        let mut count = 0;
        
        if let Some(weights) = &self.weights {
            count += weights.len();
        }
        
        if let Some(bias) = &self.bias {
            count += bias.len();
        }
        
        if let Some(extra) = &self.extra_params {
            match extra {
                ExtraParameters::LayerNorm { gamma, beta } => {
                    count += gamma.len() + beta.len();
                }
                ExtraParameters::Conv1D { filters } => {
                    count += filters.len();
                }
                ExtraParameters::Embedding { embeddings } => {
                    count += embeddings.len();
                }
            }
        }
        
        count
    }
    
    /// Get layer type as string
    pub fn layer_type(&self) -> &str {
        match &self.config {
            LayerConfig::Linear { .. } => "Linear",
            LayerConfig::Conv1D { .. } => "Conv1D",
            LayerConfig::Dropout { .. } => "Dropout",
            LayerConfig::LayerNorm { .. } => "LayerNorm",
            LayerConfig::Embedding { .. } => "Embedding",
        }
    }
    
    /// Get a summary string
    pub fn summary(&self) -> String {
        match &self.config {
            LayerConfig::Linear { input_dim, output_dim, activation, use_bias } => {
                format!(
                    "Linear: {} -> {}, activation: {:?}, bias: {}, params: {}",
                    input_dim, output_dim, activation, use_bias, self.count_parameters()
                )
            }
            LayerConfig::Conv1D { in_channels, out_channels, kernel_size, .. } => {
                format!(
                    "Conv1D: {} -> {}, kernel: {}, params: {}",
                    in_channels, out_channels, kernel_size, self.count_parameters()
                )
            }
            LayerConfig::Dropout { dropout_rate } => {
                format!("Dropout: rate = {}", dropout_rate)
            }
            LayerConfig::LayerNorm { normalized_shape, eps } => {
                format!(
                    "LayerNorm: shape = {}, eps = {}, params: {}",
                    normalized_shape, eps, self.count_parameters()
                )
            }
            LayerConfig::Embedding { vocab_size, embedding_dim } => {
                format!(
                    "Embedding: vocab = {}, dim = {}, params: {}",
                    vocab_size, embedding_dim, self.count_parameters()
                )
            }
        }
    }
}

/// Builder for creating layer states
pub struct LayerStateBuilder {
    config: LayerConfig,
    weights: Option<Array2<f32>>,
    bias: Option<Array2<f32>>,
    extra_params: Option<ExtraParameters>,
}

impl LayerStateBuilder {
    pub fn linear(
        input_dim: usize,
        output_dim: usize,
        activation: ActivationFunction,
        use_bias: bool,
    ) -> Self {
        Self {
            config: LayerConfig::Linear {
                input_dim,
                output_dim,
                activation,
                use_bias,
            },
            weights: None,
            bias: None,
            extra_params: None,
        }
    }
    
    pub fn dropout(dropout_rate: f32) -> Self {
        Self {
            config: LayerConfig::Dropout { dropout_rate },
            weights: None,
            bias: None,
            extra_params: None,
        }
    }
    
    pub fn layer_norm(normalized_shape: usize, eps: f32) -> Self {
        Self {
            config: LayerConfig::LayerNorm { normalized_shape, eps },
            weights: None,
            bias: None,
            extra_params: None,
        }
    }
    
    pub fn with_weights(mut self, weights: Array2<f32>) -> Self {
        self.weights = Some(weights);
        self
    }
    
    pub fn with_bias(mut self, bias: Array2<f32>) -> Self {
        self.bias = Some(bias);
        self
    }
    
    pub fn with_extra_params(mut self, params: ExtraParameters) -> Self {
        self.extra_params = Some(params);
        self
    }
    
    pub fn build(self) -> LayerState {
        LayerState {
            config: self.config,
            weights: self.weights,
            bias: self.bias,
            extra_params: self.extra_params,
        }
    }
}

