// Proof of concept for weight extraction implementation
// This shows how to implement weight saving for LinearLayer

use ndarray::{Array1, Array2, Array3};
use crate::persistence::layer_state::{LayerState, LayerConfig, ExtraParameters};
use crate::nn::{LinearLayer, Conv1DLayer, LayerNormLayer, DropoutLayer, Layer, WeightInit};
use crate::nn::layer::EmbeddingLayer;

/// Trait extension for weight extraction
pub trait WeightExtraction {
    /// Extract weights as a 2D array (if applicable)
    fn extract_weights(&self) -> Option<Array2<f32>>;
    
    /// Extract biases as a 1D array (if applicable)
    fn extract_biases(&self) -> Option<Array1<f32>>;
    
    /// Convert to a serializable LayerState
    fn to_layer_state(&self) -> LayerState;
}

impl WeightExtraction for LinearLayer {
    fn extract_weights(&self) -> Option<Array2<f32>> {
        Some(self.weights.clone())
    }
    
    fn extract_biases(&self) -> Option<Array1<f32>> {
        if self.use_bias {
            Some(self.bias.row(0).to_owned())
        } else {
            None
        }
    }
    
    fn to_layer_state(&self) -> LayerState {
        LayerState {
            config: LayerConfig::Linear {
                input_dim: self.weights.shape()[0],
                output_dim: self.weights.shape()[1],
                activation: self.activation.clone(),
                use_bias: self.use_bias,
            },
            weights: Some(self.weights.clone()),
            bias: if self.use_bias {
                Some(self.bias.clone())
            } else {
                None
            },
            extra_params: None,
        }
    }
}

/// Restore a LinearLayer from LayerState
impl LinearLayer {
    pub fn from_layer_state(state: &LayerState) -> Result<Self, String> {
        match &state.config {
            LayerConfig::Linear { input_dim, output_dim, activation, use_bias } => {
                // Create layer with correct dimensions
                let mut layer = Self::new(*input_dim, *output_dim, activation.clone(), *use_bias, WeightInit::Zeros);
                
                // Restore weights
                if let Some(weights) = &state.weights {
                    if weights.shape() != [*input_dim, *output_dim] {
                        return Err(format!(
                            "Weight shape mismatch: expected [{}, {}], got {:?}",
                            input_dim, output_dim, weights.shape()
                        ));
                    }
                    layer.weights = weights.clone();
                }
                
                // Restore biases
                if *use_bias {
                    if let Some(bias) = &state.bias {
                        if bias.shape() != [1, *output_dim] {
                            return Err(format!(
                                "Bias shape mismatch: expected [1, {}], got {:?}",
                                output_dim, bias.shape()
                            ));
                        }
                        layer.bias = bias.clone();
                    }
                }
                
                Ok(layer)
            }
            _ => Err("Invalid layer config for LinearLayer".to_string())
        }
    }
}


// Implementation for Conv1DLayer
impl WeightExtraction for Conv1DLayer {
    fn extract_weights(&self) -> Option<Array2<f32>> {
        // Reshape 3D filters to 2D for consistency with trait
        // Shape: (out_channels, in_channels, kernel_size) -> (out_channels, in_channels * kernel_size)
        let (out_channels, in_channels, kernel_size) = self.filters.dim();
        let reshaped = self.filters.clone()
            .into_shape((out_channels, in_channels * kernel_size))
            .ok()?;
        Some(reshaped)
    }
    
    fn extract_biases(&self) -> Option<Array1<f32>> {
        // Convert 2D bias (1, out_channels) to 1D
        Some(self.bias.row(0).to_owned())
    }
    
    fn to_layer_state(&self) -> LayerState {
        LayerState {
            config: LayerConfig::Conv1D {
                in_channels: self.filters.shape()[1],
                out_channels: self.filters.shape()[0],
                kernel_size: self.filters.shape()[2],
                stride: self.stride,
                padding: self.padding,
                activation: self.activation.clone(),
            },
            weights: None, // Conv1D stores filters in extra_params
            bias: Some(self.bias.clone()),
            extra_params: Some(ExtraParameters::Conv1D {
                filters: self.filters.clone(),
            }),
        }
    }
}

/// Restore a Conv1DLayer from LayerState
impl Conv1DLayer {
    pub fn from_layer_state(state: &LayerState) -> Result<Self, String> {
        match &state.config {
            LayerConfig::Conv1D { in_channels, out_channels, kernel_size, stride, padding, activation } => {
                // Create layer with correct dimensions
                let mut layer = Self::new(
                    *in_channels,
                    *out_channels,
                    *kernel_size,
                    *stride,
                    *padding,
                    activation.clone(),
                    WeightInit::Xavier,
                );
                
                // Restore filters from extra_params
                if let Some(ExtraParameters::Conv1D { filters }) = &state.extra_params {
                    if filters.shape() != [*out_channels, *in_channels, *kernel_size] {
                        return Err(format!(
                            "Filter shape mismatch: expected [{}, {}, {}], got {:?}",
                            out_channels, in_channels, kernel_size, filters.shape()
                        ));
                    }
                    layer.filters = filters.clone();
                } else {
                    return Err("Missing Conv1D filters in extra_params".to_string());
                }
                
                // Restore bias
                if let Some(bias) = &state.bias {
                    if bias.shape() != [1, *out_channels] {
                        return Err(format!(
                            "Bias shape mismatch: expected [1, {}], got {:?}",
                            out_channels, bias.shape()
                        ));
                    }
                    layer.bias = bias.clone();
                }
                
                Ok(layer)
            }
            _ => Err("Invalid layer config for Conv1DLayer".to_string())
        }
    }
}

// Implementation for LayerNormLayer
impl WeightExtraction for LayerNormLayer {
    fn extract_weights(&self) -> Option<Array2<f32>> {
        // Return gamma as weights (it's already 2D)
        Some(self.gamma.clone())
    }
    
    fn extract_biases(&self) -> Option<Array1<f32>> {
        // Return beta as bias (convert from 2D to 1D)
        Some(self.beta.row(0).to_owned())
    }
    
    fn to_layer_state(&self) -> LayerState {
        LayerState {
            config: LayerConfig::LayerNorm {
                normalized_shape: self.gamma.shape()[1],
                eps: self.eps,
            },
            weights: None, // LayerNorm stores params in extra_params
            bias: None,
            extra_params: Some(ExtraParameters::LayerNorm {
                gamma: self.gamma.clone(),
                beta: self.beta.clone(),
            }),
        }
    }
}

/// Restore a LayerNormLayer from LayerState
impl LayerNormLayer {
    pub fn from_layer_state(state: &LayerState) -> Result<Self, String> {
        match &state.config {
            LayerConfig::LayerNorm { normalized_shape, eps } => {
                // Create layer with correct dimensions
                let mut layer = Self::new(*normalized_shape, *eps);
                
                // Restore gamma and beta from extra_params
                if let Some(ExtraParameters::LayerNorm { gamma, beta }) = &state.extra_params {
                    if gamma.shape() != [1, *normalized_shape] {
                        return Err(format!(
                            "Gamma shape mismatch: expected [1, {}], got {:?}",
                            normalized_shape, gamma.shape()
                        ));
                    }
                    if beta.shape() != [1, *normalized_shape] {
                        return Err(format!(
                            "Beta shape mismatch: expected [1, {}], got {:?}",
                            normalized_shape, beta.shape()
                        ));
                    }
                    layer.gamma = gamma.clone();
                    layer.beta = beta.clone();
                } else {
                    return Err("Missing LayerNorm parameters in extra_params".to_string());
                }
                
                Ok(layer)
            }
            _ => Err("Invalid layer config for LayerNormLayer".to_string())
        }
    }
}

// Implementation for EmbeddingLayer
impl WeightExtraction for EmbeddingLayer {
    fn extract_weights(&self) -> Option<Array2<f32>> {
        // Return embedding matrix
        Some(self.embeddings.clone())
    }
    
    fn extract_biases(&self) -> Option<Array1<f32>> {
        // Embedding layers don't have biases
        None
    }
    
    fn to_layer_state(&self) -> LayerState {
        LayerState {
            config: LayerConfig::Embedding {
                vocab_size: self.vocab_size,
                embedding_dim: self.embedding_dim,
            },
            weights: None, // Embedding stores matrix in extra_params
            bias: None,
            extra_params: Some(ExtraParameters::Embedding {
                embeddings: self.embeddings.clone(),
            }),
        }
    }
}

/// Restore an EmbeddingLayer from LayerState
impl EmbeddingLayer {
    pub fn from_layer_state(state: &LayerState) -> Result<Self, String> {
        match &state.config {
            LayerConfig::Embedding { vocab_size, embedding_dim } => {
                // Create layer with correct dimensions
                let mut layer = Self::new(*vocab_size, *embedding_dim, WeightInit::Xavier);
                
                // Restore embeddings from extra_params
                if let Some(ExtraParameters::Embedding { embeddings }) = &state.extra_params {
                    if embeddings.shape() != [*vocab_size, *embedding_dim] {
                        return Err(format!(
                            "Embeddings shape mismatch: expected [{}, {}], got {:?}",
                            vocab_size, embedding_dim, embeddings.shape()
                        ));
                    }
                    layer.embeddings = embeddings.clone();
                } else {
                    return Err("Missing Embedding matrix in extra_params".to_string());
                }
                
                Ok(layer)
            }
            _ => Err("Invalid layer config for EmbeddingLayer".to_string())
        }
    }
}

// Implementation for DropoutLayer
impl WeightExtraction for DropoutLayer {
    fn extract_weights(&self) -> Option<Array2<f32>> {
        // Dropout has no weights
        None
    }
    
    fn extract_biases(&self) -> Option<Array1<f32>> {
        // Dropout has no biases
        None
    }
    
    fn to_layer_state(&self) -> LayerState {
        LayerState {
            config: LayerConfig::Dropout {
                dropout_rate: self.dropout_rate,
            },
            weights: None,
            bias: None,
            extra_params: None,
        }
    }
}

/// Restore a DropoutLayer from LayerState
impl DropoutLayer {
    pub fn from_layer_state(state: &LayerState) -> Result<Self, String> {
        match &state.config {
            LayerConfig::Dropout { dropout_rate } => {
                Ok(Self::new(*dropout_rate))
            }
            _ => Err("Invalid layer config for DropoutLayer".to_string())
        }
    }
}