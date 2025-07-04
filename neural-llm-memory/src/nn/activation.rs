//! Activation functions with optimization

use ndarray::Array2;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU(f32),
    GELU,
    Sigmoid,
    Tanh,
    Swish,
    Mish,
    ELU(f32),
    Identity,
    #[serde(rename = "SiLU")]
    SiLU, // Alias for Swish
}

pub struct Activation;

impl Activation {
    pub fn forward(input: &Array2<f32>, activation: ActivationFunction) -> Array2<f32> {
        match activation {
            ActivationFunction::ReLU => Self::relu(input),
            ActivationFunction::LeakyReLU(alpha) => Self::leaky_relu(input, alpha),
            ActivationFunction::GELU => Self::gelu(input),
            ActivationFunction::Sigmoid => Self::sigmoid(input),
            ActivationFunction::Tanh => Self::tanh(input),
            ActivationFunction::Swish | ActivationFunction::SiLU => Self::swish(input),
            ActivationFunction::Mish => Self::mish(input),
            ActivationFunction::ELU(alpha) => Self::elu(input, alpha),
            ActivationFunction::Identity => input.clone(),
        }
    }
    
    pub fn backward(
        grad_output: &Array2<f32>,
        input: &Array2<f32>,
        activation: ActivationFunction
    ) -> Array2<f32> {
        match activation {
            ActivationFunction::ReLU => Self::relu_backward(grad_output, input),
            ActivationFunction::LeakyReLU(alpha) => Self::leaky_relu_backward(grad_output, input, alpha),
            ActivationFunction::GELU => Self::gelu_backward(grad_output, input),
            ActivationFunction::Sigmoid => Self::sigmoid_backward(grad_output, input),
            ActivationFunction::Tanh => Self::tanh_backward(grad_output, input),
            ActivationFunction::Swish | ActivationFunction::SiLU => Self::swish_backward(grad_output, input),
            ActivationFunction::Mish => Self::mish_backward(grad_output, input),
            ActivationFunction::ELU(alpha) => Self::elu_backward(grad_output, input, alpha),
            ActivationFunction::Identity => grad_output.clone(),
        }
    }
    
    // Forward activations
    fn relu(input: &Array2<f32>) -> Array2<f32> {
        use crate::nn::tensor::{Tensor, TensorOps};
        Tensor::relu_simd(input)
    }
    
    fn leaky_relu(input: &Array2<f32>, alpha: f32) -> Array2<f32> {
        input.mapv(|x| if x > 0.0 { x } else { alpha * x })
    }
    
    fn gelu(input: &Array2<f32>) -> Array2<f32> {
        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        const COEFF: f32 = 0.044715;
        
        input.mapv(|x| {
            let cdf = 0.5 * (1.0 + ((SQRT_2_OVER_PI * (x + COEFF * x.powi(3))).tanh()));
            x * cdf
        })
    }
    
    fn sigmoid(input: &Array2<f32>) -> Array2<f32> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
    
    fn tanh(input: &Array2<f32>) -> Array2<f32> {
        input.mapv(|x| x.tanh())
    }
    
    fn swish(input: &Array2<f32>) -> Array2<f32> {
        let sigmoid = Self::sigmoid(input);
        input * &sigmoid
    }
    
    fn mish(input: &Array2<f32>) -> Array2<f32> {
        input.mapv(|x| x * (1.0 + x.exp()).ln().tanh())
    }
    
    fn elu(input: &Array2<f32>, alpha: f32) -> Array2<f32> {
        input.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
    }
    
    // Backward activations
    fn relu_backward(grad_output: &Array2<f32>, input: &Array2<f32>) -> Array2<f32> {
        grad_output * &input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
    
    fn leaky_relu_backward(grad_output: &Array2<f32>, input: &Array2<f32>, alpha: f32) -> Array2<f32> {
        grad_output * &input.mapv(|x| if x > 0.0 { 1.0 } else { alpha })
    }
    
    fn gelu_backward(grad_output: &Array2<f32>, input: &Array2<f32>) -> Array2<f32> {
        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        const COEFF: f32 = 0.044715;
        
        grad_output * &input.mapv(|x| {
            let inner = SQRT_2_OVER_PI * (x + COEFF * x.powi(3));
            let tanh_inner = inner.tanh();
            let sech2 = 1.0 - tanh_inner.powi(2);
            
            0.5 * (1.0 + tanh_inner) + 
            0.5 * x * sech2 * SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x.powi(2))
        })
    }
    
    fn sigmoid_backward(grad_output: &Array2<f32>, input: &Array2<f32>) -> Array2<f32> {
        let sigmoid = Self::sigmoid(input);
        grad_output * &sigmoid * &(1.0 - &sigmoid)
    }
    
    fn tanh_backward(grad_output: &Array2<f32>, input: &Array2<f32>) -> Array2<f32> {
        let tanh_val = input.mapv(|x| x.tanh());
        grad_output * &(1.0 - &tanh_val * &tanh_val)
    }
    
    fn swish_backward(grad_output: &Array2<f32>, input: &Array2<f32>) -> Array2<f32> {
        let sigmoid = Self::sigmoid(input);
        let swish = input * &sigmoid;
        let one_minus_swish = 1.0 - &swish;
        grad_output * &(&swish + &sigmoid * &one_minus_swish)
    }
    
    fn mish_backward(grad_output: &Array2<f32>, input: &Array2<f32>) -> Array2<f32> {
        grad_output * &input.mapv(|x| {
            let exp_x = x.exp();
            let softplus = (1.0 + exp_x).ln();
            let tanh_softplus = softplus.tanh();
            let sech2_softplus = 1.0 - tanh_softplus.powi(2);
            
            tanh_softplus + x * sech2_softplus * exp_x / (1.0 + exp_x)
        })
    }
    
    fn elu_backward(grad_output: &Array2<f32>, input: &Array2<f32>, alpha: f32) -> Array2<f32> {
        grad_output * &input.mapv(|x| if x > 0.0 { 1.0 } else { alpha * x.exp() })
    }
}