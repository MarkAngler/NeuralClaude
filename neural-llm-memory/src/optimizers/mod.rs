//! Optimizers for training neural networks

use ndarray::Array2;
use std::collections::HashMap;

pub trait Optimizer {
    fn step(&mut self, params: &mut [&mut Array2<f32>], gradients: &[Array2<f32>]);
    fn zero_grad(&mut self);
}

/// Stochastic Gradient Descent with momentum
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    velocities: HashMap<usize, Array2<f32>>,
}

impl SGD {
    pub fn new(learning_rate: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            velocities: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [&mut Array2<f32>], gradients: &[Array2<f32>]) {
        for (idx, (param, grad)) in params.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay
            let grad_with_decay = if self.weight_decay > 0.0 {
                grad + &(self.weight_decay * &**param)
            } else {
                grad.clone()
            };
            
            // Update velocity
            let velocity = self.velocities.entry(idx).or_insert_with(|| {
                Array2::zeros(param.dim())
            });
            
            *velocity = &*velocity * self.momentum + &grad_with_decay * self.learning_rate;
            
            // Update parameters
            param.zip_mut_with(&velocity, |p, &v| *p -= v);
        }
    }
    
    fn zero_grad(&mut self) {
        // SGD doesn't accumulate gradients
    }
}

/// Adam optimizer
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    
    m: HashMap<usize, Array2<f32>>, // First moment
    v: HashMap<usize, Array2<f32>>, // Second moment
    t: usize, // Time step
}

impl Adam {
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [&mut Array2<f32>], gradients: &[Array2<f32>]) {
        self.t += 1;
        let t = self.t as f32;
        
        for (idx, (param, grad)) in params.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay
            let grad_with_decay = if self.weight_decay > 0.0 {
                grad + &(self.weight_decay * &**param)
            } else {
                grad.clone()
            };
            
            // Update biased first moment estimate
            let m = self.m.entry(idx).or_insert_with(|| Array2::zeros(param.dim()));
            *m = &*m * self.beta1 + &grad_with_decay * (1.0 - self.beta1);
            
            // Update biased second raw moment estimate
            let v = self.v.entry(idx).or_insert_with(|| Array2::zeros(param.dim()));
            *v = &*v * self.beta2 + &(&grad_with_decay * &grad_with_decay) * (1.0 - self.beta2);
            
            // Compute bias-corrected moments and update
            let bias_correction1 = 1.0 - self.beta1.powf(t);
            let bias_correction2 = 1.0 - self.beta2.powf(t);
            
            // Compute bias-corrected estimates and update parameters
            let m_hat = &*m / bias_correction1;
            let v_hat = &*v / bias_correction2;
            let v_hat_sqrt = v_hat.mapv(|x| x.sqrt() + self.epsilon);
            let update = &(m_hat / v_hat_sqrt) * self.learning_rate;
            
            **param = &**param - &update;
        }
    }
    
    fn zero_grad(&mut self) {
        // Adam doesn't accumulate gradients
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
pub struct AdamW {
    adam: Adam,
}

impl AdamW {
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            adam: Adam::new(learning_rate, beta1, beta2, epsilon, weight_decay),
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [&mut Array2<f32>], gradients: &[Array2<f32>]) {
        // Apply weight decay directly to parameters
        let weight_decay = self.adam.weight_decay;
        if weight_decay > 0.0 {
            for param in params.iter_mut() {
                param.mapv_inplace(|x| x * (1.0 - self.adam.learning_rate * weight_decay));
            }
        }
        
        // Then apply Adam update
        self.adam.step(params, gradients);
    }
    
    fn zero_grad(&mut self) {
        self.adam.zero_grad();
    }
}