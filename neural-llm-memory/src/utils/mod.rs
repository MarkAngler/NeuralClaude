//! Utility functions and helpers

use ndarray::{Array2, ArrayView2};
use std::fs::File;
use std::io::{Read, Write};
use serde::{Serialize, Deserialize};

/// Save model weights to disk
pub fn save_weights(weights: &[Array2<f32>], path: &str) -> std::io::Result<()> {
    let serialized = bincode::serialize(weights).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, e)
    })?;
    
    let mut file = File::create(path)?;
    file.write_all(&serialized)?;
    Ok(())
}

/// Load model weights from disk
pub fn load_weights(path: &str) -> std::io::Result<Vec<Array2<f32>>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    bincode::deserialize(&buffer).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, e)
    })
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: ArrayView2<f32>, b: ArrayView2<f32>) -> f32 {
    let dot_product = (a.to_owned() * b.to_owned()).sum();
    let norm_a = (a.to_owned() * a.to_owned()).sum().sqrt();
    let norm_b = (b.to_owned() * b.to_owned()).sum().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot_product / (norm_a * norm_b)
}

/// Batch data for training
pub fn create_batches<T: Clone>(
    data: &[T],
    batch_size: usize,
) -> Vec<Vec<T>> {
    data.chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub loss: f32,
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            loss: 0.0,
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
        }
    }
    
    pub fn update(&mut self, predictions: &Array2<f32>, targets: &Array2<f32>) {
        // Simplified metric calculation
        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(p, t)| (*p - *t).abs() < 0.5)
            .count();
        
        self.accuracy = correct as f32 / predictions.len() as f32;
        self.f1_score = 2.0 * self.precision * self.recall / (self.precision + self.recall + 1e-8);
    }
}

/// Learning rate scheduler
pub trait LRScheduler {
    fn get_lr(&self, epoch: usize) -> f32;
}

/// Step learning rate scheduler
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
        }
    }
}

impl LRScheduler for StepLR {
    fn get_lr(&self, epoch: usize) -> f32 {
        let steps = epoch / self.step_size;
        self.initial_lr * self.gamma.powi(steps as i32)
    }
}

/// Cosine annealing learning rate scheduler
pub struct CosineAnnealingLR {
    initial_lr: f32,
    min_lr: f32,
    t_max: usize,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, min_lr: f32, t_max: usize) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_max,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, epoch: usize) -> f32 {
        use std::f32::consts::PI;
        
        let progress = (epoch % self.t_max) as f32 / self.t_max as f32;
        self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1.0 + (PI * progress).cos())
    }
}