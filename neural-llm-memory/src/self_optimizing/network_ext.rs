//! Extension methods for SelfOptimizingNetwork

use super::SelfOptimizingNetwork;
use ndarray::Array2;

impl SelfOptimizingNetwork {
    
    /// Train the network on a batch
    pub fn train(&mut self, inputs: &Array2<f32>, targets: &Array2<f32>) -> Result<f32, Box<dyn std::error::Error>> {
        // Forward pass
        let mut network = self.best_network.write().unwrap();
        let output = network.forward(inputs, true);
        
        // Calculate loss (MSE for simplicity)
        let diff = &output - targets;
        let loss = (diff.mapv(|x| x * x).sum()) / (diff.len() as f32);
        
        // Backward pass (simplified - just store the loss for adaptation)
        // Note: Full backward pass would need input history which we don't track here
        
        // Online adaptation
        self.online_optimizer.adapt_online(&loss);
        
        Ok(loss)
    }
}