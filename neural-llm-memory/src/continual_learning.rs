//! Continual Learning with Elastic Weight Consolidation and other methods
//! 
//! This module implements various continual learning strategies to prevent
//! catastrophic forgetting when learning new tasks sequentially.

use crate::nn::{Layer, NeuralNetwork, NetworkBuilder};
use crate::memory::{MemoryBank, MemoryKey};
use crate::adaptive::AdaptiveMemoryModule;
use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Continual learning strategy trait
pub trait ContinualLearningStrategy: Send + Sync {
    /// Calculate importance weights for network parameters
    fn calculate_importance(
        &mut self,
        network: &NeuralNetwork,
        task_data: &TaskData,
    ) -> Result<ImportanceWeights, String>;
    
    /// Compute regularization loss for preserving important weights
    fn regularization_loss(
        &self,
        current_params: &NetworkParams,
        previous_params: &NetworkParams,
        importance: &ImportanceWeights,
    ) -> f32;
    
    /// Update strategy after task completion
    fn update_after_task(
        &mut self,
        task_id: usize,
        network: &NeuralNetwork,
        importance: ImportanceWeights,
    );
    
    /// Get strategy name
    fn name(&self) -> &str;
}

/// Elastic Weight Consolidation (EWC) implementation
#[derive(Debug, Clone)]
pub struct ElasticWeightConsolidation {
    /// Lambda parameter for EWC loss strength
    pub lambda: f32,
    /// Number of samples for Fisher Information estimation
    pub fisher_samples: usize,
    /// Stored Fisher information matrices per task
    pub fisher_matrices: HashMap<usize, ImportanceWeights>,
    /// Stored parameters per task
    pub task_params: HashMap<usize, NetworkParams>,
}

impl ElasticWeightConsolidation {
    pub fn new(lambda: f32, fisher_samples: usize) -> Self {
        Self {
            lambda,
            fisher_samples,
            fisher_matrices: HashMap::new(),
            task_params: HashMap::new(),
        }
    }
    
    /// Calculate Fisher Information Matrix using diagonal approximation
    fn calculate_fisher_information(
        &self,
        network: &NeuralNetwork,
        data: &TaskData,
    ) -> Result<ImportanceWeights, String> {
        let mut fisher_weights = ImportanceWeights::new();
        
        // Sample subset of data for Fisher calculation
        let samples = data.sample(self.fisher_samples);
        
        for (input, target) in samples {
            // Forward pass
            let output = network.predict(&input);
            
            // Calculate log-likelihood gradient
            let loss_grad = self.log_likelihood_gradient(&output, &target);
            
            // Backward pass to get parameter gradients
            let param_grads = network.backward(&loss_grad, &[input.clone()]);
            
            // Accumulate squared gradients (diagonal Fisher approximation)
            for (layer_idx, layer) in network.get_layers().iter().enumerate() {
                let layer_params = layer.get_params();
                
                for (param_idx, param) in layer_params.iter().enumerate() {
                    let key = format!("layer_{}_param_{}", layer_idx, param_idx);
                    
                    // Get gradient for this parameter
                    if let Some(grad) = self.extract_param_gradient(&param_grads, layer_idx, param_idx) {
                        let squared_grad = grad.mapv(|x| x * x);
                        
                        fisher_weights.weights
                            .entry(key.clone())
                            .and_modify(|w| *w = &*w + &squared_grad)
                            .or_insert(squared_grad);
                    }
                }
            }
        }
        
        // Average over samples
        let n_samples = samples.len() as f32;
        for weight in fisher_weights.weights.values_mut() {
            *weight = &*weight / n_samples;
        }
        
        Ok(fisher_weights)
    }
    
    fn log_likelihood_gradient(&self, output: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
        // For classification: gradient of negative log-likelihood
        // For regression: can use MSE gradient
        output - target
    }
    
    fn extract_param_gradient(
        &self,
        gradients: &Array2<f32>,
        _layer_idx: usize,
        _param_idx: usize,
    ) -> Option<Array2<f32>> {
        // Extract specific parameter gradient from full gradient
        // This would need proper indexing based on network architecture
        Some(gradients.clone())
    }
}

impl ContinualLearningStrategy for ElasticWeightConsolidation {
    fn calculate_importance(
        &mut self,
        network: &NeuralNetwork,
        task_data: &TaskData,
    ) -> Result<ImportanceWeights, String> {
        self.calculate_fisher_information(network, task_data)
    }
    
    fn regularization_loss(
        &self,
        current_params: &NetworkParams,
        previous_params: &NetworkParams,
        importance: &ImportanceWeights,
    ) -> f32 {
        let mut loss = 0.0;
        
        for (param_name, current_value) in &current_params.params {
            if let Some(prev_value) = previous_params.params.get(param_name) {
                if let Some(fisher) = importance.weights.get(param_name) {
                    // EWC loss: λ/2 * Σ F_i * (θ_i - θ*_i)²
                    let diff = current_value - prev_value;
                    let weighted_diff = fisher * &diff.mapv(|x| x * x);
                    loss += weighted_diff.sum();
                }
            }
        }
        
        self.lambda * loss / 2.0
    }
    
    fn update_after_task(
        &mut self,
        task_id: usize,
        network: &NeuralNetwork,
        importance: ImportanceWeights,
    ) {
        // Store Fisher information for this task
        self.fisher_matrices.insert(task_id, importance);
        
        // Store network parameters after task
        let params = NetworkParams::from_network(network);
        self.task_params.insert(task_id, params);
    }
    
    fn name(&self) -> &str {
        "Elastic Weight Consolidation (EWC)"
    }
}

/// Synaptic Intelligence (SI) - Online importance tracking
#[derive(Debug, Clone)]
pub struct SynapticIntelligence {
    /// Damping parameter for SI
    pub damping: f32,
    /// Online importance accumulator
    pub omega: HashMap<String, Array2<f32>>,
    /// Parameter trajectory tracker
    pub trajectory: ParameterTrajectory,
}

impl SynapticIntelligence {
    pub fn new(damping: f32) -> Self {
        Self {
            damping,
            omega: HashMap::new(),
            trajectory: ParameterTrajectory::new(),
        }
    }
    
    /// Update importance online during training
    pub fn update_online_importance(
        &mut self,
        param_name: &str,
        gradient: &Array2<f32>,
        old_param: &Array2<f32>,
        new_param: &Array2<f32>,
    ) {
        let param_change = new_param - old_param;
        let importance_update = gradient * &param_change;
        
        self.omega
            .entry(param_name.to_string())
            .and_modify(|w| *w = w + &importance_update.mapv(|x| x.abs()))
            .or_insert(importance_update.mapv(|x| x.abs()));
    }
}

impl ContinualLearningStrategy for SynapticIntelligence {
    fn calculate_importance(
        &mut self,
        _network: &NeuralNetwork,
        _task_data: &TaskData,
    ) -> Result<ImportanceWeights, String> {
        // SI tracks importance online, so we return accumulated omega
        let mut importance = ImportanceWeights::new();
        
        for (param_name, omega_value) in &self.omega {
            importance.weights.insert(param_name.clone(), omega_value.clone());
        }
        
        Ok(importance)
    }
    
    fn regularization_loss(
        &self,
        current_params: &NetworkParams,
        previous_params: &NetworkParams,
        importance: &ImportanceWeights,
    ) -> f32 {
        let mut loss = 0.0;
        
        for (param_name, current_value) in &current_params.params {
            if let Some(prev_value) = previous_params.params.get(param_name) {
                if let Some(omega) = importance.weights.get(param_name) {
                    // SI loss: Σ Ω_i * (θ_i - θ*_i)²
                    let diff = current_value - prev_value;
                    let weighted_diff = omega * &diff.mapv(|x| x * x);
                    loss += weighted_diff.sum();
                }
            }
        }
        
        loss
    }
    
    fn update_after_task(
        &mut self,
        _task_id: usize,
        network: &NeuralNetwork,
        _importance: ImportanceWeights,
    ) {
        // Normalize omega values
        for omega in self.omega.values_mut() {
            *omega = &*omega / (omega.sum() + self.damping);
        }
        
        // Update trajectory for next task
        self.trajectory.update(network);
    }
    
    fn name(&self) -> &str {
        "Synaptic Intelligence (SI)"
    }
}

/// Memory Aware Synapses (MAS) - Importance from unlabeled data
#[derive(Debug, Clone)]
pub struct MemoryAwareSynapses {
    /// Regularization strength
    pub lambda: f32,
    /// Number of samples for importance estimation
    pub n_samples: usize,
    /// Stored importance weights
    pub importance_weights: HashMap<String, Array2<f32>>,
}

impl MemoryAwareSynapses {
    pub fn new(lambda: f32, n_samples: usize) -> Self {
        Self {
            lambda,
            n_samples,
            importance_weights: HashMap::new(),
        }
    }
    
    /// Calculate importance using output sensitivity
    fn calculate_output_sensitivity(
        &self,
        network: &NeuralNetwork,
        inputs: &[Array2<f32>],
    ) -> HashMap<String, Array2<f32>> {
        let mut importance = HashMap::new();
        
        for input in inputs.iter().take(self.n_samples) {
            // Get network output
            let output = network.predict(input);
            
            // Calculate L2 norm of output
            let output_norm = output.mapv(|x| x * x).sum().sqrt();
            
            // Approximate importance as gradient of output norm
            // In practice, this would require proper gradient computation
            for (layer_idx, layer) in network.get_layers().iter().enumerate() {
                for (param_idx, param) in layer.get_params().iter().enumerate() {
                    let key = format!("layer_{}_param_{}", layer_idx, param_idx);
                    
                    // Simplified: importance proportional to parameter magnitude
                    let param_importance = param.mapv(|x| x.abs() / (output_norm + 1e-8));
                    
                    importance
                        .entry(key)
                        .and_modify(|w: &mut Array2<f32>| *w = &*w + &param_importance)
                        .or_insert(param_importance);
                }
            }
        }
        
        // Average over samples
        let n = inputs.len().min(self.n_samples) as f32;
        for imp in importance.values_mut() {
            *imp = &*imp / n;
        }
        
        importance
    }
}

impl ContinualLearningStrategy for MemoryAwareSynapses {
    fn calculate_importance(
        &mut self,
        network: &NeuralNetwork,
        task_data: &TaskData,
    ) -> Result<ImportanceWeights, String> {
        // MAS can work with unlabeled data
        let inputs: Vec<_> = task_data.inputs.iter().cloned().collect();
        let importance_map = self.calculate_output_sensitivity(network, &inputs);
        
        let mut importance = ImportanceWeights::new();
        importance.weights = importance_map;
        
        Ok(importance)
    }
    
    fn regularization_loss(
        &self,
        current_params: &NetworkParams,
        previous_params: &NetworkParams,
        importance: &ImportanceWeights,
    ) -> f32 {
        let mut loss = 0.0;
        
        for (param_name, current_value) in &current_params.params {
            if let Some(prev_value) = previous_params.params.get(param_name) {
                if let Some(imp) = importance.weights.get(param_name) {
                    let diff = current_value - prev_value;
                    let weighted_diff = imp * &diff.mapv(|x| x * x);
                    loss += weighted_diff.sum();
                }
            }
        }
        
        self.lambda * loss
    }
    
    fn update_after_task(
        &mut self,
        _task_id: usize,
        _network: &NeuralNetwork,
        importance: ImportanceWeights,
    ) {
        // Store importance weights
        self.importance_weights = importance.weights;
    }
    
    fn name(&self) -> &str {
        "Memory Aware Synapses (MAS)"
    }
}

/// PackNet - Binary weight masks for task isolation
#[derive(Debug, Clone)]
pub struct PackNet {
    /// Pruning percentage per task
    pub prune_percentage: f32,
    /// Binary masks for each task
    pub task_masks: HashMap<usize, NetworkMask>,
    /// Free capacity tracker
    pub free_capacity: NetworkCapacity,
}

impl PackNet {
    pub fn new(prune_percentage: f32, network_shape: &NetworkShape) -> Self {
        Self {
            prune_percentage,
            task_masks: HashMap::new(),
            free_capacity: NetworkCapacity::new(network_shape),
        }
    }
    
    /// Create binary mask for new task
    fn create_task_mask(
        &mut self,
        network: &NeuralNetwork,
        task_id: usize,
    ) -> NetworkMask {
        let mut mask = NetworkMask::new();
        
        // For each layer, allocate subset of free weights
        for (layer_idx, layer) in network.get_layers().iter().enumerate() {
            for (param_idx, param) in layer.get_params().iter().enumerate() {
                let key = format!("layer_{}_param_{}", layer_idx, param_idx);
                
                // Get available capacity for this parameter
                let capacity = self.free_capacity.get_capacity(&key);
                let n_params = param.len();
                let n_available = (n_params as f32 * capacity).round() as usize;
                
                // Create binary mask
                let mut param_mask = Array2::zeros(param.dim());
                if n_available > 0 {
                    // Randomly select weights to use
                    use rand::seq::SliceRandom;
                    let mut rng = rand::thread_rng();
                    let mut indices: Vec<_> = (0..n_params).collect();
                    indices.shuffle(&mut rng);
                    
                    for &idx in indices.iter().take(n_available) {
                        let (i, j) = (idx / param.shape()[1], idx % param.shape()[1]);
                        param_mask[[i, j]] = 1.0;
                    }
                }
                
                mask.masks.insert(key, param_mask);
            }
        }
        
        // Update free capacity
        self.free_capacity.allocate_for_task(&mask, task_id);
        
        mask
    }
}

impl ContinualLearningStrategy for PackNet {
    fn calculate_importance(
        &mut self,
        network: &NeuralNetwork,
        _task_data: &TaskData,
    ) -> Result<ImportanceWeights, String> {
        // PackNet uses binary masks instead of continuous importance
        // Return uniform importance for compatibility
        let mut importance = ImportanceWeights::new();
        
        for (layer_idx, layer) in network.get_layers().iter().enumerate() {
            for (param_idx, param) in layer.get_params().iter().enumerate() {
                let key = format!("layer_{}_param_{}", layer_idx, param_idx);
                importance.weights.insert(key, Array2::ones(param.dim()));
            }
        }
        
        Ok(importance)
    }
    
    fn regularization_loss(
        &self,
        current_params: &NetworkParams,
        previous_params: &NetworkParams,
        _importance: &ImportanceWeights,
    ) -> f32 {
        let mut loss = 0.0;
        
        // For PackNet, ensure weights outside current task mask remain unchanged
        for (task_id, mask) in &self.task_masks {
            for (param_name, param_mask) in &mask.masks {
                if let (Some(current), Some(previous)) = (
                    current_params.params.get(param_name),
                    previous_params.params.get(param_name),
                ) {
                    // Penalize changes to weights not in current task mask
                    let protected_weights = 1.0 - param_mask;
                    let diff = current - previous;
                    let violation = &protected_weights * &diff.mapv(|x| x * x);
                    loss += violation.sum() * 1000.0; // High penalty
                }
            }
        }
        
        loss
    }
    
    fn update_after_task(
        &mut self,
        task_id: usize,
        network: &NeuralNetwork,
        _importance: ImportanceWeights,
    ) {
        // Create and store mask for completed task
        let mask = self.create_task_mask(network, task_id);
        self.task_masks.insert(task_id, mask);
        
        // Prune weights for this task
        self.prune_task_weights(network, task_id);
    }
    
    fn name(&self) -> &str {
        "PackNet"
    }
}

impl PackNet {
    fn prune_task_weights(&self, _network: &NeuralNetwork, _task_id: usize) {
        // Implement magnitude-based pruning within task mask
        // This would modify network weights directly
    }
}

/// Progressive Neural Networks - Add new columns for new tasks
#[derive(Debug)]
pub struct ProgressiveNeuralNetwork {
    /// Base network columns for each task
    pub task_columns: Vec<NeuralNetwork>,
    /// Lateral connections between columns
    pub lateral_connections: HashMap<(usize, usize), LateralConnection>,
    /// Current active task
    pub current_task: usize,
}

impl ProgressiveNeuralNetwork {
    pub fn new(base_architecture: NetworkBuilder) -> Self {
        Self {
            task_columns: vec![base_architecture.build(0.001)],
            lateral_connections: HashMap::new(),
            current_task: 0,
        }
    }
    
    /// Add new column for new task
    pub fn add_task_column(&mut self, architecture: NetworkBuilder) {
        let new_column = architecture.build(0.001);
        
        // Create lateral connections from previous columns
        for prev_task in 0..self.task_columns.len() {
            let connection = LateralConnection::new(
                &self.task_columns[prev_task],
                &new_column,
            );
            
            self.lateral_connections.insert(
                (prev_task, self.task_columns.len()),
                connection,
            );
        }
        
        self.task_columns.push(new_column);
        self.current_task = self.task_columns.len() - 1;
    }
    
    /// Forward pass through progressive network
    pub fn forward(&self, input: &Array2<f32>, task_id: usize) -> Array2<f32> {
        // Get output from target task column
        let mut output = self.task_columns[task_id].predict(input);
        
        // Add lateral connections from previous tasks
        for prev_task in 0..task_id {
            if let Some(lateral) = self.lateral_connections.get(&(prev_task, task_id)) {
                let prev_features = self.task_columns[prev_task].predict(input);
                let lateral_input = lateral.transform(&prev_features);
                output = output + lateral_input;
            }
        }
        
        output
    }
    
    /// Train specific task column
    pub fn train_task(
        &mut self,
        task_id: usize,
        input: &Array2<f32>,
        target: &Array2<f32>,
        loss_fn: impl Fn(&Array2<f32>, &Array2<f32>) -> (f32, Array2<f32>),
    ) -> f32 {
        // Freeze previous task columns
        let output = self.forward(input, task_id);
        let (loss, _grad) = loss_fn(&output, target);
        
        // Only update current task column
        if task_id == self.current_task {
            self.task_columns[task_id].train_step(input, target, loss_fn)
        } else {
            loss
        }
    }
}

/// Lateral connections for progressive networks
#[derive(Debug, Clone)]
pub struct LateralConnection {
    /// Projection weights
    pub weights: Array2<f32>,
    /// Adapter dimension
    pub adapter_dim: usize,
}

impl LateralConnection {
    pub fn new(source_network: &NeuralNetwork, _target_network: &NeuralNetwork) -> Self {
        // Extract feature dimension from source network
        let source_dim = source_network.get_layers()
            .last()
            .and_then(|l| l.output_size())
            .unwrap_or(512);
        
        let adapter_dim = (source_dim as f32 * 0.5).round() as usize;
        
        Self {
            weights: Array2::from_shape_fn(
                (source_dim, adapter_dim),
                |_| rand::random::<f32>() * 0.01,
            ),
            adapter_dim,
        }
    }
    
    pub fn transform(&self, features: &Array2<f32>) -> Array2<f32> {
        features.dot(&self.weights)
    }
}

/// Task boundary detection
#[derive(Debug, Clone)]
pub struct TaskBoundaryDetector {
    /// Distribution shift threshold
    pub shift_threshold: f32,
    /// Window size for statistics
    pub window_size: usize,
    /// Running statistics
    pub stats_buffer: StatisticsBuffer,
}

impl TaskBoundaryDetector {
    pub fn new(shift_threshold: f32, window_size: usize) -> Self {
        Self {
            shift_threshold,
            window_size,
            stats_buffer: StatisticsBuffer::new(window_size),
        }
    }
    
    /// Detect if new task boundary is reached
    pub fn detect_boundary(&mut self, data_batch: &Array2<f32>) -> bool {
        let current_stats = self.calculate_statistics(data_batch);
        
        if let Some(baseline_stats) = self.stats_buffer.get_baseline() {
            let shift = self.calculate_distribution_shift(&current_stats, &baseline_stats);
            
            if shift > self.shift_threshold {
                // Task boundary detected
                self.stats_buffer.reset();
                return true;
            }
        }
        
        self.stats_buffer.update(current_stats);
        false
    }
    
    fn calculate_statistics(&self, data: &Array2<f32>) -> DataStatistics {
        DataStatistics {
            mean: data.mean_axis(Axis(0)).unwrap(),
            std: data.std_axis(Axis(0), 0.0),
            min: data.fold_axis(Axis(0), f32::INFINITY, |&a, &b| a.min(b)),
            max: data.fold_axis(Axis(0), f32::NEG_INFINITY, |&a, &b| a.max(b)),
        }
    }
    
    fn calculate_distribution_shift(
        &self,
        current: &DataStatistics,
        baseline: &DataStatistics,
    ) -> f32 {
        // KL divergence approximation
        let mean_shift = (&current.mean - &baseline.mean).mapv(|x| x * x).sum();
        let std_shift = (&current.std - &baseline.std).mapv(|x| x * x).sum();
        
        (mean_shift + std_shift).sqrt()
    }
}

/// Supporting data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceWeights {
    pub weights: HashMap<String, Array2<f32>>,
}

impl ImportanceWeights {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }
    
    /// Merge importance weights from multiple tasks
    pub fn merge(&mut self, other: &ImportanceWeights, weight: f32) {
        for (param_name, importance) in &other.weights {
            self.weights
                .entry(param_name.clone())
                .and_modify(|w| *w = &*w + weight * importance)
                .or_insert(weight * importance);
        }
    }
}

#[derive(Debug, Clone)]
pub struct NetworkParams {
    pub params: HashMap<String, Array2<f32>>,
}

impl NetworkParams {
    pub fn from_network(network: &NeuralNetwork) -> Self {
        let mut params = HashMap::new();
        
        for (layer_idx, layer) in network.get_layers().iter().enumerate() {
            for (param_idx, param) in layer.get_params().iter().enumerate() {
                let key = format!("layer_{}_param_{}", layer_idx, param_idx);
                params.insert(key, param.clone().clone());
            }
        }
        
        Self { params }
    }
}

#[derive(Debug, Clone)]
pub struct TaskData {
    pub inputs: Vec<Array2<f32>>,
    pub targets: Vec<Array2<f32>>,
    pub task_id: usize,
}

impl TaskData {
    pub fn sample(&self, n: usize) -> Vec<(Array2<f32>, Array2<f32>)> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        
        let mut indices: Vec<_> = (0..self.inputs.len()).collect();
        indices.shuffle(&mut rng);
        
        indices.into_iter()
            .take(n.min(self.inputs.len()))
            .map(|i| (self.inputs[i].clone(), self.targets[i].clone()))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct NetworkMask {
    pub masks: HashMap<String, Array2<f32>>,
}

impl NetworkMask {
    pub fn new() -> Self {
        Self {
            masks: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NetworkCapacity {
    pub capacity: HashMap<String, f32>,
}

impl NetworkCapacity {
    pub fn new(shape: &NetworkShape) -> Self {
        let mut capacity = HashMap::new();
        
        for (param_name, _param_shape) in &shape.shapes {
            capacity.insert(param_name.clone(), 1.0); // Start with full capacity
        }
        
        Self { capacity }
    }
    
    pub fn get_capacity(&self, param_name: &str) -> f32 {
        self.capacity.get(param_name).copied().unwrap_or(0.0)
    }
    
    pub fn allocate_for_task(&mut self, mask: &NetworkMask, _task_id: usize) {
        for (param_name, param_mask) in &mask.masks {
            let used_capacity = param_mask.sum() / param_mask.len() as f32;
            
            self.capacity
                .entry(param_name.clone())
                .and_modify(|c| *c = (*c - used_capacity).max(0.0));
        }
    }
}

#[derive(Debug, Clone)]
pub struct NetworkShape {
    pub shapes: HashMap<String, (usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct ParameterTrajectory {
    pub trajectories: HashMap<String, Vec<Array2<f32>>>,
}

impl ParameterTrajectory {
    pub fn new() -> Self {
        Self {
            trajectories: HashMap::new(),
        }
    }
    
    pub fn update(&mut self, network: &NeuralNetwork) {
        for (layer_idx, layer) in network.get_layers().iter().enumerate() {
            for (param_idx, param) in layer.get_params().iter().enumerate() {
                let key = format!("layer_{}_param_{}", layer_idx, param_idx);
                
                self.trajectories
                    .entry(key)
                    .and_modify(|traj| traj.push(param.clone().clone()))
                    .or_insert_with(|| vec![param.clone().clone()]);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct StatisticsBuffer {
    pub window_size: usize,
    pub buffer: Vec<DataStatistics>,
}

impl StatisticsBuffer {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            buffer: Vec::new(),
        }
    }
    
    pub fn update(&mut self, stats: DataStatistics) {
        self.buffer.push(stats);
        
        if self.buffer.len() > self.window_size {
            self.buffer.remove(0);
        }
    }
    
    pub fn get_baseline(&self) -> Option<DataStatistics> {
        if self.buffer.len() >= self.window_size / 2 {
            // Average statistics over buffer
            let n = self.buffer.len() as f32;
            let mean = self.buffer.iter()
                .map(|s| &s.mean)
                .fold(Array1::zeros(self.buffer[0].mean.len()), |acc, m| acc + m) / n;
            
            let std = self.buffer.iter()
                .map(|s| &s.std)
                .fold(Array1::zeros(self.buffer[0].std.len()), |acc, s| acc + s) / n;
            
            Some(DataStatistics {
                mean,
                std,
                min: self.buffer[0].min.clone(),
                max: self.buffer[0].max.clone(),
            })
        } else {
            None
        }
    }
    
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

#[derive(Debug, Clone)]
pub struct DataStatistics {
    pub mean: Array1<f32>,
    pub std: Array1<f32>,
    pub min: Array1<f32>,
    pub max: Array1<f32>,
}

/// Continual Learning Manager
pub struct ContinualLearningManager {
    /// Current strategy
    pub strategy: Box<dyn ContinualLearningStrategy>,
    /// Task boundary detector
    pub boundary_detector: TaskBoundaryDetector,
    /// Episodic memory for replay
    pub episodic_memory: Arc<RwLock<EpisodicMemory>>,
    /// Current task ID
    pub current_task_id: usize,
    /// Task history
    pub task_history: Vec<TaskInfo>,
}

impl ContinualLearningManager {
    pub fn new(
        strategy: Box<dyn ContinualLearningStrategy>,
        boundary_detector: TaskBoundaryDetector,
    ) -> Self {
        Self {
            strategy,
            boundary_detector,
            episodic_memory: Arc::new(RwLock::new(EpisodicMemory::new(1000))),
            current_task_id: 0,
            task_history: Vec::new(),
        }
    }
    
    /// Train network with continual learning
    pub fn train_continual(
        &mut self,
        network: &mut NeuralNetwork,
        data: &Array2<f32>,
        targets: &Array2<f32>,
        base_loss_fn: impl Fn(&Array2<f32>, &Array2<f32>) -> (f32, Array2<f32>),
    ) -> Result<f32, String> {
        // Check for task boundary
        if self.boundary_detector.detect_boundary(data) {
            self.handle_task_transition(network)?;
        }
        
        // Create augmented loss function with regularization
        let total_loss = if self.current_task_id > 0 {
            // Get previous task parameters and importance
            let prev_params = self.get_previous_params()?;
            let importance = self.get_accumulated_importance()?;
            
            // Standard task loss
            let task_loss = network.train_step(data, targets, &base_loss_fn);
            
            // Regularization loss
            let current_params = NetworkParams::from_network(network);
            let reg_loss = self.strategy.regularization_loss(
                &current_params,
                &prev_params,
                &importance,
            );
            
            task_loss + reg_loss
        } else {
            // First task - no regularization
            network.train_step(data, targets, base_loss_fn)
        };
        
        // Store experiences in episodic memory
        self.store_experiences(data, targets);
        
        Ok(total_loss)
    }
    
    /// Handle transition to new task
    fn handle_task_transition(&mut self, network: &mut NeuralNetwork) -> Result<(), String> {
        // Calculate importance for completed task
        let task_data = self.episodic_memory.read()
            .get_task_data(self.current_task_id)
            .ok_or("No data for current task")?;
        
        let importance = self.strategy.calculate_importance(network, &task_data)?;
        
        // Update strategy with task completion
        self.strategy.update_after_task(self.current_task_id, network, importance.clone());
        
        // Store task info
        self.task_history.push(TaskInfo {
            task_id: self.current_task_id,
            importance,
            final_params: NetworkParams::from_network(network),
            completion_time: std::time::SystemTime::now(),
        });
        
        // Increment task ID
        self.current_task_id += 1;
        
        Ok(())
    }
    
    /// Experience replay for memory consolidation
    pub fn replay_experiences(
        &self,
        network: &mut NeuralNetwork,
        n_samples: usize,
        loss_fn: impl Fn(&Array2<f32>, &Array2<f32>) -> (f32, Array2<f32>),
    ) -> Result<f32, String> {
        let experiences = self.episodic_memory.read()
            .sample_experiences(n_samples);
        
        let mut total_loss = 0.0;
        
        for (input, target) in experiences {
            let loss = network.train_step(&input, &target, &loss_fn);
            total_loss += loss;
        }
        
        Ok(total_loss / n_samples as f32)
    }
    
    fn get_previous_params(&self) -> Result<NetworkParams, String> {
        self.task_history.last()
            .map(|info| info.final_params.clone())
            .ok_or("No previous task parameters".to_string())
    }
    
    fn get_accumulated_importance(&self) -> Result<ImportanceWeights, String> {
        let mut accumulated = ImportanceWeights::new();
        
        for task_info in &self.task_history {
            accumulated.merge(&task_info.importance, 1.0);
        }
        
        Ok(accumulated)
    }
    
    fn store_experiences(&self, inputs: &Array2<f32>, targets: &Array2<f32>) {
        self.episodic_memory.write().store_batch(
            inputs,
            targets,
            self.current_task_id,
        );
    }
}

/// Episodic memory for experience replay
#[derive(Debug)]
pub struct EpisodicMemory {
    pub capacity: usize,
    pub experiences: Vec<Experience>,
    pub task_experiences: HashMap<usize, Vec<usize>>,
}

impl EpisodicMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            experiences: Vec::new(),
            task_experiences: HashMap::new(),
        }
    }
    
    pub fn store_batch(&mut self, inputs: &Array2<f32>, targets: &Array2<f32>, task_id: usize) {
        for i in 0..inputs.shape()[0] {
            let exp = Experience {
                input: inputs.row(i).to_owned().insert_axis(Axis(0)),
                target: targets.row(i).to_owned().insert_axis(Axis(0)),
                task_id,
            };
            
            let exp_idx = self.experiences.len();
            self.experiences.push(exp);
            
            self.task_experiences
                .entry(task_id)
                .and_modify(|indices| indices.push(exp_idx))
                .or_insert_with(|| vec![exp_idx]);
            
            // Maintain capacity
            if self.experiences.len() > self.capacity {
                self.experiences.remove(0);
                // Update indices
                for indices in self.task_experiences.values_mut() {
                    indices.retain(|&idx| idx > 0);
                    for idx in indices.iter_mut() {
                        *idx -= 1;
                    }
                }
            }
        }
    }
    
    pub fn sample_experiences(&self, n: usize) -> Vec<(Array2<f32>, Array2<f32>)> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        
        self.experiences
            .choose_multiple(&mut rng, n)
            .map(|exp| (exp.input.clone(), exp.target.clone()))
            .collect()
    }
    
    pub fn get_task_data(&self, task_id: usize) -> Option<TaskData> {
        self.task_experiences.get(&task_id).map(|indices| {
            let experiences: Vec<_> = indices.iter()
                .filter_map(|&idx| self.experiences.get(idx))
                .collect();
            
            TaskData {
                inputs: experiences.iter().map(|e| e.input.clone()).collect(),
                targets: experiences.iter().map(|e| e.target.clone()).collect(),
                task_id,
            }
        })
    }
}

#[derive(Debug, Clone)]
pub struct Experience {
    pub input: Array2<f32>,
    pub target: Array2<f32>,
    pub task_id: usize,
}

#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub task_id: usize,
    pub importance: ImportanceWeights,
    pub final_params: NetworkParams,
    pub completion_time: std::time::SystemTime,
}

/// Integration with AdaptiveMemoryModule (TODO: Complete implementation)
impl AdaptiveMemoryModule {
    /// Enable continual learning for the neural network
    pub async fn enable_continual_learning(
        &mut self,
        strategy: Box<dyn ContinualLearningStrategy>,
    ) -> Result<(), String> {
        // Store continual learning configuration
        self.store(
            "continual_learning/strategy",
            &format!("Enabled: {}", strategy.name()),
        ).await.map_err(|e| format!("Failed to store strategy: {}", e))?;
        
        // Initialize importance weights in adaptive state
        self.store(
            "continual_learning/importance/initialized",
            "true",
        ).await.map_err(|e| format!("Failed to initialize: {}", e))?;
        
        Ok(())
    }
    
    /// Store Fisher information for a task
    pub async fn store_fisher_information(
        &mut self,
        task_id: usize,
        importance: &ImportanceWeights,
    ) -> Result<(), String> {
        let key = format!("continual_learning/fisher/task_{}", task_id);
        let serialized = serde_json::to_string(&importance)
            .map_err(|e| format!("Failed to serialize importance: {}", e))?;
        
        self.store(&key, &serialized).await.map_err(|e| format!("Failed to store Fisher info: {}", e))?;
        Ok(())
    }
    
    /// Retrieve accumulated importance weights (TODO: Complete implementation)
    pub async fn get_importance_weights(&self) -> Result<ImportanceWeights, String> {
        let accumulated = ImportanceWeights::new();
        
        // TODO: Implement retrieval of stored Fisher information
        // This would require iterating through stored tasks and merging importance weights
        
        Ok(accumulated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ewc_fisher_calculation() {
        let ewc = ElasticWeightConsolidation::new(0.1, 100);
        // Add test implementation
    }
    
    #[test]
    fn test_task_boundary_detection() {
        let mut detector = TaskBoundaryDetector::new(0.5, 100);
        // Add test implementation
    }
    
    #[test]
    fn test_progressive_network() {
        let builder = NetworkBuilder::new();
        let mut prog_net = ProgressiveNeuralNetwork::new(builder);
        // Add test implementation
    }
}