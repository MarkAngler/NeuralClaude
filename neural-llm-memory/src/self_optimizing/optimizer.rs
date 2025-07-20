//! Online optimization for runtime architecture adaptation

use std::sync::{Arc, RwLock};
use std::collections::VecDeque;
use crate::nn::NeuralNetwork;

/// Online optimizer for runtime adaptation
pub struct OnlineOptimizer {
    /// Reference to network being optimized
    network: Arc<RwLock<NeuralNetwork>>,
    
    /// Adaptation strategy
    strategy: AdaptationStrategy,
    
    /// Performance monitor
    monitor: PerformanceMonitor,
    
    /// Adaptation history
    history: AdaptationHistory,
}

/// Strategy for online adaptation
#[derive(Debug, Clone)]
pub struct AdaptationStrategy {
    /// Enable gradient flow monitoring
    pub monitor_gradients: bool,
    
    /// Enable overfitting detection
    pub detect_overfitting: bool,
    
    /// Enable plateau detection
    pub detect_plateau: bool,
    
    /// Enable memory pressure handling
    pub handle_memory_pressure: bool,
    
    /// Gradient vanishing threshold
    pub gradient_threshold: f32,
    
    /// Overfitting detection window
    pub overfitting_window: usize,
    
    /// Plateau detection window
    pub plateau_window: usize,
}

impl Default for AdaptationStrategy {
    fn default() -> Self {
        Self {
            monitor_gradients: true,
            detect_overfitting: true,
            detect_plateau: true,
            handle_memory_pressure: true,
            gradient_threshold: 1e-6,
            overfitting_window: 10,
            plateau_window: 20,
        }
    }
}

/// Performance monitoring for adaptation decisions
struct PerformanceMonitor {
    /// Training loss history
    train_loss_history: VecDeque<f32>,
    
    /// Validation loss history
    val_loss_history: VecDeque<f32>,
    
    /// Gradient magnitude history
    gradient_history: VecDeque<f32>,
    
    /// Learning rate history
    lr_history: VecDeque<f32>,
    
    /// Memory usage history
    memory_history: VecDeque<f32>,
    
    /// Configuration
    config: MonitorConfig,
}

#[derive(Debug, Clone)]
struct MonitorConfig {
    /// Maximum history length
    max_history: usize,
    
    /// Gradient vanishing threshold
    gradient_threshold: f32,
    
    /// Overfitting ratio threshold
    overfitting_threshold: f32,
    
    /// Plateau improvement threshold
    plateau_threshold: f32,
    
    /// Memory pressure threshold (MB)
    memory_threshold: f32,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            max_history: 100,
            gradient_threshold: 1e-6,
            overfitting_threshold: 1.2,
            plateau_threshold: 0.001,
            memory_threshold: 900.0,
        }
    }
}

/// History of adaptations made
struct AdaptationHistory {
    /// List of adaptations
    adaptations: Vec<AdaptationEvent>,
    
    /// Success rate of adaptations
    success_rate: f32,
}

#[derive(Debug, Clone)]
struct AdaptationEvent {
    /// Timestamp
    timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Type of adaptation
    adaptation_type: AdaptationType,
    
    /// Performance before adaptation
    performance_before: f32,
    
    /// Performance after adaptation
    performance_after: Option<f32>,
    
    /// Success flag
    success: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum AdaptationType {
    AddSkipConnection,
    IncreaseRegularization,
    ApplyPruning,
    AdjustLearningRate,
    PerturbWeights,
    CompressNetwork,
    ExpandCapacity,
}

impl OnlineOptimizer {
    /// Create a new online optimizer
    pub fn new(
        network: Arc<RwLock<NeuralNetwork>>,
        strategy: AdaptationStrategy,
    ) -> Self {
        Self {
            network,
            strategy,
            monitor: PerformanceMonitor::new(),
            history: AdaptationHistory {
                adaptations: Vec::new(),
                success_rate: 0.0,
            },
        }
    }
    
    /// Adapt during training epoch
    pub fn adapt_during_epoch(&mut self, epoch: usize) {
        // Update monitoring data
        self.update_monitoring_data();
        
        // Check for various conditions and adapt
        if self.strategy.monitor_gradients && self.monitor.detect_vanishing_gradients() {
            self.handle_vanishing_gradients();
        }
        
        if self.strategy.detect_overfitting && self.monitor.detect_overfitting() {
            self.handle_overfitting();
        }
        
        if self.strategy.detect_plateau && self.monitor.detect_plateau() {
            self.handle_plateau();
        }
        
        if self.strategy.handle_memory_pressure && self.monitor.memory_pressure() {
            self.handle_memory_pressure();
        }
        
        // Periodic capacity adjustment
        if epoch % 50 == 0 {
            self.consider_capacity_adjustment();
        }
    }
    
    /// Update monitoring data
    fn update_monitoring_data(&mut self) {
        let network = self.network.read().unwrap();
        
        // Get current metrics from network
        if let Some(metrics) = network.get_current_metrics() {
            self.monitor.update_train_loss(metrics.train_loss);
            self.monitor.update_val_loss(metrics.val_loss);
            self.monitor.update_gradient_norm(metrics.gradient_norm);
            self.monitor.update_learning_rate(network.get_learning_rate());
        }
        
        // Update memory usage
        self.monitor.update_memory_usage(self.estimate_memory_usage(&network));
    }
    
    /// Handle vanishing gradients
    fn handle_vanishing_gradients(&mut self) {
        println!("Detected vanishing gradients - adding skip connections");
        
        let event = AdaptationEvent {
            timestamp: chrono::Utc::now(),
            adaptation_type: AdaptationType::AddSkipConnection,
            performance_before: self.get_current_performance(),
            performance_after: None,
            success: false,
        };
        
        // Add skip connections to network
        // In a real implementation, this would modify the network structure
        // For now, we'll just log the attempt
        
        self.history.adaptations.push(event);
    }
    
    /// Handle overfitting
    fn handle_overfitting(&mut self) {
        println!("Detected overfitting - increasing regularization");
        
        let mut network = self.network.write().unwrap();
        
        // Increase dropout rates
        network.increase_dropout(0.1);
        
        // Add L2 regularization
        let current_decay = network.get_weight_decay();
        network.set_weight_decay(current_decay * 1.5);
        
        let event = AdaptationEvent {
            timestamp: chrono::Utc::now(),
            adaptation_type: AdaptationType::IncreaseRegularization,
            performance_before: self.get_current_performance(),
            performance_after: None,
            success: false,
        };
        
        self.history.adaptations.push(event);
    }
    
    /// Handle training plateau
    fn handle_plateau(&mut self) {
        println!("Detected plateau - adjusting learning rate and perturbing weights");
        
        let mut network = self.network.write().unwrap();
        
        // Reduce learning rate
        let current_lr = network.get_learning_rate();
        network.set_learning_rate(current_lr * 0.5);
        
        // Add small perturbation to weights
        network.perturb_weights(0.01);
        
        let event = AdaptationEvent {
            timestamp: chrono::Utc::now(),
            adaptation_type: AdaptationType::AdjustLearningRate,
            performance_before: self.get_current_performance(),
            performance_after: None,
            success: false,
        };
        
        self.history.adaptations.push(event);
    }
    
    /// Handle memory pressure
    fn handle_memory_pressure(&mut self) {
        println!("Detected memory pressure - compressing network");
        
        let event = AdaptationEvent {
            timestamp: chrono::Utc::now(),
            adaptation_type: AdaptationType::CompressNetwork,
            performance_before: self.get_current_performance(),
            performance_after: None,
            success: false,
        };
        
        // Apply pruning or quantization
        // In a real implementation, this would compress the network
        
        self.history.adaptations.push(event);
    }
    
    /// Consider adjusting network capacity
    fn consider_capacity_adjustment(&mut self) {
        // Check if network is underperforming
        let recent_improvements = self.monitor.get_recent_improvement();
        
        if recent_improvements < 0.001 {
            // Consider expanding capacity
            println!("Considering capacity expansion due to slow improvement");
            
            let event = AdaptationEvent {
                timestamp: chrono::Utc::now(),
                adaptation_type: AdaptationType::ExpandCapacity,
                performance_before: self.get_current_performance(),
                performance_after: None,
                success: false,
            };
            
            self.history.adaptations.push(event);
        }
    }
    
    /// Get current performance metric
    fn get_current_performance(&self) -> f32 {
        self.monitor.val_loss_history.back().cloned().unwrap_or(f32::INFINITY)
    }
    
    /// Estimate memory usage
    fn estimate_memory_usage(&self, network: &NeuralNetwork) -> f32 {
        // Simple estimation based on parameter count
        let param_count = network.get_layers().iter()
            .map(|l| l.count_parameters())
            .sum::<usize>();
        
        (param_count * 4) as f32 / (1024.0 * 1024.0) // Convert to MB
    }
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            train_loss_history: VecDeque::with_capacity(100),
            val_loss_history: VecDeque::with_capacity(100),
            gradient_history: VecDeque::with_capacity(100),
            lr_history: VecDeque::with_capacity(100),
            memory_history: VecDeque::with_capacity(100),
            config: MonitorConfig::default(),
        }
    }
    
    fn update_train_loss(&mut self, loss: f32) {
        self.train_loss_history.push_back(loss);
        if self.train_loss_history.len() > self.config.max_history {
            self.train_loss_history.pop_front();
        }
    }
    
    fn update_val_loss(&mut self, loss: f32) {
        self.val_loss_history.push_back(loss);
        if self.val_loss_history.len() > self.config.max_history {
            self.val_loss_history.pop_front();
        }
    }
    
    fn update_gradient_norm(&mut self, norm: f32) {
        self.gradient_history.push_back(norm);
        if self.gradient_history.len() > self.config.max_history {
            self.gradient_history.pop_front();
        }
    }
    
    fn update_learning_rate(&mut self, lr: f32) {
        self.lr_history.push_back(lr);
        if self.lr_history.len() > self.config.max_history {
            self.lr_history.pop_front();
        }
    }
    
    fn update_memory_usage(&mut self, memory_mb: f32) {
        self.memory_history.push_back(memory_mb);
        if self.memory_history.len() > self.config.max_history {
            self.memory_history.pop_front();
        }
    }
    
    pub fn detect_vanishing_gradients(&self) -> bool {
        if self.gradient_history.len() < 5 {
            return false;
        }
        
        // Check recent gradients
        let recent_gradients: Vec<f32> = self.gradient_history.iter()
            .rev()
            .take(5)
            .cloned()
            .collect();
        
        let avg_gradient = recent_gradients.iter().sum::<f32>() / recent_gradients.len() as f32;
        
        avg_gradient < self.config.gradient_threshold
    }
    
    pub fn detect_overfitting(&self) -> bool {
        if self.train_loss_history.len() < 10 || self.val_loss_history.len() < 10 {
            return false;
        }
        
        // Compare recent train vs validation loss
        let recent_train: Vec<f32> = self.train_loss_history.iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        
        let recent_val: Vec<f32> = self.val_loss_history.iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        
        let avg_train = recent_train.iter().sum::<f32>() / recent_train.len() as f32;
        let avg_val = recent_val.iter().sum::<f32>() / recent_val.len() as f32;
        
        // Check if validation loss is significantly higher than train loss
        avg_val > avg_train * self.config.overfitting_threshold
    }
    
    pub fn detect_plateau(&self) -> bool {
        if self.val_loss_history.len() < 20 {
            return false;
        }
        
        // Check improvement over last N epochs
        let old_loss = self.val_loss_history[self.val_loss_history.len() - 20];
        let new_loss = self.val_loss_history.back().unwrap();
        
        let improvement = (old_loss - new_loss).abs();
        
        improvement < self.config.plateau_threshold
    }
    
    pub fn memory_pressure(&self) -> bool {
        if let Some(&current_memory) = self.memory_history.back() {
            current_memory > self.config.memory_threshold
        } else {
            false
        }
    }
    
    fn get_recent_improvement(&self) -> f32 {
        if self.val_loss_history.len() < 10 {
            return 1.0; // Assume good improvement initially
        }
        
        let old = self.val_loss_history[self.val_loss_history.len() - 10];
        let new = self.val_loss_history.back().unwrap();
        
        (old - new).abs()
    }
}

/// Learning rate scheduler for adaptive training
pub struct AdaptiveLRScheduler {
    /// Base learning rate
    base_lr: f32,
    
    /// Current learning rate
    current_lr: f32,
    
    /// Scheduler type
    scheduler_type: LRSchedulerType,
    
    /// Step count
    step: usize,
}

#[derive(Debug, Clone)]
enum LRSchedulerType {
    /// Cosine annealing
    CosineAnnealing { total_steps: usize },
    
    /// Step decay
    StepDecay { step_size: usize, gamma: f32 },
    
    /// Exponential decay
    ExponentialDecay { gamma: f32 },
    
    /// Plateau-based reduction
    ReduceOnPlateau { patience: usize, factor: f32 },
}

impl AdaptiveLRScheduler {
    pub fn new(base_lr: f32, scheduler_type: LRSchedulerType) -> Self {
        Self {
            base_lr,
            current_lr: base_lr,
            scheduler_type,
            step: 0,
        }
    }
    
    pub fn step(&mut self, metrics: Option<f32>) -> f32 {
        self.step += 1;
        
        match &self.scheduler_type {
            LRSchedulerType::CosineAnnealing { total_steps } => {
                self.current_lr = self.base_lr * 0.5 * (1.0 + (std::f32::consts::PI * self.step as f32 / *total_steps as f32).cos());
            }
            LRSchedulerType::StepDecay { step_size, gamma } => {
                if self.step % step_size == 0 {
                    self.current_lr *= gamma;
                }
            }
            LRSchedulerType::ExponentialDecay { gamma } => {
                self.current_lr = self.base_lr * gamma.powf(self.step as f32);
            }
            LRSchedulerType::ReduceOnPlateau { .. } => {
                // This requires tracking metrics history
                // Simplified implementation
                if let Some(_metric) = metrics {
                    // Implementation would track history and reduce on plateau
                }
            }
        }
        
        self.current_lr
    }
    
    pub fn get_lr(&self) -> f32 {
        self.current_lr
    }
}

