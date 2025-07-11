//! Cognitive monitoring subsystem
//! 
//! Tracks and analyzes thinking patterns, attention allocation,
//! and decision-making processes in real-time.

use crate::memory::MemoryKey;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use ndarray::{Array1, Array2};

// Stub types for compilation (TODO: Move to proper modules)
#[derive(Debug, Clone)]
pub struct CognitivePattern {
    pub id: String,
    pub name: String,
    pub effectiveness: f32,
    pub usage_count: usize,
    pub last_used: chrono::DateTime<chrono::Utc>,
    pub effectiveness_history: std::collections::VecDeque<f32>,
    pub context_success_map: std::collections::HashMap<String, f32>,
    pub pattern_type: String,
}

#[derive(Debug, Clone)]
pub struct AttentionSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub attention_weights: std::collections::HashMap<String, f32>,
    pub focus_target: String,
    pub intensity: f32,
    pub duration: f32,
}

#[derive(Debug, Clone)]
pub struct Decision {
    pub id: String,
    pub timestamp: u64,
    pub context: String,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct MetaCognitiveMetrics {
    pub self_awareness: f32,
    pub bias_detection_rate: f32,
    pub strategy_effectiveness: f32,
    pub decisions_made: usize,
    pub average_confidence: f32,
    pub bias_occurrences: std::collections::HashMap<String, usize>,
    pub self_awareness_score: f32,
}

#[derive(Debug, Clone)]
pub enum BiasType {
    Confirmation,
    Anchoring,
    Availability,
    DunningKruger,
    Recency,
    Apophenia,
}
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

/// Metrics for pattern usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMetrics {
    pub pattern_id: String,
    pub effectiveness_score: f32,
    pub usage_frequency: f32,
    pub context_appropriateness: f32,
    pub resource_efficiency: f32,
    pub success_rate: f32,
}

/// Metrics for attention allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionMetrics {
    pub focus_stability: f32,
    pub coverage_breadth: f32,
    pub switching_frequency: f32,
    pub attention_entropy: f32,
    pub peak_focus_duration: Duration,
    pub attention_distribution: HashMap<String, f32>,
}

/// Cognitive monitoring implementation
pub struct CognitiveMonitor {
    /// Pattern usage tracking
    pattern_usage: Arc<RwLock<HashMap<String, CognitivePattern>>>,
    
    /// Attention history circular buffer
    attention_history: Arc<RwLock<VecDeque<AttentionSnapshot>>>,
    
    /// Decision trace for analysis
    decision_trace: Arc<RwLock<VecDeque<Decision>>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<MetaCognitiveMetrics>>,
    
    /// Monitoring configuration
    config: MonitorConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    pub history_size: usize,
    pub attention_window: usize,
    pub pattern_decay_rate: f32,
    pub metrics_update_interval: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            history_size: 1000,
            attention_window: 100,
            pattern_decay_rate: 0.95,
            metrics_update_interval: 10,
        }
    }
}

impl CognitiveMonitor {
    /// Create new cognitive monitor
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            pattern_usage: Arc::new(RwLock::new(HashMap::new())),
            attention_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.history_size))),
            decision_trace: Arc::new(RwLock::new(VecDeque::with_capacity(config.history_size))),
            metrics: Arc::new(RwLock::new(MetaCognitiveMetrics::default())),
            config,
        }
    }
    
    /// Monitor pattern usage and effectiveness
    pub fn monitor_pattern_usage(&self, pattern: &CognitivePattern) -> PatternMetrics {
        let mut usage_map = self.pattern_usage.write().unwrap();
        
        // Update or insert pattern
        let stored_pattern = usage_map.entry(pattern.id.clone())
            .or_insert_with(|| CognitivePattern {
                id: pattern.id.clone(),
                name: pattern.name.clone(),
                effectiveness: pattern.effectiveness,
                usage_count: 0,
                last_used: chrono::Utc::now(),
                effectiveness_history: std::collections::VecDeque::new(),
                context_success_map: std::collections::HashMap::new(),
                pattern_type: pattern.pattern_type.clone(),
            });
        
        // Update usage count
        stored_pattern.usage_count += 1;
        stored_pattern.last_used = chrono::Utc::now();
        
        // Calculate metrics
        let effectiveness_score = self.calculate_effectiveness(stored_pattern);
        let usage_frequency = self.calculate_usage_frequency(stored_pattern);
        let context_appropriateness = self.evaluate_context_fit(pattern);
        let resource_efficiency = self.measure_resource_efficiency(pattern);
        let success_rate = self.calculate_success_rate(stored_pattern);
        
        // Apply decay to historical effectiveness
        if stored_pattern.effectiveness_history.len() >= 100 {
            stored_pattern.effectiveness_history.pop_front();
        }
        stored_pattern.effectiveness_history.push_back(effectiveness_score);
        
        PatternMetrics {
            pattern_id: pattern.id.clone(),
            effectiveness_score,
            usage_frequency,
            context_appropriateness,
            resource_efficiency,
            success_rate,
        }
    }
    
    /// Track attention allocation across memories
    pub fn track_attention_allocation(&self, memory_accesses: &[MemoryAccess]) -> AttentionMetrics {
        let mut attention_history = self.attention_history.write().unwrap();
        
        // Create attention snapshot
        let mut attention_weights = HashMap::new();
        for access in memory_accesses {
            *attention_weights.entry(access.key.clone()).or_insert(0.0) += access.attention_weight;
        }
        
        // Normalize weights
        let total_weight: f32 = attention_weights.values().sum();
        if total_weight > 0.0 {
            for weight in attention_weights.values_mut() {
                *weight /= total_weight;
            }
        }
        
        // Find focus target (highest attention)
        let focus_target = attention_weights.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k.clone());
        
        let snapshot = AttentionSnapshot {
            timestamp: chrono::Utc::now(),
            attention_weights: attention_weights.clone(),
            focus_target: focus_target.unwrap_or_default(),
            intensity: 0.8,
            duration: 1.0,
        };
        
        // Add to history
        if attention_history.len() >= self.config.attention_window {
            attention_history.pop_front();
        }
        attention_history.push_back(snapshot);
        
        // Calculate metrics
        self.calculate_attention_metrics(&attention_history)
    }
    
    /// Calculate attention metrics from history
    fn calculate_attention_metrics(&self, history: &VecDeque<AttentionSnapshot>) -> AttentionMetrics {
        if history.is_empty() {
            return AttentionMetrics::default();
        }
        
        // Focus stability - how consistent is the focus target
        let focus_stability = self.calculate_focus_stability(history);
        
        // Coverage breadth - how many different items receive attention
        let coverage_breadth = self.calculate_coverage_breadth(history);
        
        // Switching frequency - how often focus changes
        let switching_frequency = self.calculate_switching_frequency(history);
        
        // Attention entropy - measure of attention distribution
        let attention_entropy = self.calculate_attention_entropy(history);
        
        // Peak focus duration
        let peak_focus_duration = self.calculate_peak_focus_duration(history);
        
        // Average attention distribution
        let attention_distribution = self.calculate_average_distribution(history);
        
        AttentionMetrics {
            focus_stability,
            coverage_breadth,
            switching_frequency,
            attention_entropy,
            peak_focus_duration,
            attention_distribution,
        }
    }
    
    /// Calculate focus stability metric
    fn calculate_focus_stability(&self, history: &VecDeque<AttentionSnapshot>) -> f32 {
        if history.len() < 2 {
            return 1.0;
        }
        
        let mut stable_count = 0;
        let mut prev_focus = &history[0].focus_target;
        
        for snapshot in history.iter().skip(1) {
            if snapshot.focus_target == *prev_focus {
                stable_count += 1;
            }
            prev_focus = &snapshot.focus_target;
        }
        
        stable_count as f32 / (history.len() - 1) as f32
    }
    
    /// Calculate attention entropy
    fn calculate_attention_entropy(&self, history: &VecDeque<AttentionSnapshot>) -> f32 {
        let mut combined_weights: HashMap<String, f32> = HashMap::new();
        
        for snapshot in history {
            for (key, weight) in &snapshot.attention_weights {
                *combined_weights.entry(key.clone()).or_insert(0.0) += weight;
            }
        }
        
        // Normalize
        let total: f32 = combined_weights.values().sum();
        if total == 0.0 {
            return 0.0;
        }
        
        // Calculate entropy
        let mut entropy = 0.0;
        for weight in combined_weights.values() {
            let p = weight / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        
        entropy
    }
    
    /// Analyze decision trace for patterns
    pub fn analyze_trace(&self, trace: &[Decision]) -> Vec<ThinkingPattern> {
        let mut patterns = Vec::new();
        
        // Sequential pattern mining
        for window_size in 2..=5 {
            if trace.len() >= window_size {
                for i in 0..=trace.len() - window_size {
                    let window = &trace[i..i + window_size];
                    if let Some(pattern) = self.extract_pattern(window) {
                        patterns.push(pattern);
                    }
                }
            }
        }
        
        // Frequency analysis
        let frequency_patterns = self.analyze_decision_frequencies(trace);
        patterns.extend(frequency_patterns);
        
        // Temporal patterns
        let temporal_patterns = self.analyze_temporal_patterns(trace);
        patterns.extend(temporal_patterns);
        
        patterns
    }
    
    /// Update performance metrics
    pub fn update_metrics(&self, output: &ThinkingOutput) {
        let mut metrics = self.metrics.write().unwrap();
        
        metrics.decisions_made += output.decision_trace.len();
        
        // Update average confidence
        let total_confidence: f32 = output.decision_trace.iter()
            .map(|d| d.confidence)
            .sum();
        let new_avg = if output.decision_trace.is_empty() {
            metrics.average_confidence
        } else {
            let count = metrics.decisions_made as f32;
            (metrics.average_confidence * (count - output.decision_trace.len() as f32) + total_confidence) / count
        };
        metrics.average_confidence = new_avg;
        
        // Update bias occurrences
        for bias in &output.biases_detected {
            *metrics.bias_occurrences.entry(bias.bias_type.clone()).or_insert(0) += 1;
        }
        
        // Update strategy effectiveness - use the single f32 field
        let effectiveness = output.confidence * if output.result.contains("success") { 1.0 } else { 0.5 };
        metrics.strategy_effectiveness = (metrics.strategy_effectiveness + effectiveness) / 2.0;
        
        // Calculate self-awareness score
        metrics.self_awareness_score = self.calculate_self_awareness_score(&metrics);
    }
    
    /// Calculate self-awareness score based on metrics
    fn calculate_self_awareness_score(&self, metrics: &MetaCognitiveMetrics) -> f32 {
        let mut score = 0.0;
        let mut components = 0;
        
        // Component 1: Confidence calibration
        if metrics.decisions_made > 0 {
            let confidence_calibration = 1.0 - (metrics.average_confidence - 0.7).abs();
            score += confidence_calibration;
            components += 1;
        }
        
        // Component 2: Bias awareness (detecting own biases)
        if !metrics.bias_occurrences.is_empty() {
            let bias_awareness = metrics.bias_occurrences.len() as f32 / 6.0; // 6 bias types
            score += bias_awareness.min(1.0);
            components += 1;
        }
        
        // Component 3: Strategy adaptability
        let strategy_variance = (metrics.strategy_effectiveness - 0.5).powi(2);
        let adaptability = 1.0 - strategy_variance.sqrt();
        score += adaptability;
        components += 1;
        
        if components > 0 {
            score / components as f32
        } else {
            0.5 // neutral score
        }
    }
    
    /// Capture current cognitive state
    pub fn capture_state(&self) -> CognitiveState {
        let patterns = self.pattern_usage.read().unwrap();
        let attention = self.attention_history.read().unwrap();
        let decisions = self.decision_trace.read().unwrap();
        let metrics = self.metrics.read().unwrap();
        
        CognitiveState {
            active_patterns: patterns.values()
                .filter(|p| (chrono::Utc::now() - p.last_used).num_seconds() < 300)
                .cloned()
                .collect(),
            current_attention: attention.back().cloned(),
            recent_decisions: decisions.iter().rev().take(10).cloned().collect(),
            performance_metrics: (*metrics).clone(),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Thinking pattern extracted from decision trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingPattern {
    pub pattern_type: String,
    pub occurrences: usize,
    pub average_confidence: f32,
    pub temporal_signature: Vec<f32>,
}

/// Current cognitive state snapshot
#[derive(Debug, Clone)]
pub struct CognitiveState {
    pub active_patterns: Vec<CognitivePattern>,
    pub current_attention: Option<AttentionSnapshot>,
    pub recent_decisions: Vec<Decision>,
    pub performance_metrics: MetaCognitiveMetrics,
    pub timestamp: DateTime<Utc>,
}

/// Default implementation for AttentionMetrics
impl Default for AttentionMetrics {
    fn default() -> Self {
        Self {
            focus_stability: 0.5,
            coverage_breadth: 0.5,
            switching_frequency: 0.1,
            attention_entropy: 1.0,
            peak_focus_duration: Duration::seconds(0),
            attention_distribution: HashMap::new(),
        }
    }
}

/// Default implementation for MetaCognitiveMetrics
impl Default for MetaCognitiveMetrics {
    fn default() -> Self {
        Self {
            self_awareness: 0.5,
            bias_detection_rate: 0.5,
            strategy_effectiveness: 0.5,
            decisions_made: 0,
            average_confidence: 0.5,
            bias_occurrences: HashMap::new(),
            self_awareness_score: 0.5,
        }
    }
}

// Import required types
// Stub type for compilation
#[derive(Debug, Clone)]
pub struct ThinkingOutput {
    pub result: String,
    pub confidence: f32,
    pub decision_trace: Vec<Decision>,
    pub biases_detected: Vec<BiasDetection>,
    pub strategy_used: String,
}

#[derive(Debug, Clone)]
pub struct BiasDetection {
    pub bias_type: String,
    pub confidence: f32,
}

/// Memory access information
#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub key: String,
    pub attention_weight: f32,
    pub access_type: AccessType,
}

#[derive(Debug, Clone)]
pub enum AccessType {
    Read,
    Write,
    Search,
}

// Helper implementations for the monitor
impl CognitiveMonitor {
    fn calculate_effectiveness(&self, pattern: &CognitivePattern) -> f32 {
        if pattern.effectiveness_history.is_empty() {
            return 0.5;
        }
        
        // Weighted average with recency bias
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for (i, &effectiveness) in pattern.effectiveness_history.iter().enumerate() {
            let weight = (i + 1) as f32 / pattern.effectiveness_history.len() as f32;
            weighted_sum += effectiveness * weight;
            weight_sum += weight;
        }
        
        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.5
        }
    }
    
    fn calculate_usage_frequency(&self, pattern: &CognitivePattern) -> f32 {
        let hours_since_first_use = 24.0; // Placeholder
        if hours_since_first_use > 0.0 {
            pattern.usage_count as f32 / hours_since_first_use
        } else {
            0.0
        }
    }
    
    fn evaluate_context_fit(&self, pattern: &CognitivePattern) -> f32 {
        // Placeholder implementation
        0.75
    }
    
    fn measure_resource_efficiency(&self, pattern: &CognitivePattern) -> f32 {
        // Placeholder implementation
        0.8
    }
    
    fn calculate_success_rate(&self, pattern: &CognitivePattern) -> f32 {
        if pattern.context_success_map.is_empty() {
            return 0.5;
        }
        
        let total: f32 = pattern.context_success_map.values().sum();
        total / pattern.context_success_map.len() as f32
    }
    
    fn extract_pattern(&self, window: &[Decision]) -> Option<ThinkingPattern> {
        // Placeholder implementation
        None
    }
    
    fn analyze_decision_frequencies(&self, trace: &[Decision]) -> Vec<ThinkingPattern> {
        // Placeholder implementation
        Vec::new()
    }
    
    fn analyze_temporal_patterns(&self, trace: &[Decision]) -> Vec<ThinkingPattern> {
        // Placeholder implementation
        Vec::new()
    }
    
    fn calculate_coverage_breadth(&self, history: &VecDeque<AttentionSnapshot>) -> f32 {
        let mut unique_targets = std::collections::HashSet::new();
        
        for snapshot in history {
            for key in snapshot.attention_weights.keys() {
                unique_targets.insert(key.clone());
            }
        }
        
        (unique_targets.len() as f32).log2() / 10.0
    }
    
    fn calculate_switching_frequency(&self, history: &VecDeque<AttentionSnapshot>) -> f32 {
        if history.len() < 2 {
            return 0.0;
        }
        
        let mut switches = 0;
        let mut prev_focus = &history[0].focus_target;
        
        for snapshot in history.iter().skip(1) {
            if snapshot.focus_target != *prev_focus {
                switches += 1;
            }
            prev_focus = &snapshot.focus_target;
        }
        
        switches as f32 / history.len() as f32
    }
    
    fn calculate_peak_focus_duration(&self, history: &VecDeque<AttentionSnapshot>) -> Duration {
        let mut max_duration = Duration::seconds(0);
        let mut current_duration = Duration::seconds(0);
        let mut current_focus: Option<String> = None;
        let mut start_time: Option<DateTime<Utc>> = None;
        
        for snapshot in history {
            match (&current_focus, &snapshot.focus_target) {
                (Some(cf), sf) if cf == sf => {
                    // Continue same focus
                    if let Some(st) = start_time {
                        current_duration = snapshot.timestamp - st;
                    }
                }
                (_, sf) => {
                    // New focus
                    if current_duration > max_duration {
                        max_duration = current_duration;
                    }
                    current_focus = Some(sf.clone());
                    start_time = Some(snapshot.timestamp);
                    current_duration = Duration::seconds(0);
                }
                _ => {
                    // No focus
                    if current_duration > max_duration {
                        max_duration = current_duration;
                    }
                    current_focus = None;
                    start_time = None;
                    current_duration = Duration::seconds(0);
                }
            }
        }
        
        if current_duration > max_duration {
            max_duration = current_duration;
        }
        
        max_duration
    }
    
    fn calculate_average_distribution(&self, history: &VecDeque<AttentionSnapshot>) -> HashMap<String, f32> {
        let mut total_weights: HashMap<String, f32> = HashMap::new();
        
        for snapshot in history {
            for (key, weight) in &snapshot.attention_weights {
                *total_weights.entry(key.clone()).or_insert(0.0) += weight;
            }
        }
        
        // Normalize by number of snapshots
        let snapshot_count = history.len() as f32;
        for weight in total_weights.values_mut() {
            *weight /= snapshot_count;
        }
        
        total_weights
    }
}