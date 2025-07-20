//! AttentionFusion mechanism for multi-modal queries
//! 
//! This module implements the attention fusion system that enables coherent
//! multi-modal attention across different memory types, integrating with
//! the CrossModalBridge for modality translation and MultiHeadAttention
//! for attention computation.

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::Instant;
use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Serialize, Deserialize};
use anyhow::Result;

use crate::attention::{MultiHeadAttention, CrossAttention};
use crate::graph::cross_modal::{CrossModalBridge, MemoryModality};
use crate::nn::tensor::{Tensor, TensorOps};

/// Configuration for attention fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFusionConfig {
    /// Feature dimension for embeddings
    pub feature_dimension: usize,
    
    /// Number of attention heads
    pub num_heads: usize,
    
    /// Dropout rate for attention
    pub dropout_rate: f32,
    
    /// Coherence threshold for gating
    pub coherence_threshold: f32,
    
    /// Maximum modalities to fuse
    pub max_modalities: usize,
    
    /// Enable adaptive weighting
    pub adaptive_weighting: bool,
    
    /// Performance target in milliseconds
    pub target_latency_ms: f64,
}

impl Default for AttentionFusionConfig {
    fn default() -> Self {
        Self {
            feature_dimension: 768,
            num_heads: 8,
            dropout_rate: 0.1,
            coherence_threshold: 0.5,
            max_modalities: 8,
            adaptive_weighting: true,
            target_latency_ms: 5.0,
        }
    }
}

/// Modality-specific attention weight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityWeight {
    pub modality: MemoryModality,
    pub base_weight: f32,
    pub learned_weight: f32,
    pub usage_count: u64,
    pub avg_coherence: f32,
}

/// Coherence metrics for multi-modal fusion
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoherenceMetrics {
    /// Pairwise coherence scores between modalities
    pub pairwise_coherence: HashMap<(MemoryModality, MemoryModality), f32>,
    
    /// Overall coherence score
    pub overall_coherence: f32,
    
    /// Conflict detection count
    pub conflict_count: u32,
    
    /// Successful fusion count
    pub successful_fusions: u64,
}

/// Performance metrics for fusion operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FusionMetrics {
    pub total_fusions: u64,
    pub avg_latency_ms: f64,
    pub max_latency_ms: f64,
    pub min_latency_ms: f64,
    pub modality_usage: HashMap<MemoryModality, u64>,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// AttentionFusion system for multi-modal queries
pub struct AttentionFusion {
    /// Configuration
    config: AttentionFusionConfig,
    
    /// Cross-modal bridge for translation
    cross_modal_bridge: Arc<CrossModalBridge>,
    
    /// Multi-head attention mechanism
    multi_head_attention: MultiHeadAttention,
    
    /// Cross-attention for modality fusion
    cross_attention: CrossAttention,
    
    /// Modality weights
    modality_weights: Arc<RwLock<HashMap<MemoryModality, ModalityWeight>>>,
    
    /// Coherence monitor
    coherence_monitor: Arc<RwLock<CoherenceMetrics>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<FusionMetrics>>,
    
    /// Fusion cache for performance
    fusion_cache: Arc<RwLock<HashMap<String, FusionResult>>>,
    
    /// Gating network for coherence-based filtering
    gating_weights: Arc<RwLock<Array2<f32>>>,
}

/// Result of attention fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionResult {
    /// Fused attention output
    pub fused_output: Array2<f32>,
    
    /// Attention weights per modality
    pub attention_weights: HashMap<MemoryModality, Array2<f32>>,
    
    /// Modality contributions
    pub modality_contributions: HashMap<MemoryModality, f32>,
    
    /// Coherence score
    pub coherence_score: f32,
    
    /// Processing time
    pub latency_ms: f64,
}

impl AttentionFusion {
    /// Create a new AttentionFusion system
    pub fn new(
        config: AttentionFusionConfig,
        cross_modal_bridge: Arc<CrossModalBridge>,
    ) -> Self {
        let multi_head_attention = MultiHeadAttention::new(
            config.feature_dimension,
            config.num_heads,
            config.dropout_rate,
        );
        
        let cross_attention = CrossAttention::new(
            config.feature_dimension,
            config.num_heads,
            config.dropout_rate,
        );
        
        // Initialize gating weights
        let gating_weights = Array2::from_shape_fn(
            (config.max_modalities, config.feature_dimension),
            |_| rand::random::<f32>() * 0.1 - 0.05
        );
        
        let mut fusion = Self {
            config: config.clone(),
            cross_modal_bridge,
            multi_head_attention,
            cross_attention,
            modality_weights: Arc::new(RwLock::new(HashMap::new())),
            coherence_monitor: Arc::new(RwLock::new(CoherenceMetrics::default())),
            metrics: Arc::new(RwLock::new(FusionMetrics::default())),
            fusion_cache: Arc::new(RwLock::new(HashMap::new())),
            gating_weights: Arc::new(RwLock::new(gating_weights)),
        };
        
        // Initialize modality weights
        fusion.initialize_modality_weights();
        
        fusion
    }
    
    /// Initialize modality weights
    fn initialize_modality_weights(&mut self) {
        let modalities = [
            MemoryModality::Semantic,
            MemoryModality::Episodic,
            MemoryModality::Emotional,
            MemoryModality::Procedural,
            MemoryModality::Contextual,
            MemoryModality::Temporal,
            MemoryModality::Causal,
            MemoryModality::Abstract,
        ];
        
        let mut weights = self.modality_weights.write().unwrap();
        
        for modality in &modalities {
            let weight = ModalityWeight {
                modality: *modality,
                base_weight: 1.0 / modalities.len() as f32,
                learned_weight: 0.0,
                usage_count: 0,
                avg_coherence: 0.0,
            };
            weights.insert(*modality, weight);
        }
    }
    
    /// Perform multi-modal attention fusion
    pub fn fuse_attention(
        &self,
        query: &Array2<f32>,
        memory_contents: HashMap<MemoryModality, Array3<f32>>,
        active_modalities: Vec<MemoryModality>,
    ) -> Result<FusionResult> {
        let start_time = Instant::now();
        
        // Check cache first
        let cache_key = self.generate_cache_key(query, &active_modalities);
        if let Some(cached_result) = self.check_cache(&cache_key) {
            return Ok(cached_result);
        }
        
        // Prepare modality-specific attention results
        let mut modality_outputs = HashMap::new();
        let mut attention_weights = HashMap::new();
        
        // Process each modality
        for modality in &active_modalities {
            if let Some(memory) = memory_contents.get(modality) {
                let (output, weights) = self.process_modality(query, memory, *modality)?;
                modality_outputs.insert(*modality, output);
                attention_weights.insert(*modality, weights);
            }
        }
        
        // Compute coherence scores
        let coherence_score = self.compute_coherence(&modality_outputs)?;
        
        // Apply modality weighting
        let weighted_outputs = self.apply_modality_weights(&modality_outputs)?;
        
        // Gate by coherence
        let gated_outputs = self.gate_by_coherence(weighted_outputs, coherence_score)?;
        
        // Fuse outputs
        let fused_output = self.fuse_outputs(gated_outputs)?;
        
        // Compute modality contributions
        let modality_contributions = self.compute_contributions(&modality_outputs, &fused_output)?;
        
        // Update metrics
        let latency_ms = start_time.elapsed().as_micros() as f64 / 1000.0;
        self.update_metrics(latency_ms, &active_modalities, coherence_score);
        
        // Create result
        let result = FusionResult {
            fused_output,
            attention_weights,
            modality_contributions,
            coherence_score,
            latency_ms,
        };
        
        // Cache result
        self.cache_result(cache_key, &result);
        
        Ok(result)
    }
    
    /// Process a single modality
    fn process_modality(
        &self,
        query: &Array2<f32>,
        memory: &Array3<f32>,
        modality: MemoryModality,
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        // Reshape memory for attention computation
        let batch_size = memory.shape()[0];
        let memory_2d = memory.clone().into_shape((batch_size, self.config.feature_dimension))?;
        
        // Apply cross-attention
        let (attended_output, weights) = self.cross_attention.forward(
            query,
            &memory_2d,
            None,
        );
        
        Ok((attended_output, weights))
    }
    
    /// Compute coherence between modality outputs
    fn compute_coherence(
        &self,
        modality_outputs: &HashMap<MemoryModality, Array2<f32>>,
    ) -> Result<f32> {
        if modality_outputs.len() < 2 {
            return Ok(1.0); // Perfect coherence for single modality
        }
        
        let mut coherence_scores = Vec::new();
        let modalities: Vec<_> = modality_outputs.keys().collect();
        
        // Compute pairwise coherence
        for i in 0..modalities.len() {
            for j in i+1..modalities.len() {
                let output_i = &modality_outputs[modalities[i]];
                let output_j = &modality_outputs[modalities[j]];
                
                let coherence = self.compute_pairwise_coherence(output_i, output_j)?;
                coherence_scores.push(coherence);
                
                // Update coherence monitor
                let mut monitor = self.coherence_monitor.write().unwrap();
                monitor.pairwise_coherence.insert(
                    (*modalities[i], *modalities[j]),
                    coherence
                );
            }
        }
        
        // Average coherence
        let avg_coherence = if !coherence_scores.is_empty() {
            coherence_scores.iter().sum::<f32>() / coherence_scores.len() as f32
        } else {
            1.0
        };
        
        // Update monitor
        let mut monitor = self.coherence_monitor.write().unwrap();
        monitor.overall_coherence = avg_coherence;
        
        Ok(avg_coherence)
    }
    
    /// Compute pairwise coherence between outputs
    fn compute_pairwise_coherence(
        &self,
        output_a: &Array2<f32>,
        output_b: &Array2<f32>,
    ) -> Result<f32> {
        // Cosine similarity between flattened outputs
        let flat_a = output_a.as_slice().unwrap();
        let flat_b = output_b.as_slice().unwrap();
        
        let dot_product: f32 = flat_a.iter().zip(flat_b.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let norm_a: f32 = flat_a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = flat_b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm_a * norm_b))
    }
    
    /// Apply modality-specific weights
    fn apply_modality_weights(
        &self,
        modality_outputs: &HashMap<MemoryModality, Array2<f32>>,
    ) -> Result<HashMap<MemoryModality, Array2<f32>>> {
        let weights = self.modality_weights.read().unwrap();
        let mut weighted_outputs = HashMap::new();
        
        for (modality, output) in modality_outputs {
            if let Some(weight_info) = weights.get(modality) {
                let total_weight = weight_info.base_weight + weight_info.learned_weight;
                let weighted = output * total_weight;
                weighted_outputs.insert(*modality, weighted);
            }
        }
        
        Ok(weighted_outputs)
    }
    
    /// Gate outputs by coherence score
    fn gate_by_coherence(
        &self,
        outputs: HashMap<MemoryModality, Array2<f32>>,
        coherence_score: f32,
    ) -> Result<HashMap<MemoryModality, Array2<f32>>> {
        if coherence_score < self.config.coherence_threshold {
            // Low coherence - apply stronger gating
            let gating_weights = self.gating_weights.read().unwrap();
            let mut gated_outputs = HashMap::new();
            
            for (modality, output) in outputs {
                let modality_idx = self.get_modality_index(&modality);
                let gate = gating_weights.index_axis(Axis(0), modality_idx);
                
                // Apply sigmoid gating
                let gate_values = gate.mapv(|x| 1.0 / (1.0 + (-x).exp()));
                let gated = output * gate_values.clone().insert_axis(Axis(0));
                
                gated_outputs.insert(modality, gated);
            }
            
            Ok(gated_outputs)
        } else {
            // High coherence - pass through
            Ok(outputs)
        }
    }
    
    /// Fuse multiple modality outputs
    fn fuse_outputs(
        &self,
        outputs: HashMap<MemoryModality, Array2<f32>>,
    ) -> Result<Array2<f32>> {
        if outputs.is_empty() {
            return Err(anyhow::anyhow!("No outputs to fuse"));
        }
        
        // Initialize with zeros
        let first_output = outputs.values().next().unwrap();
        let mut fused = Array2::zeros(first_output.dim());
        
        // Store the count before consuming the map
        let num_modalities = outputs.len() as f32;
        
        // Sum weighted outputs
        for (_, output) in outputs {
            fused = fused + output;
        }
        
        // Normalize
        fused = fused / num_modalities;
        
        Ok(fused)
    }
    
    /// Compute modality contributions
    fn compute_contributions(
        &self,
        modality_outputs: &HashMap<MemoryModality, Array2<f32>>,
        fused_output: &Array2<f32>,
    ) -> Result<HashMap<MemoryModality, f32>> {
        let mut contributions = HashMap::new();
        
        for (modality, output) in modality_outputs {
            // Compute contribution as correlation with fused output
            let correlation = self.compute_correlation(output, fused_output)?;
            contributions.insert(*modality, correlation.abs());
        }
        
        // Normalize contributions
        let total: f32 = contributions.values().sum();
        if total > 0.0 {
            for value in contributions.values_mut() {
                *value /= total;
            }
        }
        
        Ok(contributions)
    }
    
    /// Compute correlation between two outputs
    fn compute_correlation(
        &self,
        output_a: &Array2<f32>,
        output_b: &Array2<f32>,
    ) -> Result<f32> {
        let flat_a = output_a.as_slice().unwrap();
        let flat_b = output_b.as_slice().unwrap();
        
        let mean_a: f32 = flat_a.iter().sum::<f32>() / flat_a.len() as f32;
        let mean_b: f32 = flat_b.iter().sum::<f32>() / flat_b.len() as f32;
        
        let covariance: f32 = flat_a.iter().zip(flat_b.iter())
            .map(|(a, b)| (a - mean_a) * (b - mean_b))
            .sum::<f32>() / flat_a.len() as f32;
        
        let std_a: f32 = flat_a.iter()
            .map(|x| (x - mean_a).powi(2))
            .sum::<f32>().sqrt() / (flat_a.len() as f32).sqrt();
        
        let std_b: f32 = flat_b.iter()
            .map(|x| (x - mean_b).powi(2))
            .sum::<f32>().sqrt() / (flat_b.len() as f32).sqrt();
        
        if std_a == 0.0 || std_b == 0.0 {
            return Ok(0.0);
        }
        
        Ok(covariance / (std_a * std_b))
    }
    
    /// Update performance metrics
    fn update_metrics(
        &self,
        latency_ms: f64,
        modalities: &[MemoryModality],
        coherence_score: f32,
    ) {
        let mut metrics = self.metrics.write().unwrap();
        
        // Update fusion count
        metrics.total_fusions += 1;
        
        // Update latency
        if metrics.total_fusions == 1 {
            metrics.avg_latency_ms = latency_ms;
            metrics.max_latency_ms = latency_ms;
            metrics.min_latency_ms = latency_ms;
        } else {
            metrics.avg_latency_ms = (metrics.avg_latency_ms * (metrics.total_fusions - 1) as f64 + latency_ms) 
                / metrics.total_fusions as f64;
            metrics.max_latency_ms = metrics.max_latency_ms.max(latency_ms);
            metrics.min_latency_ms = metrics.min_latency_ms.min(latency_ms);
        }
        
        // Update modality usage
        for modality in modalities {
            *metrics.modality_usage.entry(*modality).or_insert(0) += 1;
        }
        
        // Update coherence monitor
        let mut monitor = self.coherence_monitor.write().unwrap();
        if coherence_score >= self.config.coherence_threshold {
            monitor.successful_fusions += 1;
        } else {
            monitor.conflict_count += 1;
        }
        
        // Update modality weights based on coherence
        if self.config.adaptive_weighting {
            self.update_modality_weights(modalities, coherence_score);
        }
    }
    
    /// Update modality weights based on performance
    fn update_modality_weights(&self, modalities: &[MemoryModality], coherence_score: f32) {
        let mut weights = self.modality_weights.write().unwrap();
        
        for modality in modalities {
            if let Some(weight) = weights.get_mut(modality) {
                weight.usage_count += 1;
                
                // Update average coherence
                weight.avg_coherence = (weight.avg_coherence * (weight.usage_count - 1) as f32 
                    + coherence_score) / weight.usage_count as f32;
                
                // Adjust learned weight based on coherence
                let learning_rate = 0.01;
                let coherence_factor = (coherence_score - self.config.coherence_threshold) 
                    / (1.0 - self.config.coherence_threshold);
                weight.learned_weight += learning_rate * coherence_factor;
                
                // Clamp learned weight
                weight.learned_weight = weight.learned_weight.clamp(-0.5, 0.5);
            }
        }
    }
    
    /// Generate cache key
    fn generate_cache_key(&self, query: &Array2<f32>, modalities: &[MemoryModality]) -> String {
        let query_hash = self.hash_array(query);
        let modalities_str = modalities.iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
            .join("-");
        format!("{}-{}", query_hash, modalities_str)
    }
    
    /// Hash an array for caching
    fn hash_array(&self, array: &Array2<f32>) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        array.shape().hash(&mut hasher);
        
        // Sample a few values for hashing
        if let Some(slice) = array.as_slice() {
            for i in (0..slice.len()).step_by(slice.len() / 10 + 1) {
                slice[i].to_bits().hash(&mut hasher);
            }
        }
        
        format!("{:x}", hasher.finish())
    }
    
    /// Check cache for result
    fn check_cache(&self, key: &str) -> Option<FusionResult> {
        let cache = self.fusion_cache.read().unwrap();
        let result = cache.get(key).cloned();
        
        // Update cache metrics
        let mut metrics = self.metrics.write().unwrap();
        if result.is_some() {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }
        
        result
    }
    
    /// Cache fusion result
    fn cache_result(&self, key: String, result: &FusionResult) {
        let mut cache = self.fusion_cache.write().unwrap();
        
        // Limit cache size
        if cache.len() > 1000 {
            // Remove oldest entries (simple FIFO for now)
            let keys_to_remove: Vec<_> = cache.keys().take(100).cloned().collect();
            for k in keys_to_remove {
                cache.remove(&k);
            }
        }
        
        cache.insert(key, result.clone());
    }
    
    /// Get modality index
    fn get_modality_index(&self, modality: &MemoryModality) -> usize {
        match modality {
            MemoryModality::Semantic => 0,
            MemoryModality::Episodic => 1,
            MemoryModality::Emotional => 2,
            MemoryModality::Procedural => 3,
            MemoryModality::Contextual => 4,
            MemoryModality::Temporal => 5,
            MemoryModality::Causal => 6,
            MemoryModality::Abstract => 7,
        }
    }
    
    /// Query across multiple modalities with attention fusion
    pub fn multi_modal_query(
        &self,
        query: &Array2<f32>,
        memory_contents: HashMap<MemoryModality, Array3<f32>>,
        target_modalities: Vec<MemoryModality>,
    ) -> Result<FusionResult> {
        // First, translate query to each target modality if needed
        let source_modality = MemoryModality::Semantic; // Assume semantic query by default
        
        let mut translated_memories = HashMap::new();
        for (modality, content) in &memory_contents {
            if target_modalities.contains(modality) {
                translated_memories.insert(*modality, content.clone());
            }
        }
        
        // Perform attention fusion
        self.fuse_attention(query, translated_memories, target_modalities)
    }
    
    /// Get fusion statistics
    pub fn get_stats(&self) -> FusionStats {
        let metrics = self.metrics.read().unwrap();
        let coherence = self.coherence_monitor.read().unwrap();
        let weights = self.modality_weights.read().unwrap();
        
        FusionStats {
            total_fusions: metrics.total_fusions,
            avg_latency_ms: metrics.avg_latency_ms,
            max_latency_ms: metrics.max_latency_ms,
            min_latency_ms: metrics.min_latency_ms,
            cache_hit_rate: if metrics.cache_hits + metrics.cache_misses > 0 {
                metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64
            } else {
                0.0
            },
            overall_coherence: coherence.overall_coherence,
            successful_fusion_rate: if coherence.successful_fusions + coherence.conflict_count as u64 > 0 {
                coherence.successful_fusions as f64 / 
                (coherence.successful_fusions + coherence.conflict_count as u64) as f64
            } else {
                0.0
            },
            active_modalities: weights.len(),
        }
    }
}

/// Fusion statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionStats {
    pub total_fusions: u64,
    pub avg_latency_ms: f64,
    pub max_latency_ms: f64,
    pub min_latency_ms: f64,
    pub cache_hit_rate: f64,
    pub overall_coherence: f32,
    pub successful_fusion_rate: f64,
    pub active_modalities: usize,
}

