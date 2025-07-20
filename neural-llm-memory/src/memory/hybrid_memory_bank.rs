//! Hybrid Memory Bank with Consciousness Integration
//!
//! This module provides a consciousness-aware memory bank that integrates
//! emotional and consciousness states into memory retrieval and consolidation.

use crate::memory::{MemoryKey, MemoryValue, MemoryOperations, MemoryMetadata};
use crate::memory::memory_bank::{MemoryBank, MemoryEntry};
use crate::consciousness::{ConsciousnessCore, ConsciousContent, ContentType};
use crate::emotional::EmotionalProcessor;
use ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// Consciousness context for memory operations
#[derive(Debug, Clone)]
pub struct ConsciousnessContext {
    /// Current consciousness level (0.0 to 1.0)
    pub consciousness_level: f32,
    
    /// Emotional state context
    pub emotional_state: EmotionalContext,
    
    /// Attention allocation map
    pub attention_allocation: HashMap<String, f32>,
    
    /// Self-awareness level
    pub self_awareness_level: f32,
    
    /// Metacognitive confidence
    pub metacognitive_confidence: f32,
}

/// Emotional context for memory operations
#[derive(Debug, Clone)]
pub struct EmotionalContext {
    /// Emotional valence (-1.0 to 1.0)
    pub valence: f32,
    
    /// Emotional arousal (0.0 to 1.0)
    pub arousal: f32,
    
    /// Dominant emotion type
    pub dominant_emotion: String,
    
    /// Emotion intensity (0.0 to 1.0)
    pub intensity: f32,
}

impl Default for ConsciousnessContext {
    fn default() -> Self {
        Self {
            consciousness_level: 0.5,
            emotional_state: EmotionalContext {
                valence: 0.0,
                arousal: 0.5,
                dominant_emotion: "neutral".to_string(),
                intensity: 0.5,
            },
            attention_allocation: HashMap::new(),
            self_awareness_level: 0.5,
            metacognitive_confidence: 0.5,
        }
    }
}

/// Hybrid Memory Bank with consciousness-aware retrieval
pub struct HybridMemoryBank {
    /// Base memory bank functionality
    base: MemoryBank,
    
    /// Consciousness core for awareness integration
    consciousness_core: Arc<Mutex<ConsciousnessCore>>,
    
    /// Emotional processor for emotional context
    emotional_processor: Arc<Mutex<EmotionalProcessor>>,
    
    /// Current consciousness context
    consciousness_context: Arc<Mutex<ConsciousnessContext>>,
    
    /// Configuration parameters
    config: HybridMemoryConfig,
}

/// Configuration for hybrid memory bank
#[derive(Debug, Clone)]
pub struct HybridMemoryConfig {
    /// Weight for consciousness in retrieval (0.0 to 1.0)
    pub consciousness_weight: f32,
    
    /// Weight for emotional context in retrieval (0.0 to 1.0)
    pub emotional_weight: f32,
    
    /// Threshold for consciousness-based filtering
    pub consciousness_threshold: f32,
    
    /// Enable emotional consolidation
    pub enable_emotional_consolidation: bool,
    
    /// Enable consciousness-based importance adjustment
    pub enable_consciousness_importance: bool,
}

impl Default for HybridMemoryConfig {
    fn default() -> Self {
        Self {
            consciousness_weight: 0.3,
            emotional_weight: 0.2,
            consciousness_threshold: 0.5,
            enable_emotional_consolidation: true,
            enable_consciousness_importance: true,
        }
    }
}

impl HybridMemoryBank {
    /// Create new hybrid memory bank
    pub fn new(
        max_memories: usize,
        cache_size: usize,
        config: HybridMemoryConfig,
    ) -> Self {
        Self {
            base: MemoryBank::new(max_memories, cache_size),
            consciousness_core: Arc::new(Mutex::new(ConsciousnessCore::new())),
            emotional_processor: Arc::new(Mutex::new(EmotionalProcessor::new())),
            consciousness_context: Arc::new(Mutex::new(ConsciousnessContext::default())),
            config,
        }
    }
    
    /// Update consciousness context
    pub fn update_consciousness_context(&self, context: ConsciousnessContext) {
        let mut current_context = self.consciousness_context.lock().unwrap();
        *current_context = context;
    }
    
    /// Get current consciousness context
    pub fn get_consciousness_context(&self) -> ConsciousnessContext {
        let context = self.consciousness_context.lock().unwrap();
        context.clone()
    }
    
    /// Compute consciousness-weighted similarity
    fn compute_consciousness_weighted_similarity(
        &self,
        query_embedding: &Array2<f32>,
        memory_entry: &MemoryEntry,
        context: &ConsciousnessContext,
    ) -> f32 {
        // Base similarity (cosine similarity)
        let base_similarity = self.compute_similarity(
            query_embedding,
            &memory_entry.embedding_array,
        );
        
        // Consciousness weighting factor
        let consciousness_factor = if context.consciousness_level > self.config.consciousness_threshold {
            1.0 + (context.consciousness_level - self.config.consciousness_threshold) * self.config.consciousness_weight
        } else {
            1.0
        };
        
        // Emotional relevance factor
        let emotional_factor = self.compute_emotional_relevance(
            &memory_entry.value,
            &context.emotional_state,
        );
        
        // Self-awareness factor - memories related to self get boosted
        let self_awareness_factor = if memory_entry.value.metadata.tags.contains(&"self-related".to_string()) {
            1.0 + context.self_awareness_level * 0.2
        } else {
            1.0
        };
        
        // Metacognitive factor - boost memories marked as important
        let metacognitive_factor = if memory_entry.value.metadata.importance > 0.7 {
            1.0 + context.metacognitive_confidence * 0.1
        } else {
            1.0
        };
        
        // Combine all factors
        base_similarity 
            * consciousness_factor 
            * (1.0 + emotional_factor * self.config.emotional_weight)
            * self_awareness_factor
            * metacognitive_factor
    }
    
    /// Compute emotional relevance
    fn compute_emotional_relevance(
        &self,
        memory_value: &MemoryValue,
        emotional_context: &EmotionalContext,
    ) -> f32 {
        // Check if memory has emotional tags
        if let Some(memory_emotion) = memory_value.metadata.tags.iter().find(|tag| tag.starts_with("emotion:")) {
            let emotion_name = memory_emotion.strip_prefix("emotion:").unwrap_or(memory_emotion);
            // Same emotion boost
            if emotion_name == emotional_context.dominant_emotion {
                return 1.0 * emotional_context.intensity;
            }
            
            // Related emotions get partial boost
            let emotion_similarity = self.get_emotion_similarity(
                emotion_name,
                &emotional_context.dominant_emotion,
            );
            
            emotion_similarity * emotional_context.intensity
        } else {
            // No emotional tag, check valence
            if let Some(valence_tag) = memory_value.metadata.tags.iter().find(|tag| tag.starts_with("valence:")) {
                let valence_str = valence_tag.strip_prefix("valence:").unwrap_or(valence_tag);
                if let Ok(val) = valence_str.parse::<f32>() {
                    // Similar valence boost
                    let valence_diff = (val - emotional_context.valence).abs();
                    return (1.0 - valence_diff / 2.0) * emotional_context.arousal;
                }
            }
            
            0.0
        }
    }
    
    /// Get similarity between emotions
    fn get_emotion_similarity(&self, emotion1: &str, emotion2: &str) -> f32 {
        // Simple emotion similarity matrix
        let similar_emotions = [
            (["joy", "happiness", "excitement"], 0.8),
            (["sadness", "grief", "melancholy"], 0.8),
            (["anger", "frustration", "irritation"], 0.8),
            (["fear", "anxiety", "worry"], 0.8),
            (["surprise", "amazement", "wonder"], 0.7),
        ];
        
        for (group, similarity) in &similar_emotions {
            if group.contains(&emotion1) && group.contains(&emotion2) {
                return *similarity;
            }
        }
        
        if emotion1 == emotion2 {
            1.0
        } else {
            0.0
        }
    }
    
    /// Compute similarity between embeddings
    fn compute_similarity(&self, a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        let dot_product = (a * b).sum();
        let norm_a = (a * a).sum().sqrt();
        let norm_b = (b * b).sum().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
    
    /// Search with consciousness context
    pub fn search_with_consciousness(
        &self,
        query_embedding: &Array2<f32>,
        k: usize,
        consciousness_context: Option<ConsciousnessContext>,
    ) -> Vec<(MemoryKey, MemoryValue, f32)> {
        use rayon::prelude::*;
        
        // Use provided context or current context
        let context = consciousness_context.unwrap_or_else(|| {
            self.consciousness_context.lock().unwrap().clone()
        });
        
        // Get all memories from base storage
        let all_memories = self.base.get_all_memories();
        
        // Compute consciousness-weighted similarities in parallel
        let mut results: Vec<_> = all_memories
            .par_iter()
            .map(|(key, value)| {
                // Reconstruct memory entry
                let embedding_array = Array2::from_shape_vec(
                    (1, value.embedding.len()),
                    value.embedding.clone(),
                ).unwrap_or_else(|_| Array2::zeros((1, value.embedding.len())));
                
                let entry = MemoryEntry {
                    key: key.clone(),
                    value: value.clone(),
                    embedding_array,
                };
                
                let score = self.compute_consciousness_weighted_similarity(
                    query_embedding,
                    &entry,
                    &context,
                );
                
                (key.clone(), value.clone(), score)
            })
            .collect();
        
        // Sort by similarity score (descending)
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top k results
        results.truncate(k);
        results
    }
    
    /// Store with consciousness context
    pub fn store_with_consciousness(
        &mut self,
        key: MemoryKey,
        mut value: MemoryValue,
        consciousness_context: Option<ConsciousnessContext>,
    ) -> crate::Result<()> {
        // Get consciousness context
        let context = consciousness_context.unwrap_or_else(|| {
            self.consciousness_context.lock().unwrap().clone()
        });
        
        // Adjust importance based on consciousness level if enabled
        if self.config.enable_consciousness_importance {
            value.metadata.importance *= 1.0 + (context.consciousness_level * 0.5);
            value.metadata.importance = value.metadata.importance.min(1.0);
        }
        
        // Add consciousness-related tags
        value.metadata.tags.push(
            format!("consciousness_level:{}", context.consciousness_level)
        );
        
        // Add emotional context if significant
        if context.emotional_state.intensity > 0.5 {
            value.metadata.tags.push(
                format!("emotion:{}", context.emotional_state.dominant_emotion)
            );
            value.metadata.tags.push(
                format!("valence:{}", context.emotional_state.valence)
            );
            value.metadata.tags.push(
                format!("arousal:{}", context.emotional_state.arousal)
            );
        }
        
        // Add self-awareness tag if high
        if context.self_awareness_level > 0.7 {
            value.metadata.tags.push(
                "self-related".to_string()
            );
        }
        
        // Store in base memory bank
        self.base.store(key, value)
    }
    
    /// Consolidate memories with emotional context
    pub fn consolidate_with_emotion(&mut self) -> crate::Result<()> {
        if !self.config.enable_emotional_consolidation {
            return Ok(());
        }
        
        let context = self.consciousness_context.lock().unwrap().clone();
        let emotional_state = &context.emotional_state;
        
        // Get all memories
        let all_memories = self.base.get_all_memories();
        
        // Find emotionally congruent memories for consolidation
        let mut consolidation_candidates = Vec::new();
        
        for (key, value) in all_memories {
            if let Some(emotion_tag) = value.metadata.tags.iter().find(|tag| tag.starts_with("emotion:")) {
                let memory_emotion = emotion_tag.strip_prefix("emotion:").unwrap_or(emotion_tag);
                let similarity = self.get_emotion_similarity(
                    memory_emotion,
                    &emotional_state.dominant_emotion,
                );
                
                if similarity > 0.7 {
                    consolidation_candidates.push((key, value, similarity));
                }
            }
        }
        
        // Sort by emotional similarity
        consolidation_candidates.sort_by(|a, b| {
            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Boost importance of emotionally congruent memories
        for (key, value, similarity) in consolidation_candidates.iter().take(10) {
            let new_importance = (value.metadata.importance
                + similarity * emotional_state.intensity * 0.1)
                .min(1.0);
            
            // Update decay factor to preserve emotional memories longer
            let new_decay_factor = (value.metadata.decay_factor + 0.1).min(0.99);
            
            // Update in storage
            self.base.update(key, |v| {
                v.metadata.importance = new_importance;
                v.metadata.decay_factor = new_decay_factor;
            })?;
        }
        
        Ok(())
    }
    
    /// Get memories filtered by consciousness level
    pub fn get_conscious_memories(&self, min_consciousness: f32) -> Vec<(MemoryKey, MemoryValue)> {
        self.base.get_all_memories()
            .into_iter()
            .filter(|(_, value)| {
                if let Some(level_tag) = value.metadata.tags.iter().find(|tag| tag.starts_with("consciousness_level:")) {
                    let level_str = level_tag.strip_prefix("consciousness_level:").unwrap_or(level_tag);
                    if let Ok(level) = level_str.parse::<f32>() {
                        return level >= min_consciousness;
                    }
                }
                false
            })
            .collect()
    }
    
    /// Get emotional memory summary
    pub fn get_emotional_summary(&self) -> HashMap<String, usize> {
        let mut emotion_counts = HashMap::new();
        
        for (_, value) in self.base.get_all_memories() {
            if let Some(emotion_tag) = value.metadata.tags.iter().find(|tag| tag.starts_with("emotion:")) {
                let emotion = emotion_tag.strip_prefix("emotion:").unwrap_or(emotion_tag);
                *emotion_counts.entry(emotion.to_string()).or_insert(0) += 1;
            }
        }
        
        emotion_counts
    }
}

// Implement MemoryOperations trait for compatibility
impl MemoryOperations for HybridMemoryBank {
    fn store(&mut self, key: MemoryKey, value: MemoryValue) -> crate::Result<()> {
        // Use consciousness-aware storage with current context
        self.store_with_consciousness(key, value, None)
    }
    
    fn retrieve(&mut self, key: &MemoryKey) -> crate::Result<Option<MemoryValue>> {
        self.base.retrieve(key)
    }
    
    fn search(&self, query_embedding: &Array2<f32>, k: usize) -> Vec<(MemoryKey, MemoryValue, f32)> {
        // Use consciousness-aware search with current context
        self.search_with_consciousness(query_embedding, k, None)
    }
    
    fn update(&mut self, key: &MemoryKey, update_fn: impl FnOnce(&mut MemoryValue)) -> crate::Result<()> {
        self.base.update(key, update_fn)
    }
    
    fn delete(&mut self, key: &MemoryKey) -> crate::Result<bool> {
        self.base.delete(key)
    }
    
    fn clear(&mut self) {
        self.base.clear()
    }
    
    fn size(&self) -> usize {
        self.base.size()
    }
}

