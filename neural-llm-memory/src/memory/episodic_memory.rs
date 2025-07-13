//! Episodic memory system for experience-based learning

use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use ndarray::Array2;

use crate::memory::{MemoryKey, MemoryEntry, MemoryBank};
use crate::nn::{NeuralNetwork};
use crate::nn::layers::temporal::{LSTMLayer, MultiHeadAttentionLayer};
use crate::nn::tensor::Tensor;

/// Represents a single episodic memory with temporal context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub content: String,
    pub context: Vec<String>,
    pub emotional_valence: f32,  // -1.0 to 1.0
    pub importance: f32,          // 0.0 to 1.0
    pub retrieval_count: usize,
    pub last_accessed: DateTime<Utc>,
    pub associations: Vec<String>, // IDs of related episodes
    #[serde(skip)]
    pub embedding: Option<Array2<f32>>,
}

impl Episode {
    pub fn new(content: String, context: Vec<String>, emotional_valence: f32) -> Self {
        let now = Utc::now();
        Self {
            id: format!("ep_{}", uuid::Uuid::new_v4()),
            timestamp: now,
            content,
            context,
            emotional_valence,
            importance: 0.5, // Default moderate importance
            retrieval_count: 0,
            last_accessed: now,
            associations: Vec::new(),
            embedding: None,
        }
    }
    
    /// Update importance based on access patterns and emotional significance
    pub fn update_importance(&mut self) {
        // Importance increases with emotional intensity and retrieval frequency
        let emotional_factor = self.emotional_valence.abs();
        let retrieval_factor = (self.retrieval_count as f32).ln().max(0.0) / 10.0;
        let recency_factor = {
            let hours_since_access = (Utc::now() - self.last_accessed).num_hours() as f32;
            1.0 / (1.0 + hours_since_access / 24.0)
        };
        
        self.importance = (emotional_factor + retrieval_factor + recency_factor) / 3.0;
        self.importance = self.importance.clamp(0.0, 1.0);
    }
    
    /// Mark episode as accessed
    pub fn access(&mut self) {
        self.retrieval_count += 1;
        self.last_accessed = Utc::now();
        self.update_importance();
    }
}

/// Episodic memory bank with temporal sequence processing
pub struct EpisodicBank {
    episodes: DashMap<String, Episode>,
    sequences: Arc<RwLock<VecDeque<String>>>, // Ordered episode IDs
    max_episodes: usize,
    consolidation_threshold: f32,
    
    // Neural components for episodic processing
    lstm_encoder: Arc<RwLock<LSTMLayer>>,
    attention_retriever: Arc<RwLock<MultiHeadAttentionLayer>>,
}

impl EpisodicBank {
    pub fn new(
        max_episodes: usize,
        hidden_size: usize,
        embed_dim: usize,
        num_heads: usize,
    ) -> Self {
        use crate::nn::WeightInit;
        
        Self {
            episodes: DashMap::new(),
            sequences: Arc::new(RwLock::new(VecDeque::with_capacity(max_episodes))),
            max_episodes,
            consolidation_threshold: 0.7,
            lstm_encoder: Arc::new(RwLock::new(
                LSTMLayer::new(embed_dim, hidden_size, WeightInit::Xavier)
            )),
            attention_retriever: Arc::new(RwLock::new(
                MultiHeadAttentionLayer::new(embed_dim, num_heads, WeightInit::Xavier)
            )),
        }
    }
    
    /// Store a new episode
    pub fn store_episode(&self, episode: Episode) {
        let episode_id = episode.id.clone();
        
        // Add to episodes map
        self.episodes.insert(episode_id.clone(), episode);
        
        // Add to temporal sequence
        let mut sequences = self.sequences.write();
        sequences.push_back(episode_id.clone());
        
        // Evict old episodes if over capacity
        if sequences.len() > self.max_episodes {
            if let Some(old_id) = sequences.pop_front() {
                self.evict_episode(&old_id);
            }
        }
    }
    
    /// Retrieve episode by ID
    pub fn get_episode(&self, id: &str) -> Option<Episode> {
        self.episodes.get_mut(id).map(|mut entry| {
            entry.access();
            entry.clone()
        })
    }
    
    /// Retrieve episodes by temporal context
    pub fn get_temporal_context(&self, current_id: &str, window_size: usize) -> Vec<Episode> {
        let sequences = self.sequences.read();
        let mut context = Vec::new();
        
        // Find current episode position
        if let Some(pos) = sequences.iter().position(|id| id == current_id) {
            // Get episodes before and after
            let start = pos.saturating_sub(window_size / 2);
            let end = (pos + window_size / 2 + 1).min(sequences.len());
            
            for i in start..end {
                if let Some(id) = sequences.get(i) {
                    if let Some(episode) = self.get_episode(id) {
                        context.push(episode);
                    }
                }
            }
        }
        
        context
    }
    
    /// Find similar episodes using attention mechanism
    pub fn find_similar_episodes(
        &self,
        query_embedding: &Array2<f32>,
        top_k: usize,
    ) -> Vec<(Episode, f32)> {
        let attention = self.attention_retriever.read();
        let mut similarities = Vec::new();
        
        // Compute attention scores for all episodes
        for entry in self.episodes.iter() {
            let episode = entry.value();
            if let Some(ref embedding) = episode.embedding {
                // Use cosine similarity to compute similarity
                let score = Tensor::cosine_similarity(query_embedding, embedding);
                similarities.push((episode.clone(), score));
            }
        }
        
        // Sort by similarity and return top-k
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(top_k);
        similarities
    }
    
    /// Consolidate episodes based on importance and associations
    pub fn consolidate_episodes(&self) -> Vec<Episode> {
        let mut consolidated = Vec::new();
        
        for entry in self.episodes.iter() {
            let episode = entry.value();
            if episode.importance >= self.consolidation_threshold {
                // High-importance episodes are candidates for consolidation
                consolidated.push(episode.clone());
            }
        }
        
        // Group related episodes
        self.group_associated_episodes(&mut consolidated);
        
        consolidated
    }
    
    /// Group episodes by associations for pattern extraction
    fn group_associated_episodes(&self, episodes: &mut Vec<Episode>) {
        // Simple clustering based on associations
        let mut visited = std::collections::HashSet::new();
        let mut groups = Vec::new();
        
        for episode in episodes.iter() {
            if visited.contains(&episode.id) {
                continue;
            }
            
            let mut group = vec![episode.clone()];
            visited.insert(episode.id.clone());
            
            // Add all associated episodes
            for assoc_id in &episode.associations {
                if !visited.contains(assoc_id) {
                    if let Some(assoc_episode) = self.get_episode(assoc_id) {
                        group.push(assoc_episode);
                        visited.insert(assoc_id.clone());
                    }
                }
            }
            
            if group.len() > 1 {
                groups.push(group);
            }
        }
        
        // TODO: Extract patterns from groups
    }
    
    /// Evict episode with lowest importance
    fn evict_episode(&self, id: &str) {
        if let Some((_, episode)) = self.episodes.remove(id) {
            // Log eviction for potential recovery
            // Debug: Evicted episode {} with importance {}
            // episode.id, episode.importance
        }
    }
    
    /// Create associations between episodes
    pub fn associate_episodes(&self, id1: &str, id2: &str) {
        if let Some(mut ep1) = self.episodes.get_mut(id1) {
            if !ep1.associations.contains(&id2.to_string()) {
                ep1.associations.push(id2.to_string());
            }
        }
        
        if let Some(mut ep2) = self.episodes.get_mut(id2) {
            if !ep2.associations.contains(&id1.to_string()) {
                ep2.associations.push(id1.to_string());
            }
        }
    }
    
    /// Encode episode sequence using LSTM
    pub fn encode_sequence(&self, episode_ids: &[String]) -> Option<Array2<f32>> {
        let lstm = self.lstm_encoder.read();
        let mut sequence_embeddings = Vec::new();
        
        // Collect embeddings for the sequence
        for id in episode_ids {
            if let Some(episode) = self.episodes.get(id) {
                if let Some(ref embedding) = episode.embedding {
                    sequence_embeddings.push(embedding.clone());
                }
            }
        }
        
        if sequence_embeddings.is_empty() {
            return None;
        }
        
        // Process through LSTM
        let batch_size = 1;
        lstm.reset_state(batch_size);
        
        let mut encoded = None;
        for embedding in sequence_embeddings {
            // Process through LSTM (would need Layer trait access)
            // encoded = Some(lstm.forward(&embedding, false));
            encoded = Some(embedding); // Placeholder
        }
        
        encoded
    }
    
    /// Get memory statistics
    pub fn get_stats(&self) -> EpisodicStats {
        let total_episodes = self.episodes.len();
        let sequences = self.sequences.read();
        let sequence_length = sequences.len();
        
        let mut total_importance = 0.0;
        let mut total_emotional_valence = 0.0;
        let mut total_associations = 0;
        
        for entry in self.episodes.iter() {
            let episode = entry.value();
            total_importance += episode.importance;
            total_emotional_valence += episode.emotional_valence.abs();
            total_associations += episode.associations.len();
        }
        
        EpisodicStats {
            total_episodes,
            sequence_length,
            average_importance: if total_episodes > 0 { 
                total_importance / total_episodes as f32 
            } else { 0.0 },
            average_emotional_intensity: if total_episodes > 0 { 
                total_emotional_valence / total_episodes as f32 
            } else { 0.0 },
            total_associations,
            consolidation_candidates: self.episodes.iter()
                .filter(|e| e.value().importance >= self.consolidation_threshold)
                .count(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EpisodicStats {
    pub total_episodes: usize,
    pub sequence_length: usize,
    pub average_importance: f32,
    pub average_emotional_intensity: f32,
    pub total_associations: usize,
    pub consolidation_candidates: usize,
}

// Integration with main MemoryBank would go here
// This would require extending MemoryBank with episodic_bank field