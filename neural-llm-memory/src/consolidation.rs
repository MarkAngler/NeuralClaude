//! Memory consolidation system for offline processing and pattern extraction

use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use parking_lot::RwLock;
use tokio::time::{Duration, interval};
use ndarray::Array2;
use serde::{Serialize, Deserialize};

use crate::memory::{EpisodicBank, Episode};
use crate::nn::{NeuralNetwork, NetworkBuilder, ActivationFunction, WeightInit};
use crate::adaptive::EvolvedArchitecture;

/// Trait for offline memory processing during "dream" states
pub trait DreamMode: Send + Sync {
    /// Enter dream mode for consolidation
    fn enter_dream_mode(&self) -> ConsolidationSession;
    
    /// Process memories for pattern extraction
    fn consolidate_memories(&self, session: &mut ConsolidationSession);
    
    /// Generate new patterns through replay
    fn generative_replay(&self, session: &ConsolidationSession) -> Vec<GeneratedPattern>;
    
    /// Exit dream mode and apply learnings
    fn exit_dream_mode(&self, session: ConsolidationSession);
}

/// A consolidation session representing a dream-like state
#[derive(Debug)]
pub struct ConsolidationSession {
    pub session_id: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub episodes_processed: usize,
    pub patterns_extracted: Vec<ExtractedPattern>,
    pub replay_sequences: Vec<ReplaySequence>,
    pub consolidation_stats: ConsolidationStats,
}

impl ConsolidationSession {
    pub fn new() -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            start_time: chrono::Utc::now(),
            episodes_processed: 0,
            patterns_extracted: Vec::new(),
            replay_sequences: Vec::new(),
            consolidation_stats: ConsolidationStats::default(),
        }
    }
}

/// Extracted pattern from episodic memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedPattern {
    pub pattern_id: String,
    pub pattern_type: PatternType,
    pub source_episodes: Vec<String>,
    pub strength: f32,
    pub embedding: Array2<f32>,
    pub context_signature: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Temporal,      // Time-based sequences
    Causal,        // Cause-effect relationships
    Semantic,      // Meaning-based clusters
    Emotional,     // Emotionally similar experiences
    Procedural,    // Action sequences
    Abstract,      // High-level concepts
}

/// Replay sequence for generative learning
#[derive(Debug, Clone)]
pub struct ReplaySequence {
    pub sequence_id: String,
    pub episodes: Vec<Episode>,
    pub transformations: Vec<Transformation>,
    pub novelty_score: f32,
}

#[derive(Debug, Clone)]
pub enum Transformation {
    Interpolation { factor: f32 },
    Recombination { indices: Vec<usize> },
    Abstraction { level: usize },
    EmotionalShift { delta: f32 },
}

/// Generated pattern from replay
#[derive(Debug, Clone)]
pub struct GeneratedPattern {
    pub pattern: ExtractedPattern,
    pub source_replay: String,
    pub creativity_score: f32,
}

/// Statistics for consolidation session
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ConsolidationStats {
    pub total_episodes_scanned: usize,
    pub patterns_discovered: usize,
    pub replay_sequences_generated: usize,
    pub memory_efficiency_gain: f32,
    pub consolidation_quality: f32,
}

/// Main consolidation engine
pub struct ConsolidationEngine {
    episodic_bank: Arc<RwLock<EpisodicBank>>,
    pattern_network: Arc<RwLock<NeuralNetwork>>,
    consolidation_config: ConsolidationConfig,
    active_sessions: Arc<RwLock<HashMap<String, ConsolidationSession>>>,
}

#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    pub min_episodes_for_pattern: usize,
    pub pattern_strength_threshold: f32,
    pub replay_creativity_factor: f32,
    pub consolidation_interval: Duration,
    pub max_patterns_per_session: usize,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            min_episodes_for_pattern: 3,
            pattern_strength_threshold: 0.7,
            replay_creativity_factor: 0.3,
            consolidation_interval: Duration::from_secs(3600), // 1 hour
            max_patterns_per_session: 100,
        }
    }
}

impl ConsolidationEngine {
    pub fn new(
        episodic_bank: Arc<RwLock<EpisodicBank>>,
        hidden_size: usize,
    ) -> Self {
        // Create pattern extraction network
        let pattern_network = NetworkBuilder::new()
            .add_linear(768, hidden_size, ActivationFunction::ReLU, true)
            .add_linear(hidden_size, hidden_size / 2, ActivationFunction::ReLU, true)
            .add_linear(hidden_size / 2, 128, ActivationFunction::Tanh, true)
            .build(0.001);
        
        Self {
            episodic_bank,
            pattern_network: Arc::new(RwLock::new(pattern_network)),
            consolidation_config: ConsolidationConfig::default(),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Start automatic consolidation process
    pub async fn start_auto_consolidation(self: Arc<Self>) {
        let mut interval = interval(self.consolidation_config.consolidation_interval);
        
        tokio::spawn(async move {
            loop {
                interval.tick().await;
                
                // Enter dream mode
                let session = self.enter_dream_mode();
                let session_id = session.session_id.clone();
                
                // Store session
                self.active_sessions.write().insert(session_id.clone(), session);
                
                // Perform consolidation
                if let Some(mut session) = self.active_sessions.write().get_mut(&session_id) {
                    self.consolidate_memories(session);
                    
                    // Generate new patterns
                    let generated = self.generative_replay(session);
                    // Info: Generated new patterns during consolidation: {}
                    // generated.len()
                }
                
                // Exit dream mode
                if let Some(session) = self.active_sessions.write().remove(&session_id) {
                    self.exit_dream_mode(session);
                }
            }
        });
    }
    
    /// Extract patterns from grouped episodes
    fn extract_patterns_from_group(&self, episodes: &[Episode]) -> Vec<ExtractedPattern> {
        let mut patterns = Vec::new();
        
        // Temporal pattern detection
        if episodes.len() >= self.consolidation_config.min_episodes_for_pattern {
            if let Some(temporal_pattern) = self.detect_temporal_pattern(episodes) {
                patterns.push(temporal_pattern);
            }
        }
        
        // Semantic clustering
        let semantic_clusters = self.cluster_by_semantics(episodes);
        for cluster in semantic_clusters {
            if cluster.len() >= self.consolidation_config.min_episodes_for_pattern {
                if let Some(semantic_pattern) = self.extract_semantic_pattern(&cluster) {
                    patterns.push(semantic_pattern);
                }
            }
        }
        
        // Emotional patterns
        if let Some(emotional_pattern) = self.detect_emotional_pattern(episodes) {
            patterns.push(emotional_pattern);
        }
        
        patterns
    }
    
    /// Detect temporal sequences in episodes
    fn detect_temporal_pattern(&self, episodes: &[Episode]) -> Option<ExtractedPattern> {
        // Sort by timestamp
        let mut sorted_episodes = episodes.to_vec();
        sorted_episodes.sort_by_key(|e| e.timestamp);
        
        // Look for recurring sequences
        let mut sequence_embeddings = Vec::new();
        for episode in &sorted_episodes {
            if let Some(ref embedding) = episode.embedding {
                sequence_embeddings.push(embedding.clone());
            }
        }
        
        if sequence_embeddings.is_empty() {
            return None;
        }
        
        // Process through pattern network
        let pattern_network = self.pattern_network.read();
        let mut combined_embedding = Array2::zeros((1, 768));
        
        // Average embeddings (simplified - could use attention mechanism)
        for embedding in &sequence_embeddings {
            combined_embedding = &combined_embedding + embedding;
        }
        combined_embedding = &combined_embedding / sequence_embeddings.len() as f32;
        
        let pattern_embedding = pattern_network.forward(&combined_embedding, false);
        
        Some(ExtractedPattern {
            pattern_id: uuid::Uuid::new_v4().to_string(),
            pattern_type: PatternType::Temporal,
            source_episodes: sorted_episodes.iter().map(|e| e.id.clone()).collect(),
            strength: 0.8, // Calculate based on consistency
            embedding: pattern_embedding,
            context_signature: vec![0.0; 64], // Simplified
        })
    }
    
    /// Cluster episodes by semantic similarity
    fn cluster_by_semantics(&self, episodes: &[Episode]) -> Vec<Vec<Episode>> {
        // Simple clustering - in production use proper clustering algorithm
        let mut clusters = Vec::new();
        let mut clustered = vec![false; episodes.len()];
        
        for i in 0..episodes.len() {
            if clustered[i] {
                continue;
            }
            
            let mut cluster = vec![episodes[i].clone()];
            clustered[i] = true;
            
            if let Some(ref embedding_i) = episodes[i].embedding {
                for j in (i + 1)..episodes.len() {
                    if !clustered[j] {
                        if let Some(ref embedding_j) = episodes[j].embedding {
                            let similarity = self.compute_similarity(embedding_i, embedding_j);
                            if similarity > 0.8 {
                                cluster.push(episodes[j].clone());
                                clustered[j] = true;
                            }
                        }
                    }
                }
            }
            
            clusters.push(cluster);
        }
        
        clusters
    }
    
    /// Extract semantic pattern from cluster
    fn extract_semantic_pattern(&self, cluster: &[Episode]) -> Option<ExtractedPattern> {
        if cluster.is_empty() {
            return None;
        }
        
        // Compute centroid embedding
        let mut centroid = Array2::zeros((1, 768));
        let mut count = 0;
        
        for episode in cluster {
            if let Some(ref embedding) = episode.embedding {
                centroid = &centroid + embedding;
                count += 1;
            }
        }
        
        if count == 0 {
            return None;
        }
        
        centroid = &centroid / count as f32;
        
        // Process through pattern network
        let pattern_network = self.pattern_network.read();
        let pattern_embedding = pattern_network.forward(&centroid, false);
        
        Some(ExtractedPattern {
            pattern_id: uuid::Uuid::new_v4().to_string(),
            pattern_type: PatternType::Semantic,
            source_episodes: cluster.iter().map(|e| e.id.clone()).collect(),
            strength: 0.75,
            embedding: pattern_embedding,
            context_signature: vec![0.0; 64],
        })
    }
    
    /// Detect emotional patterns
    fn detect_emotional_pattern(&self, episodes: &[Episode]) -> Option<ExtractedPattern> {
        // Group by emotional valence
        let mut positive_episodes = Vec::new();
        let mut negative_episodes = Vec::new();
        
        for episode in episodes {
            if episode.emotional_valence > 0.2 {
                positive_episodes.push(episode);
            } else if episode.emotional_valence < -0.2 {
                negative_episodes.push(episode);
            }
        }
        
        // Check if there's a strong emotional pattern
        let dominant_emotion = if positive_episodes.len() > negative_episodes.len() * 2 {
            Some((positive_episodes, 1.0))
        } else if negative_episodes.len() > positive_episodes.len() * 2 {
            Some((negative_episodes, -1.0))
        } else {
            None
        };
        
        if let Some((emotional_episodes, valence)) = dominant_emotion {
            if emotional_episodes.len() >= self.consolidation_config.min_episodes_for_pattern {
                // Create emotional pattern
                let pattern_id = uuid::Uuid::new_v4().to_string();
                let source_ids: Vec<String> = emotional_episodes.iter()
                    .map(|e| e.id.clone())
                    .collect();
                
                // Simplified embedding
                let embedding = Array2::ones((1, 128)) * valence;
                
                return Some(ExtractedPattern {
                    pattern_id,
                    pattern_type: PatternType::Emotional,
                    source_episodes: source_ids,
                    strength: 0.9,
                    embedding,
                    context_signature: vec![valence; 64],
                });
            }
        }
        
        None
    }
    
    /// Compute similarity between embeddings
    fn compute_similarity(&self, a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        // Cosine similarity
        let dot_product: f32 = (a * b).sum();
        let norm_a: f32 = (a * a).sum().sqrt();
        let norm_b: f32 = (b * b).sum().sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

impl DreamMode for ConsolidationEngine {
    fn enter_dream_mode(&self) -> ConsolidationSession {
        // Info: Entering dream mode for memory consolidation
        ConsolidationSession::new()
    }
    
    fn consolidate_memories(&self, session: &mut ConsolidationSession) {
        let episodic_bank = self.episodic_bank.read();
        let candidates = episodic_bank.consolidate_episodes();
        
        session.episodes_processed = candidates.len();
        
        // Extract patterns from high-importance episodes
        let patterns = self.extract_patterns_from_group(&candidates);
        session.patterns_extracted.extend(patterns);
        
        // Update stats
        session.consolidation_stats.total_episodes_scanned = candidates.len();
        session.consolidation_stats.patterns_discovered = session.patterns_extracted.len();
    }
    
    fn generative_replay(&self, session: &ConsolidationSession) -> Vec<GeneratedPattern> {
        let mut generated = Vec::new();
        
        // Generate replay sequences from extracted patterns
        for pattern in &session.patterns_extracted {
            if pattern.strength >= self.consolidation_config.pattern_strength_threshold {
                // Create variations of the pattern
                let variations = self.generate_pattern_variations(pattern);
                
                for (i, variation) in variations.into_iter().enumerate() {
                    generated.push(GeneratedPattern {
                        pattern: variation,
                        source_replay: format!("{}_{}", pattern.pattern_id, i),
                        creativity_score: self.consolidation_config.replay_creativity_factor,
                    });
                }
            }
        }
        
        generated
    }
    
    fn exit_dream_mode(&self, mut session: ConsolidationSession) {
        // Calculate final stats
        session.consolidation_stats.consolidation_quality = 
            session.patterns_extracted.len() as f32 / session.episodes_processed.max(1) as f32;
        
        // Info: Exiting dream mode
        // Processed episodes: session.episodes_processed
        // Extracted patterns: session.patterns_extracted.len()
        
        // Store extracted patterns back to memory
        // This would integrate with the main memory system
    }
}

impl ConsolidationEngine {
    /// Generate variations of a pattern for creative replay
    fn generate_pattern_variations(&self, pattern: &ExtractedPattern) -> Vec<ExtractedPattern> {
        let mut variations = Vec::new();
        
        // Interpolation variation
        let mut interpolated = pattern.clone();
        interpolated.pattern_id = uuid::Uuid::new_v4().to_string();
        interpolated.embedding = &pattern.embedding * 0.9; // Slight modification
        variations.push(interpolated);
        
        // Abstraction variation
        let mut abstracted = pattern.clone();
        abstracted.pattern_id = uuid::Uuid::new_v4().to_string();
        abstracted.pattern_type = PatternType::Abstract;
        // Simplify embedding by reducing dimensions (simplified)
        variations.push(abstracted);
        
        variations
    }
}