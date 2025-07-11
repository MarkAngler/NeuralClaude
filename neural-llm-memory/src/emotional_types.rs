//! Additional emotional types and implementations for the demo
//! 
//! This file provides the missing types and methods needed for the emotional demo

use crate::emotional::EmotionalState;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use ndarray::Array1;

/// Emotion enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Emotion {
    Joy,
    Sadness,
    Anger,
    Fear,
    Surprise,
    Disgust,
    Anticipation,
    Trust,
    Love,
    Curiosity,
    Confusion,
    Excitement,
    Anxiety,
    Contentment,
    Frustration,
    Pride,
    Shame,
    Guilt,
    Gratitude,
    Hope,
    Relief,
}

impl Emotion {
    /// Get valence for this emotion (-1.0 to 1.0)
    pub fn valence(&self) -> f32 {
        match self {
            Emotion::Joy | Emotion::Love | Emotion::Excitement | 
            Emotion::Contentment | Emotion::Pride | Emotion::Gratitude | 
            Emotion::Hope | Emotion::Relief => 0.6,
            
            Emotion::Sadness | Emotion::Anger | Emotion::Fear | 
            Emotion::Disgust | Emotion::Anxiety | Emotion::Frustration | 
            Emotion::Shame | Emotion::Guilt => -0.6,
            
            Emotion::Surprise | Emotion::Anticipation | Emotion::Trust | 
            Emotion::Curiosity | Emotion::Confusion => 0.0,
        }
    }
    
    /// Get typical arousal level for this emotion (0.0 to 1.0)
    pub fn arousal(&self) -> f32 {
        match self {
            Emotion::Excitement | Emotion::Anger | Emotion::Fear | 
            Emotion::Surprise | Emotion::Anxiety => 0.8,
            
            Emotion::Joy | Emotion::Love | Emotion::Disgust | 
            Emotion::Anticipation | Emotion::Frustration | 
            Emotion::Pride | Emotion::Hope => 0.6,
            
            Emotion::Curiosity | Emotion::Confusion | Emotion::Shame | 
            Emotion::Guilt | Emotion::Gratitude => 0.4,
            
            Emotion::Sadness | Emotion::Trust | Emotion::Contentment | 
            Emotion::Relief => 0.2,
        }
    }
}

impl EmotionalState {
    /// Create emotional state from a single emotion
    pub fn from_emotion(emotion: Emotion, intensity: f32) -> Self {
        let mut emotions = HashMap::new();
        emotions.insert(format!("{:?}", emotion), intensity);
        
        Self {
            emotions,
            valence: emotion.valence() * intensity,
            arousal: emotion.arousal() * intensity,
            stability: 0.7,
            complexity: 0.2,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
    
    /// Create neutral emotional state
    pub fn neutral() -> Self {
        Self {
            emotions: HashMap::new(),
            valence: 0.0,
            arousal: 0.0,
            stability: 0.8,
            complexity: 0.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
    
    /// Blend multiple emotions
    pub fn blend(emotion_weights: Vec<(Emotion, f32)>) -> Self {
        let mut emotions = HashMap::new();
        let mut total_valence = 0.0;
        let mut total_arousal = 0.0;
        let mut total_weight = 0.0;
        
        for (emotion, weight) in emotion_weights {
            emotions.insert(format!("{:?}", emotion), weight);
            total_valence += emotion.valence() * weight;
            total_arousal += emotion.arousal() * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            total_valence /= total_weight;
            total_arousal /= total_weight;
        }
        
        let complexity = emotions.len() as f32 / 5.0;
        Self {
            emotions,
            valence: total_valence,
            arousal: total_arousal,
            stability: 0.5,
            complexity,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
    
    /// Get dominant emotion
    pub fn dominant_emotion(&self) -> Option<(Emotion, f32)> {
        self.emotions.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .and_then(|(name, &intensity)| {
                // Parse emotion name back to enum
                match name.as_str() {
                    "Joy" => Some((Emotion::Joy, intensity)),
                    "Sadness" => Some((Emotion::Sadness, intensity)),
                    "Anger" => Some((Emotion::Anger, intensity)),
                    "Fear" => Some((Emotion::Fear, intensity)),
                    "Surprise" => Some((Emotion::Surprise, intensity)),
                    "Disgust" => Some((Emotion::Disgust, intensity)),
                    "Anticipation" => Some((Emotion::Anticipation, intensity)),
                    "Trust" => Some((Emotion::Trust, intensity)),
                    "Love" => Some((Emotion::Love, intensity)),
                    "Curiosity" => Some((Emotion::Curiosity, intensity)),
                    "Confusion" => Some((Emotion::Confusion, intensity)),
                    "Excitement" => Some((Emotion::Excitement, intensity)),
                    "Anxiety" => Some((Emotion::Anxiety, intensity)),
                    "Contentment" => Some((Emotion::Contentment, intensity)),
                    "Frustration" => Some((Emotion::Frustration, intensity)),
                    "Pride" => Some((Emotion::Pride, intensity)),
                    "Shame" => Some((Emotion::Shame, intensity)),
                    "Guilt" => Some((Emotion::Guilt, intensity)),
                    "Gratitude" => Some((Emotion::Gratitude, intensity)),
                    "Hope" => Some((Emotion::Hope, intensity)),
                    "Relief" => Some((Emotion::Relief, intensity)),
                    _ => None,
                }
            })
    }
    
    /// Decay emotional intensity over time
    pub fn decay(&mut self, rate: f32) {
        for (_, intensity) in self.emotions.iter_mut() {
            *intensity *= (1.0 - rate);
        }
        self.valence *= (1.0 - rate);
        self.arousal *= (1.0 - rate);
    }
}

/// Affective weighting system for memory importance
pub struct AffectiveWeighting {
    /// Weight factors for different emotional dimensions
    valence_weight: f32,
    arousal_weight: f32,
    stability_weight: f32,
}

impl AffectiveWeighting {
    pub fn new() -> Self {
        Self {
            valence_weight: 0.3,
            arousal_weight: 0.5,
            stability_weight: 0.2,
        }
    }
    
    /// Calculate importance score based on emotional state
    pub fn calculate_importance(&self, state: &EmotionalState) -> f32 {
        let valence_contrib = state.valence.abs() * self.valence_weight;
        let arousal_contrib = state.arousal * self.arousal_weight;
        let stability_contrib = (1.0 - state.stability) * self.stability_weight;
        
        (valence_contrib + arousal_contrib + stability_contrib).min(1.0)
    }
    
    /// Calculate mood congruent retrieval score
    pub fn mood_congruent_retrieval(&self, memory_emotion: &EmotionalState, 
                                   current_mood: &EmotionalState) -> f32 {
        // Calculate similarity between emotional states
        let valence_sim = 1.0 - (memory_emotion.valence - current_mood.valence).abs();
        let arousal_sim = 1.0 - (memory_emotion.arousal - current_mood.arousal).abs();
        
        (valence_sim * 0.7 + arousal_sim * 0.3).max(0.0)
    }
}

/// Emotional regulation system
pub struct EmotionalRegulation {
    /// Current emotional state
    pub current_state: Arc<RwLock<EmotionalState>>,
    /// Regulation history
    history: Vec<EmotionalState>,
    /// Regulation strategies
    strategies: Vec<String>,
    /// Regulation effectiveness
    effectiveness: f32,
}

impl EmotionalRegulation {
    pub fn new() -> Self {
        Self {
            current_state: Arc::new(RwLock::new(EmotionalState::neutral())),
            history: Vec::new(),
            strategies: vec![
                "reappraisal".to_string(),
                "suppression".to_string(),
                "acceptance".to_string(),
                "distraction".to_string(),
            ],
            effectiveness: 0.7,
        }
    }
    
    /// Update current emotional state
    pub fn update_state(&mut self, state: EmotionalState) {
        *self.current_state.write().unwrap() = state.clone();
        self.history.push(state);
        
        // Keep history limited
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }
    
    /// Apply regulation to reduce extreme emotions
    pub fn regulate(&mut self) {
        let mut state = self.current_state.write().unwrap();
        
        // Reduce extreme valence
        if state.valence.abs() > 0.8 {
            state.valence *= 0.8;
        }
        
        // Reduce high arousal
        if state.arousal > 0.8 {
            state.arousal *= 0.7;
        }
        
        // Increase stability
        state.stability = (state.stability + 0.1).min(1.0);
    }
    
    /// Detect emotional patterns
    pub fn detect_patterns(&self) -> Vec<String> {
        let mut patterns = Vec::new();
        
        if self.history.len() < 3 {
            return patterns;
        }
        
        // Check for mood swings
        let recent_valences: Vec<f32> = self.history.iter()
            .rev()
            .take(10)
            .map(|s| s.valence)
            .collect();
        
        let variance = Self::calculate_variance(&recent_valences);
        if variance > 0.5 {
            patterns.push("High emotional volatility detected".to_string());
        }
        
        // Check for persistent negative mood
        let avg_valence = recent_valences.iter().sum::<f32>() / recent_valences.len() as f32;
        if avg_valence < -0.5 {
            patterns.push("Persistent negative emotional state".to_string());
        }
        
        // Check for emotional numbing
        let avg_arousal = self.history.iter()
            .rev()
            .take(10)
            .map(|s| s.arousal)
            .sum::<f32>() / 10.0_f32.min(self.history.len() as f32);
        
        if avg_arousal < 0.2 {
            patterns.push("Low emotional arousal - possible numbing".to_string());
        }
        
        patterns
    }
    
    fn calculate_variance(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        
        variance.sqrt()
    }
}

/// Somatic marker system for gut feelings
pub struct SomaticMarker {
    /// Learned associations between patterns and emotions
    associations: HashMap<String, EmotionalState>,
    /// Learning rate
    learning_rate: f32,
}

impl SomaticMarker {
    pub fn new() -> Self {
        Self {
            associations: HashMap::new(),
            learning_rate: 0.1,
        }
    }
    
    /// Learn association between pattern and emotional outcome
    pub fn learn_association(&mut self, pattern: Vec<f32>, emotion: EmotionalState) {
        let pattern_key = Self::pattern_to_key(&pattern);
        
        if let Some(existing) = self.associations.get_mut(&pattern_key) {
            // Blend with existing association
            existing.valence = existing.valence * (1.0 - self.learning_rate) + 
                              emotion.valence * self.learning_rate;
            existing.arousal = existing.arousal * (1.0 - self.learning_rate) + 
                              emotion.arousal * self.learning_rate;
        } else {
            self.associations.insert(pattern_key, emotion);
        }
    }
    
    /// Get gut feeling for a pattern
    pub fn get_gut_feeling(&self, pattern: &[f32]) -> Option<EmotionalState> {
        let pattern_key = Self::pattern_to_key(pattern);
        
        // Exact match
        if let Some(emotion) = self.associations.get(&pattern_key) {
            return Some(emotion.clone());
        }
        
        // Find similar patterns
        for (key, emotion) in &self.associations {
            if Self::patterns_similar(&pattern_key, key) {
                return Some(emotion.clone());
            }
        }
        
        None
    }
    
    fn pattern_to_key(pattern: &[f32]) -> String {
        pattern.iter()
            .map(|&v| format!("{:.1}", v))
            .collect::<Vec<_>>()
            .join(",")
    }
    
    fn patterns_similar(p1: &str, p2: &str) -> bool {
        let vals1: Vec<f32> = p1.split(',')
            .filter_map(|s| s.parse().ok())
            .collect();
        let vals2: Vec<f32> = p2.split(',')
            .filter_map(|s| s.parse().ok())
            .collect();
        
        if vals1.len() != vals2.len() {
            return false;
        }
        
        let diff: f32 = vals1.iter()
            .zip(vals2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        
        (diff / vals1.len() as f32) < 0.2
    }
}

/// Empathy simulator for theory of mind
pub struct EmpathySimulator {
    /// Models of other agents' emotional states
    pub other_models: HashMap<String, EmotionalState>,
    /// Contagion factor
    contagion_factor: f32,
    /// Empathy accuracy
    accuracy: f32,
}

impl EmpathySimulator {
    pub fn new() -> Self {
        Self {
            other_models: HashMap::new(),
            contagion_factor: 0.3,
            accuracy: 0.7,
        }
    }
    
    /// Simulate another agent's emotional state
    pub fn simulate_other(&self, agent_id: &str, context: &[f32]) -> EmotionalState {
        if let Some(model) = self.other_models.get(agent_id) {
            // Add noise based on accuracy
            let mut simulated = model.clone();
            let noise = 1.0 - self.accuracy;
            simulated.valence += (rand::random::<f32>() - 0.5) * noise;
            simulated.arousal += (rand::random::<f32>() - 0.5) * noise;
            simulated
        } else {
            // Default neutral if no model
            EmotionalState::neutral()
        }
    }
    
    /// Apply emotional contagion
    pub fn emotional_contagion(&self, my_state: &mut EmotionalState, other_state: &EmotionalState) {
        my_state.valence = my_state.valence * (1.0 - self.contagion_factor) + 
                          other_state.valence * self.contagion_factor;
        my_state.arousal = my_state.arousal * (1.0 - self.contagion_factor) + 
                          other_state.arousal * self.contagion_factor;
    }
}

/// Emotional intelligence metrics
pub struct EmotionalIntelligence {
    /// Self-awareness score
    pub self_awareness: f32,
    /// Self-regulation score
    pub self_regulation: f32,
    /// Social awareness score
    pub social_awareness: f32,
    /// Relationship management score
    pub relationship_skill: f32,
}

impl EmotionalIntelligence {
    pub fn new() -> Self {
        Self {
            self_awareness: 0.5,
            self_regulation: 0.5,
            social_awareness: 0.5,
            relationship_skill: 0.5,
        }
    }
    
    /// Calculate overall EQ score
    pub fn eq_score(&self) -> f32 {
        (self.self_awareness + self.self_regulation + 
         self.social_awareness + self.relationship_skill) / 4.0
    }
    
    /// Update from feedback
    pub fn update_from_feedback(&mut self, dimension: &str, success: bool) {
        let delta = if success { 0.05 } else { -0.03 };
        
        match dimension {
            "self_awareness" => self.self_awareness = (self.self_awareness + delta).clamp(0.0, 1.0),
            "self_regulation" => self.self_regulation = (self.self_regulation + delta).clamp(0.0, 1.0),
            "social_awareness" => self.social_awareness = (self.social_awareness + delta).clamp(0.0, 1.0),
            "relationship_skill" => self.relationship_skill = (self.relationship_skill + delta).clamp(0.0, 1.0),
            _ => {}
        }
    }
}

/// Trait for emotional memory operations
pub trait EmotionalMemory {
    fn tag_with_emotion(&mut self, memory_key: &str, state: EmotionalState);
    fn retrieve_by_emotion(&self, current_state: &EmotionalState, threshold: f32) -> Vec<String>;
    fn get_emotional_context(&self, memory_key: &str) -> Option<EmotionalState>;
    fn decay_emotions(&mut self, decay_rate: f32);
}