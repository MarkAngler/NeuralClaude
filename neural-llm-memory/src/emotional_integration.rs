use crate::emotional::EmotionalState;
use crate::emotional_types::{
    EmotionalMemory, AffectiveWeighting, EmotionalRegulation,
    SomaticMarker, EmpathySimulator, EmotionalIntelligence, Emotion
};
use crate::memory::{MemoryBank, MemoryOperations, MemoryKey, MemoryValue, MemoryMetadata};
// use crate::metacognition::{MetaCognitiveMonitor, CognitivePattern};
use crate::adaptive::AdaptiveMemoryModule;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use ndarray::Array1;

/// Integrates emotional intelligence with the memory and metacognitive systems
pub struct EmotionallyAwareMemory {
    /// Core memory bank
    memory: Arc<RwLock<MemoryBank>>,
    /// Emotional states associated with memories
    emotional_tags: Arc<RwLock<HashMap<String, EmotionalState>>>,
    /// Affective weighting system
    affective_weighting: AffectiveWeighting,
    /// Emotional regulation
    regulation: Arc<RwLock<EmotionalRegulation>>,
    /// Somatic marker system
    somatic_markers: Arc<RwLock<SomaticMarker>>,
    /// Empathy simulator
    empathy: Arc<RwLock<EmpathySimulator>>,
    /// Emotional intelligence metrics
    eq_metrics: Arc<RwLock<EmotionalIntelligence>>,
}

impl EmotionallyAwareMemory {
    pub fn new(memory_bank: Arc<RwLock<MemoryBank>>) -> Self {
        Self {
            memory: memory_bank,
            emotional_tags: Arc::new(RwLock::new(HashMap::new())),
            affective_weighting: AffectiveWeighting::new(),
            regulation: Arc::new(RwLock::new(EmotionalRegulation::new())),
            somatic_markers: Arc::new(RwLock::new(SomaticMarker::new())),
            empathy: Arc::new(RwLock::new(EmpathySimulator::new())),
            eq_metrics: Arc::new(RwLock::new(EmotionalIntelligence::new())),
        }
    }
    
    /// Store memory with emotional context
    pub fn store_with_emotion(&self, key: String, content: Array1<f32>, 
                            emotional_state: EmotionalState) -> Result<(), String> {
        // Calculate importance based on emotional state
        let importance = self.affective_weighting.calculate_importance(&emotional_state);
        
        // Store in memory bank with emotional weighting
        let mut memory = self.memory.write().map_err(|_| "Lock error")?;
        let memory_key = MemoryKey::new(key.clone(), &emotional_state.emotions.iter().map(|(k, v)| format!("{}: {}", k, v)).collect::<Vec<_>>().join(", "));
        let memory_value = MemoryValue {
            embedding: content.to_vec(),
            content: key.clone(),
            metadata: MemoryMetadata {
                importance,
                ..Default::default()
            },
        };
        memory.store(memory_key, memory_value).map_err(|e| format!("{}", e))?;
        
        // Tag with emotional state
        let mut tags = self.emotional_tags.write().map_err(|_| "Lock error")?;
        tags.insert(key.clone(), emotional_state.clone());
        
        // Update somatic markers if high arousal
        if emotional_state.arousal > 0.7 {
            let mut markers = self.somatic_markers.write().map_err(|_| "Lock error")?;
            markers.learn_association(content.to_vec(), emotional_state.clone());
        }
        
        // Update regulation system
        let mut regulation = self.regulation.write().map_err(|_| "Lock error")?;
        regulation.update_state(emotional_state);
        
        Ok(())
    }
    
    /// Retrieve memories with mood-congruent bias
    pub fn retrieve_mood_congruent(&self, query: &Array1<f32>, 
                                  current_mood: &EmotionalState, 
                                  k: usize) -> Vec<(String, f32)> {
        // Convert 1D query to 2D for memory search
        let query_2d = query.clone().insert_axis(ndarray::Axis(0));
        
        let memory = self.memory.read().unwrap();
        let base_results = memory.search(&query_2d, k * 2); // Get more candidates
        
        let tags = self.emotional_tags.read().unwrap();
        
        // Re-rank based on mood congruence
        let mut scored_results: Vec<(String, f32, f32)> = base_results.into_iter()
            .filter_map(|(key, _value, similarity)| {
                let key_str = key.id.clone(); // Get the string ID from MemoryKey
                if let Some(memory_emotion) = tags.get(&key_str) {
                    let mood_score = self.affective_weighting
                        .mood_congruent_retrieval(memory_emotion, current_mood);
                    let combined_score = similarity * 0.7 + mood_score * 0.3;
                    Some((key_str, combined_score, similarity))
                } else {
                    Some((key_str, similarity * 0.7, similarity)) // Neutral memories get lower weight
                }
            })
            .collect();
        
        // Sort by combined score and take top k
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_results.into_iter()
            .take(k)
            .map(|(key, score, _)| (key, score))
            .collect()
    }
    
    /// Get intuitive assessment of a pattern
    pub fn get_intuition(&self, pattern: &Array1<f32>) -> Option<EmotionalState> {
        let markers = self.somatic_markers.read().unwrap();
        markers.get_gut_feeling(&pattern.to_vec())
    }
    
    /// Simulate emotional response for another agent
    pub fn simulate_empathy(&self, agent_id: &str, context: &Array1<f32>) -> EmotionalState {
        let empathy = self.empathy.write().unwrap();
        empathy.simulate_other(agent_id, &context.to_vec())
    }
    
    /// Apply emotional contagion from interaction
    pub fn apply_contagion(&self, other_emotion: &EmotionalState) {
        let regulation = self.regulation.write().unwrap();
        let mut current = regulation.current_state.write().unwrap();
        
        let empathy = self.empathy.read().unwrap();
        empathy.emotional_contagion(&mut current, other_emotion);
    }
    
    /// Get current emotional intelligence score
    pub fn get_eq_score(&self) -> f32 {
        let eq = self.eq_metrics.read().unwrap();
        eq.eq_score()
    }
    
    /// Update EQ based on interaction success
    pub fn update_eq(&self, dimension: &str, success: bool) {
        let mut eq = self.eq_metrics.write().unwrap();
        eq.update_from_feedback(dimension, success);
    }
}

impl EmotionalMemory for EmotionallyAwareMemory {
    fn tag_with_emotion(&mut self, memory_key: &str, state: EmotionalState) {
        let mut tags = self.emotional_tags.write().unwrap();
        tags.insert(memory_key.to_string(), state);
    }
    
    fn retrieve_by_emotion(&self, current_state: &EmotionalState, threshold: f32) -> Vec<String> {
        let tags = self.emotional_tags.read().unwrap();
        
        tags.iter()
            .filter_map(|(key, state)| {
                let similarity = self.affective_weighting
                    .mood_congruent_retrieval(state, current_state);
                if similarity > threshold {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect()
    }
    
    fn get_emotional_context(&self, memory_key: &str) -> Option<EmotionalState> {
        let tags = self.emotional_tags.read().unwrap();
        tags.get(memory_key).cloned()
    }
    
    fn decay_emotions(&mut self, decay_rate: f32) {
        let mut tags = self.emotional_tags.write().unwrap();
        for (_, state) in tags.iter_mut() {
            state.decay(decay_rate);
        }
    }
}

/// Extension trait for MetaCognitiveMonitor to include emotional awareness
pub trait EmotionalMetaCognition {
    /// Monitor emotional impact on decision making
    fn monitor_emotional_influence(&self, decision: &str, emotional_state: &EmotionalState) -> f32;
    
    /// Detect emotional biases in reasoning
    fn detect_emotional_bias(&self, pattern: &str, emotion: &EmotionalState) -> Vec<String>;
    
    /// Evaluate emotional regulation effectiveness
    fn evaluate_regulation(&self, before: &EmotionalState, after: &EmotionalState) -> f32;
}

/// Integrate emotional system with adaptive memory
pub struct EmotionalAdaptiveMemory {
    emotional_memory: Arc<EmotionallyAwareMemory>,
}

impl EmotionalAdaptiveMemory {
    pub fn new(emotional: Arc<EmotionallyAwareMemory>) -> Self {
        Self {
            emotional_memory: emotional,
        }
    }
    
    /// Store with emotional context
    pub fn store_adaptive_emotional(&mut self, key: String, content: Array1<f32>, 
                                  emotion: EmotionalState) -> Result<(), String> {
        // Store with emotional context
        self.emotional_memory.store_with_emotion(key.clone(), content.clone(), emotion.clone())?;
        
        Ok(())
    }
    
    /// Retrieve using emotional similarity
    pub fn retrieve_holistic(&self, query: &Array1<f32>, 
                           current_emotion: &EmotionalState, 
                           k: usize) -> Vec<(String, f32)> {
        // Get mood-congruent results
        self.emotional_memory
            .retrieve_mood_congruent(query, current_emotion, k)
    }
    
    /// Learn from emotional feedback
    pub fn learn_from_emotion(&mut self, success: bool, emotion: EmotionalState) {
        // Update emotional intelligence
        let dimension = if emotion.dominant_emotion()
            .map(|(e, _)| matches!(e, Emotion::Joy | Emotion::Gratitude))
            .unwrap_or(false) {
            "self_regulation"
        } else {
            "self_awareness"
        };
        
        self.emotional_memory.update_eq(dimension, success);
    }
}

/// Emotional decision-making integration
pub struct EmotionalDecisionMaker {
    emotional_memory: Arc<EmotionallyAwareMemory>,
    decision_history: Vec<(String, EmotionalState, f32)>, // decision, emotion, outcome
}

impl EmotionalDecisionMaker {
    pub fn new(emotional_memory: Arc<EmotionallyAwareMemory>) -> Self {
        Self {
            emotional_memory,
            decision_history: Vec::new(),
        }
    }
    
    /// Make decision considering emotional factors
    pub fn decide_with_emotion(&mut self, options: Vec<(String, Array1<f32>)>, 
                             current_emotion: &EmotionalState) -> String {
        let mut best_option = String::new();
        let mut best_score = f32::NEG_INFINITY;
        
        for (option, features) in options {
            // Get gut feeling about this option
            let intuition = self.emotional_memory.get_intuition(&features);
            
            // Calculate emotional compatibility
            let emotional_score = if let Some(gut_feeling) = intuition {
                // Positive gut feeling in positive mood = good
                // Negative gut feeling in risk-averse mood = good (caution)
                if current_emotion.valence > 0.5 && gut_feeling.valence > 0.0 {
                    0.8
                } else if current_emotion.arousal < 0.3 && gut_feeling.arousal < 0.5 {
                    0.7 // Low arousal prefers calm options
                } else if gut_feeling.dominant_emotion()
                    .map(|(e, _)| matches!(e, Emotion::Fear))
                    .unwrap_or(false) && current_emotion.arousal > 0.7 {
                    0.2 // High arousal + fear = avoid
                } else {
                    0.5
                }
            } else {
                0.5 // Neutral for unknown options
            };
            
            // Combine with rational features (simplified)
            let rational_score: f32 = features.iter().sum::<f32>() / features.len() as f32;
            let combined_score = rational_score * 0.6 + emotional_score * 0.4;
            
            if combined_score > best_score {
                best_score = combined_score;
                best_option = option.clone();
            }
        }
        
        // Record decision with emotional context
        self.decision_history.push((best_option.clone(), current_emotion.clone(), 0.0));
        
        best_option
    }
    
    /// Update decision history with outcome
    pub fn record_outcome(&mut self, decision: &str, outcome: f32) {
        if let Some(entry) = self.decision_history.iter_mut()
            .find(|(d, _, _)| d == decision) {
            entry.2 = outcome;
            
            // Learn association for future decisions
            let mut markers = self.emotional_memory.somatic_markers.write().unwrap();
            let outcome_emotion = if outcome > 0.5 {
                EmotionalState::from_emotion(Emotion::Joy, outcome)
            } else {
                EmotionalState::from_emotion(Emotion::Sadness, 1.0 - outcome)
            };
            
            // Simple feature vector from decision
            let features = decision.as_bytes().iter()
                .take(10)
                .map(|&b| b as f32 / 255.0)
                .collect();
            
            markers.learn_association(features, outcome_emotion);
        }
    }
}


