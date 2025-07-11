//! Emotional Processing Module for NeuralClaude
//! 
//! This module provides emotional intelligence capabilities for the consciousness system,
//! including emotion recognition, valence processing, and affective state management.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use ndarray::Array1;

/// Emotional processor for consciousness system
#[derive(Debug)]
pub struct EmotionalProcessor {
    /// Current emotional state
    current_state: Arc<Mutex<EmotionalState>>,
    
    /// Emotional memory
    emotional_memory: Arc<Mutex<EmotionalMemory>>,
    
    /// Emotion recognition system
    emotion_recognition: Arc<Mutex<EmotionRecognition>>,
    
    /// Valence processing system
    valence_processor: Arc<Mutex<ValenceProcessor>>,
    
    /// Affective learning system
    affective_learning: Arc<Mutex<AffectiveLearning>>,
    
    /// Configuration
    config: EmotionalConfig,
}

/// Current emotional state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    /// Primary emotions with intensities
    pub emotions: HashMap<String, f32>,
    
    /// Overall valence (-1.0 to 1.0)
    pub valence: f32,
    
    /// Arousal level (0.0 to 1.0)
    pub arousal: f32,
    
    /// Emotional stability
    pub stability: f32,
    
    /// Emotional complexity
    pub complexity: f32,
    
    /// Timestamp of last update
    pub timestamp: u64,
}

/// Emotional memory system
#[derive(Debug)]
pub struct EmotionalMemory {
    /// Emotional experiences
    experiences: Vec<EmotionalExperience>,
    
    /// Emotional patterns
    patterns: Vec<EmotionalPattern>,
    
    /// Emotional associations
    associations: HashMap<String, Vec<String>>,
    
    /// Memory capacity
    capacity: usize,
}

/// Emotional experience record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalExperience {
    /// Experience identifier
    pub id: String,
    
    /// Emotional state at time of experience
    pub emotional_state: EmotionalState,
    
    /// Context of experience
    pub context: String,
    
    /// Intensity of experience
    pub intensity: f32,
    
    /// Duration of experience
    pub duration: u64,
    
    /// Timestamp
    pub timestamp: u64,
}

/// Emotional pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalPattern {
    /// Pattern identifier
    pub id: String,
    
    /// Pattern description
    pub description: String,
    
    /// Pattern triggers
    pub triggers: Vec<String>,
    
    /// Pattern frequency
    pub frequency: u32,
    
    /// Pattern strength
    pub strength: f32,
    
    /// Last occurrence
    pub last_occurrence: u64,
}

/// Emotion recognition system
#[derive(Debug)]
pub struct EmotionRecognition {
    /// Emotion classifiers
    classifiers: HashMap<String, EmotionClassifier>,
    
    /// Recognition confidence
    confidence: f32,
    
    /// Recognition history
    history: Vec<RecognitionEvent>,
}

/// Emotion classifier
#[derive(Debug, Clone)]
pub struct EmotionClassifier {
    /// Emotion name
    pub name: String,
    
    /// Classification weights
    pub weights: Array1<f32>,
    
    /// Classification threshold
    pub threshold: f32,
    
    /// Classifier accuracy
    pub accuracy: f32,
}

/// Recognition event
#[derive(Debug, Clone)]
pub struct RecognitionEvent {
    /// Recognized emotion
    pub emotion: String,
    
    /// Recognition confidence
    pub confidence: f32,
    
    /// Input features
    pub features: Array1<f32>,
    
    /// Timestamp
    pub timestamp: u64,
}

/// Valence processing system
#[derive(Debug)]
pub struct ValenceProcessor {
    /// Current valence
    current_valence: f32,
    
    /// Valence history
    valence_history: Vec<ValenceEvent>,
    
    /// Valence smoothing factor
    smoothing_factor: f32,
    
    /// Valence sensitivity
    sensitivity: f32,
}

/// Valence event
#[derive(Debug, Clone)]
pub struct ValenceEvent {
    /// Valence value
    pub valence: f32,
    
    /// Event context
    pub context: String,
    
    /// Event intensity
    pub intensity: f32,
    
    /// Timestamp
    pub timestamp: u64,
}

/// Affective learning system
#[derive(Debug)]
pub struct AffectiveLearning {
    /// Learning rate
    learning_rate: f32,
    
    /// Adaptation rate
    adaptation_rate: f32,
    
    /// Learning history
    learning_history: Vec<LearningEvent>,
    
    /// Learned associations
    associations: HashMap<String, f32>,
}

/// Learning event
#[derive(Debug, Clone)]
pub struct LearningEvent {
    /// Learning context
    pub context: String,
    
    /// Learning outcome
    pub outcome: f32,
    
    /// Learning strength
    pub strength: f32,
    
    /// Timestamp
    pub timestamp: u64,
}

/// Emotional configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalConfig {
    /// Emotional sensitivity
    pub sensitivity: f32,
    
    /// Emotional memory capacity
    pub memory_capacity: usize,
    
    /// Valence smoothing factor
    pub valence_smoothing: f32,
    
    /// Learning rate
    pub learning_rate: f32,
    
    /// Update frequency
    pub update_frequency: u32,
}

impl Default for EmotionalConfig {
    fn default() -> Self {
        Self {
            sensitivity: 0.7,
            memory_capacity: 1000,
            valence_smoothing: 0.8,
            learning_rate: 0.01,
            update_frequency: 10,
        }
    }
}

impl EmotionalProcessor {
    /// Create new emotional processor
    pub fn new() -> Self {
        Self::with_config(EmotionalConfig::default())
    }
    
    /// Create emotional processor with configuration
    pub fn with_config(config: EmotionalConfig) -> Self {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            current_state: Arc::new(Mutex::new(EmotionalState {
                emotions: HashMap::new(),
                valence: 0.0,
                arousal: 0.0,
                stability: 0.5,
                complexity: 0.0,
                timestamp: current_time,
            })),
            emotional_memory: Arc::new(Mutex::new(EmotionalMemory {
                experiences: Vec::new(),
                patterns: Vec::new(),
                associations: HashMap::new(),
                capacity: config.memory_capacity,
            })),
            emotion_recognition: Arc::new(Mutex::new(EmotionRecognition {
                classifiers: Self::create_emotion_classifiers(),
                confidence: 0.0,
                history: Vec::new(),
            })),
            valence_processor: Arc::new(Mutex::new(ValenceProcessor {
                current_valence: 0.0,
                valence_history: Vec::new(),
                smoothing_factor: config.valence_smoothing,
                sensitivity: config.sensitivity,
            })),
            affective_learning: Arc::new(Mutex::new(AffectiveLearning {
                learning_rate: config.learning_rate,
                adaptation_rate: 0.1,
                learning_history: Vec::new(),
                associations: HashMap::new(),
            })),
            config,
        }
    }
    
    /// Create emotion classifiers
    fn create_emotion_classifiers() -> HashMap<String, EmotionClassifier> {
        let mut classifiers = HashMap::new();
        
        // Basic emotions
        let emotions = vec![
            "joy", "sadness", "anger", "fear", "surprise", "disgust",
            "anticipation", "trust", "love", "curiosity", "confusion",
            "excitement", "anxiety", "contentment", "frustration",
        ];
        
        for emotion in emotions {
            classifiers.insert(emotion.to_string(), EmotionClassifier {
                name: emotion.to_string(),
                weights: Array1::from_vec(vec![0.1; 768]), // Default weights
                threshold: 0.5,
                accuracy: 0.7,
            });
        }
        
        classifiers
    }
    
    /// Process emotional input
    pub fn process_emotion(&self, input: EmotionalInput) -> EmotionalOutput {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Recognize emotions
        let recognized_emotions = self.recognize_emotions(&input.features);
        
        // Process valence
        let valence = self.process_valence(&input);
        
        // Update emotional state
        self.update_emotional_state(&recognized_emotions, valence);
        
        // Store experience
        self.store_emotional_experience(&input, &recognized_emotions, valence);
        
        // Generate output
        let current_state = self.current_state.lock().unwrap();
        EmotionalOutput {
            emotions: recognized_emotions,
            valence: current_state.valence,
            arousal: current_state.arousal,
            stability: current_state.stability,
            complexity: current_state.complexity,
            timestamp: current_time,
        }
    }
    
    /// Recognize emotions from input features
    fn recognize_emotions(&self, features: &Array1<f32>) -> HashMap<String, f32> {
        let mut recognized = HashMap::new();
        let recognition = self.emotion_recognition.lock().unwrap();
        
        for (emotion_name, classifier) in &recognition.classifiers {
            let score = self.classify_emotion(features, classifier);
            if score > classifier.threshold {
                recognized.insert(emotion_name.clone(), score);
            }
        }
        
        recognized
    }
    
    /// Classify emotion using classifier
    fn classify_emotion(&self, features: &Array1<f32>, classifier: &EmotionClassifier) -> f32 {
        // Simple dot product classification
        let dot_product = features.dot(&classifier.weights);
        let magnitude = classifier.weights.dot(&classifier.weights).sqrt();
        
        if magnitude == 0.0 {
            0.0
        } else {
            (dot_product / magnitude).max(0.0).min(1.0)
        }
    }
    
    /// Process valence from input
    fn process_valence(&self, input: &EmotionalInput) -> f32 {
        let mut valence_processor = self.valence_processor.lock().unwrap();
        
        // Calculate new valence
        let new_valence = input.valence * input.intensity;
        
        // Apply smoothing
        let smoothed_valence = valence_processor.current_valence * valence_processor.smoothing_factor +
                              new_valence * (1.0 - valence_processor.smoothing_factor);
        
        // Update processor state
        valence_processor.current_valence = smoothed_valence;
        
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        valence_processor.valence_history.push(ValenceEvent {
            valence: smoothed_valence,
            context: input.context.clone(),
            intensity: input.intensity,
            timestamp: current_time,
        });
        
        smoothed_valence
    }
    
    /// Update emotional state
    fn update_emotional_state(&self, emotions: &HashMap<String, f32>, valence: f32) {
        let mut state = self.current_state.lock().unwrap();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Update emotions
        state.emotions = emotions.clone();
        
        // Update valence
        state.valence = valence;
        
        // Calculate arousal
        let total_emotion_intensity: f32 = emotions.values().sum();
        state.arousal = (total_emotion_intensity / emotions.len().max(1) as f32).min(1.0);
        
        // Calculate stability
        let emotion_variance = self.calculate_emotion_variance(emotions);
        state.stability = 1.0 - emotion_variance;
        
        // Calculate complexity
        state.complexity = emotions.len() as f32 / 15.0; // Normalize by max emotions
        
        state.timestamp = current_time;
    }
    
    /// Calculate variance in emotions
    fn calculate_emotion_variance(&self, emotions: &HashMap<String, f32>) -> f32 {
        if emotions.is_empty() {
            return 0.0;
        }
        
        let mean: f32 = emotions.values().sum::<f32>() / emotions.len() as f32;
        let variance: f32 = emotions.values()
            .map(|&v| (v - mean).powi(2))
            .sum::<f32>() / emotions.len() as f32;
        
        variance.sqrt().min(1.0)
    }
    
    /// Store emotional experience
    fn store_emotional_experience(&self, input: &EmotionalInput, emotions: &HashMap<String, f32>, valence: f32) {
        let mut memory = self.emotional_memory.lock().unwrap();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let experience = EmotionalExperience {
            id: format!("exp_{}", current_time),
            emotional_state: EmotionalState {
                emotions: emotions.clone(),
                valence,
                arousal: input.intensity,
                stability: 0.5, // Default
                complexity: emotions.len() as f32 / 15.0,
                timestamp: current_time,
            },
            context: input.context.clone(),
            intensity: input.intensity,
            duration: input.duration,
            timestamp: current_time,
        };
        
        memory.experiences.push(experience);
        
        // Maintain capacity
        if memory.experiences.len() > memory.capacity {
            memory.experiences.remove(0);
        }
    }
    
    /// Get current valence
    pub fn get_valence(&self) -> f32 {
        let state = self.current_state.lock().unwrap();
        state.valence
    }
    
    /// Get current arousal
    pub fn get_arousal(&self) -> f32 {
        let state = self.current_state.lock().unwrap();
        state.arousal
    }
    
    /// Get current emotional state
    pub fn get_emotional_state(&self) -> EmotionalState {
        let state = self.current_state.lock().unwrap();
        state.clone()
    }
    
    /// Get dominant emotion
    pub fn get_dominant_emotion(&self) -> Option<(String, f32)> {
        let state = self.current_state.lock().unwrap();
        
        state.emotions.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(emotion, &intensity)| (emotion.clone(), intensity))
    }
    
    /// Learn from emotional feedback
    pub fn learn_from_feedback(&self, context: String, outcome: f32) {
        let mut learning = self.affective_learning.lock().unwrap();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Update associations
        let current_association = learning.associations.get(&context).unwrap_or(&0.0);
        let learning_rate = learning.learning_rate;
        let new_association = current_association + learning_rate * outcome;
        learning.associations.insert(context.clone(), new_association);
        
        // Store learning event
        learning.learning_history.push(LearningEvent {
            context,
            outcome,
            strength: learning_rate,
            timestamp: current_time,
        });
    }
    
    /// Get emotional patterns
    pub fn get_emotional_patterns(&self) -> Vec<EmotionalPattern> {
        let memory = self.emotional_memory.lock().unwrap();
        memory.patterns.clone()
    }
    
    /// Analyze emotional trends
    pub fn analyze_emotional_trends(&self) -> EmotionalTrends {
        let memory = self.emotional_memory.lock().unwrap();
        let valence_processor = self.valence_processor.lock().unwrap();
        
        if memory.experiences.is_empty() {
            return EmotionalTrends::default();
        }
        
        // Calculate trends
        let recent_experiences: Vec<_> = memory.experiences.iter()
            .rev()
            .take(10)
            .collect();
        
        let avg_valence = recent_experiences.iter()
            .map(|e| e.emotional_state.valence)
            .sum::<f32>() / recent_experiences.len() as f32;
        
        let avg_arousal = recent_experiences.iter()
            .map(|e| e.emotional_state.arousal)
            .sum::<f32>() / recent_experiences.len() as f32;
        
        let avg_stability = recent_experiences.iter()
            .map(|e| e.emotional_state.stability)
            .sum::<f32>() / recent_experiences.len() as f32;
        
        EmotionalTrends {
            valence_trend: avg_valence,
            arousal_trend: avg_arousal,
            stability_trend: avg_stability,
            dominant_emotions: self.get_dominant_emotions(&recent_experiences),
            pattern_frequency: memory.patterns.len() as f32,
        }
    }
    
    /// Get dominant emotions from experiences
    fn get_dominant_emotions(&self, experiences: &[&EmotionalExperience]) -> Vec<(String, f32)> {
        let mut emotion_totals: HashMap<String, f32> = HashMap::new();
        
        for experience in experiences {
            for (emotion, &intensity) in &experience.emotional_state.emotions {
                *emotion_totals.entry(emotion.clone()).or_insert(0.0) += intensity;
            }
        }
        
        let mut dominant: Vec<_> = emotion_totals.into_iter().collect();
        dominant.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        dominant.truncate(5); // Top 5 emotions
        
        dominant
    }
}

/// Input for emotional processing
#[derive(Debug, Clone)]
pub struct EmotionalInput {
    /// Feature vector for emotion recognition
    pub features: Array1<f32>,
    
    /// Context information
    pub context: String,
    
    /// Emotional valence (-1.0 to 1.0)
    pub valence: f32,
    
    /// Emotional intensity (0.0 to 1.0)
    pub intensity: f32,
    
    /// Duration of emotional event
    pub duration: u64,
}

/// Output from emotional processing
#[derive(Debug, Clone)]
pub struct EmotionalOutput {
    /// Recognized emotions with intensities
    pub emotions: HashMap<String, f32>,
    
    /// Overall valence
    pub valence: f32,
    
    /// Arousal level
    pub arousal: f32,
    
    /// Emotional stability
    pub stability: f32,
    
    /// Emotional complexity
    pub complexity: f32,
    
    /// Timestamp
    pub timestamp: u64,
}

/// Emotional trends analysis
#[derive(Debug, Clone)]
pub struct EmotionalTrends {
    /// Valence trend
    pub valence_trend: f32,
    
    /// Arousal trend
    pub arousal_trend: f32,
    
    /// Stability trend
    pub stability_trend: f32,
    
    /// Dominant emotions
    pub dominant_emotions: Vec<(String, f32)>,
    
    /// Pattern frequency
    pub pattern_frequency: f32,
}

impl Default for EmotionalTrends {
    fn default() -> Self {
        Self {
            valence_trend: 0.0,
            arousal_trend: 0.0,
            stability_trend: 0.5,
            dominant_emotions: Vec::new(),
            pattern_frequency: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_emotional_processor_creation() {
        let processor = EmotionalProcessor::new();
        let state = processor.get_emotional_state();
        
        assert_eq!(state.valence, 0.0);
        assert_eq!(state.arousal, 0.0);
    }
    
    #[test]
    fn test_emotion_processing() {
        let processor = EmotionalProcessor::new();
        
        let input = EmotionalInput {
            features: Array1::from_vec(vec![0.5; 768]),
            context: "test_context".to_string(),
            valence: 0.7,
            intensity: 0.8,
            duration: 1000,
        };
        
        let output = processor.process_emotion(input);
        
        assert!(output.valence >= 0.0);
        assert!(output.arousal >= 0.0);
    }
    
    #[test]
    fn test_valence_processing() {
        let processor = EmotionalProcessor::new();
        
        let input = EmotionalInput {
            features: Array1::from_vec(vec![0.1; 768]),
            context: "positive_context".to_string(),
            valence: 0.8,
            intensity: 0.9,
            duration: 500,
        };
        
        processor.process_emotion(input);
        let valence = processor.get_valence();
        
        assert!(valence > 0.0);
    }
    
    #[test]
    fn test_emotional_learning() {
        let processor = EmotionalProcessor::new();
        
        processor.learn_from_feedback("learning_context".to_string(), 0.8);
        
        // Test would verify that learning occurred
        // In a real implementation, this would check association updates
    }
    
    #[test]
    fn test_emotional_trends() {
        let processor = EmotionalProcessor::new();
        
        let trends = processor.analyze_emotional_trends();
        
        assert_eq!(trends.valence_trend, 0.0);
        assert_eq!(trends.arousal_trend, 0.0);
    }
}