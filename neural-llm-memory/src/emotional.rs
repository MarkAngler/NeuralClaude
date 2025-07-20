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

impl EmotionalProcessor {
    /// Create new emotional processor
    pub fn new() -> Self {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            current_state: Arc::new(Mutex::new(EmotionalState {
                emotions: HashMap::new(),
                valence: 0.0,
                arousal: 0.5,
                stability: 0.8,
                complexity: 0.2,
                timestamp: current_time,
            })),
            emotional_memory: Arc::new(Mutex::new(EmotionalMemory {
                episodes: Vec::new(),
                patterns: Vec::new(),
                associations: HashMap::new(),
            })),
            emotion_recognition: Arc::new(Mutex::new(EmotionRecognition {
                recognition_models: Vec::new(),
                feature_extractors: Vec::new(),
                confidence_threshold: 0.7,
            })),
            valence_processor: Arc::new(Mutex::new(ValenceProcessor {
                valence_history: Vec::new(),
                valence_trends: Vec::new(),
                adaptation_rate: 0.1,
            })),
            affective_learning: Arc::new(Mutex::new(AffectiveLearning {
                learning_episodes: Vec::new(),
                emotion_outcomes: HashMap::new(),
                learning_rate: 0.01,
            })),
            config: EmotionalConfig::default(),
        }
    }
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
    episodes: Vec<EmotionalExperience>,
    
    /// Emotional patterns
    patterns: Vec<EmotionalPattern>,
    
    /// Emotional associations
    associations: HashMap<String, Vec<String>>,
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
    /// Recognition models
    recognition_models: Vec<RecognitionModel>,
    
    /// Feature extractors
    feature_extractors: Vec<FeatureExtractor>,
    
    /// Confidence threshold
    confidence_threshold: f32,
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
    /// Valence history
    valence_history: Vec<ValenceEvent>,
    
    /// Valence trends
    valence_trends: Vec<ValenceTrend>,
    
    /// Adaptation rate
    adaptation_rate: f32,
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
    /// Learning episodes
    learning_episodes: Vec<LearningEvent>,
    
    /// Emotion outcomes
    emotion_outcomes: HashMap<String, f32>,
    
    /// Learning rate
    learning_rate: f32,
}

/// Recognition model
#[derive(Debug, Clone)]
pub struct RecognitionModel {
    /// Model name
    pub name: String,
    /// Model weights
    pub weights: Array1<f32>,
}

/// Feature extractor
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Extractor name
    pub name: String,
    /// Feature dimension
    pub dimension: usize,
}

/// Valence trend
#[derive(Debug, Clone)]
pub struct ValenceTrend {
    /// Trend direction
    pub direction: f32,
    /// Trend strength
    pub strength: f32,
    /// Trend duration
    pub duration: u64,
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
                episodes: Vec::new(),
                patterns: Vec::new(),
                associations: HashMap::new(),
            })),
            emotion_recognition: Arc::new(Mutex::new(EmotionRecognition {
                recognition_models: Vec::new(),
                feature_extractors: Vec::new(),
                confidence_threshold: 0.7,
            })),
            valence_processor: Arc::new(Mutex::new(ValenceProcessor {
                valence_history: Vec::new(),
                valence_trends: Vec::new(),
                adaptation_rate: 0.1,
            })),
            affective_learning: Arc::new(Mutex::new(AffectiveLearning {
                learning_episodes: Vec::new(),
                emotion_outcomes: HashMap::new(),
                learning_rate: config.learning_rate,
            })),
            config,
        }
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
        
        // Basic emotion recognition based on threshold
        // This is a simplified version - proper implementation would use recognition_models
        let base_emotions = vec!["joy", "sadness", "anger", "fear", "surprise", "disgust"];
        for emotion in base_emotions {
            let score = self.classify_emotion_simple(features, emotion);
            if score > recognition.confidence_threshold {
                recognized.insert(emotion.to_string(), score);
            }
        }
        
        recognized
    }
    
    /// Classify emotion using simplified approach
    fn classify_emotion_simple(&self, features: &Array1<f32>, emotion: &str) -> f32 {
        // Simple heuristic-based classification
        let avg_feature = features.mean().unwrap_or(0.0);
        match emotion {
            "joy" => (avg_feature + 0.5).max(0.0).min(1.0),
            "sadness" => (0.5 - avg_feature).max(0.0).min(1.0),
            "anger" => (avg_feature.abs() * 0.7).max(0.0).min(1.0),
            "fear" => (avg_feature * 0.3).max(0.0).min(1.0),
            "surprise" => (avg_feature * 0.6).max(0.0).min(1.0),
            "disgust" => (avg_feature * 0.4).max(0.0).min(1.0),
            _ => 0.0,
        }
    }
    
    /// Process valence from input
    fn process_valence(&self, input: &EmotionalInput) -> f32 {
        let mut valence_processor = self.valence_processor.lock().unwrap();
        
        // Calculate new valence
        let new_valence = input.valence * input.intensity;
        
        // Apply smoothing using adaptation rate
        let smoothed_valence = if let Some(last_event) = valence_processor.valence_history.last() {
            last_event.valence * (1.0 - valence_processor.adaptation_rate) +
            new_valence * valence_processor.adaptation_rate
        } else {
            new_valence
        };
        
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
        
        memory.episodes.push(experience);
        
        // Maintain capacity (using a reasonable default)
        if memory.episodes.len() > 1000 {
            memory.episodes.remove(0);
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
        
        // Update emotion outcomes
        let current_association = learning.emotion_outcomes.get(&context).unwrap_or(&0.0);
        let learning_rate = learning.learning_rate;
        let new_association = current_association + learning_rate * outcome;
        learning.emotion_outcomes.insert(context.clone(), new_association);
        
        // Store learning event
        learning.learning_episodes.push(LearningEvent {
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
        
        if memory.episodes.is_empty() {
            return EmotionalTrends::default();
        }
        
        // Calculate trends
        let recent_experiences: Vec<_> = memory.episodes.iter()
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

