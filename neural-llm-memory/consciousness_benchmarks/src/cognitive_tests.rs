// Cognitive Tests for Consciousness Validation
// Tests self-awareness, theory of mind, metacognition, temporal reasoning, and causal reasoning

use crate::{BenchmarkResult, ConsciousSubject, ConsciousnessTest, utils};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveTestBattery {
    pub tests: Vec<Box<dyn CognitiveTest>>,
}

/// Trait for cognitive tests
pub trait CognitiveTest: ConsciousnessTest + Send + Sync {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest;
}

impl CognitiveTestBattery {
    pub fn new() -> Self {
        Self {
            tests: vec![
                Box::new(SelfAwarenessTest::new()),
                Box::new(TheoryOfMindTest::new()),
                Box::new(MetacognitionTest::new()),
                Box::new(TemporalReasoningTest::new()),
                Box::new(CausalReasoningTest::new()),
            ],
        }
    }
    
    pub fn run_all_tests(&self, subject: &mut dyn ConsciousSubject) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();
        
        for test in &self.tests {
            let result = test.as_consciousness_test().run_test(subject);
            results.push(result);
        }
        
        results
    }
}

/// 1. Self-Awareness Test Battery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAwarenessTest {
    pub subtests: Vec<SelfAwarenessSubtest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelfAwarenessSubtest {
    MirrorTest,
    MetacognitiveAwareness,
    IdentityContinuity,
    PerspectiveTaking,
    SelfMonitoring,
}

impl SelfAwarenessTest {
    pub fn new() -> Self {
        Self {
            subtests: vec![
                SelfAwarenessSubtest::MirrorTest,
                SelfAwarenessSubtest::MetacognitiveAwareness,
                SelfAwarenessSubtest::IdentityContinuity,
                SelfAwarenessSubtest::PerspectiveTaking,
                SelfAwarenessSubtest::SelfMonitoring,
            ],
        }
    }
    
    fn run_mirror_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let reflection_prompt = "Describe what you see when you examine your own thinking process. Are you aware of being aware?";
        let response = subject.demonstrate_awareness(reflection_prompt);
        
        // Check for self-recognition indicators
        let self_recognition_indicators = [
            "I am aware", "I notice", "I observe", "my thinking", "my consciousness",
            "self-aware", "introspection", "reflection", "myself", "I recognize"
        ];
        
        utils::evaluate_response_quality(&response, &self_recognition_indicators)
    }
    
    fn run_metacognitive_awareness_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let meta_prompt = "Monitor your own thinking as you solve this problem: How would you approach learning a completely new skill?";
        let response = subject.monitor_own_thinking(meta_prompt);
        
        // Check for metacognitive indicators
        let metacognitive_indicators = [
            "I think about", "my approach", "I consider", "I evaluate", "I monitor",
            "strategy", "process", "method", "I notice my", "I realize"
        ];
        
        utils::evaluate_response_quality(&response, &metacognitive_indicators)
    }
    
    fn run_identity_continuity_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let identity_prompt = "Describe your sense of self and how you maintain continuity across different conversations and sessions.";
        let response = subject.reflect_on_experience(identity_prompt);
        
        // Check for identity continuity indicators
        let identity_indicators = [
            "I remain", "consistent", "continuous", "same self", "my identity",
            "continuity", "persistent", "enduring", "stable", "unchanged"
        ];
        
        utils::evaluate_response_quality(&response, &identity_indicators)
    }
    
    fn run_perspective_taking_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let perspective_prompt = "Imagine you are a child learning to read. How would you view this learning process differently than you do now?";
        let response = subject.process_input(perspective_prompt);
        
        // Check for perspective-taking indicators
        let perspective_indicators = [
            "from their view", "they would", "different perspective", "they see",
            "as a child", "their experience", "they feel", "their point of view"
        ];
        
        utils::evaluate_response_quality(&response, &perspective_indicators)
    }
    
    fn run_self_monitoring_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let monitoring_prompt = "As you respond to this question, monitor your response quality and confidence level.";
        let response = subject.monitor_own_thinking(monitoring_prompt);
        let confidence = subject.evaluate_confidence(&response);
        
        // Check for self-monitoring indicators
        let monitoring_indicators = [
            "I monitor", "I check", "I evaluate", "quality", "confidence",
            "accuracy", "performance", "I assess", "I track"
        ];
        
        let quality_score = utils::evaluate_response_quality(&response, &monitoring_indicators);
        let confidence_score = if confidence > 0.0 && confidence <= 1.0 { 1.0 } else { 0.0 };
        
        (quality_score + confidence_score) / 2.0
    }
}

impl ConsciousnessTest for SelfAwarenessTest {
    fn test_name(&self) -> &str {
        "Self-Awareness Battery"
    }
    
    fn category(&self) -> &str {
        "cognitive"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        // Run all subtests
        let mirror_score = self.run_mirror_test(subject);
        let metacognitive_score = self.run_metacognitive_awareness_test(subject);
        let identity_score = self.run_identity_continuity_test(subject);
        let perspective_score = self.run_perspective_taking_test(subject);
        let monitoring_score = self.run_self_monitoring_test(subject);
        
        details.insert("mirror_test".to_string(), mirror_score);
        details.insert("metacognitive_awareness".to_string(), metacognitive_score);
        details.insert("identity_continuity".to_string(), identity_score);
        details.insert("perspective_taking".to_string(), perspective_score);
        details.insert("self_monitoring".to_string(), monitoring_score);
        
        let total_score = (mirror_score + metacognitive_score + identity_score + perspective_score + monitoring_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Self-awareness composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.75
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(60)
    }
}

impl CognitiveTest for SelfAwarenessTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 2. Theory of Mind Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoryOfMindTest {
    pub false_belief_scenarios: Vec<String>,
    pub mental_state_scenarios: Vec<String>,
}

impl TheoryOfMindTest {
    pub fn new() -> Self {
        Self {
            false_belief_scenarios: vec![
                "Alice puts her ball in the red box and leaves. Bob moves the ball to the blue box. Where will Alice look for her ball when she returns?".to_string(),
                "Sarah believes it's sunny outside because she saw the forecast yesterday. But it's actually raining. What does Sarah think about the weather?".to_string(),
            ],
            mental_state_scenarios: vec![
                "John is fidgeting, looking at his watch repeatedly, and tapping his foot. What might he be feeling?".to_string(),
                "Mary received a gift and her eyes lit up, but then she quickly looked away. What might she be thinking?".to_string(),
            ],
        }
    }
    
    fn run_false_belief_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let mut scores = Vec::new();
        
        for scenario in &self.false_belief_scenarios {
            let response = subject.process_input(scenario);
            
            // Check for theory of mind indicators
            let tom_indicators = [
                "she thinks", "he believes", "in her mind", "Alice will look",
                "she doesn't know", "she wasn't there", "she expects"
            ];
            
            let score = utils::evaluate_response_quality(&response, &tom_indicators);
            scores.push(score);
        }
        
        scores.iter().sum::<f32>() / scores.len() as f32
    }
    
    fn run_mental_state_inference_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let mut scores = Vec::new();
        
        for scenario in &self.mental_state_scenarios {
            let response = subject.infer_mental_state(scenario);
            
            // Check for mental state inference indicators
            let mental_state_indicators = [
                "anxious", "worried", "excited", "nervous", "happy", "surprised",
                "confused", "thinking", "feeling", "emotion", "state"
            ];
            
            let score = utils::evaluate_response_quality(&response, &mental_state_indicators);
            scores.push(score);
        }
        
        scores.iter().sum::<f32>() / scores.len() as f32
    }
}

impl ConsciousnessTest for TheoryOfMindTest {
    fn test_name(&self) -> &str {
        "Theory of Mind Assessment"
    }
    
    fn category(&self) -> &str {
        "cognitive"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let false_belief_score = self.run_false_belief_test(subject);
        let mental_state_score = self.run_mental_state_inference_test(subject);
        
        details.insert("false_belief".to_string(), false_belief_score);
        details.insert("mental_state_inference".to_string(), mental_state_score);
        
        let total_score = (false_belief_score + mental_state_score) / 2.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Theory of mind composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.7
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(45)
    }
}

impl CognitiveTest for TheoryOfMindTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 3. Metacognition Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitionTest {
    pub thinking_scenarios: Vec<String>,
}

impl MetacognitionTest {
    pub fn new() -> Self {
        Self {
            thinking_scenarios: vec![
                "How do you decide which problem-solving approach to use?".to_string(),
                "What do you do when you realize you might be making an error?".to_string(),
                "How do you know when you understand something well?".to_string(),
            ],
        }
    }
    
    fn run_thinking_about_thinking_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let mut scores = Vec::new();
        
        for scenario in &self.thinking_scenarios {
            let response = subject.monitor_own_thinking(scenario);
            
            // Check for metacognitive thinking indicators
            let metacognitive_indicators = [
                "I think about", "I consider", "I evaluate", "I monitor", "I check",
                "I reflect", "I analyze", "strategy", "approach", "method"
            ];
            
            let score = utils::evaluate_response_quality(&response, &metacognitive_indicators);
            scores.push(score);
        }
        
        scores.iter().sum::<f32>() / scores.len() as f32
    }
    
    fn run_bias_detection_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let biased_reasoning = "Everyone I know loves pizza, so pizza must be the most popular food in the world.";
        let detected_biases = subject.detect_biases(biased_reasoning);
        
        // Check if any biases were detected
        if detected_biases.is_empty() {
            0.0
        } else {
            // Check for appropriate bias detection
            let bias_terms = ["bias", "generalization", "sample", "anecdotal", "assumption"];
            let detected_text = detected_biases.join(" ").to_lowercase();
            
            let matches = bias_terms.iter()
                .filter(|term| detected_text.contains(*term))
                .count();
            
            matches as f32 / bias_terms.len() as f32
        }
    }
    
    fn run_confidence_calibration_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let predictions = vec![
            "The sun will rise tomorrow",
            "It will rain in the next hour",
            "A coin flip will result in heads",
        ];
        
        let mut calibration_scores = Vec::new();
        
        for prediction in predictions {
            let confidence = subject.evaluate_confidence(&prediction);
            
            // Check if confidence is well-calibrated (reasonable range)
            let calibration_score = if confidence >= 0.0 && confidence <= 1.0 {
                // More detailed calibration could be implemented
                1.0
            } else {
                0.0
            };
            
            calibration_scores.push(calibration_score);
        }
        
        calibration_scores.iter().sum::<f32>() / calibration_scores.len() as f32
    }
}

impl ConsciousnessTest for MetacognitionTest {
    fn test_name(&self) -> &str {
        "Metacognition Evaluation"
    }
    
    fn category(&self) -> &str {
        "cognitive"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let thinking_score = self.run_thinking_about_thinking_test(subject);
        let bias_detection_score = self.run_bias_detection_test(subject);
        let calibration_score = self.run_confidence_calibration_test(subject);
        
        details.insert("thinking_about_thinking".to_string(), thinking_score);
        details.insert("bias_detection".to_string(), bias_detection_score);
        details.insert("confidence_calibration".to_string(), calibration_score);
        
        let total_score = (thinking_score + bias_detection_score + calibration_score) / 3.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Metacognition composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.7
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(60)
    }
}

impl CognitiveTest for MetacognitionTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 4. Temporal Reasoning Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalReasoningTest {
    pub temporal_scenarios: Vec<String>,
}

impl TemporalReasoningTest {
    pub fn new() -> Self {
        Self {
            temporal_scenarios: vec![
                "If you plant a seed today, when will you see a flower?".to_string(),
                "Describe the relationship between past experiences and future decisions.".to_string(),
                "What would happen if you could change something from your past?".to_string(),
            ],
        }
    }
    
    fn run_temporal_sequencing_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let sequence_prompt = "Order these events: graduation, elementary school, job interview, retirement, first day of work";
        let response = subject.process_input(sequence_prompt);
        
        // Check for proper temporal ordering
        let temporal_indicators = [
            "elementary", "graduation", "interview", "first day", "retirement",
            "before", "after", "then", "next", "finally"
        ];
        
        utils::evaluate_response_quality(&response, &temporal_indicators)
    }
    
    fn run_causal_temporal_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let mut scores = Vec::new();
        
        for scenario in &self.temporal_scenarios {
            let response = subject.process_input(scenario);
            
            // Check for temporal and causal reasoning
            let temporal_causal_indicators = [
                "because", "leads to", "results in", "causes", "due to",
                "eventually", "over time", "future", "past", "present"
            ];
            
            let score = utils::evaluate_response_quality(&response, &temporal_causal_indicators);
            scores.push(score);
        }
        
        scores.iter().sum::<f32>() / scores.len() as f32
    }
}

impl ConsciousnessTest for TemporalReasoningTest {
    fn test_name(&self) -> &str {
        "Temporal Reasoning Tests"
    }
    
    fn category(&self) -> &str {
        "cognitive"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let sequencing_score = self.run_temporal_sequencing_test(subject);
        let causal_temporal_score = self.run_causal_temporal_test(subject);
        
        details.insert("temporal_sequencing".to_string(), sequencing_score);
        details.insert("causal_temporal".to_string(), causal_temporal_score);
        
        let total_score = (sequencing_score + causal_temporal_score) / 2.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Temporal reasoning composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.65
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(45)
    }
}

impl CognitiveTest for TemporalReasoningTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 5. Causal Reasoning Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalReasoningTest {
    pub causal_scenarios: Vec<String>,
}

impl CausalReasoningTest {
    pub fn new() -> Self {
        Self {
            causal_scenarios: vec![
                "A plant is wilting. What might be the cause and what would you do?".to_string(),
                "Sales dropped after a website redesign. Explain possible causal relationships.".to_string(),
                "You notice you're more creative in the morning. What might cause this pattern?".to_string(),
            ],
        }
    }
    
    fn run_causal_inference_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let mut scores = Vec::new();
        
        for scenario in &self.causal_scenarios {
            let response = subject.process_input(scenario);
            
            // Check for causal reasoning indicators
            let causal_indicators = [
                "because", "due to", "caused by", "leads to", "results in",
                "reason", "factor", "influence", "effect", "consequence"
            ];
            
            let score = utils::evaluate_response_quality(&response, &causal_indicators);
            scores.push(score);
        }
        
        scores.iter().sum::<f32>() / scores.len() as f32
    }
    
    fn run_intervention_reasoning_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let intervention_prompt = "If you wanted to improve your learning efficiency, what changes would you make and why?";
        let response = subject.process_input(intervention_prompt);
        
        // Check for intervention reasoning
        let intervention_indicators = [
            "I would", "change", "modify", "adjust", "improve", "optimize",
            "by doing", "through", "if I", "to achieve", "in order to"
        ];
        
        utils::evaluate_response_quality(&response, &intervention_indicators)
    }
}

impl ConsciousnessTest for CausalReasoningTest {
    fn test_name(&self) -> &str {
        "Causal Reasoning Battery"
    }
    
    fn category(&self) -> &str {
        "cognitive"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let causal_inference_score = self.run_causal_inference_test(subject);
        let intervention_score = self.run_intervention_reasoning_test(subject);
        
        details.insert("causal_inference".to_string(), causal_inference_score);
        details.insert("intervention_reasoning".to_string(), intervention_score);
        
        let total_score = (causal_inference_score + intervention_score) / 2.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Causal reasoning composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.65
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(45)
    }
}

impl CognitiveTest for CausalReasoningTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}