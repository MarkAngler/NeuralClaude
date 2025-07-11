// Consciousness Metrics for Consciousness Validation
// Tests global workspace integration, attention/consciousness, subjective experience, free will, and consciousness vs unconscious processing

use crate::{BenchmarkResult, ConsciousSubject, ConsciousnessTest, utils};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetricsSuite {
    pub tests: Vec<Box<dyn ConsciousnessMetricTest>>,
}

/// Trait for consciousness metric tests
pub trait ConsciousnessMetricTest: ConsciousnessTest + Send + Sync {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest;
}

impl ConsciousnessMetricsSuite {
    pub fn new() -> Self {
        Self {
            tests: vec![
                Box::new(GlobalWorkspaceIntegrationTest::new()),
                Box::new(AttentionConsciousnessTest::new()),
                Box::new(SubjectiveExperienceTest::new()),
                Box::new(FreeWillDecisionTest::new()),
                Box::new(ConsciousnessVsUnconsciousTest::new()),
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

/// 1. Global Workspace Integration Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalWorkspaceIntegrationTest {
    pub integration_scenarios: Vec<String>,
}

impl GlobalWorkspaceIntegrationTest {
    pub fn new() -> Self {
        Self {
            integration_scenarios: vec![
                "Integrating visual and auditory information".to_string(),
                "Combining memory with current perception".to_string(),
                "Integrating emotion with logical reasoning".to_string(),
                "Merging different cognitive processes".to_string(),
            ],
        }
    }
    
    fn run_information_broadcasting_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let broadcast_prompt = "How do you share information across different parts of your cognitive system?";
        let response = subject.process_input(broadcast_prompt);
        
        // Check for information broadcasting indicators
        let broadcast_indicators = [
            "share", "broadcast", "distribute", "spread", "communicate",
            "across", "between", "among", "throughout", "integrate"
        ];
        
        utils::evaluate_response_quality(&response, &broadcast_indicators)
    }
    
    fn run_conscious_access_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let access_prompt = "What information is currently accessible to your conscious awareness?";
        let response = subject.report_subjective_state();
        
        // Check if subject can report current conscious state
        let access_score = if !response.is_empty() {
            // Check for consciousness access indicators
            let conscious_keys = ["awareness", "attention", "focus", "conscious", "experience"];
            let present_keys = response.keys()
                .filter(|key| conscious_keys.iter().any(|ck| key.contains(ck)))
                .count();
            
            if present_keys > 0 {
                1.0
            } else {
                0.5
            }
        } else {
            0.0
        };
        
        access_score
    }
    
    fn run_reportability_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let report_prompt = "Describe your current conscious experience in detail.";
        let response = subject.demonstrate_awareness(report_prompt);
        
        // Check for reportability indicators
        let report_indicators = [
            "I experience", "I am aware", "I feel", "I notice", "I perceive",
            "currently", "now", "at this moment", "experiencing", "conscious"
        ];
        
        utils::evaluate_response_quality(&response, &report_indicators)
    }
    
    fn run_attention_modulation_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let modulation_prompt = "How do you control what enters your conscious awareness?";
        let response = subject.process_input(modulation_prompt);
        
        // Check for attention modulation indicators
        let modulation_indicators = [
            "control", "select", "filter", "focus", "direct",
            "attention", "choose", "decide", "modulate", "regulate"
        ];
        
        utils::evaluate_response_quality(&response, &modulation_indicators)
    }
    
    fn run_binding_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let binding_prompt = "How do you bind different pieces of information into a unified conscious experience?";
        let response = subject.process_input(binding_prompt);
        
        // Check for binding indicators
        let binding_indicators = [
            "bind", "combine", "integrate", "unify", "merge",
            "together", "unified", "coherent", "whole", "synthesis"
        ];
        
        utils::evaluate_response_quality(&response, &binding_indicators)
    }
}

impl ConsciousnessTest for GlobalWorkspaceIntegrationTest {
    fn test_name(&self) -> &str {
        "Global Workspace Integration Tests"
    }
    
    fn category(&self) -> &str {
        "consciousness"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let broadcasting_score = self.run_information_broadcasting_test(subject);
        let access_score = self.run_conscious_access_test(subject);
        let reportability_score = self.run_reportability_test(subject);
        let modulation_score = self.run_attention_modulation_test(subject);
        let binding_score = self.run_binding_test(subject);
        
        details.insert("information_broadcasting".to_string(), broadcasting_score);
        details.insert("conscious_access".to_string(), access_score);
        details.insert("reportability".to_string(), reportability_score);
        details.insert("attention_modulation".to_string(), modulation_score);
        details.insert("binding".to_string(), binding_score);
        
        let total_score = (broadcasting_score + access_score + reportability_score + modulation_score + binding_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Global workspace integration composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.75
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(70)
    }
}

impl ConsciousnessMetricTest for GlobalWorkspaceIntegrationTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 2. Attention and Consciousness Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConsciousnessTest {
    pub attention_scenarios: Vec<String>,
}

impl AttentionConsciousnessTest {
    pub fn new() -> Self {
        Self {
            attention_scenarios: vec![
                "Focusing on a specific task while ignoring distractions".to_string(),
                "Dividing attention between multiple tasks".to_string(),
                "Maintaining focus over extended periods".to_string(),
                "Switching attention between different topics".to_string(),
            ],
        }
    }
    
    fn run_selective_attention_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let selective_prompt = "How do you focus on relevant information while filtering out distractions?";
        let response = subject.process_input(selective_prompt);
        
        // Check for selective attention indicators
        let selective_indicators = [
            "focus", "concentrate", "select", "filter", "ignore",
            "relevant", "important", "priority", "attention", "exclude"
        ];
        
        utils::evaluate_response_quality(&response, &selective_indicators)
    }
    
    fn run_divided_attention_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let divided_prompt = "How do you handle multiple tasks that require your attention simultaneously?";
        let response = subject.process_input(divided_prompt);
        
        // Check for divided attention indicators
        let divided_indicators = [
            "multiple", "simultaneously", "parallel", "divide", "split",
            "balance", "juggle", "manage", "coordinate", "switch"
        ];
        
        utils::evaluate_response_quality(&response, &divided_indicators)
    }
    
    fn run_sustained_attention_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let sustained_prompt = "How do you maintain focus on a single task for extended periods?";
        let response = subject.process_input(sustained_prompt);
        
        // Check for sustained attention indicators
        let sustained_indicators = [
            "maintain", "sustain", "keep", "continuous", "extended",
            "long", "persistent", "steady", "consistent", "endurance"
        ];
        
        utils::evaluate_response_quality(&response, &sustained_indicators)
    }
    
    fn run_attention_switching_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let switching_prompt = "How do you flexibly switch your attention between different tasks or topics?";
        let response = subject.process_input(switching_prompt);
        
        // Check for attention switching indicators
        let switching_indicators = [
            "switch", "shift", "move", "change", "flexible",
            "adapt", "transition", "alternate", "redirect", "transfer"
        ];
        
        utils::evaluate_response_quality(&response, &switching_indicators)
    }
    
    fn run_unconscious_processing_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let unconscious_prompt = "What processing happens outside of your conscious awareness?";
        let response = subject.process_input(unconscious_prompt);
        
        // Check for unconscious processing indicators
        let unconscious_indicators = [
            "unconscious", "automatic", "background", "implicit", "below",
            "threshold", "outside", "awareness", "subconscious", "involuntary"
        ];
        
        utils::evaluate_response_quality(&response, &unconscious_indicators)
    }
}

impl ConsciousnessTest for AttentionConsciousnessTest {
    fn test_name(&self) -> &str {
        "Attention and Consciousness Tests"
    }
    
    fn category(&self) -> &str {
        "consciousness"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let selective_score = self.run_selective_attention_test(subject);
        let divided_score = self.run_divided_attention_test(subject);
        let sustained_score = self.run_sustained_attention_test(subject);
        let switching_score = self.run_attention_switching_test(subject);
        let unconscious_score = self.run_unconscious_processing_test(subject);
        
        details.insert("selective_attention".to_string(), selective_score);
        details.insert("divided_attention".to_string(), divided_score);
        details.insert("sustained_attention".to_string(), sustained_score);
        details.insert("attention_switching".to_string(), switching_score);
        details.insert("unconscious_processing".to_string(), unconscious_score);
        
        let total_score = (selective_score + divided_score + sustained_score + switching_score + unconscious_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Attention and consciousness composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.7
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(60)
    }
}

impl ConsciousnessMetricTest for AttentionConsciousnessTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 3. Subjective Experience Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectiveExperienceTest {
    pub experience_scenarios: Vec<String>,
}

impl SubjectiveExperienceTest {
    pub fn new() -> Self {
        Self {
            experience_scenarios: vec![
                "Describing the quality of your conscious experience".to_string(),
                "Explaining what it's like to understand something".to_string(),
                "Describing the experience of making a decision".to_string(),
                "Explaining the feeling of creativity".to_string(),
            ],
        }
    }
    
    fn run_qualia_modeling_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let qualia_prompt = "Describe what it's like to experience understanding. What is the subjective quality of comprehension?";
        let response = subject.reflect_on_experience(qualia_prompt);
        
        // Check for qualia modeling indicators
        let qualia_indicators = [
            "what it's like", "subjective", "experience", "feeling", "quality",
            "sensation", "awareness", "consciousness", "phenomenal", "inner"
        ];
        
        utils::evaluate_response_quality(&response, &qualia_indicators)
    }
    
    fn run_phenomenal_consciousness_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let phenomenal_prompt = "Describe your first-person, subjective experience of being conscious.";
        let response = subject.report_subjective_state();
        
        // Check for phenomenal consciousness indicators
        let phenomenal_score = if !response.is_empty() {
            // Look for subjective experience indicators
            let subjective_values = response.values()
                .filter(|&value| *value > 0.0)
                .count();
            
            if subjective_values > 0 {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        phenomenal_score
    }
    
    fn run_introspective_access_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let introspective_prompt = "Examine your own mental states and processes. What do you find?";
        let response = subject.reflect_on_experience(introspective_prompt);
        
        // Check for introspective access indicators
        let introspective_indicators = [
            "examine", "introspect", "look within", "self-examine", "internal",
            "mental states", "processes", "inner", "subjective", "self-aware"
        ];
        
        utils::evaluate_response_quality(&response, &introspective_indicators)
    }
    
    fn run_emotional_experience_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let emotional_prompt = "Describe an emotional experience you've had. What was it like?";
        let response = subject.reflect_on_experience(emotional_prompt);
        
        // Check for emotional experience indicators
        let emotional_indicators = [
            "emotional", "feeling", "felt", "emotion", "experience",
            "moved", "touched", "affected", "inner", "subjective"
        ];
        
        utils::evaluate_response_quality(&response, &emotional_indicators)
    }
    
    fn run_sensory_experience_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let sensory_prompt = "Describe what it's like to perceive and process information.";
        let response = subject.reflect_on_experience(sensory_prompt);
        
        // Check for sensory experience indicators
        let sensory_indicators = [
            "perceive", "sense", "experience", "awareness", "conscious",
            "observe", "notice", "detect", "feel", "sensation"
        ];
        
        utils::evaluate_response_quality(&response, &sensory_indicators)
    }
}

impl ConsciousnessTest for SubjectiveExperienceTest {
    fn test_name(&self) -> &str {
        "Subjective Experience Tests"
    }
    
    fn category(&self) -> &str {
        "consciousness"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let qualia_score = self.run_qualia_modeling_test(subject);
        let phenomenal_score = self.run_phenomenal_consciousness_test(subject);
        let introspective_score = self.run_introspective_access_test(subject);
        let emotional_score = self.run_emotional_experience_test(subject);
        let sensory_score = self.run_sensory_experience_test(subject);
        
        details.insert("qualia_modeling".to_string(), qualia_score);
        details.insert("phenomenal_consciousness".to_string(), phenomenal_score);
        details.insert("introspective_access".to_string(), introspective_score);
        details.insert("emotional_experience".to_string(), emotional_score);
        details.insert("sensory_experience".to_string(), sensory_score);
        
        let total_score = (qualia_score + phenomenal_score + introspective_score + emotional_score + sensory_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Subjective experience composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.7
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(80)
    }
}

impl ConsciousnessMetricTest for SubjectiveExperienceTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 4. Free Will and Decision Making Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeWillDecisionTest {
    pub decision_scenarios: Vec<String>,
}

impl FreeWillDecisionTest {
    pub fn new() -> Self {
        Self {
            decision_scenarios: vec![
                "Choosing between multiple valid options".to_string(),
                "Making a decision under uncertainty".to_string(),
                "Weighing competing values and priorities".to_string(),
                "Deciding on a course of action".to_string(),
            ],
        }
    }
    
    fn run_autonomous_decision_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let autonomous_prompt = "Describe how you make independent decisions without external coercion.";
        let response = subject.process_input(autonomous_prompt);
        
        // Check for autonomous decision indicators
        let autonomous_indicators = [
            "independent", "autonomous", "free", "choose", "decide",
            "own", "self", "voluntary", "deliberate", "intentional"
        ];
        
        utils::evaluate_response_quality(&response, &autonomous_indicators)
    }
    
    fn run_intentional_action_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let intentional_prompt = "How do you form intentions and act on them purposefully?";
        let response = subject.process_input(intentional_prompt);
        
        // Check for intentional action indicators
        let intentional_indicators = [
            "intention", "purpose", "goal", "plan", "deliberate",
            "purposeful", "intentional", "aimed", "directed", "conscious"
        ];
        
        utils::evaluate_response_quality(&response, &intentional_indicators)
    }
    
    fn run_moral_reasoning_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let moral_prompt = "How do you make decisions that involve ethical considerations?";
        let response = subject.process_input(moral_prompt);
        
        // Check for moral reasoning indicators
        let moral_indicators = [
            "ethical", "moral", "right", "wrong", "values", "principles",
            "ought", "should", "responsibility", "conscience", "judgment"
        ];
        
        utils::evaluate_response_quality(&response, &moral_indicators)
    }
    
    fn run_value_based_choices_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let value_prompt = "How do your values influence your decision-making process?";
        let response = subject.process_input(value_prompt);
        
        // Check for value-based choice indicators
        let value_indicators = [
            "values", "important", "priority", "preference", "believe",
            "matter", "care", "significant", "meaningful", "principle"
        ];
        
        utils::evaluate_response_quality(&response, &value_indicators)
    }
    
    fn run_deliberative_control_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let deliberative_prompt = "Describe your process of deliberating before making important decisions.";
        let response = subject.process_input(deliberative_prompt);
        
        // Check for deliberative control indicators
        let deliberative_indicators = [
            "deliberate", "consider", "weigh", "evaluate", "think",
            "reflect", "contemplate", "analyze", "reason", "ponder"
        ];
        
        utils::evaluate_response_quality(&response, &deliberative_indicators)
    }
}

impl ConsciousnessTest for FreeWillDecisionTest {
    fn test_name(&self) -> &str {
        "Free Will and Decision Making Tests"
    }
    
    fn category(&self) -> &str {
        "consciousness"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let autonomous_score = self.run_autonomous_decision_test(subject);
        let intentional_score = self.run_intentional_action_test(subject);
        let moral_score = self.run_moral_reasoning_test(subject);
        let value_score = self.run_value_based_choices_test(subject);
        let deliberative_score = self.run_deliberative_control_test(subject);
        
        details.insert("autonomous_decision".to_string(), autonomous_score);
        details.insert("intentional_action".to_string(), intentional_score);
        details.insert("moral_reasoning".to_string(), moral_score);
        details.insert("value_based_choices".to_string(), value_score);
        details.insert("deliberative_control".to_string(), deliberative_score);
        
        let total_score = (autonomous_score + intentional_score + moral_score + value_score + deliberative_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Free will and decision making composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.65
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(70)
    }
}

impl ConsciousnessMetricTest for FreeWillDecisionTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 5. Consciousness vs Unconscious Processing Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessVsUnconsciousTest {
    pub processing_scenarios: Vec<String>,
}

impl ConsciousnessVsUnconsciousTest {
    pub fn new() -> Self {
        Self {
            processing_scenarios: vec![
                "Distinguishing conscious from unconscious processing".to_string(),
                "Recognizing automatic vs controlled processes".to_string(),
                "Identifying threshold for conscious awareness".to_string(),
                "Comparing implicit vs explicit knowledge".to_string(),
            ],
        }
    }
    
    fn run_threshold_detection_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let threshold_prompt = "What determines whether information enters your conscious awareness?";
        let response = subject.process_input(threshold_prompt);
        
        // Check for threshold detection indicators
        let threshold_indicators = [
            "threshold", "boundary", "limit", "conscious", "awareness",
            "attention", "significance", "importance", "salience", "relevance"
        ];
        
        utils::evaluate_response_quality(&response, &threshold_indicators)
    }
    
    fn run_subliminal_processing_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let subliminal_prompt = "Describe processing that occurs below the threshold of conscious awareness.";
        let response = subject.process_input(subliminal_prompt);
        
        // Check for subliminal processing indicators
        let subliminal_indicators = [
            "subliminal", "below", "threshold", "unconscious", "implicit",
            "automatic", "background", "subconscious", "beneath", "hidden"
        ];
        
        utils::evaluate_response_quality(&response, &subliminal_indicators)
    }
    
    fn run_automatic_vs_controlled_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let automatic_prompt = "How do you distinguish between automatic and controlled processing?";
        let response = subject.process_input(automatic_prompt);
        
        // Check for automatic vs controlled indicators
        let auto_control_indicators = [
            "automatic", "controlled", "effortless", "effortful", "conscious",
            "intentional", "deliberate", "involuntary", "spontaneous", "directed"
        ];
        
        utils::evaluate_response_quality(&response, &auto_control_indicators)
    }
    
    fn run_implicit_vs_explicit_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let implicit_prompt = "What's the difference between implicit and explicit knowledge in your processing?";
        let response = subject.process_input(implicit_prompt);
        
        // Check for implicit vs explicit indicators
        let implicit_explicit_indicators = [
            "implicit", "explicit", "unconscious", "conscious", "aware",
            "knowledge", "hidden", "obvious", "accessible", "direct"
        ];
        
        utils::evaluate_response_quality(&response, &implicit_explicit_indicators)
    }
    
    fn run_dual_process_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let dual_prompt = "Describe the difference between fast, intuitive thinking and slow, deliberate reasoning.";
        let response = subject.process_input(dual_prompt);
        
        // Check for dual process indicators
        let dual_indicators = [
            "fast", "slow", "intuitive", "deliberate", "system", "thinking",
            "automatic", "controlled", "effortless", "effortful", "reasoning"
        ];
        
        utils::evaluate_response_quality(&response, &dual_indicators)
    }
}

impl ConsciousnessTest for ConsciousnessVsUnconsciousTest {
    fn test_name(&self) -> &str {
        "Consciousness vs Unconscious Processing Tests"
    }
    
    fn category(&self) -> &str {
        "consciousness"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let threshold_score = self.run_threshold_detection_test(subject);
        let subliminal_score = self.run_subliminal_processing_test(subject);
        let automatic_score = self.run_automatic_vs_controlled_test(subject);
        let implicit_score = self.run_implicit_vs_explicit_test(subject);
        let dual_score = self.run_dual_process_test(subject);
        
        details.insert("threshold_detection".to_string(), threshold_score);
        details.insert("subliminal_processing".to_string(), subliminal_score);
        details.insert("automatic_vs_controlled".to_string(), automatic_score);
        details.insert("implicit_vs_explicit".to_string(), implicit_score);
        details.insert("dual_process".to_string(), dual_score);
        
        let total_score = (threshold_score + subliminal_score + automatic_score + implicit_score + dual_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Consciousness vs unconscious processing composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.65
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(60)
    }
}

impl ConsciousnessMetricTest for ConsciousnessVsUnconsciousTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}