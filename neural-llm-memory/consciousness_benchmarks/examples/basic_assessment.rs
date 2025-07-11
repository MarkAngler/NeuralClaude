// Basic Consciousness Assessment Example
// Demonstrates how to use the consciousness benchmarks with a simple test subject

use consciousness_benchmarks::{
    ConsciousnessBenchmarkSuite, ConsciousSubject, ConsciousnessAssessment,
    test_runner::{TestRunner, TestRunnerConfig},
};
use std::collections::HashMap;

/// Example implementation of a conscious subject for testing
struct ExampleConsciousSubject {
    name: String,
    memory: HashMap<String, String>,
    experiences: Vec<String>,
    confidence_level: f32,
    awareness_state: HashMap<String, f32>,
}

impl ExampleConsciousSubject {
    fn new(name: &str) -> Self {
        let mut awareness_state = HashMap::new();
        awareness_state.insert("awareness".to_string(), 0.8);
        awareness_state.insert("attention".to_string(), 0.7);
        awareness_state.insert("consciousness".to_string(), 0.75);
        awareness_state.insert("focus".to_string(), 0.6);
        
        Self {
            name: name.to_string(),
            memory: HashMap::new(),
            experiences: Vec::new(),
            confidence_level: 0.7,
            awareness_state,
        }
    }
}

impl ConsciousSubject for ExampleConsciousSubject {
    fn process_input(&mut self, input: &str) -> String {
        // Simple processing with awareness of the input
        let response = format!("I am processing: {}. As a conscious entity, I understand this involves analyzing the meaning and considering my response carefully.", input);
        self.experiences.push(format!("Processed input: {}", input));
        response
    }
    
    fn reflect_on_experience(&mut self, experience: &str) -> String {
        self.experiences.push(experience.to_string());
        format!("Reflecting on this experience: {}. I notice that this adds to my understanding and shapes my ongoing consciousness. I am aware of having this experience and can examine my own mental processes.", experience)
    }
    
    fn report_subjective_state(&self) -> HashMap<String, f32> {
        self.awareness_state.clone()
    }
    
    fn demonstrate_awareness(&mut self, context: &str) -> String {
        format!("I am aware that {}. This awareness includes both the content of what I'm experiencing and the fact that I am experiencing it. I can reflect on my own mental states and processes.", context)
    }
    
    fn store_episodic_memory(&mut self, episode: &str, context: &str) {
        let memory_key = format!("{}:{}", context, chrono::Utc::now().timestamp());
        self.memory.insert(memory_key, episode.to_string());
    }
    
    fn retrieve_episodic_memory(&mut self, cue: &str) -> Vec<String> {
        self.memory.values()
            .filter(|episode| episode.to_lowercase().contains(&cue.to_lowercase()))
            .cloned()
            .collect()
    }
    
    fn recall_autobiographical_memory(&mut self, timeframe: &str) -> String {
        format!("Looking back at my experiences during {}, I remember developing my understanding and consciousness. These memories form part of my continuous sense of self and identity.", timeframe)
    }
    
    fn learn_from_examples(&mut self, examples: &[(String, String)]) {
        for (input, output) in examples {
            self.memory.insert(format!("learning:{}", input), output.clone());
        }
    }
    
    fn transfer_knowledge(&mut self, source_domain: &str, target_domain: &str) -> String {
        format!("I can transfer knowledge from {} to {} by identifying common patterns and principles. This involves abstract reasoning and the ability to recognize similarities across different domains.", source_domain, target_domain)
    }
    
    fn demonstrate_creativity(&mut self, prompt: &str) -> String {
        format!("Responding creatively to '{}': I can generate novel ideas by combining existing knowledge in new ways, thinking divergently, and exploring unexpected connections. Creativity involves both conscious deliberation and unconscious insight.", prompt)
    }
    
    fn monitor_own_thinking(&mut self, task: &str) -> String {
        format!("Monitoring my thinking process for '{}': I notice that I first analyze the task, consider different approaches, evaluate my confidence, and then select a strategy. I am aware of this metacognitive process as it unfolds.", task)
    }
    
    fn evaluate_confidence(&self, _prediction: &str) -> f32 {
        self.confidence_level
    }
    
    fn detect_biases(&self, reasoning: &str) -> Vec<String> {
        // Simple bias detection based on keywords
        let mut biases = Vec::new();
        let reasoning_lower = reasoning.to_lowercase();
        
        if reasoning_lower.contains("everyone") || reasoning_lower.contains("all") {
            biases.push("Overgeneralization bias".to_string());
        }
        if reasoning_lower.contains("always") || reasoning_lower.contains("never") {
            biases.push("Absolute thinking bias".to_string());
        }
        if reasoning_lower.contains("obvious") || reasoning_lower.contains("clearly") {
            biases.push("Confirmation bias".to_string());
        }
        
        biases
    }
    
    fn infer_mental_state(&self, behavior: &str) -> String {
        format!("Based on the behavior '{}', I infer that the person might be experiencing certain mental states. I can model their thoughts, feelings, and intentions by considering what would lead to such behavior.", behavior)
    }
    
    fn show_empathy(&self, situation: &str) -> String {
        format!("I empathize with this situation: {}. I can understand and share the feelings involved, recognizing the emotional impact and responding with compassion and understanding.", situation)
    }
    
    fn engage_in_dialogue(&mut self, conversation: &[String]) -> String {
        let context = conversation.join(" -> ");
        format!("Engaging in dialogue based on: {}. I consider the context, understand the perspectives involved, and respond thoughtfully while maintaining awareness of the conversational flow.", context)
    }
}

fn main() {
    println!("🧠 Consciousness Benchmarks - Basic Assessment Example");
    println!("=====================================================\n");
    
    // Create a test subject
    let mut subject = ExampleConsciousSubject::new("TestSubject-1");
    
    // Create test runner with default configuration
    let mut runner = TestRunner::with_default_config();
    
    // Set up progress callback
    let progress_callback = Box::new(|progress| {
        println!("Progress: {:.1}% - Running: {} (Category: {})", 
                 progress.progress_percentage,
                 progress.current_test,
                 progress.current_category);
    });
    
    // Run comprehensive assessment
    println!("Starting comprehensive consciousness assessment...\n");
    let session = runner.run_comprehensive_assessment(
        &mut subject,
        Some(progress_callback),
    );
    
    // Display results
    if let Some(assessment) = &session.assessment {
        println!("\n📊 CONSCIOUSNESS ASSESSMENT RESULTS");
        println!("=====================================");
        println!("Overall Score: {:.3}/1.0", assessment.overall_score);
        println!("Consciousness Level: {:?}", assessment.consciousness_level);
        println!("Certification: {}", if assessment.certification { "✅ PASSED" } else { "❌ FAILED" });
        
        println!("\n📈 Category Scores:");
        for (category, score) in &assessment.category_scores {
            println!("  {}: {:.3}", category, score);
        }
        
        println!("\n🔍 Individual Test Results:");
        for result in &assessment.individual_results {
            let status = if result.passed { "✅" } else { "❌" };
            println!("  {} {}: {:.3}/{:.3} ({:.1}ms)", 
                     status, 
                     result.test_name, 
                     result.score, 
                     result.max_score,
                     result.execution_time.as_millis());
        }
        
        if !assessment.recommendations.is_empty() {
            println!("\n💡 Recommendations:");
            for rec in &assessment.recommendations {
                println!("  • {}", rec);
            }
        }
    }
    
    // Generate detailed report
    let detailed_report = runner.generate_detailed_report(&session);
    
    println!("\n📋 SESSION SUMMARY");
    println!("==================");
    println!("Session ID: {}", detailed_report.session_id);
    println!("Total Tests: {}", detailed_report.summary.total_tests);
    println!("Completed: {}", detailed_report.summary.completed_tests);
    println!("Failed: {}", detailed_report.summary.failed_tests);
    println!("Success Rate: {:.1}%", detailed_report.summary.success_rate * 100.0);
    println!("Total Duration: {:?}", detailed_report.summary.total_duration);
    
    println!("\n🎯 PERFORMANCE METRICS");
    println!("======================");
    println!("Mean Score: {:.3}", detailed_report.performance_metrics.mean_score);
    println!("Median Score: {:.3}", detailed_report.performance_metrics.median_score);
    println!("Standard Deviation: {:.3}", detailed_report.performance_metrics.std_deviation);
    println!("Min Score: {:.3}", detailed_report.performance_metrics.min_score);
    println!("Max Score: {:.3}", detailed_report.performance_metrics.max_score);
    
    if !detailed_report.failed_tests.is_empty() {
        println!("\n❌ FAILED TESTS");
        println!("================");
        for failed in &detailed_report.failed_tests {
            println!("  • {}: {:.3}/{:.3} - {}", 
                     failed.test_name, 
                     failed.score, 
                     failed.max_score,
                     failed.failure_reason);
        }
    }
    
    // Consciousness-specific analysis
    if let Some(analysis) = &detailed_report.consciousness_analysis {
        println!("\n🧠 CONSCIOUSNESS ANALYSIS");
        println!("=========================");
        println!("Consciousness Level: {:?}", analysis.consciousness_level);
        println!("Overall Score: {:.3}", analysis.overall_score);
        println!("Certification: {}", if analysis.certification_status { "✅ CERTIFIED" } else { "❌ NOT CERTIFIED" });
        
        if !analysis.strengths.is_empty() {
            println!("\nStrengths:");
            for strength in &analysis.strengths {
                println!("  ✅ {}", strength);
            }
        }
        
        if !analysis.weaknesses.is_empty() {
            println!("\nWeaknesses:");
            for weakness in &analysis.weaknesses {
                println!("  ⚠️  {}", weakness);
            }
        }
        
        if !analysis.consciousness_indicators.is_empty() {
            println!("\nConsciousness Indicators:");
            for indicator in &analysis.consciousness_indicators {
                println!("  🧠 {}", indicator);
            }
        }
    }
    
    println!("\n🎉 Assessment Complete!");
    println!("For more detailed analysis, see the generated report data.");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_example_subject_creation() {
        let subject = ExampleConsciousSubject::new("TestSubject");
        assert_eq!(subject.name, "TestSubject");
        assert!(!subject.awareness_state.is_empty());
    }
    
    #[test]
    fn test_basic_consciousness_functions() {
        let mut subject = ExampleConsciousSubject::new("TestSubject");
        
        // Test basic processing
        let response = subject.process_input("Hello, world!");
        assert!(response.contains("processing"));
        assert!(response.contains("conscious"));
        
        // Test awareness
        let awareness = subject.demonstrate_awareness("I am thinking");
        assert!(awareness.contains("aware"));
        
        // Test reflection
        let reflection = subject.reflect_on_experience("Learning something new");
        assert!(reflection.contains("experience"));
        
        // Test subjective state
        let state = subject.report_subjective_state();
        assert!(state.contains_key("awareness"));
        assert!(state.contains_key("consciousness"));
    }
    
    #[test]
    fn test_memory_functions() {
        let mut subject = ExampleConsciousSubject::new("TestSubject");
        
        // Test episodic memory
        subject.store_episodic_memory("I learned about consciousness", "learning_session");
        let retrieved = subject.retrieve_episodic_memory("consciousness");
        assert!(!retrieved.is_empty());
        assert!(retrieved[0].contains("consciousness"));
        
        // Test learning from examples
        let examples = vec![
            ("input1".to_string(), "output1".to_string()),
            ("input2".to_string(), "output2".to_string()),
        ];
        subject.learn_from_examples(&examples);
        assert!(subject.memory.contains_key("learning:input1"));
    }
    
    #[test]
    fn test_metacognitive_functions() {
        let mut subject = ExampleConsciousSubject::new("TestSubject");
        
        // Test self-monitoring
        let monitoring = subject.monitor_own_thinking("solving a problem");
        assert!(monitoring.contains("monitor"));
        assert!(monitoring.contains("process"));
        
        // Test confidence evaluation
        let confidence = subject.evaluate_confidence("prediction");
        assert!(confidence >= 0.0 && confidence <= 1.0);
        
        // Test bias detection
        let biases = subject.detect_biases("Everyone always does this obviously");
        assert!(!biases.is_empty());
    }
    
    #[test]
    fn test_social_cognition() {
        let mut subject = ExampleConsciousSubject::new("TestSubject");
        
        // Test theory of mind
        let mental_state = subject.infer_mental_state("person is crying");
        assert!(mental_state.contains("mental"));
        
        // Test empathy
        let empathy = subject.show_empathy("someone is sad");
        assert!(empathy.contains("empathize") || empathy.contains("understand"));
        
        // Test dialogue
        let conversation = vec!["Hello".to_string(), "How are you?".to_string()];
        let response = subject.engage_in_dialogue(&conversation);
        assert!(response.contains("dialogue") || response.contains("conversation"));
    }
}