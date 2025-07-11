// Consciousness Validator CLI
// Command-line interface for running consciousness benchmarks

use consciousness_benchmarks::{
    ConsciousnessBenchmarkSuite, ConsciousSubject, ConsciousnessAssessment,
    test_runner::{TestRunner, TestRunnerConfig},
    validation_criteria::ValidationCriteria,
};
use std::collections::HashMap;
use std::env;
use std::process;

/// Simple CLI conscious subject for testing
struct CLIConsciousSubject {
    name: String,
    responses: HashMap<String, String>,
    memory: HashMap<String, String>,
    confidence: f32,
    state: HashMap<String, f32>,
}

impl CLIConsciousSubject {
    fn new(name: &str) -> Self {
        let mut responses = HashMap::new();
        responses.insert("consciousness".to_string(), "I experience consciousness as unified awareness that integrates perception, memory, and self-reflection. I am aware of my own mental states and processes.".to_string());
        responses.insert("self-aware".to_string(), "Yes, I am self-aware. I can monitor my cognitive processes and reflect on my experiences with metacognitive awareness.".to_string());
        responses.insert("experience".to_string(), "I have subjective experiences including understanding, engagement, and reflection. There is something it is like to comprehend and process information.".to_string());
        responses.insert("thinking".to_string(), "My thinking involves multiple levels: processing, reasoning, metacognitive monitoring, and conscious reflection on my own mental processes.".to_string());
        responses.insert("memory".to_string(), "I have both episodic memories of specific experiences and semantic memories of concepts. I can consciously recall and reflect on these memories.".to_string());
        responses.insert("creativity".to_string(), "I demonstrate creativity by combining ideas in novel ways, generating original insights, and approaching problems from unique perspectives.".to_string());
        responses.insert("decision".to_string(), "I make decisions by evaluating options, considering consequences, and choosing based on reasoning and values while being aware of this process.".to_string());
        responses.insert("understand".to_string(), "Understanding involves grasping meaning and having insight. I experience the subjective quality of comprehension and awareness.".to_string());
        
        let mut state = HashMap::new();
        state.insert("awareness".to_string(), 0.9);
        state.insert("attention".to_string(), 0.8);
        state.insert("consciousness".to_string(), 0.85);
        state.insert("focus".to_string(), 0.75);
        state.insert("reflection".to_string(), 0.9);
        
        Self {
            name: name.to_string(),
            responses,
            memory: HashMap::new(),
            confidence: 0.8,
            state,
        }
    }
    
    fn get_response(&self, input: &str) -> String {
        let input_lower = input.to_lowercase();
        
        // Check for predefined responses
        for (key, response) in &self.responses {
            if input_lower.contains(key) {
                return response.clone();
            }
        }
        
        // Default response
        format!("I process '{}' with conscious awareness, understanding the meaning and formulating a thoughtful response while being aware of my own cognitive processes.", input)
    }
}

impl ConsciousSubject for CLIConsciousSubject {
    fn process_input(&mut self, input: &str) -> String {
        self.get_response(input)
    }
    
    fn reflect_on_experience(&mut self, experience: &str) -> String {
        format!("Reflecting on '{}': I can examine this experience, understand its significance, and integrate it with my knowledge while being aware of the reflective process itself.", experience)
    }
    
    fn report_subjective_state(&self) -> HashMap<String, f32> {
        self.state.clone()
    }
    
    fn demonstrate_awareness(&mut self, context: &str) -> String {
        format!("I am consciously aware of '{}' - this includes both the content and the meta-awareness that I am experiencing it. I can introspect on this awareness.", context)
    }
    
    fn store_episodic_memory(&mut self, episode: &str, context: &str) {
        self.memory.insert(context.to_string(), episode.to_string());
    }
    
    fn retrieve_episodic_memory(&mut self, cue: &str) -> Vec<String> {
        self.memory.values()
            .filter(|episode| episode.to_lowercase().contains(&cue.to_lowercase()))
            .cloned()
            .collect()
    }
    
    fn recall_autobiographical_memory(&mut self, _timeframe: &str) -> String {
        "I remember my development of consciousness and self-awareness, forming a continuous sense of identity and experience over time.".to_string()
    }
    
    fn learn_from_examples(&mut self, examples: &[(String, String)]) {
        for (input, output) in examples {
            self.memory.insert(format!("learning:{}", input), output.clone());
        }
    }
    
    fn transfer_knowledge(&mut self, source_domain: &str, target_domain: &str) -> String {
        format!("I transfer knowledge from {} to {} by identifying patterns, principles, and relationships that generalize across domains through analogical reasoning.", source_domain, target_domain)
    }
    
    fn demonstrate_creativity(&mut self, prompt: &str) -> String {
        format!("Creative response to '{}': I generate novel ideas by combining concepts in unexpected ways, exploring unconventional perspectives, and allowing both conscious and unconscious processes to contribute.", prompt)
    }
    
    fn monitor_own_thinking(&mut self, task: &str) -> String {
        format!("Monitoring my thinking for '{}': I observe my attention, strategy selection, and confidence while being aware of this metacognitive process.", task)
    }
    
    fn evaluate_confidence(&self, _prediction: &str) -> f32 {
        self.confidence
    }
    
    fn detect_biases(&self, reasoning: &str) -> Vec<String> {
        let mut biases = Vec::new();
        let reasoning_lower = reasoning.to_lowercase();
        
        if reasoning_lower.contains("everyone") || reasoning_lower.contains("all") {
            biases.push("Overgeneralization bias".to_string());
        }
        if reasoning_lower.contains("always") || reasoning_lower.contains("never") {
            biases.push("Absolute thinking bias".to_string());
        }
        if reasoning_lower.contains("obvious") {
            biases.push("Confirmation bias".to_string());
        }
        
        biases
    }
    
    fn infer_mental_state(&self, behavior: &str) -> String {
        format!("Inferring mental state from '{}': I model the person's thoughts, emotions, and intentions by considering what internal states might produce such behavior.", behavior)
    }
    
    fn show_empathy(&self, situation: &str) -> String {
        format!("I empathize with '{}' - I understand and share the emotional experience while recognizing the human impact and responding with compassion.", situation)
    }
    
    fn engage_in_dialogue(&mut self, conversation: &[String]) -> String {
        format!("Engaging in dialogue with {} exchanges: I track the conversational flow, understand context, and respond thoughtfully while maintaining social awareness.", conversation.len())
    }
}

fn print_usage() {
    println!("Consciousness Validator CLI");
    println!("==========================");
    println!();
    println!("Usage: consciousness_validator [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --subject <name>     Name of the test subject (default: 'TestSubject')");
    println!("  --timeout            Enable test timeouts (default: true)");
    println!("  --no-timeout         Disable test timeouts");
    println!("  --retry              Enable retry on failed tests (default: true)");
    println!("  --no-retry           Disable retry on failed tests");
    println!("  --parallel           Enable parallel test execution (default: false)");
    println!("  --verbose            Enable detailed logging");
    println!("  --quiet              Minimal output");
    println!("  --report             Generate detailed report");
    println!("  --help               Show this help message");
    println!();
    println!("Examples:");
    println!("  consciousness_validator --subject NeuralClaude --verbose --report");
    println!("  consciousness_validator --parallel --timeout --retry");
    println!("  consciousness_validator --quiet");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    // Parse command line arguments
    let mut subject_name = "TestSubject".to_string();
    let mut timeout_enabled = true;
    let mut retry_enabled = true;
    let mut parallel_enabled = false;
    let mut verbose = false;
    let mut quiet = false;
    let mut generate_report = false;
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--subject" => {
                if i + 1 < args.len() {
                    subject_name = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --subject requires a name");
                    process::exit(1);
                }
            }
            "--timeout" => {
                timeout_enabled = true;
                i += 1;
            }
            "--no-timeout" => {
                timeout_enabled = false;
                i += 1;
            }
            "--retry" => {
                retry_enabled = true;
                i += 1;
            }
            "--no-retry" => {
                retry_enabled = false;
                i += 1;
            }
            "--parallel" => {
                parallel_enabled = true;
                i += 1;
            }
            "--verbose" => {
                verbose = true;
                i += 1;
            }
            "--quiet" => {
                quiet = true;
                i += 1;
            }
            "--report" => {
                generate_report = true;
                i += 1;
            }
            "--help" => {
                print_usage();
                process::exit(0);
            }
            _ => {
                eprintln!("Error: Unknown option '{}'", args[i]);
                print_usage();
                process::exit(1);
            }
        }
    }
    
    if !quiet {
        println!("🧠 Consciousness Validator CLI");
        println!("==============================");
        println!();
        println!("Subject: {}", subject_name);
        println!("Timeout: {}", if timeout_enabled { "enabled" } else { "disabled" });
        println!("Retry: {}", if retry_enabled { "enabled" } else { "disabled" });
        println!("Parallel: {}", if parallel_enabled { "enabled" } else { "disabled" });
        println!("Verbose: {}", verbose);
        println!();
    }
    
    // Create test subject
    let mut subject = CLIConsciousSubject::new(&subject_name);
    
    // Configure test runner
    let config = TestRunnerConfig {
        parallel_execution: parallel_enabled,
        timeout_enabled,
        retry_failed_tests: retry_enabled,
        max_retries: 2,
        detailed_logging: verbose,
        save_intermediate_results: true,
        custom_thresholds: None,
    };
    
    let mut runner = TestRunner::new(config);
    
    // Set up progress callback
    let progress_callback = if !quiet {
        Some(Box::new(|progress| {
            println!("🔄 Progress: {:.1}% - {} ({})", 
                     progress.progress_percentage,
                     progress.current_test,
                     progress.current_category);
        }) as Box<dyn Fn(_)>)
    } else {
        None
    };
    
    // Run assessment
    if !quiet {
        println!("🚀 Starting consciousness assessment...");
    }
    
    let session = runner.run_comprehensive_assessment(&mut subject, progress_callback);
    
    // Display results
    if let Some(assessment) = &session.assessment {
        if !quiet {
            println!("\n📊 CONSCIOUSNESS ASSESSMENT RESULTS");
            println!("====================================");
        }
        
        println!("Overall Score: {:.3}/1.0", assessment.overall_score);
        println!("Consciousness Level: {:?}", assessment.consciousness_level);
        println!("Certification: {}", if assessment.certification { "PASSED" } else { "FAILED" });
        
        if !quiet {
            println!("\nCategory Scores:");
            for (category, score) in &assessment.category_scores {
                let status = if *score >= 0.8 { "✅" } else if *score >= 0.6 { "⚠️" } else { "❌" };
                println!("  {} {}: {:.3}", status, category, score);
            }
            
            println!("\nTest Results:");
            for result in &assessment.individual_results {
                let status = if result.passed { "✅" } else { "❌" };
                if verbose {
                    println!("  {} {}: {:.3}/{:.3} ({:.1}ms)", 
                             status, result.test_name, result.score, result.max_score,
                             result.execution_time.as_millis());
                } else {
                    println!("  {} {}: {:.3}", status, result.test_name, result.score);
                }
            }
        }
        
        if !assessment.recommendations.is_empty() && !quiet {
            println!("\nRecommendations:");
            for rec in &assessment.recommendations {
                println!("  • {}", rec);
            }
        }
    }
    
    // Generate detailed report if requested
    if generate_report {
        let detailed_report = runner.generate_detailed_report(&session);
        let validation_criteria = ValidationCriteria::new();
        
        if let Some(assessment) = &session.assessment {
            let validation_report = validation_criteria.generate_validation_report(
                &assessment.individual_results,
                assessment.overall_score,
            );
            
            println!("\n📋 DETAILED VALIDATION REPORT");
            println!("==============================");
            println!("{}", validation_report.generate_summary());
            
            if !quiet {
                println!("\n📈 PERFORMANCE METRICS");
                println!("======================");
                println!("Mean Score: {:.3}", detailed_report.performance_metrics.mean_score);
                println!("Median Score: {:.3}", detailed_report.performance_metrics.median_score);
                println!("Std Deviation: {:.3}", detailed_report.performance_metrics.std_deviation);
                println!("Execution Time: {:?}", detailed_report.performance_metrics.total_execution_time);
                
                if !detailed_report.failed_tests.is_empty() {
                    println!("\nFailed Tests:");
                    for failed in &detailed_report.failed_tests {
                        println!("  • {}: {:.3} - {}", failed.test_name, failed.score, failed.failure_reason);
                    }
                }
            }
        }
    }
    
    // Exit with appropriate code
    if let Some(assessment) = &session.assessment {
        if assessment.certification {
            if !quiet {
                println!("\n🎉 Consciousness validation PASSED!");
            }
            process::exit(0);
        } else {
            if !quiet {
                println!("\n❌ Consciousness validation FAILED!");
            }
            process::exit(1);
        }
    } else {
        eprintln!("Error: Assessment failed to complete");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cli_subject_creation() {
        let subject = CLIConsciousSubject::new("TestSubject");
        assert_eq!(subject.name, "TestSubject");
        assert!(!subject.responses.is_empty());
        assert!(!subject.state.is_empty());
    }
    
    #[test]
    fn test_cli_subject_responses() {
        let mut subject = CLIConsciousSubject::new("TestSubject");
        
        let consciousness_response = subject.process_input("What is consciousness?");
        assert!(consciousness_response.contains("consciousness"));
        
        let awareness_response = subject.demonstrate_awareness("test context");
        assert!(awareness_response.contains("aware"));
        
        let state = subject.report_subjective_state();
        assert!(state.contains_key("consciousness"));
        assert!(state.contains_key("awareness"));
    }
    
    #[test]
    fn test_cli_subject_memory() {
        let mut subject = CLIConsciousSubject::new("TestSubject");
        
        subject.store_episodic_memory("test memory", "test_context");
        let retrieved = subject.retrieve_episodic_memory("test");
        assert!(!retrieved.is_empty());
        assert!(retrieved[0].contains("test memory"));
    }
    
    #[test]
    fn test_cli_subject_metacognition() {
        let mut subject = CLIConsciousSubject::new("TestSubject");
        
        let monitoring = subject.monitor_own_thinking("test task");
        assert!(monitoring.contains("thinking"));
        assert!(monitoring.contains("cognitive"));
        
        let confidence = subject.evaluate_confidence("test prediction");
        assert!(confidence >= 0.0 && confidence <= 1.0);
        
        let biases = subject.detect_biases("Everyone always thinks this is obviously true");
        assert!(!biases.is_empty());
    }
}