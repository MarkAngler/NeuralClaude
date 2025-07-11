//! Demonstration of the metacognitive layer in action
//! 
//! This example shows how the metacognitive system monitors thinking,
//! selects strategies, and achieves self-aware learning.

use neural_llm_memory::metacognition::{
    MetaCognition, MetaCognitiveConfig, ThinkingInput, ThinkingStrategy,
};
use neural_llm_memory::adaptive::AdaptiveMemoryModule;
use neural_llm_memory::memory::MemoryConfig;

use std::collections::HashMap;
use std::time::Duration;
use chrono::Utc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 NeuralClaude Metacognitive Layer Demo\n");
    
    // Initialize metacognitive system
    let config = MetaCognitiveConfig {
        monitoring_interval: 5,
        bias_sensitivity: 0.8,
        strategy_switch_threshold: 0.4,
        optimization_rate: 0.02,
        introspection_depth: 3,
        ..Default::default()
    };
    
    let mut metacognition = MetaCognition::new(config);
    
    // Initialize adaptive memory
    let memory_config = MemoryConfig::default();
    let mut memory = AdaptiveMemoryModule::new(memory_config).await?;
    
    // Example 1: Problem-solving with metacognitive monitoring
    println!("📋 Example 1: Solving a complex problem with self-awareness\n");
    
    let problem = ThinkingInput {
        task: "Design a distributed caching system that maintains consistency across nodes".to_string(),
        context: HashMap::from([
            ("domain".to_string(), "system_design".to_string()),
            ("scale".to_string(), "large".to_string()),
            ("constraints".to_string(), "low_latency,high_availability".to_string()),
        ]),
        constraints: vec![
            Constraint::TimeLimit(Duration::from_secs(300)),
            Constraint::ResourceLimit("memory", 1000),
        ],
        available_time: Duration::from_secs(300),
    };
    
    // Process with metacognitive monitoring
    let result = metacognition.monitored_thinking_step(&problem, &mut memory);
    
    println!("Strategy used: {:?}", result.strategy_used);
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!("Result: {}", result.result);
    
    if !result.biases_detected.is_empty() {
        println!("\n⚠️  Cognitive biases detected:");
        for bias in &result.biases_detected {
            println!("  - {:?} (strength: {:.2})", bias.bias_type, bias.strength);
            println!("    Mitigation: {}", bias.mitigation_strategy);
        }
    }
    
    // Example 2: Demonstrating strategy switching
    println!("\n\n📋 Example 2: Adaptive strategy switching\n");
    
    let creative_task = ThinkingInput {
        task: "Generate innovative solutions for reducing carbon emissions in urban transport".to_string(),
        context: HashMap::from([
            ("domain".to_string(), "creative_problem_solving".to_string()),
            ("type".to_string(), "brainstorming".to_string()),
        ]),
        constraints: vec![],
        available_time: Duration::from_secs(600),
    };
    
    let creative_result = metacognition.monitored_thinking_step(&creative_task, &mut memory);
    
    println!("Initial strategy: {:?}", creative_result.strategy_used);
    
    // Simulate poor performance to trigger strategy switch
    metacognition.strategy_selector.update_performance(
        &creative_result.strategy_used,
        "creative_problem_solving",
        false, // failure
        120.0, // time taken
        0.3,   // low confidence
    );
    
    // Try again - should switch strategy
    let retry_result = metacognition.monitored_thinking_step(&creative_task, &mut memory);
    println!("Adapted strategy: {:?}", retry_result.strategy_used);
    
    // Example 3: Self-reflection and improvement
    println!("\n\n📋 Example 3: Self-reflection and meta-learning\n");
    
    // Reflect on thinking process
    let reflection = metacognition.reflect_on_thinking(&create_thinking_trace());
    
    println!("Self-awareness score: {:.2}%", reflection.effectiveness_score * 100.0);
    println!("\nPatterns identified:");
    for pattern in &reflection.patterns_identified {
        println!("  - {}: {} occurrences", pattern.pattern_type, pattern.occurrences);
    }
    
    println!("\nSuggested improvements:");
    for improvement in &reflection.suggested_improvements {
        println!("  - {}", improvement);
    }
    
    // Example 4: Theory of mind simulation
    println!("\n\n📋 Example 4: Theory of Mind - Understanding other agents\n");
    
    let other_agent = AgentModel {
        id: "user_123".to_string(),
        observed_actions: vec![
            "asked about performance optimization".to_string(),
            "requested code examples".to_string(),
            "focused on practical implementation".to_string(),
        ],
        communication_style: "technical".to_string(),
    };
    
    let mind_state = metacognition.simulate_other_mind(&other_agent);
    
    println!("Inferred mental state of {}:", mind_state.agent_id);
    println!("Beliefs: {:?}", mind_state.beliefs);
    println!("Goals: {:?}", mind_state.goals);
    println!("Theory confidence: {:.2}%", mind_state.theory_confidence * 100.0);
    
    // Example 5: Introspection API
    println!("\n\n📋 Example 5: Introspection and self-examination\n");
    
    let cognitive_state = metacognition.introspection.get_cognitive_state();
    
    println!("Current cognitive state:");
    println!("  Active patterns: {}", cognitive_state.active_patterns.len());
    println!("  Self-awareness: {:.2}%", cognitive_state.performance_metrics.self_awareness_score * 100.0);
    println!("  Average confidence: {:.2}%", cognitive_state.performance_metrics.average_confidence * 100.0);
    
    // Explain a specific decision
    if let Some(decision) = cognitive_state.recent_decisions.first() {
        let explanation = metacognition.introspection.explain_decision(&decision.id);
        println!("\nDecision explanation for '{}':", decision.option_selected);
        println!("  Reasoning: {}", explanation.reasoning);
        println!("  Key factors: {:?}", explanation.key_factors);
        println!("  Confidence basis: {}", explanation.confidence_basis);
    }
    
    println!("\n✅ Metacognitive demonstration complete!");
    
    Ok(())
}

// Helper structures for the demo

#[derive(Debug, Clone)]
pub enum Constraint {
    TimeLimit(Duration),
    ResourceLimit(&'static str, usize),
}

#[derive(Debug)]
struct AgentModel {
    id: String,
    observed_actions: Vec<String>,
    communication_style: String,
}

#[derive(Debug)]
struct MindState {
    agent_id: String,
    beliefs: Vec<String>,
    goals: Vec<String>,
    intentions: Vec<String>,
    theory_confidence: f32,
}

#[derive(Debug)]
struct ThinkingTrace {
    steps: Vec<ThinkingStep>,
    total_time: Duration,
}

#[derive(Debug)]
struct ThinkingStep {
    action: String,
    result: String,
    time_taken: Duration,
}

#[derive(Debug)]
struct Reflection {
    patterns_identified: Vec<ThinkingPattern>,
    effectiveness_score: f32,
    suggested_improvements: Vec<String>,
    meta_insights: Vec<String>,
}

#[derive(Debug)]
struct ThinkingPattern {
    pattern_type: String,
    occurrences: usize,
}

// Stub implementations for demo
impl MetaCognition {
    fn simulate_other_mind(&self, agent: &AgentModel) -> MindState {
        // Simplified theory of mind simulation
        let mut beliefs = vec![];
        let mut goals = vec![];
        
        // Infer from observed actions
        for action in &agent.observed_actions {
            if action.contains("optimization") {
                beliefs.push("Values efficiency and performance".to_string());
                goals.push("Optimize system performance".to_string());
            }
            if action.contains("examples") {
                beliefs.push("Prefers learning by example".to_string());
                goals.push("Understand through practical application".to_string());
            }
        }
        
        MindState {
            agent_id: agent.id.clone(),
            beliefs,
            goals,
            intentions: vec!["Learn and apply knowledge".to_string()],
            theory_confidence: 0.75,
        }
    }
    
    fn reflect_on_thinking(&mut self, _trace: &ThinkingTrace) -> Reflection {
        Reflection {
            patterns_identified: vec![
                ThinkingPattern {
                    pattern_type: "Sequential analysis".to_string(),
                    occurrences: 5,
                },
                ThinkingPattern {
                    pattern_type: "Hypothesis testing".to_string(),
                    occurrences: 3,
                },
            ],
            effectiveness_score: 0.82,
            suggested_improvements: vec![
                "Consider more creative alternatives in early stages".to_string(),
                "Reduce cognitive bias through systematic questioning".to_string(),
            ],
            meta_insights: vec![
                "Problem decomposition was effective".to_string(),
                "Time allocation could be optimized".to_string(),
            ],
        }
    }
}

// Stub for missing types
mod stubs {
    use super::*;
    
    pub struct IntrospectionAPI;
    
    impl IntrospectionAPI {
        pub fn get_cognitive_state(&self) -> CognitiveState {
            unimplemented!("Stub implementation")
        }
        
        pub fn explain_decision(&self, _id: &str) -> DecisionExplanation {
            DecisionExplanation {
                reasoning: "Logical deduction based on constraints".to_string(),
                key_factors: vec!["Performance requirements".to_string()],
                confidence_basis: "Historical success with similar problems".to_string(),
            }
        }
    }
    
    pub struct CognitiveState {
        pub active_patterns: Vec<CognitivePattern>,
        pub recent_decisions: Vec<Decision>,
        pub performance_metrics: MetaCognitiveMetrics,
    }
    
    pub struct DecisionExplanation {
        pub reasoning: String,
        pub key_factors: Vec<String>,
        pub confidence_basis: String,
    }
    
    pub struct CognitivePattern;
    pub struct Decision {
        pub id: String,
        pub option_selected: String,
    }
    pub struct MetaCognitiveMetrics {
        pub self_awareness_score: f32,
        pub average_confidence: f32,
    }
}

use stubs::*;

fn create_thinking_trace() -> ThinkingTrace {
    ThinkingTrace {
        steps: vec![
            ThinkingStep {
                action: "Analyze problem constraints".to_string(),
                result: "Identified 3 key constraints".to_string(),
                time_taken: Duration::from_secs(10),
            },
            ThinkingStep {
                action: "Generate solution candidates".to_string(),
                result: "Created 5 potential approaches".to_string(),
                time_taken: Duration::from_secs(30),
            },
        ],
        total_time: Duration::from_secs(40),
    }
}