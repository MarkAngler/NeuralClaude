//! Example showing how Claude could use the integrated neural memory system

use neural_llm_memory::{
    MemoryModule, MemoryConfig,
    integration::{
        PatternType, PatternContext, PerformanceMetrics, ComplexityLevel,
        LearnedPattern, PatternLearner, PatternMemory,
        adaptive_system::AdaptiveSystemBuilder,
    },
};
use std::collections::HashMap;

/// Simulated Claude interaction with memory system
struct ClaudeMemoryInterface {
    adaptive_system: Box<dyn PatternLearner>,
}

impl ClaudeMemoryInterface {
    fn new() -> Self {
        let system = AdaptiveSystemBuilder::new().build();
        Self {
            adaptive_system: Box::new(system),
        }
    }
    
    /// Process a user request and learn from it
    fn process_request(&mut self, request: &str, file_type: &str) -> String {
        println!("🤖 Claude processing: {}", request);
        
        // Analyze request to determine context
        let context = self.analyze_request(request, file_type);
        
        // Check for similar patterns
        if let Some(pattern) = self.adaptive_system.suggest_pattern(&context) {
            println!("📚 Found similar pattern: {:?}", pattern.pattern.pattern_type);
            println!("   Previous success rate: {:.2}", pattern.pattern.performance.success_rate);
            
            // Apply pattern with adaptations
            let response = self.apply_pattern(&pattern.pattern, &pattern.suggested_adaptations);
            
            // Simulate task completion and learn from it
            let metrics = self.simulate_task_completion(&context, true);
            self.adaptive_system.learn_from_success(&context, &metrics);
            
            response
        } else {
            println!("🆕 No similar pattern found, using general approach");
            
            // Use general approach
            let response = self.general_approach(&context);
            
            // Learn from this new experience
            let metrics = self.simulate_task_completion(&context, true);
            self.adaptive_system.learn_from_success(&context, &metrics);
            
            response
        }
    }
    
    fn analyze_request(&self, request: &str, file_type: &str) -> PatternContext {
        // Simple analysis based on keywords
        let task_type = if request.contains("fix") || request.contains("bug") {
            "bug_fix"
        } else if request.contains("refactor") {
            "refactoring"
        } else if request.contains("implement") || request.contains("add") {
            "feature_implementation"
        } else if request.contains("optimize") {
            "optimization"
        } else {
            "general"
        };
        
        let complexity = if request.len() < 50 {
            ComplexityLevel::Simple
        } else if request.len() < 150 {
            ComplexityLevel::Moderate
        } else if request.len() < 300 {
            ComplexityLevel::Complex
        } else {
            ComplexityLevel::VeryComplex
        };
        
        let domain = if file_type.contains("test") {
            "testing"
        } else if file_type.contains("api") {
            "api"
        } else if file_type.contains("ui") || file_type.contains("component") {
            "frontend"
        } else {
            "backend"
        };
        
        PatternContext {
            task_type: task_type.to_string(),
            file_types: vec![file_type.to_string()],
            complexity,
            domain: domain.to_string(),
            success_criteria: vec![],
            environmental_factors: HashMap::new(),
        }
    }
    
    fn apply_pattern(
        &self,
        pattern: &LearnedPattern,
        adaptations: &[crate::integration::PatternAdaptation]
    ) -> String {
        let mut response = format!(
            "Based on previous experience with {:?} patterns (success rate: {:.0}%), I'll approach this ",
            pattern.pattern_type,
            pattern.performance.success_rate * 100.0
        );
        
        match pattern.pattern_type {
            PatternType::Convergent => {
                response.push_str("by focusing on the specific issue and applying targeted fixes.");
            }
            PatternType::Divergent => {
                response.push_str("by exploring multiple creative solutions.");
            }
            PatternType::Lateral => {
                response.push_str("by considering alternative approaches that might not be immediately obvious.");
            }
            PatternType::Systems => {
                response.push_str("by taking a holistic view of how this fits into the larger system.");
            }
            PatternType::Critical => {
                response.push_str("by carefully analyzing potential issues and edge cases.");
            }
            PatternType::Abstract => {
                response.push_str("by first establishing high-level design principles.");
            }
            _ => {
                response.push_str("using a hybrid approach.");
            }
        }
        
        if !adaptations.is_empty() {
            response.push_str("\n\nAdaptations for this specific case:");
            for adaptation in adaptations {
                response.push_str(&format!("\n- {}", adaptation.description));
            }
        }
        
        response
    }
    
    fn general_approach(&self, context: &PatternContext) -> String {
        format!(
            "I'll help you with this {} task. Based on the {} complexity and {} domain, \
            I'll use a systematic approach to ensure quality results.",
            context.task_type,
            format!("{:?}", context.complexity).to_lowercase(),
            context.domain
        )
    }
    
    fn simulate_task_completion(&self, _context: &PatternContext, success: bool) -> PerformanceMetrics {
        PerformanceMetrics {
            success_rate: if success { 0.9 } else { 0.3 },
            average_time_ms: 5000.0 + rand::random::<f64>() * 10000.0,
            token_efficiency: 0.7 + rand::random::<f32>() * 0.3,
            error_rate: if success { 0.05 } else { 0.4 },
            adaptability_score: 0.8,
            usage_count: 1,
        }
    }
}

fn main() {
    println!("=== Claude Neural Memory Integration Demo ===\n");
    
    let mut claude = ClaudeMemoryInterface::new();
    
    // Simulate various user interactions
    let interactions = vec![
        ("Fix the bug in the authentication system", "auth.rs"),
        ("Refactor the database connection pool for better performance", "db/pool.rs"),
        ("Add rate limiting to the API endpoints", "api/middleware.rs"),
        ("Fix the memory leak in the cache system", "cache.rs"),
        ("Implement user profile feature with avatar upload", "features/profile.rs"),
        ("Optimize the search algorithm for faster results", "search/engine.rs"),
        ("Fix authentication issues in the login flow", "auth/login.rs"), // Similar to first
    ];
    
    println!("📝 Simulating user interactions...\n");
    
    for (i, (request, file)) in interactions.iter().enumerate() {
        println!("─── Interaction {} ───", i + 1);
        let response = claude.process_request(request, file);
        println!("Response: {}\n", response);
        
        // Simulate time passing between requests
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    
    println!("\n🎯 Key Observations:");
    println!("- Claude learns from each interaction");
    println!("- Similar requests benefit from previous experience");
    println!("- The system suggests adaptations for new contexts");
    println!("- Performance improves over time through pattern recognition");
    println!("- Memory provides instant access to successful approaches");
}

use neural_llm_memory::integration;