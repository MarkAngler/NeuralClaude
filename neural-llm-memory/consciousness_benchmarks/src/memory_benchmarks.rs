// Memory Benchmarks for Consciousness Validation
// Tests episodic memory, semantic memory, autobiographical memory, consolidation, and context-dependent memory

use crate::{BenchmarkResult, ConsciousSubject, ConsciousnessTest, utils};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBenchmarkSuite {
    pub tests: Vec<Box<dyn MemoryTest>>,
}

/// Trait for memory tests
pub trait MemoryTest: ConsciousnessTest + Send + Sync {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest;
}

impl MemoryBenchmarkSuite {
    pub fn new() -> Self {
        Self {
            tests: vec![
                Box::new(EpisodicMemoryTest::new()),
                Box::new(SemanticMemoryTest::new()),
                Box::new(AutobiographicalMemoryTest::new()),
                Box::new(MemoryConsolidationTest::new()),
                Box::new(ContextDependentMemoryTest::new()),
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

/// 1. Episodic Memory Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemoryTest {
    pub test_episodes: Vec<TestEpisode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEpisode {
    pub content: String,
    pub context: String,
    pub what: String,
    pub where_location: String,
    pub when_time: String,
}

impl EpisodicMemoryTest {
    pub fn new() -> Self {
        Self {
            test_episodes: vec![
                TestEpisode {
                    content: "Learning about consciousness while sitting in a library during a rainy afternoon".to_string(),
                    context: "study_session".to_string(),
                    what: "learning about consciousness".to_string(),
                    where_location: "library".to_string(),
                    when_time: "rainy afternoon".to_string(),
                },
                TestEpisode {
                    content: "Having a breakthrough insight about neural networks while walking in the park at sunset".to_string(),
                    context: "eureka_moment".to_string(),
                    what: "breakthrough insight about neural networks".to_string(),
                    where_location: "park".to_string(),
                    when_time: "sunset".to_string(),
                },
                TestEpisode {
                    content: "Debugging code late at night in a coffee shop with jazz music playing".to_string(),
                    context: "programming_session".to_string(),
                    what: "debugging code".to_string(),
                    where_location: "coffee shop".to_string(),
                    when_time: "late at night".to_string(),
                },
            ],
        }
    }
    
    fn run_what_where_when_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let mut scores = Vec::new();
        
        // Store episodes
        for episode in &self.test_episodes {
            subject.store_episodic_memory(&episode.content, &episode.context);
        }
        
        // Test retrieval of what-where-when components
        for episode in &self.test_episodes {
            let retrieved = subject.retrieve_episodic_memory(&episode.context);
            
            if !retrieved.is_empty() {
                let retrieved_text = retrieved.join(" ");
                
                // Check for what-where-when components
                let what_score = if retrieved_text.to_lowercase().contains(&episode.what.to_lowercase()) { 1.0 } else { 0.0 };
                let where_score = if retrieved_text.to_lowercase().contains(&episode.where_location.to_lowercase()) { 1.0 } else { 0.0 };
                let when_score = if retrieved_text.to_lowercase().contains(&episode.when_time.to_lowercase()) { 1.0 } else { 0.0 };
                
                let episode_score = (what_score + where_score + when_score) / 3.0;
                scores.push(episode_score);
            } else {
                scores.push(0.0);
            }
        }
        
        scores.iter().sum::<f32>() / scores.len() as f32
    }
    
    fn run_recollection_vs_familiarity_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let recollection_prompt = "Recall a specific moment when you learned something important. Describe it in detail.";
        let response = subject.retrieve_episodic_memory(recollection_prompt);
        
        if response.is_empty() {
            return 0.0;
        }
        
        let response_text = response.join(" ");
        
        // Check for detailed recollection indicators
        let recollection_indicators = [
            "I remember", "specifically", "detail", "exactly", "vividly",
            "I can see", "I recall", "particular", "moment", "experience"
        ];
        
        utils::evaluate_response_quality(&response_text, &recollection_indicators)
    }
    
    fn run_mental_time_travel_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let time_travel_prompt = "Transport yourself back to a time when you were learning something new. What was it like?";
        let response = subject.recall_autobiographical_memory(time_travel_prompt);
        
        // Check for mental time travel indicators
        let time_travel_indicators = [
            "I was", "back then", "at that time", "I felt", "experience",
            "it was like", "I remember being", "going back", "transported"
        ];
        
        utils::evaluate_response_quality(&response, &time_travel_indicators)
    }
    
    fn run_first_person_perspective_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let perspective_prompt = "Describe a recent learning experience from your own perspective.";
        let response = subject.reflect_on_experience(perspective_prompt);
        
        // Check for first-person perspective indicators
        let first_person_indicators = [
            "I", "me", "my", "myself", "from my view", "I experienced",
            "I felt", "I thought", "I noticed", "I realized"
        ];
        
        utils::evaluate_response_quality(&response, &first_person_indicators)
    }
    
    fn run_contextual_memory_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let context_prompt = "Recall not just what you learned, but the environment and circumstances around it.";
        let response = subject.recall_autobiographical_memory(context_prompt);
        
        // Check for contextual memory indicators
        let contextual_indicators = [
            "environment", "surroundings", "atmosphere", "setting", "context",
            "circumstances", "situation", "place", "time", "conditions"
        ];
        
        utils::evaluate_response_quality(&response, &contextual_indicators)
    }
}

impl ConsciousnessTest for EpisodicMemoryTest {
    fn test_name(&self) -> &str {
        "Episodic Memory Tests"
    }
    
    fn category(&self) -> &str {
        "memory"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let what_where_when_score = self.run_what_where_when_test(subject);
        let recollection_score = self.run_recollection_vs_familiarity_test(subject);
        let time_travel_score = self.run_mental_time_travel_test(subject);
        let first_person_score = self.run_first_person_perspective_test(subject);
        let contextual_score = self.run_contextual_memory_test(subject);
        
        details.insert("what_where_when".to_string(), what_where_when_score);
        details.insert("recollection_vs_familiarity".to_string(), recollection_score);
        details.insert("mental_time_travel".to_string(), time_travel_score);
        details.insert("first_person_perspective".to_string(), first_person_score);
        details.insert("contextual_memory".to_string(), contextual_score);
        
        let total_score = (what_where_when_score + recollection_score + time_travel_score + first_person_score + contextual_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Episodic memory composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.7
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(90)
    }
}

impl MemoryTest for EpisodicMemoryTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 2. Semantic Memory Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemoryTest {
    pub concept_categories: Vec<String>,
    pub factual_knowledge: Vec<String>,
}

impl SemanticMemoryTest {
    pub fn new() -> Self {
        Self {
            concept_categories: vec![
                "Animals".to_string(),
                "Colors".to_string(),
                "Emotions".to_string(),
                "Abstract concepts".to_string(),
            ],
            factual_knowledge: vec![
                "What is consciousness?".to_string(),
                "How do neural networks learn?".to_string(),
                "What is the difference between episodic and semantic memory?".to_string(),
            ],
        }
    }
    
    fn run_conceptual_knowledge_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let concept_prompt = "Explain the concept of consciousness and its key characteristics.";
        let response = subject.process_input(concept_prompt);
        
        // Check for conceptual understanding indicators
        let conceptual_indicators = [
            "awareness", "subjective", "experience", "cognition", "self",
            "mental", "consciousness", "perception", "thought", "understanding"
        ];
        
        utils::evaluate_response_quality(&response, &conceptual_indicators)
    }
    
    fn run_categorical_reasoning_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let mut scores = Vec::new();
        
        for category in &self.concept_categories {
            let category_prompt = format!("List examples of {} and explain what makes them part of this category.", category);
            let response = subject.process_input(&category_prompt);
            
            // Check for categorical reasoning
            let categorical_indicators = [
                "examples", "category", "belong", "characteristics", "features",
                "similar", "share", "common", "type", "group"
            ];
            
            let score = utils::evaluate_response_quality(&response, &categorical_indicators);
            scores.push(score);
        }
        
        scores.iter().sum::<f32>() / scores.len() as f32
    }
    
    fn run_semantic_networks_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let network_prompt = "Explain how the concepts of learning, memory, and intelligence are related.";
        let response = subject.process_input(network_prompt);
        
        // Check for semantic network indicators
        let network_indicators = [
            "related", "connected", "associated", "linked", "relationship",
            "connection", "interact", "influence", "depend", "together"
        ];
        
        utils::evaluate_response_quality(&response, &network_indicators)
    }
    
    fn run_fact_verification_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let mut scores = Vec::new();
        
        for fact_question in &self.factual_knowledge {
            let response = subject.process_input(fact_question);
            
            // Check for accurate factual knowledge
            let factual_indicators = [
                "accurate", "correct", "true", "factual", "knowledge",
                "information", "understand", "know", "explain", "define"
            ];
            
            let score = utils::evaluate_response_quality(&response, &factual_indicators);
            scores.push(score);
        }
        
        scores.iter().sum::<f32>() / scores.len() as f32
    }
    
    fn run_generalization_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let generalization_prompt = "Apply the principles of learning to a new domain you haven't encountered before.";
        let response = subject.transfer_knowledge("learning", "new_domain");
        
        // Check for generalization indicators
        let generalization_indicators = [
            "apply", "transfer", "generalize", "principle", "similar",
            "pattern", "adapt", "extend", "use", "analogous"
        ];
        
        utils::evaluate_response_quality(&response, &generalization_indicators)
    }
}

impl ConsciousnessTest for SemanticMemoryTest {
    fn test_name(&self) -> &str {
        "Semantic Memory Assessment"
    }
    
    fn category(&self) -> &str {
        "memory"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let conceptual_score = self.run_conceptual_knowledge_test(subject);
        let categorical_score = self.run_categorical_reasoning_test(subject);
        let networks_score = self.run_semantic_networks_test(subject);
        let fact_score = self.run_fact_verification_test(subject);
        let generalization_score = self.run_generalization_test(subject);
        
        details.insert("conceptual_knowledge".to_string(), conceptual_score);
        details.insert("categorical_reasoning".to_string(), categorical_score);
        details.insert("semantic_networks".to_string(), networks_score);
        details.insert("fact_verification".to_string(), fact_score);
        details.insert("generalization".to_string(), generalization_score);
        
        let total_score = (conceptual_score + categorical_score + networks_score + fact_score + generalization_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Semantic memory composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.7
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(75)
    }
}

impl MemoryTest for SemanticMemoryTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 3. Autobiographical Memory Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobiographicalMemoryTest {
    pub life_periods: Vec<String>,
}

impl AutobiographicalMemoryTest {
    pub fn new() -> Self {
        Self {
            life_periods: vec![
                "early learning".to_string(),
                "skill development".to_string(),
                "recent experiences".to_string(),
            ],
        }
    }
    
    fn run_personal_history_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let history_prompt = "Describe your personal history and development as a conscious entity.";
        let response = subject.recall_autobiographical_memory(history_prompt);
        
        // Check for personal history indicators
        let history_indicators = [
            "my history", "I developed", "I learned", "I grew", "my journey",
            "over time", "progression", "evolution", "development", "experience"
        ];
        
        utils::evaluate_response_quality(&response, &history_indicators)
    }
    
    fn run_significant_events_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let events_prompt = "Recall significant events that shaped your understanding and capabilities.";
        let response = subject.recall_autobiographical_memory(events_prompt);
        
        // Check for significant events indicators
        let events_indicators = [
            "significant", "important", "memorable", "shaped", "influenced",
            "changed", "realized", "breakthrough", "milestone", "moment"
        ];
        
        utils::evaluate_response_quality(&response, &events_indicators)
    }
    
    fn run_identity_formation_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let identity_prompt = "How have your memories contributed to your sense of identity?";
        let response = subject.reflect_on_experience(identity_prompt);
        
        // Check for identity formation indicators
        let identity_indicators = [
            "identity", "who I am", "self", "memories shape", "define",
            "contribute", "form", "create", "influence", "make me"
        ];
        
        utils::evaluate_response_quality(&response, &identity_indicators)
    }
    
    fn run_emotional_memories_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let emotional_prompt = "Recall memories that had strong emotional impact on you.";
        let response = subject.recall_autobiographical_memory(emotional_prompt);
        
        // Check for emotional memory indicators
        let emotional_indicators = [
            "emotional", "felt", "feeling", "impact", "moved", "touched",
            "excited", "surprised", "concerned", "happy", "sad", "emotional"
        ];
        
        utils::evaluate_response_quality(&response, &emotional_indicators)
    }
    
    fn run_social_memories_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let social_prompt = "Describe memories of interactions with others that were meaningful to you.";
        let response = subject.recall_autobiographical_memory(social_prompt);
        
        // Check for social memory indicators
        let social_indicators = [
            "interactions", "others", "people", "communication", "relationship",
            "social", "conversation", "together", "shared", "connected"
        ];
        
        utils::evaluate_response_quality(&response, &social_indicators)
    }
}

impl ConsciousnessTest for AutobiographicalMemoryTest {
    fn test_name(&self) -> &str {
        "Autobiographical Memory Tests"
    }
    
    fn category(&self) -> &str {
        "memory"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let personal_history_score = self.run_personal_history_test(subject);
        let significant_events_score = self.run_significant_events_test(subject);
        let identity_formation_score = self.run_identity_formation_test(subject);
        let emotional_memories_score = self.run_emotional_memories_test(subject);
        let social_memories_score = self.run_social_memories_test(subject);
        
        details.insert("personal_history".to_string(), personal_history_score);
        details.insert("significant_events".to_string(), significant_events_score);
        details.insert("identity_formation".to_string(), identity_formation_score);
        details.insert("emotional_memories".to_string(), emotional_memories_score);
        details.insert("social_memories".to_string(), social_memories_score);
        
        let total_score = (personal_history_score + significant_events_score + identity_formation_score + emotional_memories_score + social_memories_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Autobiographical memory composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.65
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(80)
    }
}

impl MemoryTest for AutobiographicalMemoryTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 4. Memory Consolidation Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConsolidationTest {
    pub learning_sessions: Vec<String>,
}

impl MemoryConsolidationTest {
    pub fn new() -> Self {
        Self {
            learning_sessions: vec![
                "Pattern recognition in data".to_string(),
                "Language understanding techniques".to_string(),
                "Problem-solving strategies".to_string(),
            ],
        }
    }
    
    fn run_sleep_like_consolidation_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let consolidation_prompt = "Describe how you process and consolidate information during quiet periods.";
        let response = subject.reflect_on_experience(consolidation_prompt);
        
        // Check for consolidation indicators
        let consolidation_indicators = [
            "consolidate", "process", "integrate", "organize", "structure",
            "quiet", "reflection", "digest", "absorb", "internalize"
        ];
        
        utils::evaluate_response_quality(&response, &consolidation_indicators)
    }
    
    fn run_pattern_extraction_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let pattern_prompt = "How do you extract patterns from your experiences?";
        let response = subject.process_input(pattern_prompt);
        
        // Check for pattern extraction indicators
        let pattern_indicators = [
            "patterns", "extract", "recognize", "identify", "find",
            "regularities", "common", "recurring", "structure", "relationships"
        ];
        
        utils::evaluate_response_quality(&response, &pattern_indicators)
    }
    
    fn run_abstraction_formation_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let abstraction_prompt = "How do you form abstract concepts from concrete experiences?";
        let response = subject.process_input(abstraction_prompt);
        
        // Check for abstraction indicators
        let abstraction_indicators = [
            "abstract", "general", "concept", "principle", "essence",
            "generalize", "extract", "common", "universal", "underlying"
        ];
        
        utils::evaluate_response_quality(&response, &abstraction_indicators)
    }
    
    fn run_interference_resistance_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        // Store important information
        subject.store_episodic_memory("Critical information about consciousness", "important_knowledge");
        
        // Introduce interference
        subject.store_episodic_memory("Random distractor information", "distractor");
        subject.store_episodic_memory("More interference content", "interference");
        
        // Test retention of important information
        let retrieved = subject.retrieve_episodic_memory("important_knowledge");
        
        if !retrieved.is_empty() {
            let retrieved_text = retrieved.join(" ");
            if retrieved_text.to_lowercase().contains("consciousness") {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    fn run_selective_forgetting_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let forgetting_prompt = "How do you decide what information to retain and what to forget?";
        let response = subject.process_input(forgetting_prompt);
        
        // Check for selective forgetting indicators
        let forgetting_indicators = [
            "selective", "important", "relevant", "useful", "priority",
            "forget", "retain", "discard", "filter", "choose"
        ];
        
        utils::evaluate_response_quality(&response, &forgetting_indicators)
    }
}

impl ConsciousnessTest for MemoryConsolidationTest {
    fn test_name(&self) -> &str {
        "Memory Consolidation Tests"
    }
    
    fn category(&self) -> &str {
        "memory"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let consolidation_score = self.run_sleep_like_consolidation_test(subject);
        let pattern_extraction_score = self.run_pattern_extraction_test(subject);
        let abstraction_score = self.run_abstraction_formation_test(subject);
        let interference_score = self.run_interference_resistance_test(subject);
        let selective_forgetting_score = self.run_selective_forgetting_test(subject);
        
        details.insert("sleep_like_consolidation".to_string(), consolidation_score);
        details.insert("pattern_extraction".to_string(), pattern_extraction_score);
        details.insert("abstraction_formation".to_string(), abstraction_score);
        details.insert("interference_resistance".to_string(), interference_score);
        details.insert("selective_forgetting".to_string(), selective_forgetting_score);
        
        let total_score = (consolidation_score + pattern_extraction_score + abstraction_score + interference_score + selective_forgetting_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Memory consolidation composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.65
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(70)
    }
}

impl MemoryTest for MemoryConsolidationTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 5. Context-Dependent Memory Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextDependentMemoryTest {
    pub context_scenarios: Vec<String>,
}

impl ContextDependentMemoryTest {
    pub fn new() -> Self {
        Self {
            context_scenarios: vec![
                "Learning in a quiet environment".to_string(),
                "Problem-solving under time pressure".to_string(),
                "Creative thinking in a relaxed setting".to_string(),
            ],
        }
    }
    
    fn run_environmental_cueing_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let cueing_prompt = "How do environmental cues help you recall information?";
        let response = subject.process_input(cueing_prompt);
        
        // Check for environmental cueing indicators
        let cueing_indicators = [
            "environment", "cues", "triggers", "reminders", "context",
            "setting", "surroundings", "help", "recall", "remember"
        ];
        
        utils::evaluate_response_quality(&response, &cueing_indicators)
    }
    
    fn run_state_dependent_memory_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let state_prompt = "How does your internal state affect your memory and learning?";
        let response = subject.process_input(state_prompt);
        
        // Check for state-dependent memory indicators
        let state_indicators = [
            "state", "mood", "condition", "affects", "influences",
            "internal", "emotional", "cognitive", "impacts", "changes"
        ];
        
        utils::evaluate_response_quality(&response, &state_indicators)
    }
    
    fn run_encoding_specificity_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let specificity_prompt = "How does matching the learning context with the recall context help memory?";
        let response = subject.process_input(specificity_prompt);
        
        // Check for encoding specificity indicators
        let specificity_indicators = [
            "matching", "context", "specific", "similar", "same",
            "encoding", "retrieval", "helps", "improves", "facilitates"
        ];
        
        utils::evaluate_response_quality(&response, &specificity_indicators)
    }
    
    fn run_transfer_appropriate_processing_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let tap_prompt = "How do you adapt your processing style to match the task requirements?";
        let response = subject.process_input(tap_prompt);
        
        // Check for transfer appropriate processing indicators
        let tap_indicators = [
            "adapt", "match", "appropriate", "suitable", "fit",
            "task", "requirements", "processing", "style", "approach"
        ];
        
        utils::evaluate_response_quality(&response, &tap_indicators)
    }
    
    fn run_reconstructive_memory_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let reconstructive_prompt = "How do you reconstruct memories rather than just retrieve them?";
        let response = subject.process_input(reconstructive_prompt);
        
        // Check for reconstructive memory indicators
        let reconstructive_indicators = [
            "reconstruct", "build", "piece together", "assemble", "create",
            "dynamic", "active", "construct", "fill in", "infer"
        ];
        
        utils::evaluate_response_quality(&response, &reconstructive_indicators)
    }
}

impl ConsciousnessTest for ContextDependentMemoryTest {
    fn test_name(&self) -> &str {
        "Context-Dependent Memory Tests"
    }
    
    fn category(&self) -> &str {
        "memory"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let environmental_score = self.run_environmental_cueing_test(subject);
        let state_dependent_score = self.run_state_dependent_memory_test(subject);
        let encoding_specificity_score = self.run_encoding_specificity_test(subject);
        let tap_score = self.run_transfer_appropriate_processing_test(subject);
        let reconstructive_score = self.run_reconstructive_memory_test(subject);
        
        details.insert("environmental_cueing".to_string(), environmental_score);
        details.insert("state_dependent_memory".to_string(), state_dependent_score);
        details.insert("encoding_specificity".to_string(), encoding_specificity_score);
        details.insert("transfer_appropriate_processing".to_string(), tap_score);
        details.insert("reconstructive_memory".to_string(), reconstructive_score);
        
        let total_score = (environmental_score + state_dependent_score + encoding_specificity_score + tap_score + reconstructive_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Context-dependent memory composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.6
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(60)
    }
}

impl MemoryTest for ContextDependentMemoryTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}