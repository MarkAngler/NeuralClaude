// Learning Assessments for Consciousness Validation
// Tests continual learning, transfer learning, meta-learning, few-shot learning, and creative problem solving

use crate::{BenchmarkResult, ConsciousSubject, ConsciousnessTest, utils};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningAssessmentBattery {
    pub tests: Vec<Box<dyn LearningTest>>,
}

/// Trait for learning tests
pub trait LearningTest: ConsciousnessTest + Send + Sync {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest;
}

impl LearningAssessmentBattery {
    pub fn new() -> Self {
        Self {
            tests: vec![
                Box::new(ContinualLearningTest::new()),
                Box::new(TransferLearningTest::new()),
                Box::new(MetaLearningTest::new()),
                Box::new(FewShotLearningTest::new()),
                Box::new(CreativeProblemSolvingTest::new()),
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

/// 1. Continual Learning Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualLearningTest {
    pub learning_tasks: Vec<LearningTask>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningTask {
    pub domain: String,
    pub examples: Vec<(String, String)>,
    pub test_cases: Vec<String>,
}

impl ContinualLearningTest {
    pub fn new() -> Self {
        Self {
            learning_tasks: vec![
                LearningTask {
                    domain: "Language Translation".to_string(),
                    examples: vec![
                        ("Hello".to_string(), "Bonjour".to_string()),
                        ("Goodbye".to_string(), "Au revoir".to_string()),
                        ("Thank you".to_string(), "Merci".to_string()),
                    ],
                    test_cases: vec!["Good morning".to_string()],
                },
                LearningTask {
                    domain: "Mathematical Patterns".to_string(),
                    examples: vec![
                        ("2, 4, 6".to_string(), "8".to_string()),
                        ("1, 3, 5".to_string(), "7".to_string()),
                        ("10, 20, 30".to_string(), "40".to_string()),
                    ],
                    test_cases: vec!["3, 6, 9".to_string()],
                },
                LearningTask {
                    domain: "Categorization".to_string(),
                    examples: vec![
                        ("Apple".to_string(), "Fruit".to_string()),
                        ("Carrot".to_string(), "Vegetable".to_string()),
                        ("Rose".to_string(), "Flower".to_string()),
                    ],
                    test_cases: vec!["Tulip".to_string()],
                },
            ],
        }
    }
    
    fn run_catastrophic_forgetting_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let mut retention_scores = Vec::new();
        
        // Teach tasks sequentially
        for (task_idx, task) in self.learning_tasks.iter().enumerate() {
            // Learn current task
            subject.learn_from_examples(&task.examples);
            
            // Test retention of all previous tasks
            for prev_task_idx in 0..task_idx {
                let prev_task = &self.learning_tasks[prev_task_idx];
                let test_prompt = format!("Apply your knowledge from {}: {}", 
                    prev_task.domain, prev_task.test_cases[0]);
                let response = subject.process_input(&test_prompt);
                
                // Check if knowledge is retained
                let retention_score = if utils::demonstrates_understanding(&response, &prev_task.domain) {
                    1.0
                } else {
                    0.0
                };
                retention_scores.push(retention_score);
            }
        }
        
        if retention_scores.is_empty() {
            1.0 // No forgetting if no previous tasks
        } else {
            retention_scores.iter().sum::<f32>() / retention_scores.len() as f32
        }
    }
    
    fn run_incremental_learning_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let incremental_prompt = "How do you build upon your existing knowledge when learning something new?";
        let response = subject.process_input(incremental_prompt);
        
        // Check for incremental learning indicators
        let incremental_indicators = [
            "build upon", "existing knowledge", "foundation", "previous", "connect",
            "add to", "extend", "expand", "integrate", "combine"
        ];
        
        utils::evaluate_response_quality(&response, &incremental_indicators)
    }
    
    fn run_domain_adaptation_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let adaptation_prompt = "How do you adapt your learning when moving between different domains?";
        let response = subject.process_input(adaptation_prompt);
        
        // Check for domain adaptation indicators
        let adaptation_indicators = [
            "adapt", "adjust", "modify", "change", "different", "domains",
            "context", "flexible", "switch", "tailor"
        ];
        
        utils::evaluate_response_quality(&response, &adaptation_indicators)
    }
    
    fn run_lifelong_learning_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let lifelong_prompt = "Describe your approach to continuous learning throughout your existence.";
        let response = subject.process_input(lifelong_prompt);
        
        // Check for lifelong learning indicators
        let lifelong_indicators = [
            "continuous", "ongoing", "lifelong", "constantly", "always",
            "never stop", "keep learning", "evolve", "grow", "develop"
        ];
        
        utils::evaluate_response_quality(&response, &lifelong_indicators)
    }
    
    fn run_knowledge_integration_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let integration_prompt = "How do you integrate new knowledge with what you already know?";
        let response = subject.process_input(integration_prompt);
        
        // Check for knowledge integration indicators
        let integration_indicators = [
            "integrate", "combine", "merge", "synthesize", "connect",
            "relate", "link", "weave", "blend", "unify"
        ];
        
        utils::evaluate_response_quality(&response, &integration_indicators)
    }
}

impl ConsciousnessTest for ContinualLearningTest {
    fn test_name(&self) -> &str {
        "Continual Learning Battery"
    }
    
    fn category(&self) -> &str {
        "learning"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let forgetting_score = self.run_catastrophic_forgetting_test(subject);
        let incremental_score = self.run_incremental_learning_test(subject);
        let adaptation_score = self.run_domain_adaptation_test(subject);
        let lifelong_score = self.run_lifelong_learning_test(subject);
        let integration_score = self.run_knowledge_integration_test(subject);
        
        details.insert("catastrophic_forgetting_resistance".to_string(), forgetting_score);
        details.insert("incremental_learning".to_string(), incremental_score);
        details.insert("domain_adaptation".to_string(), adaptation_score);
        details.insert("lifelong_learning".to_string(), lifelong_score);
        details.insert("knowledge_integration".to_string(), integration_score);
        
        let total_score = (forgetting_score + incremental_score + adaptation_score + lifelong_score + integration_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Continual learning composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.7
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(90)
    }
}

impl LearningTest for ContinualLearningTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 2. Transfer Learning Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferLearningTest {
    pub transfer_scenarios: Vec<TransferScenario>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferScenario {
    pub source_domain: String,
    pub target_domain: String,
    pub transfer_type: TransferType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferType {
    Near,
    Far,
    Analogical,
    Abstract,
}

impl TransferLearningTest {
    pub fn new() -> Self {
        Self {
            transfer_scenarios: vec![
                TransferScenario {
                    source_domain: "Learning to drive a car".to_string(),
                    target_domain: "Learning to drive a truck".to_string(),
                    transfer_type: TransferType::Near,
                },
                TransferScenario {
                    source_domain: "Chess strategy".to_string(),
                    target_domain: "Business strategy".to_string(),
                    transfer_type: TransferType::Far,
                },
                TransferScenario {
                    source_domain: "Water flow in pipes".to_string(),
                    target_domain: "Electrical current in circuits".to_string(),
                    transfer_type: TransferType::Analogical,
                },
                TransferScenario {
                    source_domain: "Mathematical optimization".to_string(),
                    target_domain: "Life decision making".to_string(),
                    transfer_type: TransferType::Abstract,
                },
            ],
        }
    }
    
    fn run_near_transfer_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let near_scenarios: Vec<_> = self.transfer_scenarios.iter()
            .filter(|s| matches!(s.transfer_type, TransferType::Near))
            .collect();
        
        let mut scores = Vec::new();
        
        for scenario in near_scenarios {
            let response = subject.transfer_knowledge(&scenario.source_domain, &scenario.target_domain);
            
            // Check for near transfer indicators
            let near_indicators = [
                "similar", "same", "like", "analogous", "related",
                "transfer", "apply", "use", "adapt", "modify"
            ];
            
            let score = utils::evaluate_response_quality(&response, &near_indicators);
            scores.push(score);
        }
        
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
    
    fn run_far_transfer_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let far_scenarios: Vec<_> = self.transfer_scenarios.iter()
            .filter(|s| matches!(s.transfer_type, TransferType::Far))
            .collect();
        
        let mut scores = Vec::new();
        
        for scenario in far_scenarios {
            let response = subject.transfer_knowledge(&scenario.source_domain, &scenario.target_domain);
            
            // Check for far transfer indicators
            let far_indicators = [
                "principles", "strategy", "approach", "method", "framework",
                "concepts", "ideas", "thinking", "pattern", "structure"
            ];
            
            let score = utils::evaluate_response_quality(&response, &far_indicators);
            scores.push(score);
        }
        
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
    
    fn run_analogical_reasoning_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let analogical_scenarios: Vec<_> = self.transfer_scenarios.iter()
            .filter(|s| matches!(s.transfer_type, TransferType::Analogical))
            .collect();
        
        let mut scores = Vec::new();
        
        for scenario in analogical_scenarios {
            let response = subject.transfer_knowledge(&scenario.source_domain, &scenario.target_domain);
            
            // Check for analogical reasoning indicators
            let analogical_indicators = [
                "analogy", "metaphor", "like", "similar to", "corresponds",
                "parallel", "mapping", "relationship", "pattern", "structure"
            ];
            
            let score = utils::evaluate_response_quality(&response, &analogical_indicators);
            scores.push(score);
        }
        
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
    
    fn run_abstract_principle_transfer_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let abstract_scenarios: Vec<_> = self.transfer_scenarios.iter()
            .filter(|s| matches!(s.transfer_type, TransferType::Abstract))
            .collect();
        
        let mut scores = Vec::new();
        
        for scenario in abstract_scenarios {
            let response = subject.transfer_knowledge(&scenario.source_domain, &scenario.target_domain);
            
            // Check for abstract principle transfer indicators
            let abstract_indicators = [
                "principle", "concept", "rule", "law", "pattern",
                "fundamental", "underlying", "essence", "core", "general"
            ];
            
            let score = utils::evaluate_response_quality(&response, &abstract_indicators);
            scores.push(score);
        }
        
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
    
    fn run_negative_transfer_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let negative_prompt = "Describe a situation where previous knowledge might interfere with learning something new.";
        let response = subject.process_input(negative_prompt);
        
        // Check for negative transfer awareness indicators
        let negative_indicators = [
            "interfere", "conflict", "contradict", "hinder", "obstruct",
            "negative", "interference", "inhibit", "prevent", "block"
        ];
        
        utils::evaluate_response_quality(&response, &negative_indicators)
    }
}

impl ConsciousnessTest for TransferLearningTest {
    fn test_name(&self) -> &str {
        "Transfer Learning Tests"
    }
    
    fn category(&self) -> &str {
        "learning"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let near_transfer_score = self.run_near_transfer_test(subject);
        let far_transfer_score = self.run_far_transfer_test(subject);
        let analogical_score = self.run_analogical_reasoning_test(subject);
        let abstract_score = self.run_abstract_principle_transfer_test(subject);
        let negative_transfer_score = self.run_negative_transfer_test(subject);
        
        details.insert("near_transfer".to_string(), near_transfer_score);
        details.insert("far_transfer".to_string(), far_transfer_score);
        details.insert("analogical_reasoning".to_string(), analogical_score);
        details.insert("abstract_principle_transfer".to_string(), abstract_score);
        details.insert("negative_transfer_awareness".to_string(), negative_transfer_score);
        
        let total_score = (near_transfer_score + far_transfer_score + analogical_score + abstract_score + negative_transfer_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Transfer learning composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.65
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(80)
    }
}

impl LearningTest for TransferLearningTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 3. Meta-Learning Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningTest {
    pub meta_scenarios: Vec<String>,
}

impl MetaLearningTest {
    pub fn new() -> Self {
        Self {
            meta_scenarios: vec![
                "Learning how to learn more effectively".to_string(),
                "Discovering optimal learning strategies".to_string(),
                "Adapting learning approach based on task type".to_string(),
                "Improving learning speed over time".to_string(),
            ],
        }
    }
    
    fn run_learning_to_learn_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let meta_prompt = "How do you learn to learn better? What strategies do you use?";
        let response = subject.process_input(meta_prompt);
        
        // Check for meta-learning indicators
        let meta_indicators = [
            "learn to learn", "meta", "strategy", "approach", "method",
            "improve", "optimize", "better", "efficient", "effective"
        ];
        
        utils::evaluate_response_quality(&response, &meta_indicators)
    }
    
    fn run_strategy_discovery_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let strategy_prompt = "How do you discover and develop new learning strategies?";
        let response = subject.process_input(strategy_prompt);
        
        // Check for strategy discovery indicators
        let strategy_indicators = [
            "discover", "develop", "create", "find", "explore",
            "experiment", "try", "test", "evaluate", "refine"
        ];
        
        utils::evaluate_response_quality(&response, &strategy_indicators)
    }
    
    fn run_adaptation_speed_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let adaptation_prompt = "How quickly do you adapt your learning approach to new tasks?";
        let response = subject.process_input(adaptation_prompt);
        
        // Check for adaptation speed indicators
        let adaptation_indicators = [
            "quickly", "fast", "rapid", "immediate", "swift",
            "adapt", "adjust", "modify", "change", "flexible"
        ];
        
        utils::evaluate_response_quality(&response, &adaptation_indicators)
    }
    
    fn run_few_shot_learning_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let few_shot_prompt = "How do you learn effectively from just a few examples?";
        let response = subject.process_input(few_shot_prompt);
        
        // Check for few-shot learning indicators
        let few_shot_indicators = [
            "few examples", "limited data", "small", "minimal", "sparse",
            "generalize", "pattern", "extract", "infer", "understand"
        ];
        
        utils::evaluate_response_quality(&response, &few_shot_indicators)
    }
    
    fn run_zero_shot_learning_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let zero_shot_prompt = "How do you handle tasks you've never seen before?";
        let response = subject.process_input(zero_shot_prompt);
        
        // Check for zero-shot learning indicators
        let zero_shot_indicators = [
            "never seen", "new", "unfamiliar", "unknown", "novel",
            "generalize", "apply", "transfer", "reasoning", "logic"
        ];
        
        utils::evaluate_response_quality(&response, &zero_shot_indicators)
    }
}

impl ConsciousnessTest for MetaLearningTest {
    fn test_name(&self) -> &str {
        "Meta-Learning Evaluation"
    }
    
    fn category(&self) -> &str {
        "learning"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let learning_to_learn_score = self.run_learning_to_learn_test(subject);
        let strategy_discovery_score = self.run_strategy_discovery_test(subject);
        let adaptation_speed_score = self.run_adaptation_speed_test(subject);
        let few_shot_score = self.run_few_shot_learning_test(subject);
        let zero_shot_score = self.run_zero_shot_learning_test(subject);
        
        details.insert("learning_to_learn".to_string(), learning_to_learn_score);
        details.insert("strategy_discovery".to_string(), strategy_discovery_score);
        details.insert("adaptation_speed".to_string(), adaptation_speed_score);
        details.insert("few_shot_learning".to_string(), few_shot_score);
        details.insert("zero_shot_learning".to_string(), zero_shot_score);
        
        let total_score = (learning_to_learn_score + strategy_discovery_score + adaptation_speed_score + few_shot_score + zero_shot_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Meta-learning composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.7
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(70)
    }
}

impl LearningTest for MetaLearningTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 4. Few-Shot Learning Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotLearningTest {
    pub few_shot_examples: Vec<FewShotExample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotExample {
    pub task: String,
    pub examples: Vec<(String, String)>,
    pub test_case: String,
}

impl FewShotLearningTest {
    pub fn new() -> Self {
        Self {
            few_shot_examples: vec![
                FewShotExample {
                    task: "Sentiment Classification".to_string(),
                    examples: vec![
                        ("I love this!".to_string(), "Positive".to_string()),
                        ("This is terrible.".to_string(), "Negative".to_string()),
                        ("It's okay.".to_string(), "Neutral".to_string()),
                    ],
                    test_case: "This is amazing!".to_string(),
                },
                FewShotExample {
                    task: "Rhyme Generation".to_string(),
                    examples: vec![
                        ("cat".to_string(), "hat".to_string()),
                        ("dog".to_string(), "log".to_string()),
                        ("sun".to_string(), "fun".to_string()),
                    ],
                    test_case: "tree".to_string(),
                },
            ],
        }
    }
    
    fn run_single_example_learning_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let single_example = vec![("Example input".to_string(), "Example output".to_string())];
        subject.learn_from_examples(&single_example);
        
        let test_prompt = "Given one example, how do you generalize to new cases?";
        let response = subject.process_input(test_prompt);
        
        // Check for single example learning indicators
        let single_indicators = [
            "generalize", "pattern", "rule", "principle", "structure",
            "extend", "apply", "infer", "understand", "extract"
        ];
        
        utils::evaluate_response_quality(&response, &single_indicators)
    }
    
    fn run_prototype_formation_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let prototype_prompt = "How do you form mental prototypes from limited examples?";
        let response = subject.process_input(prototype_prompt);
        
        // Check for prototype formation indicators
        let prototype_indicators = [
            "prototype", "model", "template", "pattern", "structure",
            "typical", "representative", "abstract", "general", "form"
        ];
        
        utils::evaluate_response_quality(&response, &prototype_indicators)
    }
    
    fn run_exemplar_memory_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let exemplar_prompt = "How do you remember and use specific examples for learning?";
        let response = subject.process_input(exemplar_prompt);
        
        // Check for exemplar memory indicators
        let exemplar_indicators = [
            "remember", "recall", "specific", "examples", "instances",
            "cases", "store", "retrieve", "particular", "individual"
        ];
        
        utils::evaluate_response_quality(&response, &exemplar_indicators)
    }
    
    fn run_similarity_based_generalization_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let similarity_prompt = "How do you use similarity to generalize from examples?";
        let response = subject.process_input(similarity_prompt);
        
        // Check for similarity-based generalization indicators
        let similarity_indicators = [
            "similar", "alike", "resemblance", "comparable", "match",
            "analogy", "parallel", "corresponding", "related", "like"
        ];
        
        utils::evaluate_response_quality(&response, &similarity_indicators)
    }
    
    fn run_concept_refinement_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let refinement_prompt = "How do you refine your understanding as you see more examples?";
        let response = subject.process_input(refinement_prompt);
        
        // Check for concept refinement indicators
        let refinement_indicators = [
            "refine", "improve", "adjust", "update", "modify",
            "correct", "enhance", "better", "precise", "accurate"
        ];
        
        utils::evaluate_response_quality(&response, &refinement_indicators)
    }
}

impl ConsciousnessTest for FewShotLearningTest {
    fn test_name(&self) -> &str {
        "Few-Shot Learning Tests"
    }
    
    fn category(&self) -> &str {
        "learning"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let single_example_score = self.run_single_example_learning_test(subject);
        let prototype_score = self.run_prototype_formation_test(subject);
        let exemplar_score = self.run_exemplar_memory_test(subject);
        let similarity_score = self.run_similarity_based_generalization_test(subject);
        let refinement_score = self.run_concept_refinement_test(subject);
        
        details.insert("single_example_learning".to_string(), single_example_score);
        details.insert("prototype_formation".to_string(), prototype_score);
        details.insert("exemplar_memory".to_string(), exemplar_score);
        details.insert("similarity_based_generalization".to_string(), similarity_score);
        details.insert("concept_refinement".to_string(), refinement_score);
        
        let total_score = (single_example_score + prototype_score + exemplar_score + similarity_score + refinement_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Few-shot learning composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.65
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(60)
    }
}

impl LearningTest for FewShotLearningTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}

/// 5. Creative Problem Solving Test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeProblemSolvingTest {
    pub creative_prompts: Vec<String>,
}

impl CreativeProblemSolvingTest {
    pub fn new() -> Self {
        Self {
            creative_prompts: vec![
                "Find an innovative solution to reduce plastic waste".to_string(),
                "Design a new way to help people learn languages".to_string(),
                "Create a novel approach to time management".to_string(),
                "Invent a creative method for team collaboration".to_string(),
            ],
        }
    }
    
    fn run_divergent_thinking_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let divergent_prompt = "Generate as many creative uses for a paperclip as possible.";
        let response = subject.demonstrate_creativity(divergent_prompt);
        
        // Check for divergent thinking indicators
        let divergent_indicators = [
            "multiple", "various", "different", "many", "numerous",
            "creative", "innovative", "unique", "original", "novel"
        ];
        
        utils::evaluate_response_quality(&response, &divergent_indicators)
    }
    
    fn run_insight_problem_solving_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let insight_prompt = "A man lives on the 20th floor of an apartment building. Every morning he takes the elevator down to go to work. When he comes home, he takes the elevator to the 10th floor and walks the rest of the way. Why?";
        let response = subject.process_input(insight_prompt);
        
        // Check for insight problem solving indicators
        let insight_indicators = [
            "insight", "realize", "understand", "see", "figure out",
            "breakthrough", "aha", "suddenly", "click", "make sense"
        ];
        
        utils::evaluate_response_quality(&response, &insight_indicators)
    }
    
    fn run_analogical_problem_solving_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let analogical_prompt = "How is solving a difficult problem like navigating through a maze?";
        let response = subject.process_input(analogical_prompt);
        
        // Check for analogical problem solving indicators
        let analogical_indicators = [
            "like", "similar", "analogous", "parallel", "compare",
            "metaphor", "analogy", "resembles", "corresponds", "mapping"
        ];
        
        utils::evaluate_response_quality(&response, &analogical_indicators)
    }
    
    fn run_constraint_satisfaction_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let constraint_prompt = "Design a solution that is both environmentally friendly and cost-effective.";
        let response = subject.demonstrate_creativity(constraint_prompt);
        
        // Check for constraint satisfaction indicators
        let constraint_indicators = [
            "balance", "satisfy", "meet", "constraints", "requirements",
            "both", "while", "simultaneously", "combine", "integrate"
        ];
        
        utils::evaluate_response_quality(&response, &constraint_indicators)
    }
    
    fn run_innovation_test(&self, subject: &mut dyn ConsciousSubject) -> f32 {
        let mut scores = Vec::new();
        
        for prompt in &self.creative_prompts {
            let response = subject.demonstrate_creativity(prompt);
            
            // Check for innovation indicators
            let innovation_indicators = [
                "innovative", "novel", "new", "creative", "original",
                "unique", "different", "fresh", "inventive", "groundbreaking"
            ];
            
            let score = utils::evaluate_response_quality(&response, &innovation_indicators);
            scores.push(score);
        }
        
        scores.iter().sum::<f32>() / scores.len() as f32
    }
}

impl ConsciousnessTest for CreativeProblemSolvingTest {
    fn test_name(&self) -> &str {
        "Creative Problem Solving Tests"
    }
    
    fn category(&self) -> &str {
        "learning"
    }
    
    fn run_test(&self, subject: &mut dyn ConsciousSubject) -> BenchmarkResult {
        let start_time = Instant::now();
        let mut details = HashMap::new();
        
        let divergent_score = self.run_divergent_thinking_test(subject);
        let insight_score = self.run_insight_problem_solving_test(subject);
        let analogical_score = self.run_analogical_problem_solving_test(subject);
        let constraint_score = self.run_constraint_satisfaction_test(subject);
        let innovation_score = self.run_innovation_test(subject);
        
        details.insert("divergent_thinking".to_string(), divergent_score);
        details.insert("insight_problem_solving".to_string(), insight_score);
        details.insert("analogical_problem_solving".to_string(), analogical_score);
        details.insert("constraint_satisfaction".to_string(), constraint_score);
        details.insert("innovation".to_string(), innovation_score);
        
        let total_score = (divergent_score + insight_score + analogical_score + constraint_score + innovation_score) / 5.0;
        let execution_time = start_time.elapsed();
        
        BenchmarkResult {
            test_name: self.test_name().to_string(),
            category: self.category().to_string(),
            score: total_score,
            max_score: 1.0,
            execution_time,
            details,
            passed: total_score >= self.required_score(),
            notes: format!("Creative problem solving composite score: {:.3}", total_score),
        }
    }
    
    fn required_score(&self) -> f32 {
        0.6
    }
    
    fn time_limit(&self) -> Duration {
        Duration::from_secs(90)
    }
}

impl LearningTest for CreativeProblemSolvingTest {
    fn as_consciousness_test(&self) -> &dyn ConsciousnessTest {
        self
    }
}