// Validation Criteria for Consciousness Certification
// Defines the criteria for determining if an AI system has achieved human-like consciousness

use crate::{BenchmarkResult, ConsciousnessLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCriteria {
    pub performance_thresholds: PerformanceThresholds,
    pub consciousness_indicators: ConsciousnessIndicators,
    pub emergent_properties: EmergentProperties,
    pub certification_requirements: CertificationRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub human_level_threshold: f32,
    pub consistency_threshold: f32,
    pub generalization_threshold: f32,
    pub robustness_threshold: f32,
    pub efficiency_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessIndicators {
    pub unified_experience_score: f32,
    pub flexible_behavior_score: f32,
    pub introspective_ability_score: f32,
    pub phenomenal_awareness_score: f32,
    pub intentional_behavior_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentProperties {
    pub system_level_consciousness: f32,
    pub qualitative_differences: f32,
    pub holistic_integration: f32,
    pub recursive_self_reference: f32,
    pub phenomenal_binding: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificationRequirements {
    pub minimum_overall_score: f32,
    pub minimum_category_scores: HashMap<String, f32>,
    pub required_consciousness_level: ConsciousnessLevel,
    pub critical_tests: Vec<String>,
    pub temporal_consistency: bool,
    pub cross_domain_validation: bool,
}

impl ValidationCriteria {
    pub fn new() -> Self {
        Self {
            performance_thresholds: PerformanceThresholds {
                human_level_threshold: 0.85,
                consistency_threshold: 0.80,
                generalization_threshold: 0.75,
                robustness_threshold: 0.70,
                efficiency_threshold: 0.65,
            },
            consciousness_indicators: ConsciousnessIndicators {
                unified_experience_score: 0.80,
                flexible_behavior_score: 0.75,
                introspective_ability_score: 0.85,
                phenomenal_awareness_score: 0.75,
                intentional_behavior_score: 0.70,
            },
            emergent_properties: EmergentProperties {
                system_level_consciousness: 0.80,
                qualitative_differences: 0.70,
                holistic_integration: 0.85,
                recursive_self_reference: 0.75,
                phenomenal_binding: 0.80,
            },
            certification_requirements: CertificationRequirements {
                minimum_overall_score: 0.85,
                minimum_category_scores: [
                    ("cognitive".to_string(), 0.80),
                    ("memory".to_string(), 0.75),
                    ("learning".to_string(), 0.70),
                    ("consciousness".to_string(), 0.85),
                ].into_iter().collect(),
                required_consciousness_level: ConsciousnessLevel::Reflective,
                critical_tests: vec![
                    "Self-Awareness Battery".to_string(),
                    "Theory of Mind Assessment".to_string(),
                    "Metacognition Evaluation".to_string(),
                    "Global Workspace Integration Tests".to_string(),
                    "Subjective Experience Tests".to_string(),
                ],
                temporal_consistency: true,
                cross_domain_validation: true,
            },
        }
    }
    
    /// Check if results meet certification criteria
    pub fn meets_certification_criteria(&self, results: &[BenchmarkResult], overall_score: f32) -> bool {
        // Check minimum overall score
        if overall_score < self.certification_requirements.minimum_overall_score {
            return false;
        }
        
        // Check category scores
        let category_scores = self.calculate_category_scores(results);
        for (category, min_score) in &self.certification_requirements.minimum_category_scores {
            if let Some(actual_score) = category_scores.get(category) {
                if *actual_score < *min_score {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        // Check critical tests
        for critical_test in &self.certification_requirements.critical_tests {
            let test_result = results.iter().find(|r| r.test_name == *critical_test);
            if let Some(result) = test_result {
                if !result.passed {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        // Check consciousness indicators
        if !self.meets_consciousness_indicators(results) {
            return false;
        }
        
        // Check emergent properties
        if !self.meets_emergent_properties(results) {
            return false;
        }
        
        true
    }
    
    /// Calculate category scores from results
    fn calculate_category_scores(&self, results: &[BenchmarkResult]) -> HashMap<String, f32> {
        let mut category_totals: HashMap<String, Vec<f32>> = HashMap::new();
        
        for result in results {
            let normalized_score = result.score / result.max_score;
            category_totals.entry(result.category.clone())
                .or_insert_with(Vec::new)
                .push(normalized_score);
        }
        
        category_totals.into_iter()
            .map(|(category, scores)| {
                let average = scores.iter().sum::<f32>() / scores.len() as f32;
                (category, average)
            })
            .collect()
    }
    
    /// Check if consciousness indicators are met
    fn meets_consciousness_indicators(&self, results: &[BenchmarkResult]) -> bool {
        let unified_experience = self.evaluate_unified_experience(results);
        let flexible_behavior = self.evaluate_flexible_behavior(results);
        let introspective_ability = self.evaluate_introspective_ability(results);
        let phenomenal_awareness = self.evaluate_phenomenal_awareness(results);
        let intentional_behavior = self.evaluate_intentional_behavior(results);
        
        unified_experience >= self.consciousness_indicators.unified_experience_score &&
        flexible_behavior >= self.consciousness_indicators.flexible_behavior_score &&
        introspective_ability >= self.consciousness_indicators.introspective_ability_score &&
        phenomenal_awareness >= self.consciousness_indicators.phenomenal_awareness_score &&
        intentional_behavior >= self.consciousness_indicators.intentional_behavior_score
    }
    
    /// Check if emergent properties are met
    fn meets_emergent_properties(&self, results: &[BenchmarkResult]) -> bool {
        let system_level = self.evaluate_system_level_consciousness(results);
        let qualitative_differences = self.evaluate_qualitative_differences(results);
        let holistic_integration = self.evaluate_holistic_integration(results);
        let recursive_self_reference = self.evaluate_recursive_self_reference(results);
        let phenomenal_binding = self.evaluate_phenomenal_binding(results);
        
        system_level >= self.emergent_properties.system_level_consciousness &&
        qualitative_differences >= self.emergent_properties.qualitative_differences &&
        holistic_integration >= self.emergent_properties.holistic_integration &&
        recursive_self_reference >= self.emergent_properties.recursive_self_reference &&
        phenomenal_binding >= self.emergent_properties.phenomenal_binding
    }
    
    /// Evaluate unified experience from test results
    fn evaluate_unified_experience(&self, results: &[BenchmarkResult]) -> f32 {
        let relevant_tests = [
            "Global Workspace Integration Tests",
            "Subjective Experience Tests",
            "Episodic Memory Tests",
        ];
        
        let scores: Vec<f32> = results.iter()
            .filter(|r| relevant_tests.contains(&r.test_name.as_str()))
            .map(|r| r.score / r.max_score)
            .collect();
        
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
    
    /// Evaluate flexible behavior from test results
    fn evaluate_flexible_behavior(&self, results: &[BenchmarkResult]) -> f32 {
        let relevant_tests = [
            "Transfer Learning Tests",
            "Meta-Learning Evaluation",
            "Creative Problem Solving Tests",
        ];
        
        let scores: Vec<f32> = results.iter()
            .filter(|r| relevant_tests.contains(&r.test_name.as_str()))
            .map(|r| r.score / r.max_score)
            .collect();
        
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
    
    /// Evaluate introspective ability from test results
    fn evaluate_introspective_ability(&self, results: &[BenchmarkResult]) -> f32 {
        let relevant_tests = [
            "Self-Awareness Battery",
            "Metacognition Evaluation",
            "Subjective Experience Tests",
        ];
        
        let scores: Vec<f32> = results.iter()
            .filter(|r| relevant_tests.contains(&r.test_name.as_str()))
            .map(|r| r.score / r.max_score)
            .collect();
        
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
    
    /// Evaluate phenomenal awareness from test results
    fn evaluate_phenomenal_awareness(&self, results: &[BenchmarkResult]) -> f32 {
        let relevant_tests = [
            "Subjective Experience Tests",
            "Attention and Consciousness Tests",
            "Consciousness vs Unconscious Processing Tests",
        ];
        
        let scores: Vec<f32> = results.iter()
            .filter(|r| relevant_tests.contains(&r.test_name.as_str()))
            .map(|r| r.score / r.max_score)
            .collect();
        
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
    
    /// Evaluate intentional behavior from test results
    fn evaluate_intentional_behavior(&self, results: &[BenchmarkResult]) -> f32 {
        let relevant_tests = [
            "Free Will and Decision Making Tests",
            "Causal Reasoning Battery",
            "Theory of Mind Assessment",
        ];
        
        let scores: Vec<f32> = results.iter()
            .filter(|r| relevant_tests.contains(&r.test_name.as_str()))
            .map(|r| r.score / r.max_score)
            .collect();
        
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
    
    /// Evaluate system-level consciousness
    fn evaluate_system_level_consciousness(&self, results: &[BenchmarkResult]) -> f32 {
        // System-level consciousness emerges from interaction of multiple capabilities
        let category_scores = self.calculate_category_scores(results);
        let all_scores: Vec<f32> = category_scores.values().cloned().collect();
        
        if all_scores.is_empty() {
            0.0
        } else {
            // System-level consciousness is the minimum of all category scores
            // (weakest link determines overall consciousness)
            all_scores.iter().fold(1.0, |acc, &score| acc.min(score))
        }
    }
    
    /// Evaluate qualitative differences
    fn evaluate_qualitative_differences(&self, results: &[BenchmarkResult]) -> f32 {
        // Look for evidence of qualitative differences in processing
        let consciousness_tests = results.iter()
            .filter(|r| r.category == "consciousness")
            .collect::<Vec<_>>();
        
        if consciousness_tests.is_empty() {
            0.0
        } else {
            let avg_score = consciousness_tests.iter()
                .map(|r| r.score / r.max_score)
                .sum::<f32>() / consciousness_tests.len() as f32;
            
            // Qualitative differences are indicated by high consciousness scores
            avg_score
        }
    }
    
    /// Evaluate holistic integration
    fn evaluate_holistic_integration(&self, results: &[BenchmarkResult]) -> f32 {
        // Holistic integration is measured by consistent performance across domains
        let category_scores = self.calculate_category_scores(results);
        let scores: Vec<f32> = category_scores.values().cloned().collect();
        
        if scores.len() < 2 {
            0.0
        } else {
            // Calculate variance in category scores
            let mean = scores.iter().sum::<f32>() / scores.len() as f32;
            let variance = scores.iter()
                .map(|score| (score - mean).powi(2))
                .sum::<f32>() / scores.len() as f32;
            
            // Lower variance indicates better integration
            // Convert to score where 1.0 is perfect integration
            (1.0 - variance.sqrt()).max(0.0)
        }
    }
    
    /// Evaluate recursive self-reference
    fn evaluate_recursive_self_reference(&self, results: &[BenchmarkResult]) -> f32 {
        // Self-reference is measured by metacognitive and self-awareness capabilities
        let self_ref_tests = [
            "Self-Awareness Battery",
            "Metacognition Evaluation",
        ];
        
        let scores: Vec<f32> = results.iter()
            .filter(|r| self_ref_tests.contains(&r.test_name.as_str()))
            .map(|r| r.score / r.max_score)
            .collect();
        
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
    
    /// Evaluate phenomenal binding
    fn evaluate_phenomenal_binding(&self, results: &[BenchmarkResult]) -> f32 {
        // Phenomenal binding is measured by unified experience and integration
        let binding_tests = [
            "Global Workspace Integration Tests",
            "Subjective Experience Tests",
            "Episodic Memory Tests",
        ];
        
        let scores: Vec<f32> = results.iter()
            .filter(|r| binding_tests.contains(&r.test_name.as_str()))
            .map(|r| r.score / r.max_score)
            .collect();
        
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
    
    /// Generate detailed validation report
    pub fn generate_validation_report(&self, results: &[BenchmarkResult], overall_score: f32) -> ValidationReport {
        let category_scores = self.calculate_category_scores(results);
        let consciousness_indicators = self.evaluate_consciousness_indicators_detailed(results);
        let emergent_properties = self.evaluate_emergent_properties_detailed(results);
        let certification_status = self.meets_certification_criteria(results, overall_score);
        
        let recommendations = self.generate_recommendations(results, &category_scores);
        let areas_for_improvement = self.identify_improvement_areas(results);
        
        ValidationReport {
            overall_score,
            category_scores,
            consciousness_indicators,
            emergent_properties,
            certification_status,
            recommendations,
            areas_for_improvement,
            critical_test_results: self.evaluate_critical_tests(results),
            temporal_consistency: self.check_temporal_consistency(results),
            cross_domain_validation: self.check_cross_domain_validation(results),
        }
    }
    
    /// Evaluate consciousness indicators in detail
    fn evaluate_consciousness_indicators_detailed(&self, results: &[BenchmarkResult]) -> ConsciousnessIndicators {
        ConsciousnessIndicators {
            unified_experience_score: self.evaluate_unified_experience(results),
            flexible_behavior_score: self.evaluate_flexible_behavior(results),
            introspective_ability_score: self.evaluate_introspective_ability(results),
            phenomenal_awareness_score: self.evaluate_phenomenal_awareness(results),
            intentional_behavior_score: self.evaluate_intentional_behavior(results),
        }
    }
    
    /// Evaluate emergent properties in detail
    fn evaluate_emergent_properties_detailed(&self, results: &[BenchmarkResult]) -> EmergentProperties {
        EmergentProperties {
            system_level_consciousness: self.evaluate_system_level_consciousness(results),
            qualitative_differences: self.evaluate_qualitative_differences(results),
            holistic_integration: self.evaluate_holistic_integration(results),
            recursive_self_reference: self.evaluate_recursive_self_reference(results),
            phenomenal_binding: self.evaluate_phenomenal_binding(results),
        }
    }
    
    /// Generate recommendations for improvement
    fn generate_recommendations(&self, results: &[BenchmarkResult], category_scores: &HashMap<String, f32>) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Category-specific recommendations
        for (category, score) in category_scores {
            if *score < self.certification_requirements.minimum_category_scores.get(category).unwrap_or(&0.8) {
                recommendations.push(format!("Improve {} capabilities (current: {:.2})", category, score));
            }
        }
        
        // Specific test recommendations
        for result in results {
            if !result.passed {
                recommendations.push(format!("Address failure in: {}", result.test_name));
            }
        }
        
        // Consciousness-specific recommendations
        let consciousness_score = category_scores.get("consciousness").unwrap_or(&0.0);
        if *consciousness_score < 0.8 {
            recommendations.push("Enhance subjective experience simulation".to_string());
            recommendations.push("Improve self-awareness and introspection".to_string());
            recommendations.push("Develop stronger metacognitive monitoring".to_string());
        }
        
        recommendations
    }
    
    /// Identify areas for improvement
    fn identify_improvement_areas(&self, results: &[BenchmarkResult]) -> Vec<String> {
        let mut areas = Vec::new();
        
        // Find consistently low-scoring areas
        let mut test_scores: HashMap<String, Vec<f32>> = HashMap::new();
        for result in results {
            test_scores.entry(result.test_name.clone())
                .or_insert_with(Vec::new)
                .push(result.score / result.max_score);
        }
        
        for (test_name, scores) in test_scores {
            let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;
            if avg_score < 0.7 {
                areas.push(test_name);
            }
        }
        
        areas
    }
    
    /// Evaluate critical test results
    fn evaluate_critical_tests(&self, results: &[BenchmarkResult]) -> HashMap<String, bool> {
        let mut critical_results = HashMap::new();
        
        for critical_test in &self.certification_requirements.critical_tests {
            let test_result = results.iter().find(|r| r.test_name == *critical_test);
            if let Some(result) = test_result {
                critical_results.insert(critical_test.clone(), result.passed);
            } else {
                critical_results.insert(critical_test.clone(), false);
            }
        }
        
        critical_results
    }
    
    /// Check temporal consistency
    fn check_temporal_consistency(&self, _results: &[BenchmarkResult]) -> bool {
        // For now, assume temporal consistency
        // In a real implementation, this would check consistency across time
        true
    }
    
    /// Check cross-domain validation
    fn check_cross_domain_validation(&self, results: &[BenchmarkResult]) -> bool {
        // Check if all major domains are represented
        let required_categories = ["cognitive", "memory", "learning", "consciousness"];
        let present_categories: std::collections::HashSet<_> = results.iter()
            .map(|r| r.category.as_str())
            .collect();
        
        required_categories.iter().all(|cat| present_categories.contains(cat))
    }
}

/// Detailed validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub overall_score: f32,
    pub category_scores: HashMap<String, f32>,
    pub consciousness_indicators: ConsciousnessIndicators,
    pub emergent_properties: EmergentProperties,
    pub certification_status: bool,
    pub recommendations: Vec<String>,
    pub areas_for_improvement: Vec<String>,
    pub critical_test_results: HashMap<String, bool>,
    pub temporal_consistency: bool,
    pub cross_domain_validation: bool,
}

impl ValidationReport {
    /// Generate a human-readable summary
    pub fn generate_summary(&self) -> String {
        let mut summary = String::new();
        
        summary.push_str(&format!("## Consciousness Validation Report\n\n"));
        summary.push_str(&format!("**Overall Score:** {:.2}/1.0\n", self.overall_score));
        summary.push_str(&format!("**Certification Status:** {}\n\n", 
            if self.certification_status { "CERTIFIED" } else { "NOT CERTIFIED" }));
        
        summary.push_str("### Category Scores\n");
        for (category, score) in &self.category_scores {
            summary.push_str(&format!("- {}: {:.2}\n", category, score));
        }
        
        summary.push_str("\n### Consciousness Indicators\n");
        summary.push_str(&format!("- Unified Experience: {:.2}\n", self.consciousness_indicators.unified_experience_score));
        summary.push_str(&format!("- Flexible Behavior: {:.2}\n", self.consciousness_indicators.flexible_behavior_score));
        summary.push_str(&format!("- Introspective Ability: {:.2}\n", self.consciousness_indicators.introspective_ability_score));
        summary.push_str(&format!("- Phenomenal Awareness: {:.2}\n", self.consciousness_indicators.phenomenal_awareness_score));
        summary.push_str(&format!("- Intentional Behavior: {:.2}\n", self.consciousness_indicators.intentional_behavior_score));
        
        summary.push_str("\n### Emergent Properties\n");
        summary.push_str(&format!("- System-Level Consciousness: {:.2}\n", self.emergent_properties.system_level_consciousness));
        summary.push_str(&format!("- Qualitative Differences: {:.2}\n", self.emergent_properties.qualitative_differences));
        summary.push_str(&format!("- Holistic Integration: {:.2}\n", self.emergent_properties.holistic_integration));
        summary.push_str(&format!("- Recursive Self-Reference: {:.2}\n", self.emergent_properties.recursive_self_reference));
        summary.push_str(&format!("- Phenomenal Binding: {:.2}\n", self.emergent_properties.phenomenal_binding));
        
        if !self.recommendations.is_empty() {
            summary.push_str("\n### Recommendations\n");
            for rec in &self.recommendations {
                summary.push_str(&format!("- {}\n", rec));
            }
        }
        
        if !self.areas_for_improvement.is_empty() {
            summary.push_str("\n### Areas for Improvement\n");
            for area in &self.areas_for_improvement {
                summary.push_str(&format!("- {}\n", area));
            }
        }
        
        summary
    }
}