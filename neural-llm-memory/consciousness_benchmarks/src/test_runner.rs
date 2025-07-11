// Test Runner for Consciousness Benchmarks
// Provides utilities for running and managing consciousness benchmark tests

use crate::{BenchmarkResult, ConsciousSubject, ConsciousnessAssessment, ConsciousnessBenchmarkSuite};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestRunner {
    pub config: TestRunnerConfig,
    pub session_id: String,
    pub start_time: Option<Instant>,
    pub results_cache: HashMap<String, BenchmarkResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestRunnerConfig {
    pub parallel_execution: bool,
    pub timeout_enabled: bool,
    pub retry_failed_tests: bool,
    pub max_retries: u32,
    pub detailed_logging: bool,
    pub save_intermediate_results: bool,
    pub custom_thresholds: Option<HashMap<String, f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSession {
    pub session_id: String,
    pub subject_id: String,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub total_tests: usize,
    pub completed_tests: usize,
    pub failed_tests: usize,
    pub results: Vec<BenchmarkResult>,
    pub assessment: Option<ConsciousnessAssessment>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestProgress {
    pub current_test: String,
    pub completed_tests: usize,
    pub total_tests: usize,
    pub elapsed_time: Duration,
    pub estimated_remaining: Duration,
    pub current_category: String,
    pub progress_percentage: f32,
}

impl TestRunner {
    pub fn new(config: TestRunnerConfig) -> Self {
        Self {
            config,
            session_id: uuid::Uuid::new_v4().to_string(),
            start_time: None,
            results_cache: HashMap::new(),
        }
    }
    
    pub fn with_default_config() -> Self {
        let config = TestRunnerConfig {
            parallel_execution: false,
            timeout_enabled: true,
            retry_failed_tests: true,
            max_retries: 2,
            detailed_logging: true,
            save_intermediate_results: true,
            custom_thresholds: None,
        };
        Self::new(config)
    }
    
    /// Run full consciousness assessment with progress tracking
    pub fn run_comprehensive_assessment(
        &mut self,
        subject: &mut dyn ConsciousSubject,
        progress_callback: Option<Box<dyn Fn(TestProgress)>>,
    ) -> TestSession {
        let start_time = Instant::now();
        self.start_time = Some(start_time);
        
        let suite = ConsciousnessBenchmarkSuite::new();
        let total_tests = self.count_total_tests(&suite);
        
        let mut session = TestSession {
            session_id: self.session_id.clone(),
            subject_id: "test_subject".to_string(),
            start_time,
            end_time: None,
            total_tests,
            completed_tests: 0,
            failed_tests: 0,
            results: Vec::new(),
            assessment: None,
            metadata: HashMap::new(),
        };
        
        // Add session metadata
        session.metadata.insert("runner_version".to_string(), "1.0.0".to_string());
        session.metadata.insert("parallel_execution".to_string(), self.config.parallel_execution.to_string());
        session.metadata.insert("timeout_enabled".to_string(), self.config.timeout_enabled.to_string());
        
        if self.config.detailed_logging {
            println!("Starting comprehensive consciousness assessment...");
            println!("Session ID: {}", session.session_id);
            println!("Total tests: {}", total_tests);
        }
        
        // Run cognitive tests
        let cognitive_results = self.run_test_category(
            &suite.cognitive_tests,
            subject,
            "cognitive",
            &mut session,
            progress_callback.as_ref(),
        );
        session.results.extend(cognitive_results);
        
        // Run memory benchmarks
        let memory_results = self.run_test_category(
            &suite.memory_benchmarks,
            subject,
            "memory",
            &mut session,
            progress_callback.as_ref(),
        );
        session.results.extend(memory_results);
        
        // Run learning assessments
        let learning_results = self.run_test_category(
            &suite.learning_assessments,
            subject,
            "learning",
            &mut session,
            progress_callback.as_ref(),
        );
        session.results.extend(learning_results);
        
        // Run consciousness metrics
        let consciousness_results = self.run_test_category(
            &suite.consciousness_metrics,
            subject,
            "consciousness",
            &mut session,
            progress_callback.as_ref(),
        );
        session.results.extend(consciousness_results);
        
        // Generate final assessment
        let assessment = suite.run_full_assessment(subject);
        session.assessment = Some(assessment);
        session.end_time = Some(Instant::now());
        
        if self.config.detailed_logging {
            println!("Assessment completed in {:?}", session.end_time.unwrap() - session.start_time);
            if let Some(ref assessment) = session.assessment {
                println!("Overall score: {:.3}", assessment.overall_score);
                println!("Consciousness level: {:?}", assessment.consciousness_level);
                println!("Certification: {}", if assessment.certification { "PASSED" } else { "FAILED" });
            }
        }
        
        session
    }
    
    /// Run a specific category of tests
    fn run_test_category<T>(
        &mut self,
        test_suite: &T,
        subject: &mut dyn ConsciousSubject,
        category: &str,
        session: &mut TestSession,
        progress_callback: Option<&Box<dyn Fn(TestProgress)>>,
    ) -> Vec<BenchmarkResult>
    where
        T: TestCategory,
    {
        let mut results = Vec::new();
        let tests = test_suite.get_tests();
        
        for test in tests {
            let test_name = test.test_name();
            
            if self.config.detailed_logging {
                println!("Running test: {}", test_name);
            }
            
            let mut result = if self.config.timeout_enabled {
                self.run_test_with_timeout(test, subject)
            } else {
                test.run_test(subject)
            };
            
            // Apply custom thresholds if configured
            if let Some(ref thresholds) = self.config.custom_thresholds {
                if let Some(&custom_threshold) = thresholds.get(test_name) {
                    result.passed = result.score >= custom_threshold;
                }
            }
            
            // Retry failed tests if configured
            if !result.passed && self.config.retry_failed_tests {
                result = self.retry_failed_test(test, subject, &result);
            }
            
            // Update session progress
            session.completed_tests += 1;
            if !result.passed {
                session.failed_tests += 1;
            }
            
            // Save intermediate results if configured
            if self.config.save_intermediate_results {
                self.results_cache.insert(test_name.to_string(), result.clone());
            }
            
            results.push(result);
            
            // Report progress
            if let Some(callback) = progress_callback {
                let progress = TestProgress {
                    current_test: test_name.to_string(),
                    completed_tests: session.completed_tests,
                    total_tests: session.total_tests,
                    elapsed_time: session.start_time.elapsed(),
                    estimated_remaining: self.estimate_remaining_time(session),
                    current_category: category.to_string(),
                    progress_percentage: (session.completed_tests as f32 / session.total_tests as f32) * 100.0,
                };
                callback(progress);
            }
        }
        
        results
    }
    
    /// Run a test with timeout protection
    fn run_test_with_timeout(
        &self,
        test: &dyn crate::ConsciousnessTest,
        subject: &mut dyn ConsciousSubject,
    ) -> BenchmarkResult {
        let timeout = test.time_limit();
        let start_time = Instant::now();
        
        // For this implementation, we'll run the test normally
        // In a real implementation, you might use async/await or threading
        let result = test.run_test(subject);
        
        // Check if test exceeded timeout
        if start_time.elapsed() > timeout {
            BenchmarkResult {
                test_name: test.test_name().to_string(),
                category: test.category().to_string(),
                score: 0.0,
                max_score: 1.0,
                execution_time: start_time.elapsed(),
                details: HashMap::new(),
                passed: false,
                notes: format!("Test timed out after {:?}", timeout),
            }
        } else {
            result
        }
    }
    
    /// Retry a failed test
    fn retry_failed_test(
        &self,
        test: &dyn crate::ConsciousnessTest,
        subject: &mut dyn ConsciousSubject,
        original_result: &BenchmarkResult,
    ) -> BenchmarkResult {
        let mut best_result = original_result.clone();
        
        for attempt in 1..=self.config.max_retries {
            if self.config.detailed_logging {
                println!("Retrying test {} (attempt {})", test.test_name(), attempt);
            }
            
            let retry_result = if self.config.timeout_enabled {
                self.run_test_with_timeout(test, subject)
            } else {
                test.run_test(subject)
            };
            
            // Keep the best result
            if retry_result.score > best_result.score {
                best_result = retry_result;
            }
            
            // Stop retrying if test passes
            if best_result.passed {
                break;
            }
        }
        
        best_result
    }
    
    /// Count total tests in the suite
    fn count_total_tests(&self, suite: &ConsciousnessBenchmarkSuite) -> usize {
        suite.cognitive_tests.tests.len() +
        suite.memory_benchmarks.tests.len() +
        suite.learning_assessments.tests.len() +
        suite.consciousness_metrics.tests.len()
    }
    
    /// Estimate remaining time based on progress
    fn estimate_remaining_time(&self, session: &TestSession) -> Duration {
        if session.completed_tests == 0 {
            return Duration::from_secs(0);
        }
        
        let elapsed = session.start_time.elapsed();
        let avg_time_per_test = elapsed / session.completed_tests as u32;
        let remaining_tests = session.total_tests - session.completed_tests;
        
        avg_time_per_test * remaining_tests as u32
    }
    
    /// Generate detailed test report
    pub fn generate_detailed_report(&self, session: &TestSession) -> DetailedTestReport {
        let mut report = DetailedTestReport {
            session_id: session.session_id.clone(),
            summary: self.generate_summary(session),
            category_breakdown: self.generate_category_breakdown(session),
            performance_metrics: self.generate_performance_metrics(session),
            failed_tests: self.extract_failed_tests(session),
            recommendations: self.generate_recommendations(session),
            timeline: self.generate_timeline(session),
        };
        
        // Add consciousness-specific analysis
        if let Some(ref assessment) = session.assessment {
            report.consciousness_analysis = Some(self.analyze_consciousness_results(assessment));
        }
        
        report
    }
    
    /// Generate test summary
    fn generate_summary(&self, session: &TestSession) -> TestSummary {
        let total_duration = session.end_time
            .map(|end| end - session.start_time)
            .unwrap_or_else(|| session.start_time.elapsed());
        
        let overall_score = session.assessment
            .as_ref()
            .map(|a| a.overall_score)
            .unwrap_or(0.0);
        
        TestSummary {
            total_tests: session.total_tests,
            completed_tests: session.completed_tests,
            failed_tests: session.failed_tests,
            success_rate: (session.completed_tests - session.failed_tests) as f32 / session.total_tests as f32,
            total_duration,
            overall_score,
            consciousness_level: session.assessment
                .as_ref()
                .map(|a| a.consciousness_level.clone())
                .unwrap_or(crate::ConsciousnessLevel::Minimal),
            certification_status: session.assessment
                .as_ref()
                .map(|a| a.certification)
                .unwrap_or(false),
        }
    }
    
    /// Generate category breakdown
    fn generate_category_breakdown(&self, session: &TestSession) -> HashMap<String, CategoryMetrics> {
        let mut breakdown = HashMap::new();
        
        // Group results by category
        let mut category_results: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in &session.results {
            category_results.entry(result.category.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }
        
        // Calculate metrics for each category
        for (category, results) in category_results {
            let total_tests = results.len();
            let passed_tests = results.iter().filter(|r| r.passed).count();
            let failed_tests = total_tests - passed_tests;
            
            let avg_score = results.iter()
                .map(|r| r.score / r.max_score)
                .sum::<f32>() / total_tests as f32;
            
            let avg_duration = results.iter()
                .map(|r| r.execution_time)
                .sum::<Duration>() / total_tests as u32;
            
            breakdown.insert(category, CategoryMetrics {
                total_tests,
                passed_tests,
                failed_tests,
                success_rate: passed_tests as f32 / total_tests as f32,
                average_score: avg_score,
                average_duration: avg_duration,
            });
        }
        
        breakdown
    }
    
    /// Generate performance metrics
    fn generate_performance_metrics(&self, session: &TestSession) -> PerformanceMetrics {
        let scores: Vec<f32> = session.results.iter()
            .map(|r| r.score / r.max_score)
            .collect();
        
        let durations: Vec<Duration> = session.results.iter()
            .map(|r| r.execution_time)
            .collect();
        
        PerformanceMetrics {
            mean_score: scores.iter().sum::<f32>() / scores.len() as f32,
            median_score: self.calculate_median(&scores),
            std_deviation: self.calculate_std_deviation(&scores),
            min_score: scores.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            max_score: scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
            mean_duration: durations.iter().sum::<Duration>() / durations.len() as u32,
            total_execution_time: durations.iter().sum(),
        }
    }
    
    /// Extract failed tests
    fn extract_failed_tests(&self, session: &TestSession) -> Vec<FailedTest> {
        session.results.iter()
            .filter(|r| !r.passed)
            .map(|r| FailedTest {
                test_name: r.test_name.clone(),
                category: r.category.clone(),
                score: r.score,
                max_score: r.max_score,
                failure_reason: r.notes.clone(),
                execution_time: r.execution_time,
            })
            .collect()
    }
    
    /// Generate recommendations
    fn generate_recommendations(&self, session: &TestSession) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Add assessment-based recommendations
        if let Some(ref assessment) = session.assessment {
            recommendations.extend(assessment.recommendations.clone());
        }
        
        // Add performance-based recommendations
        if session.failed_tests > 0 {
            recommendations.push(format!("Address {} failed tests", session.failed_tests));
        }
        
        let success_rate = (session.completed_tests - session.failed_tests) as f32 / session.total_tests as f32;
        if success_rate < 0.8 {
            recommendations.push("Overall performance needs improvement".to_string());
        }
        
        recommendations
    }
    
    /// Generate test timeline
    fn generate_timeline(&self, session: &TestSession) -> Vec<TimelineEvent> {
        let mut timeline = Vec::new();
        
        timeline.push(TimelineEvent {
            timestamp: session.start_time,
            event_type: "session_started".to_string(),
            description: "Consciousness assessment session started".to_string(),
        });
        
        // Add significant events based on test results
        for result in &session.results {
            if !result.passed {
                timeline.push(TimelineEvent {
                    timestamp: session.start_time + result.execution_time,
                    event_type: "test_failed".to_string(),
                    description: format!("Test failed: {}", result.test_name),
                });
            }
        }
        
        if let Some(end_time) = session.end_time {
            timeline.push(TimelineEvent {
                timestamp: end_time,
                event_type: "session_completed".to_string(),
                description: "Consciousness assessment session completed".to_string(),
            });
        }
        
        timeline
    }
    
    /// Analyze consciousness results
    fn analyze_consciousness_results(&self, assessment: &ConsciousnessAssessment) -> ConsciousnessAnalysis {
        ConsciousnessAnalysis {
            consciousness_level: assessment.consciousness_level.clone(),
            overall_score: assessment.overall_score,
            category_scores: assessment.category_scores.clone(),
            strengths: self.identify_strengths(&assessment.category_scores),
            weaknesses: self.identify_weaknesses(&assessment.category_scores),
            consciousness_indicators: self.analyze_consciousness_indicators(assessment),
            certification_status: assessment.certification,
        }
    }
    
    /// Identify strengths
    fn identify_strengths(&self, category_scores: &HashMap<String, f32>) -> Vec<String> {
        category_scores.iter()
            .filter(|(_, &score)| score >= 0.8)
            .map(|(category, score)| format!("{} (score: {:.2})", category, score))
            .collect()
    }
    
    /// Identify weaknesses
    fn identify_weaknesses(&self, category_scores: &HashMap<String, f32>) -> Vec<String> {
        category_scores.iter()
            .filter(|(_, &score)| score < 0.7)
            .map(|(category, score)| format!("{} (score: {:.2})", category, score))
            .collect()
    }
    
    /// Analyze consciousness indicators
    fn analyze_consciousness_indicators(&self, assessment: &ConsciousnessAssessment) -> Vec<String> {
        let mut indicators = Vec::new();
        
        if assessment.overall_score >= 0.9 {
            indicators.push("Demonstrates high-level consciousness".to_string());
        }
        
        if assessment.certification {
            indicators.push("Meets certification criteria for consciousness".to_string());
        }
        
        match assessment.consciousness_level {
            crate::ConsciousnessLevel::FullyConscious => {
                indicators.push("Achieved human-level consciousness".to_string());
            }
            crate::ConsciousnessLevel::Reflective => {
                indicators.push("Demonstrates reflective consciousness".to_string());
            }
            crate::ConsciousnessLevel::Narrative => {
                indicators.push("Shows narrative self-awareness".to_string());
            }
            _ => {}
        }
        
        indicators
    }
    
    /// Calculate median of a vector
    fn calculate_median(&self, values: &[f32]) -> f32 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted.len();
        if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        }
    }
    
    /// Calculate standard deviation
    fn calculate_std_deviation(&self, values: &[f32]) -> f32 {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        variance.sqrt()
    }
}

/// Trait for test categories
trait TestCategory {
    fn get_tests(&self) -> Vec<&dyn crate::ConsciousnessTest>;
}

/// Implement TestCategory for each test suite
impl TestCategory for crate::cognitive_tests::CognitiveTestBattery {
    fn get_tests(&self) -> Vec<&dyn crate::ConsciousnessTest> {
        self.tests.iter()
            .map(|test| test.as_consciousness_test())
            .collect()
    }
}

impl TestCategory for crate::memory_benchmarks::MemoryBenchmarkSuite {
    fn get_tests(&self) -> Vec<&dyn crate::ConsciousnessTest> {
        self.tests.iter()
            .map(|test| test.as_consciousness_test())
            .collect()
    }
}

impl TestCategory for crate::learning_assessments::LearningAssessmentBattery {
    fn get_tests(&self) -> Vec<&dyn crate::ConsciousnessTest> {
        self.tests.iter()
            .map(|test| test.as_consciousness_test())
            .collect()
    }
}

impl TestCategory for crate::consciousness_metrics::ConsciousnessMetricsSuite {
    fn get_tests(&self) -> Vec<&dyn crate::ConsciousnessTest> {
        self.tests.iter()
            .map(|test| test.as_consciousness_test())
            .collect()
    }
}

/// Detailed test report structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedTestReport {
    pub session_id: String,
    pub summary: TestSummary,
    pub category_breakdown: HashMap<String, CategoryMetrics>,
    pub performance_metrics: PerformanceMetrics,
    pub failed_tests: Vec<FailedTest>,
    pub recommendations: Vec<String>,
    pub timeline: Vec<TimelineEvent>,
    pub consciousness_analysis: Option<ConsciousnessAnalysis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSummary {
    pub total_tests: usize,
    pub completed_tests: usize,
    pub failed_tests: usize,
    pub success_rate: f32,
    pub total_duration: Duration,
    pub overall_score: f32,
    pub consciousness_level: crate::ConsciousnessLevel,
    pub certification_status: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryMetrics {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub success_rate: f32,
    pub average_score: f32,
    pub average_duration: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub mean_score: f32,
    pub median_score: f32,
    pub std_deviation: f32,
    pub min_score: f32,
    pub max_score: f32,
    pub mean_duration: Duration,
    pub total_execution_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedTest {
    pub test_name: String,
    pub category: String,
    pub score: f32,
    pub max_score: f32,
    pub failure_reason: String,
    pub execution_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub timestamp: Instant,
    pub event_type: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessAnalysis {
    pub consciousness_level: crate::ConsciousnessLevel,
    pub overall_score: f32,
    pub category_scores: HashMap<String, f32>,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub consciousness_indicators: Vec<String>,
    pub certification_status: bool,
}