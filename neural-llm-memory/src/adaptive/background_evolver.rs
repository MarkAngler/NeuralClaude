use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::task::JoinHandle;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use crate::self_optimizing::{SelfOptimizingNetwork, EvolutionConfig, SelfOptimizingConfig};
use crate::memory::{MemoryConfig};
use crate::adaptive::{TrainingCorpus, AdaptiveConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionStatus {
    pub is_running: bool,
    pub current_generation: usize,
    pub best_fitness: f32,
    pub started_at: Option<DateTime<Utc>>,
    pub progress_percent: f32,
    pub estimated_completion: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct EvolvedArchitecture {
    pub generation: usize,
    pub fitness_score: f32,
    pub memory_config: MemoryConfig,
    pub architecture_summary: String,
    pub improvements: Vec<String>,
}

pub struct BackgroundEvolver {
    // Evolution happens in separate thread
    evolution_handle: Option<JoinHandle<()>>,
    
    // Channel for sending new architectures
    architecture_tx: mpsc::Sender<EvolvedArchitecture>,
    architecture_rx: Arc<Mutex<mpsc::Receiver<EvolvedArchitecture>>>,
    
    // Current evolution status
    pub(crate) status: Arc<RwLock<EvolutionStatus>>,
    
    // Last evolution time
    last_evolution: Arc<RwLock<Option<DateTime<Utc>>>>,
}

impl BackgroundEvolver {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(10);
        
        Self {
            evolution_handle: None,
            architecture_tx: tx,
            architecture_rx: Arc::new(Mutex::new(rx)),
            status: Arc::new(RwLock::new(EvolutionStatus {
                is_running: false,
                current_generation: 0,
                best_fitness: 0.0,
                started_at: None,
                progress_percent: 0.0,
                estimated_completion: None,
            })),
            last_evolution: Arc::new(RwLock::new(None)),
        }
    }
    
    pub async fn start_evolution(
        &mut self,
        config: AdaptiveConfig,
        training_data: TrainingCorpus,
        current_config: MemoryConfig,
    ) -> Result<(), String> {
        // Check if already running
        if self.status.read().await.is_running {
            return Err("Evolution already in progress".to_string());
        }
        
        // Update status
        {
            let mut status = self.status.write().await;
            status.is_running = true;
            status.current_generation = 0;
            status.started_at = Some(Utc::now());
            status.progress_percent = 0.0;
        }
        
        // Clone necessary data for the evolution thread
        let status_clone = Arc::clone(&self.status);
        let tx = self.architecture_tx.clone();
        let evolution_config = self.build_evolution_config(&config);
        
        // Spawn background evolution task
        let handle = tokio::spawn(async move {
            if let Err(e) = Self::run_evolution(
                config,
                training_data,
                current_config,
                evolution_config,
                status_clone.clone(),
                tx,
            ).await {
                eprintln!("Evolution error: {}", e);
                let mut status = status_clone.write().await;
                status.is_running = false;
            }
        });
        
        self.evolution_handle = Some(handle);
        *self.last_evolution.write().await = Some(Utc::now());
        
        Ok(())
    }
    
    async fn run_evolution(
        config: AdaptiveConfig,
        training_data: TrainingCorpus,
        current_config: MemoryConfig,
        _evolution_config: EvolutionConfig,
        status: Arc<RwLock<EvolutionStatus>>,
        tx: mpsc::Sender<EvolvedArchitecture>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Create optimizer config from current memory config
        let opt_config = SelfOptimizingConfig {
            enabled: true,
            max_layers: 5,
            min_layers: 2,
            layer_size_range: (128, 1024),
            population_size: config.population_size,
            mutation_rate: config.mutation_rate,
            crossover_rate: config.crossover_rate,
            elite_size: config.population_size / 10,
            objectives: config.objectives.clone(),
            max_memory_mb: 512,
            target_inference_ms: 20.0,
            fitness_threshold: 0.8,
            adaptation_interval: 100,
        };
        let mut optimizer = SelfOptimizingNetwork::new(opt_config);
        
        // Convert usage metrics to training samples
        let training_samples = Self::prepare_training_data(&training_data);
        
        // Set correct input/output sizes based on training data
        if !training_samples.is_empty() {
            let (input, target) = &training_samples[0];
            optimizer.set_input_size(input.len());
            optimizer.set_output_size(target.len());
        } else {
            // Default sizes if no training data
            optimizer.set_input_size(3);  // Matches our dummy data
            optimizer.set_output_size(1); // Matches our dummy data
        }
        
        // Run evolution
        for generation in 0..config.generations {
            // Update progress
            {
                let mut status = status.write().await;
                status.current_generation = generation;
                status.progress_percent = (generation as f32 / config.generations as f32) * 100.0;
                
                // Estimate completion time
                if generation > 0 {
                    let elapsed = Utc::now().signed_duration_since(status.started_at.unwrap());
                    let per_generation = elapsed / generation as i32;
                    let remaining = per_generation * (config.generations - generation) as i32;
                    status.estimated_completion = Some(Utc::now() + remaining);
                }
            }
            
            // Prepare validation data for evolution
            let validation_data: Vec<(ndarray::Array2<f32>, ndarray::Array2<f32>)> = if !training_samples.is_empty() {
                training_samples.iter()
                    .take(100) // Use more samples for better evaluation
                    .map(|(input, target)| {
                        (
                            ndarray::Array2::from_shape_vec((1, input.len()), input.clone()).unwrap(),
                            ndarray::Array2::from_shape_vec((1, target.len()), target.clone()).unwrap()
                        )
                    })
                    .collect()
            } else {
                // Create dummy validation data if no training samples
                vec![(
                    ndarray::Array2::from_shape_vec((1, 3), vec![0.0; 3]).unwrap(),
                    ndarray::Array2::from_shape_vec((1, 1), vec![0.0]).unwrap()
                )]
            };
            
            // Actually evolve the architecture!
            if let Err(e) = optimizer.evolve_architecture(&validation_data) {
                eprintln!("Evolution step failed: {}", e);
                // Continue with next generation even if this one fails
            }
            
            // Get insights
            let insights = optimizer.get_insights();
            {
                let mut status = status.write().await;
                status.best_fitness = insights.generation_stats.best_fitness;
            }
            
            // Check for significant improvement
            if generation > 0 && generation % 5 == 0 {
                let best_arch = optimizer.get_best_architecture();
                let evolved_config = best_arch.to_memory_config();
                
                let evolved = EvolvedArchitecture {
                    generation,
                    fitness_score: insights.generation_stats.best_fitness,
                    memory_config: evolved_config.clone(),
                    architecture_summary: best_arch.get_summary(),
                    improvements: extract_architecture_improvements(&evolved_config, &current_config),
                };
                
                // Send intermediate result
                let _ = tx.send(evolved).await;
            }
        }
        
        // Get final best architecture
        let best_arch = optimizer.get_best_architecture();
        let evolved_config = best_arch.to_memory_config();
        let insights = optimizer.get_insights();
        
        let evolved = EvolvedArchitecture {
            generation: config.generations,
            fitness_score: insights.generation_stats.best_fitness,
            memory_config: evolved_config.clone(),
            architecture_summary: best_arch.get_summary(),
            improvements: extract_architecture_improvements(&evolved_config, &current_config),
        };
        
        // Send final result
        let _ = tx.send(evolved).await;
        
        // Update status
        {
            let mut status = status.write().await;
            status.is_running = false;
            status.progress_percent = 100.0;
        }
        
        Ok(())
    }
    
    fn build_evolution_config(&self, config: &AdaptiveConfig) -> EvolutionConfig {
        EvolutionConfig {
            population_size: config.population_size,
            mutation_rate: config.mutation_rate,
            crossover_rate: config.crossover_rate,
            elite_size: config.population_size / 10,
            tournament_size: 3,
        }
    }
    
    fn prepare_training_data(corpus: &TrainingCorpus) -> Vec<(Vec<f32>, Vec<f32>)> {
        // Convert usage metrics to training samples
        // This is a simplified version - in production, you'd want more sophisticated conversion
        corpus.metrics.iter()
            .filter_map(|metric| {
                if metric.similarity_scores.is_empty() {
                    return None;
                }
                
                // Create input features
                let input = vec![
                    metric.input_size as f32 / 1000.0,  // Normalize input size
                    metric.response_time_ms / 100.0,     // Normalize response time
                    if metric.cache_hit { 1.0 } else { 0.0 },
                ];
                
                // Use similarity scores as output, or default to single value
                let output = if !metric.similarity_scores.is_empty() {
                    metric.similarity_scores.clone()
                } else {
                    // Default output based on cache hit
                    vec![if metric.cache_hit { 1.0 } else { 0.0 }]
                };
                Some((input, output))
            })
            .collect()
    }
    
    pub async fn get_status(&self) -> EvolutionStatus {
        self.status.read().await.clone()
    }
    
    pub async fn try_receive_architecture(&self) -> Option<EvolvedArchitecture> {
        let mut rx = self.architecture_rx.lock().await;
        rx.try_recv().ok()
    }
    
    pub async fn stop_evolution(&mut self) {
        if let Some(handle) = self.evolution_handle.take() {
            handle.abort();
            let mut status = self.status.write().await;
            status.is_running = false;
        }
    }
    
    pub async fn get_last_evolution_time(&self) -> Option<DateTime<Utc>> {
        *self.last_evolution.read().await
    }
}

/// Extract improvements between evolved and original config
fn extract_architecture_improvements(
    evolved_config: &MemoryConfig, 
    original_config: &MemoryConfig
) -> Vec<String> {
    let mut improvements = vec![];
    
    if evolved_config.embedding_dim != original_config.embedding_dim {
        improvements.push(format!("Embedding: {} → {}", 
            original_config.embedding_dim, evolved_config.embedding_dim));
    }
    
    if evolved_config.hidden_dim != original_config.hidden_dim {
        improvements.push(format!("Hidden: {} → {}", 
            original_config.hidden_dim, evolved_config.hidden_dim));
    }
    
    if evolved_config.num_layers != original_config.num_layers {
        improvements.push(format!("Layers: {} → {}", 
            original_config.num_layers, evolved_config.num_layers));
    }
    
    if evolved_config.num_heads != original_config.num_heads {
        improvements.push(format!("Heads: {} → {}", 
            original_config.num_heads, evolved_config.num_heads));
    }
    
    if (evolved_config.dropout_rate - original_config.dropout_rate).abs() > 0.01 {
        improvements.push(format!("Dropout: {:.2} → {:.2}", 
            original_config.dropout_rate, evolved_config.dropout_rate));
    }
    
    improvements
}