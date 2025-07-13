#[cfg(test)]
mod genome_to_memory_config_tests {
    use crate::self_optimizing::genome::{
        ArchitectureGenome, LayerGene, LayerType, ConnectionGene, ConnectionType,
        HyperparameterSet, OptimizerType, HardwareMetrics
    };
    use crate::self_optimizing::SelfOptimizingConfig;
    use crate::nn::ActivationFunction;
    use crate::memory::MemoryConfig;
    
    #[test]
    fn test_genome_to_memory_config_basic() {
        let config = SelfOptimizingConfig::default();
        let genome = ArchitectureGenome::random(&config);
        
        let memory_config = genome.to_memory_config();
        
        // Check basic constraints
        assert!(memory_config.embedding_dim >= 32);
        assert!(memory_config.embedding_dim <= 4096);
        assert!(memory_config.num_heads > 0);
        assert!(memory_config.num_layers >= 2);
        assert!(memory_config.dropout_rate >= 0.0);
        assert!(memory_config.dropout_rate <= 0.5);
        
        // Check that num_heads divides embedding_dim evenly
        assert_eq!(memory_config.embedding_dim % memory_config.num_heads, 0);
    }
    
    #[test]
    fn test_genome_with_attention_layers() {
        let mut genome = ArchitectureGenome {
            id: uuid::Uuid::new_v4(),
            layers: vec![
                LayerGene {
                    id: 0,
                    layer_type: LayerType::Linear,
                    size: 768,
                    activation: ActivationFunction::GELU,
                    dropout_rate: Some(0.1),
                    use_batch_norm: false,
                    use_layer_norm: true,
                },
                LayerGene {
                    id: 1,
                    layer_type: LayerType::Attention { num_heads: 12 },
                    size: 768,
                    activation: ActivationFunction::Identity,
                    dropout_rate: Some(0.2),
                    use_batch_norm: false,
                    use_layer_norm: true,
                },
                LayerGene {
                    id: 2,
                    layer_type: LayerType::Linear,
                    size: 2048,
                    activation: ActivationFunction::ReLU,
                    dropout_rate: None,
                    use_batch_norm: false,
                    use_layer_norm: false,
                },
            ],
            connections: vec![
                ConnectionGene {
                    from_layer: 0,
                    to_layer: 1,
                    connection_type: ConnectionType::Sequential,
                    enabled: true,
                    weight_scale: 1.0,
                },
                ConnectionGene {
                    from_layer: 1,
                    to_layer: 2,
                    connection_type: ConnectionType::Sequential,
                    enabled: true,
                    weight_scale: 1.0,
                },
            ],
            hyperparameters: HyperparameterSet {
                learning_rate: 0.001,
                batch_size: 32,
                weight_decay: 0.0001,
                gradient_clip: Some(1.0),
                warmup_steps: 100,
                optimizer_type: OptimizerType::Adam { 
                    beta1: 0.9, 
                    beta2: 0.999, 
                    epsilon: 1e-8 
                },
            },
            fitness_scores: std::collections::HashMap::new(),
            hardware_metrics: HardwareMetrics {
                inference_time_ms: 5.0,
                memory_usage_mb: 100.0,
                flops: 1000000,
                energy_consumption: None,
            },
            created_at: chrono::Utc::now(),
            num_parameters: 0,
            mutation_history: vec![],
        };
        
        let memory_config = genome.to_memory_config();
        
        assert_eq!(memory_config.embedding_dim, 768);
        assert_eq!(memory_config.hidden_dim, 768); // From attention layer
        assert_eq!(memory_config.num_heads, 12); // From attention layer
        assert_eq!(memory_config.num_layers, 3);
        assert_eq!(memory_config.dropout_rate, 0.15); // Average of 0.1, 0.2, None
    }
    
    #[test]
    fn test_num_heads_adjustment() {
        // Test the adjust_num_heads function indirectly
        let mut genome = ArchitectureGenome {
            id: uuid::Uuid::new_v4(),
            layers: vec![
                LayerGene {
                    id: 0,
                    layer_type: LayerType::Linear,
                    size: 100, // Not divisible by 12
                    activation: ActivationFunction::ReLU,
                    dropout_rate: None,
                    use_batch_norm: false,
                    use_layer_norm: false,
                },
            ],
            connections: vec![],
            hyperparameters: HyperparameterSet {
                learning_rate: 0.001,
                batch_size: 32,
                weight_decay: 0.0,
                gradient_clip: None,
                warmup_steps: 0,
                optimizer_type: OptimizerType::SGD { momentum: 0.9 },
            },
            fitness_scores: std::collections::HashMap::new(),
            hardware_metrics: HardwareMetrics {
                inference_time_ms: 0.0,
                memory_usage_mb: 0.0,
                flops: 0,
                energy_consumption: None,
            },
            created_at: chrono::Utc::now(),
            num_parameters: 0,
            mutation_history: vec![],
        };
        
        let memory_config = genome.to_memory_config();
        
        // 100 is divisible by 1, 2, 4, 5, 10, 20, 25, 50, 100
        // Default for size 100 would be 4 heads
        assert_eq!(memory_config.embedding_dim, 100);
        assert!(memory_config.num_heads <= 4);
        assert_eq!(100 % memory_config.num_heads, 0);
    }
    
    #[test]
    fn test_evolution_to_memory_config_improvements() {
        // Simulate an evolution by creating two genomes
        let original_config = MemoryConfig {
            memory_size: 10000,
            embedding_dim: 768,
            hidden_dim: 2048,
            num_heads: 12,
            num_layers: 6,
            dropout_rate: 0.1,
            max_sequence_length: 512,
            use_positional_encoding: true,
        };
        
        // Create an evolved genome with different architecture
        let evolved_genome = ArchitectureGenome {
            id: uuid::Uuid::new_v4(),
            layers: vec![
                LayerGene {
                    id: 0,
                    layer_type: LayerType::Linear,
                    size: 512, // Smaller embedding
                    activation: ActivationFunction::GELU,
                    dropout_rate: Some(0.2),
                    use_batch_norm: false,
                    use_layer_norm: true,
                },
                LayerGene {
                    id: 1,
                    layer_type: LayerType::Linear,
                    size: 1024, // Smaller hidden
                    activation: ActivationFunction::ReLU,
                    dropout_rate: Some(0.15),
                    use_batch_norm: true,
                    use_layer_norm: false,
                },
                LayerGene {
                    id: 2,
                    layer_type: LayerType::Attention { num_heads: 8 },
                    size: 512,
                    activation: ActivationFunction::Identity,
                    dropout_rate: Some(0.1),
                    use_batch_norm: false,
                    use_layer_norm: true,
                },
                LayerGene {
                    id: 3,
                    layer_type: LayerType::Linear,
                    size: 256,
                    activation: ActivationFunction::Tanh,
                    dropout_rate: None,
                    use_batch_norm: false,
                    use_layer_norm: false,
                },
            ],
            connections: vec![],
            hyperparameters: HyperparameterSet {
                learning_rate: 0.0005,
                batch_size: 64,
                weight_decay: 0.001,
                gradient_clip: Some(0.5),
                warmup_steps: 500,
                optimizer_type: OptimizerType::AdamW { 
                    beta1: 0.9, 
                    beta2: 0.999, 
                    epsilon: 1e-8 
                },
            },
            fitness_scores: std::collections::HashMap::from([
                ("accuracy".to_string(), 0.95),
                ("speed".to_string(), 0.90),
                ("memory".to_string(), 0.85),
            ]),
            hardware_metrics: HardwareMetrics {
                inference_time_ms: 3.5,
                memory_usage_mb: 75.0,
                flops: 500000,
                energy_consumption: Some(0.1),
            },
            created_at: chrono::Utc::now(),
            num_parameters: 1000000,
            mutation_history: vec![],
        };
        
        let evolved_config = evolved_genome.to_memory_config();
        
        // Verify the evolution produced different architecture
        assert_ne!(evolved_config.embedding_dim, original_config.embedding_dim);
        assert_ne!(evolved_config.hidden_dim, original_config.hidden_dim);
        assert_ne!(evolved_config.num_layers, original_config.num_layers);
        assert_ne!(evolved_config.num_heads, original_config.num_heads);
        
        // Verify evolved config is valid
        assert_eq!(evolved_config.embedding_dim, 512);
        assert_eq!(evolved_config.hidden_dim, 1024);
        assert_eq!(evolved_config.num_layers, 4);
        assert_eq!(evolved_config.num_heads, 8);
        // Average of 0.2, 0.15, 0.1 (None excluded) = 0.45/3 = 0.15
        assert_eq!(evolved_config.dropout_rate, 0.15);
    }
}