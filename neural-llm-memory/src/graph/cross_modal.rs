//! Cross-Modal Bridge for connecting different memory types
//! 
//! This module implements a bridge system that enables translation and connections
//! between different memory modalities (semantic, episodic, emotional, procedural, etc.)

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use ndarray::{Array2, Array1};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use chrono::{DateTime, Utc};

use crate::graph::core::{NodeId, ConsciousNode, ConsciousEdge, EdgeType};

/// Memory modality types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryModality {
    Semantic,      // Facts and concepts
    Episodic,      // Personal experiences
    Emotional,     // Emotional associations
    Procedural,    // How-to knowledge
    Contextual,    // Environmental context
    Temporal,      // Time-based memories
    Causal,        // Cause-effect relationships
    Abstract,      // Abstract concepts
}

/// Cross-modal bridge strength
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStrength {
    pub base_strength: f32,
    pub learned_strength: f32,
    pub usage_count: u64,
    pub last_used: DateTime<Utc>,
}

/// Translation matrix between modalities
#[derive(Debug, Clone)]
pub struct ModalityTranslator {
    /// Translation matrix: source features -> target features
    pub translation_matrix: Array2<f32>,
    /// Bias vector for the translation
    pub bias: Array1<f32>,
    /// Learning rate for updates
    pub learning_rate: f32,
}

/// Cross-modal connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalConnection {
    pub source_modality: MemoryModality,
    pub target_modality: MemoryModality,
    pub strength: BridgeStrength,
    pub example_pairs: Vec<(NodeId, NodeId)>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Cross-modal bridge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalConfig {
    pub enable_all_connections: bool,
    pub min_bridge_strength: f32,
    pub max_example_pairs: usize,
    pub learning_rate: f32,
    pub feature_dimension: usize,
}

impl Default for CrossModalConfig {
    fn default() -> Self {
        Self {
            enable_all_connections: true,
            min_bridge_strength: 0.3,
            max_example_pairs: 100,
            learning_rate: 0.01,
            feature_dimension: 768, // Standard embedding size
        }
    }
}

/// Cross-modal bridge system
pub struct CrossModalBridge {
    /// Configuration
    config: CrossModalConfig,
    
    /// Modality encoders (modality -> encoder)
    encoders: Arc<RwLock<HashMap<MemoryModality, ModalityEncoder>>>,
    
    /// Translation matrices between modalities
    translators: Arc<RwLock<HashMap<(MemoryModality, MemoryModality), ModalityTranslator>>>,
    
    /// Active connections between modalities
    connections: Arc<RwLock<Vec<CrossModalConnection>>>,
    
    /// Node modality mappings
    node_modalities: Arc<RwLock<HashMap<NodeId, MemoryModality>>>,
    
    /// Performance metrics
    metrics: Arc<RwLock<CrossModalMetrics>>,
}

/// Modality-specific encoder
pub struct ModalityEncoder {
    pub modality: MemoryModality,
    pub encoding_matrix: Array2<f32>,
    pub feature_extractors: Vec<Box<dyn FeatureExtractor>>,
}

/// Trait for modality-specific feature extraction
pub trait FeatureExtractor: Send + Sync {
    fn extract(&self, node: &ConsciousNode) -> Result<Vec<f32>>;
    fn name(&self) -> &str;
}

/// Cross-modal performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossModalMetrics {
    pub total_translations: u64,
    pub successful_translations: u64,
    pub avg_translation_time_ms: f64,
    pub modality_usage: HashMap<MemoryModality, u64>,
    pub bridge_utilization: HashMap<(MemoryModality, MemoryModality), u64>,
}

impl CrossModalBridge {
    /// Create a new cross-modal bridge
    pub fn new(config: CrossModalConfig) -> Self {
        let mut bridge = Self {
            config: config.clone(),
            encoders: Arc::new(RwLock::new(HashMap::new())),
            translators: Arc::new(RwLock::new(HashMap::new())),
            connections: Arc::new(RwLock::new(Vec::new())),
            node_modalities: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(CrossModalMetrics::default())),
        };
        
        // Initialize modality encoders
        bridge.initialize_encoders();
        
        // Initialize translation matrices
        if config.enable_all_connections {
            bridge.initialize_all_translators();
        }
        
        bridge
    }
    
    /// Initialize modality encoders
    fn initialize_encoders(&mut self) {
        let modalities = [
            MemoryModality::Semantic,
            MemoryModality::Episodic,
            MemoryModality::Emotional,
            MemoryModality::Procedural,
            MemoryModality::Contextual,
            MemoryModality::Temporal,
            MemoryModality::Causal,
            MemoryModality::Abstract,
        ];
        
        let mut encoders = self.encoders.write().unwrap();
        
        for modality in &modalities {
            let encoder = ModalityEncoder {
                modality: *modality,
                encoding_matrix: Array2::from_shape_fn(
                    (self.config.feature_dimension, self.config.feature_dimension),
                    |_| rand::random::<f32>() * 0.1 - 0.05
                ),
                feature_extractors: vec![], // Will be populated with specific extractors
            };
            encoders.insert(*modality, encoder);
        }
    }
    
    /// Initialize all possible translation matrices
    fn initialize_all_translators(&mut self) {
        let modalities = [
            MemoryModality::Semantic,
            MemoryModality::Episodic,
            MemoryModality::Emotional,
            MemoryModality::Procedural,
            MemoryModality::Contextual,
            MemoryModality::Temporal,
            MemoryModality::Causal,
            MemoryModality::Abstract,
        ];
        
        let mut translators = self.translators.write().unwrap();
        
        for source in &modalities {
            for target in &modalities {
                if source != target {
                    let translator = ModalityTranslator {
                        translation_matrix: Array2::from_shape_fn(
                            (self.config.feature_dimension, self.config.feature_dimension),
                            |_| rand::random::<f32>() * 0.1 - 0.05
                        ),
                        bias: Array1::zeros(self.config.feature_dimension),
                        learning_rate: self.config.learning_rate,
                    };
                    translators.insert((*source, *target), translator);
                }
            }
        }
    }
    
    /// Register a node with a specific modality
    pub fn register_node(&self, node_id: NodeId, modality: MemoryModality) -> Result<()> {
        let mut node_modalities = self.node_modalities.write().unwrap();
        node_modalities.insert(node_id, modality);
        
        // Update metrics
        let mut metrics = self.metrics.write().unwrap();
        *metrics.modality_usage.entry(modality).or_insert(0) += 1;
        
        Ok(())
    }
    
    /// Translate features from one modality to another
    pub fn translate_features(
        &self,
        features: &Array1<f32>,
        source_modality: MemoryModality,
        target_modality: MemoryModality,
    ) -> Result<Array1<f32>> {
        let start_time = std::time::Instant::now();
        
        // Get translator
        let translators = self.translators.read().unwrap();
        let translator = translators
            .get(&(source_modality, target_modality))
            .ok_or_else(|| anyhow::anyhow!("No translator found for {:?} -> {:?}", 
                source_modality, target_modality))?;
        
        // Perform translation: y = Wx + b
        let translated = translator.translation_matrix.dot(features) + &translator.bias;
        
        // Update metrics
        let mut metrics = self.metrics.write().unwrap();
        metrics.total_translations += 1;
        metrics.successful_translations += 1;
        let elapsed_ms = start_time.elapsed().as_millis() as f64;
        metrics.avg_translation_time_ms = 
            (metrics.avg_translation_time_ms * (metrics.total_translations - 1) as f64 + elapsed_ms) 
            / metrics.total_translations as f64;
        *metrics.bridge_utilization
            .entry((source_modality, target_modality))
            .or_insert(0) += 1;
        
        Ok(translated)
    }
    
    /// Create a cross-modal connection between nodes
    pub fn create_connection(
        &self,
        source_node: &ConsciousNode,
        target_node: &ConsciousNode,
        source_modality: MemoryModality,
        target_modality: MemoryModality,
    ) -> Result<ConsciousEdge> {
        // Create bridge strength
        let strength = BridgeStrength {
            base_strength: 0.5,
            learned_strength: 0.0,
            usage_count: 0,
            last_used: Utc::now(),
        };
        
        // Create or update connection
        let mut connections = self.connections.write().unwrap();
        let connection = connections.iter_mut().find(|c| 
            c.source_modality == source_modality && c.target_modality == target_modality
        );
        
        match connection {
            Some(conn) => {
                // Add example pair
                conn.example_pairs.push((source_node.id_string(), target_node.id_string()));
                if conn.example_pairs.len() > self.config.max_example_pairs {
                    conn.example_pairs.remove(0);
                }
                conn.strength.usage_count += 1;
                conn.strength.last_used = Utc::now();
            }
            None => {
                // Create new connection
                let new_connection = CrossModalConnection {
                    source_modality,
                    target_modality,
                    strength,
                    example_pairs: vec![(source_node.id_string(), target_node.id_string())],
                    metadata: HashMap::new(),
                };
                connections.push(new_connection);
            }
        }
        
        // Create conscious edge
        let edge = ConsciousEdge::new(
            source_node.id_string(),
            target_node.id_string(),
            EdgeType::Association { 
                strength: 0.5 // Will be updated based on bridge strength
            },
        );
        
        Ok(edge)
    }
    
    /// Find cross-modal connections for a node
    pub fn find_connections(&self, node_id: &NodeId) -> Result<Vec<CrossModalConnection>> {
        let connections = self.connections.read().unwrap();
        let node_modalities = self.node_modalities.read().unwrap();
        
        let node_modality = node_modalities.get(node_id)
            .ok_or_else(|| anyhow::anyhow!("Node modality not found"))?;
        
        let relevant_connections: Vec<_> = connections
            .iter()
            .filter(|conn| {
                conn.source_modality == *node_modality || 
                conn.target_modality == *node_modality
            })
            .cloned()
            .collect();
        
        Ok(relevant_connections)
    }
    
    /// Update translator weights based on feedback
    pub fn update_translator(
        &self,
        source_modality: MemoryModality,
        target_modality: MemoryModality,
        source_features: &Array1<f32>,
        target_features: &Array1<f32>,
        learning_rate: Option<f32>,
    ) -> Result<()> {
        let mut translators = self.translators.write().unwrap();
        let translator = translators
            .get_mut(&(source_modality, target_modality))
            .ok_or_else(|| anyhow::anyhow!("Translator not found"))?;
        
        let lr = learning_rate.unwrap_or(translator.learning_rate);
        
        // Compute prediction
        let prediction = translator.translation_matrix.dot(source_features) + &translator.bias;
        
        // Compute error
        let error = target_features - &prediction;
        
        // Update weights using gradient descent
        // W = W + lr * error * source^T
        let gradient = error.clone().insert_axis(ndarray::Axis(1))
            .dot(&source_features.clone().insert_axis(ndarray::Axis(0)));
        translator.translation_matrix = &translator.translation_matrix + &(gradient * lr);
        
        // Update bias
        translator.bias = &translator.bias + &(error * lr);
        
        Ok(())
    }
    
    /// Get cross-modal query results
    pub fn cross_modal_query(
        &self,
        query_embedding: &Array1<f32>,
        source_modality: MemoryModality,
        target_modalities: Vec<MemoryModality>,
    ) -> Result<Vec<(MemoryModality, Array1<f32>)>> {
        let mut results = Vec::new();
        
        for target_modality in target_modalities {
            let translated = self.translate_features(
                query_embedding,
                source_modality,
                target_modality,
            )?;
            results.push((target_modality, translated));
        }
        
        Ok(results)
    }
    
    /// Get bridge statistics
    pub fn get_stats(&self) -> CrossModalStats {
        let connections = self.connections.read().unwrap();
        let metrics = self.metrics.read().unwrap();
        let translators = self.translators.read().unwrap();
        
        CrossModalStats {
            total_modalities: 8,
            active_connections: connections.len(),
            total_translators: translators.len(),
            total_translations: metrics.total_translations,
            success_rate: if metrics.total_translations > 0 {
                metrics.successful_translations as f64 / metrics.total_translations as f64
            } else {
                0.0
            },
            avg_translation_time_ms: metrics.avg_translation_time_ms,
        }
    }
}

/// Cross-modal statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalStats {
    pub total_modalities: usize,
    pub active_connections: usize,
    pub total_translators: usize,
    pub total_translations: u64,
    pub success_rate: f64,
    pub avg_translation_time_ms: f64,
}

/// Default feature extractor for basic embeddings
pub struct EmbeddingExtractor;

impl FeatureExtractor for EmbeddingExtractor {
    fn extract(&self, node: &ConsciousNode) -> Result<Vec<f32>> {
        Ok(node.embeddings.clone())
    }
    
    fn name(&self) -> &str {
        "embedding"
    }
}

// Tests removed - outdated imports
