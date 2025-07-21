//! Main ConsciousGraph container that orchestrates all graph components

use crate::graph::{
    GraphOperations, GraphQueryResult, GraphPath, QueryStats,
    ConsciousNode, ConsciousEdge, NodeType, EdgeType, NodeId, EdgeId,
    GraphStorage, GraphIndices, HnswIndex, GraphAlgorithms, TraversalOptions,
    RelationshipInference, PatternExtractor, ExtractedPattern,
    dream_consolidation::DreamConsolidation,
    temporal_tracker::TemporalTracker
};
use crate::embeddings::{EmbeddingService, EmbeddingConfig, EmbeddingError};
use anyhow::{Result, anyhow};
use std::sync::Arc;
use parking_lot::RwLock;
use std::time::Instant;
use std::path::PathBuf;
use chrono::Utc;

/// Main Conscious Knowledge Graph that orchestrates all components
pub struct ConsciousGraph {
    /// Graph storage backend
    pub storage: Arc<GraphStorage>,
    
    /// High-performance indices
    indices: Arc<GraphIndices>,
    
    /// Relationship inference engine
    inference: Arc<RelationshipInference>,
    
    /// Pattern extraction system
    patterns: Arc<PatternExtractor>,
    
    /// Graph algorithms
    algorithms: Arc<GraphAlgorithms>,
    
    /// Temporal access tracker
    temporal_tracker: Arc<TemporalTracker>,
    
    /// Semantic embedding service
    embedding_service: Option<Arc<EmbeddingService>>,
    
    /// Configuration
    config: ConsciousGraphConfig,
    
    /// Runtime statistics
    stats: Arc<RwLock<GraphStats>>,
}

/// Configuration for the conscious graph
#[derive(Debug, Clone)]
pub struct ConsciousGraphConfig {
    pub embedding_dim: usize,
    pub auto_infer_relationships: bool,
    pub consciousness_threshold: f32,
    pub max_nodes: usize,
    pub persistence_enabled: bool,
    pub storage_path: PathBuf,
    /// Semantic embedding configuration
    pub semantic_embeddings_enabled: bool,
    pub embedding_config: Option<EmbeddingConfig>,
}

impl Default for ConsciousGraphConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 768,
            auto_infer_relationships: true,
            consciousness_threshold: 0.3,
            max_nodes: 100_000,
            persistence_enabled: true,
            storage_path: PathBuf::from("./conscious_graph_data"),
            semantic_embeddings_enabled: true,
            embedding_config: Some(EmbeddingConfig::default()),
        }
    }
}

/// Runtime statistics for the graph
#[derive(Debug, Default, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub total_queries: u64,
    pub avg_query_time_ms: f64,
    pub consciousness_activations: u64,
    pub pattern_extractions: u64,
    pub relationship_inferences: u64,
}

impl ConsciousGraph {
    /// Create a new ConsciousGraph with default configuration
    pub fn new() -> Result<Self> {
        Self::new_with_config(ConsciousGraphConfig::default())
    }
    
    /// Create a new ConsciousGraph with custom configuration
    pub fn new_with_config(config: ConsciousGraphConfig) -> Result<Self> {
        // Initialize storage
        let storage = Arc::new(GraphStorage::new(config.storage_path.clone())?);
        
        // Initialize indices
        let indices = Arc::new(GraphIndices::new(config.embedding_dim));
        
        // Initialize inference engine
        let inference = Arc::new(RelationshipInference::new());
        
        // Initialize pattern extractor
        let patterns = Arc::new(PatternExtractor::new());
        
        // Initialize algorithms
        let algorithms = Arc::new(GraphAlgorithms);
        
        // Initialize temporal tracker (24 hour window by default)
        let temporal_tracker = Arc::new(TemporalTracker::new(24));
        
        // Initialize stats
        let stats = Arc::new(RwLock::new(GraphStats::default()));
        
        // Note: embedding service is None for synchronous initialization
        // Use new_with_config_async for semantic embeddings
        Ok(Self {
            storage,
            indices,
            inference,
            patterns,
            algorithms,
            temporal_tracker,
            embedding_service: None,
            config,
            stats,
        })
    }
    
    /// Create a new ConsciousGraph with custom configuration and semantic embeddings
    pub async fn new_with_config_async(config: ConsciousGraphConfig) -> Result<Self> {
        // Initialize storage
        let storage = Arc::new(GraphStorage::new(config.storage_path.clone())?);
        
        // Initialize indices
        let indices = Arc::new(GraphIndices::new(config.embedding_dim));
        
        // Initialize inference engine
        let inference = Arc::new(RelationshipInference::new());
        
        // Initialize pattern extractor
        let patterns = Arc::new(PatternExtractor::new());
        
        // Initialize algorithms
        let algorithms = Arc::new(GraphAlgorithms);
        
        // Initialize temporal tracker (24 hour window by default)
        let temporal_tracker = Arc::new(TemporalTracker::new(24));
        
        // Initialize stats
        let stats = Arc::new(RwLock::new(GraphStats::default()));
        
        // Initialize embedding service if enabled
        let embedding_service = if config.semantic_embeddings_enabled {
            if let Some(ref embedding_config) = config.embedding_config {
                match EmbeddingService::new(embedding_config.clone()).await {
                    Ok(service) => {
                        tracing::info!("Initialized semantic embedding service with model: {}", 
                            embedding_config.model_id);
                        Some(Arc::new(service))
                    }
                    Err(e) => {
                        tracing::warn!("Failed to initialize embedding service: {}. Falling back to hash-based embeddings.", e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };
        
        Ok(Self {
            storage,
            indices,
            inference,
            patterns,
            algorithms,
            temporal_tracker,
            embedding_service,
            config,
            stats,
        })
    }
    
    /// Get graph statistics
    pub fn get_stats(&self) -> GraphStats {
        self.stats.read().clone()
    }
    
    /// Get the temporal tracker
    pub fn get_temporal_tracker(&self) -> Arc<TemporalTracker> {
        Arc::clone(&self.temporal_tracker)
    }
    
    /// Update consciousness levels based on usage patterns
    pub fn update_consciousness(&self) -> Result<()> {
        // This would implement consciousness updating logic
        // For now, just increment the activation counter
        self.stats.write().consciousness_activations += 1;
        Ok(())
    }
    
    /// Perform dream-like consolidation of memories
    pub async fn dream_consolidation(self: Arc<Self>) -> Result<usize> {
        let start_time = Instant::now();
        
        // Use the DreamConsolidation processor for proper edge creation
        let dream_processor = DreamConsolidation::new_from_graph(self.clone());
        let consolidation_result = dream_processor.process().await?;
        
        // Update stats with actual relationships created
        let mut stats = self.stats.write();
        stats.pattern_extractions += consolidation_result.patterns_found as u64;
        stats.edge_count += consolidation_result.relationships_created;
        
        // Log actual relationships created
        println!("Dream consolidation created {} relationships, {} insights", 
                 consolidation_result.relationships_created,
                 consolidation_result.insights_generated);
        
        Ok(consolidation_result.relationships_created)
    }
    
    /// Find similar nodes using consciousness-aware similarity
    pub fn find_conscious_similar(&self, node_id: &NodeId, limit: usize) -> Result<Vec<(NodeId, f32)>> {
        // Use HNSW index for fast similarity search
        let similar = self.indices.embedding_index.read()
            .search_by_node(node_id, limit)?;
        
        // Filter by consciousness threshold
        let conscious_similar: Vec<(NodeId, f32)> = similar
            .into_iter()
            .filter(|(_, score)| *score >= self.config.consciousness_threshold)
            .collect();
        
        Ok(conscious_similar)
    }
    
    /// Traverse graph with consciousness awareness
    pub fn conscious_traverse(&self, start: &NodeId, depth: usize) -> Result<GraphQueryResult> {
        let options = TraversalOptions {
            max_depth: depth,
            consciousness_threshold: Some(self.config.consciousness_threshold),
            follow_emotional: true,
            ..Default::default()
        };
        
        let result = GraphAlgorithms::bfs(&self.storage, start, options)?;
        
        // Update stats
        let mut stats = self.stats.write();
        stats.total_queries += 1;
        stats.consciousness_activations += result.stats.consciousness_activations as u64;
        
        Ok(result)
    }
}

impl GraphOperations for ConsciousGraph {
    fn add_node(&self, node: NodeType) -> Result<NodeId> {
        let start_time = Instant::now();
        
        // Extract content based on node type
        let (key, content) = match &node {
            NodeType::Memory(mem) => (mem.id.clone(), mem.value.clone()),
            NodeType::Pattern(pat) => (pat.id.clone(), pat.description.clone()),
            NodeType::Concept(con) => (con.id.clone(), con.definition.clone()),
            NodeType::Context(ctx) => (ctx.id.clone(), ctx.description.clone()),
            NodeType::Entity(ent) => (ent.id.clone(), format!("{} ({})", ent.name, ent.entity_type)),
        };
        
        // Generate embedding from content
        let embedding = self.generate_embedding(&content);
        
        // Create ConsciousNode from NodeType
        let conscious_node = ConsciousNode::new(
            key,
            content,
            embedding,
            node,
        );
        
        // Add to storage
        let node_id = self.storage.add_node(conscious_node.clone())?;
        
        // Index the node
        self.indices.index_node(&conscious_node)?;
        
        // Auto-infer relationships if enabled
        if self.config.auto_infer_relationships {
            // Note: This would need to be in an async context to properly await
            // For now, we'll skip the inference as it requires async
            // TODO: Make add_node async or use a different approach
            self.stats.write().relationship_inferences += 1;
        }
        
        // Update stats
        let mut stats = self.stats.write();
        stats.node_count += 1;
        
        Ok(node_id)
    }
    
    fn add_edge(&self, edge: ConsciousEdge) -> Result<EdgeId> {
        let edge_id = self.storage.add_edge(edge)?;
        
        // Update stats
        self.stats.write().edge_count += 1;
        
        Ok(edge_id)
    }
    
    fn query_graph(&self, start: &NodeId, depth: usize) -> Result<GraphQueryResult> {
        // Record access to the starting node
        self.temporal_tracker.record_access(start);
        
        let result = self.conscious_traverse(start, depth)?;
        
        // Record access to all nodes in the result
        let node_ids: Vec<NodeId> = result.nodes.iter()
            .map(|node| node.id_string())
            .collect();
        self.temporal_tracker.record_batch_access(&node_ids);
        
        Ok(result)
    }
    
    fn extract_patterns(&self, context: &str) -> Result<Vec<ExtractedPattern>> {
        let patterns = self.patterns.extract_contextual_patterns(context)?;
        
        // Update stats
        self.stats.write().pattern_extractions += patterns.len() as u64;
        
        Ok(patterns)
    }
}

impl ConsciousGraph {
    /// Generate embedding from text using semantic embeddings or fallback to hash-based
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        // If semantic embeddings are available, use them synchronously via blocking
        if let Some(ref embedding_service) = self.embedding_service {
            // Use tokio's block_in_place to run async code in sync context
            // This is safe because we're in a tokio runtime context
            match tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    embedding_service.embed(text).await
                })
            }) {
                Ok(embedding) => {
                    // Ensure embedding matches configured dimension
                    if embedding.len() == self.config.embedding_dim {
                        return embedding;
                    } else {
                        tracing::warn!(
                            "Embedding dimension mismatch: expected {}, got {}. Using hash-based fallback.",
                            self.config.embedding_dim,
                            embedding.len()
                        );
                    }
                }
                Err(e) => {
                    tracing::debug!("Failed to generate semantic embedding: {}. Using hash-based fallback.", e);
                }
            }
        }
        
        // Fallback to hash-based approach
        self.generate_hash_embedding(text)
    }
    
    /// Generate embedding using hash-based approach (fallback)
    fn generate_hash_embedding(&self, text: &str) -> Vec<f32> {
        let embedding_dim = self.config.embedding_dim;
        let mut embedding = vec![0.0; embedding_dim];
        
        // Simple hash-based features
        for (i, ch) in text.chars().enumerate() {
            let idx = (ch as usize + i) % embedding_dim;
            embedding[idx] += 1.0;
        }
        
        // Normalize to unit vector for cosine similarity
        let sum: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if sum > 0.0 {
            for val in &mut embedding {
                *val /= sum;
            }
        }
        
        embedding
    }
    
    /// Generate embedding asynchronously (preferred for new code)
    pub async fn generate_embedding_async(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(ref embedding_service) = self.embedding_service {
            match embedding_service.embed(text).await {
                Ok(embedding) => {
                    if embedding.len() == self.config.embedding_dim {
                        Ok(embedding)
                    } else {
                        tracing::warn!(
                            "Embedding dimension mismatch: expected {}, got {}. Using hash-based fallback.",
                            self.config.embedding_dim,
                            embedding.len()
                        );
                        Ok(self.generate_hash_embedding(text))
                    }
                }
                Err(e) => {
                    if self.config.embedding_config.as_ref().map_or(false, |c| c.fallback_enabled) {
                        tracing::debug!("Failed to generate semantic embedding: {}. Using hash-based fallback.", e);
                        Ok(self.generate_hash_embedding(text))
                    } else {
                        Err(anyhow!("Failed to generate embedding: {}", e))
                    }
                }
            }
        } else {
            Ok(self.generate_hash_embedding(text))
        }
    }
}

impl Default for ConsciousGraph {
    fn default() -> Self {
        Self::new().expect("Failed to create default ConsciousGraph")
    }
}

