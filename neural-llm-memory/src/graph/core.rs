//! Core data structures for the Conscious Knowledge Graph

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Unique identifier for nodes
pub type NodeId = String;

/// Unique identifier for edges
pub type EdgeId = String;

/// Type of node in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Memory(MemoryNode),
    Concept(ConceptNode),
    Entity(EntityNode),
    Context(ContextNode),
    Pattern(PatternNode),
}

/// A memory node containing stored information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: NodeId,
    pub key: String,
    pub value: String,
    pub embedding: Vec<f32>,
    pub created_at: DateTime<Utc>,
    pub accessed_at: DateTime<Utc>,
    pub access_count: u32,
}

/// A concept node representing abstract ideas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    pub id: NodeId,
    pub name: String,
    pub definition: String,
    pub embedding: Vec<f32>,
    pub confidence: f32,
    pub source_memories: Vec<NodeId>,
}

/// An entity node representing concrete objects or beings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    pub id: NodeId,
    pub name: String,
    pub entity_type: String,
    pub attributes: HashMap<String, serde_json::Value>,
    pub embedding: Vec<f32>,
}

/// A context node representing situational information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextNode {
    pub id: NodeId,
    pub description: String,
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub location: Option<String>,
    pub participants: Vec<NodeId>,
}

/// A pattern node representing discovered patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternNode {
    pub id: NodeId,
    pub pattern_type: String,
    pub description: String,
    pub frequency: usize,
    pub confidence: f32,
    pub examples: Vec<NodeId>,
}

/// A conscious node with awareness and emotional attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousNode {
    // Identity
    pub id: Uuid,
    pub key: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    
    // Content
    pub content: String,
    pub embeddings: Vec<f32>, // 768-dimensional
    pub properties: HashMap<String, serde_json::Value>,
    
    // Node type
    pub node_type: NodeType,
    
    // Consciousness attributes
    pub awareness: ConsciousAwareness,
    pub emotional_state: EmotionalState,
    pub cognitive_metadata: CognitiveMetadata,
    pub learning_state: LearningState,
}

/// Consciousness awareness attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousAwareness {
    pub level: f32,                    // 0.0-1.0
    pub last_activation: DateTime<Utc>,
    pub activation_count: u64,
    pub broadcast_priority: f32,
}

/// Emotional state of a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub primary: Emotion,
    pub valence: f32,                  // -1.0 to 1.0
    pub arousal: f32,                  // 0.0 to 1.0
    pub tags: Vec<EmotionalTag>,
}

/// Primary emotions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Emotion {
    Joy,
    Curiosity,
    Fear,
    Surprise,
    Sadness,
    Anger,
    Neutral,
}

/// Emotional tags for nuanced states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalTag(pub String);

/// Cognitive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMetadata {
    pub pattern: PatternType,
    pub thinking_style: ThinkingStyle,
    pub abstraction_level: u8,         // 0-10
    pub metacognitive_notes: Vec<String>,
}

/// Cognitive pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Convergent,
    Divergent,
    Lateral,
    Systems,
    Critical,
    Abstract,
}

/// Thinking styles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThinkingStyle {
    Analytical,
    Creative,
    Practical,
    Theoretical,
    Intuitive,
}

/// Learning state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningState {
    pub importance_score: f32,
    pub access_frequency: f32,
    pub last_consolidated: Option<DateTime<Utc>>,
    pub evolution_generation: u64,
}

/// Types of edges in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeType {
    Related { weight: f32 },
    PartOf,
    CausedBy,
    Temporal { delta_ms: i64 },
    Derived,
    Association { strength: f32 },
}

/// A conscious edge with traversal tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousEdge {
    // Relationship
    pub id: Uuid,
    pub source: NodeId,
    pub target: NodeId,
    pub edge_type: EdgeType,
    pub created_at: DateTime<Utc>,
    pub properties: HashMap<String, serde_json::Value>,
    
    // Strength and usage
    pub strength: EdgeStrength,
    pub traversal_stats: TraversalStats,
    pub consciousness: EdgeConsciousness,
}

/// Edge strength components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeStrength {
    pub base: f32,                     // Initial strength
    pub learned: f32,                  // Neural network weight
    pub usage: f32,                    // Based on traversal count
    pub temporal: f32,                 // Time-decay factor
    pub combined: f32,                 // Calculated overall strength
}

/// Traversal statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalStats {
    pub count: u64,
    pub last_traversed: DateTime<Utc>,
    pub avg_traversal_time: std::time::Duration,
    pub success_rate: f32,
}

/// Edge consciousness attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConsciousness {
    pub emotional_resonance: f32,
    pub cognitive_pathway: PathwayType,
    pub bidirectional: bool,
    pub inhibitory: bool,              // Can block activation
}

/// Types of cognitive pathways
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathwayType {
    Logical,
    Associative,
    Emotional,
    Temporal,
    Causal,
    Creative,
}

impl ConsciousNode {
    /// Create a new ConsciousNode from basic information
    pub fn new(key: String, content: String, embeddings: Vec<f32>, node_type: NodeType) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            key,
            created_at: now,
            updated_at: now,
            content,
            embeddings,
            properties: HashMap::new(),
            node_type,
            awareness: ConsciousAwareness {
                level: 0.5,
                last_activation: now,
                activation_count: 0,
                broadcast_priority: 0.5,
            },
            emotional_state: EmotionalState {
                primary: Emotion::Neutral,
                valence: 0.0,
                arousal: 0.5,
                tags: vec![],
            },
            cognitive_metadata: CognitiveMetadata {
                pattern: PatternType::Convergent,
                thinking_style: ThinkingStyle::Analytical,
                abstraction_level: 5,
                metacognitive_notes: vec![],
            },
            learning_state: LearningState {
                importance_score: 0.5,
                access_frequency: 0.0,
                last_consolidated: None,
                evolution_generation: 0,
            },
        }
    }
    
    /// Get the node's ID as a string
    pub fn id_string(&self) -> NodeId {
        self.id.to_string()
    }
}

impl ConsciousEdge {
    /// Create a new ConsciousEdge
    pub fn new(source: NodeId, target: NodeId, edge_type: EdgeType) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            source,
            target,
            edge_type,
            created_at: now,
            properties: HashMap::new(),
            strength: EdgeStrength {
                base: 0.5,
                learned: 0.5,
                usage: 0.0,
                temporal: 1.0,
                combined: 0.5,
            },
            traversal_stats: TraversalStats {
                count: 0,
                last_traversed: now,
                avg_traversal_time: std::time::Duration::from_millis(0),
                success_rate: 1.0,
            },
            consciousness: EdgeConsciousness {
                emotional_resonance: 0.0,
                cognitive_pathway: PathwayType::Logical,
                bidirectional: true,
                inhibitory: false,
            },
        }
    }
    
    /// Get the edge's ID as a string
    pub fn id_string(&self) -> EdgeId {
        self.id.to_string()
    }
}