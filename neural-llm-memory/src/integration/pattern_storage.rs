//! Pattern storage implementation for neural memory integration

use crate::integration::{LearnedPattern, PatternType, PatternContext, PerformanceMetrics, PatternMemory, PatternMatch, ComplexityLevel};
use crate::memory::{MemoryModule, MemoryKey, MemoryMetadata, MemoryOperations};
use ndarray::Array2;
use std::collections::HashMap;
use chrono::Utc;

/// Pattern storage system that integrates with neural memory
pub struct PatternStorage {
    memory_module: MemoryModule,
    pattern_index: HashMap<String, MemoryKey>,
    embedding_dim: usize,
}

impl PatternStorage {
    pub fn new(memory_module: MemoryModule) -> Self {
        let embedding_dim = memory_module.config().embedding_dim;
        Self {
            memory_module,
            pattern_index: HashMap::new(),
            embedding_dim,
        }
    }
    
    /// Convert pattern to embedding vector
    fn pattern_to_embedding(&self, pattern: &LearnedPattern) -> Array2<f32> {
        let mut embedding = vec![0.0f32; self.embedding_dim];
        
        // Encode pattern type (first 6 dimensions)
        match &pattern.pattern_type {
            PatternType::Convergent => embedding[0] = 1.0,
            PatternType::Divergent => embedding[1] = 1.0,
            PatternType::Lateral => embedding[2] = 1.0,
            PatternType::Systems => embedding[3] = 1.0,
            PatternType::Critical => embedding[4] = 1.0,
            PatternType::Abstract => embedding[5] = 1.0,
            PatternType::Hybrid(_, _) => {
                embedding[0] = 0.5;
                embedding[1] = 0.5;
            }
        }
        
        // Encode performance metrics (dimensions 6-15)
        embedding[6] = pattern.performance.success_rate;
        embedding[7] = (pattern.performance.average_time_ms / 10000.0).min(1.0) as f32;
        embedding[8] = pattern.performance.token_efficiency;
        embedding[9] = 1.0 - pattern.performance.error_rate;
        embedding[10] = pattern.performance.adaptability_score;
        embedding[11] = (pattern.performance.usage_count as f32 / 100.0).min(1.0);
        embedding[12] = pattern.confidence;
        
        // Encode complexity (dimensions 13-16)
        match pattern.context.complexity {
            ComplexityLevel::Simple => embedding[13] = 1.0,
            ComplexityLevel::Moderate => embedding[14] = 1.0,
            ComplexityLevel::Complex => embedding[15] = 1.0,
            ComplexityLevel::VeryComplex => embedding[16] = 1.0,
        }
        
        // Encode domain hash (dimensions 17-32)
        let domain_hash = self.hash_string(&pattern.context.domain);
        for i in 0..16 {
            embedding[17 + i] = ((domain_hash >> (i * 2)) & 0x3) as f32 / 3.0;
        }
        
        // Encode file types (dimensions 33-48)
        for (i, file_type) in pattern.context.file_types.iter().take(16).enumerate() {
            let type_hash = self.hash_string(file_type);
            embedding[33 + i] = (type_hash & 0xFF) as f32 / 255.0;
        }
        
        // Use neural weights if available (dimensions 49+)
        let weights_start = 49;
        let weights_to_use = (self.embedding_dim - weights_start).min(pattern.neural_weights.len());
        for i in 0..weights_to_use {
            embedding[weights_start + i] = pattern.neural_weights[i];
        }
        
        // Normalize the embedding
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        
        Array2::from_shape_vec((1, self.embedding_dim), embedding).unwrap()
    }
    
    /// Simple string hashing for consistent encoding
    fn hash_string(&self, s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Create context embedding for search
    fn context_to_embedding(&self, context: &PatternContext) -> Array2<f32> {
        // Similar to pattern_to_embedding but only uses context information
        let mut embedding = vec![0.0f32; self.embedding_dim];
        
        // Leave pattern type dimensions as 0 for neutral search
        
        // Encode complexity (dimensions 13-16)
        match context.complexity {
            ComplexityLevel::Simple => embedding[13] = 1.0,
            ComplexityLevel::Moderate => embedding[14] = 1.0,
            ComplexityLevel::Complex => embedding[15] = 1.0,
            ComplexityLevel::VeryComplex => embedding[16] = 1.0,
        }
        
        // Encode domain
        let domain_hash = self.hash_string(&context.domain);
        for i in 0..16 {
            embedding[17 + i] = ((domain_hash >> (i * 2)) & 0x3) as f32 / 3.0;
        }
        
        // Encode file types
        for (i, file_type) in context.file_types.iter().take(16).enumerate() {
            let type_hash = self.hash_string(file_type);
            embedding[33 + i] = (type_hash & 0xFF) as f32 / 255.0;
        }
        
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        
        Array2::from_shape_vec((1, self.embedding_dim), embedding).unwrap()
    }
    
    /// Calculate pattern relevance based on context
    fn calculate_relevance(&self, pattern: &LearnedPattern, context: &PatternContext) -> f32 {
        let mut relevance = 0.0f32;
        
        // Domain match
        if pattern.context.domain == context.domain {
            relevance += 0.3;
        }
        
        // File type overlap
        let file_type_overlap = pattern.context.file_types.iter()
            .filter(|ft| context.file_types.contains(ft))
            .count() as f32;
        let max_types = pattern.context.file_types.len().max(context.file_types.len()) as f32;
        if max_types > 0.0 {
            relevance += 0.2 * (file_type_overlap / max_types);
        }
        
        // Complexity match
        if pattern.context.complexity == context.complexity {
            relevance += 0.2;
        } else {
            let complexity_diff = (pattern.context.complexity as i32 - context.complexity as i32).abs();
            relevance += 0.2 * (1.0 - complexity_diff as f32 / 3.0).max(0.0);
        }
        
        // Success rate weight
        relevance += 0.2 * pattern.performance.success_rate;
        
        // Recency weight
        let age_days = (Utc::now() - pattern.timestamp).num_days() as f32;
        let recency_factor = (-age_days / 30.0).exp(); // Exponential decay over 30 days
        relevance += 0.1 * recency_factor;
        
        relevance
    }
}

impl PatternMemory for PatternStorage {
    fn store_pattern(&mut self, pattern: LearnedPattern) -> Result<String, Box<dyn std::error::Error>> {
        // Generate pattern ID if not present
        let pattern_id = pattern.id.clone();
        
        // Convert pattern to embedding
        let embedding = self.pattern_to_embedding(&pattern);
        
        // Serialize pattern data
        let pattern_json = serde_json::to_string(&pattern)?;
        
        // Create memory metadata
        let mut metadata = MemoryMetadata::default();
        metadata.importance = pattern.confidence;
        metadata.tags = vec![
            pattern.pattern_type.to_string(),
            pattern.context.domain.clone(),
            format!("complexity:{:?}", pattern.context.complexity),
        ];
        
        // Store in memory module
        let memory_key = self.memory_module.store_memory(pattern_json, embedding)?;
        
        // Update index
        self.pattern_index.insert(pattern_id.clone(), memory_key);
        
        Ok(pattern_id)
    }
    
    fn retrieve_pattern(&self, pattern_id: &str) -> Option<LearnedPattern> {
        // Get memory key from index
        let memory_key = self.pattern_index.get(pattern_id)?;
        
        // Retrieve from memory module
        let memory_value = self.memory_module.memory_bank().write()
            .retrieve(memory_key).ok()??;
        
        // Deserialize pattern
        serde_json::from_str(&memory_value.content).ok()
    }
    
    fn search_patterns(&self, context: &PatternContext, limit: usize) -> Vec<PatternMatch> {
        // Create context embedding
        let context_embedding = self.context_to_embedding(context);
        
        // Search in memory module
        let search_results = self.memory_module.retrieve_with_attention(&context_embedding, limit * 2);
        
        // Convert to pattern matches
        let mut matches = Vec::new();
        for (_key, value, similarity) in search_results {
            if let Ok(pattern) = serde_json::from_str::<LearnedPattern>(&value.content) {
                let relevance = self.calculate_relevance(&pattern, context);
                
                matches.push(PatternMatch {
                    pattern,
                    similarity: similarity[[0, 0]],
                    relevance_score: relevance,
                    suggested_adaptations: vec![], // TODO: Implement adaptation suggestions
                });
            }
        }
        
        // Sort by combined score
        matches.sort_by(|a, b| {
            let score_a = a.similarity * 0.6 + a.relevance_score * 0.4;
            let score_b = b.similarity * 0.6 + b.relevance_score * 0.4;
            score_b.partial_cmp(&score_a).unwrap()
        });
        
        matches.truncate(limit);
        matches
    }
    
    fn update_pattern_metrics(&mut self, pattern_id: &str, metrics: &PerformanceMetrics) {
        if let Some(mut pattern) = self.retrieve_pattern(pattern_id) {
            // Update metrics
            pattern.performance = metrics.clone();
            pattern.timestamp = Utc::now();
            
            // Re-store with updated data
            let _ = self.store_pattern(pattern);
        }
    }
    
    fn prune_old_patterns(&mut self, max_age_days: u32) -> usize {
        let cutoff_date = Utc::now() - chrono::Duration::days(max_age_days as i64);
        let mut pruned = 0;
        
        // Collect patterns to prune
        let patterns_to_prune: Vec<String> = self.pattern_index.keys()
            .filter_map(|id| {
                self.retrieve_pattern(id).and_then(|pattern| {
                    if pattern.timestamp < cutoff_date && pattern.performance.usage_count < 5 {
                        Some(id.clone())
                    } else {
                        None
                    }
                })
            })
            .collect();
        
        // Remove from memory
        for pattern_id in patterns_to_prune {
            if let Some(memory_key) = self.pattern_index.remove(&pattern_id) {
                if self.memory_module.memory_bank().write().delete(&memory_key).unwrap_or(false) {
                    pruned += 1;
                }
            }
        }
        
        pruned
    }
}

impl PatternType {
    fn to_string(&self) -> String {
        match self {
            PatternType::Convergent => "convergent".to_string(),
            PatternType::Divergent => "divergent".to_string(),
            PatternType::Lateral => "lateral".to_string(),
            PatternType::Systems => "systems".to_string(),
            PatternType::Critical => "critical".to_string(),
            PatternType::Abstract => "abstract".to_string(),
            PatternType::Hybrid(a, b) => format!("hybrid:{}+{}", a.to_string(), b.to_string()),
        }
    }
}