//! Pattern retrieval optimization for fast access

use crate::integration::{PatternType, PatternContext, LearnedPattern};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Optimized pattern retrieval system with indexing
pub struct PatternRetrieval {
    type_index: Arc<RwLock<HashMap<PatternType, Vec<String>>>>,
    domain_index: Arc<RwLock<HashMap<String, Vec<String>>>>,
    complexity_index: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl PatternRetrieval {
    pub fn new() -> Self {
        Self {
            type_index: Arc::new(RwLock::new(HashMap::new())),
            domain_index: Arc::new(RwLock::new(HashMap::new())),
            complexity_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Index a pattern for fast retrieval
    pub fn index_pattern(&self, pattern: &LearnedPattern) {
        let pattern_id = pattern.id.clone();
        
        // Index by type
        self.type_index.write()
            .entry(pattern.pattern_type.clone())
            .or_insert_with(Vec::new)
            .push(pattern_id.clone());
        
        // Index by domain
        self.domain_index.write()
            .entry(pattern.context.domain.clone())
            .or_insert_with(Vec::new)
            .push(pattern_id.clone());
        
        // Index by complexity
        let complexity_key = format!("{:?}", pattern.context.complexity);
        self.complexity_index.write()
            .entry(complexity_key)
            .or_insert_with(Vec::new)
            .push(pattern_id);
    }
    
    /// Get pattern IDs by type
    pub fn get_by_type(&self, pattern_type: &PatternType) -> Vec<String> {
        self.type_index.read()
            .get(pattern_type)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Get pattern IDs by domain
    pub fn get_by_domain(&self, domain: &str) -> Vec<String> {
        self.domain_index.read()
            .get(domain)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Get intersection of multiple criteria
    pub fn get_by_criteria(&self, context: &PatternContext) -> Vec<String> {
        let domain_patterns = self.get_by_domain(&context.domain);
        let complexity_key = format!("{:?}", context.complexity);
        let complexity_patterns = self.complexity_index.read()
            .get(&complexity_key)
            .cloned()
            .unwrap_or_default();
        
        // Find intersection
        domain_patterns.into_iter()
            .filter(|id| complexity_patterns.contains(id))
            .collect()
    }
}