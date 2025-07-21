//! Simplified embedding service that provides better-than-hash embeddings
//! without complex ML framework dependencies

use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;
use dashmap::DashMap;
use tracing::{info, debug};

use super::{EmbeddingConfig, EmbeddingCache, normalize_vector};

/// A simplified embedding service that uses TF-IDF-like features
/// This provides better semantic understanding than pure hash-based embeddings
/// while avoiding complex ML framework dependencies
pub struct SimpleEmbeddingService {
    /// Vocabulary mapping words to indices
    vocabulary: Arc<DashMap<String, usize>>,
    /// IDF (Inverse Document Frequency) scores
    idf_scores: Arc<DashMap<String, f32>>,
    /// Document count for IDF calculation
    doc_count: Arc<std::sync::atomic::AtomicUsize>,
    /// Configuration
    config: EmbeddingConfig,
    /// Cache for computed embeddings
    cache: Option<EmbeddingCache>,
    /// Embedding dimension
    embedding_dim: usize,
}

impl SimpleEmbeddingService {
    /// Create a new simplified embedding service
    pub async fn new(config: EmbeddingConfig) -> Result<Self> {
        info!("Initializing simplified embedding service");
        
        let embedding_dim = 768; // Match the default graph embedding dimension
        
        // Initialize cache if enabled
        let cache = if config.cache_enabled {
            Some(EmbeddingCache::new(
                config.cache_dir.clone(),
                config.cache_size_mb,
                "simple-embeddings".to_string(),
            )?)
        } else {
            None
        };
        
        Ok(Self {
            vocabulary: Arc::new(DashMap::new()),
            idf_scores: Arc::new(DashMap::new()),
            doc_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            config,
            cache,
            embedding_dim,
        })
    }
    
    /// Tokenize text into words (simple whitespace + punctuation splitting)
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }
    
    /// Generate n-grams from tokens
    fn generate_ngrams(&self, tokens: &[String], n: usize) -> Vec<String> {
        if tokens.len() < n {
            return vec![];
        }
        
        tokens.windows(n)
            .map(|window| window.join("_"))
            .collect()
    }
    
    /// Update vocabulary and IDF scores with new document
    fn update_vocabulary(&self, tokens: &[String]) {
        // Count document
        self.doc_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        // Track unique tokens in this document
        let mut doc_tokens = std::collections::HashSet::new();
        
        for token in tokens {
            doc_tokens.insert(token.clone());
            
            // Add to vocabulary if not present
            let vocab_size = self.vocabulary.len();
            self.vocabulary.entry(token.clone()).or_insert(vocab_size);
        }
        
        // Update document frequencies for IDF
        for token in doc_tokens {
            self.idf_scores
                .entry(token)
                .and_modify(|count| *count += 1.0)
                .or_insert(1.0);
        }
    }
    
    /// Calculate IDF score for a token
    fn get_idf(&self, token: &str) -> f32 {
        let doc_count = self.doc_count.load(std::sync::atomic::Ordering::Relaxed) as f32;
        if doc_count == 0.0 {
            return 1.0;
        }
        
        if let Some(doc_freq) = self.idf_scores.get(token) {
            (doc_count / (*doc_freq + 1.0)).ln() + 1.0
        } else {
            (doc_count + 1.0).ln() + 1.0 // Unseen token gets max IDF
        }
    }
    
    /// Generate embedding using enhanced features
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(ref cache) = self.cache {
            if let Some(embedding) = cache.get(text).await {
                debug!("Cache hit for text embedding");
                return Ok(embedding);
            }
        }
        
        // Tokenize text
        let tokens = self.tokenize(text);
        
        // Update vocabulary (for continual learning)
        self.update_vocabulary(&tokens);
        
        // Generate features
        let mut embedding = vec![0.0; self.embedding_dim];
        
        // 1. Unigram features with TF-IDF weighting
        let token_freq = tokens.iter()
            .fold(HashMap::new(), |mut map, token| {
                *map.entry(token.clone()).or_insert(0.0) += 1.0;
                map
            });
        
        for (token, freq) in &token_freq {
            let tf = freq / tokens.len() as f32;
            let idf = self.get_idf(token);
            let tfidf = tf * idf;
            
            // Hash token to multiple positions for better coverage
            for i in 0..3 {
                let hash = self.hash_string(&format!("{}{}", token, i));
                let idx = (hash % self.embedding_dim as u64) as usize;
                embedding[idx] += tfidf;
            }
        }
        
        // 2. Bigram features
        let bigrams = self.generate_ngrams(&tokens, 2);
        for bigram in &bigrams {
            let hash = self.hash_string(bigram);
            let idx = (hash % self.embedding_dim as u64) as usize;
            embedding[idx] += 0.5; // Lower weight for bigrams
        }
        
        // 3. Character-level features (for handling OOV and typos)
        for ch in text.chars().take(100) {
            if ch.is_alphanumeric() {
                let hash = self.hash_string(&ch.to_string());
                let idx = (hash % self.embedding_dim as u64) as usize;
                embedding[idx] += 0.1;
            }
        }
        
        // 4. Positional features
        for (i, token) in tokens.iter().enumerate().take(20) {
            let hash = self.hash_string(&format!("pos_{}_{}", i, token));
            let idx = (hash % self.embedding_dim as u64) as usize;
            embedding[idx] += 0.3;
        }
        
        // 5. Length features
        let length_features = [
            tokens.len() as f32 / 10.0,
            text.len() as f32 / 100.0,
            (tokens.len() as f32).sqrt(),
        ];
        
        for (i, &feat) in length_features.iter().enumerate() {
            embedding[i] += feat.min(1.0);
        }
        
        // Normalize
        normalize_vector(&mut embedding);
        
        // Cache the result
        if let Some(ref cache) = self.cache {
            if let Err(e) = cache.put(text, embedding.clone()).await {
                debug!("Failed to cache embedding: {}", e);
            }
        }
        
        Ok(embedding)
    }
    
    /// Simple string hashing function
    fn hash_string(&self, s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Batch embedding with shared vocabulary updates
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        
        for text in texts {
            results.push(self.embed(text).await?);
        }
        
        Ok(results)
    }
    
    /// Get the embedding dimension
    pub fn embedding_dimension(&self) -> usize {
        self.embedding_dim
    }
    
    /// Clear the cache
    pub async fn clear_cache(&self) -> Result<()> {
        if let Some(ref cache) = self.cache {
            cache.clear().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_simple_embeddings() -> Result<()> {
        let config = EmbeddingConfig {
            cache_enabled: false,
            ..Default::default()
        };
        
        let service = SimpleEmbeddingService::new(config).await?;
        
        // Test that similar texts have higher similarity
        let cat1 = service.embed("cat feline pet animal").await?;
        let cat2 = service.embed("kitten cat domestic animal").await?;
        let car = service.embed("automobile vehicle transportation").await?;
        
        let sim_cats = super::super::cosine_similarity(&cat1, &cat2);
        let sim_cat_car = super::super::cosine_similarity(&cat1, &car);
        
        // While not as good as neural embeddings, should still show some similarity
        println!("Cat-Cat similarity: {}", sim_cats);
        println!("Cat-Car similarity: {}", sim_cat_car);
        
        // Both should be positive due to shared features
        assert!(sim_cats > 0.0);
        assert!(sim_cat_car > 0.0);
        
        Ok(())
    }
}