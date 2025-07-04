use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

// Import from our existing modules
use crate::memory::{MemoryBank, MemoryModule, MemoryConfig, PersistentMemoryBuilder};
use crate::attention::MultiHeadAttention;
use crate::nn::{NeuralNetwork, Layer, Activation};
use crate::FrameworkConfig;

#[derive(Debug, Serialize, Deserialize)]
#[napi(object)]
pub struct NeuralConfig {
    pub dimensions: Option<u32>,
    pub capacity: Option<u32>,
    pub threshold: Option<f64>,
    pub persist_path: Option<String>,
    pub num_heads: Option<u32>,
    pub hidden_dim: Option<u32>,
    pub dropout_rate: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
#[napi(object)]
pub struct MemoryEntry {
    pub key: String,
    pub content: String,
    pub embedding: Option<Vec<f64>>,
    pub timestamp: Option<i64>,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[napi(object)]
pub struct SearchResult {
    pub key: String,
    pub content: String,
    pub similarity: f64,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[napi(object)]
pub struct MemoryStats {
    pub total_entries: u32,
    pub cache_size: u32,
    pub hit_rate: f64,
    pub avg_retrieval_time: f64,
}

struct InternalMemoryEntry {
    key: String,
    content: String,
    embedding: ndarray::Array1<f32>,
    metadata: HashMap<String, String>,
    timestamp: chrono::DateTime<chrono::Utc>,
}

#[napi]
pub struct NeuralMemorySystem {
    memory_module: Arc<RwLock<PersistentMemoryModule>>,
    attention: Arc<RwLock<MultiHeadAttention>>,
    entries: Arc<RwLock<HashMap<String, InternalMemoryEntry>>>,
    config: FrameworkConfig,
}

use crate::memory::PersistentMemoryModule;

#[napi]
impl NeuralMemorySystem {
    #[napi(constructor)]
    pub fn new(config: Option<NeuralConfig>) -> Result<Self> {
        let cfg = config.unwrap_or(NeuralConfig {
            dimensions: Some(768),
            capacity: Some(10000),
            threshold: Some(0.7),
            persist_path: Some("./neural_memory.db".to_string()),
            num_heads: Some(12),
            hidden_dim: Some(3072),
            dropout_rate: Some(0.1),
        });

        let framework_config = FrameworkConfig {
            memory_size: cfg.capacity.unwrap_or(10000) as usize,
            embedding_dim: cfg.dimensions.unwrap_or(768) as usize,
            num_heads: cfg.num_heads.unwrap_or(12) as usize,
            hidden_dim: cfg.hidden_dim.unwrap_or(3072) as usize,
            dropout_rate: cfg.dropout_rate.unwrap_or(0.1) as f32,
            learning_rate: 0.001,
            batch_size: 32,
            use_simd: true,
            use_gpu: false,
        };

        let memory_config = MemoryConfig {
            memory_size: framework_config.memory_size,
            key_size: framework_config.embedding_dim,
            value_size: framework_config.embedding_dim,
            num_heads: framework_config.num_heads,
            dropout_rate: framework_config.dropout_rate,
        };

        let persist_path = cfg.persist_path.unwrap_or("./neural_memory.db".to_string());
        
        let memory_module = PersistentMemoryBuilder::new()
            .with_memory_size(memory_config.memory_size)
            .with_key_size(memory_config.key_size)
            .with_value_size(memory_config.value_size)
            .with_num_heads(memory_config.num_heads)
            .with_dropout_rate(memory_config.dropout_rate)
            .with_file_path(persist_path)
            .build()
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        let attention = MultiHeadAttention::new(
            framework_config.embedding_dim,
            framework_config.num_heads,
            framework_config.dropout_rate,
        );

        Ok(Self {
            memory_module: Arc::new(RwLock::new(memory_module)),
            attention: Arc::new(RwLock::new(attention)),
            entries: Arc::new(RwLock::new(HashMap::new())),
            config: framework_config,
        })
    }

    #[napi]
    pub async fn store(
        &self,
        key: String,
        content: String,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()> {
        // Simple embedding generation (in production, use a proper encoder)
        let embedding = ndarray::Array1::<f32>::from_vec(
            content.chars()
                .take(self.config.embedding_dim)
                .enumerate()
                .map(|(i, c)| (c as u32 as f32 + i as f32) / 1000.0)
                .chain(std::iter::repeat(0.0))
                .take(self.config.embedding_dim)
                .collect()
        );

        let entry = InternalMemoryEntry {
            key: key.clone(),
            content: content.clone(),
            embedding: embedding.clone(),
            metadata: metadata.unwrap_or_default(),
            timestamp: chrono::Utc::now(),
        };

        let mut entries = self.entries.write().await;
        entries.insert(key.clone(), entry);

        // Store in persistent memory module
        let query = ndarray::Array2::from_shape_vec(
            (1, self.config.embedding_dim),
            embedding.to_vec()
        ).map_err(|e| napi::Error::from_reason(e.to_string()))?;

        let mut memory_module = self.memory_module.write().await;
        memory_module.write(query.clone(), query)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        Ok(())
    }

    #[napi]
    pub async fn retrieve(&self, key: String) -> Result<Option<MemoryEntry>> {
        let entries = self.entries.read().await;
        
        match entries.get(&key) {
            Some(entry) => Ok(Some(MemoryEntry {
                key: entry.key.clone(),
                content: entry.content.clone(),
                embedding: Some(entry.embedding.to_vec().iter().map(|&x| x as f64).collect()),
                timestamp: Some(entry.timestamp.timestamp()),
                metadata: Some(entry.metadata.clone()),
            })),
            None => Ok(None),
        }
    }

    #[napi]
    pub async fn search(&self, query: String, limit: Option<u32>) -> Result<Vec<SearchResult>> {
        let query_embedding = ndarray::Array1::<f32>::from_vec(
            query.chars()
                .take(self.config.embedding_dim)
                .enumerate()
                .map(|(i, c)| (c as u32 as f32 + i as f32) / 1000.0)
                .chain(std::iter::repeat(0.0))
                .take(self.config.embedding_dim)
                .collect()
        );

        let entries = self.entries.read().await;
        let mut similarities: Vec<(String, f64)> = Vec::new();

        for (key, entry) in entries.iter() {
            // Compute cosine similarity
            let dot_product: f32 = query_embedding.dot(&entry.embedding);
            let query_norm: f32 = query_embedding.dot(&query_embedding).sqrt();
            let entry_norm: f32 = entry.embedding.dot(&entry.embedding).sqrt();
            let similarity = if query_norm > 0.0 && entry_norm > 0.0 {
                (dot_product / (query_norm * entry_norm)) as f64
            } else {
                0.0
            };
            
            similarities.push((key.clone(), similarity));
        }

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let limit = limit.unwrap_or(10) as usize;
        let mut results = Vec::new();
        
        for (key, similarity) in similarities.into_iter().take(limit) {
            if let Some(entry) = entries.get(&key) {
                results.push(SearchResult {
                    key: entry.key.clone(),
                    content: entry.content.clone(),
                    similarity,
                    metadata: Some(entry.metadata.clone()),
                });
            }
        }

        Ok(results)
    }

    #[napi]
    pub async fn get_stats(&self) -> Result<MemoryStats> {
        let entries = self.entries.read().await;
        
        Ok(MemoryStats {
            total_entries: entries.len() as u32,
            cache_size: entries.len() as u32,
            hit_rate: 0.0, // Would need to track this
            avg_retrieval_time: 0.0, // Would need to track this
        })
    }

    #[napi]
    pub async fn clear(&self) -> Result<()> {
        let mut entries = self.entries.write().await;
        entries.clear();
        
        // Clear persistent storage
        let mut memory_module = self.memory_module.write().await;
        memory_module.clear()
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        
        Ok(())
    }

    #[napi]
    pub async fn delete(&self, key: String) -> Result<bool> {
        let mut entries = self.entries.write().await;
        Ok(entries.remove(&key).is_some())
    }

    #[napi]
    pub async fn list_keys(&self) -> Result<Vec<String>> {
        let entries = self.entries.read().await;
        Ok(entries.keys().cloned().collect())
    }

    #[napi]
    pub async fn export_to_file(&self, file_path: String) -> Result<()> {
        let entries = self.entries.read().await;
        let json_entries: Vec<MemoryEntry> = entries.values().map(|entry| {
            MemoryEntry {
                key: entry.key.clone(),
                content: entry.content.clone(),
                embedding: Some(entry.embedding.to_vec().iter().map(|&x| x as f64).collect()),
                timestamp: Some(entry.timestamp.timestamp()),
                metadata: Some(entry.metadata.clone()),
            }
        }).collect();

        let json = serde_json::to_string_pretty(&json_entries)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        
        tokio::fs::write(&file_path, json)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        
        Ok(())
    }

    #[napi]
    pub async fn import_from_file(&self, file_path: String) -> Result<()> {
        let json = tokio::fs::read_to_string(&file_path)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        
        let json_entries: Vec<MemoryEntry> = serde_json::from_str(&json)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        let mut entries = self.entries.write().await;
        entries.clear();

        for entry in json_entries {
            let embedding = entry.embedding
                .unwrap_or_else(|| vec![0.0; self.config.embedding_dim])
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<f32>>();
            
            let internal_entry = InternalMemoryEntry {
                key: entry.key.clone(),
                content: entry.content,
                embedding: ndarray::Array1::from_vec(embedding),
                metadata: entry.metadata.unwrap_or_default(),
                timestamp: chrono::DateTime::from_timestamp(
                    entry.timestamp.unwrap_or(0), 
                    0
                ).unwrap_or_else(chrono::Utc::now),
            };
            
            entries.insert(entry.key, internal_entry);
        }

        Ok(())
    }
}