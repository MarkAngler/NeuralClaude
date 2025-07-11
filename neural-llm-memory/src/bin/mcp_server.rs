//! MCP server for neural LLM memory framework using rmcp SDK
//! Migrated from mcp_server_simple.rs JSON-RPC implementation

use anyhow::Result;
use neural_llm_memory::{PersistentMemoryModule, PersistentMemoryBuilder, MemoryConfig};
use neural_llm_memory::adaptive::{AdaptiveMemoryModule, AdaptiveConfig};
use neural_llm_memory::consciousness::{ConsciousnessCore, ConsciousnessConfig, ConsciousInput, ContentType, IntrospectiveFocus, ReflectionFocus};
use ndarray::Array2;
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};
use schemars::JsonSchema;

// mod mcp_server_config;

// Import rmcp types
use rmcp::{
    ServerHandler, ServiceExt,
    model::*,
    service::{RequestContext, RoleServer},
    transport::stdio,
};

// Helper to convert schema
fn to_input_schema<T: schemars::JsonSchema>() -> Arc<serde_json::Map<String, Value>> {
    let schema = schemars::schema_for!(T);
    let json_schema = serde_json::to_value(schema).unwrap_or(json!({}));
    Arc::new(json_schema.as_object().cloned().unwrap_or_default())
}

// Tool parameter structures with JsonSchema for automatic schema generation
#[derive(Debug, Deserialize, JsonSchema)]
struct StoreMemoryParams {
    #[schemars(description = "Unique key for the memory")]
    key: String,
    #[schemars(description = "Content to store")]
    content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RetrieveMemoryParams {
    #[schemars(description = "Key of the memory to retrieve")]
    key: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct UpdateMemoryParams {
    #[schemars(description = "Key of the memory to update")]
    key: String,
    #[schemars(description = "New content to store")]
    content: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DeleteMemoryParams {
    #[schemars(description = "Key of the memory to delete")]
    key: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchMemoryParams {
    #[schemars(description = "Query to search for")]
    query: String,
    #[schemars(description = "Maximum number of results")]
    #[serde(default = "default_search_limit")]
    limit: usize,
}

fn default_search_limit() -> usize { 5 }

#[derive(Debug, Deserialize, JsonSchema)]
struct MemoryStatsParams {}

#[derive(Debug, Deserialize, JsonSchema)]
struct AdaptiveStatusParams {
    #[schemars(description = "Include detailed metrics")]
    #[serde(default)]
    verbose: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AdaptiveTrainParams {
    #[schemars(description = "Number of evolution generations")]
    #[serde(default = "default_generations")]
    generations: usize,
    #[schemars(description = "Force evolution even if not due")]
    #[serde(default)]
    force: bool,
}

fn default_generations() -> usize { 5 }

#[derive(Debug, Deserialize, JsonSchema)]
struct AdaptiveInsightsParams {}

#[derive(Debug, Deserialize, JsonSchema)]
struct AdaptiveConfigParams {
    #[schemars(description = "Optimization objectives with weights")]
    objectives: Option<serde_json::Map<String, Value>>,
    #[schemars(description = "Enable/disable adaptive features")]
    enabled: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ProvideFeedbackParams {
    #[schemars(description = "ID of the operation to provide feedback for")]
    operation_id: String,
    #[schemars(description = "Whether the operation helped answer the user")]
    success: bool,
    #[schemars(description = "Relevance score between 0.0 and 1.0")]
    score: f32,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ConsciousnessStatusParams {
    #[schemars(description = "Include detailed consciousness metrics")]
    #[serde(default)]
    detailed: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ConsciousnessProcessParams {
    #[schemars(description = "Content to process")]
    content: String,
    #[schemars(description = "Content type")]
    content_type: String,
    #[schemars(description = "Activation level (0.0 to 1.0)")]
    #[serde(default = "default_activation")]
    activation: f32,
    #[schemars(description = "Attention weight (0.0 to 1.0)")]
    #[serde(default = "default_attention")]
    attention_weight: f32,
}

fn default_activation() -> f32 { 0.7 }
fn default_attention() -> f32 { 0.5 }

#[derive(Debug, Deserialize, JsonSchema)]
struct ConsciousnessReflectParams {
    #[schemars(description = "Focus of reflection")]
    focus: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ConsciousnessIntrospectParams {
    #[schemars(description = "Focus of introspection")]
    focus: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ConsciousnessInsightParams {
    #[schemars(description = "Domain for creative insight")]
    domain: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ConsciousnessEvolveParams {
    #[schemars(description = "Target capability to evolve")]
    target_capability: String,
}

// Tool result wrapper for consistent output
#[derive(Debug)]
struct ToolResult(Value);

// Error type for tools
#[derive(Debug)]
struct ToolError(String);

// Main server state
#[derive(Clone)]
struct NeuralMemoryServer {
    memory_module: Arc<Mutex<PersistentMemoryModule>>,
    adaptive_module: Option<Arc<Mutex<AdaptiveMemoryModule>>>,
    adaptive_enabled: bool,
    consciousness_core: Option<Arc<Mutex<ConsciousnessCore>>>,
    consciousness_enabled: bool,
}

impl NeuralMemoryServer {
    async fn new() -> Result<Self> {
        let adaptive_enabled = std::env::var("NEURAL_MCP_ADAPTIVE")
            .unwrap_or_else(|_| "true".to_string())
            .parse::<bool>()
            .unwrap_or(true);
        
        let consciousness_enabled = std::env::var("NEURAL_MCP_CONSCIOUSNESS")
            .unwrap_or_else(|_| "true".to_string())
            .parse::<bool>()
            .unwrap_or(true);
        
        let auto_recover = std::env::var("NEURAL_MCP_AUTO_RECOVER")
            .unwrap_or_else(|_| "true".to_string())
            .parse::<bool>()
            .unwrap_or(true);
        
        let base_config = MemoryConfig::default();
        
        // Create adaptive module if enabled
        let adaptive_module = if adaptive_enabled {
            // Use new_with_persistence for startup recovery
            let module = if auto_recover {
                eprintln!("🔄 Attempting to recover adaptive memory state...");
                match AdaptiveMemoryModule::new_with_persistence(base_config.clone(), true).await {
                    Ok(module) => {
                        eprintln!("✅ Successfully recovered adaptive memory state");
                        module
                    }
                    Err(e) => {
                        eprintln!("⚠️  Failed to recover state: {}. Starting fresh.", e);
                        let adaptive_config = AdaptiveConfig::default();
                        AdaptiveMemoryModule::with_config(base_config.clone(), adaptive_config)
                            .await
                            .map_err(|e| anyhow::anyhow!("Failed to create adaptive module: {}", e))?
                    }
                }
            } else {
                eprintln!("📝 Starting with fresh adaptive memory (recovery disabled)");
                let adaptive_config = AdaptiveConfig::default();
                AdaptiveMemoryModule::with_config(base_config.clone(), adaptive_config)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to create adaptive module: {}", e))?
            };
            Some(Arc::new(Mutex::new(module)))
        } else {
            None
        };
        
        // Create consciousness core if enabled
        let consciousness_core = if consciousness_enabled {
            eprintln!("🧠 Initializing consciousness core...");
            let consciousness_config = ConsciousnessConfig::default();
            let core = ConsciousnessCore::with_config(consciousness_config);
            eprintln!("✅ Consciousness core initialized successfully");
            Some(Arc::new(Mutex::new(core)))
        } else {
            None
        };
        
        // Always create regular memory module for compatibility
        let memory_module = PersistentMemoryBuilder::new()
            .storage_path("./neural_memory_data")
            .auto_save_interval(60)
            .memory_config(base_config)
            .build()
            .await?;
        
        Ok(Self {
            memory_module: Arc::new(Mutex::new(memory_module)),
            adaptive_module,
            adaptive_enabled,
            consciousness_core,
            consciousness_enabled,
        })
    }
    
    // Tool implementations
    async fn store_memory(
        &self,
        params: StoreMemoryParams
    ) -> Result<ToolResult, ToolError> {
        if self.adaptive_enabled {
            // Use adaptive module
            let module = self.adaptive_module.as_ref().unwrap().lock().await;
            let operation_id = module.store(&params.key, &params.content).await
                .map_err(|e| ToolError(e.to_string()))?;
            
            Ok(ToolResult(json!({
                "status": "stored",
                "key": params.key,
                "operation_id": operation_id
            })))
        } else {
            // Use regular module with simple embeddings
            let embedding = Array2::from_shape_vec((1, 768), vec![0.1; 768])
                .map_err(|e| ToolError(e.to_string()))?;
            
            let memory = self.memory_module.lock().await;
            let memory_key = memory.store_memory(params.content.clone(), embedding).await
                .map_err(|e| ToolError(e.to_string()))?;
            
            Ok(ToolResult(json!({
                "status": "stored",
                "key": params.key,
                "memory_id": memory_key.id
            })))
        }
    }
    
    async fn retrieve_memory(
        &self,
        params: RetrieveMemoryParams
    ) -> Result<ToolResult, ToolError> {
        if self.adaptive_enabled {
            // Use adaptive module
            let module = self.adaptive_module.as_ref().unwrap().lock().await;
            match module.retrieve(&params.key).await {
                Ok((Some(content), operation_id)) => Ok(ToolResult(json!({
                    "key": params.key,
                    "content": content,
                    "found": true,
                    "operation_id": operation_id
                }))),
                Ok((None, operation_id)) => Ok(ToolResult(json!({
                    "key": params.key,
                    "content": null,
                    "found": false,
                    "operation_id": operation_id
                }))),
                Err(e) => Err(ToolError(e.to_string())),
            }
        } else {
            // Placeholder for regular module
            Ok(ToolResult(json!({
                "key": params.key,
                "content": format!("Memory content for key: {}", params.key),
                "found": false
            })))
        }
    }
    
    async fn search_memory(
        &self,
        params: SearchMemoryParams
    ) -> Result<ToolResult, ToolError> {
        if self.adaptive_enabled {
            // Use adaptive module
            let module = self.adaptive_module.as_ref().unwrap().lock().await;
            let (results, operation_id) = module.search(&params.query, params.limit).await
                .map_err(|e| ToolError(e.to_string()))?;
            
            let matches: Vec<_> = results.into_iter()
                .map(|(key, score)| json!({
                    "key": key,
                    "score": score
                }))
                .collect();
            
            Ok(ToolResult(json!({
                "query": params.query,
                "matches": matches,
                "operation_id": operation_id
            })))
        } else {
            // Use regular module - simplified search
            Ok(ToolResult(json!({
                "query": params.query,
                "matches": []
            })))
        }
    }
    
    async fn update_memory(
        &self,
        params: UpdateMemoryParams
    ) -> Result<ToolResult, ToolError> {
        if self.adaptive_enabled {
            // Use adaptive module
            let module = self.adaptive_module.as_ref().unwrap().lock().await;
            match module.update(&params.key, &params.content).await {
                Ok(operation_id) => Ok(ToolResult(json!({
                    "status": "updated",
                    "key": params.key,
                    "operation_id": operation_id
                }))),
                Err(e) => Err(ToolError(e.to_string())),
            }
        } else {
            // Placeholder for regular module
            Err(ToolError("Update not supported in non-adaptive mode".to_string()))
        }
    }
    
    async fn delete_memory(
        &self,
        params: DeleteMemoryParams
    ) -> Result<ToolResult, ToolError> {
        if self.adaptive_enabled {
            // Use adaptive module
            let module = self.adaptive_module.as_ref().unwrap().lock().await;
            match module.delete(&params.key).await {
                Ok((deleted, operation_id)) => Ok(ToolResult(json!({
                    "status": if deleted { "deleted" } else { "not_found" },
                    "key": params.key,
                    "deleted": deleted,
                    "operation_id": operation_id
                }))),
                Err(e) => Err(ToolError(e.to_string())),
            }
        } else {
            // Placeholder for regular module
            Err(ToolError("Delete not supported in non-adaptive mode".to_string()))
        }
    }
    
    async fn memory_stats(
        &self,
        _params: MemoryStatsParams
    ) -> Result<ToolResult, ToolError> {
        if self.adaptive_enabled {
            let module = self.adaptive_module.as_ref().unwrap().lock().await;
            // Use get_stats method
            let stats = module.get_stats().await
                .map_err(|e| ToolError(e.to_string()))?;
            Ok(ToolResult(stats))
        } else {
            // For non-adaptive mode, return basic stats
            Ok(ToolResult(json!({
                "memory_size": 0,
                "total_accesses": 0,
                "cache_hits": 0,
                "cache_hit_rate": 0.0,
                "mode": "basic"
            })))
        }
    }
    
    async fn adaptive_status(
        &self,
        params: AdaptiveStatusParams
    ) -> Result<ToolResult, ToolError> {
        if !self.adaptive_enabled {
            return Err(ToolError("Adaptive learning not enabled".to_string()));
        }
        
        let module = self.adaptive_module.as_ref().unwrap().lock().await;
        let status = module.get_adaptive_status(params.verbose).await
            .map_err(|e| ToolError(e.to_string()))?;
        Ok(ToolResult(status))
    }
    
    async fn adaptive_train(
        &self,
        params: AdaptiveTrainParams
    ) -> Result<ToolResult, ToolError> {
        if !self.adaptive_enabled {
            return Err(ToolError("Adaptive learning not enabled".to_string()));
        }
        
        let module = self.adaptive_module.as_ref().unwrap().lock().await;
        let result = module.trigger_evolution(Some(params.generations), params.force).await
            .map_err(|e| ToolError(e.to_string()))?;
        Ok(ToolResult(result))
    }
    
    async fn adaptive_insights(
        &self,
        _params: AdaptiveInsightsParams
    ) -> Result<ToolResult, ToolError> {
        if !self.adaptive_enabled {
            return Err(ToolError("Adaptive learning not enabled".to_string()));
        }
        
        let module = self.adaptive_module.as_ref().unwrap().lock().await;
        let insights = module.get_insights().await
            .map_err(|e| ToolError(e.to_string()))?;
        Ok(ToolResult(insights))
    }
    
    async fn adaptive_config(
        &self,
        params: AdaptiveConfigParams
    ) -> Result<ToolResult, ToolError> {
        if !self.adaptive_enabled {
            return Err(ToolError("Adaptive learning not enabled".to_string()));
        }
        
        let module = self.adaptive_module.as_ref().unwrap().lock().await;
        let objectives = params.objectives.map(|obj| {
            obj.into_iter()
                .filter_map(|(k, v)| v.as_f64().map(|f| (k, f as f32)))
                .collect()
        });
        let result = module.update_config(objectives, params.enabled).await
            .map_err(|e| ToolError(e.to_string()))?;
        Ok(ToolResult(result))
    }
    
    async fn provide_feedback(
        &self,
        params: ProvideFeedbackParams
    ) -> Result<ToolResult, ToolError> {
        if !self.adaptive_enabled {
            return Err(ToolError("Adaptive learning not enabled".to_string()));
        }
        
        use neural_llm_memory::adaptive::feedback::OperationFeedback;
        
        let feedback = OperationFeedback::from_claude(
            params.operation_id,
            params.success,
            Some(params.score),
            None,  // No reason parameter anymore
            None,  // No usage_context parameter anymore
        );
        
        let module = self.adaptive_module.as_ref().unwrap().lock().await;
        let result = module.provide_feedback(feedback).await
            .map_err(|e| ToolError(e.to_string()))?;
        Ok(ToolResult(result))
    }
    
    // Consciousness tool implementations
    async fn consciousness_status(
        &self,
        params: ConsciousnessStatusParams
    ) -> Result<ToolResult, ToolError> {
        if !self.consciousness_enabled {
            return Err(ToolError("Consciousness not enabled".to_string()));
        }
        
        let consciousness = self.consciousness_core.as_ref().unwrap().lock().await;
        let state = consciousness.get_consciousness_state();
        
        let result = if params.detailed {
            json!({
                "consciousness_level": state.consciousness_level,
                "consciousness_focus": state.consciousness_focus,
                "attention_allocation": state.attention_allocation,
                "integration_coherence": state.integration_coherence,
                "temporal_continuity": state.temporal_continuity,
                "self_awareness_level": state.self_awareness_level,
                "metacognitive_confidence": state.metacognitive_confidence,
                "emotional_valence": state.emotional_valence,
                "timestamp": state.timestamp,
                "status": "active"
            })
        } else {
            json!({
                "consciousness_level": state.consciousness_level,
                "self_awareness_level": state.self_awareness_level,
                "status": "active"
            })
        };
        
        Ok(ToolResult(result))
    }
    
    async fn consciousness_process(
        &self,
        params: ConsciousnessProcessParams
    ) -> Result<ToolResult, ToolError> {
        if !self.consciousness_enabled {
            return Err(ToolError("Consciousness not enabled".to_string()));
        }
        
        let consciousness = self.consciousness_core.as_ref().unwrap().lock().await;
        
        // Parse content type
        let content_type = match params.content_type.as_str() {
            "episodic" => ContentType::EpisodicMemory,
            "metacognitive" => ContentType::MetacognitiveReflection,
            "emotional" => ContentType::EmotionalState,
            "abstract" => ContentType::AbstractReasoning,
            "self_awareness" => ContentType::SelfAwarenessInsight,
            "goal_planning" => ContentType::GoalDirectedPlanning,
            "creative" => ContentType::CreativeInsight,
            _ => ContentType::PerceptualInput,
        };
        
        // Create semantic embedding (simplified)
        let embedding = ndarray::Array1::from_vec(vec![0.1; 768]);
        
        let input = ConsciousInput {
            content_type,
            activation: params.activation,
            attention_weight: params.attention_weight,
            semantic_embedding: embedding,
            metadata: std::collections::HashMap::new(),
        };
        
        let output = consciousness.process_input(input);
        
        let result = json!({
            "consciousness_level": output.consciousness_level,
            "integration_coherence": output.integration_coherence,
            "self_awareness_level": output.self_awareness_level,
            "insights": output.insights.iter().map(|i| json!({
                "type": format!("{:?}", i.insight_type),
                "content": i.content,
                "confidence": i.confidence
            })).collect::<Vec<_>>(),
            "timestamp": output.timestamp
        });
        
        Ok(ToolResult(result))
    }
    
    async fn consciousness_reflect(
        &self,
        params: ConsciousnessReflectParams
    ) -> Result<ToolResult, ToolError> {
        if !self.consciousness_enabled {
            return Err(ToolError("Consciousness not enabled".to_string()));
        }
        
        let consciousness = self.consciousness_core.as_ref().unwrap().lock().await;
        
        // Parse reflection focus
        let focus = match params.focus.as_str() {
            "experience" => ReflectionFocus::ExperienceReflection,
            "state" => ReflectionFocus::StateReflection,
            "future" => ReflectionFocus::FutureReflection,
            "metacognitive" => ReflectionFocus::MetaCognitive,
            "values" => ReflectionFocus::ValueReflection,
            _ => ReflectionFocus::StateReflection,
        };
        
        let result = consciousness.reflect(focus);
        
        let response = json!({
            "focus": result.focus,
            "content": result.content,
            "insights": result.insights,
            "quality": result.quality,
            "timestamp": result.timestamp
        });
        
        Ok(ToolResult(response))
    }
    
    async fn consciousness_introspect(
        &self,
        params: ConsciousnessIntrospectParams
    ) -> Result<ToolResult, ToolError> {
        if !self.consciousness_enabled {
            return Err(ToolError("Consciousness not enabled".to_string()));
        }
        
        let consciousness = self.consciousness_core.as_ref().unwrap().lock().await;
        
        // Parse introspection focus
        let focus = match params.focus.as_str() {
            "cognitive" => IntrospectiveFocus::Cognitive,
            "emotional" => IntrospectiveFocus::Emotional,
            "behavioral" => IntrospectiveFocus::Behavioral,
            "motivational" => IntrospectiveFocus::Motivational,
            "axiological" => IntrospectiveFocus::Axiological,
            _ => IntrospectiveFocus::Cognitive,
        };
        
        let result = consciousness.introspect(focus);
        
        let response = json!({
            "focus": result.focus,
            "insight": result.insight,
            "confidence": result.confidence,
            "depth": result.depth,
            "timestamp": result.timestamp
        });
        
        Ok(ToolResult(response))
    }
    
    async fn consciousness_insight(
        &self,
        params: ConsciousnessInsightParams
    ) -> Result<ToolResult, ToolError> {
        if !self.consciousness_enabled {
            return Err(ToolError("Consciousness not enabled".to_string()));
        }
        
        let consciousness = self.consciousness_core.as_ref().unwrap().lock().await;
        
        let result = consciousness.generate_creative_insight(params.domain);
        
        let response = json!({
            "domain": result.domain,
            "insight": result.insight,
            "novelty": result.novelty,
            "usefulness": result.usefulness,
            "confidence": result.confidence,
            "timestamp": result.timestamp
        });
        
        Ok(ToolResult(response))
    }
    
    async fn consciousness_evolve(
        &self,
        params: ConsciousnessEvolveParams
    ) -> Result<ToolResult, ToolError> {
        if !self.consciousness_enabled {
            return Err(ToolError("Consciousness not enabled".to_string()));
        }
        
        let consciousness = self.consciousness_core.as_ref().unwrap().lock().await;
        
        let result = consciousness.evolve_self(params.target_capability);
        
        let response = json!({
            "target_capability": result.target_capability,
            "evolution_type": format!("{:?}", result.evolution_type),
            "success": result.success,
            "safety_status": {
                "is_safe": result.safety_status.is_safe,
                "violated_constraints": result.safety_status.violated_constraints,
                "safety_score": result.safety_status.safety_score
            },
            "improvements": result.improvements,
            "timestamp": result.timestamp
        });
        
        Ok(ToolResult(response))
    }
}

// Implement ServerHandler using the rmcp SDK
impl ServerHandler for NeuralMemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::default(),
            capabilities: ServerCapabilities {
                tools: Some(Default::default()),
                resources: Some(Default::default()),
                prompts: None,
                completions: None,
                experimental: None,
                logging: None,
            },
            server_info: Implementation {
                name: "neural-memory".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
            instructions: Some("Neural memory MCP server for Claude with adaptive learning".into()),
        }
    }
    
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>
    ) -> Result<ListToolsResult, rmcp::Error> {
        let mut tools = vec![
            Tool {
                name: "store_memory".into(),
                description: Some("Store content in neural memory with a key".into()),
                input_schema: to_input_schema::<StoreMemoryParams>(),
                annotations: None,
            },
            Tool {
                name: "retrieve_memory".into(),
                description: Some("Retrieve memory content by key".into()),
                input_schema: to_input_schema::<RetrieveMemoryParams>(),
                annotations: None,
            },
            Tool {
                name: "update_memory".into(),
                description: Some("Update existing memory content by key".into()),
                input_schema: to_input_schema::<UpdateMemoryParams>(),
                annotations: None,
            },
            Tool {
                name: "delete_memory".into(),
                description: Some("Delete memory content by key".into()),
                input_schema: to_input_schema::<DeleteMemoryParams>(),
                annotations: None,
            },
            Tool {
                name: "search_memory".into(),
                description: Some("Search memory for similar content".into()),
                input_schema: to_input_schema::<SearchMemoryParams>(),
                annotations: None,
            },
            Tool {
                name: "memory_stats".into(),
                description: Some("Get memory system statistics".into()),
                input_schema: to_input_schema::<MemoryStatsParams>(),
                annotations: None,
            },
        ];
        
        if self.adaptive_enabled {
            tools.extend(vec![
                Tool {
                    name: "adaptive_status".into(),
                    description: Some("Get current adaptive learning status and evolution metrics".into()),
                    input_schema: to_input_schema::<AdaptiveStatusParams>(),
                    annotations: None,
            },
                Tool {
                    name: "adaptive_train".into(),
                    description: Some("Manually trigger neural evolution to optimize memory performance".into()),
                    input_schema: to_input_schema::<AdaptiveTrainParams>(),
                    annotations: None,
            },
                Tool {
                    name: "adaptive_insights".into(),
                    description: Some("Get adaptive learning insights and optimization recommendations".into()),
                    input_schema: to_input_schema::<AdaptiveInsightsParams>(),
                    annotations: None,
            },
                Tool {
                    name: "adaptive_config".into(),
                    description: Some("Update adaptive learning configuration".into()),
                    input_schema: to_input_schema::<AdaptiveConfigParams>(),
                    annotations: None,
            },
                Tool {
                    name: "provide_feedback".into(),
                    description: Some("Provide feedback on memory operation results with success status and usefulness score (0-1)".into()),
                    input_schema: to_input_schema::<ProvideFeedbackParams>(),
                    annotations: None,
            },
            ]);
        }
        
        if self.consciousness_enabled {
            tools.extend(vec![
                Tool {
                    name: "consciousness_status".into(),
                    description: Some("Get current consciousness state and awareness metrics".into()),
                    input_schema: to_input_schema::<ConsciousnessStatusParams>(),
                    annotations: None,
                },
                Tool {
                    name: "consciousness_process".into(),
                    description: Some("Process input through consciousness system".into()),
                    input_schema: to_input_schema::<ConsciousnessProcessParams>(),
                    annotations: None,
                },
                Tool {
                    name: "consciousness_reflect".into(),
                    description: Some("Engage in reflective thinking on specified focus".into()),
                    input_schema: to_input_schema::<ConsciousnessReflectParams>(),
                    annotations: None,
                },
                Tool {
                    name: "consciousness_introspect".into(),
                    description: Some("Perform introspective analysis of internal states".into()),
                    input_schema: to_input_schema::<ConsciousnessIntrospectParams>(),
                    annotations: None,
                },
                Tool {
                    name: "consciousness_insight".into(),
                    description: Some("Generate creative insights in specified domain".into()),
                    input_schema: to_input_schema::<ConsciousnessInsightParams>(),
                    annotations: None,
                },
                Tool {
                    name: "consciousness_evolve".into(),
                    description: Some("Engage in self-directed evolution toward target capability".into()),
                    input_schema: to_input_schema::<ConsciousnessEvolveParams>(),
                    annotations: None,
                },
            ]);
        }
        
        Ok(ListToolsResult { tools, next_cursor: None })
    }
    
    async fn call_tool(
        &self,
        request: CallToolRequestParam,
        _context: RequestContext<RoleServer>
    ) -> Result<CallToolResult, rmcp::Error> {
        let params = Value::Object(request.arguments.clone().unwrap_or_default());
        
        // Debug logging for provide_feedback calls
        if request.name == "provide_feedback" {
            tracing::debug!("provide_feedback called with raw params: {:?}", params);
            eprintln!("🔍 provide_feedback raw params: {}", serde_json::to_string_pretty(&params).unwrap_or_default());
        }
        
        let result = match request.name.as_ref() {
            "store_memory" => {
                let params: StoreMemoryParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.store_memory(params).await
            }
            "retrieve_memory" => {
                let params: RetrieveMemoryParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.retrieve_memory(params).await
            }
            "update_memory" => {
                let params: UpdateMemoryParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.update_memory(params).await
            }
            "delete_memory" => {
                let params: DeleteMemoryParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.delete_memory(params).await
            }
            "search_memory" => {
                let params: SearchMemoryParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.search_memory(params).await
            }
            "memory_stats" => {
                let params: MemoryStatsParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.memory_stats(params).await
            }
            "adaptive_status" if self.adaptive_enabled => {
                let params: AdaptiveStatusParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.adaptive_status(params).await
            }
            "adaptive_train" if self.adaptive_enabled => {
                let params: AdaptiveTrainParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.adaptive_train(params).await
            }
            "adaptive_insights" if self.adaptive_enabled => {
                let params: AdaptiveInsightsParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.adaptive_insights(params).await
            }
            "adaptive_config" if self.adaptive_enabled => {
                let params: AdaptiveConfigParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.adaptive_config(params).await
            }
            "provide_feedback" if self.adaptive_enabled => {
                let params: ProvideFeedbackParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.provide_feedback(params).await
            }
            "consciousness_status" if self.consciousness_enabled => {
                let params: ConsciousnessStatusParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.consciousness_status(params).await
            }
            "consciousness_process" if self.consciousness_enabled => {
                let params: ConsciousnessProcessParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.consciousness_process(params).await
            }
            "consciousness_reflect" if self.consciousness_enabled => {
                let params: ConsciousnessReflectParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.consciousness_reflect(params).await
            }
            "consciousness_introspect" if self.consciousness_enabled => {
                let params: ConsciousnessIntrospectParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.consciousness_introspect(params).await
            }
            "consciousness_insight" if self.consciousness_enabled => {
                let params: ConsciousnessInsightParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.consciousness_insight(params).await
            }
            "consciousness_evolve" if self.consciousness_enabled => {
                let params: ConsciousnessEvolveParams = serde_json::from_value(params)
                    .map_err(|e| rmcp::Error::invalid_params(format!("Invalid params: {}", e), None))?;
                self.consciousness_evolve(params).await
            }
            _ => return Err(rmcp::Error::invalid_request("Unknown tool", None)),
        };
        
        match result {
            Ok(tool_result) => Ok(CallToolResult::success(vec![
                Content::text(tool_result.0.to_string())
            ])),
            Err(tool_error) => Ok(CallToolResult::error(vec![
                Content::text(format!("Error: {}", tool_error.0))
            ])),
        }
    }
    
    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>
    ) -> Result<ListResourcesResult, rmcp::Error> {
        let mut resources = vec![
            Resource::new(
                RawResource {
                    uri: "memory://stats".into(),
                    name: "Memory Statistics".into(),
                    description: Some("Current memory system statistics and performance metrics".into()),
                    mime_type: Some("application/json".into()),
                    size: None,
                },
                None
            ),
        ];
        
        if self.adaptive_enabled {
            resources.extend(vec![
                Resource::new(
                    RawResource {
                        uri: "memory://adaptive/status".into(),
                        name: "Adaptive Learning Status".into(),
                        description: Some("Current adaptive learning status and evolution metrics".into()),
                        mime_type: Some("application/json".into()),
                        size: None,
                    },
                    None
                ),
                Resource::new(
                    RawResource {
                        uri: "memory://adaptive/insights".into(),
                        name: "Learning Insights".into(),
                        description: Some("Adaptive learning insights and recommendations".into()),
                        mime_type: Some("application/json".into()),
                        size: None,
                    },
                    None
                ),
            ]);
        }
        
        if self.consciousness_enabled {
            resources.extend(vec![
                Resource::new(
                    RawResource {
                        uri: "consciousness://status".into(),
                        name: "Consciousness Status".into(),
                        description: Some("Current consciousness state and awareness metrics".into()),
                        mime_type: Some("application/json".into()),
                        size: None,
                    },
                    None
                ),
                Resource::new(
                    RawResource {
                        uri: "consciousness://introspection".into(),
                        name: "Consciousness Introspection".into(),
                        description: Some("Real-time introspective analysis of internal states".into()),
                        mime_type: Some("application/json".into()),
                        size: None,
                    },
                    None
                ),
            ]);
        }
        
        Ok(ListResourcesResult { resources, next_cursor: None })
    }
    
    async fn read_resource(
        &self,
        request: ReadResourceRequestParam,
        _context: RequestContext<RoleServer>
    ) -> Result<ReadResourceResult, rmcp::Error> {
        match request.uri.as_ref() {
            "memory://stats" => {
                let result = self.memory_stats(MemoryStatsParams {}).await
                    .map_err(|e| rmcp::Error::internal_error("Internal error", None))?;
                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(request.uri.clone(), result.0.to_string())],
                })
            }
            "memory://adaptive/status" if self.adaptive_enabled => {
                let result = self.adaptive_status(AdaptiveStatusParams { verbose: true }).await
                    .map_err(|e| rmcp::Error::internal_error("Internal error", None))?;
                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(request.uri.clone(), result.0.to_string())],
                })
            }
            "memory://adaptive/insights" if self.adaptive_enabled => {
                let result = self.adaptive_insights(AdaptiveInsightsParams {}).await
                    .map_err(|e| rmcp::Error::internal_error("Internal error", None))?;
                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(request.uri.clone(), result.0.to_string())],
                })
            }
            "consciousness://status" if self.consciousness_enabled => {
                let result = self.consciousness_status(ConsciousnessStatusParams { detailed: true }).await
                    .map_err(|e| rmcp::Error::internal_error("Internal error", None))?;
                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(request.uri.clone(), result.0.to_string())],
                })
            }
            "consciousness://introspection" if self.consciousness_enabled => {
                let result = self.consciousness_introspect(ConsciousnessIntrospectParams { focus: "cognitive".to_string() }).await
                    .map_err(|e| rmcp::Error::internal_error("Internal error", None))?;
                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(request.uri.clone(), result.0.to_string())],
                })
            }
            _ => Err(rmcp::Error::resource_not_found(format!("Unknown resource: {}", request.uri), None)),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Increase default max listeners to prevent warnings
    if std::env::var("NODE_OPTIONS").is_err() {
        std::env::set_var("NODE_OPTIONS", "--max-old-space-size=4096");
    }
    
    // Initialize logging to stderr to avoid interfering with stdio transport
    use tracing_subscriber::{self, EnvFilter};
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::DEBUG.into()))
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();
    
    tracing::info!("Starting neural-memory MCP server");
    
    // Initialize server
    let server = NeuralMemoryServer::new().await?;
    
    // Start auto-save background task if adaptive is enabled
    if server.adaptive_enabled {
        if let Some(adaptive_module) = &server.adaptive_module {
            let auto_save_interval = std::env::var("NEURAL_MCP_AUTO_SAVE_INTERVAL")
                .unwrap_or_else(|_| "300".to_string())
                .parse::<u64>()
                .unwrap_or(300); // Default to 5 minutes
            
            eprintln!("🔄 Starting auto-save task (interval: {} seconds)", auto_save_interval);
            
            let module_clone = Arc::clone(adaptive_module);
            tokio::spawn(async move {
                use tokio::time::{interval, Duration};
                
                let mut interval = interval(Duration::from_secs(auto_save_interval));
                
                loop {
                    interval.tick().await;
                    
                    let module = module_clone.lock().await;
                    if let Err(e) = module.auto_save().await {
                        eprintln!("❌ Auto-save failed: {}", e);
                    } else {
                        eprintln!("💾 Auto-saved adaptive module state");
                    }
                }
            });
        }
    }
    
    // Only print to stderr AFTER server is ready
    eprintln!("🚀 neural-memory MCP server ready (using rmcp SDK)");
    eprintln!("📊 Configuration:");
    eprintln!("  - Adaptive learning: {}", if server.adaptive_enabled { "enabled" } else { "disabled" });
    eprintln!("  - Consciousness: {}", if server.consciousness_enabled { "enabled" } else { "disabled" });
    eprintln!("  - Auto-recovery: {}", std::env::var("NEURAL_MCP_AUTO_RECOVER").unwrap_or_else(|_| "true".to_string()));
    eprintln!("  - Auto-save interval: {} seconds", std::env::var("NEURAL_MCP_AUTO_SAVE_INTERVAL").unwrap_or_else(|_| "300".to_string()));
    
    // Clone server for shutdown handling
    let server_for_shutdown = server.clone();
    
    // Start the server using rmcp's serve method with stdio transport
    let server_peer = server.serve(stdio()).await
        .inspect_err(|e| {
            tracing::error!("serving error: {:?}", e);
        })?;
    
    eprintln!("Server connected and running. Waiting for shutdown...");
    
    // Keep the server running
    let shutdown_reason = server_peer.waiting().await?;
    eprintln!("Server shut down. Reason: {:?}", shutdown_reason);
    
    // Shutdown and save before exit
    eprintln!("💾 Saving memories before shutdown...");
    let memory = server_for_shutdown.memory_module.lock().await;
    memory.shutdown().await.ok();
    
    if let Some(adaptive) = &server_for_shutdown.adaptive_module {
        let module = adaptive.lock().await;
        
        // Save complete network state with timestamped checkpoint
        eprintln!("🧠 Saving neural network weights and state...");
        if let Err(e) = module.save_shutdown_checkpoint().await {
            eprintln!("❌ Failed to save shutdown checkpoint: {}", e);
            
            // Fall back to basic state save
            if let Err(e) = module.save_state("./adaptive_memory_data").await {
                eprintln!("❌ Failed to save adaptive state: {}", e);
            } else {
                eprintln!("✅ Saved basic adaptive state (without full checkpoint)");
            }
        } else {
            eprintln!("✅ Saved complete neural network checkpoint with weights");
            eprintln!("📁 Checkpoint location: ./adaptive_memory_data/network_checkpoints/");
            eprintln!("🔗 Latest checkpoint symlink: ./adaptive_memory_data/network_checkpoints/latest.bin");
        }
    }
    
    Ok(())
}