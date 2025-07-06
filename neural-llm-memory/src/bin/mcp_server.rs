//! MCP server for neural LLM memory framework using rmcp SDK

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use neural_llm_memory::{PersistentMemoryModule, PersistentMemoryBuilder, MemoryConfig};
use neural_llm_memory::adaptive::{AdaptiveMemoryModule, AdaptiveConfig};
use ndarray::Array2;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;
use schemars::JsonSchema;

// Import rmcp types and macros
use rmcp::{
    ServerHandler, ServiceExt,
    model::*,
    service::RequestContext,
    error::Error as McpError,
};
use rmcp::{tool, tool_box};
use async_trait::async_trait;

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
    generations: Option<usize>,
    #[schemars(description = "Force training even with few samples")]
    #[serde(default)]
    force: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AdaptiveInsightsParams {}

#[derive(Debug, Deserialize, JsonSchema)]
struct AdaptiveConfigParams {
    #[schemars(description = "Optimization objectives with weights")]
    objectives: Option<HashMap<String, f32>>,
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
    score: Option<f32>,
    #[schemars(description = "Why the operation succeeded or failed")]
    reason: Option<String>,
    #[schemars(description = "How the result was used (or not used)")]
    usage_context: Option<String>,
}

// Result types for tool responses
#[derive(Debug)]
struct ToolResult(Value);

impl rmcp::IntoContents for ToolResult {
    fn into_contents(self) -> Contents {
        Contents::Text(self.0.to_string())
    }
}

// Error type for tools
#[derive(Debug)]
struct ToolError(String);

impl rmcp::IntoContents for ToolError {
    fn into_contents(self) -> Contents {
        Contents::Error(ErrorContents {
            code: "TOOL_ERROR".to_string(),
            message: self.0,
            data: None,
        })
    }
}

// Main server state
#[derive(Clone)]
struct NeuralMemoryServer {
    memory_module: Arc<Mutex<PersistentMemoryModule>>,
    adaptive_module: Option<Arc<Mutex<AdaptiveMemoryModule>>>,
    runtime: Arc<tokio::runtime::Runtime>,
    adaptive_enabled: bool,
}

// Implement tools using the tool_box macro
#[tool_box]
impl NeuralMemoryServer {
    #[tool(description = "Store content in neural memory with a key")]
    async fn store_memory(
        &self,
        #[tool(aggr)] params: StoreMemoryParams
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
            // Use regular module
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
    
    #[tool(description = "Retrieve a specific memory by key")]
    async fn retrieve_memory(
        &self,
        #[tool(aggr)] params: RetrieveMemoryParams
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
    
    #[tool(description = "Search for similar memories")]
    async fn search_memory(
        &self,
        #[tool(aggr)] params: SearchMemoryParams
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
                "count": matches.len(),
                "operation_id": operation_id
            })))
        } else {
            // Use regular module
            let query_embedding = Array2::from_shape_vec((1, 768), vec![0.1; 768])
                .map_err(|e| ToolError(e.to_string()))?;
            
            let memory = self.memory_module.lock().await;
            let results = memory.retrieve_with_attention(&query_embedding, params.limit).await;
            
            let matches: Vec<_> = results.iter()
                .map(|(key, value, score)| json!({
                    "key": key.id,
                    "content": value.content,
                    "score": score[[0, 0]]
                }))
                .collect();
            
            Ok(ToolResult(json!({
                "query": params.query,
                "matches": matches,
                "count": matches.len()
            })))
        }
    }
    
    #[tool(description = "Get memory system statistics")]
    async fn memory_stats(
        &self,
        #[tool(aggr)] _params: MemoryStatsParams
    ) -> Result<ToolResult, ToolError> {
        if self.adaptive_enabled {
            // Use adaptive module stats
            let module = self.adaptive_module.as_ref().unwrap().lock().await;
            let stats = module.get_stats().await
                .map_err(|e| ToolError(e.to_string()))?;
            Ok(ToolResult(stats))
        } else {
            // Use regular module stats
            let memory = self.memory_module.lock().await;
            let (size, total_accesses, cache_hits, hit_rate) = memory.get_stats().await;
            
            Ok(ToolResult(json!({
                "memory_size": size,
                "total_accesses": total_accesses,
                "cache_hits": cache_hits,
                "cache_hit_rate": hit_rate
            })))
        }
    }
    
    #[tool(description = "Get current adaptive learning status")]
    async fn adaptive_status(
        &self,
        #[tool(aggr)] params: AdaptiveStatusParams
    ) -> Result<ToolResult, ToolError> {
        if !self.adaptive_enabled {
            return Err(ToolError("Adaptive learning not enabled".to_string()));
        }
        
        let module = self.adaptive_module.as_ref().unwrap().lock().await;
        let status = module.get_adaptive_status(params.verbose).await
            .map_err(|e| ToolError(e.to_string()))?;
        Ok(ToolResult(status))
    }
    
    #[tool(description = "Trigger adaptive training cycle")]
    async fn adaptive_train(
        &self,
        #[tool(aggr)] params: AdaptiveTrainParams
    ) -> Result<ToolResult, ToolError> {
        if !self.adaptive_enabled {
            return Err(ToolError("Adaptive learning not enabled".to_string()));
        }
        
        let module = self.adaptive_module.as_ref().unwrap().lock().await;
        let result = module.trigger_evolution(params.generations, params.force).await
            .map_err(|e| ToolError(e.to_string()))?;
        Ok(ToolResult(result))
    }
    
    #[tool(description = "Get learning insights and recommendations")]
    async fn adaptive_insights(
        &self,
        #[tool(aggr)] _params: AdaptiveInsightsParams
    ) -> Result<ToolResult, ToolError> {
        if !self.adaptive_enabled {
            return Err(ToolError("Adaptive learning not enabled".to_string()));
        }
        
        let module = self.adaptive_module.as_ref().unwrap().lock().await;
        let insights = module.get_insights().await
            .map_err(|e| ToolError(e.to_string()))?;
        Ok(ToolResult(insights))
    }
    
    #[tool(description = "Update adaptive learning configuration")]
    async fn adaptive_config(
        &self,
        #[tool(aggr)] params: AdaptiveConfigParams
    ) -> Result<ToolResult, ToolError> {
        if !self.adaptive_enabled {
            return Err(ToolError("Adaptive learning not enabled".to_string()));
        }
        
        let module = self.adaptive_module.as_ref().unwrap().lock().await;
        let result = module.update_config(params.objectives, params.enabled).await
            .map_err(|e| ToolError(e.to_string()))?;
        Ok(ToolResult(result))
    }
    
    #[tool(description = "Claude provides success/failure feedback for memory operations")]
    async fn provide_feedback(
        &self,
        #[tool(aggr)] params: ProvideFeedbackParams
    ) -> Result<ToolResult, ToolError> {
        if !self.adaptive_enabled {
            return Err(ToolError("Adaptive learning not enabled".to_string()));
        }
        
        use neural_llm_memory::adaptive::feedback::OperationFeedback;
        
        let feedback = OperationFeedback::from_claude(
            params.operation_id,
            params.success,
            params.score,
            params.reason,
            params.usage_context,
        );
        
        let module = self.adaptive_module.as_ref().unwrap().lock().await;
        let result = module.provide_feedback(feedback).await
            .map_err(|e| ToolError(e.to_string()))?;
        Ok(ToolResult(result))
    }
}

// Implement ServerHandler using the rmcp SDK
#[async_trait]
#[tool_box]
impl ServerHandler for NeuralMemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            name: "neural-memory".into(),
            version: "0.1.0".into(),
            instructions: Some("A neural memory system with adaptive learning capabilities".into()),
            capabilities: None,
        }
    }
    
    // The tool_box! macro automatically handles ListToolsRequest and CallToolRequest
    tool_box!(@impl ServerHandler for NeuralMemoryServer {
        store_memory,
        retrieve_memory,
        search_memory,
        memory_stats,
        adaptive_status,
        adaptive_train,
        adaptive_insights,
        adaptive_config,
        provide_feedback
    });
    
    async fn handle_request(
        &self,
        request: ClientRequest,
        context: RequestContext<rmcp::service::RoleServer>,
    ) -> Result<ServerResult, McpError> {
        match request {
            ClientRequest::InitializeRequest(req) => {
                eprintln!("Client connected: {:?}", req.params.client_info);
                Ok(ServerResult::InitializeResult(InitializeResult {
                    server_info: self.get_info().into(),
                    capabilities: ServerCapabilities {
                        tools: Some(ToolCapabilities {
                            list_changed: None,
                        }),
                        resources: Some(ResourceCapabilities {
                            subscribe: None,
                            list_changed: None,
                        }),
                        ..Default::default()
                    },
                }))
            }
            ClientRequest::ListResourcesRequest(_) => {
                let resources = vec![
                    Resource {
                        uri: "memory://stats".to_string(),
                        name: "Memory Statistics".to_string(),
                        description: Some("Current memory system statistics and performance metrics".to_string()),
                        mime_type: Some("application/json".to_string()),
                    },
                    Resource {
                        uri: "memory://adaptive/status".to_string(),
                        name: "Adaptive Learning Status".to_string(),
                        description: Some("Current adaptive learning status and evolution metrics".to_string()),
                        mime_type: Some("application/json".to_string()),
                    },
                    Resource {
                        uri: "memory://adaptive/insights".to_string(),
                        name: "Learning Insights".to_string(),
                        description: Some("Adaptive learning insights and recommendations".to_string()),
                        mime_type: Some("application/json".to_string()),
                    },
                    Resource {
                        uri: "memory://keys".to_string(),
                        name: "Stored Memory Keys".to_string(),
                        description: Some("List of all stored memory keys".to_string()),
                        mime_type: Some("application/json".to_string()),
                    },
                ];
                
                Ok(ServerResult::ListResourcesResult(ListResourcesResult {
                    resources,
                    next_cursor: None,
                }))
            }
            ClientRequest::ReadResourceRequest(req) => {
                let content = match req.params.uri.as_str() {
                    "memory://stats" => {
                        if self.adaptive_enabled {
                            let module = self.adaptive_module.as_ref().unwrap().lock().await;
                            module.get_stats().await
                                .map_err(|e| McpError::internal_error(e.to_string()))?
                        } else {
                            let memory = self.memory_module.lock().await;
                            let (size, total_accesses, cache_hits, hit_rate) = memory.get_stats().await;
                            json!({
                                "memory_size": size,
                                "total_accesses": total_accesses,
                                "cache_hits": cache_hits,
                                "cache_hit_rate": hit_rate
                            })
                        }
                    }
                    "memory://adaptive/status" => {
                        if self.adaptive_enabled {
                            let module = self.adaptive_module.as_ref().unwrap().lock().await;
                            module.get_adaptive_status(true).await
                                .map_err(|e| McpError::internal_error(e.to_string()))?
                        } else {
                            json!({"error": "Adaptive learning not enabled"})
                        }
                    }
                    "memory://adaptive/insights" => {
                        if self.adaptive_enabled {
                            let module = self.adaptive_module.as_ref().unwrap().lock().await;
                            module.get_insights().await
                                .map_err(|e| McpError::internal_error(e.to_string()))?
                        } else {
                            json!({"error": "Adaptive learning not enabled"})
                        }
                    }
                    "memory://keys" => {
                        json!({
                            "keys": [],
                            "count": 0,
                            "message": "Key listing not yet implemented"
                        })
                    }
                    _ => return Err(McpError::resource_not_found(&req.params.uri)),
                };
                
                Ok(ServerResult::ReadResourceResult(ReadResourceResult {
                    contents: vec![ResourceContents {
                        uri: req.params.uri.clone(),
                        mime_type: Some("application/json".to_string()),
                        text: Some(content.to_string()),
                        blob: None,
                    }],
                }))
            }
            _ => {
                // The tool_box! macro handles ListToolsRequest and CallToolRequest
                // Other requests are handled here or return method_not_found
                Err(McpError::method_not_found("Unknown request type"))
            }
        }
    }
    
    async fn handle_notification(
        &self,
        notification: ClientNotification,
    ) -> Result<(), McpError> {
        match notification {
            ClientNotification::CancelledNotification(note) => {
                eprintln!("Client cancelled request: {:?}", note.params.request_id);
                Ok(())
            }
            _ => {
                eprintln!("Received unhandled notification");
                Ok(())
            }
        }
    }
}

impl NeuralMemoryServer {
    async fn new() -> Result<Self> {
        // Create tokio runtime for sync operations if needed
        let runtime = Arc::new(
            tokio::runtime::Runtime::new()
                .expect("Failed to create tokio runtime")
        );
        
        // Always enable adaptive learning
        let adaptive_enabled = true;
        
        // Create adaptive memory module
        let base_config = MemoryConfig::default();
        let adaptive_config = AdaptiveConfig::default();
        
        let adaptive_module = AdaptiveMemoryModule::with_config(base_config.clone(), adaptive_config)
            .await?;
        
        // For compatibility, we still need a regular memory module
        let memory_module = PersistentMemoryBuilder::new()
            .storage_path("./neural_memory_data")
            .auto_save_interval(60)
            .memory_config(base_config)
            .build()
            .await?;
        
        Ok(Self {
            memory_module: Arc::new(Mutex::new(memory_module)),
            adaptive_module: Some(Arc::new(Mutex::new(adaptive_module))),
            runtime,
            adaptive_enabled,
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize server
    let server = NeuralMemoryServer::new().await?;
    
    // Only print to stderr AFTER server is ready
    eprintln!("🚀 neural-memory MCP server ready (using rmcp SDK)");
    
    // Use stdio transport
    use tokio::io::{stdin, stdout};
    let transport = (stdin(), stdout());
    
    // Start the server using rmcp's serve method
    let server_peer = server.serve(transport).await?;
    
    eprintln!("Server connected and running. Waiting for shutdown...");
    
    // Keep the server running
    let shutdown_reason = server_peer.waiting().await?;
    eprintln!("Server shut down. Reason: {:?}", shutdown_reason);
    
    // Shutdown and save before exit
    eprintln!("💾 Saving memories before shutdown...");
    let memory = server.memory_module.lock().await;
    memory.shutdown().await.ok();
    
    Ok(())
}