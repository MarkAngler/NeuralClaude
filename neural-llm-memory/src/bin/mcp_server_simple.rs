//! Simple MCP server for neural LLM memory framework - without macros

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use neural_llm_memory::{PersistentMemoryModule, PersistentMemoryBuilder, MemoryConfig};
use neural_llm_memory::adaptive::AdaptiveMemoryModule;
use ndarray::Array2;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcMessage {
    jsonrpc: String,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
    id: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

struct MemoryServer {
    memory_module: Arc<Mutex<PersistentMemoryModule>>,
    adaptive_module: Option<Arc<Mutex<AdaptiveMemoryModule>>>,
    runtime: Arc<tokio::runtime::Runtime>,
    adaptive_enabled: bool,
}

impl MemoryServer {
    fn new() -> Self {
        // Create tokio runtime for async operations
        let runtime = Arc::new(
            tokio::runtime::Runtime::new()
                .expect("Failed to create tokio runtime")
        );
        
        // Always enable adaptive learning
        let adaptive_enabled = true;
        
        // Remove debug print during initialization
        
        // Create adaptive memory module with persistence
        let adaptive_module = runtime.block_on(async {
            let base_config = MemoryConfig::default();
            
            // Try to load from saved state
            AdaptiveMemoryModule::new_with_persistence(base_config, true)
                .await
                .expect("Failed to create adaptive memory module")
        });
        
        // For compatibility, we still need a regular memory module
        let memory_module = runtime.block_on(async {
            PersistentMemoryBuilder::new()
                .storage_path("./neural_memory_data")
                .auto_save_interval(60)
                .memory_config(MemoryConfig::default())
                .build()
                .await
                .expect("Failed to create persistent memory module")
        });
        
        // Wrap adaptive module in Arc
        let adaptive_module_arc = Arc::new(Mutex::new(adaptive_module));
        
        // Start auto-save task for adaptive module (every 5 minutes)
        {
            let module_clone = Arc::clone(&adaptive_module_arc);
            runtime.spawn(async move {
                neural_llm_memory::adaptive::start_auto_save_task(
                    Arc::new(module_clone.lock().await.clone()),
                    300
                ).await;
            });
        }
        
        Self {
            memory_module: Arc::new(Mutex::new(memory_module)),
            adaptive_module: Some(adaptive_module_arc),
            runtime,
            adaptive_enabled,
        }
    }
    
    fn handle_request(&self, request: JsonRpcMessage) -> Option<JsonRpcResponse> {
        // Handle notifications (no id field)
        if request.id.is_none() {
            match request.method.as_str() {
                "notifications/initialized" => {
                    // Notification acknowledged, no response needed
                    return None;
                }
                _ => {
                    // Unknown notification, ignore
                    return None;
                }
            }
        }
        
        // Handle regular requests (with id field)
        let id = request.id.clone().unwrap();
        let response = match request.method.as_str() {
            "initialize" => self.handle_initialize(request),
            "initialized" => self.handle_initialized(request),
            "tools/list" => self.handle_list_tools(request),
            "tools/call" => self.handle_tool_call(request),
            "resources/list" => self.handle_list_resources(request),
            "resources/read" => self.handle_read_resource(request),
            _ => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Method not found: {}", request.method),
                    data: None,
                }),
                id,
            },
        };
        Some(response)
    }
    
    fn handle_initialize(&self, request: JsonRpcMessage) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "list": true
                    },
                    "resources": {
                        "list": true,
                        "read": true
                    }
                },
                "serverInfo": {
                    "name": "neural-memory",
                    "version": "0.1.0"
                }
            })),
            error: None,
            id: request.id.unwrap(),
        }
    }
    
    fn handle_initialized(&self, request: JsonRpcMessage) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(json!({})),
            error: None,
            id: request.id.unwrap(),
        }
    }
    
    fn handle_list_resources(&self, request: JsonRpcMessage) -> JsonRpcResponse {
        // Return available resources for the neural memory system
        let resources = vec![
            json!({
                "uri": "memory://stats",
                "name": "Memory Statistics",
                "description": "Current memory system statistics and performance metrics",
                "mimeType": "application/json"
            }),
            json!({
                "uri": "memory://adaptive/status",
                "name": "Adaptive Learning Status",
                "description": "Current adaptive learning status and evolution metrics",
                "mimeType": "application/json"
            }),
            json!({
                "uri": "memory://adaptive/insights",
                "name": "Learning Insights",
                "description": "Adaptive learning insights and recommendations",
                "mimeType": "application/json"
            }),
            json!({
                "uri": "memory://keys",
                "name": "Stored Memory Keys",
                "description": "List of all stored memory keys",
                "mimeType": "application/json"
            })
        ];
        
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(json!(resources)),
            error: None,
            id: request.id.unwrap(),
        }
    }
    
    fn handle_read_resource(&self, request: JsonRpcMessage) -> JsonRpcResponse {
        let params = match request.params {
            Some(Value::Object(map)) => map,
            _ => {
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: "Invalid params".to_string(),
                        data: None,
                    }),
                    id: request.id.unwrap(),
                };
            }
        };
        
        let uri = match params.get("uri").and_then(|v| v.as_str()) {
            Some(uri) => uri,
            None => {
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: "Missing uri parameter".to_string(),
                        data: None,
                    }),
                    id: request.id.unwrap(),
                };
            }
        };
        
        let content = match uri {
            "memory://stats" => {
                match self.memory_stats() {
                    Ok(stats) => stats,
                    Err(e) => {
                        return JsonRpcResponse {
                            jsonrpc: "2.0".to_string(),
                            result: None,
                            error: Some(JsonRpcError {
                                code: -32603,
                                message: format!("Failed to get stats: {}", e),
                                data: None,
                            }),
                            id: request.id.unwrap(),
                        };
                    }
                }
            },
            "memory://adaptive/status" => {
                if self.adaptive_enabled {
                    match self.adaptive_status(json!({"verbose": true})) {
                        Ok(status) => status,
                        Err(e) => {
                            return JsonRpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: None,
                                error: Some(JsonRpcError {
                                    code: -32603,
                                    message: format!("Failed to get adaptive status: {}", e),
                                    data: None,
                                }),
                                id: request.id.unwrap(),
                            };
                        }
                    }
                } else {
                    json!({"error": "Adaptive learning not enabled"})
                }
            },
            "memory://adaptive/insights" => {
                if self.adaptive_enabled {
                    match self.adaptive_insights() {
                        Ok(insights) => insights,
                        Err(e) => {
                            return JsonRpcResponse {
                                jsonrpc: "2.0".to_string(),
                                result: None,
                                error: Some(JsonRpcError {
                                    code: -32603,
                                    message: format!("Failed to get insights: {}", e),
                                    data: None,
                                }),
                                id: request.id.unwrap(),
                            };
                        }
                    }
                } else {
                    json!({"error": "Adaptive learning not enabled"})
                }
            },
            "memory://keys" => {
                // For now, return a placeholder. In a real implementation,
                // this would query all stored keys from the memory module
                json!({
                    "keys": [],
                    "count": 0,
                    "message": "Key listing not yet implemented"
                })
            },
            _ => {
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: format!("Unknown resource URI: {}", uri),
                        data: None,
                    }),
                    id: request.id.unwrap(),
                };
            }
        };
        
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(json!({
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": content.to_string()
                }]
            })),
            error: None,
            id: request.id.unwrap(),
        }
    }
    
    fn handle_list_tools(&self, request: JsonRpcMessage) -> JsonRpcResponse {
        let mut tools_list = vec![
            json!({
                "name": "store_memory",
                    "description": "Store content in neural memory with a key",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Unique key for the memory"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to store"
                            }
                        },
                        "required": ["key", "content"]
                }
            }),
            json!({
                "name": "retrieve_memory",
                    "description": "Retrieve a specific memory by key",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Key of the memory to retrieve"
                            }
                        },
                        "required": ["key"]
                }
            }),
            json!({
                "name": "search_memory",
                    "description": "Search for similar memories",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query to search for"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                }
            }),
            json!({
                "name": "memory_stats",
                    "description": "Get memory system statistics",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                }
            })
        ];
        
        // Add adaptive tools if enabled
        if self.adaptive_enabled {
            tools_list.extend(vec![
                json!({
                    "name": "adaptive_status",
                    "description": "Get current adaptive learning status",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "verbose": {
                                "type": "boolean",
                                "description": "Include detailed metrics"
                            }
                        }
                    }
                }),
                json!({
                    "name": "adaptive_train",
                    "description": "Trigger adaptive training cycle",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "generations": {
                                "type": "integer",
                                "description": "Number of evolution generations"
                            },
                            "force": {
                                "type": "boolean",
                                "description": "Force training even with few samples"
                            }
                        }
                    }
                }),
                json!({
                    "name": "adaptive_insights",
                    "description": "Get learning insights and recommendations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }),
                json!({
                    "name": "adaptive_config",
                    "description": "Update adaptive learning configuration",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "objectives": {
                                "type": "object",
                                "description": "Optimization objectives with weights"
                            },
                            "enabled": {
                                "type": "boolean",
                                "description": "Enable/disable adaptive features"
                            }
                        }
                    }
                }),
                json!({
                    "name": "provide_feedback",
                    "description": "Claude provides success/failure feedback for memory operations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "operation_id": {
                                "type": "string",
                                "description": "ID of the operation to provide feedback for"
                            },
                            "success": {
                                "type": "boolean",
                                "description": "Whether the operation helped answer the user"
                            },
                            "score": {
                                "type": "number",
                                "description": "Relevance score between 0.0 and 1.0"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Why the operation succeeded or failed"
                            },
                            "usage_context": {
                                "type": "string",
                                "description": "How the result was used (or not used)"
                            }
                        },
                        "required": ["operation_id", "success"]
                    }
                }),
            ]);
        }
        
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(json!(tools_list)),
            error: None,
            id: request.id.unwrap(),
        }
    }
    
    fn handle_tool_call(&self, request: JsonRpcMessage) -> JsonRpcResponse {
        let params = match request.params {
            Some(Value::Object(map)) => map,
            _ => {
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: "Invalid params".to_string(),
                        data: None,
                    }),
                    id: request.id.unwrap(),
                };
            }
        };
        
        let tool_name = match params.get("name").and_then(|v| v.as_str()) {
            Some(name) => name,
            None => {
                return JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32602,
                        message: "Missing tool name".to_string(),
                        data: None,
                    }),
                    id: request.id.unwrap(),
                };
            }
        };
        
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
        
        let result = match tool_name {
            "store_memory" => self.store_memory(arguments),
            "retrieve_memory" => self.retrieve_memory(arguments),
            "search_memory" => self.search_memory(arguments),
            "memory_stats" => self.memory_stats(),
            // Adaptive tools (only available when adaptive is enabled)
            "adaptive_status" if self.adaptive_enabled => self.adaptive_status(arguments),
            "adaptive_train" if self.adaptive_enabled => self.adaptive_train(arguments),
            "adaptive_insights" if self.adaptive_enabled => self.adaptive_insights(),
            "adaptive_config" if self.adaptive_enabled => self.adaptive_config(arguments),
            "provide_feedback" if self.adaptive_enabled => self.provide_feedback(arguments),
            _ => Err(format!("Unknown tool: {}", tool_name)),
        };
        
        match result {
            Ok(value) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(json!({
                    "content": [
                        {
                            "type": "text",
                            "text": value.to_string()
                        }
                    ]
                })),
                error: None,
                id: request.id.unwrap(),
            },
            Err(msg) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32603,
                    message: msg,
                    data: None,
                }),
                id: request.id.unwrap(),
            },
        }
    }
    
    fn store_memory(&self, args: Value) -> Result<Value, String> {
        let key = args.get("key")
            .and_then(|v| v.as_str())
            .ok_or("Missing key")?;
        let content = args.get("content")
            .and_then(|v| v.as_str())
            .ok_or("Missing content")?;
        
        if self.adaptive_enabled {
            // Use adaptive module
            self.runtime.block_on(async {
                let module = self.adaptive_module.as_ref().unwrap().lock().await;
                let operation_id = module.store(key, content).await
                    .map_err(|e| e.to_string())?;
                Ok(json!({
                    "status": "stored",
                    "key": key,
                    "operation_id": operation_id
                }))
            })
        } else {
            // Use regular module
            let embedding = Array2::from_shape_vec((1, 768), vec![0.1; 768])
                .map_err(|e| e.to_string())?;
            
            let memory_key = self.runtime.block_on(async {
                let memory = self.memory_module.lock().await;
                memory.store_memory(content.to_string(), embedding).await
            }).map_err(|e| e.to_string())?;
            
            Ok(json!({
                "status": "stored",
                "key": key,
                "memory_id": memory_key.id
            }))
        }
    }
    
    fn retrieve_memory(&self, args: Value) -> Result<Value, String> {
        let key = args.get("key")
            .and_then(|v| v.as_str())
            .ok_or("Missing key")?;
        
        if self.adaptive_enabled {
            // Use adaptive module
            self.runtime.block_on(async {
                let module = self.adaptive_module.as_ref().unwrap().lock().await;
                match module.retrieve(key).await {
                    Ok((Some(content), operation_id)) => Ok(json!({
                        "key": key,
                        "content": content,
                        "found": true,
                        "operation_id": operation_id
                    })),
                    Ok((None, operation_id)) => Ok(json!({
                        "key": key,
                        "content": null,
                        "found": false,
                        "operation_id": operation_id
                    })),
                    Err(e) => Err(e.to_string()),
                }
            })
        } else {
            // Placeholder for regular module
            Ok(json!({
                "key": key,
                "content": format!("Memory content for key: {}", key),
                "found": false
            }))
        }
    }
    
    fn search_memory(&self, args: Value) -> Result<Value, String> {
        let query = args.get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing query")?;
        let limit = args.get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;
        
        if self.adaptive_enabled {
            // Use adaptive module
            self.runtime.block_on(async {
                let module = self.adaptive_module.as_ref().unwrap().lock().await;
                let (results, operation_id) = module.search(query, limit).await
                    .map_err(|e| e.to_string())?;
                
                let matches: Vec<_> = results.into_iter()
                    .map(|(key, score)| json!({
                        "key": key,
                        "score": score
                    }))
                    .collect();
                
                Ok(json!({
                    "query": query,
                    "matches": matches,
                    "count": matches.len(),
                    "operation_id": operation_id
                }))
            })
        } else {
            // Use regular module
            let query_embedding = Array2::from_shape_vec((1, 768), vec![0.1; 768])
                .map_err(|e| e.to_string())?;
            
            let results = self.runtime.block_on(async {
                let memory = self.memory_module.lock().await;
                memory.retrieve_with_attention(&query_embedding, limit).await
            });
            
            let matches: Vec<_> = results.iter()
                .map(|(key, value, score)| json!({
                    "key": key.id,
                    "content": value.content,
                    "score": score[[0, 0]]
                }))
                .collect();
            
            Ok(json!({
                "query": query,
                "matches": matches,
                "count": matches.len()
            }))
        }
    }
    
    fn memory_stats(&self) -> Result<Value, String> {
        if self.adaptive_enabled {
            // Use adaptive module stats
            self.runtime.block_on(async {
                let module = self.adaptive_module.as_ref().unwrap().lock().await;
                module.get_stats().await
                    .map_err(|e| e.to_string())
            })
        } else {
            // Use regular module stats
            let (size, total_accesses, cache_hits, hit_rate) = self.runtime.block_on(async {
                let memory = self.memory_module.lock().await;
                memory.get_stats().await
            });
            
            Ok(json!({
                "memory_size": size,
                "total_accesses": total_accesses,
                "cache_hits": cache_hits,
                "cache_hit_rate": hit_rate
            }))
        }
    }
    
    // Adaptive learning methods
    
    fn adaptive_status(&self, args: Value) -> Result<Value, String> {
        let verbose = args.get("verbose")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        self.runtime.block_on(async {
            let module = self.adaptive_module.as_ref().unwrap().lock().await;
            module.get_adaptive_status(verbose).await
                .map_err(|e| e.to_string())
        })
    }
    
    fn adaptive_train(&self, args: Value) -> Result<Value, String> {
        let generations = args.get("generations")
            .and_then(|v| v.as_u64())
            .map(|g| g as usize);
        let force = args.get("force")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        self.runtime.block_on(async {
            let module = self.adaptive_module.as_ref().unwrap().lock().await;
            module.trigger_evolution(generations, force).await
                .map_err(|e| e.to_string())
        })
    }
    
    fn adaptive_insights(&self) -> Result<Value, String> {
        self.runtime.block_on(async {
            let module = self.adaptive_module.as_ref().unwrap().lock().await;
            module.get_insights().await
                .map_err(|e| e.to_string())
        })
    }
    
    fn adaptive_config(&self, args: Value) -> Result<Value, String> {
        let objectives = args.get("objectives")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f as f32)))
                    .collect::<HashMap<String, f32>>()
            });
        
        let enabled = args.get("enabled")
            .and_then(|v| v.as_bool());
        
        self.runtime.block_on(async {
            let module = self.adaptive_module.as_ref().unwrap().lock().await;
            module.update_config(objectives, enabled).await
                .map_err(|e| e.to_string())
        })
    }
    
    fn provide_feedback(&self, args: Value) -> Result<Value, String> {
        let operation_id = args.get("operation_id")
            .and_then(|v| v.as_str())
            .ok_or("Missing operation_id")?
            .to_string();
        
        let success = args.get("success")
            .and_then(|v| v.as_bool())
            .ok_or("Missing success")?;
        
        let score = args.get("score")
            .and_then(|v| v.as_f64())
            .map(|s| s as f32);
        
        let reason = args.get("reason")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        let usage_context = args.get("usage_context")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        // Import the feedback module
        use neural_llm_memory::adaptive::feedback::OperationFeedback;
        
        let feedback = OperationFeedback::from_claude(
            operation_id,
            success,
            score,
            reason,
            usage_context,
        );
        
        self.runtime.block_on(async {
            let module = self.adaptive_module.as_ref().unwrap().lock().await;
            module.provide_feedback(feedback).await
                .map_err(|e| e.to_string())
        })
    }
}

fn main() -> Result<()> {
    // Initialize server first, before any output
    let server = MemoryServer::new();
    
    // Only print to stderr AFTER server is ready
    eprintln!("🚀 neural-memory MCP server ready");
    
    let stdin = io::stdin();
    let stdout = io::stdout();
    
    // Process JSON-RPC messages line by line
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        
        match serde_json::from_str::<JsonRpcMessage>(&line) {
            Ok(request) => {
                if let Some(response) = server.handle_request(request) {
                    let response_str = serde_json::to_string(&response)?;
                    
                    // Write and flush immediately - acquire lock, write, flush, release
                    {
                        let mut stdout = stdout.lock();
                        writeln!(stdout, "{}", response_str)?;
                        stdout.flush()?;
                    }
                }
            }
            Err(e) => {
                let error_response = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                        data: None,
                    }),
                    id: json!("error"),
                };
                let response_str = serde_json::to_string(&error_response)?;
                
                // Write and flush immediately - acquire lock, write, flush, release
                {
                    let mut stdout = stdout.lock();
                    writeln!(stdout, "{}", response_str)?;
                    stdout.flush()?;
                }
            }
        }
    }
    
    // Shutdown and save before exit
    eprintln!("💾 Saving memories before shutdown...");
    server.runtime.block_on(async {
        // Save regular memory module
        let memory = server.memory_module.lock().await;
        memory.shutdown().await.ok();
        
        // Save adaptive module state
        if let Some(adaptive) = &server.adaptive_module {
            let module = adaptive.lock().await;
            if let Err(e) = module.save_state("./adaptive_memory_data").await {
                eprintln!("Failed to save adaptive state: {}", e);
            } else {
                eprintln!("✅ Saved adaptive neural network state");
            }
        }
    });
    
    Ok(())
}