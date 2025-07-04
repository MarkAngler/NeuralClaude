//! Simple MCP server for neural LLM memory framework - without macros

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use neural_llm_memory::{PersistentMemoryModule, PersistentMemoryBuilder, MemoryConfig};
use ndarray::Array2;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    method: String,
    params: Option<Value>,
    id: Value,
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
    runtime: Arc<tokio::runtime::Runtime>,
}

impl MemoryServer {
    fn new() -> Self {
        // Create tokio runtime for async operations
        let runtime = Arc::new(
            tokio::runtime::Runtime::new()
                .expect("Failed to create tokio runtime")
        );
        
        // Create persistent memory module
        let memory_module = runtime.block_on(async {
            PersistentMemoryBuilder::new()
                .storage_path("./neural_memory_data")
                .auto_save_interval(60) // Save every minute
                .memory_config(MemoryConfig::default())
                .build()
                .await
                .expect("Failed to create persistent memory module")
        });
        
        Self {
            memory_module: Arc::new(Mutex::new(memory_module)),
            runtime,
        }
    }
    
    fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request),
            "initialized" => self.handle_initialized(request),
            "tools/list" => self.handle_list_tools(request),
            "tools/call" => self.handle_tool_call(request),
            _ => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Method not found: {}", request.method),
                    data: None,
                }),
                id: request.id,
            },
        }
    }
    
    fn handle_initialize(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {
                        "list": false,
                        "read": false
                    }
                },
                "serverInfo": {
                    "name": "neural-memory",
                    "version": "0.1.0"
                }
            })),
            error: None,
            id: request.id,
        }
    }
    
    fn handle_initialized(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(json!({})),
            error: None,
            id: request.id,
        }
    }
    
    fn handle_list_tools(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let tools = json!({
            "tools": [
                {
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
                },
                {
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
                },
                {
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
                },
                {
                    "name": "memory_stats",
                    "description": "Get memory system statistics",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        });
        
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(tools),
            error: None,
            id: request.id,
        }
    }
    
    fn handle_tool_call(&self, request: JsonRpcRequest) -> JsonRpcResponse {
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
                    id: request.id,
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
                    id: request.id,
                };
            }
        };
        
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
        
        let result = match tool_name {
            "store_memory" => self.store_memory(arguments),
            "retrieve_memory" => self.retrieve_memory(arguments),
            "search_memory" => self.search_memory(arguments),
            "memory_stats" => self.memory_stats(),
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
                id: request.id,
            },
            Err(msg) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32603,
                    message: msg,
                    data: None,
                }),
                id: request.id,
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
        
        // Create a simple embedding
        let embedding = Array2::from_shape_vec((1, 768), vec![0.1; 768])
            .map_err(|e| e.to_string())?;
        
        // Use runtime to execute async operation
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
    
    fn retrieve_memory(&self, args: Value) -> Result<Value, String> {
        let key = args.get("key")
            .and_then(|v| v.as_str())
            .ok_or("Missing key")?;
        
        Ok(json!({
            "key": key,
            "content": format!("Memory content for key: {}", key),
            "found": false
        }))
    }
    
    fn search_memory(&self, args: Value) -> Result<Value, String> {
        let query = args.get("query")
            .and_then(|v| v.as_str())
            .ok_or("Missing query")?;
        let limit = args.get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;
        
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
    
    fn memory_stats(&self) -> Result<Value, String> {
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

fn main() -> Result<()> {
    // Print startup to stderr
    eprintln!("🚀 neural-memory MCP server starting in stdio mode...");
    eprintln!("🧠 Initializing neural memory framework with persistent storage...");
    
    let server = MemoryServer::new();
    
    eprintln!("✅ Neural memory initialized successfully");
    eprintln!("📁 Storage directory: ./neural_memory_data");
    eprintln!("💾 Auto-save interval: 60 seconds");
    
    let stdin = io::stdin();
    let stdout = io::stdout();
    
    // Process JSON-RPC messages line by line
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        
        match serde_json::from_str::<JsonRpcRequest>(&line) {
            Ok(request) => {
                let response = server.handle_request(request);
                let response_str = serde_json::to_string(&response)?;
                writeln!(stdout.lock(), "{}", response_str)?;
                stdout.lock().flush()?;
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
                    id: json!(null),
                };
                let response_str = serde_json::to_string(&error_response)?;
                writeln!(stdout.lock(), "{}", response_str)?;
                stdout.lock().flush()?;
            }
        }
    }
    
    // Shutdown and save before exit
    eprintln!("💾 Saving memories before shutdown...");
    server.runtime.block_on(async {
        let memory = server.memory_module.lock().await;
        memory.shutdown().await.ok();
    });
    
    Ok(())
}