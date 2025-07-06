//! Test MCP server to verify rmcp SDK is working

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use neural_llm_memory::{PersistentMemoryModule, PersistentMemoryBuilder, MemoryConfig};
use std::sync::Arc;
use tokio::sync::Mutex;
use schemars::JsonSchema;

// Import rmcp types
use rmcp::{
    ServerHandler, ServiceExt,
    model::*,
    service::RequestContext,
    error::Error as McpError,
};
use async_trait::async_trait;

// Simple test server
#[derive(Clone)]
struct TestServer;

#[async_trait]
impl ServerHandler for TestServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            name: "test-server".into(),
            version: "0.1.0".into(),
            instructions: Some("A test server".into()),
            capabilities: None,
        }
    }
    
    async fn handle_request(
        &self,
        request: ClientRequest,
        _context: RequestContext<rmcp::service::RoleServer>,
    ) -> Result<ServerResult, McpError> {
        match request {
            ClientRequest::InitializeRequest(_) => {
                Ok(ServerResult::InitializeResult(InitializeResult {
                    server_info: self.get_info().into(),
                    capabilities: ServerCapabilities::default(),
                }))
            }
            ClientRequest::ListToolsRequest(_) => {
                // Return a simple tool list
                Ok(ServerResult::ListToolsResult(ListToolsResult {
                    tools: vec![
                        Tool {
                            name: "test_tool".into(),
                            description: Some("A test tool".into()),
                            input_schema: None,
                        }
                    ],
                    next_cursor: None,
                }))
            }
            _ => Err(McpError::method_not_found("Unknown request")),
        }
    }
    
    async fn handle_notification(
        &self,
        _notification: ClientNotification,
    ) -> Result<(), McpError> {
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    eprintln!("Starting test server...");
    
    let server = TestServer;
    
    use tokio::io::{stdin, stdout};
    let transport = (stdin(), stdout());
    
    let server_peer = server.serve(transport).await?;
    
    eprintln!("Test server running...");
    
    server_peer.waiting().await?;
    
    Ok(())
}