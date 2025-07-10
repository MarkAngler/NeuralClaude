use crate::adaptive::AdaptiveConfig;
use std::env;
use serial_test::serial;

#[test]
#[serial]
fn test_default_evolution_threshold() {
    // Clear any existing env vars
    env::remove_var("NEURAL_MCP_EVOLUTION_THRESHOLD");
    env::remove_var("MCP_SERVER");
    env::remove_var("CLAUDE_MCP");
    
    let config = AdaptiveConfig::new();
    assert_eq!(config.evolution_after_operations, 1000);
}

#[test]
#[serial]
fn test_mcp_environment_default() {
    // Clear and set MCP environment
    env::remove_var("NEURAL_MCP_EVOLUTION_THRESHOLD");
    env::remove_var("CLAUDE_MCP");
    env::set_var("MCP_SERVER", "true");
    
    let config = AdaptiveConfig::new();
    assert_eq!(config.evolution_after_operations, 50);
    
    env::remove_var("MCP_SERVER");
}

#[test]
#[serial]
fn test_custom_threshold_from_env() {
    // Clear other env vars first
    env::remove_var("MCP_SERVER");
    env::remove_var("CLAUDE_MCP");
    env::set_var("NEURAL_MCP_EVOLUTION_THRESHOLD", "100");
    
    let config = AdaptiveConfig::new();
    assert_eq!(config.evolution_after_operations, 100);
    
    env::remove_var("NEURAL_MCP_EVOLUTION_THRESHOLD");
}

#[test]
#[serial]
fn test_update_from_env() {
    let mut config = AdaptiveConfig::new();
    config.evolution_after_operations = 500;
    
    env::set_var("NEURAL_MCP_EVOLUTION_THRESHOLD", "75");
    config.update_from_env();
    
    assert_eq!(config.evolution_after_operations, 75);
    
    env::remove_var("NEURAL_MCP_EVOLUTION_THRESHOLD");
}

#[test]
#[serial]
fn test_with_evolution_threshold() {
    let config = AdaptiveConfig::new()
        .with_evolution_threshold(200);
    
    assert_eq!(config.evolution_after_operations, 200);
}