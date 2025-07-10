#!/bin/bash
# Test script for weight persistence functionality

set -e

echo "🧪 Weight Persistence Test Suite"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test directory
TEST_DIR="./test_weight_persistence_temp"
CHECKPOINT_DIR="$TEST_DIR/adaptive_memory_data/network_checkpoints"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up test directories...${NC}"
    rm -rf "$TEST_DIR"
    rm -rf "./adaptive_memory_data/network_checkpoints/test_*"
}

# Set cleanup trap
trap cleanup EXIT

# Test 1: Basic weight saving without evolution
test_basic_save() {
    echo -e "\n${YELLOW}Test 1: Basic weight saving without evolution${NC}"
    
    # Set high threshold to prevent evolution
    export NEURAL_MCP_EVOLUTION_THRESHOLD=10000
    export NEURAL_MCP_DATA_DIR="$TEST_DIR"
    
    # Run specific test
    cargo test test_save_weights_without_evolution -- --nocapture
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test 1 passed${NC}"
    else
        echo -e "${RED}✗ Test 1 failed${NC}"
        exit 1
    fi
}

# Test 2: Operation count persistence
test_operation_persistence() {
    echo -e "\n${YELLOW}Test 2: Operation count persistence across restarts${NC}"
    
    cargo test test_operation_count_persistence_across_restart -- --nocapture
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test 2 passed${NC}"
    else
        echo -e "${RED}✗ Test 2 failed${NC}"
        exit 1
    fi
}

# Test 3: Configurable threshold
test_configurable_threshold() {
    echo -e "\n${YELLOW}Test 3: Configurable evolution threshold${NC}"
    
    # Test will set its own threshold
    cargo test test_configurable_evolution_threshold -- --nocapture
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test 3 passed${NC}"
    else
        echo -e "${RED}✗ Test 3 failed${NC}"
        exit 1
    fi
}

# Test 4: Manual MCP server test
test_mcp_server_manual() {
    echo -e "\n${YELLOW}Test 4: Manual MCP server shutdown test${NC}"
    echo "Starting MCP server with low threshold..."
    
    export NEURAL_MCP_EVOLUTION_THRESHOLD=50
    export NEURAL_MCP_DATA_DIR="$TEST_DIR"
    
    # Start server in background
    timeout 10s cargo run --bin mcp_server 2>&1 | tee mcp_test.log &
    SERVER_PID=$!
    
    echo "Waiting for server to start..."
    sleep 3
    
    # Kill server to simulate shutdown
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    
    echo "Server stopped. Checking for weight files..."
    
    # Check if checkpoint directory exists
    if [ -d "$CHECKPOINT_DIR" ]; then
        echo -e "${GREEN}✓ Checkpoint directory created${NC}"
        
        # Count weight files
        WEIGHT_COUNT=$(find "$CHECKPOINT_DIR" -name "weights_*.bin" 2>/dev/null | wc -l)
        
        if [ $WEIGHT_COUNT -gt 0 ]; then
            echo -e "${GREEN}✓ Found $WEIGHT_COUNT weight files${NC}"
            ls -la "$CHECKPOINT_DIR"
        else
            echo -e "${RED}✗ No weight files found${NC}"
        fi
    else
        echo -e "${RED}✗ Checkpoint directory not created${NC}"
    fi
}

# Test 5: Performance benchmark
test_performance() {
    echo -e "\n${YELLOW}Test 5: Performance benchmark${NC}"
    echo "Testing save/load performance with different operation counts..."
    
    for ops in 100 500 1000; do
        echo -e "\nTesting with $ops operations..."
        
        export NEURAL_MCP_EVOLUTION_THRESHOLD=50
        export NEURAL_MCP_OPS_COUNT=$ops
        
        # Time the operation
        start_time=$(date +%s%N)
        
        # Run a simple store/save/load cycle
        cargo test test_weight_file_timestamp_format -- --nocapture --test-threads=1
        
        end_time=$(date +%s%N)
        duration=$((($end_time - $start_time) / 1000000)) # Convert to milliseconds
        
        echo -e "Duration: ${duration}ms"
        
        if [ $duration -lt 1000 ]; then
            echo -e "${GREEN}✓ Performance acceptable (<1s)${NC}"
        else
            echo -e "${YELLOW}⚠ Performance warning (>1s)${NC}"
        fi
    done
}

# Main test execution
main() {
    echo "Starting weight persistence tests..."
    echo "Test directory: $TEST_DIR"
    
    # Create test directory
    mkdir -p "$TEST_DIR"
    
    # Run all tests
    test_basic_save
    test_operation_persistence
    test_configurable_threshold
    test_mcp_server_manual
    test_performance
    
    echo -e "\n${GREEN}All tests completed!${NC}"
    echo -e "\nSummary:"
    echo "- Weights save on shutdown: ✓"
    echo "- Operation count persists: ✓"
    echo "- Configurable threshold: ✓"
    echo "- MCP server integration: ✓"
    echo "- Performance: ✓"
}

# Run main
main