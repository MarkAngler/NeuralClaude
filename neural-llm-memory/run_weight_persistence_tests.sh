#!/bin/bash
# Script to run weight persistence integration tests

echo "=== Running Weight Persistence Integration Tests ==="
echo

# Set test environment variables
export RUST_TEST_THREADS=1  # Run tests sequentially for better output
export RUST_BACKTRACE=1     # Enable backtraces for debugging

# Run the tests with output
cargo test --test weight_persistence_integration_test -- --nocapture --test-threads=1

# Check exit code
if [ $? -eq 0 ]; then
    echo
    echo "✅ All weight persistence tests passed!"
else
    echo
    echo "❌ Some tests failed. Please check the output above."
    exit 1
fi