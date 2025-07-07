# Weight Persistence Integration Tests

This document describes the comprehensive integration tests for weight persistence in NeuralClaude.

## Test Coverage

### 1. Full Training Cycle Test (`test_full_training_cycle_persistence`)
- **Purpose**: Verifies that training can be interrupted and resumed without loss of progress
- **Process**:
  1. Create and train a network for 10 epochs
  2. Save the network state mid-training
  3. Load the network from disk
  4. Verify weights are exactly preserved
  5. Continue training for 10 more epochs
  6. Confirm training continues smoothly with loss improvement

### 2. Corrupted File Handling (`test_corrupted_file_handling`)
- **Purpose**: Ensures robust error handling for corrupted save files
- **Test Cases**:
  - Truncated files (simulated disk full or crash during write)
  - Corrupted headers (invalid binary data)
  - Wrong format files (JSON parsed as binary, etc.)
  - Partial write recovery (incomplete atomic saves)

### 3. Performance Benchmarks (`test_save_load_performance_benchmarks`)
- **Purpose**: Measures save/load performance for different network sizes
- **Network Sizes**:
  - Small: 5K parameters
  - Medium: 700K parameters  
  - Large: 12M parameters
- **Metrics**:
  - Save/load time
  - File size
  - Throughput (MB/s)
- **Formats Tested**:
  - Binary (fastest, exact precision)
  - JSON (human-readable, slight precision loss)
  - Compressed (smallest size, moderate speed)

### 4. Recovery from Incomplete Saves (`test_recovery_from_incomplete_saves`)
- **Purpose**: Tests atomic save operations and crash recovery
- **Scenarios**:
  - Simulated crash during file rename
  - Temp file cleanup
  - Primary file integrity preservation
  - Sequential save consistency

### 5. Concurrent Access Safety (`test_concurrent_save_load_safety`)
- **Purpose**: Verifies thread-safe file operations
- **Tests**:
  - Multiple threads loading simultaneously
  - Concurrent save operations with atomic writes
  - Race condition prevention
  - File lock handling

### 6. Weight Precision Preservation (`test_weight_precision_preservation`)
- **Purpose**: Ensures numerical precision is maintained
- **Test Values**:
  - High-precision decimals
  - Very small numbers (near epsilon)
  - Mathematical constants (π, e)
  - Edge cases (MIN_POSITIVE, MAX/1000)
- **Verification**:
  - Binary formats preserve exact f32 representation
  - JSON maintains at least 6 decimal places
  - No precision loss for critical values

## Running the Tests

```bash
# Run all weight persistence tests
./run_weight_persistence_tests.sh

# Run a specific test
cargo test --test weight_persistence_integration_test test_full_training_cycle_persistence -- --nocapture

# Run with detailed output
RUST_LOG=debug cargo test --test weight_persistence_integration_test -- --nocapture
```

## Expected Performance

Based on the benchmarks, you should expect:

- **Binary Format**:
  - Save: < 1 second for networks up to 12M parameters
  - Load: < 1 second for networks up to 12M parameters
  - No precision loss

- **Compressed Format**:
  - 40-60% smaller than binary
  - 2-3x slower than binary
  - No precision loss

- **JSON Format**:
  - Human-readable
  - 2-4x larger than binary
  - Minimal precision loss (< 1e-6 relative error)

## Integration with Adaptive Learning

The tests include hooks for the adaptive learning system:
- Each test operation generates an operation_id
- Test results are fed back to the learning system
- This helps the neural memory optimize for persistence operations

## Common Issues and Solutions

1. **Test Timeout**: Large network tests may timeout on slow systems
   - Solution: Increase test timeout or reduce network size

2. **Disk Space**: Performance tests create large temporary files
   - Solution: Ensure at least 1GB free space in /tmp

3. **Permission Errors**: Atomic saves require write permissions
   - Solution: Run tests with appropriate file permissions

4. **Memory Usage**: Large network tests may use significant RAM
   - Solution: Run tests individually or increase available memory