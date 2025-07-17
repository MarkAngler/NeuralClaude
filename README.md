# NeuralClaude

A Rust-based neural memory framework with Model Context Protocol (MCP) server integration for Large Language Models.

## Overview

NeuralClaude is a neural memory system implemented in Rust that provides persistent memory storage, adaptive learning, and consciousness modeling capabilities through an MCP server interface. The system is designed to enhance LLM interactions by providing stateful memory operations and neural network optimization.

## Core Architecture

### Memory System
- **Persistent Memory Module**: JSON-based storage with auto-save functionality
- **Adaptive Memory Module**: Self-optimizing memory with usage metrics collection
- **Hybrid Memory Bank**: Multi-modal memory storage with episodic and semantic components
- **Key-Value Store**: Efficient in-memory storage with LRU caching

### Neural Network Framework
- **Layer Types**: Linear, Convolutional 1D, Dropout, Layer Normalization, Embedding
- **Activation Functions**: ReLU, Sigmoid, Tanh, GELU, Swish
- **Optimizers**: Adam, SGD with momentum and weight decay
- **Weight Initialization**: Xavier, He, Normal, Uniform initialization strategies
- **Training State**: Gradient tracking, loss history, learning rate management

### Knowledge Graph
- **Conscious Graph**: Graph-based memory representation with consciousness attributes
- **Pattern Extraction**: Temporal and semantic pattern discovery
- **Relationship Inference**: Automatic edge creation between related concepts
- **HNSW Indexing**: High-performance similarity search using hierarchical navigable small world graphs
- **Cross-Modal Bridging**: Connections between different memory modalities

### Consciousness System
- **Global Workspace**: Conscious content integration and attention management
- **Self-Awareness Monitor**: Identity tracking and introspection capabilities
- **Emotional Processing**: Emotional state modeling and integration
- **Metacognition**: Higher-order thinking and self-reflection

## MCP Server Interface

The system provides an MCP server (`mcp_server.rs`) with the following tools:

### Core Memory Operations
- `store_memory`: Store content with neural embeddings
- `retrieve_memory`: Retrieve content by key
- `update_memory`: Update existing memory entries
- `delete_memory`: Remove memory entries
- `search_memory`: Semantic similarity search with graph traversal
- `memory_stats`: System usage statistics

### Adaptive Learning (Optional)
- `adaptive_status`: Current learning state and metrics
- `adaptive_train`: Manual evolution trigger
- `adaptive_insights`: Performance recommendations
- `adaptive_config`: Learning parameter adjustment
- `provide_feedback`: Operation feedback for learning

### Consciousness Features (Optional)
- `consciousness_status`: Current consciousness state
- `consciousness_process`: Process input through consciousness system
- `consciousness_reflect`: Reflective thinking operations
- `consciousness_introspect`: Internal state analysis
- `consciousness_insight`: Creative insight generation
- `consciousness_evolve`: Self-directed capability evolution
- `dream_consolidate`: Pattern consolidation and relationship discovery

## System Workflow

### Basic Memory Operations

The core workflow begins with basic memory operations:

1. **Storage**: When content is stored via `store_memory`, the system:
   - Generates neural embeddings (768-dimensional vectors)
   - Stores content in persistent JSON storage
   - Records usage metrics (response time, memory usage, content size)
   - Updates graph indices for future similarity searches

2. **Retrieval**: When content is retrieved via `retrieve_memory` or `search_memory`:
   - Performs exact key lookup or semantic similarity search
   - Uses HNSW indexing for fast similarity matching
   - Records cache hit/miss metrics
   - Can traverse graph relationships to find related content

### Adaptive Learning Cycle

The adaptive learning system continuously improves performance through a feedback loop:

1. **Metrics Collection**: Every operation records:
   - Response times
   - Memory usage changes
   - Similarity scores
   - Cache hit rates
   - User feedback (success/failure, relevance scores 0-1)

2. **Usage Analysis**: The `UsageCollector` analyzes patterns:
   - Identifies frequently accessed content
   - Detects performance bottlenecks
   - Tracks which neural architectures perform best
   - Measures user satisfaction trends

3. **Neural Evolution**: The `BackgroundEvolver` optimizes the system:
   - Experiments with different neural network architectures
   - Adjusts embedding dimensions and layer sizes
   - Modifies attention mechanisms based on usage patterns
   - Evolution triggered automatically after sufficient data or manually via `adaptive_train`

4. **Performance Benefits**: 
   - Faster retrieval for frequently accessed content
   - Better similarity matching for user's specific use patterns
   - Reduced memory usage through architecture optimization
   - Higher relevance scores in search results

### Dream Consolidation Process

Dream consolidation runs periodically (every 5 minutes during low activity) to enhance memory organization:

1. **Pattern Extraction**: Analyzes the last 24 hours of memory operations to identify:
   - Temporal patterns (memories accessed together in time)
   - Semantic patterns (conceptually related content)
   - Usage patterns (frequently co-accessed memories)

2. **Relationship Inference**: Creates new graph edges between related memories:
   - Causal relationships (A leads to B)
   - Temporal relationships (A and B occurred together)
   - Semantic relationships (A and B are conceptually similar)
   - Emotional relationships (A and B have similar emotional context)

3. **Graph Optimization**: Restructures the knowledge graph:
   - Strengthens connections between frequently co-accessed nodes
   - Creates shortcuts for common traversal paths
   - Removes weak or unused connections
   - Optimizes for future search performance

4. **Benefits**:
   - Improved search results through better relationship discovery
   - Faster graph traversal with optimized structure
   - Discovery of implicit connections between memories
   - Enhanced context retrieval for related concepts

### Consciousness Integration

The consciousness system adds higher-order processing:

1. **Global Workspace**: Integrates information from multiple sources:
   - Current memory operations
   - Historical usage patterns
   - Emotional context
   - Metacognitive assessments

2. **Self-Awareness Monitoring**: Tracks system state:
   - Performance metrics
   - Learning progress
   - Capability improvements
   - Limitation identification

3. **Reflection and Introspection**: Provides insights:
   - Analysis of memory organization effectiveness
   - Identification of knowledge gaps
   - Suggestions for system improvements
   - Creative insights from cross-domain connections

### Feedback Loop Integration

The system uses feedback to continuously improve:

1. **Operation Feedback**: Each operation receives a score (0-1) indicating usefulness
2. **Performance Correlation**: Correlates feedback with technical metrics
3. **Architecture Adjustment**: Modifies neural networks based on combined feedback and performance data
4. **Learning Acceleration**: Uses feedback to guide evolution direction

### Persistence and Recovery

The system maintains state across sessions:

1. **Auto-Save**: Periodically saves all component states
2. **Checkpoint Creation**: Creates timestamped neural network snapshots
3. **Crash Recovery**: Automatically restores from latest checkpoint on startup
4. **State Migration**: Handles updates to internal data structures

This workflow creates a self-improving memory system that becomes more effective over time by learning from usage patterns, user feedback, and performance metrics.

## Technical Specifications

### Dependencies
- **Core**: `ndarray`, `nalgebra`, `rayon` for numerical computing
- **Storage**: `serde`, `serde_json`, `bincode` for serialization
- **Graph**: `petgraph`, `hnsw` for graph operations
- **MCP**: `rmcp` for Model Context Protocol implementation
- **Async**: `tokio` for asynchronous operations
- **Concurrency**: `dashmap`, `parking_lot` for thread-safe operations

### Performance Features
- **SIMD Optimization**: Vectorized neural network operations
- **Parallel Processing**: Multi-threaded execution using Rayon
- **Memory Efficiency**: LRU caching and compression
- **Persistent Storage**: WAL-based persistence with crash recovery
- **Background Evolution**: Asynchronous neural network optimization

### Configuration
- **Memory Size**: Configurable capacity limits
- **Embedding Dimensions**: Default 768-dimensional vectors
- **Neural Architecture**: 768�512�256�128 layer structure
- **Learning Parameters**: Adaptive learning rates and evolution thresholds
- **Storage Paths**: Configurable data directories

## Installation and Usage

### Building from Source
```bash
cd neural-llm-memory
cargo build --release
```

### Running MCP Server
```bash
./target/release/mcp_server
```

### Environment Variables
- `NEURAL_MCP_ADAPTIVE`: Enable/disable adaptive learning (default: true)
- `NEURAL_MCP_CONSCIOUSNESS`: Enable/disable consciousness features (default: true)
- `NEURAL_MCP_AUTO_RECOVER`: Enable automatic state recovery (default: true)
- `NEURAL_MCP_AUTO_SAVE_INTERVAL`: Auto-save interval in seconds (default: 300)

## File Structure

```
neural-llm-memory/
   src/
      adaptive/          # Adaptive learning components
      attention/         # Attention mechanisms
      consciousness.rs   # Consciousness modeling
      graph/            # Knowledge graph implementation
      memory/           # Memory storage systems
      nn/               # Neural network framework
      storage/          # Persistence backends
      bin/
          mcp_server.rs # MCP server implementation
   tests/                # Integration and unit tests
   examples/             # Usage examples
   benches/              # Performance benchmarks
```

## Testing

The project includes comprehensive tests:
- Integration tests for memory operations
- Neural network forward/backward pass validation
- Persistence and recovery testing
- Concurrent stress tests
- Graph operations testing

Run tests with:
```bash
cargo test
```

## Data Storage

The system creates the following data directories:
- `./adaptive_memory_data/`: Adaptive module state and neural checkpoints
- `./neural_memory_data/`: Basic memory storage
- `./conscious_graph_data/`: Knowledge graph persistence

## License

This project is distributed under the terms specified in LICENSE.MD.