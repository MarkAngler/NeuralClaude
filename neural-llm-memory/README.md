# NeuralClaude MCP Server

Neural LLM Memory MCP Server - An adaptive learning memory bank for AI assistants via the Model Context Protocol.

## What is NeuralClaude?

NeuralClaude is an **adaptive learning memory system** that goes beyond traditional storage:

- **Adaptive Neural Memory**: Stores and retrieves information with semantic understanding
- **Self-Optimizing**: Evolves based on usage patterns and feedback to improve over time
- **Learning System**: Uses genetic algorithms to optimize neural network topology
- **Feedback-Driven**: Learns from your interactions (70% user feedback, 30% performance metrics)
- **Personalized**: Adapts retrieval strategies to your specific needs

Unlike static memory systems, NeuralClaude gets smarter about what to remember and how to retrieve it, creating a co-evolutionary relationship with AI assistants.

## Installation

```bash
npm install -g neuralclaude
```

**Note:** Currently only supports Linux ARM64 (aarch64) platforms.

## Usage

After installation, run the MCP server:

```bash
neuralclaude
```

The server will start and listen for MCP protocol connections on stdin/stdout.

## Configuring with Claude Desktop

1. Find your Claude Desktop config file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. Add neuralclaude to the `mcpServers` section:

```json
{
  "mcpServers": {
    "neuralclaude": {
      "command": "neuralclaude"
    }
  }
}
```

3. Restart Claude Desktop

See [MCP_SETUP.md](./MCP_SETUP.md) for detailed setup instructions.

## Features

- High-performance Rust-based neural memory system
- Persistent memory storage with automatic recovery
- Semantic search capabilities
- Attention-based retrieval
- Self-optimizing neural architecture
- Neural network weights persist across restarts

## MCP Tools Available

Once configured, Claude will have access to:

### Core Memory Operations
- `store_memory` - Store information with neural embeddings
- `retrieve_memory` - Retrieve specific memories by key
- `search_memory` - Search for similar memories using semantic similarity
- `memory_stats` - Get memory system statistics
- `provide_feedback` - Provide feedback on operation success/failure

### Adaptive Learning Tools
- `adaptive_status` - Check learning progress and evolution metrics
- `adaptive_train` - Manually trigger neural network evolution
- `adaptive_insights` - Get optimization recommendations
- `adaptive_config` - Adjust learning objectives and parameters

## MCP Resources Available

The server also exposes read-only resources:
- `memory://stats` - Current memory system statistics
- `memory://adaptive/status` - Adaptive learning status
- `memory://adaptive/insights` - Learning insights and recommendations
- `memory://keys` - List of stored memory keys

## Data Storage

NeuralClaude persists all data in the `./adaptive_memory_data/` directory:

```
./adaptive_memory_data/
├── adaptive_state.json          # Adaptive module state & metrics
├── memories.json                # Memory content storage
├── network_checkpoints/         # Neural network weights
│   ├── evolved_config.json      # Evolved network architecture
│   └── checkpoint_*.bin         # Weight checkpoints
└── recovery.lock               # Concurrent access prevention
```

### Environment Variables

- `NEURAL_MCP_ADAPTIVE` - Enable/disable adaptive learning (default: `true`)
- `NEURAL_MCP_AUTO_RECOVER` - Auto-load saved state on startup (default: `true`)
- `NEURAL_MCP_AUTO_SAVE_INTERVAL` - Auto-save interval in seconds (default: `300`)
- `NEURAL_MCP_CHECKPOINT_INTERVAL` - Checkpoint save interval (default: `300`)
- `NEURAL_MCP_MAX_CHECKPOINTS` - Maximum checkpoints to keep (default: `10`)

## Platform Support

Currently, this package includes a prebuilt binary for:
- **Linux ARM64** (aarch64)

For other platforms, please build from source at https://github.com/markangler/NeuralClaude

## License

MIT