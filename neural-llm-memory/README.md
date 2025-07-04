# NeuralClaude MCP Server

Neural LLM Memory MCP Server - A high-performance neural memory system for AI assistants via the Model Context Protocol.

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
- Persistent memory storage
- Semantic search capabilities
- Attention-based retrieval
- Self-optimizing neural architecture

## MCP Tools Available

Once configured, Claude will have access to:
- `store_memory` - Store information with neural embeddings
- `retrieve_memory` - Retrieve specific memories
- `search_memory` - Search for similar memories
- `memory_stats` - Get memory system statistics

## Platform Support

Currently, this package includes a prebuilt binary for:
- **Linux ARM64** (aarch64)

For other platforms, please build from source at https://github.com/markangler/NeuralClaude

## License

MIT