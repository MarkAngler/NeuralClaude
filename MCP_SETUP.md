# Setting up NeuralClaude with Claude Desktop

## 1. Install the MCP Server

```bash
npm install -g neuralclaude
```

## 2. Configure Claude Desktop

Find your Claude Desktop configuration file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

Add the neuralclaude server to the `mcpServers` section:

```json
{
  "mcpServers": {
    "neuralclaude": {
      "command": "neuralclaude"
    }
  }
}
```

If you already have other MCP servers configured, add it to the existing list:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    },
    "neuralclaude": {
      "command": "neuralclaude"
    }
  }
}
```

## 3. Restart Claude Desktop

After saving the configuration, restart Claude Desktop for the changes to take effect.

## 4. Verify It's Working

In a new Claude conversation, you should now have access to these tools:

- `store_memory` - Store information with a key
- `retrieve_memory` - Retrieve memory by key
- `search_memory` - Search for similar memories
- `memory_stats` - Get memory statistics

Try it out:
- "Store a memory about my favorite color being blue"
- "What memories do you have about my preferences?"

## Troubleshooting

If the tools don't appear:

1. Check the server is installed:
   ```bash
   which neuralclaude
   # Should output: /usr/local/bin/neuralclaude or similar
   ```

2. Test the server manually:
   ```bash
   echo '{"jsonrpc":"2.0","method":"initialize","params":{},"id":1}' | neuralclaude
   ```
   You should see a JSON response with server capabilities.

3. Check Claude Desktop logs for any errors

4. Make sure the path to `neuralclaude` is in your system PATH

## Alternative: Specify Full Path

If Claude can't find the command, you can specify the full path:

```json
{
  "mcpServers": {
    "neuralclaude": {
      "command": "/usr/local/bin/neuralclaude"
    }
  }
}
```

Find the full path with:
```bash
which neuralclaude
```