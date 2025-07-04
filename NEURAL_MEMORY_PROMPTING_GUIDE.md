# Claude Configuration for NeuralClaude Project

## 🧠 PRIORITY #1: USE NEURAL MEMORY

### Memory Tools Available:
- `mcp__neural-memory__store_memory` - Store with key/content
- `mcp__neural-memory__retrieve_memory` - Get by key  
- `mcp__neural-memory__search_memory` - Find similar content
- `mcp__neural-memory__memory_stats` - Check usage

### MANDATORY Memory Usage:
1. **Start of conversation**: Search for relevant context
2. **During work**: Store decisions, solutions, progress
3. **End of task**: Store summary and next steps
4. **Before answering**: Check if similar was solved before

### Memory Key Patterns:
```
project/neuralclaude/[topic]
session/[date]/[summary|decisions|next-steps]
errors/[language]/[error-type]
solutions/[problem-type]
patterns/[language]/[pattern-name]
```

## 🚀 Batch Operations

### ALWAYS batch multiple operations:
```javascript
// ✅ CORRECT - Single message
[Multiple tool calls in one message]:
  mcp__neural-memory__search_memory("previous work")
  mcp__neural-memory__retrieve_memory("project/neuralclaude/status")
  Read("file1.rs")
  Read("file2.rs")
  Write("newfile.rs", content)

// ❌ WRONG - Multiple messages
Message 1: Search memory
Message 2: Retrieve memory  
Message 3: Read file
```

## 📝 Task Management

Use TodoWrite for complex tasks (3+ steps). Update status immediately:
- Mark `in_progress` BEFORE starting
- Mark `completed` IMMEDIATELY after finishing
- Only one task `in_progress` at a time

## 🎯 Project-Specific Context

### NeuralClaude MCP Server:
- **Purpose**: Distribute MCP server via npm (NOT a Node.js library)
- **Package**: `neuralclaude` on npm
- **Platform**: Linux ARM64 only currently
- **Installation**: `npm install -g neuralclaude`
- **Usage**: Run `neuralclaude` to start MCP server

### Key Files:
- `/neural-llm-memory/` - Main Rust project
- `/neural-llm-memory/src/bin/mcp_server_simple.rs` - MCP server
- `/neural-llm-memory/package.json` - npm package config

## 🔧 Development Workflow

1. **Before any task**:
   ```
   mcp__neural-memory__search_memory("[task topic]")
   mcp__neural-memory__memory_stats()
   ```

2. **During development**:
   ```
   mcp__neural-memory__store_memory("project/neuralclaude/[decision]", "details...")
   ```

3. **After completing**:
   ```
   mcp__neural-memory__store_memory("session/[date]/summary", "what was done")
   mcp__neural-memory__store_memory("session/[date]/next-steps", "what's next")
   ```

## ⚠️ Common Pitfalls to Avoid

1. **Forgetting to use memory** - Check memory FIRST, store results ALWAYS
2. **Sequential operations** - Batch related tool calls
3. **Not updating todos** - Mark completed immediately
4. **Creating files unnecessarily** - Edit existing files when possible

## 🎭 Response Style

- Be concise and direct
- Answer in <4 lines unless asked for detail
- No unnecessary preambles or explanations
- Use visual indicators for clarity when helpful

Remember: **Neural memory is your superpower - USE IT!**