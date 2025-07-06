# NeuralClaude Memory Prompting Guide

## 🧠 PRIORITY #1: USE ADAPTIVE NEURAL MEMORY

### Core Memory Tools:
1. **`mcp__neural-memory__store_memory`** - Store content with key
   - Parameters: `key` (string), `content` (string)
   - Returns: operation_id (for feedback)

2. **`mcp__neural-memory__retrieve_memory`** - Get content by key
   - Parameters: `key` (string)
   - Returns: content + operation_id

3. **`mcp__neural-memory__search_memory`** - Find similar content
   - Parameters: `query` (string), `limit` (optional, default: 5)
   - Returns: matches with scores + operation_id

4. **`mcp__neural-memory__memory_stats`** - Check system usage
   - Parameters: none
   - Returns: memory statistics

5. **`mcp__neural-memory__provide_feedback`** - **CRITICAL: Always provide feedback!**
   - Parameters: `operation_id` (required), `success` (required), `score` (required, 0-1)
   - Returns: feedback status
   - **Note**: Score should reflect how useful the result was (1.0 = perfect, 0.0 = useless)

### Adaptive Learning Tools (when NEURAL_MCP_ADAPTIVE=true):
6. **`mcp__neural-memory__adaptive_status`** - Check learning progress
   - Parameters: `verbose` (optional, default: false)
   - Returns: evolution metrics

7. **`mcp__neural-memory__adaptive_train`** - Trigger neural evolution
   - Parameters: `generations` (optional, default: 5), `force` (optional, default: false)
   - Returns: evolution results

8. **`mcp__neural-memory__adaptive_insights`** - Get optimization tips
   - Parameters: none
   - Returns: learning insights

9. **`mcp__neural-memory__adaptive_config`** - Adjust learning settings
   - Parameters: `objectives` (optional), `enabled` (optional)
   - Returns: config status

## 🎯 MANDATORY Memory + Feedback Workflow

### 1. **Start of Every Conversation**:
```
Step 1: Search for relevant context
- Tool: mcp__neural-memory__search_memory
- Parameters:
  - query: "[user's topic/question]"
  - limit: 10

Step 2: Provide feedback on search results
- Tool: mcp__neural-memory__provide_feedback
- Parameters:
  - operation_id: "<from Step 1>"
  - success: true/false
  - score: 0.0-1.0
  - reason: "Found exact solution" or "No relevant matches"
  - usage_context: "Used in response" or "Ignored, searching elsewhere"
```

### 2. **During Work**:
```
Store key decisions/solutions:
- Tool: mcp__neural-memory__store_memory
- Parameters:
  - key: "project/neuralclaude/[feature]/[decision]"
  - content: "Detailed explanation of what was done and why"

Then provide feedback:
- Tool: mcp__neural-memory__provide_feedback
- Parameters:
  - operation_id: "<from store operation>"
  - success: true
  - score: 1.0
  - reason: "Stored critical decision for future reference"
```

### 3. **End of Task**:
```
Step 1: Store session summary
- Tool: mcp__neural-memory__store_memory
- Parameters:
  - key: "session/[YYYY-MM-DD]/summary"
  - content: "What was accomplished, key changes made"

Step 2: Store next steps
- Tool: mcp__neural-memory__store_memory
- Parameters:
  - key: "session/[YYYY-MM-DD]/next-steps"
  - content: "What needs to be done next, blockers, questions"

Step 3: Provide feedback on both operations
```

## 📋 Memory Key Patterns

Use consistent, hierarchical keys:
```
project/neuralclaude/[component]/[topic]
project/neuralclaude/bugs/[issue-description]
project/neuralclaude/features/[feature-name]
session/[YYYY-MM-DD]/[summary|decisions|next-steps]
errors/[language]/[error-type]/[specific-error]
solutions/[problem-category]/[specific-solution]
patterns/[language]/[pattern-type]/[pattern-name]
knowledge/[domain]/[topic]/[subtopic]
```

## 🚀 Batch Operations Pattern

**ALWAYS batch related operations in a single message:**
```
Correct MCP tool usage (single message with multiple tools):
1. mcp__neural-memory__search_memory
   - query: "Docker deployment issues"
   - limit: 5

2. mcp__neural-memory__retrieve_memory
   - key: "project/neuralclaude/deployment/config"

3. mcp__neural-memory__provide_feedback
   - operation_id: "<from search>"
   - success: true
   - score: 0.8
   - reason: "Found related deployment patterns"
   - usage_context: "Applied Docker config from results"

4. Read
   - file_path: "/deployment/docker-compose.yml"

5. mcp__neural-memory__store_memory
   - key: "solutions/docker/compose-optimization"
   - content: "Optimized compose file with caching..."
```

## 🎭 Feedback Quality Examples

### ✅ HIGH-QUALITY Feedback:
```
SUCCESS (score: 0.95):
- reason: "Retrieved exact JWT implementation from previous session. Code pattern matched current needs perfectly."
- usage_context: "Copied auth middleware directly into new endpoint with minor variable name changes"

PARTIAL (score: 0.6):
- reason: "Found related WebSocket patterns but for different framework. Had to adapt significantly."
- usage_context: "Used connection handling logic but rewrote event system for current architecture"

FAILURE (score: 0.1):
- reason: "Search returned Python examples but working in Rust. Syntax completely different."
- usage_context: "Ignored results, searched Rust documentation instead"
```

### ❌ POOR Feedback (avoid these):
- "It worked" (too vague)
- "Found something" (no specifics)
- Missing operation_id
- No usage_context provided

## 📊 Adaptive Learning Workflow

### Monitor Evolution:
```
1. Check current status:
   - Tool: mcp__neural-memory__adaptive_status
   - Parameters: verbose: true

2. Review insights periodically:
   - Tool: mcp__neural-memory__adaptive_insights

3. Trigger evolution when needed:
   - Tool: mcp__neural-memory__adaptive_train
   - Parameters: generations: 10
```

### Configure Objectives:
```
Tool: mcp__neural-memory__adaptive_config
Parameters:
  objectives: {
    "search_accuracy": 0.4,
    "retrieval_speed": 0.3,
    "storage_efficiency": 0.3
  }
```

## 📝 Task Management Integration

For complex tasks (3+ steps), combine with TodoWrite:
1. Create todos for main objectives
2. Use memory to check previous similar work
3. Store decisions as you complete each todo
4. Provide feedback on memory usefulness
5. Store final summary when todos complete

## 🔧 Project-Specific Context

### NeuralClaude Architecture:
- **Main Project**: `/neural-llm-memory/` (Rust)
- **MCP Server**: `/neural-llm-memory/src/bin/mcp_server.rs`
- **Adaptive Module**: `/neural-llm-memory/src/adaptive/`
- **Neural Networks**: `/neural-llm-memory/src/nn/`
- **Memory Core**: `/neural-llm-memory/src/memory/`
- **npm Package**: `neuralclaude` (Linux ARM64)

### Key Features:
- 768-dimensional embeddings with SIMD optimization
- Adaptive learning through genetic algorithms
- Background evolution every 100 operations
- Compressed storage (bincode, JSON, MessagePack)
- Lock-free concurrent access

## ⚠️ Critical Best Practices

### DO:
✅ ALWAYS provide feedback with operation_ids
✅ Search memory BEFORE implementing solutions
✅ Store decisions/solutions IMMEDIATELY
✅ Use descriptive, hierarchical keys
✅ Batch related operations
✅ Check adaptive_insights regularly

### DON'T:
❌ Forget to provide feedback
❌ Use vague keys like "temp" or "data"
❌ Make sequential tool calls
❌ Ignore operation_ids
❌ Skip memory search at start
❌ Provide generic feedback

## 🧬 The Feedback Loop

Your feedback drives evolution:
1. **You search/store/retrieve** → Get operation_id
2. **You provide specific feedback** → System learns patterns
3. **Neural network evolves** → Better embeddings
4. **Results improve** → You work more efficiently
5. **Repeat** → Continuous improvement

**Remember: The more specific your feedback, the faster the system adapts to your needs!**

## 🚨 Quick Reference

```
# Start of conversation:
search → feedback → proceed

# During work:
store decisions → feedback → continue

# Finding solutions:
search → retrieve → feedback → implement

# End of task:
store summary → store next → feedback both
```

**Adaptive memory with feedback is your AI superpower - TRAIN IT WELL!**