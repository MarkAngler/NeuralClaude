# Neural Memory MCP - Prompting Guide

This guide explains how to effectively prompt Claude to use the neural memory MCP server for persistent memory across conversations.

## Quick Start

Your neural memory MCP server is configured as `mcp__neural-memory__` and provides these tools:

- `mcp__neural-memory__store_memory` - Store important information with a key
- `mcp__neural-memory__retrieve_memory` - Retrieve specific memories by key
- `mcp__neural-memory__search_memory` - Search for similar memories using neural similarity
- `mcp__neural-memory__memory_stats` - Get memory system statistics

## Initial Setup Prompt

Use this prompt when starting a new conversation:

```markdown
I have a neural memory MCP server configured as `mcp__neural-memory__`. This provides persistent memory across our conversations using these tools:

- `mcp__neural-memory__store_memory`: Store important information with a key
- `mcp__neural-memory__retrieve_memory`: Retrieve specific memories by key
- `mcp__neural-memory__search_memory`: Search for similar memories using neural similarity
- `mcp__neural-memory__memory_stats`: Get memory system statistics

Please use this memory system to:
1. Store key decisions, context, and learnings from our work
2. Retrieve relevant memories when starting new tasks
3. Search for similar past experiences when solving problems
```

## Effective Usage Patterns

### 1. Project Context Storage

```markdown
"As we work on [project], please store important decisions and context in memory using descriptive keys like:
- `project/[name]/architecture` - Design decisions
- `project/[name]/bugs/[id]` - Bug fixes and solutions
- `project/[name]/features/[name]` - Feature implementations
- `project/[name]/dependencies` - Key dependencies and versions"
```

### 2. Session Continuity

```markdown
"Before starting, search memory for any relevant context from previous sessions about [topic]. Store our progress at key milestones using keys like:
- `session/[date]/summary` - Session summaries
- `session/[date]/decisions` - Key decisions made
- `session/[date]/next-steps` - Action items for next session"
```

### 3. Learning Pattern Storage

```markdown
"When we solve complex problems or discover patterns, store them as:
- `patterns/[language]/[pattern-name]` - Code patterns
- `solutions/[problem-type]` - Problem solutions
- `errors/[error-type]/fix` - Error resolutions
- `optimizations/[technique]` - Performance improvements"
```

### 4. Code Snippet Library

```markdown
"Store useful code snippets and examples:
- `snippets/[language]/[functionality]` - Reusable code
- `examples/[framework]/[feature]` - Framework examples
- `templates/[type]` - Project templates"
```

## Example Prompts by Use Case

### Starting a New Session

```markdown
"Check memory for any previous work on neural networks or Rust projects. Specifically search for:
1. Previous architecture decisions
2. Unresolved issues or bugs
3. Next steps from last session
Then let's continue building the memory framework, storing key decisions as we go."
```

### Debugging Session

```markdown
"I'm encountering error: '[error message]'
1. Search memory for similar error patterns
2. If we find a solution, apply it
3. If this is new, store our solution with key 'errors/rust/[error-type]' once resolved
4. Also store the debugging process for future reference"
```

### Code Review

```markdown
"We're reviewing [component]. Please:
1. Search memory for previous reviews of similar components
2. Store this architecture decision with key 'architecture/neural-memory/[component]'
3. Note any patterns that should be applied elsewhere
4. Store review feedback for future improvements"
```

### Knowledge Building

```markdown
"As you learn about [technology]:
1. Store key insights with descriptive keys
2. Before answering questions, search memory for relevant stored knowledge
3. Update existing memories if you find better information
4. Create a knowledge map with keys like 'knowledge/[topic]/[subtopic]'"
```

### Feature Implementation

```markdown
"Implementing [feature name]:
1. Search memory for similar features we've built
2. Check for relevant design patterns
3. Store the implementation approach as 'features/[project]/[feature]/approach'
4. Save any reusable components for future use"
```

## Best Practices for Memory Keys

### Key Structure

Use hierarchical, descriptive keys:

```
category/subcategory/specific-item
```

Examples:
- `project/neural-memory/architecture/tensor-design`
- `errors/rust/lifetime/async-trait-fix`
- `patterns/rust/builder/with-validation`
- `meeting/2024-01-15/action-items`

### Key Conventions

1. **Use lowercase with hyphens**: `my-key-name` not `MyKeyName`
2. **Be specific**: `bugs/auth/jwt-refresh-race-condition` not `bugs/auth-bug`
3. **Include dates when relevant**: `release/2024-01-15/notes`
4. **Version important items**: `api/v2/endpoints/user-create`

## Integration with ruv-swarm

When using both neural memory and swarm coordination:

```markdown
"Initialize a swarm for [task]. Configure each agent to:
1. Check neural memory for relevant context using search before starting
2. Store findings with keys like 'swarm/[task]/[agent]/[finding]'
3. Share important discoveries by storing in shared memory namespace
4. Search memory for similar patterns when encountering obstacles

Memory namespaces:
- `swarm/shared/` - Information all agents should know
- `swarm/[task]/[agent]/` - Agent-specific findings
- `swarm/patterns/` - Discovered coordination patterns"
```

## Advanced Prompting Strategies

### 1. Memory-Augmented Problem Solving

```markdown
"For this problem:
1. First search memory for similar problems and their solutions
2. Identify patterns from past solutions
3. Apply relevant patterns to current problem
4. Store our solution with comparisons to past approaches"
```

### 2. Incremental Learning

```markdown
"As we work:
1. Continuously update memory with new learnings
2. Revise previous memories if we find better approaches
3. Build a knowledge graph by linking related memories
4. Track which memories are most useful (access patterns)"
```

### 3. Context Switching

```markdown
"Switching to work on [different project]:
1. Store current context with key 'context/[project]/[timestamp]'
2. Search and load context for new project
3. Note any relevant crossover patterns
4. Maintain separate memory namespaces for each project"
```

### 4. Team Knowledge Sharing

```markdown
"For team collaboration:
1. Store decisions in 'team/decisions/[date]/[topic]'
2. Document patterns in 'team/patterns/[category]'
3. Share debugging solutions in 'team/solutions/[problem]'
4. Create onboarding memories in 'team/onboarding/[topic]'"
```

## Example Conversation Starters

### General Development

```markdown
"I have neural memory available via MCP. Let's work on [task]. Please:
1. First run memory stats to see what's available
2. Search for any relevant previous work on [topic]
3. Store important decisions as 'decisions/[date]/[topic]'
4. Build on previous knowledge where possible

What relevant memories do you find?"
```

### Continuing Previous Work

```markdown
"Let's continue our neural memory framework. Please:
1. Retrieve memory with key 'project/neural-memory/last-session'
2. Search for any unresolved issues or TODOs
3. Load the current architecture from memory
4. Update progress as we work

What was our last status?"
```

### Learning New Technology

```markdown
"I want to learn about [technology]. Please:
1. Search memory for any existing knowledge
2. As you explain, store key concepts
3. Save practical examples and use cases
4. Create a learning path in memory

Start with searching what we already know."
```

## Memory Maintenance

### Periodic Cleanup

```markdown
"Let's review our memory usage:
1. Show memory stats
2. Search for duplicate or outdated entries
3. Consolidate related memories
4. Archive old project memories with prefix 'archive/'"
```

### Memory Organization

```markdown
"Help me organize memories:
1. List all memory keys matching pattern '[pattern]'
2. Suggest better key structures
3. Identify missing categories
4. Create an index of important memories"
```

## Troubleshooting

### If Memory Retrieval Fails

```markdown
"If you can't retrieve a memory:
1. Try searching with different keywords
2. Check memory stats to ensure system is working
3. Use broader search terms
4. Store a new version if the old one is inaccessible"
```

### Performance Optimization

```markdown
"To optimize memory usage:
1. Check cache hit rates with memory stats
2. Store frequently accessed items with shorter keys
3. Use search instead of retrieve when unsure of exact keys
4. Batch related memory operations"
```

## Conclusion

Effective use of neural memory transforms Claude from a stateless assistant to a learning partner that builds knowledge over time. The key is being explicit about memory operations and using consistent, descriptive naming conventions.

Remember:
- **Store early and often** - Capture decisions and learnings as they happen
- **Search before solving** - Leverage past experience
- **Organize thoughtfully** - Good key structure makes retrieval easier
- **Maintain actively** - Keep memory relevant and organized

With these practices, your neural memory becomes a powerful knowledge base that grows more valuable with each conversation.