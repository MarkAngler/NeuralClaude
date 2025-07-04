# NeuralClaude

Neural LLM Memory Framework - A high-performance Rust-based neural memory system for LLMs, available as an npm package.

## Installation

```bash
npm install neuralclaude
```

Or with yarn:

```bash
yarn add neuralclaude
```

## Features

- 🚀 High-performance Rust implementation
- 🧠 Neural memory with attention mechanisms
- 💾 Persistent storage support
- 🔍 Semantic search capabilities
- 📊 Memory usage statistics
- 🔄 Import/export functionality
- ⚡ Async/await API

## Quick Start

```javascript
const { NeuralMemorySystem } = require('neuralclaude');

// Create a new memory system
const memory = new NeuralMemorySystem({
  dimensions: 768,      // Embedding dimensions
  capacity: 10000,      // Maximum number of memories
  threshold: 0.7,       // Similarity threshold
  persistPath: './memory.db'  // Path to persistent storage
});

// Store a memory
await memory.store('user_preference', 'User likes dark mode', {
  category: 'preferences',
  timestamp: Date.now().toString()
});

// Retrieve a specific memory
const result = await memory.retrieve('user_preference');
console.log(result);

// Search for similar memories
const similar = await memory.search('user settings', 5);
console.log(similar);

// Get memory statistics
const stats = await memory.getStats();
console.log(stats);
```

## API Reference

### `new NeuralMemorySystem(config?)`

Creates a new neural memory system instance.

**Parameters:**
- `config` (optional): Configuration object
  - `dimensions`: Number of embedding dimensions (default: 768)
  - `capacity`: Maximum number of memories (default: 10000)
  - `threshold`: Similarity threshold for searches (default: 0.7)
  - `persistPath`: Path to persistent storage file (default: './neural_memory.db')

### `store(key, content, metadata?)`

Stores a memory with the given key and content.

**Parameters:**
- `key`: Unique identifier for the memory
- `content`: The content to store
- `metadata`: Optional metadata object

**Returns:** Promise<void>

### `retrieve(key)`

Retrieves a memory by its key.

**Parameters:**
- `key`: The key of the memory to retrieve

**Returns:** Promise<MemoryEntry | null>

### `search(query, limit?)`

Searches for memories similar to the query.

**Parameters:**
- `query`: The search query
- `limit`: Maximum number of results (default: 10)

**Returns:** Promise<SearchResult[]>

### `getStats()`

Gets memory system statistics.

**Returns:** Promise<MemoryStats>

### `clear()`

Clears all memories.

**Returns:** Promise<void>

### `delete(key)`

Deletes a specific memory by key.

**Parameters:**
- `key`: The key of the memory to delete

**Returns:** Promise<boolean>

### `listKeys()`

Lists all memory keys.

**Returns:** Promise<string[]>

### `exportToFile(filePath)`

Exports all memories to a file.

**Parameters:**
- `filePath`: Path to the export file

**Returns:** Promise<void>

### `importFromFile(filePath)`

Imports memories from a file.

**Parameters:**
- `filePath`: Path to the import file

**Returns:** Promise<void>

## Examples

### Building a Chatbot Memory

```javascript
const { NeuralMemorySystem } = require('neuralclaude');

class ChatbotMemory {
  constructor() {
    this.memory = new NeuralMemorySystem({
      dimensions: 768,
      capacity: 5000,
      persistPath: './chatbot_memory.db'
    });
  }

  async rememberConversation(userId, message, response) {
    const key = `conversation_${userId}_${Date.now()}`;
    await this.memory.store(key, message, {
      userId,
      response,
      timestamp: new Date().toISOString()
    });
  }

  async findSimilarConversations(query, userId) {
    const results = await this.memory.search(query, 10);
    return results.filter(r => r.metadata?.userId === userId);
  }
}
```

### Document Storage System

```javascript
const { NeuralMemorySystem } = require('neuralclaude');

class DocumentStore {
  constructor() {
    this.memory = new NeuralMemorySystem({
      dimensions: 1024,
      capacity: 50000,
      persistPath: './documents.db'
    });
  }

  async indexDocument(docId, content, metadata) {
    // Split document into chunks
    const chunks = this.splitIntoChunks(content, 500);
    
    for (let i = 0; i < chunks.length; i++) {
      const key = `${docId}_chunk_${i}`;
      await this.memory.store(key, chunks[i], {
        ...metadata,
        docId,
        chunkIndex: i,
        totalChunks: chunks.length
      });
    }
  }

  async searchDocuments(query, limit = 20) {
    return await this.memory.search(query, limit);
  }

  splitIntoChunks(text, chunkSize) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
      chunks.push(text.slice(i, i + chunkSize));
    }
    return chunks;
  }
}
```

## TypeScript Support

Full TypeScript definitions are included. Simply import the types:

```typescript
import { 
  NeuralMemorySystem, 
  NeuralConfig, 
  MemoryEntry, 
  SearchResult,
  MemoryStats 
} from 'neuralclaude';

const memory = new NeuralMemorySystem({
  dimensions: 768,
  capacity: 10000
} as NeuralConfig);
```

## Performance

NeuralClaude is built with Rust for maximum performance:

- Fast memory operations with O(1) retrieval by key
- Efficient similarity search using optimized vector operations
- Low memory overhead with smart caching
- Persistent storage with minimal I/O operations

## Building from Source

If you want to build the native module from source:

```bash
git clone https://github.com/yourusername/NeuralClaude.git
cd NeuralClaude/neural-llm-memory
npm install
npm run build
```

## Requirements

- Node.js >= 14.0.0
- npm or yarn
- Rust toolchain (only for building from source)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and feature requests, please use the [GitHub issue tracker](https://github.com/yourusername/NeuralClaude/issues).