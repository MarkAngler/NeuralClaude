# Neural Memory Persistent Storage

This document describes the persistent storage implementation for the neural memory system.

## Overview

The persistent storage system provides:
- **Automatic persistence** of memories to disk
- **JSON-based storage** format (with optional compression planned)
- **Write-ahead logging (WAL)** for crash recovery
- **Periodic saves** to minimize performance impact
- **Backup and restore** functionality
- **Concurrent access safety**

## Architecture

### Storage Layers

1. **`PersistentStorageBackend`** - High-level interface combining in-memory cache with persistent storage
2. **`JsonStorage`** - Low-level JSON file storage with WAL support
3. **`StorageConfig`** - Configuration for storage behavior

### Key Features

#### Write-Ahead Logging (WAL)
- All operations are first written to a WAL file
- On crash, WAL is replayed to recover unsaved changes
- WAL is cleared after successful full saves

#### Periodic Saves
- Configurable save intervals (default: 5 minutes)
- Background thread handles saves without blocking operations
- Dirty tracking ensures only changed data is saved

#### Atomic Writes
- Uses temporary files and atomic renames
- Prevents corruption from partial writes

## Usage

### Basic Usage

```rust
use neural_llm_memory::memory::PersistentMemoryBuilder;

// Create a new persistent memory system
let mut memory = PersistentMemoryBuilder::new()
    .with_storage_path("./my_memories")
    .with_memory_size(10000)
    .with_embedding_dim(768)
    .with_auto_save(false)  // Use periodic saves
    .with_save_interval(300) // Save every 5 minutes
    .build()?;

// Store memories - automatically persisted
let key = memory.store_memory(
    "Important fact".to_string(),
    embedding_array
)?;

// Memories are automatically loaded on restart
let memory = PersistentMemoryModule::load("./my_memories")?;
```

### Configuration Options

```rust
let config = StorageConfig {
    base_path: PathBuf::from("./storage"),
    auto_save: false,           // Save on every write vs periodic
    save_interval_secs: 300,    // Seconds between saves
    max_backups: 5,            // Number of backups to keep
    compress: false,           // Compress storage files
    enable_wal: true,          // Enable write-ahead logging
    max_file_size_mb: 100,     // Max file size before rotation
};
```

### Backup and Restore

```rust
// Create a backup
let backup_path = memory.backup()?;

// Restore from backup
memory.restore_from_backup(&backup_path)?;
```

## Storage Format

The JSON storage format:

```json
{
  "version": "1.0",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "memories": [
    {
      "key": {
        "id": "unique-id",
        "timestamp": 1234567890,
        "context_hash": 987654321
      },
      "value": {
        "embedding": [0.1, 0.2, 0.3],
        "content": "Memory content",
        "metadata": {
          "importance": 1.0,
          "access_count": 5,
          "last_accessed": 1234567890,
          "decay_factor": 0.95,
          "tags": ["tag1", "tag2"]
        }
      }
    }
  ]
}
```

## Performance Considerations

1. **Memory Usage**: The system maintains an in-memory cache for fast access
2. **I/O Operations**: Periodic saves reduce disk I/O impact
3. **Concurrent Access**: Uses read-write locks for thread safety
4. **Large Datasets**: Consider enabling compression for large memory banks

## Error Handling

The storage system handles:
- Missing directories (auto-created)
- Corrupted files (via backups)
- Concurrent access conflicts
- Disk space issues
- Permission errors

## Future Enhancements

- [ ] Compression support (gzip/zstd)
- [ ] Database backend option (SQLite/PostgreSQL)
- [ ] Incremental backups
- [ ] Memory sharding for very large datasets
- [ ] Cloud storage backends (S3, GCS)
- [ ] Encryption at rest