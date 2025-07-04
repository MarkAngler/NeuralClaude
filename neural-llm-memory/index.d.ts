/**
 * NeuralClaude - Neural LLM Memory Framework
 * High-performance Rust-based neural memory system for LLMs
 */

export interface NeuralConfig {
  dimensions?: number;
  capacity?: number;
  threshold?: number;
  persistPath?: string;
}

export interface MemoryEntry {
  key: string;
  content: string;
  embedding?: number[];
  timestamp?: number;
  metadata?: Record<string, any>;
}

export interface SearchResult {
  key: string;
  content: string;
  similarity: number;
  metadata?: Record<string, any>;
}

export interface MemoryStats {
  totalEntries: number;
  cacheSize: number;
  hitRate: number;
  avgRetrievalTime: number;
}

export declare class NeuralMemorySystem {
  constructor(config?: NeuralConfig);
  
  /**
   * Store a memory with the given key and content
   */
  store(key: string, content: string, metadata?: Record<string, any>): Promise<void>;
  
  /**
   * Retrieve a memory by its key
   */
  retrieve(key: string): Promise<MemoryEntry | null>;
  
  /**
   * Search for similar memories based on content
   */
  search(query: string, limit?: number): Promise<SearchResult[]>;
  
  /**
   * Get memory system statistics
   */
  getStats(): Promise<MemoryStats>;
  
  /**
   * Clear all memories
   */
  clear(): Promise<void>;
  
  /**
   * Delete a specific memory by key
   */
  delete(key: string): Promise<boolean>;
  
  /**
   * List all memory keys
   */
  listKeys(): Promise<string[]>;
  
  /**
   * Export all memories to a file
   */
  exportToFile(filePath: string): Promise<void>;
  
  /**
   * Import memories from a file
   */
  importFromFile(filePath: string): Promise<void>;
}

export function createMemorySystem(config?: NeuralConfig): NeuralMemorySystem;

export const VERSION: string;