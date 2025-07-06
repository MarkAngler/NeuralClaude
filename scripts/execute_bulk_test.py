#!/usr/bin/env python3
"""
Execute bulk memory operations to trigger evolution
This script will store memories and provide feedback through MCP tool calls
"""
import json
import time

# Load test data
with open("/tmp/bulk_memories.json", "r") as f:
    data = json.load(f)

memories = data["memories"]
searches = data["searches"]

print(f"📊 Loaded {len(memories)} memories and {len(searches)} search queries")
print("\n⚡ Starting bulk operations...")
print("=" * 60)

# Instructions for manual execution
print("\n📋 INSTRUCTIONS FOR BULK OPERATIONS:")
print("\n1️⃣ STORE ALL MEMORIES (70 operations):")
print("Execute these store operations in batches of 5-10 at a time:\n")

# Generate store commands
for i, mem in enumerate(memories[:10]):  # Show first 10 as example
    print(f"Store #{i+1}:")
    print(f"  mcp__neural-memory__store_memory")
    print(f"  - key: \"{mem['key']}\"")
    print(f"  - content: \"{mem['content'][:50]}...\"")
    print()

print(f"... and {len(memories)-10} more memories")
print("\n2️⃣ PROVIDE FEEDBACK FOR STORES:")
print("After each batch of stores, provide feedback:")
print("  mcp__neural-memory__provide_feedback")
print("  - operation_id: <from store response>")
print("  - success: true")
print("  - score: 1.0")

print("\n3️⃣ PERFORM SEARCHES (20 operations):")
print("Execute these search operations:\n")

for i, search in enumerate(searches[:5]):  # Show first 5 as example
    print(f"Search #{i+1}:")
    print(f"  mcp__neural-memory__search_memory")
    print(f"  - query: \"{search['query']}\"")
    print(f"  - limit: 5")
    print()

print(f"... and {len(searches)-5} more searches")

print("\n4️⃣ PROVIDE SEARCH FEEDBACK:")
print("Rate search quality based on relevance (0.0-1.0)")

print("\n5️⃣ CHECK EVOLUTION STATUS:")
print("After ~100 operations, check if evolution triggered:")
print("  mcp__neural-memory__adaptive_status")
print("  - verbose: true")

print("\n6️⃣ FORCE EVOLUTION IF NEEDED:")
print("If not triggered automatically:")
print("  mcp__neural-memory__adaptive_train")
print("  - generations: 10")
print("  - force: true")

print("\n" + "=" * 60)
print("🎯 Total operations needed: ~100 to trigger evolution")
print("📈 Current operations: 22")
print("🔄 Operations to perform: 78+")
print("=" * 60)