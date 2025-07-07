#!/usr/bin/env python3
"""
Test script to verify ML state persistence in NeuralClaude
"""
import json
import os
from pathlib import Path

def check_ml_state_persistence():
    """Check what ML state is actually persisted"""
    
    print("🔍 Checking ML State Persistence in NeuralClaude\n")
    
    # Check adaptive state file
    adaptive_state_paths = [
        "./adaptive_memory_data/adaptive_state.json",
        "./neural-llm-memory/adaptive_memory_data/adaptive_state.json",
        "../adaptive_memory_data/adaptive_state.json"
    ]
    
    adaptive_state = None
    for path in adaptive_state_paths:
        if os.path.exists(path):
            print(f"✅ Found adaptive state at: {path}")
            with open(path, 'r') as f:
                adaptive_state = json.load(f)
            break
    else:
        print("❌ No adaptive state file found")
    
    if adaptive_state:
        print("\n📊 Adaptive State Contents:")
        print(f"  - Operation Count: {adaptive_state.get('operation_count', 0)}")
        print(f"  - Evolution Status: {adaptive_state.get('evolution_status', {})}")
        print(f"  - Best Fitness: {adaptive_state['evolution_status'].get('best_fitness', 0.0)}")
        print(f"  - Current Generation: {adaptive_state['evolution_status'].get('current_generation', 0)}")
        print(f"  - Last Evolution: {adaptive_state.get('last_evolution_time', 'Never')}")
        print(f"  - Saved At: {adaptive_state.get('saved_at', 'Unknown')}")
    
    # Check for network checkpoints
    print("\n🧠 Checking Neural Network Checkpoints:")
    checkpoint_paths = [
        "./adaptive_memory_data/network_checkpoints",
        "./neural-llm-memory/adaptive_memory_data/network_checkpoints",
        "./network_checkpoints"
    ]
    
    found_checkpoints = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"✅ Found checkpoint directory: {path}")
            files = list(Path(path).glob("*"))
            if files:
                print("  Files found:")
                for f in files:
                    print(f"    - {f.name}")
            else:
                print("  ⚠️  Directory is empty")
            found_checkpoints = True
            break
    
    if not found_checkpoints:
        print("❌ No network checkpoint directories found")
    
    # Check for memory data
    print("\n💾 Checking Memory Data:")
    memory_paths = [
        "./adaptive_memory_data/memories.json",
        "./neural-llm-memory/adaptive_memory_data/memories.json",
        "./adaptive_memory_data/memory_bank.bin"
    ]
    
    for path in memory_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"✅ Found memory file: {path} ({size:.1f} KB)")
    
    # Summary
    print("\n📋 ML State Persistence Summary:")
    print("=" * 50)
    
    issues = []
    
    if adaptive_state:
        if adaptive_state.get('operation_count', 0) == 0:
            issues.append("Operation count is 0 - state may not be updating")
        if adaptive_state['evolution_status'].get('current_generation', 0) == 0:
            issues.append("No evolution generations recorded")
    else:
        issues.append("No adaptive state file found")
    
    if not found_checkpoints:
        issues.append("No neural network weight checkpoints found")
    
    if issues:
        print("⚠️  Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All persistence mechanisms appear to be working")
    
    print("\n🔧 Recommendations:")
    print("  1. Neural network weights are NOT being saved (implementation incomplete)")
    print("  2. Evolution state is tracked but network architecture isn't persisted")
    print("  3. Only metadata and configuration are saved, not the actual ML model")
    print("  4. To fix: Implement LayerState extraction in neural network classes")

if __name__ == "__main__":
    check_ml_state_persistence()