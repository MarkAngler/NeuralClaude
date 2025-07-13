#!/usr/bin/env python3
"""
NeuralClaude Progress Tracking Dashboard
Monitors and visualizes the improvement of the neural memory system over time.
"""

import json
import subprocess
import datetime
from typing import Dict, List, Any
import os
from pathlib import Path

class NeuralMemoryTracker:
    def __init__(self):
        self.metrics_file = Path.home() / ".neuralclaude" / "metrics_history.json"
        self.metrics_file.parent.mkdir(exist_ok=True)
        self.load_history()
    
    def load_history(self):
        """Load historical metrics from file."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []
    
    def save_history(self):
        """Save metrics history to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def query_mcp_tool(self, tool: str, params: Dict[str, Any] = None) -> Dict:
        """Query MCP tools through MCP client."""
        try:
            # Use MCP client to query tools
            import sys
            sys.path.append('/home/markangler/Github/NeuralClaude/neural-llm-memory')
            
            # Import and use the MCP client directly
            # Note: In production, this would use the proper MCP client library
            # For now, we'll use subprocess to call the MCP server
            
            # Return actual data from calling the tool
            # This is a placeholder - in real usage, you'd integrate with the MCP client
            print(f"Note: To get live data, run this script through Claude with MCP tools enabled")
            
            # Return realistic placeholder data that matches actual schema
            if tool == "adaptive_status":
                return {
                    "adaptive_enabled": True,
                    "evolution_status": {
                        "current_generation": 0,
                        "best_fitness": 0.0,
                        "is_running": False,
                        "progress_percent": 0.0
                    },
                    "usage_stats": {
                        "avg_response_time_ms": 0.0,
                        "cache_hit_rate": 0.0,
                        "total_operations": 22
                    },
                    "memory_stats": {
                        "total_memories": 22,
                        "cache_hit_rate": 0.0,
                        "total_operations": 22
                    }
                }
            elif tool == "memory_stats":
                return {
                    "memory_stats": {
                        "total_memories": 22,
                        "cache_hit_rate": 0.0,
                        "total_operations": 22
                    },
                    "usage_stats": {
                        "avg_response_time_ms": 0.0,
                        "cache_hit_rate": 0.0,
                        "total_operations": 22
                    }
                }
        except Exception as e:
            print(f"Error querying MCP tool {tool}: {e}")
            return {}
        
        return {}
    
    def collect_metrics(self) -> Dict:
        """Collect current metrics from the neural memory system."""
        timestamp = datetime.datetime.now().isoformat()
        
        # Get adaptive status
        adaptive_status = self.query_mcp_tool("adaptive_status", {"verbose": True})
        
        # Get memory stats
        memory_stats = self.query_mcp_tool("memory_stats")
        
        # Extract data based on actual response structure
        evolution_status = adaptive_status.get("evolution_status", {})
        usage_stats = adaptive_status.get("usage_stats", {})
        mem_stats = memory_stats.get("memory_stats", {})
        
        metrics = {
            "timestamp": timestamp,
            "operation_count": usage_stats.get("total_operations", 0),
            "generation": evolution_status.get("current_generation", 0),
            "best_fitness": evolution_status.get("best_fitness", 0),
            "avg_response_ms": usage_stats.get("avg_response_time_ms", 0),
            "cache_hit_rate": usage_stats.get("cache_hit_rate", 0),
            "total_memories": mem_stats.get("total_memories", 0),
            "improvements": self.calculate_improvements()
        }
        
        return metrics
    
    def calculate_improvements(self) -> Dict:
        """Calculate improvement percentages from baseline."""
        if len(self.history) < 2:
            return {}
        
        baseline = self.history[0]
        current = self.history[-1] if self.history else {}
        
        improvements = {}
        
        # Response time improvement (lower is better)
        if baseline.get("avg_response_ms", 0) > 0:
            rt_improvement = (baseline["avg_response_ms"] - current.get("avg_response_ms", 0)) / baseline["avg_response_ms"] * 100
            improvements["response_time"] = rt_improvement
        
        # Cache hit rate improvement (higher is better)
        if "cache_hit_rate" in baseline:
            chr_improvement = (current.get("cache_hit_rate", 0) - baseline["cache_hit_rate"]) * 100
            improvements["cache_hit_rate"] = chr_improvement
        
        # Fitness improvement (higher is better)
        if "best_fitness" in baseline:
            fitness_improvement = (current.get("best_fitness", 0) - baseline["best_fitness"]) * 100
            improvements["fitness"] = fitness_improvement
        
        return improvements
    
    def display_dashboard(self):
        """Display a text-based dashboard of current metrics."""
        current = self.collect_metrics()
        self.history.append(current)
        self.save_history()
        
        print("\n" + "="*60)
        print("🧠 NeuralClaude Progress Dashboard")
        print("="*60)
        print(f"📅 Timestamp: {current['timestamp']}")
        print(f"🔢 Total Operations: {current['operation_count']}")
        print(f"🧬 Evolution Generation: {current['generation']}")
        print(f"💪 Best Fitness Score: {current['best_fitness']:.4f}")
        print(f"⚡ Avg Response Time: {current['avg_response_ms']:.2f}ms")
        print(f"💾 Cache Hit Rate: {current['cache_hit_rate']:.2%}")
        print(f"🗃️  Total Memories: {current['total_memories']}")
        
        if current['improvements']:
            print("\n📈 Improvements from Baseline:")
            for metric, improvement in current['improvements'].items():
                symbol = "📈" if improvement > 0 else "📉"
                print(f"  {symbol} {metric.replace('_', ' ').title()}: {improvement:+.2f}%")
        
        print("\n" + "="*60)
        
        # Show evolution progress
        ops_until_evolution = 100 - (current['operation_count'] % 100)
        print(f"⏳ Next evolution in {ops_until_evolution} operations")
        
        # Show recommendations
        print("\n💡 Recommendations:")
        if current['cache_hit_rate'] < 0.5:
            print("  • Low cache hit rate - consider adjusting cache size")
        if current['avg_response_ms'] > 50:
            print("  • Response time could be improved - check index optimization")
        if current['generation'] < 10:
            print("  • System is still learning - continue providing feedback")
        
        print("="*60 + "\n")

def main():
    tracker = NeuralMemoryTracker()
    tracker.display_dashboard()

if __name__ == "__main__":
    main()