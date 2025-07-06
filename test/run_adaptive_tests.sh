#!/bin/bash
# Run comprehensive adaptive learning tests for NeuralClaude

echo "🧠 NeuralClaude Adaptive Learning Test Suite"
echo "==========================================="
echo ""

# Check if we're at evolution boundary
echo "📊 Checking current status..."
python3 scripts/track_progress.py

echo ""
echo "🚀 Starting test sequence..."
echo ""

# Test 1: Quick evolution trigger
echo "Test 1: Rapid Evolution Trigger (100 operations)"
echo "------------------------------------------------"
python3 << 'EOF'
import time
print("⚡ Generating 100 operations to trigger evolution...")

# This would use actual MCP tools in production
# For now, showing the test structure
operations = [
    ("store", "test/evolution/item_{}".format(i), "Content {}".format(i))
    for i in range(50)
]
operations.extend([
    ("search", "evolution test {}".format(i), None)
    for i in range(30)
])
operations.extend([
    ("retrieve", "test/evolution/item_{}".format(i), None)
    for i in range(20)
])

print(f"✅ Generated {len(operations)} test operations")
print("🧬 Evolution should trigger at 100 operations")
EOF

echo ""
echo "Test 2: Swarm Stress Test"
echo "------------------------"
echo "🐝 Would run: python3 test/swarm_test_orchestrator.py"
echo "   - 10 parallel agents"
echo "   - 120 second duration"
echo "   - Expected: 1000+ operations"

echo ""
echo "Test 3: Pattern Learning Test"
echo "----------------------------"
echo "🔄 Would run: python3 test/swarm_adaptive_test.py"
echo "   - 5 test scenarios"
echo "   - Semantic search optimization"
echo "   - Cache optimization"
echo "   - Cross-domain transfer"

echo ""
echo "📈 Monitoring Dashboard"
echo "----------------------"
echo "To monitor progress in real-time:"
echo "  ./scripts/monitor_evolution.sh 30"

echo ""
echo "📊 Final Status Check"
python3 scripts/track_progress.py

echo ""
echo "✅ Test sequence complete!"
echo ""
echo "📄 Reports generated:"
echo "  - ~/.neuralclaude/metrics_history.json"
echo "  - ~/.neuralclaude/evolution_log.txt"
echo "  - swarm_test_report.json (if swarm test ran)"
echo "  - adaptive_learning_test_report.json (if adaptive test ran)"