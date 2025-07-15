#\!/bin/bash
# Notification hook: Real-time monitoring of neural-memory usage

# Read JSON input from stdin
INPUT=$(cat)

# Extract session ID
SESSION_ID=$(echo "$INPUT"  < /dev/null |  grep -o '"session_id"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*: *"\([^"]*\)".*/\1/')
SESSION_ID="${SESSION_ID:-default}"

METRICS_FILE="/tmp/claude-metrics-${SESSION_ID}.json"
OPERATION_COUNT_FILE="/tmp/claude-ops-${SESSION_ID}.count"
SESSION_FILE="/tmp/claude-session-${SESSION_ID}.state"

# Increment operation counter
CURRENT_OPS=$(cat "$OPERATION_COUNT_FILE" 2>/dev/null || echo "0")
CURRENT_OPS=$((CURRENT_OPS + 1))
echo "$CURRENT_OPS" > "$OPERATION_COUNT_FILE"

# Check every 5 operations
if [ $((CURRENT_OPS % 5)) -eq 0 ]; then
    # Get current session metrics
    MEMORIES_SEARCHED=$(grep -c "memory_searched" "$SESSION_FILE" 2>/dev/null || echo "0")
    WISDOMS_STORED=$(grep -c "wisdom_stored" "$SESSION_FILE" 2>/dev/null || echo "0")
    FEEDBACKS_PROVIDED=$(grep -c "feedback_provided" "$SESSION_FILE" 2>/dev/null || echo "0")
    
    # Calculate ratios
    SEARCH_RATIO=$(echo "scale=2; $MEMORIES_SEARCHED / 5" | bc)
    FEEDBACK_RATIO=$(echo "scale=2; $FEEDBACKS_PROVIDED / $MEMORIES_SEARCHED" | bc 2>/dev/null || echo "0")
    
    # Generate reminder if below threshold
    if (( $(echo "$SEARCH_RATIO < 0.4" | bc -l) )); then
        cat << EOF2

🔔 NEURAL-MEMORY REMINDER (Operation #$CURRENT_OPS)
════════════════════════════════════════════════
📊 Last 5 operations:
   Memory Searches: $MEMORIES_SEARCHED (Target: 2+)
   Wisdoms Stored:  $WISDOMS_STORED
   Feedbacks Given: $FEEDBACKS_PROVIDED

⚠️  Low memory usage detected\!

Before your next code change:
1. Search: mcp__neural-memory__search_memory
   Query: "[framework] [feature] limitations patterns"
   
2. After search: mcp__neural-memory__provide_feedback
   Score the relevance (0.0-1.0)

This prevents repeating past mistakes\!
EOF2
    fi
    
    # Alert on missing feedback
    if [ "$MEMORIES_SEARCHED" -gt 0 ] && [ "$FEEDBACKS_PROVIDED" -eq 0 ]; then
        echo "❗ Feedback Missing: You searched memories but didn't rate them\!"
        echo "   This breaks the learning loop. Always provide feedback\!"
    fi
fi

# Track specific operations for metrics (extract tool name)
TOOL_NAME=$(echo "$INPUT" | grep -o '"tool_name"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*: *"\([^"]*\)".*/\1/')

if [[ "$TOOL_NAME" == "mcp__neural-memory__search_memory" ]]; then
    echo "memory_searched=$(date +%s)" >> "$SESSION_FILE"
elif [[ "$TOOL_NAME" == "mcp__neural-memory__store_memory" ]] && echo "$INPUT" | grep -q "wisdom/"; then
    echo "wisdom_stored=$(date +%s)" >> "$SESSION_FILE"
elif [[ "$TOOL_NAME" == "mcp__neural-memory__provide_feedback" ]]; then
    echo "feedback_provided=$(date +%s)" >> "$SESSION_FILE"
fi

exit 0
