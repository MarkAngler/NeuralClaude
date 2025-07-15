#\!/bin/bash
# Stop/SubagentStop hook: Enforce session summary and wisdom check

# Read JSON input from stdin
INPUT=$(cat)

# Extract session ID from JSON
SESSION_ID=$(echo "$INPUT"  < /dev/null |  grep -o '"session_id"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*: *"\([^"]*\)".*/\1/')
SESSION_ID="${SESSION_ID:-default}"
SESSION_FILE="/tmp/claude-session-${SESSION_ID}.state"
METRICS_FILE="/tmp/claude-metrics-${SESSION_ID}.json"

# Calculate session metrics
PROBLEMS_SOLVED=$(grep -c "fixed\|solved\|complete" "$SESSION_FILE" 2>/dev/null || echo "0")
WISDOMS_STORED=$(grep -c "wisdom_stored" "$SESSION_FILE" 2>/dev/null || echo "0")
MEMORIES_SEARCHED=$(grep -c "memory_searched" "$SESSION_FILE" 2>/dev/null || echo "0")
FEEDBACKS_PROVIDED=$(grep -c "feedback_provided" "$SESSION_FILE" 2>/dev/null || echo "0")

# Check wisdom storage ratio
if [ "$PROBLEMS_SOLVED" -gt 0 ]; then
    WISDOM_RATIO=$(echo "scale=2; $WISDOMS_STORED / $PROBLEMS_SOLVED" | bc)
else
    WISDOM_RATIO="N/A"
fi

# Generate enforcement report
cat << EOF2

📊 NEURAL-MEMORY USAGE REPORT
════════════════════════════════════════
Problems Solved:    $PROBLEMS_SOLVED
Wisdoms Stored:     $WISDOMS_STORED  
Memory Searches:    $MEMORIES_SEARCHED
Feedbacks Given:    $FEEDBACKS_PROVIDED
Wisdom Ratio:       $WISDOM_RATIO

EOF2

# Check for pending wisdom
if grep -q "WISDOM_PENDING" "$SESSION_FILE" 2>/dev/null; then
    echo "❌ BLOCKED: Unsaved wisdom detected\!"
    echo "You solved problems but didn't store the learnings."
    echo "Required: Store wisdom for problems you solved"
    # Use JSON output for blocking
    echo '{"decision": "block", "reason": "Unsaved wisdom detected - store learnings before ending session"}'
    exit 2
fi

# Enforce minimum standards
if [ "$PROBLEMS_SOLVED" -gt 0 ] && [ "$WISDOMS_STORED" -eq 0 ]; then
    echo "⚠️  WARNING: No wisdom stored despite solving $PROBLEMS_SOLVED problems"
    echo "Required: Store at least one key learning"
    
    cat << EOF2

MANDATORY SESSION SUMMARY:
Run these commands before ending session:

mcp__neural-memory__store_memory
  key: "session/$(date +%Y%m%d)/summary"
  content: "
    SESSION: $(date)
    PROBLEMS_SOLVED: $PROBLEMS_SOLVED
    KEY_LEARNINGS: [List main insights]
    NEXT_STEPS: [What to remember]
    PATTERNS_NOTICED: [Recurring issues]
  "

EOF2
    echo '{"decision": "block", "reason": "No wisdom stored - capture at least one learning"}'
    exit 2
fi

# Success message
if [ "$WISDOM_RATIO" \!= "N/A" ] && (( $(echo "$WISDOM_RATIO >= 0.8" | bc -l) )); then
    echo "✅ Excellent wisdom capture rate: ${WISDOM_RATIO}"
fi

# Cleanup
rm -f "$SESSION_FILE" "$METRICS_FILE" 2>/dev/null

exit 0
