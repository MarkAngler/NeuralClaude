#\!/bin/bash
# Post-tool hook: Extract wisdom after solving problems

# Read JSON input from stdin
INPUT=$(cat)

# Extract tool name and result from JSON
TOOL_NAME=$(echo "$INPUT"  < /dev/null |  grep -o '"tool_name"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*: *"\([^"]*\)".*/\1/')
# For PostToolUse, we need to check tool_output instead of tool_result
TOOL_OUTPUT=$(echo "$INPUT" | grep -o '"tool_output"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*: *"\([^"]*\)".*/\1/')

# Extract session ID
SESSION_ID=$(echo "$INPUT" | grep -o '"session_id"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*: *"\([^"]*\)".*/\1/')
SESSION_ID="${SESSION_ID:-default}"
WISDOM_FILE="/tmp/claude-wisdom-${SESSION_ID}.pending"

# Pattern detection for successful problem solving
SUCCESS_PATTERNS="fixed|solved|working|complete|success|implemented|resolved"
ERROR_PATTERNS="error|failed|wrong|issue|problem|bug"

# Check if result contains success after errors (learning opportunity)
if echo "$TOOL_OUTPUT" | grep -qiE "$ERROR_PATTERNS" && echo "$TOOL_OUTPUT" | grep -qiE "$SUCCESS_PATTERNS"; then
    cat << EOF2 > "$WISDOM_FILE"
🧠 WISDOM EXTRACTION REQUIRED\!

You just solved a problem. Store the lesson learned:

mcp__neural-memory__store_memory
  key: "wisdom/[category]/[specific-pattern]"
  content: "
    PROBLEM: [What went wrong]
    SOLUTION: [What fixed it]
    PATTERN: [Reusable insight]
    CONFIDENCE: [High/Medium/Low]
    TIME_WASTED: [Estimate]
  "

Then provide feedback:
mcp__neural-memory__provide_feedback
  operation_id: "[from store response]"
  score: [0.0-1.0]
  success: true

⚠️  This wisdom could save 30+ minutes next time\!
EOF2
    
    echo "$(cat $WISDOM_FILE)"
    echo "WISDOM_PENDING" >> "/tmp/claude-session-${SESSION_ID}.state"
fi

# Track if wisdom was stored
if [[ "$TOOL_NAME" == "mcp__neural-memory__store_memory" ]] && echo "$TOOL_OUTPUT" | grep -q "wisdom/"; then
    sed -i '/WISDOM_PENDING/d' "/tmp/claude-session-${SESSION_ID}.state" 2>/dev/null
    echo "✅ Wisdom captured\! This will help future sessions."
fi

exit 0
