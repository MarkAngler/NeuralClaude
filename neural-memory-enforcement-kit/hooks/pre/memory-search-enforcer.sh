#\!/bin/bash
# Pre-tool hook: Enforce memory search before code operations

# Read JSON input from stdin
INPUT=$(cat)

# Extract tool name from JSON using grep and sed (portable approach)
TOOL_NAME=$(echo "$INPUT"  < /dev/null |  grep -o '"tool_name"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*: *"\([^"]*\)".*/\1/')

# Extract session ID from JSON
SESSION_ID=$(echo "$INPUT" | grep -o '"session_id"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*: *"\([^"]*\)".*/\1/')
SESSION_ID="${SESSION_ID:-default}"
SESSION_FILE="/tmp/claude-session-${SESSION_ID}.state"

# Debug logging
echo "[$(date)] Tool: $TOOL_NAME, Session: $SESSION_ID" >> /tmp/claude-enforcement.log

# Check if this is a code-writing operation
if [[ "$TOOL_NAME" =~ ^(Edit|Write|MultiEdit)$ ]]; then
    # Check if memory search was performed in this session
    if [ \! -f "$SESSION_FILE" ] || \! grep -q "memory_searched" "$SESSION_FILE"; then
        # Output JSON response to block the operation
        cat << EOF2
{
  "decision": "block",
  "reason": "❌ BLOCKED: Must search neural-memory before writing code\!\n\nRequired: mcp__neural-memory__search_memory with relevant query\nExample: Search for 'framework patterns' before implementing features"
}
EOF2
        exit 2  # Exit code 2 blocks the operation
    fi
fi

# Check if this is a memory search operation
# Handle both possible formats: mcp__neural-memory__search_memory or neural_memory
if [[ "$TOOL_NAME" =~ neural.?memory.?search || "$TOOL_NAME" == "search_memory" ]]; then
    echo "memory_searched=$(date +%s)" >> "$SESSION_FILE"
    echo "[$(date)] Memory search recorded for session $SESSION_ID" >> /tmp/claude-enforcement.log
fi

# Allow the operation by default (no output needed for approval)
exit 0
