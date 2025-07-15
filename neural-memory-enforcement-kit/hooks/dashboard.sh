#\!/bin/bash
# Neural-Memory Enforcement Dashboard

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        🧠 NEURAL-MEMORY ENFORCEMENT DASHBOARD 🧠             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Get all session files
SESSION_FILES=($(ls /tmp/claude-session-*.state 2>/dev/null))

if [ ${#SESSION_FILES[@]} -eq 0 ]; then
    echo "No active sessions found."
    exit 0
fi

# Aggregate metrics across all sessions
TOTAL_SEARCHES=0
TOTAL_WISDOMS=0
TOTAL_FEEDBACKS=0
TOTAL_PROBLEMS=0

for SESSION in "${SESSION_FILES[@]}"; do
    SEARCHES=$(grep -c "memory_searched" "$SESSION" 2>/dev/null || echo "0")
    WISDOMS=$(grep -c "wisdom_stored" "$SESSION" 2>/dev/null || echo "0")
    FEEDBACKS=$(grep -c "feedback_provided" "$SESSION" 2>/dev/null || echo "0")
    PROBLEMS=$(grep -c "fixed\ < /dev/null | solved\|complete" "$SESSION" 2>/dev/null || echo "0")
    
    TOTAL_SEARCHES=$((TOTAL_SEARCHES + SEARCHES))
    TOTAL_WISDOMS=$((TOTAL_WISDOMS + WISDOMS))
    TOTAL_FEEDBACKS=$((TOTAL_FEEDBACKS + FEEDBACKS))
    TOTAL_PROBLEMS=$((TOTAL_PROBLEMS + PROBLEMS))
    
    # Show per-session details
    SESSION_ID=$(basename "$SESSION" .state | sed 's/claude-session-//')
    echo "Session $SESSION_ID:"
    echo "  Searches: $SEARCHES | Wisdoms: $WISDOMS | Feedbacks: $FEEDBACKS"
done

# Calculate ratios
if [ "$TOTAL_PROBLEMS" -gt 0 ]; then
    WISDOM_RATIO=$(echo "scale=2; $TOTAL_WISDOMS * 100 / $TOTAL_PROBLEMS" | bc)%
else
    WISDOM_RATIO="N/A"
fi

if [ "$TOTAL_SEARCHES" -gt 0 ]; then
    FEEDBACK_RATIO=$(echo "scale=2; $TOTAL_FEEDBACKS * 100 / $TOTAL_SEARCHES" | bc)%
else
    FEEDBACK_RATIO="N/A"
fi

# Display aggregate metrics
echo
echo "📊 AGGREGATE METRICS"
echo "═══════════════════════════════════════════════════════════════"
printf "%-30s %10s\n" "Active Sessions:" "${#SESSION_FILES[@]}"
printf "%-30s %10s\n" "Total Memory Searches:" "$TOTAL_SEARCHES"
printf "%-30s %10s\n" "Total Wisdoms Stored:" "$TOTAL_WISDOMS"
printf "%-30s %10s\n" "Total Feedbacks Given:" "$TOTAL_FEEDBACKS"
printf "%-30s %10s\n" "Problems Solved:" "$TOTAL_PROBLEMS"
echo "═══════════════════════════════════════════════════════════════"
printf "%-30s %10s\n" "Wisdom Capture Rate:" "$WISDOM_RATIO"
printf "%-30s %10s\n" "Feedback Rate:" "$FEEDBACK_RATIO"
echo

# Check enforcement status
echo "🛡️  ENFORCEMENT STATUS"
echo "═══════════════════════════════════════════════════════════════"

# Check if hooks are active
if [ -f .claude/settings.json ] && grep -q "PreToolUse" .claude/settings.json; then
    echo "✅ Neural Memory Enforcement: ACTIVE"
else
    echo "❌ Neural Memory Enforcement: INACTIVE"
fi

# Check hook files
HOOKS=(
    ".claude/hooks/pre/memory-search-enforcer.sh"
    ".claude/hooks/post/wisdom-extractor.sh"
    ".claude/hooks/session/summary-enforcer.sh"
    ".claude/hooks/monitoring/realtime-tracker.sh"
)

for HOOK in "${HOOKS[@]}"; do
    if [ -x "$HOOK" ]; then
        echo "✅ $(basename $HOOK): INSTALLED"
    else
        echo "❌ $(basename $HOOK): MISSING"
    fi
done

echo

# Recommendations
if [ "$TOTAL_SEARCHES" -lt 3 ]; then
    echo "⚠️  RECOMMENDATION: Increase memory search frequency"
    echo "   Current: $TOTAL_SEARCHES | Target: 3+ per session"
fi

if [ "$WISDOM_RATIO" \!= "N/A" ] && [ "${WISDOM_RATIO%\%}" -lt 80 ]; then
    echo "⚠️  RECOMMENDATION: Improve wisdom capture rate"
    echo "   Current: $WISDOM_RATIO | Target: 80%+"
fi

# Recent violations
echo
echo "🚨 RECENT VIOLATIONS (Last 10)"
echo "═══════════════════════════════════════════════════════════════"
grep -h "BLOCKED\|WARNING" /tmp/claude-*.log 2>/dev/null | tail -10 || echo "No violations found"

echo
echo "Run this dashboard anytime with: bash .claude/hooks/dashboard.sh"
