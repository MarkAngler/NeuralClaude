#!/bin/bash
# NeuralClaude Evolution Monitor
# Continuously monitors the neural memory system's evolution progress

INTERVAL=${1:-300}  # Default 5 minutes
LOG_FILE="$HOME/.neuralclaude/evolution_log.txt"

mkdir -p "$HOME/.neuralclaude"

echo "🧠 NeuralClaude Evolution Monitor Started"
echo "📊 Checking every $INTERVAL seconds"
echo "📝 Logging to: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] Checking evolution status..." | tee -a "$LOG_FILE"
    
    # Run the progress tracker
    python3 "$(dirname "$0")/track_progress.py" | tee -a "$LOG_FILE"
    
    # Sleep for the interval
    sleep "$INTERVAL"
done