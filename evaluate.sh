#!/bin/bash

# Simple Evaluation Script

SCORER="./evaluation/NADI2024-ST1-Scorer.py"
GOLD="./evaluation/NADI2024_subtask1_dev2_gold.txt"
PREDICTIONS="output.txt"

if [ -z "$PREDICTIONS" ]; then
    echo "Usage: $0 <predictions_file>"
    echo ""
    echo "Example:"
    echo "  $0 ./output.txt"
    echo "  $0 /path/to/predictions.txt"
    exit 1
fi

if [ ! -f "$PREDICTIONS" ]; then
    echo "ERROR: Predictions file not found: $PREDICTIONS"
    exit 1
fi

echo "Running evaluation..."
python3 "$SCORER" "$GOLD" "$PREDICTIONS"