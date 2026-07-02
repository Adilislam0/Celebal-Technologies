#!/bin/bash
# Usage: bash scripts/run_scoring.sh <input_csv> <output_csv>
# Example: bash scripts/run_scoring.sh data/Leads.csv data/scored_leads.csv

INPUT=${1:-data/Leads.csv}
OUTPUT=${2:-data/scored_leads.csv}
CONFIG=config/scoring_config.json

echo "Scoring leads from: $INPUT"
python src/pipeline.py "$INPUT" "$OUTPUT" --config "$CONFIG"
