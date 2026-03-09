#!/bin/bash
# AFL Live Learning Pipeline — Daily Trigger
#
# After each round: scrapes results from FootyWire, rebuilds data,
# learns from completed rounds, and predicts the next round.
#
# Setup cron (runs at 6am daily during AFL season, Mar-Sep):
#   crontab -e
#   0 6 * * * /Users/akash/Desktop/AFL/daily_update.sh >> /Users/akash/Desktop/AFL/logs/daily.log 2>&1
#
# Or run manually:
#   ./daily_update.sh
#   ./daily_update.sh 2026

cd "$(dirname "$0")"
mkdir -p logs

YEAR="${1:-2026}"

echo "=== AFL Live Learning $(date) ==="
python3 pipeline.py --daily --year "$YEAR"
echo "=== Done $(date) ==="
