#!/bin/bash
# Daily job: refresh data, then rebalance (rebalance.py exits unless it's the first trading day of the month).
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"
mkdir -p logs
LOG="logs/$(date +%Y-%m-%d).log"

# Activate the conda env (Anaconda install in your home folder)
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate deskfund

{
  echo "=== $(date) ==="
  python scripts/ingest.py
  python scripts/rebalance.py
} >> "$LOG" 2>&1
