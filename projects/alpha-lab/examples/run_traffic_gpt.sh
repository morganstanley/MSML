#!/bin/bash
# ==========================================================================
# Paper Run: Traffic Forecasting — GPT-5.2
# ==========================================================================
# Full pipeline run (Phase 0 → 1 → 2 → 3). GPT-5.2 is the reference model
# that builds the Phase 2 evaluation framework. Opus will reuse this
# framework for apples-to-apples comparison.
#
# Dataset: 862 road sensors, hourly occupancy (0-1), ~17 months, 24h horizon
# Metric:  RMSE (minimize) — values already normalized, directly comparable
# Budget:  50 experiments on 4x H100
#
# Launch in tmux. Expects OPENAI_API_KEY to be set.
#
# Results: paper_final_results/traffic/gpt52/
# ==========================================================================

set -euo pipefail

PY="${ALPHALAB_PYTHON:-python3}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$DIR/paper_final_results/traffic/gpt52"

mkdir -p "$WORKSPACE"

echo "=== Traffic Forecasting — GPT-5.2 ==="
echo "Config:    $DIR/data/paper_traffic_gpt.json"
echo "Workspace: $WORKSPACE"
echo "Python:    $PY"
echo "Started:   $(date)"
echo ""

$PY "$DIR/run.py" \
  --config "$DIR/data/paper_traffic_gpt.json" \
  --workspace "$WORKSPACE" \
  2>&1 | tee "$WORKSPACE/run.log"
