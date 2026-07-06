#!/bin/bash
# ==========================================================================
# Paper Run: LLM Speedrun — GPT-5.2
# ==========================================================================
# Launch in tmux. Expects OPENAI_API_KEY to be set.
#
# Results: paper_final_results/llm_speedrun/gpt52/
# ==========================================================================

set -euo pipefail

PY="${ALPHALAB_PYTHON:-python3}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$DIR/paper_final_results/llm_speedrun/gpt52"

mkdir -p "$WORKSPACE"

echo "=== LLM Speedrun — GPT-5.2 ==="
echo "Config:    $DIR/data/paper_llm_speedrun_gpt.json"
echo "Workspace: $WORKSPACE"
echo "Python:    $PY"
echo "Started:   $(date)"
echo ""

$PY "$DIR/run.py" \
  --config "$DIR/data/paper_llm_speedrun_gpt.json" \
  --workspace "$WORKSPACE" \
  2>&1 | tee "$WORKSPACE/run.log"
