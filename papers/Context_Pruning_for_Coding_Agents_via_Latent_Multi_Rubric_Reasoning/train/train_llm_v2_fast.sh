#!/bin/bash
# Usage:
#   ./train/train_llm_v2_fast.sh <NUM_GPUS> <INPUT_FILE> [extra train args...]
#
# Example:
#   ./train/train_llm_v2_fast.sh 8 \
#     swe-pruner-training-dataset-py-v2-all-rubric-fixed.jsonl \
#     --log-dir llm_experiments/swe-pruner-py-v2-fast-3obj
#
# This preserves the original train_llm.sh baseline path and provides a
# dedicated wrapper for the faster v2 FFN configuration:
#   - max_length=4096
#   - compression_head_type=ffn
#   - 3 objectives: semantic,dependency,context
#   - MoE gating enabled

set -euo pipefail

NUM_GPUS="${1:?Usage: train_llm_v2_fast.sh NUM_GPUS INPUT_FILE [extra args...]}"
INPUT_FILE="${2:?Usage: train_llm_v2_fast.sh NUM_GPUS INPUT_FILE [extra args...]}"
shift 2 || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_MODEL_PATH="$REPO_ROOT/hf_models/Qwen3-Reranker-0.6B"
MASTER_PORT="${MASTER_PORT:-29500}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

"$TORCHRUN_BIN" \
  --nproc_per_node="$NUM_GPUS" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port="$MASTER_PORT" \
  --module train.train_llm.train \
  -i "$INPUT_FILE" \
  --model-name "$DEFAULT_MODEL_PATH" \
  --epochs 3 \
  --lr 1e-4 \
  --log-dir "$REPO_ROOT/llm_experiments/swe-pruner-py-v2-fast-3obj" \
  --num-finetune-layers 2 \
  --num-fusion-layers 1 \
  --batch-size 16 \
  --max-length 4096 \
  --compression-head-type ffn \
  --compression-loss-type focal \
  --dropout 0.4 \
  --auto-focal-alpha \
  --use-multi-layer-fusion \
  --use-sample-level-aggregation \
  --split-strategy label-stratified \
  --num-objectives 3 \
  --objective-names semantic,dependency,context \
  --objective-weights 1.0,0.7,0.5 \
  --use-moe-gating \
  --gating-type softmax \
  --lambda-rubric 0.6 \
  --gate-entropy-weight 0.002 \
  "$@"
