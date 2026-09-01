#!/bin/bash
# Usage:
#   ./train/train_llm_v2_crf_semantic_context.sh <NUM_GPUS> <INPUT_FILE> [extra train args...]
#
# Example:
#   ./train/train_llm_v2_crf_semantic_context.sh 8 \
#     swe-pruner-training-dataset-py-v2-all-rubric-fixed.jsonl \
#     --log-dir llm_experiments/swe-pruner-py-v2-crf-semantic-context
#
# Dedicated wrapper for the v2 CRF path with:
#   - max_length=4096
#   - compression_head_type=crf
#   - use_final_crf enabled
#   - 2 objectives: semantic,context
#   - MoE gating enabled

set -euo pipefail

NUM_GPUS="${1:?Usage: train_llm_v2_crf_semantic_context.sh NUM_GPUS INPUT_FILE [extra args...]}"
INPUT_FILE="${2:?Usage: train_llm_v2_crf_semantic_context.sh NUM_GPUS INPUT_FILE [extra args...]}"
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
  --log-dir "$REPO_ROOT/llm_experiments/swe-pruner-py-v2-crf-semantic-context" \
  --num-finetune-layers 2 \
  --num-fusion-layers 1 \
  --batch-size 16 \
  --max-length 4096 \
  --compression-head-type crf \
  --compression-loss-type focal \
  --dropout 0.4 \
  --auto-focal-alpha \
  --use-multi-layer-fusion \
  --use-sample-level-aggregation \
  --split-strategy label-stratified \
  --num-objectives 2 \
  --objective-names semantic,context \
  --objective-weights 1.0,0.5 \
  --use-moe-gating \
  --gating-type softmax \
  --use-final-crf \
  --lambda-rubric 0.6 \
  --gate-entropy-weight 0.002 \
  "$@"
