#!/bin/bash
# Final LaMR sem+dep 8192 training recipe.
#
# Usage:
#   ./train/train_llm_v2_crf_semdep_8192.sh <NUM_GPUS> <INPUT_FILE> [extra train args...]
#
# Example:
#   ./train/train_llm_v2_crf_semdep_8192.sh 8 \
#     /path/to/swe-pruner-training-dataset-py-v2-rubric.jsonl
#
# LOG_DIR, MODEL_NAME, BATCH_SIZE, LR, and TORCHRUN_BIN can be overridden
# through environment variables.

set -euo pipefail

NUM_GPUS="${1:?Usage: train_llm_v2_crf_semdep_8192.sh NUM_GPUS INPUT_FILE [extra args...]}"
INPUT_FILE="${2:?Usage: train_llm_v2_crf_semdep_8192.sh NUM_GPUS INPUT_FILE [extra args...]}"
shift 2 || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_NAME="${MODEL_NAME:-$REPO_ROOT/hf_models/Qwen3-Reranker-0.6B}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/llm_experiments/swe-pruner-py-v2-crf-semdep-8192}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
MASTER_PORT="${MASTER_PORT:-29500}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"

"$TORCHRUN_BIN" \
  --nproc_per_node="$NUM_GPUS" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port="$MASTER_PORT" \
  --module train.train_llm.train \
  -i "$INPUT_FILE" \
  --model-name "$MODEL_NAME" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --log-dir "$LOG_DIR" \
  --num-finetune-layers 2 \
  --num-fusion-layers 1 \
  --num-heads 8 \
  --batch-size "$BATCH_SIZE" \
  --max-length 8192 \
  --compression-head-type crf \
  --compression-loss-type focal \
  --dropout 0.4 \
  --auto-focal-alpha \
  --focal-gamma 2.0 \
  --use-multi-layer-fusion \
  --early-layer-ratio 0.25 \
  --middle-layer-ratio 0.5 \
  --use-sample-level-aggregation \
  --split-strategy label-stratified \
  --num-objectives 2 \
  --objective-names semantic,dependency \
  --objective-weights 1.0,0.8 \
  --use-moe-gating \
  --gating-type softmax \
  --use-final-crf \
  --lambda-score 0.05 \
  --lambda-rubric 0.3 \
  --gate-entropy-weight 0.002 \
  "$@"
