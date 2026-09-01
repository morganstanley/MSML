#!/bin/bash
set -euo pipefail

echo "=== Experiment 1: No-Correction Ablation ==="
date

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="<REPO_DIR>"
DATA_DIR="<DATA_DIR>"

# ── Parameters ────────────────────────────────────────────────────────────────
SCDD_CKPT="${DATA_DIR}/scdd_checkpoints/scdd.ckpt"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-512}"
NUM_STEPS="${NUM_STEPS:-128}"
NUCLEUS_P="${NUCLEUS_P:-0.9}"
NUM_BATCHES="${NUM_BATCHES:-16}"
OUTPUT_DIR="${DATA_DIR}/correction_results"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_PATH="${OUTPUT_DIR}/ablation_correction.json"

export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Configuration:"
echo "  SCDD_CKPT    : ${SCDD_CKPT}"
echo "  BATCH_SIZE   : ${BATCH_SIZE}"
echo "  SEQ_LEN      : ${SEQ_LEN}"
echo "  NUM_STEPS    : ${NUM_STEPS}"
echo "  NUCLEUS_P    : ${NUCLEUS_P}"
echo "  NUM_BATCHES  : ${NUM_BATCHES}"
echo "  OUTPUT_PATH  : ${OUTPUT_PATH}"
echo ""

cd "${REPO_DIR}"
python -u "${REPO_DIR}/supplementary/correction/ablation_correction.py" \
    --checkpoint_path "${SCDD_CKPT}" \
    --num_steps "${NUM_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --seq_len "${SEQ_LEN}" \
    --nucleus_p "${NUCLEUS_P}" \
    --num_batches "${NUM_BATCHES}" \
    --output_path "${OUTPUT_PATH}"

echo ""
echo "=== Done ==="
date
