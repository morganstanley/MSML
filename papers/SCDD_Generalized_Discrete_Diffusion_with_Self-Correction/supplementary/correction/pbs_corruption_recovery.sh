#!/bin/bash
set -euo pipefail

echo "=== Experiment 2: Corruption Recovery at Last Step ==="
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
CORRUPT_COUNTS="${CORRUPT_COUNTS:-5,10,20,50}"
NUM_BATCHES="${NUM_BATCHES:-8}"
OUTPUT_DIR="${DATA_DIR}/correction_results"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_PATH="${OUTPUT_DIR}/corruption_recovery.json"

export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Configuration:"
echo "  SCDD_CKPT       : ${SCDD_CKPT}"
echo "  BATCH_SIZE      : ${BATCH_SIZE}"
echo "  SEQ_LEN         : ${SEQ_LEN}"
echo "  NUM_STEPS       : ${NUM_STEPS}"
echo "  NUCLEUS_P       : ${NUCLEUS_P}"
echo "  CORRUPT_COUNTS  : ${CORRUPT_COUNTS}"
echo "  NUM_BATCHES     : ${NUM_BATCHES}"
echo "  OUTPUT_PATH     : ${OUTPUT_PATH}"
echo ""

cd "${REPO_DIR}"
python -u "${REPO_DIR}/supplementary/correction/corruption_recovery.py" \
    --checkpoint_path "${SCDD_CKPT}" \
    --num_steps "${NUM_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --seq_len "${SEQ_LEN}" \
    --nucleus_p "${NUCLEUS_P}" \
    --corrupt_counts "${CORRUPT_COUNTS}" \
    --num_batches "${NUM_BATCHES}" \
    --data_cache_dir "${DATA_DIR}/data" \
    --output_path "${OUTPUT_PATH}"

echo ""
echo "=== Done ==="
date
