#!/bin/bash
# SCDD vs GIDD+ Semantic Comparison (shared corruption only, single GPU)
# Usage: bash eval_semantic_comparison.sh
# Or with overrides: T_START=0.8 bash eval_semantic_comparison.sh
# NOTE: This script requires two conda environments: scdd and gidd.

set -euo pipefail

# ── Paths (adapt to your setup) ──────────────────────────────────────────────
REPO_DIR="<REPO_DIR>"
STORAGE_BASE="<FAST_STORAGE_DIR>"

export HF_HOME=${STORAGE_BASE}/hf_cache
export TRANSFORMERS_CACHE=${HF_HOME}

# ── Checkpoints ──────────────────────────────────────────────────────────────
SCDD_CKPT="${SCDD_CKPT:-${STORAGE_BASE}/scdd_checkpoints/checkpoint.ckpt}"
GIDD_CKPT="${GIDD_CKPT:-${STORAGE_BASE}/gidd_checkpoints/model.pt}"

# ── Experiment settings ──────────────────────────────────────────────────────
T_START="${T_START:-0.5}"
EXP_TAG="${EXP_TAG:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUCLEUS_P="${NUCLEUS_P:-0.9}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4}"
OPENAI_API_KEY="${OPENAI_API_KEY:?ERROR: OPENAI_API_KEY must be set}"

# ── Output directories ───────────────────────────────────────────────────────
T_START_SUFFIX="t$(echo "${T_START}" | tr -d '.')"
RUN_SUFFIX="shared_${T_START_SUFFIX}${EXP_TAG:+_${EXP_TAG}}"

WORK_DIR="${WORK_DIR:-${STORAGE_BASE}/eval_comparison/${RUN_SUFFIX}/work}"
OUTPUT_DIR="${OUTPUT_DIR:-${STORAGE_BASE}/eval_comparison/${RUN_SUFFIX}/results}"
mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}"

EVAL_SCRIPT="${REPO_DIR}/supplementary/llm_as_a_judge/eval_semantic_comparison.py"

echo "=== SCDD vs GIDD+ Semantic Comparison ==="
echo "  SCDD_CKPT  : ${SCDD_CKPT}"
echo "  GIDD_CKPT  : ${GIDD_CKPT}"
echo "  T_START    : ${T_START}"
echo "  WORK_DIR   : ${WORK_DIR}"
echo "  OUTPUT_DIR : ${OUTPUT_DIR}"
echo ""

# ── Stage 1: SCDD generation + shared corruption setup ──────────────────────
echo "Stage 1: SCDD generation (builds shared corruption)"
# Activate SCDD conda env and cd to its codebase root
conda activate scdd
cd "${REPO_DIR}"

python -u "${EVAL_SCRIPT}" \
    --stage scdd \
    --scdd_ckpt "${SCDD_CKPT}" \
    --t_start "${T_START}" \
    --work_dir "${WORK_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --nucleus_p "${NUCLEUS_P}" \
    --rank 0 --world_size 1

# ── Stage 2: GIDD+ generation from the shared corruption ────────────────────
echo "Stage 2: GIDD+ generation"
# Activate GIDD conda env and cd to its codebase root
conda activate gidd
cd "${REPO_DIR}/gidd"

python -u "${EVAL_SCRIPT}" \
    --stage gidd \
    --gidd_ckpt "${GIDD_CKPT}" \
    --work_dir "${WORK_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --nucleus_p "${NUCLEUS_P}" \
    --rank 0 --world_size 1

# ── Stage 2.5: Merge text shards ─────────────────────────────────────────────
cd "${REPO_DIR}"
python -u "${EVAL_SCRIPT}" \
    --stage merge \
    --work_dir "${WORK_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --world_size 1

# ── Stage 3: LLM judging ─────────────────────────────────────────────────────
echo "Stage 3: LLM judging (${JUDGE_MODEL})"
python -u "${EVAL_SCRIPT}" \
    --stage judge \
    --work_dir "${WORK_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --judge_model "${JUDGE_MODEL}" \
    --openai_key "${OPENAI_API_KEY}" \
    --rank 0 --world_size 1

# ── Stage 4: Merge results & statistics ─────────────────────────────────────
echo "Stage 4: Merging results"
python -u "${EVAL_SCRIPT}" \
    --stage merge \
    --work_dir "${WORK_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --world_size 1

echo "=== Done. Results in: ${OUTPUT_DIR} ==="
