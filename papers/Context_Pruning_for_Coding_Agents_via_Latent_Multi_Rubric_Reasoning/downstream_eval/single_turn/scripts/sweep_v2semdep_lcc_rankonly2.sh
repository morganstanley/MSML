#!/usr/bin/env bash
# LCC reproduction sweep for the v2 sem+dep checkpoint.
# The reported best cell is:
#   longcodezip_with_pruner, rank_only=True, rate=0.25, threshold=0.55.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_DIR="${REPO_ROOT}/downstream_eval/single_turn/hard_code_pruner/downstream_task/LCC"

V2_CKPT="${V2_CKPT:-${SWEPRUNER_MODEL_PATH:-${REPO_ROOT}/runtime_models/swe-pruner-py-v2-semdep-5ep-8192}}"
QWEN_MODEL="${QWEN_MODEL:-${REPO_ROOT}/hf_models/Qwen2.5-Coder-7B-Instruct}"
EMBED_MODEL="${EMBED_MODEL:-${REPO_ROOT}/hf_models/unixcoder-base}"
LONGCODEZIP_MODEL="${LONGCODEZIP_MODEL:-${REPO_ROOT}/hf_models/Seed-Coder-8B-Instruct}"
LCC_DATASET="${LCC_DATASET:-${REPO_ROOT}/lcc}"

LABEL="${LABEL:-v2semdep-lcc-rankonly2}"
NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/downstream_eval/results/lcc-${LABEL}}"
LOG_ROOT="${RESULT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

PORT="${PORT:-8062}"
PRUNER_CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES:-0}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-1,2}"

PRUNER_PID=""
cleanup() {
    if [ -n "${PRUNER_PID}" ] && kill -0 "${PRUNER_PID}" 2>/dev/null; then
        kill "${PRUNER_PID}" || true
        wait "${PRUNER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

start_pruner() {
    if [ -n "${PRUNER_PID}" ] && kill -0 "${PRUNER_PID}" 2>/dev/null; then return 0; fi
    echo "Starting pruner on port ${PORT} with ${V2_CKPT}"
    CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES}" \
    SWEPRUNER_MODEL_PATH="${V2_CKPT}" \
    PYTHONPATH="${REPO_ROOT}/swe-pruner/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m swe_pruner.online_serving --port "${PORT}" --host 127.0.0.1 \
        >"${LOG_ROOT}/pruner_server.log" 2>&1 &
    PRUNER_PID=$!
    local attempt=0
    while true; do
        if ! kill -0 "${PRUNER_PID}" 2>/dev/null; then
            echo "[ERROR] pruner exited early"; tail -n 60 "${LOG_ROOT}/pruner_server.log"; return 1
        fi
        if curl -s -f "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "[ok] pruner ready"; return 0
        fi
        attempt=$((attempt + 1))
        if [ $((attempt % 15)) -eq 0 ]; then echo "waiting pruner ${attempt}..."; fi
        sleep 2
    done
}

run_cell() {
    local CELL="$1"; shift
    local RUN_DIR="${RESULT_ROOT}/${CELL}"
    local LOG_FILE="${LOG_ROOT}/${CELL}.log"
    mkdir -p "${RUN_DIR}"
    if find "${RUN_DIR}" -name "*SCORES.json" 2>/dev/null | grep -q .; then
        echo "[skip] ${CELL} already has SCORES on disk"; return 0
    fi
    echo "=========================================="
    echo "Cell ${CELL}  n=${NUM_EXAMPLES}"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
    LCC_TOKENIZER_MODEL="${QWEN_MODEL}" \
    PYTHONPATH="${REPO_ROOT}/swe-pruner/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${WORK_DIR}/main.py" "$@" --result_dir "${RUN_DIR}" 2>&1 | tee "${LOG_FILE}"
}

start_pruner

COMMON_ARGS=(
    --model_name "${QWEN_MODEL}"
    --dataset_path "${LCC_DATASET}"
    --dataset_split test
    --num_examples "${NUM_EXAMPLES}"
    --batch_size "${BATCH_SIZE}"
    --tensor_parallel_size 1
    --embed_model_name "${EMBED_MODEL}"
    --embed_model_type bert
    --syntax_check True
    --syntax_language python
    --syntax_check_chunk True
    --pruner_type online_rerank
    --rerank_api_base "http://127.0.0.1:${PORT}"
    --rerank_aggregate_method line
    --rerank_language python
    --method longcodezip_with_pruner
    --rerank_always_keep_first_frags False
    --longcodezip_model_name "${LONGCODEZIP_MODEL}"
    --longcodezip_rank_only True
)

# rate × th
GRID=(
    "0.25  0.45"
    "0.25  0.55"
    "0.125 0.58"
    "0.125 0.60"
    "0.125 0.62"
)
for entry in "${GRID[@]}"; do
    read -r RATE TH <<< "${entry}"
    run_cell "lczp_rankonly_r${RATE}_th${TH}" \
        "${COMMON_ARGS[@]}" \
        --longcodezip_rate "${RATE}" \
        --rerank_threshold "${TH}"
done

echo "rank_only2 sweep done. Results under ${RESULT_ROOT}"
