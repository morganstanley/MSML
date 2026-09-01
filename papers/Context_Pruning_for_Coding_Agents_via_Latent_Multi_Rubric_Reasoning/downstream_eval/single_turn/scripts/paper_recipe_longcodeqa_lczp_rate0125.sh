#!/usr/bin/env bash
# LongCodeQA reproduction for the v2 sem+dep checkpoint.
# H100 x 8 layout used for the reported run:
#   GPU 0: pruner (online_serving)
#   GPU 1: embedder (unixcoder-base)
#   GPU 2: main vLLM (TP=1)
# Set NUM_EXAMPLES=2 for smoke; default 113 for paper-faithful 32K split.
#
# Methods (paper script): full, no_context, rag, llmlingua2, longcodezip,
#                         longcodezip_with_pruner, selective_context
# Skips any method whose SCORES.json already exists.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_DIR="${REPO_ROOT}/downstream_eval/single_turn/hard_code_pruner/downstream_task/longcodeqa"

ORIG_MODEL="${ORIG_MODEL:-${SWEPRUNER_MODEL_PATH:-${REPO_ROOT}/runtime_models/swe-pruner-py-v2-semdep-5ep-8192}}"
QWEN_MODEL="${QWEN_MODEL:-${REPO_ROOT}/hf_models/Qwen2.5-Coder-7B-Instruct}"
EMBED_MODEL="${EMBED_MODEL:-${REPO_ROOT}/hf_models/unixcoder-base}"
LLMLINGUA2_MODEL="${LLMLINGUA2_MODEL:-${REPO_ROOT}/hf_models/llmlingua-2-xlm-roberta-large-meetingbank}"
LONGCODEZIP_MODEL="${LONGCODEZIP_MODEL:-${REPO_ROOT}/hf_models/Seed-Coder-8B-Instruct}"
SUMMARY_MODEL="${SUMMARY_MODEL:-${REPO_ROOT}/hf_models/Qwen3-0.6B}"
LCQA_DATASET="${LCQA_DATASET:-${REPO_ROOT}/downstream_eval/single_turn/datasets/longcodeqa_32k.jsonl}"

LABEL="${LABEL:-v2semdep-8192-lczp-r0125-th055-rankTrue}"
NUM_EXAMPLES="${NUM_EXAMPLES:-113}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/downstream_eval/results/longcodeqa-${LABEL}}"
LOG_ROOT="${RESULT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

PORT="${PORT:-8037}"
PRUNER_CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES:-0}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-1,2}"

# Best reported v2 sem+dep LongCodeQA cell:
#   longcodezip_with_pruner rate=0.125, rank_only=True, threshold=0.55
# Other methods can still be run by overriding METHODS_CSV.
METHODS_CSV="${METHODS_CSV:-longcodezip_with_pruner}"

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
    echo "Starting pruner on port ${PORT} with ${ORIG_MODEL}"
    CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES}" \
    SWEPRUNER_MODEL_PATH="${ORIG_MODEL}" \
    PYTHONPATH="${REPO_ROOT}/swe-pruner/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m swe_pruner.online_serving --port "${PORT}" --host 127.0.0.1 \
        >"${LOG_ROOT}/pruner_server.log" 2>&1 &
    PRUNER_PID=$!
    local attempt=0
    while true; do
        if ! kill -0 "${PRUNER_PID}" 2>/dev/null; then
            echo "[ERROR] pruner exited early"; tail -n 60 "${LOG_ROOT}/pruner_server.log"; return 1
        fi
        if curl -s -f "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then return 0; fi
        attempt=$((attempt + 1))
        if [ $((attempt % 15)) -eq 0 ]; then echo "waiting pruner ${attempt}..."; fi
        sleep 2
    done
}

stop_pruner() {
    if [ -n "${PRUNER_PID}" ] && kill -0 "${PRUNER_PID}" 2>/dev/null; then
        kill "${PRUNER_PID}" || true
        wait "${PRUNER_PID}" 2>/dev/null || true
    fi
    PRUNER_PID=""
}

IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"
for method in "${METHODS[@]}"; do
    RUN_DIR="${RESULT_ROOT}/method_${method}"
    LOG_FILE="${LOG_ROOT}/${method}.log"
    mkdir -p "${RUN_DIR}"

    if find "${RUN_DIR}" -name "*SCORES.json" 2>/dev/null | grep -q .; then
        echo "[skip] LongCodeQA method=${method} already has SCORES on disk"
        continue
    fi

    echo "=========================================="
    echo "Running LongCodeQA-32K method=${method}  n=${NUM_EXAMPLES}"
    echo "=========================================="

    ARGS=(
        --model_name "${QWEN_MODEL}"
        --method "${method}"
        --result_dir "${RUN_DIR}"
        --dataset_path "${LCQA_DATASET}"
        --num_examples "${NUM_EXAMPLES}"
        --batch_size "${BATCH_SIZE}"
        --tensor_parallel_size 1
        --max_model_len 32768
        --embedder_type bert
        --embed_model_name "${EMBED_MODEL}"
    )

    if [[ "${method}" == "rag" ]]; then
        ARGS+=(--rag_window_size 90 --rag_overlap 10 --rag_top_k 8)
    elif [[ "${method}" == "longcodezip" ]]; then
        ARGS+=(--longcodezip_model_name "${LONGCODEZIP_MODEL}" --longcodezip_rate 0.4 --longcodezip_rank_only False
               --qwen_summary_model_name "${SUMMARY_MODEL}")
    elif [[ "${method}" == "longcodezip_with_pruner" ]]; then
        start_pruner
        ARGS+=(
            --pruner_type online_rerank
            --rerank_api_base "http://127.0.0.1:${PORT}"
            --rerank_threshold "${RERANK_THRESHOLD:-0.55}"
            --rerank_always_keep_first_frags False
            --rerank_aggregate_method line
            --rerank_language python
            --longcodezip_model_name "${LONGCODEZIP_MODEL}"
            --longcodezip_rate "${LONGCODEZIP_RATE:-0.125}"
            --longcodezip_rank_only "${LONGCODEZIP_RANK_ONLY:-True}"
            --qwen_summary_model_name "${SUMMARY_MODEL}"
        )
    elif [[ "${method}" == "llmlingua2" ]]; then
        ARGS+=(--llmlingua2_model_name "${LLMLINGUA2_MODEL}" --llmlingua2_rate 0.3)
    elif [[ "${method}" == "selective_context" ]]; then
        ARGS+=(--selective_context_model_type gpt2 --selective_context_lang en --selective_context_reduce_ratio 0.75)
    fi

    CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
    PYTHONPATH="${REPO_ROOT}/swe-pruner/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${WORK_DIR}/eval.py" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
done

stop_pruner
echo "Paper-recipe LongCodeQA-32K reproduction (${LABEL}) finished. Results in ${RESULT_ROOT}"
