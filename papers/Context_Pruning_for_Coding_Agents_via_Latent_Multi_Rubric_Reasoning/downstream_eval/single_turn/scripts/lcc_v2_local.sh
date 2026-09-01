#!/usr/bin/env bash
set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "Error: this script must be run with bash" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_DIR="${REPO_ROOT}/downstream_eval/single_turn/hard_code_pruner/downstream_task/LCC"
SERVER_SCRIPT="${REPO_ROOT}/tests/smoke_prune_http_server.py"
PRUNER_CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES:-1}"
MAIN_CUDA_VISIBLE_DEVICES="${MAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL_NAME="${1:-${REPO_ROOT}/hf_models/Qwen2.5-Coder-7B-Instruct}"
PRUNER_MODEL_PATH="${2:-${REPO_ROOT}/runtime_models/swe-pruner-py-v2-semdep-5ep-8192}"
RESULT_DIR="${3:-${REPO_ROOT}/downstream_eval/results/lcc-v2-semdep-5ep-8192}"
DATASET_PATH="${4:-${REPO_ROOT}/lcc}"
EMBED_MODEL_NAME="${5:-${LCC_EMBED_MODEL:-${REPO_ROOT}/hf_models/unixcoder-base}}"
NUM_EXAMPLES="${6:-200}"
BATCH_SIZE="${7:-16}"
TENSOR_PARALLEL_SIZE="${8:-2}"
TOP_K="${9:-5}"
RERANK_API_PORT="${10:-8400}"
DATASET_SPLIT="${11:-test}"
METHODS_CSV="${METHODS_CSV:-full,no_context,selective_context,llmlingua2,rag,longcodezip,rag_with_pruner}"

LONGCODEZIP_MODEL_NAME="${LONGCODEZIP_MODEL_NAME:-${REPO_ROOT}/hf_models/Seed-Coder-8B-Instruct}"
LLMLINGUA2_MODEL_NAME="${LLMLINGUA2_MODEL_NAME:-${REPO_ROOT}/hf_models/llmlingua-2-xlm-roberta-large-meetingbank}"
SELECTIVE_CONTEXT_MODEL_TYPE="${SELECTIVE_CONTEXT_MODEL_TYPE:-gpt2}"
RAG_WINDOW_SIZE="${RAG_WINDOW_SIZE:-80}"
RAG_OVERLAP="${RAG_OVERLAP:-40}"
RAG_TOP_K="${RAG_TOP_K:-${TOP_K}}"
RAG_WITH_PRUNER_WINDOW_SIZE="${RAG_WITH_PRUNER_WINDOW_SIZE:-${RAG_WINDOW_SIZE}}"
RAG_WITH_PRUNER_OVERLAP="${RAG_WITH_PRUNER_OVERLAP:-${RAG_OVERLAP}}"
RAG_WITH_PRUNER_TOP_K="${RAG_WITH_PRUNER_TOP_K:-${TOP_K}}"
RERANK_THRESHOLD="${RERANK_THRESHOLD:-0.42}"
RERANK_ALWAYS_KEEP_FIRST_FRAGS="${RERANK_ALWAYS_KEEP_FIRST_FRAGS:-True}"
RERANK_AGGREGATE_METHOD="${RERANK_AGGREGATE_METHOD:-line}"
RERANK_LANGUAGE="${RERANK_LANGUAGE:-python}"

BASE_LOG_DIR="${RESULT_DIR}/logs"
mkdir -p "${BASE_LOG_DIR}" "${RESULT_DIR}"

if [ ! -d "${PRUNER_MODEL_PATH}" ]; then
    echo "Error: pruner model directory not found at ${PRUNER_MODEL_PATH}" >&2
    exit 1
fi

if [ ! -f "${PYTHON_BIN}" ]; then
    echo "Error: python binary not found at ${PYTHON_BIN}" >&2
    exit 1
fi

if [ ! -f "${SERVER_SCRIPT}" ]; then
    echo "Error: pruner server script not found at ${SERVER_SCRIPT}" >&2
    exit 1
fi

if [ ! -d "${DATASET_PATH}" ]; then
    echo "Error: LCC dataset dir not found at ${DATASET_PATH}" >&2
    exit 1
fi

check_port() {
    local port="$1"
    if lsof -Pi :"${port}" -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

wait_for_server() {
    local url="$1"
    local pid="$2"
    local log_file="$3"
    local attempt=0

    while true; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "Pruner server exited unexpectedly." >&2
            tail -n 200 "${log_file}" >&2 || true
            return 1
        fi

        if curl -s -f "${url}/health" >/dev/null 2>&1; then
            return 0
        fi

        attempt=$((attempt + 1))
        if [ $((attempt % 15)) -eq 0 ]; then
            echo "Waiting for pruner server on ${url} ..."
        fi
        sleep 2
    done
}

cleanup() {
    if [ -n "${RERANK_SERVER_PID:-}" ] && kill -0 "${RERANK_SERVER_PID}" 2>/dev/null; then
        kill "${RERANK_SERVER_PID}" || true
        wait "${RERANK_SERVER_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

if check_port "${RERANK_API_PORT}"; then
    echo "Error: port ${RERANK_API_PORT} is already in use." >&2
    exit 1
fi

RERANK_API_BASE="http://127.0.0.1:${RERANK_API_PORT}"
RERANK_SERVER_LOG="${BASE_LOG_DIR}/pruner_server.log"

echo "Starting local pruner server on ${RERANK_API_BASE}"
CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES}" \
SWEPRUNER_MODEL_PATH="${PRUNER_MODEL_PATH}" \
SWEPRUNER_SMOKE_PORT="${RERANK_API_PORT}" \
"${PYTHON_BIN}" "${SERVER_SCRIPT}" >"${RERANK_SERVER_LOG}" 2>&1 &
RERANK_SERVER_PID=$!

wait_for_server "${RERANK_API_BASE}" "${RERANK_SERVER_PID}" "${RERANK_SERVER_LOG}"

IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"
for method in "${METHODS[@]}"; do
    RUN_DIR="${RESULT_DIR}/method_${method}"
    LOG_FILE="${BASE_LOG_DIR}/${method}.log"
    mkdir -p "${RUN_DIR}"

    echo "=========================================="
    echo "Running LCC method=${method}"
    echo "Model: ${MODEL_NAME}"
    echo "Pruner: ${PRUNER_MODEL_PATH}"
    echo "Result dir: ${RUN_DIR}"
    echo "=========================================="

    ARGS=(
        --model_name "${MODEL_NAME}"
        --method "${method}"
        --result_dir "${RUN_DIR}"
        --dataset_path "${DATASET_PATH}"
        --dataset_split "${DATASET_SPLIT}"
        --num_examples "${NUM_EXAMPLES}"
        --batch_size "${BATCH_SIZE}"
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
        --embed_model_name "${EMBED_MODEL_NAME}"
        --embed_model_type bert
        --syntax_check True
        --syntax_language "${RERANK_LANGUAGE}"
        --syntax_check_chunk True
    )

    if [[ "${method}" == "rag" ]]; then
        ARGS+=(
            --rag_window_size "${RAG_WINDOW_SIZE}"
            --rag_overlap "${RAG_OVERLAP}"
            --rag_top_k "${RAG_TOP_K}"
        )
    elif [[ "${method}" == "rag_with_pruner" ]]; then
        ARGS+=(
            --pruner_type online_rerank
            --rerank_api_base "${RERANK_API_BASE}"
            --rerank_threshold "${RERANK_THRESHOLD}"
            --rerank_always_keep_first_frags "${RERANK_ALWAYS_KEEP_FIRST_FRAGS}"
            --rerank_aggregate_method "${RERANK_AGGREGATE_METHOD}"
            --rerank_language "${RERANK_LANGUAGE}"
            --rag_window_size "${RAG_WITH_PRUNER_WINDOW_SIZE}"
            --rag_overlap "${RAG_WITH_PRUNER_OVERLAP}"
            --rag_top_k "${RAG_WITH_PRUNER_TOP_K}"
        )
    elif [[ "${method}" == "llmlingua2" ]]; then
        ARGS+=(
            --llmlingua2_model_name "${LLMLINGUA2_MODEL_NAME}"
            --llmlingua2_rate 0.33
        )
    elif [[ "${method}" == "selective_context" ]]; then
        ARGS+=(
            --selective_context_model_type "${SELECTIVE_CONTEXT_MODEL_TYPE}"
            --selective_context_lang en
            --selective_context_reduce_ratio 0.5
        )
    elif [[ "${method}" == "longcodezip" ]]; then
        ARGS+=(
            --longcodezip_model_name "${LONGCODEZIP_MODEL_NAME}"
            --longcodezip_rate 0.5
            --longcodezip_rank_only False
        )
    fi

    LCC_TOKENIZER_MODEL="${MODEL_NAME}" \
    CUDA_VISIBLE_DEVICES="${MAIN_CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" "${WORK_DIR}/main.py" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
done

echo "LCC runs finished. Results are in ${RESULT_DIR}"
