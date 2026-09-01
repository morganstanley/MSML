#!/usr/bin/env bash
set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "Error: this script must be run with bash" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORK_DIR="${REPO_ROOT}/downstream_eval/single_turn/hard_code_pruner/downstream_task/longcodeqa"
SERVER_SCRIPT="${REPO_ROOT}/tests/smoke_prune_http_server.py"
PRUNER_CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES:-1}"

MODEL_NAME="${1:-${REPO_ROOT}/hf_models/Qwen2.5-Coder-7B-Instruct}"
PRUNER_MODEL_PATH="${2:-${REPO_ROOT}/runtime_models/swe-pruner-py-v2-semdep-5ep-8192}"
RESULT_DIR="${3:-${REPO_ROOT}/downstream_eval/results/longcodeqa-v2-semdep-5ep-8192}"
DATASET_PATH="${4:-}"
EMBED_MODEL_NAME="${5:-${LONGCODEQA_EMBED_MODEL:-${REPO_ROOT}/hf_models/unixcoder-base}}"
NUM_EXAMPLES="${6:-200}"
BATCH_SIZE="${7:-16}"
TENSOR_PARALLEL_SIZE="${8:-1}"
TOP_K="${9:-4}"
RERANK_API_PORT="${10:-8000}"
MAX_MODEL_LEN="${11:-32768}"
METHODS_CSV="${METHODS_CSV:-full,no_context,rag,llmlingua2,longcodezip,selective_context,rag_with_pruner}"
LONGCODEZIP_MODEL_NAME="${LONGCODEZIP_MODEL_NAME:-${REPO_ROOT}/hf_models/Seed-Coder-8B-Instruct}"
LLMLINGUA2_MODEL_NAME="${LLMLINGUA2_MODEL_NAME:-${REPO_ROOT}/hf_models/llmlingua-2-xlm-roberta-large-meetingbank}"
SELECTIVE_CONTEXT_MODEL_TYPE="${SELECTIVE_CONTEXT_MODEL_TYPE:-gpt2}"
RAG_WINDOW_SIZE="${RAG_WINDOW_SIZE:-90}"
RAG_OVERLAP="${RAG_OVERLAP:-15}"
RAG_TOP_K="${RAG_TOP_K:-4}"
RAG_WITH_PRUNER_WINDOW_SIZE="${RAG_WITH_PRUNER_WINDOW_SIZE:-${RAG_WINDOW_SIZE}}"
RAG_WITH_PRUNER_OVERLAP="${RAG_WITH_PRUNER_OVERLAP:-${RAG_OVERLAP}}"
RAG_WITH_PRUNER_TOP_K="${RAG_WITH_PRUNER_TOP_K:-${RAG_TOP_K}}"
RERANK_THRESHOLD="${RERANK_THRESHOLD:-0.4}"
RERANK_AGGREGATE_METHOD="${RERANK_AGGREGATE_METHOD:-line}"
RERANK_LANGUAGE="${RERANK_LANGUAGE:-python}"

BASE_LOG_DIR="${RESULT_DIR}/logs"
mkdir -p "${BASE_LOG_DIR}" "${RESULT_DIR}"

if [ -z "${DATASET_PATH}" ]; then
    echo "Error: DATASET_PATH is required." >&2
    echo "Usage: bash ${BASH_SOURCE[0]} [model_name] [pruner_model_path] [result_dir] <dataset_path> [embed_model_name]" >&2
    exit 1
fi

if [ ! -f "${DATASET_PATH}" ]; then
    echo "Error: dataset file not found at ${DATASET_PATH}" >&2
    exit 1
fi

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

needs_embed=false
IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"
for method in "${METHODS[@]}"; do
    case "${method}" in
        rag|function_rag|rag_with_pruner|function_rag_with_pruner|rag_with_silver_label_pruner|function_rag_with_silver_label_pruner|rag_with_pruner_rerank)
            needs_embed=true
            ;;
    esac
done

if [ "${needs_embed}" = true ] && [ -z "${EMBED_MODEL_NAME}" ]; then
    echo "Error: EMBED_MODEL_NAME is required for methods: ${METHODS_CSV}" >&2
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

for method in "${METHODS[@]}"; do
    RUN_DIR="${RESULT_DIR}/method_${method}"
    LOG_FILE="${BASE_LOG_DIR}/${method}.log"
    mkdir -p "${RUN_DIR}"

    echo "=========================================="
    echo "Running LongCodeQA method=${method}"
    echo "Model: ${MODEL_NAME}"
    echo "Pruner: ${PRUNER_MODEL_PATH}"
    echo "Result dir: ${RUN_DIR}"
    echo "=========================================="

    ARGS=(
        --model_name "${MODEL_NAME}"
        --method "${method}"
        --result_dir "${RUN_DIR}"
        --dataset_path "${DATASET_PATH}"
        --num_examples "${NUM_EXAMPLES}"
        --batch_size "${BATCH_SIZE}"
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
        --max_model_len "${MAX_MODEL_LEN}"
        --embedder_type bert
    )

    if [ -n "${EMBED_MODEL_NAME}" ]; then
        ARGS+=(--embed_model_name "${EMBED_MODEL_NAME}")
    fi

    if [[ "${method}" == "rag" ]]; then
        ARGS+=(
            --rag_window_size "${RAG_WINDOW_SIZE}"
            --rag_overlap "${RAG_OVERLAP}"
            --rag_top_k "${RAG_TOP_K}"
        )
    elif [[ "${method}" == "longcodezip" ]]; then
        ARGS+=(
            --longcodezip_model_name "${LONGCODEZIP_MODEL_NAME}"
            --longcodezip_rate 0.2
            --longcodezip_rank_only False
        )
    elif [[ "${method}" == "rag_with_pruner" ]]; then
        ARGS+=(
            --pruner_type online_rerank
            --rerank_api_base "${RERANK_API_BASE}"
            --rerank_threshold "${RERANK_THRESHOLD}"
            --rerank_aggregate_method "${RERANK_AGGREGATE_METHOD}"
            --rerank_language "${RERANK_LANGUAGE}"
            --rag_window_size "${RAG_WITH_PRUNER_WINDOW_SIZE}"
            --rag_overlap "${RAG_WITH_PRUNER_OVERLAP}"
            --rag_top_k "${RAG_WITH_PRUNER_TOP_K}"
        )
    elif [[ "${method}" == "llmlingua2" ]]; then
        ARGS+=(
            --llmlingua2_model_name "${LLMLINGUA2_MODEL_NAME}"
            --llmlingua2_rate 0.15
        )
    elif [[ "${method}" == "selective_context" ]]; then
        ARGS+=(
            --selective_context_model_type "${SELECTIVE_CONTEXT_MODEL_TYPE}"
            --selective_context_lang "en"
            --selective_context_reduce_ratio 0.86
        )
    fi

    "${PYTHON_BIN}" "${WORK_DIR}/eval.py" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
done

echo "LongCodeQA runs finished. Results are in ${RESULT_DIR}"
