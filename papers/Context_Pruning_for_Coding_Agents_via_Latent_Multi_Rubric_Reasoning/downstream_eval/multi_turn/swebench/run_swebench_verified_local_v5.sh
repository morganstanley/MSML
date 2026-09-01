#!/usr/bin/env bash
set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "Error: this script must be run with bash" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MINI_ROOT="${SCRIPT_DIR}/mini-swe-agent--with-pruning"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRUNER_SERVER_SCRIPT="${REPO_ROOT}/tests/smoke_prune_http_server.py"
VLLM_SERVER_SCRIPT="${SCRIPT_DIR}/run_local_vllm_server.py"
MODEL_PATH="${1:-${REPO_ROOT}/hf_models/Qwen2.5-Coder-7B-Instruct}"
PRUNER_MODEL_PATH="${2:-${REPO_ROOT}/runtime_models/swe-pruner-py-v2-semdep-5ep-8192}"
OUTPUT_DIR="${3:-${REPO_ROOT}/downstream_eval/results/swebench-v5-semdep-smoke}"
INSTANCE_FILTER="${4:-^(django__django-11099)$}"
WORKERS="${5:-1}"
PRUNER_PORT="${PRUNER_PORT:-8000}"
VLLM_PORT="${VLLM_PORT:-8010}"
PRUNER_CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES:-0}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-1}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${REPO_ROOT}/cache/swebench_hf}"
APPTAINER_CACHE_DIR="${APPTAINER_CACHE_DIR:-${REPO_ROOT}/cache/apptainer}"
APPTAINER_TMP_DIR="${APPTAINER_TMP_DIR:-${TMPDIR:-${REPO_ROOT}/cache/apptainer-tmp}}"
MSWEA_GLOBAL_CONFIG_DIR="${MSWEA_GLOBAL_CONFIG_DIR:-${REPO_ROOT}/downstream_eval/multi_turn/swebench/.mswea_local}"
CONFIG_PATH="${CONFIG_PATH:-${MINI_ROOT}/templates/pruner_local_v5.yaml}"

mkdir -p "${OUTPUT_DIR}" "${HF_CACHE_DIR}" "${MSWEA_GLOBAL_CONFIG_DIR}" "${APPTAINER_CACHE_DIR}" "${APPTAINER_TMP_DIR}"

check_port() {
    local port="$1"
    if lsof -Pi :"${port}" -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

wait_for_http() {
    local url="$1"
    local pid="$2"
    local log_file="$3"
    local attempt=0

    while true; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "Server exited unexpectedly: ${url}" >&2
            tail -n 200 "${log_file}" >&2 || true
            return 1
        fi

        if curl -s -f "${url}" >/dev/null 2>&1; then
            return 0
        fi

        attempt=$((attempt + 1))
        if [ $((attempt % 15)) -eq 0 ]; then
            echo "Waiting for ${url} ..."
        fi
        sleep 2
    done
}

cleanup() {
    if [ -n "${VLLM_SERVER_PID:-}" ] && kill -0 "${VLLM_SERVER_PID}" 2>/dev/null; then
        kill "${VLLM_SERVER_PID}" || true
        wait "${VLLM_SERVER_PID}" 2>/dev/null || true
    fi
    if [ -n "${PRUNER_SERVER_PID:-}" ] && kill -0 "${PRUNER_SERVER_PID}" 2>/dev/null; then
        kill "${PRUNER_SERVER_PID}" || true
        wait "${PRUNER_SERVER_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

if check_port "${PRUNER_PORT}"; then
    echo "Error: pruner port ${PRUNER_PORT} is already in use." >&2
    exit 1
fi

if check_port "${VLLM_PORT}"; then
    echo "Error: vLLM port ${VLLM_PORT} is already in use." >&2
    exit 1
fi

PRUNER_LOG="${OUTPUT_DIR}/pruner_server.log"
VLLM_LOG="${OUTPUT_DIR}/vllm_server.log"

echo "Starting v5 pruner server on http://127.0.0.1:${PRUNER_PORT}"
CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES}" \
SWEPRUNER_MODEL_PATH="${PRUNER_MODEL_PATH}" \
SWEPRUNER_SMOKE_PORT="${PRUNER_PORT}" \
"${PYTHON_BIN}" "${PRUNER_SERVER_SCRIPT}" >"${PRUNER_LOG}" 2>&1 &
PRUNER_SERVER_PID=$!
wait_for_http "http://127.0.0.1:${PRUNER_PORT}/health" "${PRUNER_SERVER_PID}" "${PRUNER_LOG}"

echo "Starting local vLLM server on http://127.0.0.1:${VLLM_PORT}/v1"
CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES}" \
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
"${PYTHON_BIN}" "${VLLM_SERVER_SCRIPT}" \
    --model "${MODEL_PATH}" \
    --served-model-name Qwen2.5-Coder-7B-Instruct \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --trust-remote-code >"${VLLM_LOG}" 2>&1 &
VLLM_SERVER_PID=$!
wait_for_http "http://127.0.0.1:${VLLM_PORT}/v1/models" "${VLLM_SERVER_PID}" "${VLLM_LOG}"

echo "Running SWE-bench Verified smoke on ${INSTANCE_FILTER}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHE_DIR}"
export APPTAINER_TMPDIR="${APPTAINER_TMP_DIR}"
export TMPDIR="${APPTAINER_TMP_DIR}"
MSWEA_GLOBAL_CONFIG_DIR="${MSWEA_GLOBAL_CONFIG_DIR}" \
HF_HOME="${HF_CACHE_DIR}" \
HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets" \
TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers" \
HF_HUB_CACHE="${HF_CACHE_DIR}/hub" \
MSWEA_SILENT_STARTUP=1 \
"${PYTHON_BIN}" -m minisweagent.run.extra.swebench \
    --subset verified \
    --split test \
    --filter "${INSTANCE_FILTER}" \
    --output "${OUTPUT_DIR}" \
    --workers "${WORKERS}" \
    --config "${CONFIG_PATH}" \
    --environment-class singularity \
    --pruner-url "http://127.0.0.1:${PRUNER_PORT}/prune"

echo "SWE-bench smoke run finished. Output in ${OUTPUT_DIR}"
