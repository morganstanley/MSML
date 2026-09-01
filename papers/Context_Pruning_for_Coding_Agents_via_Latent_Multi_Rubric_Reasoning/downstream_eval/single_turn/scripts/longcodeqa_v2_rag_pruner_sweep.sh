#!/usr/bin/env bash
set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "Error: this script must be run with bash" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNNER="${SCRIPT_DIR}/longcodeqa_v2_local.sh"

MODEL_NAME="${1:-${REPO_ROOT}/hf_models/Qwen2.5-Coder-7B-Instruct}"
PRUNER_MODEL_PATH="${2:-${REPO_ROOT}/runtime_models/swe-pruner-py-v2-semdep-5ep-8192}"
RESULT_BASE_DIR="${3:-${REPO_ROOT}/downstream_eval/results/longcodeqa-ragpruner-sweep}"
DATASET_PATH="${4:-}"
EMBED_MODEL_NAME="${5:-${REPO_ROOT}/hf_models/unixcoder-base}"
THRESHOLDS_CSV="${6:-0.34,0.36,0.37,0.38,0.39,0.41,0.42}"
TOPK_CSV="${7:-3,4,5}"
AGGREGATE_METHODS_CSV="${8:-line}"
NUM_EXAMPLES="${9:-200}"
BATCH_SIZE="${10:-16}"
TENSOR_PARALLEL_SIZE="${11:-2}"
PRUNER_CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES:-3}"
RERANK_API_PORT_BASE="${RERANK_API_PORT_BASE:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

if [ -z "${DATASET_PATH}" ]; then
    echo "Error: DATASET_PATH is required." >&2
    echo "Usage: bash ${BASH_SOURCE[0]} [model] [pruner] [result_base_dir] <dataset_path> [embed_model] [thresholds_csv] [topk_csv] [aggregate_methods_csv]" >&2
    exit 1
fi

if [ ! -f "${RUNNER}" ]; then
    echo "Error: runner script not found at ${RUNNER}" >&2
    exit 1
fi

mkdir -p "${RESULT_BASE_DIR}"

IFS=',' read -r -a THRESHOLDS <<< "${THRESHOLDS_CSV}"
IFS=',' read -r -a TOPKS <<< "${TOPK_CSV}"
IFS=',' read -r -a AGG_METHODS <<< "${AGGREGATE_METHODS_CSV}"

port_offset=0

for aggregate_method in "${AGG_METHODS[@]}"; do
    for top_k in "${TOPKS[@]}"; do
        for threshold in "${THRESHOLDS[@]}"; do
            run_name="agg_${aggregate_method}_topk${top_k}_th${threshold}"
            result_dir="${RESULT_BASE_DIR}/${run_name}"
            port=$((RERANK_API_PORT_BASE + port_offset))
            port_offset=$((port_offset + 1))

            echo "===== START ${run_name} ====="

            METHODS_CSV="rag_with_pruner" \
            PRUNER_CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES}" \
            RAG_WINDOW_SIZE=90 \
            RAG_OVERLAP=15 \
            RAG_TOP_K="${top_k}" \
            RAG_WITH_PRUNER_WINDOW_SIZE=90 \
            RAG_WITH_PRUNER_OVERLAP=15 \
            RAG_WITH_PRUNER_TOP_K="${top_k}" \
            RERANK_THRESHOLD="${threshold}" \
            RERANK_AGGREGATE_METHOD="${aggregate_method}" \
            bash "${RUNNER}" \
                "${MODEL_NAME}" \
                "${PRUNER_MODEL_PATH}" \
                "${result_dir}" \
                "${DATASET_PATH}" \
                "${EMBED_MODEL_NAME}" \
                "${NUM_EXAMPLES}" \
                "${BATCH_SIZE}" \
                "${TENSOR_PARALLEL_SIZE}" \
                "${top_k}" \
                "${port}" \
                "${MAX_MODEL_LEN}"

            echo "===== END ${run_name} ====="
        done
    done
done
