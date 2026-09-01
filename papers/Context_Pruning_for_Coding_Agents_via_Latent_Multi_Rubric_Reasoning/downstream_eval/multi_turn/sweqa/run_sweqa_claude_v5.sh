#!/usr/bin/env bash
set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
    echo "Error: this script must be run with bash" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SWEQA_ROOT="${SCRIPT_DIR}"

OPENHANDS_PYTHON="${OPENHANDS_PYTHON:-${SWEQA_ROOT}/.venv/bin/python}"
PRUNER_PYTHON="${PRUNER_PYTHON:-python}"
PRUNER_SERVER_SCRIPT="${PRUNER_SERVER_SCRIPT:-${REPO_ROOT}/tests/smoke_prune_http_server.py}"

REPOS="${1:-streamlink}"
EXPERIMENT_TYPE="${2:-pruner}"

MODEL_NAME="${OPENHANDS_MODEL_NAME:-anthropic/claude-sonnet-4-5-20250929}"
API_TYPE="${API_TYPE:-openai}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.anthropic.com}"
# This launcher targets Claude via Anthropic's OpenAI-compatible endpoint.
# Prefer ANTHROPIC_API_KEY over any unrelated OPENAI_API_KEY left in the shell.
OPENAI_API_KEY="${ANTHROPIC_API_KEY:-${OPENAI_API_KEY:-}}"

BASE_REPO_PATH="${BASE_REPO_PATH:-${SWEQA_ROOT}/swe-repos}"
QUESTIONS_PATH="${QUESTIONS_PATH:-${SWEQA_ROOT}/questions}"
ANSWER_OUTPUT_PATH="${ANSWER_OUTPUT_PATH:-${REPO_ROOT}/downstream_eval/results/sweqa-answer-claude45-v5-${EXPERIMENT_TYPE}}"
TRAJ_OUTPUT_PATH="${TRAJ_OUTPUT_PATH:-${REPO_ROOT}/downstream_eval/results/sweqa-traj-claude45-v5-${EXPERIMENT_TYPE}}"

MAX_ITERATION_PER_RUN="${MAX_ITERATION_PER_RUN:-50}"
MAX_TIME_PER_QUESTION="${MAX_TIME_PER_QUESTION:-1800}"

PRUNER_MODEL_PATH="${PRUNER_MODEL_PATH:-${REPO_ROOT}/runtime_models/swe-pruner-py-v2-semdep-5ep-8192}"
PRUNER_PORT="${PRUNER_PORT:-8000}"
PRUNER_CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES:-0}"
PRUNER_URL="${PRUNER_URL:-http://127.0.0.1:${PRUNER_PORT}/prune}"
PRUNE_THRESHOLD="${PRUNE_THRESHOLD:-0.4}"
PRUNER_MIN_CHARS="${PRUNER_MIN_CHARS:-1500}"
PRUNER_CHUNK_OVERLAP_TOKENS="${PRUNER_CHUNK_OVERLAP_TOKENS:-100}"
PRUNER_ALWAYS_KEEP_FIRST_FRAGS="${PRUNER_ALWAYS_KEEP_FIRST_FRAGS:-true}"
PRUNER_ALWAYS_KEEP_LAST_FRAGS="${PRUNER_ALWAYS_KEEP_LAST_FRAGS:-false}"
PRUNER_BYPASS_ERROR_OUTPUTS="${PRUNER_BYPASS_ERROR_OUTPUTS:-true}"

mkdir -p "${ANSWER_OUTPUT_PATH}" "${TRAJ_OUTPUT_PATH}"

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
    if [ -n "${PRUNER_SERVER_PID:-}" ] && kill -0 "${PRUNER_SERVER_PID}" 2>/dev/null; then
        kill "${PRUNER_SERVER_PID}" || true
        wait "${PRUNER_SERVER_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

if [ ! -x "${OPENHANDS_PYTHON}" ]; then
    echo "Error: OpenHands Python not found at ${OPENHANDS_PYTHON}" >&2
    exit 1
fi

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "Error: OPENAI_API_KEY or ANTHROPIC_API_KEY must be set" >&2
    exit 1
fi

if [ "${EXPERIMENT_TYPE}" = "pruner" ]; then
    if check_port "${PRUNER_PORT}"; then
        echo "Error: pruner port ${PRUNER_PORT} is already in use." >&2
        exit 1
    fi
    PRUNER_LOG="${TRAJ_OUTPUT_PATH}/pruner_server.log"
    echo "Starting pruner server on http://127.0.0.1:${PRUNER_PORT}"
    CUDA_VISIBLE_DEVICES="${PRUNER_CUDA_VISIBLE_DEVICES}" \
    SWEPRUNER_MODEL_PATH="${PRUNER_MODEL_PATH}" \
    SWEPRUNER_SMOKE_PORT="${PRUNER_PORT}" \
    "${PRUNER_PYTHON}" "${PRUNER_SERVER_SCRIPT}" >"${PRUNER_LOG}" 2>&1 &
    PRUNER_SERVER_PID=$!
    wait_for_http "http://127.0.0.1:${PRUNER_PORT}/health" "${PRUNER_SERVER_PID}" "${PRUNER_LOG}"
fi

echo "Running SWE-QA for repos: ${REPOS}"
echo "Experiment: ${EXPERIMENT_TYPE}"
echo "Model: ${MODEL_NAME}"
if [ "${EXPERIMENT_TYPE}" = "pruner" ]; then
    echo "Pruner model: ${PRUNER_MODEL_PATH}"
    echo "Pruner threshold: ${PRUNE_THRESHOLD}"
    echo "Pruner min_chars: ${PRUNER_MIN_CHARS}"
    echo "Pruner overlap: ${PRUNER_CHUNK_OVERLAP_TOKENS}"
    echo "Pruner keep first: ${PRUNER_ALWAYS_KEEP_FIRST_FRAGS}"
    echo "Pruner keep last: ${PRUNER_ALWAYS_KEEP_LAST_FRAGS}"
    echo "Pruner bypass error outputs: ${PRUNER_BYPASS_ERROR_OUTPUTS}"
fi

cd "${SWEQA_ROOT}"

API_TYPE="${API_TYPE}" \
OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
OPENAI_API_KEY="${OPENAI_API_KEY}" \
OPENHANDS_MODEL_NAME="${MODEL_NAME}" \
OPENHANDS_REPOS="${REPOS}" \
BASE_REPO_PATH="${BASE_REPO_PATH}" \
QUESTIONS_PATH="${QUESTIONS_PATH}" \
ANSWER_OUTPUT_PATH="${ANSWER_OUTPUT_PATH}" \
TRAJ_OUTPUT_PATH="${TRAJ_OUTPUT_PATH}" \
EXPERIMENT_TYPE="${EXPERIMENT_TYPE}" \
PRUNER_URL="${PRUNER_URL}" \
PRUNE_THRESHOLD="${PRUNE_THRESHOLD}" \
PRUNER_MIN_CHARS="${PRUNER_MIN_CHARS}" \
PRUNER_CHUNK_OVERLAP_TOKENS="${PRUNER_CHUNK_OVERLAP_TOKENS}" \
PRUNER_ALWAYS_KEEP_FIRST_FRAGS="${PRUNER_ALWAYS_KEEP_FIRST_FRAGS}" \
PRUNER_ALWAYS_KEEP_LAST_FRAGS="${PRUNER_ALWAYS_KEEP_LAST_FRAGS}" \
PRUNER_BYPASS_ERROR_OUTPUTS="${PRUNER_BYPASS_ERROR_OUTPUTS}" \
MAX_ITERATION_PER_RUN="${MAX_ITERATION_PER_RUN}" \
MAX_TIME_PER_QUESTION="${MAX_TIME_PER_QUESTION}" \
LITELLM_DROP_PARAMS=true \
"${OPENHANDS_PYTHON}" openhands-qa/main.py

echo "SWE-QA run finished."
echo "Answers: ${ANSWER_OUTPUT_PATH}/openhands"
echo "Trajectories: ${TRAJ_OUTPUT_PATH}"
