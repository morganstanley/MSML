# Single-Turn Evaluation: LCC and LongCodeQA

This directory contains the single-turn downstream evaluation harness for
LaMR/SWE-Pruner. It covers:

- **LCC**: code completion on the LCC Python benchmark.
- **LongCodeQA**: long-context code question answering.

The code is included here; large datasets, model weights, and result logs are
not. The scripts are written so a user can place assets under the repository
root or pass absolute paths through environment variables.

## Directory Layout

| Path | Purpose |
|---|---|
| `hard_code_pruner/downstream_task/LCC/` | LCC evaluator, metrics, and compression helpers. |
| `hard_code_pruner/downstream_task/longcodeqa/` | LongCodeQA evaluator, dataset loader, metrics, and compression helpers. |
| `hard_code_pruner/downstream_task/{embedder,reranker,model}.py` | Shared RAG, reranking, and online-pruner adapters. |
| `hard_code_pruner/utils/` | Shared chunking and code-splitting utilities. |
| `scripts/sweep_v2semdep_lcc_rankonly2.sh` | LCC reproduction sweep for the reported best cell. |
| `scripts/paper_recipe_longcodeqa_lczp_rate0125.sh` | LongCodeQA reproduction script for the reported best cell. |
| `pyproject.toml` | Dependency manifest for this evaluation harness. |

## Required Artifacts

The following files/directories must be available locally. They are ignored by
Git and should be provided through external storage, HuggingFace, or symlinks.

| Artifact | Default path from repo root | Override variable |
|---|---|---|
| SWE-Pruner v2 sem+dep checkpoint | `runtime_models/swe-pruner-py-v2-semdep-5ep-8192` | `SWEPRUNER_MODEL_PATH` or `V2_CKPT` |
| Qwen generator | `hf_models/Qwen2.5-Coder-7B-Instruct` | `QWEN_MODEL` |
| UnixCoder embedder | `hf_models/unixcoder-base` | `EMBED_MODEL` |
| Seed-Coder compressor | `hf_models/Seed-Coder-8B-Instruct` | `LONGCODEZIP_MODEL` |
| Qwen summary model | `hf_models/Qwen3-0.6B` | `SUMMARY_MODEL` |
| LCC dataset | `lcc` | `LCC_DATASET` |
| LongCodeQA 32K JSONL | `downstream_eval/single_turn/datasets/longcodeqa_32k.jsonl` | `LCQA_DATASET` |

Suggested local layout:

```text
hf_models/
  Qwen2.5-Coder-7B-Instruct/
  Qwen3-0.6B/
  Seed-Coder-8B-Instruct/
  unixcoder-base/
runtime_models/
  swe-pruner-py-v2-semdep-5ep-8192/
lcc/
downstream_eval/single_turn/datasets/
  longcodeqa_32k.jsonl
```

Dataset sources:

- LCC: HuggingFace dataset `microsoft/LCC_python`.
- LongCodeQA: LongCodeQA archive from the LCB dataset release, converted to
  the `longcodeqa_32k.jsonl` format used by the evaluator.

## Environment

Use Python 3.12 if possible. Install the runtime pruner package and the
single-turn evaluation dependencies:

```bash
cd /path/to/LaMR
python -m pip install -e swe-pruner
python -m pip install \
  accelerate datasets evaluate FlagEmbedding fire huggingface-hub \
  langchain langchain-core langchain-openai langgraph llmlingua loguru \
  modelscope mteb numpy openai pandas peft pydantic requests \
  sentence-transformers selective-context torch tqdm transformers \
  tree-sitter tree-sitter-python vllm editdistance astchunk tiktoken
```

On clusters, use your existing CUDA-compatible environment instead of installing
GPU packages from scratch.

## Smoke Test

Use `NUM_EXAMPLES=2` before running the full benchmark:

```bash
cd /path/to/LaMR
NUM_EXAMPLES=2 \
SWEPRUNER_MODEL_PATH=/path/to/swe-pruner-py-v2-semdep-5ep-8192 \
QWEN_MODEL=/path/to/Qwen2.5-Coder-7B-Instruct \
EMBED_MODEL=/path/to/unixcoder-base \
LONGCODEZIP_MODEL=/path/to/Seed-Coder-8B-Instruct \
LCQA_DATASET=/path/to/longcodeqa_32k.jsonl \
bash downstream_eval/single_turn/scripts/paper_recipe_longcodeqa_lczp_rate0125.sh
```

The script starts a local pruner service, waits for `/health`, runs the
evaluator, and writes logs/results under `downstream_eval/results/`.

## LongCodeQA Reproduction

Reported cell:

| Task | Method | Config | Accuracy | Correct / N | CR |
|---|---|---|---:|---:|---:|
| LongCodeQA | `longcodezip_with_pruner` | `rank_only=True`, `rate=0.125`, `threshold=0.55` | 62.04 | 67 / 108 | 0.0715 |

Command:

```bash
cd /path/to/LaMR
PYTHON_BIN=/path/to/python \
SWEPRUNER_MODEL_PATH=/path/to/swe-pruner-py-v2-semdep-5ep-8192 \
QWEN_MODEL=/path/to/Qwen2.5-Coder-7B-Instruct \
EMBED_MODEL=/path/to/unixcoder-base \
LONGCODEZIP_MODEL=/path/to/Seed-Coder-8B-Instruct \
SUMMARY_MODEL=/path/to/Qwen3-0.6B \
LCQA_DATASET=/path/to/longcodeqa_32k.jsonl \
bash downstream_eval/single_turn/scripts/paper_recipe_longcodeqa_lczp_rate0125.sh
```

Important defaults:

- `METHODS_CSV=longcodezip_with_pruner`
- `LONGCODEZIP_RATE=0.125`
- `LONGCODEZIP_RANK_ONLY=True`
- `RERANK_THRESHOLD=0.55`
- `NUM_EXAMPLES=113`
- `PORT=8037`

## LCC Reproduction

Reported best cell from the sweep:

| Task | Method | Config | EM | ES | CR |
|---|---|---|---:|---:|---:|
| LCC | `longcodezip_with_pruner` | `rank_only=True`, `rate=0.25`, `threshold=0.55` | 35.50 | 61.15 | 0.2285 |

Command:

```bash
cd /path/to/LaMR
PYTHON_BIN=/path/to/python \
V2_CKPT=/path/to/swe-pruner-py-v2-semdep-5ep-8192 \
QWEN_MODEL=/path/to/Qwen2.5-Coder-7B-Instruct \
EMBED_MODEL=/path/to/unixcoder-base \
LONGCODEZIP_MODEL=/path/to/Seed-Coder-8B-Instruct \
LCC_DATASET=/path/to/LCC_python \
bash downstream_eval/single_turn/scripts/sweep_v2semdep_lcc_rankonly2.sh
```

The script runs these cells:

| Rate | Threshold | Notes |
|---:|---:|---|
| 0.25 | 0.45 | Rank-only LCC sweep cell. |
| 0.25 | 0.55 | Reported best EM/ES cell. |
| 0.125 | 0.58 | Higher-compression sweep cell. |
| 0.125 | 0.60 | Higher-compression sweep cell. |
| 0.125 | 0.62 | Higher-compression sweep cell. |

Important defaults:

- `NUM_EXAMPLES=200`
- `BATCH_SIZE=16`
- `PORT=8062`
- `PRUNER_CUDA_VISIBLE_DEVICES=0`
- `EVAL_CUDA_VISIBLE_DEVICES=1,2`

## Outputs

Each script writes to `downstream_eval/results/` by default:

```text
downstream_eval/results/
  longcodeqa-<label>/
    logs/
    method_longcodezip_with_pruner/
      ...SCORES.json
  lcc-<label>/
    logs/
    lczp_rankonly_r0.25_th0.55/
      ...SCORES.json
```

If a run directory already contains a `*SCORES.json` file, the scripts skip
that cell. Delete or move the existing result directory to force a rerun.

## GPU and Port Controls

Both scripts start a local SWE-Pruner HTTP service with
`python -m swe_pruner.online_serving`.

Override these variables when needed:

| Variable | Default | Meaning |
|---|---|---|
| `PRUNER_CUDA_VISIBLE_DEVICES` | `0` | GPU used by the pruner service. |
| `EVAL_CUDA_VISIBLE_DEVICES` | `1,2` | GPUs used by the evaluator/vLLM. |
| `PORT` | `8037` or `8062` | Local HTTP port for the pruner service. |
| `RESULT_ROOT` | task-specific path under `downstream_eval/results/` | Output directory. |
| `NUM_EXAMPLES` | task-specific | Number of examples to evaluate. |

## Troubleshooting

- If the script cannot import `swe_pruner`, install the runtime package with
  `python -m pip install -e swe-pruner` from the repository root.
- If `/health` never becomes ready, inspect `pruner_server.log` under the run's
  `logs/` directory.
- If a CUDA out-of-memory error occurs, reduce `BATCH_SIZE` or change
  `EVAL_CUDA_VISIBLE_DEVICES`.
- If the evaluator immediately skips a cell, a previous `*SCORES.json` already
  exists in that result directory.
- If HuggingFace download is disabled, make sure all model paths point to local
  model directories.
