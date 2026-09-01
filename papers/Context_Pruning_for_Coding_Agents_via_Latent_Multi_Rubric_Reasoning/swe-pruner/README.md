# SWE-Pruner Runtime

This directory contains the runtime package used by LaMR for code-context
pruning. It loads an exported SWE-Pruner checkpoint, exposes a Python API, and
serves the same functionality through a small FastAPI HTTP service.

The repository does not track model weights. You must provide an exported
serving bundle such as `runtime_models/swe-pruner-py-v2-semdep-5ep-8192/`.

## Install

From the repository root:

```bash
python -m pip install -e swe-pruner
```

Install the GPU stack that matches your machine separately. At minimum, runtime
serving needs PyTorch, Transformers, FastAPI, Uvicorn, Typer, and Pydantic:

```bash
python -m pip install torch transformers fastapi uvicorn typer pydantic
```

If your exported model/backbone requires FlashAttention, install a wheel that
matches your CUDA and PyTorch versions:

```bash
python -m pip install flash-attn --no-build-isolation
```

## Model Layout

`SwePrunerForCodePruning.from_pretrained()` expects a HuggingFace-style exported
directory:

```text
runtime_models/swe-pruner-py-v2-semdep-5ep-8192/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── ...
```

If you trained from `train/`, export the checkpoint before serving:

```bash
python swe-pruner/export_serving_model.py \
  --checkpoint-dir llm_experiments/swe-pruner-py-v2-crf-semdep-8192 \
  --backbone-model /path/to/Qwen3-Reranker-0.6B \
  --output-dir runtime_models/swe-pruner-py-v2-semdep-5ep-8192
```

The exported directory can then be passed with `--model-path` or
`SWEPRUNER_MODEL_PATH`.

## HTTP Server

Start the server from the repository root:

```bash
PYTHONPATH=swe-pruner/src \
python -m swe_pruner.online_serving \
  --model-path runtime_models/swe-pruner-py-v2-semdep-5ep-8192 \
  --host 127.0.0.1 \
  --port 8000
```

Equivalent console-script form after `pip install -e swe-pruner`:

```bash
SWEPRUNER_MODEL_PATH=runtime_models/swe-pruner-py-v2-semdep-5ep-8192 \
swe-pruner --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prune request:

```bash
curl -X POST http://127.0.0.1:8000/prune \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Where is authentication handled?",
    "code": "def login(user, password):\n    return verify(user, password)\n\ndef logout(session):\n    session.close()\n",
    "threshold": 0.5,
    "chunk_overlap_tokens": 50,
    "always_keep_first_frags": false,
    "always_keep_last_frags": false
  }'
```

## Python API

```python
from swe_pruner.prune_wrapper import PruneRequest, SwePrunerForCodePruning

model = SwePrunerForCodePruning.from_pretrained(
    "runtime_models/swe-pruner-py-v2-semdep-5ep-8192"
)

request = PruneRequest(
    query="Where is authentication handled?",
    code="""\
def login(user, password):
    return verify(user, password)

def logout(session):
    session.close()
""",
    threshold=0.5,
    chunk_overlap_tokens=50,
    always_keep_first_frags=False,
    always_keep_last_frags=False,
)

response = model.prune(request)
print(response.pruned_code)
print(response.kept_frags)
print(response.origin_token_cnt, response.left_token_cnt)
```

## Request Fields

| Field | Default | Meaning |
|---|---:|---|
| `query` | required | Natural-language information need. |
| `code` | required | Source code context to prune. |
| `threshold` | `0.5` | Line keep threshold after token-to-line aggregation. |
| `chunk_overlap_tokens` | `50` | Token overlap when long code must be chunked. |
| `always_keep_first_frags` | `false` | Pin the first line before AST repair. |
| `always_keep_last_frags` | `false` | Pin the last line before AST repair. |

The runtime computes token-level keep decisions, aggregates them to lines,
thresholds the line scores, and applies Python AST-aware repair to keep
structural support lines such as scope headers, imports, and paired control
flow. For CRF checkpoints, runtime decoding uses the CRF/Viterbi path exposed by
the model.

## Response Fields

| Field | Meaning |
|---|---|
| `score` | Document-level relevance score. |
| `pruned_code` | Code after pruning, with filtered spans replaced by comments or `pass` placeholders when needed. |
| `token_scores` | List of `[token, score]` pairs used for line aggregation. |
| `kept_frags` | 1-based line numbers retained after AST repair. |
| `origin_token_cnt` | Estimated token count before pruning. |
| `left_token_cnt` | Estimated token count after pruning. |
| `model_input_token_cnt` | Estimated model input length for the request. |
| `error_msg` | Non-null only when pruning is bypassed because the request cannot be processed safely. |

## Notes

- This runtime is for Python code paths used in the LaMR experiments.
- Large checkpoints should stay outside Git and be referenced by path.
- The default runtime path uses thresholding plus AST-aware structural repair.
- Experimental Lagrangian decoding files from internal swe-pruner3 runs are not
  required for the reported LCC/LongCodeQA reproduction scripts in this repo.
