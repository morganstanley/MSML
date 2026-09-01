# Training SWE-Pruner

Run all commands from the **repository root** so that the `train` package is importable.

---

## Quick start: train with the released dataset

If you just want to reproduce or finetune the swe-pruner, download the pre-built dataset and start training directly.

### 1. Install dependencies

```bash
pip install torch transformers flash-attn torchmetrics typer rich pydantic tqdm
```

### 2. Download data and model

| Resource | Link |
|----------|------|
| Training data (61k, Python) | [Google Drive](https://drive.google.com/file/d/18g_kWeyvd8EICEDZcKylEEf8mnOFhwdi) — `swe-pruner-training-dataset-py.jsonl` |
| Base model | [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) |

### 3. Launch training

```bash
bash train/train_llm.sh 8 /path/to/swe-pruner-training-dataset-py.jsonl \
  --model-name /path/to/Qwen3-Reranker-0.6B \
  --epochs 3 \
  --lr 1e-4 \
  --log-dir llm_experiments/swe-pruner-py \
  --num-finetune-layers 2 \
  --num-fusion-layers 1 \
  --batch-size 16 \
  --compression-head-type crf \
  --compression-loss-type focal \
  --dropout 0.4 \
  --auto-focal-alpha \
  --use-multi-layer-fusion \
  --use-sample-level-aggregation
```

Replace `8` with your GPU count. With 8 × A100 80GB, each epoch takes ~75 min (~10 s/step, 431 steps/epoch); full run finishes in ~4 hours.

### Main LaMR sem+dep 8192 recipe

The main downstream checkpoint uses the semantic+dependency CRF recipe with
8192-token inputs. This is different from
`train/train_llm_v2_crf_semantic_dependency.sh`, which is the older 4096-token
ablation wrapper.

Use:

```bash
MODEL_NAME=/path/to/Qwen3-Reranker-0.6B \
bash train/train_llm_v2_crf_semdep_8192.sh 8 /path/to/swe-pruner-training-dataset-py-v2-rubric.jsonl
```

The reference training dataset from the original workspace is
`swe-pruner-training-dataset-py-v2-rubric.jsonl` (39,649 JSONL rows locally).
It has rubric columns for `semantic,syntax,dependency,context`; this recipe
selects only `semantic,dependency` with `--objective-names`.

Recipe settings:

| Setting | Value |
|---|---|
| Objectives | `semantic,dependency` |
| Max length | `8192` |
| Epochs | `5` |
| Objective weights | `1.0,0.8` |
| `lambda_score` | `0.05` |
| `lambda_rubric` | `0.3` |
| Head | CRF with MoE softmax gating and final CRF |

### Expected results

The verified reference artifact in the original training workspace is:

```text
llm_experiments/swe-pruner-py-v2-crf-semdep-rubric03-5ep-8192
```

Its TensorBoard event file logs five validation epochs:

| Epoch | Loss | C_Loss | S_Loss | Accuracy | Precision | Recall | F1 |
|-------|------|--------|--------|----------|-----------|--------|----|
| 1 | 0.4518 | 0.4754 | 0.0005 | 0.7724 | 0.7614 | 0.6997 | 0.7293 |
| 2 | 0.3687 | 0.3879 | 0.0005 | 0.8130 | 0.7931 | 0.7755 | 0.7842 |
| 3 | 0.3442 | 0.3620 | 0.0005 | 0.8253 | 0.8154 | 0.7772 | 0.7958 |
| 4 | 0.3391 | 0.3566 | 0.0005 | 0.8296 | 0.8226 | 0.7792 | 0.8003 |
| 5 | 0.3333 | 0.3505 | 0.0005 | 0.8307 | 0.8133 | 0.7966 | **0.8048** |

### Monitoring

```bash
tensorboard --logdir llm_experiments/swe-pruner-py
```

### Output layout

All outputs land under `--log-dir`:

```
llm_experiments/swe-pruner-py/
├── best_model.pt                # Best checkpoint (by val F1)
├── model_config.json            # Full config snapshot for reproducibility
├── eval_with_token_scores.jsonl # Val set with per-token scores
└── events.out.tfevents.*        # TensorBoard logs
```

### Export for serving

`best_model.pt` and `model_config.json` are training artifacts. They are enough
to resume or evaluate training, but they are not yet the Hugging Face-style
model directory expected by `swe_pruner.prune_wrapper` or the HTTP server.

After training, export the checkpoint into a serving bundle:

```bash
python swe-pruner/export_serving_model.py \
  --checkpoint-dir llm_experiments/swe-pruner-py-v2-crf-semdep-8192 \
  --backbone-model /path/to/Qwen3-Reranker-0.6B \
  --output-dir runtime_models/swe-pruner-py-v2-semdep-5ep-8192
```

Use the `--checkpoint-dir` you trained with. For the verified reference artifact,
that path is `llm_experiments/swe-pruner-py-v2-crf-semdep-rubric03-5ep-8192`.

That produces a serving-ready directory like:

```text
runtime_models/swe-pruner-py-v2-semdep-5ep-8192/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

Then you can serve it with:

```bash
PYTHONPATH=swe-pruner/src \
SWEPRUNER_MODEL_PATH=runtime_models/swe-pruner-py-v2-semdep-5ep-8192 \
python -m swe_pruner.online_serving --port 8000
```

---

## Build your own training data

The full pipeline: **pull code → dedup → query gen → score → label → train**.

### JSONL format at each stage

Each line is one JSON object. Fields accumulate as the pipeline progresses.

| Stage | Fields | Note |
|-------|--------|------|
| **1. Pull** | `code`, `repo` | `repo`: `repo_id/file_path`. Long files chunked by `--max-lines`/`--min-lines`. |
| **2. Dedup** | same | Rows whose `code` appears in the eval set are removed. |
| **3. Query gen** | + `query` | One generated query per code snippet. |
| **4. Score** | + `score` | `score` ∈ [0,1]: query–code relevance from reranker. |
| **5. Label** | + `kept_frags` | 1-based line indices to keep (line-level pruning label). |
| **6. Train** | must have: `query`, `code`, `kept_frags`, `score` | Extra fields ignored. |

Example labeled line:
```json
{"query": "Where is auth configured?", "code": "def foo():\n  x = 1\n  return x", "score": 0.92, "kept_frags": [1, 3]}
```

### Step-by-step commands

**1. Pull GitHub code (ModelScope)**
```bash
python -m train.scripts.gh_code_dataset --output-prefix ghcode --want-rows 200000 --lang python
```
We used the first 200k samples. Scaling to 2M did not improve results much — better labeling models or a larger base model (e.g. Qwen3-Reranker-8B) may help more.

**2. Dedup against eval set**
```bash
python -m train.scripts.dedup --final-dataset final_dataset.jsonl --eval-dataset eval_ds.jsonl --output final_dedup.jsonl
```

**3. Generate queries**
```bash
python -m train.inference.qgen -i data.jsonl -o generated_queries.jsonl --model <vLLM_MODEL_PATH>
```

**4. Score (query, code) pairs**
```bash
python -m train.inference.score -i generated_queries.jsonl -o scored.jsonl --model <RERANKER_MODEL_PATH>
```

**5. Line-level labeling**
```bash
python -m train.inference.build_label \
  --input-file scored.jsonl \
  --output-jsonl labeled.jsonl \
  --model-name <vLLM_MODEL_PATH> \
  --tensor-parallel-size 8
```

**Optional: AST-aware v2 repair + judge filtering**

The safer v2 path keeps the original `kept_frags` as the semantic ground truth,
uses AST analysis to repair structurally incomplete masks, and then filters the
repaired examples with a judge.

Install static-analysis dependencies:

```bash
pip install tree-sitter tree-sitter-python
```

Build the repaired dataset:

```bash
python -m train.inference.build_repair_dataset \
  -i swe-pruner-training-dataset-py.jsonl \
  -o swe-pruner-training-dataset-py-v2.jsonl \
  --require-tree-sitter
```

This preserves the original fields and adds:

| Field | Meaning |
|-------|---------|
| `original_kept_frags` | Original semantic mask from the v1 dataset. |
| `repaired_kept_frags` | AST-repaired mask after adding structural closure. |
| `final_kept_frags` | Final accepted mask used for training. |
| `repair_actions` | Why each extra line was added. |
| `ast_metadata` | Parse status, scopes, controls, dependencies, symbols. |
| `judge_evaluation` | Heuristic or LLM-judge quality decision. |
| `accepted` | Whether the repaired sample passed filtering. |

For trainer compatibility, accepted rows also set `kept_frags = final_kept_frags`.

Quick smoke test:

```bash
python -m train.inference.build_repair_dataset \
  -i swe-pruner-training-dataset-py.jsonl \
  -o /tmp/repaired-smoke.jsonl \
  --max-items 10 \
  --judge-mode heuristic \
  --no-include-ast-metadata
```

If you want to run a real LLM judge instead of the heuristic smoke-test filter:

```bash
python -m train.inference.build_repair_dataset \
  -i swe-pruner-training-dataset-py.jsonl \
  -o swe-pruner-training-dataset-py-v2.jsonl \
  --judge-mode llm \
  --model-name <vLLM_JUDGE_MODEL_PATH> \
  --require-tree-sitter
```

The earlier `build_rubric_label` script remains in the repo as an experiment,
but it is not the recommended v2 training-data path.

If you want to enrich the accepted v2 dataset with rubric vectors for auxiliary
experiments, run it after `build_repair_dataset`:

```bash
python -m train.inference.build_rubric_label \
  -i swe-pruner-training-dataset-py-v2.jsonl \
  -o swe-pruner-training-dataset-py-v2-rubric.jsonl \
  --accepted-only \
  --semantic-label-source auto \
  --require-tree-sitter
```

That script now treats the repaired v2 dataset as its parent dataset and, by
default, prefers `final_kept_frags` over older mask fields when seeding the
semantic dimension.

**6. Train** — see [Quick start](#3-launch-training) above.

---

## Parameter reference

### Data and I/O
- **`-i` / `--input-file`** – Input JSONL with `query`, `code`, `kept_frags`, `score`.
- **`--log-dir`** – Output directory for checkpoints, config, TensorBoard (default: `llm_experiments/swe-pruner`).
- **`--train-split`** – Train/val split ratio (default: 0.9).

### Model
- **`--model-name`** – Base model name or path. **Required.**
- **`--num-finetune-layers`** – Top transformer layers to unfreeze; 0 = freeze all (default: 0).
- **`--instruction`** – System instruction for the query–document task.

### Compression head
- **`--compression-head-type`** – `ffn` | `simple` | `residual` | `crf` (default: `ffn`).
- **`--hidden-size`** – Bottleneck dimension (default: 256).
- **`--dropout`** – Dropout rate (default: 0.1).
- **`--num-fusion-layers`** – Self-attention fusion layers (default: 1).
- **`--num-heads`** – Attention heads per fusion layer (default: 8).

### Loss
- **`--lambda-score`** – Score loss weight; compression weight = 1 − lambda (default: 0.05).
- **`--compression-loss-type`** – `bce` | `focal` (default: `focal`).
- **`--focal-alpha`** – Focal loss alpha (default: 0.25). Use **`--auto-focal-alpha`** to compute from data.
- **`--focal-gamma`** – Focal loss gamma (default: 2.0).
- **`--use-sample-level-aggregation`** – Per-sample loss averaging before batch mean (default: true).

### Optimization
- **`--lr`** – Learning rate (default: 1e-4).
- **`--weight-decay`** – AdamW weight decay (default: 0.01).
- **`--warmup-ratio`** – Linear warmup fraction (default: 0.1).
- **`--batch-size`** – Batch size per GPU (default: 4).
- **`--epochs`** – Training epochs (default: 2).

### Multi-layer fusion
- **`--use-multi-layer-fusion`** – Concatenate early/middle/final hidden states.
- **`--early-layer-ratio`**, **`--middle-layer-ratio`** – Layer index ratios (defaults: 0.25, 0.5).

### Eval-only mode
- **`--eval-only`** – Skip training, only evaluate (requires `--eval-dataset` and `--model-paths`).
- **`--eval-dataset`** – JSONL for evaluation.
- **`--model-paths`** – Checkpoint path(s) to evaluate or compare.

---

## Shell scripts

- **train_llm.sh** – `./train/train_llm.sh <NUM_GPUS> <INPUT_JSONL> [--model-name MODEL] ...`
- **qgen.sh** – `./train/qgen.sh <DATASET_NAME> <RESULT_DIR> [--model MODEL]`
- **label.sh** – `./train/label.sh <DATASET_NAME> <RESULT_DIR> [--model-name MODEL] ...`

All run from repo root and forward extra arguments to the underlying Python module.
