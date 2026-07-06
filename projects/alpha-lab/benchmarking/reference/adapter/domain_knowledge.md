# LLM Speedrun Domain Knowledge — PleIAs SYNTH (Parquet) Pretraining

## Objective

Train a **<100M trainable parameter** language model **from scratch** under a **hard 20-minute wall-clock training budget** to minimize **validation bits-per-byte (`val_bpb`)** on the **PleIAs SYNTH** corpus.

This is a *throughput-constrained pretraining* problem: the best configuration is usually the one that (1) is stable at high learning rates, (2) processes the most useful tokens per second, and (3) allocates parameters efficiently (width/depth/FFN) for fast early convergence.

---

## Primary Metric: `val_bpb` (Validation Bits-Per-Byte)

- **Definition**: `bpb = loss_nats * log2(e) / bytes_per_token`
- `bytes_per_token` must be computed on the **actual text** used for training/validation with the **exact tokenizer** (total UTF-8 bytes / total tokens).
- Lower is better.
- **Do not** compute BPB from perplexity; always convert from raw cross-entropy in *nats*.

Practical note: because `bytes_per_token` depends on the tokenizer and dataset mix (multilingual text changes UTF-8 byte patterns), **tokenizer choice and the exact sampled subset** can shift absolute BPB.

---

## Hard Constraints

### 1) <100M trainable parameters
- Enforce via: `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
- Tie input/output embeddings when possible to save parameters.
- For parameter budgeting, embeddings can dominate: `vocab_size * d_model`.

### 2) 20-minute wall-clock training budget
- Budget is **1200 seconds of training time**; compilation/startup is excluded.
- Final score is **best `val_bpb` observed at any evaluation checkpoint**.
- Evaluate frequently enough to get multiple checkpoints (aim 5–10+ evals).

---

## Dataset: PleIAs SYNTH (local parquet shards)

**Location**: `/path/to/pleias-synth`

**Format**: 500 parquet shards (`synth_001.parquet` … `synth_500.parquet`)
- Total size: **~220 GB**
- Total rows: **77,908,583** (~155,817 rows/shard; ~450.8 MB/shard)
- Each shard is a single row group (fast metadata access; full reads are still heavy).

### Schema (observed in shards)
| column | type | notes |
|---|---|---|
| `synth_id` | string | unique id |
| `language` | string | mostly `en`; small fraction is null/other |
| `exercise` | string | synthetic task type |
| `model` | string | generator model tag (e.g., `qwen-3-8b-memorization`) |
| `query` | string | user query / instruction |
| `query_seed_url` | string | often present; some null |
| `query_seed_text` | string | seed passage; can be long |
| `additional_seed_url` | string | ~9% blank |
| `seed_license` | string | typically CC license |
| `constraints` | string | **~91% blank** (in shard 001) |
| `script` | string | **~98% blank** (in shard 001) |
| `synthetic_reasoning` | string | intermediate reasoning trace (often markdown-like) |
| `synthetic_answer` | string | final answer text |
| `words` | int64 | word count of full sample (query+reasoning+answer) |

### High-level distributions (quick sample: 200k rows from shards 001–010)
- **Languages**: `en` **~80.7%**; `fr/de/es/it/pl` ~3.1–3.2% each; `nl/la` ~1.6% each; ~0.15% null; rare extra codes (`pt`, `uk`, `ru`, …).
- **Exercises**: `memorization` **~90.6%**; remaining mass across `mcq`, `math exercise`, `rag`, `constrained writing`, `creative writing`, `editing`, `cooking`, etc.
- **`words`**: mean ~515, median ~522, p90 ~699, p99 ~1118, max ~3971.
- Text length (chars, mean): `query` ~221; `synthetic_reasoning` ~2078; `synthetic_answer` ~1454. There are long outliers (10k–30k chars).

### What text to train on (important)
The task description specifies using **`synthetic_answer`** (and optionally **`query`**) as training text.

Recommended text construction options (make this configurable in the harness):
1. **Answer-only (fastest, simplest)**:
   - `text = synthetic_answer`
2. **Query + Answer (instruction-style)**:
   - `text = "<Q>" + query + "\n<A>" + synthetic_answer`
3. **Full trace (more tokens, may teach formatting patterns)**:
   - `text = "<Q>"+query+"\n<R>"+synthetic_reasoning+"\n<A>"+synthetic_answer`

Because the 20-minute budget is tight, **answer-only** or **query+answer** is usually best to maximize useful tokens/sec.

### Sampling is mandatory
Reading/tokenizing the full 220GB corpus is not feasible within 20 minutes.

Make sampling explicit and reproducible:
- Sample **N shards** (e.g., 2–20) and/or **cap rows per shard**.
- Consider filtering to `language == "en"` (dominant in the metric if the val split follows the corpus mix).
- Fix the **val split deterministically** (same shards/rows/tokens across runs), otherwise comparisons are noisy.

Implementation note: parquet scan + Python string handling can be a bottleneck; prefer:
- Arrow dataset scanning with column projection (`synthetic_answer`, optionally `query`, `language`).
- Streaming tokenization and caching to a `.bin` (mmap) once per sampled subset.

---

## Model & Training Recommendations for SYNTH speedruns

### Architecture (high-impact under 100M)
- **LLaMA-style block** (recommended default): **RMSNorm + RoPE + SwiGLU + pre-norm**.
- Use `torch.nn.functional.scaled_dot_product_attention` to get Flash/SDPA kernels.
- Typical sweet spots under 100M:
  - 6–12 layers, d_model 512–768, heads 8–12, RoPE.
  - Keep FFN ratio ~2.5–4.0× (SwiGLU usually uses `ffn_dim ≈ 8/3 * d_model`).
- Consider **GQA/MQA** if it improves throughput enough to buy more tokens in 20 minutes.

### Optimizers / schedules (fast early convergence matters)
- **AdamW** baseline: `betas=(0.9, 0.95)`, `eps=1e-8`, `weight_decay≈0.1`, try `lr 3e-4 → 1e-3`.
- **Fused AdamW** when available: `torch.optim.AdamW(..., fused=True)`.
- **Sophia / Muon**: can converge faster in some regimes; worth trying if already implemented and stable.
- Schedules:
  - **Cosine decay** is robust.
  - **WSD (warmup → stable → decay)** is simple and competitive for short runs.
- Warmup: keep short (hundreds to ~2000 steps) given the time cap.

### Efficiency knobs (often the difference-maker)
- **bf16 autocast** on GPU (no GradScaler needed).
- `torch.compile(mode="reduce-overhead")` can help, but only if compilation is excluded from the 20-min timer.
- Data loader: minimize Python overhead; use memmap’d token arrays; pinned memory; enough prefetch.

---

## What not to spend time on
- Heavy regularization (dropout): short-run pretraining rarely overfits.
- Data augmentation: irrelevant; the dataset is enormous; sampling/throughput dominates.
- Complex curriculum: too much engineering for a 20-minute budget.
