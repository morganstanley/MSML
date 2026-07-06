# Harness & Backtest Correctness Review (Alpha Lab Critic)

## Scope reviewed

- `harness/`: `runner.py`, `baseline_train.py`, `metrics.py`, `data_prep.py`, `config.py`, `plot_curves.py`
- `backtest/`: wrapper modules (`runner.py`, `baseline_train.py`, `metrics.py`, `data_prep.py`, `config.py`, `__init__.py`)

## Smoke test executed

```bash
python -m harness.runner harness/baseline_train.py \
  --out_dir harness/results/review_smoke_backtest \
  --time_limit_seconds 10 --param_cap 100000000 -- \
  --run_name review_smoke_backtest --train_shard_end 1 --max_rows_per_shard 500 --no_compile
```

Observed:

- Runner enforced timeout and returned `status="timeout"`.
- `METRIC {...}` records were emitted.
- `best_val_bpb` was tracked and reported.

---

## Critical checklist items

### 1) `val_bpb` correctness — **OK**

**Required**: `bpb = loss_nats * log2(e) / bytes_per_token`, and `bytes_per_token` must be computed from the actual UTF-8 bytes and actual token count of the evaluated text.

- Conversion constant + formula are correct:
  - `harness/metrics.py:13–18` (`LOG2E = log2(e)`, `compute_bpb(loss_nats, bytes_per_token)`)
- `bytes_per_token` is computed from the actual sampled text:
  - `harness/data_prep.py:173–185` collects per-row token arrays and per-row `len(text.encode('utf-8'))`
  - `harness/data_prep.py:221–223` computes:
    - `bytes_per_token = total_bytes / total_tokens`
    - `bytes_per_token_val = val_bytes / val_tokens`
- Validation uses the **validation-split** denominator:
  - `harness/baseline_train.py:284–288` uses `compute_bpb(loss_nats, bytes_per_token_val)`
  - `harness/baseline_train.py:368–370` loads `bytes_per_token_val` from `meta`

### 2) Time enforcement (1200s budget) — **OK, with minor overrun risk**

- Runner starts the budget at `TRAINING_START` (so compile/data prep are excluded):
  - `harness/runner.py:144–151`
- Runner enforces timeout by SIGINT at the boundary and escalates quickly:
  - `harness/runner.py:179–199`
- Training code also self-terminates on the same budget (internal safety stop):
  - `harness/baseline_train.py:469–540`
- CUDA timing correctness inside the training process:
  - `torch.cuda.synchronize()` is called before `time.perf_counter()` for budget checks and throughput measurements
    - e.g. `harness/baseline_train.py:533–536`, `615–620`

**Minor issue**: the runner only polls every `0.25s` (`harness/runner.py:203`), and signal handling can add a small delay. In the smoke test the training process reported `wall_clock_seconds ≈ time_limit + 0.02s`. If you need *extremely strict* cutoffs, reduce the polling sleep and/or make the training loop check budget more frequently.

### 3) Compilation exclusion — **OK**

- `torch.compile(...)` happens before `TRAINING_START`:
  - `harness/baseline_train.py:423–426`
- Warm-up steps are excluded from timer and do **not** update weights (no `opt.step()`):
  - `harness/baseline_train.py:427–446`

### 4) Parameter counting + strict <100M cap — **OK**

- Count uses exactly: `sum(p.numel() for p in model.parameters() if p.requires_grad)`:
  - `harness/metrics.py:21–32`
- Cap is enforced **before** training starts and uses `>=` (strictly `< cap`):
  - `harness/runner.py:94–103`

### 5) Data isolation (no train/val overlap) — **OK for row-level leakage; note on fairness**

- Split is done on **row boundaries** to avoid splitting a single row across train/val:
  - `harness/data_prep.py:12–19`, `192–212`

**Fair-comparison note**: the validation set is deterministic for a given `DataConfig`, but it is not enforced globally across *all* experiments if you change sampling knobs (`shard_end`, `max_rows_per_shard`, language filter, include_query, tokenizer, etc.). If the benchmark requires a single fixed val set regardless of experimental configuration, you should lock these sampling parameters and/or externalize a fixed val shard/row list.

### 6) Metric extraction + saving — **Mostly OK**

- Machine-parseable metric lines are emitted (`METRIC {...}`) and summarized:
  - `harness/metrics.py:49–105` parses metrics
  - `harness/runner.py:205–238` writes compact `metrics.json`
- Partial results on timeout are preserved because:
  - Runner writes a summary even when killing the process.
  - Training prints `TRAINING_END` from `finally:` on normal SIGINT handling.

**Gap vs spec**: the runner’s top-level `metrics.json` does not guarantee presence of *all* keys the spec lists (e.g., `train_loss`, `tokens_per_sec`, `peak_memory_gb`, `wall_clock_seconds`) at the top level; they are present in `last` and/or `training_end_meta` when available. If a consumer expects flat keys, add them explicitly.

### 7) Reproducibility — **OK**

- Global seeds are set for Python/NumPy/Torch:
  - `harness/baseline_train.py:242–253`, called at `:315–317`
- Evaluation uses a fixed, precomputed set of validation windows (`val_ix`) so repeated evals are deterministic given fixed weights:
  - `harness/baseline_train.py:395–402`

### 8) Best-so-far tracking — **OK**

- `MetricsTracker` tracks `best_val_bpb` across checkpoints:
  - `harness/metrics.py:117–129`
- Runner extracts best from logs and also tracks it online:
  - `harness/runner.py:169–176`, `205–236`

---

## Backtest directory correctness

`backtest/` is intended as a compatibility layer, but it is currently **not usable as a CLI/entrypoint**:

### Critical: backtest wrapper files break when executed

- `backtest/runner.py` and `backtest/baseline_train.py` do `from harness... import ...` without adding the repo root to `sys.path`.
- If you run them as scripts (`python backtest/runner.py` or `python backtest/baseline_train.py`), Python sets `sys.path[0]` to `.../backtest`, so `import harness` fails.

Additionally, `backtest/baseline_train.py` is not a runnable training entrypoint (it only re-exports symbols; it does not call `main()`).

**Files/lines**:

- `backtest/runner.py:2`
- `backtest/baseline_train.py:1–2`

**Recommended fix**: make these true forwarders:

- In `backtest/runner.py`:
  - add a repo-root `sys.path` shim (copy from `harness/runner.py`), and
  - `if __name__ == '__main__': from harness.runner import main; main()`
- In `backtest/baseline_train.py`:
  - add the same shim, and
  - `if __name__ == '__main__': from harness.baseline_train import main; main()`

---

## Final verdict

- `harness/`: **PASS** (core metric/time/param-cap logic is correct)
- `backtest/`: **NEEDS FIXES** (wrapper entrypoints are not executable and cannot serve as `train_py` targets)
