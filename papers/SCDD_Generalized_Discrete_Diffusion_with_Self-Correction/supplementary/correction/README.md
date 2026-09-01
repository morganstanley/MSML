# Correction Ablation Experiments: SCDD

This folder contains the code and instructions to reproduce the two correction ablation experiments demonstrating that SCDD's correction mechanism is beneficial.

## Files

| File | Description |
|------|-------------|
| `ablation_correction.py` | Experiment 1: Gen-PPL with/without corrections |
| `corruption_recovery.py` | Experiment 2: Corruption recovery at last step |
| `pbs_ablation_correction.sh` | Shell script for Experiment 1 (template) |
| `pbs_corruption_recovery.sh` | Shell script for Experiment 2 (template) |

## Running the Experiments

The scripts import the repository's root `diffusion.py`; no source replacement
is required.

### 1. Experiment 1: No-Correction Ablation

Generates text with corrections enabled and disabled, compares Gen-PPL (GPT-2-large). Reports per-batch PPL with mean ± standard error.

```bash
python ablation_correction.py \
    --checkpoint_path /path/to/scdd.ckpt \
    --num_steps 128 \
    --batch_size 16 \
    --num_batches 16 \
    --nucleus_p 0.9 \
    --output_path ablation_correction.json
```

Or edit `pbs_ablation_correction.sh` to fill in `<REPO_DIR>` and `<DATA_DIR>`, then:

```bash
bash pbs_ablation_correction.sh
```

### 2. Experiment 2: Corruption Recovery

Corrupts K tokens in clean validation text, runs one SCDD denoising step at the last-step noise level, measures touch/recovery/damage rates. Saves clean, corrupted, and corrected text per sample.

```bash
python corruption_recovery.py \
    --checkpoint_path /path/to/scdd.ckpt \
    --num_steps 128 \
    --batch_size 16 \
    --num_batches 8 \
    --corrupt_counts 5,10,20,50 \
    --nucleus_p 0.9 \
    --data_cache_dir /path/to/data \
    --output_path corruption_recovery.json
```

Or edit `pbs_corruption_recovery.sh` to fill in `<REPO_DIR>` and `<DATA_DIR>`, then:

```bash
bash pbs_corruption_recovery.sh
```

**Note:** The `--data_cache_dir` argument overrides the `data.cache_dir` stored in the checkpoint config, which may point to a path that does not exist on your system.

## Output Format

### Experiment 1

JSON with per-variant results:
- `gen_ppl_mean`, `gen_ppl_stderr`, `gen_ppl_per_batch`: per-batch PPL values
- `text_samples`: all generated text

### Experiment 2

JSON with per-corruption-level results:
- `touch_rate_mean/stderr`, `exact_recovery_rate_mean/stderr`, `damage_rate_mean/stderr`: per-batch metrics
- `ppl_clean/corrupted/corrected_mean/stderr`: perplexity before and after correction
- `samples`: list of per-sample records, each containing:
  - `clean_text`, `corrupted_text`, `corrected_text`
  - `corrupted_positions`, `touched_positions`, `recovered_positions`
