# Alpha-Lab Benchmark Leaderboard

## LLM Speedrun — val_bpb (lower is better)

**Task**: Train a small language model (<100M params) on PleIAs SYNTH corpus
within a 20-minute wall-clock budget, minimizing validation bits-per-byte.

**Setup**: Frozen Phase 2 evaluation harness, 20 experiments per run, 4x GPUs.
Phases 0 and 1 are skipped (adapter pre-loaded), Phase 3 runs fresh each time.

See [`run_benchmark.py`](benchmarking/run_benchmark.py) for details on how to run.

---

<!-- RESULTS_TABLE_START -->
| Rank | Method | Mean val_bpb | Best val_bpb | Success Rate | Runs | Date |
|------|--------|-------------|-------------|-------------|------|------|
| 1 | Opus 4.6 (paper) | 0.7578 | 0.7578 | 72% | 1 | 2025-03-27 |
| 2 | GPT-5.2 + Phase 3 variance (paper) | 0.9866 ± 0.0273 | 0.9455 | 71% | 4 | 2025-03-27 |
| 3 | GPT-5.2 (paper) | 0.9697 | 0.9697 | 64% | 1 | 2025-03-19 |
<!-- RESULTS_TABLE_END -->

### How to read this table

- **Mean val_bpb**: Average best val_bpb across all runs (± standard deviation if multiple runs).
  Lower is better.
- **Best val_bpb**: Single best val_bpb achieved across all runs.
- **Success Rate**: Percentage of experiments that produced a valid val_bpb result.
- **Runs**: Number of independent benchmark runs.
- **Date**: When the benchmark was run.

### Paper results notes

- **Opus 4.6 (paper)**: Claude Opus 4.6 via Bedrock, 50 experiments, full pipeline.
  Best single experiment: `shallow_10l_752d` at 0.7578.
- **GPT-5.2 (paper)**: GPT-5.2 via OpenAI, 50 experiments, full pipeline.
- **GPT-5.2 + Phase 3 variance (paper)**: 4 independent Phase 3 runs using
  GPT-5.2, seeded from the same Phase 1/2 artifacts. Shows natural variance
  in the experiment generation process. Individual run bests:
  0.9455, 0.9642, 1.0059, 1.0109.
