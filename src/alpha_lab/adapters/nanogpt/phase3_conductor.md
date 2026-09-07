## NanoGPT-speedrun-specific Conductor guidance

The metric is wall-clock seconds (lower is better) for a fixed validation-loss target. Specific failure modes to watch for:

- **Loss-target gaming.** A run that exits early because it hit the target by luck on a noisy step is not a real speed-up. Direct the strategist (via directive) to require N consecutive steps below the target before declaring done.
- **Setup cost excluded from timing.** Some experiments start the wall-clock after dataloader warm-up or compile. The honest number includes everything. Peek at the timing helper; if setup is excluded, mark `quarantined` and request a fix.
- **Numerical instability disguised as speed.** Aggressive bf16 / muP / large-LR tricks can pass the target on one seed and diverge on others. Request a 3-seed re-run via a directive before annotating a row as `champion`.
- **Same-arch microoptimizations stacking.** Many tiny kernel/optimizer tweaks each report +1% but don't compose. When a "stacked" experiment underperforms the sum of its parts, mark it `quarantined` and direct the strategist to a controlled additivity check.
