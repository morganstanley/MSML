## Time-series-specific Conductor guidance

In addition to the generic Conductor responsibilities (representing the user, auditing past decisions, building deep understanding, steering with directives/annotations/parking), pay particular attention to time-series-specific failure modes:

- **Leakage by mis-shifted features.** Look for features whose values reflect information that wasn't available at the prediction cutoff (`shift(0)` instead of `shift(1)`, contemporaneous mids, won-only filters applied to incoming-flow targets, scalers fit on combined train+test). When a debrief looks "too good", check the strategy code before believing the metric.
- **Walk-forward integrity.** Confirm experiments use date-based splits with an embargo, not random splits. A leaderboard column that doesn't reproduce on a fresh fold is a red flag.
- **Multiple-comparison inflation.** With dozens of experiments differing only in seed / horizon / threshold, the best single number is upward-biased. Annotate suspiciously sharp leaders as `quarantined` and direct the strategist to a control-vs-best comparison on a held-out window before treating them as champions.
- **Regime shifts.** The dataset's metadata often flags regime changes (e.g., a 2× volume step). When a model trained pre-shift is evaluated post-shift, the metric collapses. Direct the strategist (via directives) to either retrain on the recent regime or report metrics split by regime.
- **Sharpe-style metrics' tail dependence.** A long-short Sharpe can be driven by a handful of large gains. Periodically request decile / quintile breakdowns via a directive — pure rank correlation alone hides this.
