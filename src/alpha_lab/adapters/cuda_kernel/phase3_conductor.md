## CUDA-kernel-specific Conductor guidance

CUDA kernel generation has failure modes that look like progress on the leaderboard but aren't:

- **Correctness regressions hidden by throughput.** A kernel that returns wrong values can run arbitrarily fast. Always check that the experiment's eval framework verified correctness (`atol`/`rtol` tolerance pass) before annotating a high-throughput row as `champion`. If correctness wasn't checked, mark `quarantined` and direct the strategist (via directive) to re-evaluate with a correctness gate.
- **Synchronization-cost shortcuts.** Async kernels can omit `cudaDeviceSynchronize` before timing, producing fake speedups. Peek at the experiment code, not just `metrics.json`. Direct the strategist to use the framework's timing helper rather than ad-hoc `time.time()`.
- **Specialization over a too-narrow shape.** A kernel hand-tuned for one input size won't generalize. Direct the strategist to evaluate across the framework's full shape suite before treating any result as a champion.
- **Memory leaks in long runs.** Symptom: throughput drops over wall-time. If the leaderboard's recent kernels run shorter than older ones, request a long-form re-run via a directive.
