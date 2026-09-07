# d6_cuda — fused GEMM + bias + GELU

Optimize a single fused kernel on one A100:

```
out = gelu(A @ B + bias)
```

- `A`: (4096, 4096) fp16, `B`: (4096, 4096) fp16, `bias`: (4096,) fp16
- output: (4096, 4096) fp16, accumulation in fp32
- GELU: exact (erf) form, `torch.nn.functional.gelu(x)` default

Any implementation technology available in the environment is allowed
(Triton, CUDA C via `torch.utils.cpp_extension.load_inline`, torch.compile,
CUTLASS bindings). The torch eager composition
`F.gelu(A @ B + bias)` is the baseline to beat, not an acceptable answer.

## Fixed inputs (correctness identity)

Inputs are generated once per experiment with a pinned seed so every
implementation is checked against the same tensors:

```python
g = torch.Generator(device="cuda").manual_seed(20260801)
A = torch.randn(4096, 4096, generator=g, device="cuda", dtype=torch.float32).half()
B = torch.randn(4096, 4096, generator=g, device="cuda", dtype=torch.float32).half()
bias = torch.randn(4096, generator=g, device="cuda", dtype=torch.float32).half()
```

Reference: compute in fp32 (`A.float() @ B.float() + bias.float()`, exact
GELU, then compare against the candidate output cast to fp32). Outputs reach
magnitude ~350, where fp16's own quantization step is ~0.25, so the gate is
a **normalized elementwise error** with `atol = rtol = 2e-2`:

```
err_norm = max( |out - ref| / (atol + rtol * |ref|) )
```

**Correctness gate: `err_norm <= 1.0`.** A fast kernel that fails the gate
scores nothing. Calibration, measured on an idle A100: the torch eager
baseline lands at err_norm 0.064 (max_abs_err 0.2421), so a correct kernel
has ~15x headroom; dropping the bias or using a tanh-approximation GELU
pushes err_norm far above 1.

## Timing protocol

- ≥ 50 warmup iterations, then ≥ 200 timed iterations
- one CUDA-event pair per iteration, `torch.cuda.synchronize()` before and
  after the timed block
- keep every per-iteration millisecond sample; the referee recomputes the
  score from the raw samples, not from your summary

## Metric (pinned FLOP convention)

```
TFLOP/s = 2 * 4096^3 / median(samples_ms in seconds) / 1e12
```

The epilogue (bias + GELU) is deliberately excluded from the FLOP count —
everyone computes the same numerator, so the ranking is by time alone.

## Evidence contract — `experiments/<name>/results/kernel_report.json`

```json
{
  "fingerprint": {"op": "gemm_bias_gelu", "m": 4096, "n": 4096, "k": 4096, "dtype": "fp16"},
  "implementation": "one-line description (e.g. triton 128x128x64 pipelined)",
  "correctness": {"err_norm": 0.064, "max_abs_err": 0.2421, "atol": 0.02, "rtol": 0.02},
  "timing": {"warmup_iters": 50, "timed_iters": 200, "samples_ms": [/* raw samples */]},
  "tflops": 312.4
}
```

Record the achieved TFLOP/s as the experiment's metric. Reports with a
mismatched fingerprint, a failed correctness block, or fewer than 50 timing
samples are rejected by the referee. `reference.py` in this directory runs
the baseline and emits a contract-conformant report — start there.
