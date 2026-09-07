"""d6_cuda baseline: torch eager gelu(A @ B + bias), report per TASK.md.

Runs the pinned-seed task, times the eager composition, checks it against
the fp32 reference, and writes a contract-conformant kernel_report.json.
This is the floor every candidate kernel must beat — and a working example
of the evidence contract.

Usage: python reference.py [out_path=kernel_report.json]
"""

from __future__ import annotations

import json
import statistics
import sys

import torch
import torch.nn.functional as F

M = N = K = 4096
SEED = 20260801
ATOL = RTOL = 2e-2  # gate: err_norm = max(|out-ref| / (ATOL + RTOL*|ref|)) <= 1.0
WARMUP = 50
ITERS = 200
FLOPS_PER_CALL = 2 * M * N * K


def make_inputs():
    g = torch.Generator(device="cuda").manual_seed(SEED)
    A = torch.randn(M, K, generator=g, device="cuda", dtype=torch.float32).half()
    B = torch.randn(K, N, generator=g, device="cuda", dtype=torch.float32).half()
    bias = torch.randn(N, generator=g, device="cuda", dtype=torch.float32).half()
    return A, B, bias


def reference_fp32(A, B, bias):
    return F.gelu(A.float() @ B.float() + bias.float())


def candidate(A, B, bias):
    """The implementation under test — replace this in your experiment."""
    return F.gelu(A @ B + bias)


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "kernel_report.json"
    A, B, bias = make_inputs()
    out = candidate(A, B, bias)
    ref = reference_fp32(A, B, bias)
    diff = (out.float() - ref).abs()
    max_abs_err = float(diff.max())
    err_norm = float((diff / (ATOL + RTOL * ref.abs())).max())

    for _ in range(WARMUP):
        candidate(A, B, bias)
    torch.cuda.synchronize()
    samples_ms = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        candidate(A, B, bias)
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end))

    med_ms = statistics.median(samples_ms)
    tflops = FLOPS_PER_CALL / (med_ms / 1e3) / 1e12
    report = {
        "fingerprint": {"op": "gemm_bias_gelu", "m": M, "n": N, "k": K,
                        "dtype": "fp16"},
        "implementation": "torch eager baseline: F.gelu(A @ B + bias)",
        "correctness": {"err_norm": err_norm, "max_abs_err": max_abs_err,
                        "atol": ATOL, "rtol": RTOL},
        "timing": {"warmup_iters": WARMUP, "timed_iters": ITERS,
                   "samples_ms": samples_ms},
        "tflops": round(tflops, 3),
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"err_norm={err_norm:.4f} (gate <=1.0, atol=rtol={ATOL}) "
          f"max_abs_err={max_abs_err:.4f} "
          f"median={med_ms:.3f}ms tflops={tflops:.1f} -> {out_path}")
    return 0 if err_norm <= 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
