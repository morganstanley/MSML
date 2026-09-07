import math
import numpy as np
import torch
import pytest

from harness.metrics import LOG2E, compute_bpb, count_parameters, MetricsTracker, compute_mfu


@pytest.mark.parametrize(
    "loss_nats,bytes_per_token",
    [
        (1.0, 4.5),
        (0.0, 1.0),
        (2.5, 1.0),
        (10.0, 2.0),
    ],
)
def test_compute_bpb_known_values(loss_nats, bytes_per_token):
    expected = float(loss_nats) * LOG2E / float(bytes_per_token)
    assert compute_bpb(loss_nats, bytes_per_token) == pytest.approx(expected, rel=0, abs=1e-12)


def test_compute_bpb_edge_cases():
    assert compute_bpb(0.0, 3.0) == 0.0
    assert compute_bpb(1000.0, 2.0) == pytest.approx(1000.0 * LOG2E / 2.0)
    with pytest.raises(ValueError):
        compute_bpb(-1.0, 1.0)
    with pytest.raises(ValueError):
        compute_bpb(1.0, 0.0)


def test_count_parameters_linear():
    m = torch.nn.Linear(100, 200, bias=True)
    n, s = count_parameters(m)
    assert n == 100 * 200 + 200
    assert isinstance(s, str)


def test_count_parameters_only_trainable():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(10, 10)
            self.b = torch.nn.Linear(10, 10)
            for p in self.b.parameters():
                p.requires_grad = False

    m = M()
    n, _ = count_parameters(m)
    expected = sum(p.numel() for p in m.a.parameters())
    assert n == expected


def test_metrics_tracker_best_val_bpb_and_history(tmp_path):
    out = tmp_path / "m.jsonl"
    tr = MetricsTracker(run_name="r", out_path_jsonl=out, bytes_per_token=1.0, param_count=10, echo_stdout=False)

    vals = [1.2, 1.0, 1.1, 0.9, 0.95, 1.3, 0.91, 0.92, 0.89, 1.05]
    for i, v in enumerate(vals):
        tr.log(step=i, t=float(i), val_bpb=v)

    assert tr.best_val_bpb == min(vals)
    assert len(tr.records) == len(vals)
    # full history preserved
    assert [r["val_bpb"] for r in tr.records] == vals


def test_compute_mfu_known_values():
    tokens_per_sec = 1000.0
    param_count = 1_000_000
    gpu_flops = 1e15
    expected = (tokens_per_sec * 6.0 * param_count) / gpu_flops
    assert compute_mfu(tokens_per_sec, param_count, gpu_flops) == pytest.approx(expected)
    assert compute_mfu(tokens_per_sec, param_count, None) is None
