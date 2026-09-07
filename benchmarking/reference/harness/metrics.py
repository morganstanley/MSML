"""Metrics utilities for the experiment harness."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOG2E = math.log2(math.e)


def compute_bpb(loss_nats: float, bytes_per_token: float) -> float:
    """Convert cross-entropy loss in nats/token to bits/byte."""
    loss_nats = float(loss_nats)
    bytes_per_token = float(bytes_per_token)
    if loss_nats < 0:
        raise ValueError(f"loss_nats must be >= 0, got {loss_nats}")
    if bytes_per_token <= 0:
        raise ValueError(f"bytes_per_token must be > 0, got {bytes_per_token}")
    return loss_nats * LOG2E / bytes_per_token


def count_parameters(model) -> Tuple[int, str]:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n >= 1_000_000_000:
        s = f"{n/1e9:.2f}B"
    elif n >= 1_000_000:
        s = f"{n/1e6:.2f}M"
    elif n >= 1_000:
        s = f"{n/1e3:.2f}K"
    else:
        s = str(n)
    return int(n), s


def compute_mfu(tokens_per_sec: float, param_count: int, gpu_flops: Optional[float]) -> Optional[float]:
    """Approximate MFU. Assumes ~6*P FLOPs per token for a dense decoder-only transformer."""
    if gpu_flops is None:
        return None
    flops_per_token = 6.0 * float(param_count)
    achieved = float(tokens_per_sec) * flops_per_token
    return achieved / float(gpu_flops)


_METRIC_JSON_RE = re.compile(r"^METRIC\s+(\{.*\})\s*$")
_TRAINING_START_RE = re.compile(r"^TRAINING_START\s+(\{.*\})\s*$")
_TRAINING_END_RE = re.compile(r"^TRAINING_END\s+(\{.*\})\s*$")
_PARAM_RE = re.compile(r"^HARNESS_PARAM_COUNT\s+(\{.*\})\s*$")


def extract_metrics_from_log(log_path: str | Path) -> Dict[str, Any]:
    log_path = Path(log_path)
    metrics: List[Dict[str, Any]] = []
    best = None
    last = None
    training_start_meta = None
    training_end_meta = None
    param_meta = None

    if not log_path.exists():
        return {"metrics": [], "best_val_bpb": None, "last": None}

    for line in log_path.read_text(errors="ignore").splitlines():
        m = _METRIC_JSON_RE.match(line)
        if m:
            try:
                d = json.loads(m.group(1))
            except Exception:
                continue
            metrics.append(d)
            last = d
            vb = d.get("val_bpb")
            if vb is not None:
                best = vb if best is None else min(best, vb)
            continue

        m = _TRAINING_START_RE.match(line)
        if m:
            try:
                training_start_meta = json.loads(m.group(1))
            except Exception:
                training_start_meta = None
            continue

        m = _TRAINING_END_RE.match(line)
        if m:
            try:
                training_end_meta = json.loads(m.group(1))
            except Exception:
                training_end_meta = None
            continue

        m = _PARAM_RE.match(line)
        if m:
            try:
                param_meta = json.loads(m.group(1))
            except Exception:
                param_meta = None

    return {
        "metrics": metrics,
        "best_val_bpb": best,
        "last": last,
        "training_start_meta": training_start_meta,
        "training_end_meta": training_end_meta,
        "param_meta": param_meta,
    }


@dataclass
class MetricsTracker:
    run_name: str
    out_path_jsonl: Path
    bytes_per_token: float
    param_count: int
    gpu_peak_flops: Optional[float] = None
    echo_stdout: bool = True

    best_val_bpb: float | None = None
    records: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, **kwargs):
        rec = {"t": kwargs.pop("t", None), "step": kwargs.pop("step", None), **kwargs}

        if rec.get("val_loss") is not None and rec.get("val_bpb") is None:
            rec["val_bpb"] = compute_bpb(rec["val_loss"], self.bytes_per_token)

        vb = rec.get("val_bpb")
        if vb is not None:
            self.best_val_bpb = vb if self.best_val_bpb is None else min(self.best_val_bpb, vb)

        tps = rec.get("tokens_per_sec")
        if tps is not None and rec.get("mfu") is None:
            rec["mfu"] = compute_mfu(tps, self.param_count, self.gpu_peak_flops)

        self.records.append(rec)

        self.out_path_jsonl.parent.mkdir(parents=True, exist_ok=True)
        line = "METRIC " + json.dumps(rec, sort_keys=True)
        with self.out_path_jsonl.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        if self.echo_stdout:
            print(line, flush=True)

    def summary(self) -> Dict[str, Any]:
        last = self.records[-1] if self.records else None
        return {"run_name": self.run_name, "best_val_bpb": self.best_val_bpb, "last": last, "num_records": len(self.records)}
