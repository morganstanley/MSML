"""Experiment runner.

Runs a train.py-like script under:
- wall-clock training budget (default 1200s)
- parameter cap (default 100M)

Assumptions about the train.py interface (fulfilled by harness/baseline_train.py):
- When env var HARNESS_COUNT_ONLY=1 is set, it prints:
    HARNESS_PARAM_COUNT {"param_count": ..., ...}
  and exits quickly.
- When training begins (after data load + optional compile), it prints:
    TRAINING_START {...}
- It emits metric JSON lines:
    METRIC {...}

The runner parses the log and writes a compact results/metrics.json file.

Invocation
----------
Prefer:  python -m harness.runner ...
This file also supports: python harness/runner.py ... (best effort).
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure repo root is on sys.path when executed as a script
_repo_root = _Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Support both module execution and script execution.
try:
    from .metrics import extract_metrics_from_log
except Exception:  # pragma: no cover
    from harness.metrics import extract_metrics_from_log  # type: ignore


def _run_count_only(train_py: Path, extra_args: list[str]) -> Dict[str, Any]:
    env = os.environ.copy()
    env["HARNESS_COUNT_ONLY"] = "1"
    cmd = [sys.executable, str(train_py)] + extra_args
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout

    meta = None
    for line in out.splitlines():
        if line.startswith("HARNESS_PARAM_COUNT"):
            try:
                meta = json.loads(line.split(" ", 1)[1])
            except Exception:
                meta = None
    if meta is None:
        raise RuntimeError(
            "Failed to get param count from train.py. Expected it to print 'HARNESS_PARAM_COUNT {...}'.\n"
            f"Output:\n{out[-4000:]}"
        )
    meta["_count_only_returncode"] = p.returncode
    return meta


def run_experiment(
    *,
    train_py: str,
    out_dir: str = "harness/results",
    time_limit_seconds: int = 1200,
    param_cap: int = 100_000_000,
    extra_args: Optional[list[str]] = None,
    grace_seconds: int = 5,
) -> Dict[str, Any]:
    train_py_p = Path(train_py)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    extra_args = extra_args or []

    # 1) parameter count (pre-flight)
    count_meta = _run_count_only(train_py_p, extra_args)
    param_count = int(count_meta.get("param_count", -1))
    if param_count < 0:
        raise RuntimeError(f"Bad param_count from train.py: {count_meta}")

    # Spec: STRICTLY < param_cap
    if param_count >= param_cap:
        res = {
            # Required top-level summary keys (for leaderboard/backtests)
            'val_bpb': None,
            'train_loss': None,
            'tokens_per_sec': None,
            'peak_memory_gb': None,
            'wall_clock_seconds': None,
            'status': 'rejected_param_cap',
            'param_count': param_count,
            'param_cap': param_cap,
            'count_meta': count_meta,
        }
        (out_dir_p / 'metrics.json').write_text(json.dumps(res, indent=2, sort_keys=True))
        return res

    # 2) training run
    log_path = out_dir_p / "train.log"
    env = os.environ.copy()
    env.pop("HARNESS_COUNT_ONLY", None)
    env["HARNESS_OUT_DIR"] = str(out_dir_p)
    env["HARNESS_TIME_LIMIT_SECONDS"] = str(time_limit_seconds)

    # Systemic stability guard:
    # Many Phase 3 failures were caused by `torch.compile` + TorchInductor CUDA Graphs
    # interacting badly with lazy RoPE cache initialization inside the compiled forward.
    # Disabling Inductor cudagraphs avoids the
    #   "accessing tensor output of CUDAGraphs that has been overwritten" RuntimeError
    # while still allowing `torch.compile` to be used.
    #
    # If a user explicitly set this env var, respect it.
    # NOTE: Some torch versions also gate cudagraph usage behind
    # TORCHINDUCTOR_CUDAGRAPH_OR_ERROR. Setting both makes the intent explicit and
# has proven to eliminate a large fraction of systemic Phase-3 crashes.
    env.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
    env.setdefault("TORCHINDUCTOR_CUDAGRAPH_OR_ERROR", "0")

    cmd = [sys.executable, "-u", str(train_py_p)] + extra_args
    with log_path.open("w", encoding="utf-8") as logf:
        p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # tail the log while enforcing budget AFTER TRAINING_START
    t_budget_start: Optional[float] = None
    sent_interrupt = False
    interrupt_time: Optional[float] = None

    def _read_new_lines(fp, pos):
        fp.seek(pos)
        data = fp.read()
        return data, fp.tell()

    pos = 0
    best_val_bpb = None
    training_start_meta = None
    training_end_meta = None
    last_metric = None

    while True:
        ret = p.poll()

        # parse incremental logs
        try:
            with log_path.open("r", encoding="utf-8", errors="ignore") as fp:
                chunk, pos = _read_new_lines(fp, pos)
        except Exception:
            chunk = ""

        if chunk:
            for line in chunk.splitlines():
                if line.startswith("TRAINING_START"):
                    try:
                        training_start_meta = json.loads(line.split(" ", 1)[1])
                    except Exception:
                        training_start_meta = None
                    if t_budget_start is None:
                        t_budget_start = time.perf_counter()

                    # Optional: if runtime reports param_count, enforce again.
                    try:
                        runtime_pc = int(training_start_meta.get("param_count", -1)) if training_start_meta else -1
                        if runtime_pc >= param_cap:
                            # Stop immediately; status will be rejected below.
                            p.send_signal(signal.SIGINT)
                            sent_interrupt = True
                            interrupt_time = time.perf_counter()
                    except Exception:
                        pass

                if line.startswith("TRAINING_END"):
                    try:
                        training_end_meta = json.loads(line.split(" ", 1)[1])
                    except Exception:
                        training_end_meta = None

                if line.startswith("METRIC "):
                    try:
                        d = json.loads(line.split(" ", 1)[1])
                        last_metric = d
                        vb = d.get("val_bpb")
                        if vb is not None:
                            best_val_bpb = vb if best_val_bpb is None else min(best_val_bpb, vb)
                    except Exception:
                        pass

        if t_budget_start is not None and not sent_interrupt:
            if (time.perf_counter() - t_budget_start) >= time_limit_seconds:
                try:
                    p.send_signal(signal.SIGINT)
                finally:
                    sent_interrupt = True
                    interrupt_time = time.perf_counter()

        # Escalate termination quickly to keep the wall-clock budget hard.
        # SIGINT is sent at the budget boundary; if the process does not exit promptly,
        # escalate: SIGTERM then SIGKILL.
        if sent_interrupt and ret is None and interrupt_time is not None:
            dt = (time.perf_counter() - interrupt_time)
            try:
                if dt > max(1.0, grace_seconds * 0.5):
                    p.send_signal(signal.SIGTERM)
                if dt > grace_seconds:
                    p.kill()
            except Exception:
                pass

        if ret is not None:
            break

        time.sleep(0.25)

    parsed = extract_metrics_from_log(log_path)
    metrics_list = parsed.get('metrics', [])

    merged_end = parsed.get("training_end_meta") or training_end_meta

    # status handling
    status = "ok" if p.returncode == 0 else "error"
    if sent_interrupt:
        # Treat our own interrupt as a normal timeout outcome.
        status = "timeout"
    # Also treat explicit TRAINING_END reason as timeout.
    if isinstance(merged_end, dict) and merged_end.get("reason") in {"time_budget", "signal_2", "signal_15"}:
        status = "timeout"

    # runtime param-cap reject
    if isinstance(training_start_meta, dict):
        rpc = int(training_start_meta.get("param_count", -1))
        if rpc >= param_cap:
            status = "rejected_param_cap"


    # Build compact, spec-friendly summary fields.
    # Prefer the best val_bpb seen; for other fields, take the most recent non-null occurrence.
    def _last_non_null(key: str):
        for d in reversed(metrics_list):
            if isinstance(d, dict) and d.get(key) is not None:
                return d.get(key)
        return None

    val_bpb = parsed.get('best_val_bpb')
    train_loss = _last_non_null('train_loss')
    tokens_per_sec = _last_non_null('tokens_per_sec')

    peak_memory_gb = None
    for d in metrics_list:
        pm = d.get('peak_memory_gb') if isinstance(d, dict) else None
        if pm is not None:
            peak_memory_gb = pm if peak_memory_gb is None else max(peak_memory_gb, pm)

    wall_clock_seconds = None
    if isinstance(merged_end, dict):
        wall_clock_seconds = merged_end.get('wall_clock_seconds')
    res = {
        # Required top-level summary keys (for leaderboard/backtests)
        'val_bpb': val_bpb,
        'train_loss': train_loss,
        'tokens_per_sec': tokens_per_sec,
        'peak_memory_gb': peak_memory_gb,
        'wall_clock_seconds': wall_clock_seconds,
        "status": status,
        "returncode": p.returncode,
        "time_limit_seconds": time_limit_seconds,
        "param_cap": param_cap,
        "param_count": param_count,
        "count_meta": count_meta,
        "best_val_bpb": parsed.get("best_val_bpb") if parsed.get("best_val_bpb") is not None else best_val_bpb,
        "last": parsed.get("last") if parsed.get("last") is not None else last_metric,
        "training_start_meta": parsed.get("training_start_meta") or training_start_meta,
        "training_end_meta": merged_end,
        "log_path": str(log_path),
        "num_metric_records": len(parsed.get("metrics", [])),
    }
    (out_dir_p / "metrics.json").write_text(json.dumps(res, indent=2, sort_keys=True))

    # Convenience: if the training script wrote a detailed metrics.json in a run subdir,
    # copy it up to the runner out_dir for easier discovery.
    try:
        run_name = None
        if isinstance(res.get("training_start_meta"), dict):
            run_name = res["training_start_meta"].get("run_name")
        # baseline_train writes to <HARNESS_OUT_DIR>/<cfg.run_name>/metrics.json
        if run_name:
            detailed = out_dir_p / str(run_name) / "metrics.json"
            if detailed.exists():
                (out_dir_p / "metrics_detailed.json").write_text(detailed.read_text())
                res["metrics_detailed_path"] = str(detailed)
                # update summary file with the extra pointer
                (out_dir_p / "metrics.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    except Exception:
        pass

    return res


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("train_py", type=str)
    ap.add_argument("--out_dir", type=str, default="harness/results")
    ap.add_argument("--time_limit_seconds", type=int, default=1200)
    ap.add_argument("--param_cap", type=int, default=100_000_000)

    args, extra = ap.parse_known_args()

    # allow conventional '--' separator
    if extra and extra[0] == "--":
        extra = extra[1:]

    res = run_experiment(
        train_py=args.train_py,
        out_dir=args.out_dir,
        time_limit_seconds=args.time_limit_seconds,
        param_cap=args.param_cap,
        extra_args=extra,
    )
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
