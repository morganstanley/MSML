import json
import os
import sys
import time
from pathlib import Path

import pytest

from harness.runner import run_experiment


def _write_script(path: Path, text: str):
    path.write_text(text)
    path.chmod(0o755)


def test_time_limit_enforcement_kills_infinite_loop(tmp_path):
    # Script prints TRAINING_START then loops.
    train_py = tmp_path / "train_forever.py"
    _write_script(
        train_py,
        r'''
import json, os, time, signal, sys

if os.environ.get("HARNESS_COUNT_ONLY","0") == "1":
    print("HARNESS_PARAM_COUNT " + json.dumps({"param_count": 1}))
    raise SystemExit(0)

print("TRAINING_START " + json.dumps({"run_name": "forever", "param_count": 1}))
sys.stdout.flush()

# Ignore SIGINT so the runner has to escalate.
signal.signal(signal.SIGINT, signal.SIG_IGN)

while True:
    time.sleep(0.5)
''',
    )

    out_dir = tmp_path / "out"
    t0 = time.perf_counter()
    res = run_experiment(
        train_py=str(train_py),
        out_dir=str(out_dir),
        time_limit_seconds=2,
        param_cap=100,
        grace_seconds=2,
    )
    dt = time.perf_counter() - t0

    assert dt < 10, "runner should enforce timeout promptly"
    assert res["status"] == "timeout"


def test_parameter_cap_enforcement_rejects_before_training(tmp_path):
    train_py = tmp_path / "train_big.py"
    _write_script(
        train_py,
        r'''
import json, os
if os.environ.get("HARNESS_COUNT_ONLY","0") == "1":
    print("HARNESS_PARAM_COUNT " + json.dumps({"param_count": 200_000_000}))
    raise SystemExit(0)
print("TRAINING_START " + json.dumps({"run_name": "big", "param_count": 200_000_000}))
''',
    )
    out_dir = tmp_path / "out"
    res = run_experiment(train_py=str(train_py), out_dir=str(out_dir), time_limit_seconds=5, param_cap=100_000_000)
    assert res["status"] == "rejected_param_cap"
    assert (out_dir / "metrics.json").exists()


def test_parameter_cap_boundary(tmp_path):
    def run_with(pc: int):
        train_py = tmp_path / f"train_{pc}.py"
        _write_script(
            train_py,
            f"""
import json, os, time
if os.environ.get('HARNESS_COUNT_ONLY','0') == '1':
    print('HARNESS_PARAM_COUNT ' + json.dumps({{'param_count': {pc}}}))
    raise SystemExit(0)
print('TRAINING_START ' + json.dumps({{'run_name':'r','param_count': {pc}}}))
print('METRIC ' + json.dumps({{'val_bpb': 1.0, 'train_loss': 2.0, 'tokens_per_sec': 3.0, 'peak_memory_gb': 0.1}}))
print('TRAINING_END ' + json.dumps({{'wall_clock_seconds': 0.01, 'reason':'done'}}))
""",
        )
        out_dir = tmp_path / f"out_{pc}"
        return run_experiment(train_py=str(train_py), out_dir=str(out_dir), time_limit_seconds=5, param_cap=100_000_000)

    assert run_with(99_900_000)["status"] in {"ok", "timeout"}  # should be accepted
    assert run_with(100_100_000)["status"] == "rejected_param_cap"


def test_best_so_far_tracking(tmp_path):
    vals = [1.2, 1.0, 0.9, 0.95, 1.1]
    train_py = tmp_path / "train_metrics.py"
    metric_lines = "\n".join([f"print('METRIC ' + json.dumps({{'val_bpb': {v}}}))" for v in vals])
    _write_script(
        train_py,
        f"""
import json, os
if os.environ.get('HARNESS_COUNT_ONLY','0') == '1':
    print('HARNESS_PARAM_COUNT ' + json.dumps({{'param_count': 1}}))
    raise SystemExit(0)
print('TRAINING_START ' + json.dumps({{'run_name':'m','param_count': 1}}))
{metric_lines}
print('TRAINING_END ' + json.dumps({{'wall_clock_seconds': 0.02, 'reason':'done'}}))
""",
    )
    out_dir = tmp_path / "out"
    res = run_experiment(train_py=str(train_py), out_dir=str(out_dir), time_limit_seconds=5, param_cap=100)
    assert res["best_val_bpb"] == pytest.approx(0.9)
    # compact key should match
    mj = json.loads((out_dir / "metrics.json").read_text())
    assert mj["val_bpb"] == pytest.approx(0.9)


def test_error_handling_saves_partial_results(tmp_path):
    train_py = tmp_path / "train_crash.py"
    _write_script(
        train_py,
        r'''
import json, os
if os.environ.get("HARNESS_COUNT_ONLY","0") == "1":
    print("HARNESS_PARAM_COUNT " + json.dumps({"param_count": 1}))
    raise SystemExit(0)
print("TRAINING_START " + json.dumps({"run_name": "crash", "param_count": 1}))
print("METRIC " + json.dumps({"val_bpb": 1.5}))
print("METRIC " + json.dumps({"val_bpb": 1.1}))
raise RuntimeError("boom")
''',
    )
    out_dir = tmp_path / "out"
    res = run_experiment(train_py=str(train_py), out_dir=str(out_dir), time_limit_seconds=5, param_cap=100)
    assert res["status"] == "error"
    assert res["best_val_bpb"] == pytest.approx(1.1)
    assert (out_dir / "metrics.json").exists()


def test_results_file_format_contains_required_keys(tmp_path):
    train_py = tmp_path / "train_one_metric.py"
    _write_script(
        train_py,
        r'''
import json, os
if os.environ.get("HARNESS_COUNT_ONLY","0") == "1":
    print("HARNESS_PARAM_COUNT " + json.dumps({"param_count": 123}))
    raise SystemExit(0)
print("TRAINING_START " + json.dumps({"run_name": "r", "param_count": 123}))
print("METRIC " + json.dumps({"val_bpb": 1.0, "train_loss": 2.0, "tokens_per_sec": 10.0, "peak_memory_gb": 0.0, "param_count": 123}))
print("TRAINING_END " + json.dumps({"wall_clock_seconds": 0.01, "reason":"done"}))
''',
    )
    out_dir = tmp_path / "out"
    run_experiment(train_py=str(train_py), out_dir=str(out_dir), time_limit_seconds=5, param_cap=1000)

    data = json.loads((out_dir / "metrics.json").read_text())
    for k in ["val_bpb", "train_loss", "tokens_per_sec", "param_count", "peak_memory_gb", "wall_clock_seconds"]:
        assert k in data
