import json
import sys

from harness.config import ExperimentConfig
from harness.metrics import count_parameters
import harness.baseline_train as baseline_train


def test_baseline_train_count_only_mode_fast_and_correct(tmp_path, monkeypatch, capsys):
    # Count-only mode is what the runner uses for param-cap enforcement.
    monkeypatch.setenv("HARNESS_COUNT_ONLY", "1")
    monkeypatch.setenv("HARNESS_OUT_DIR", str(tmp_path))

    argv = [
        "baseline_train.py",
        "--run_name",
        "count_only",
        "--no_compile",
        "--device",
        "cpu",
        "--dtype",
        "fp32",
        "--block_size",
        "16",
        "--batch_size",
        "2",
        "--n_layer",
        "1",
        "--n_head",
        "2",
        "--n_embd",
        "32",
        "--ffn_mult",
        "2.0",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    baseline_train.main()
    out = capsys.readouterr().out

    # Parse the printed HARNESS_PARAM_COUNT JSON.
    lines = [ln for ln in out.splitlines() if ln.startswith("HARNESS_PARAM_COUNT ")]
    assert len(lines) == 1
    meta = json.loads(lines[0].split(" ", 1)[1])

    assert "param_count" in meta
    assert meta["vocab_size"] == 256
    assert meta["block_size"] == 16

    # Manual check: should match build_model param count.
    cfg = ExperimentConfig()
    cfg.data.block_size = 16
    cfg.model.n_layer = 1
    cfg.model.n_head = 2
    cfg.model.n_embd = 32
    cfg.model.ffn_mult = 2.0
    cfg.model.vocab_size = 256
    model = baseline_train.build_model(cfg, vocab_size_override=256)
    manual_pc, _ = count_parameters(model)
    assert int(meta["param_count"]) == manual_pc
