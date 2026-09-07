"""Standalone entry point for the Alpha Lab finding-verifier.

Reads an existing workspace (read-only except its `verify/` subtree) and runs the
three-agent verifier (User-Rep / Worker / Critic) to independently re-implement and
stress-test the system's findings, emitting executed Jupyter notebooks as proof.

Usage:
  cd <repo root>   # so .token_cache.json is found
  PYTHONPATH=src <p312-python> scripts/verify_workspace.py \
      --config data/verify_etfflow_o48_config.json

Auth: uses the same .token_cache.json as the running system (bedrock provider).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from alpha_lab.adapter_loader import load_adapter
from alpha_lab.client import get_provider
from alpha_lab.config import load_config
from alpha_lab.verifier import Verifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("verify_workspace")


def make_event_callback(log_file):
    def cb(ev):
        if isinstance(ev, str):
            line = ev
        else:
            parts = [getattr(ev, "type", ev.__class__.__name__)]
            for attr in ("name", "status", "detail"):
                v = getattr(ev, attr, None)
                if v:
                    parts.append(str(v))
            line = " ".join(parts)
        line = ("[verify] " + line)[:400]
        print(line, flush=True)
        try:
            log_file.write(line + "\n")
            log_file.flush()
        except Exception:
            pass
    return cb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workspace", default=None,
                    help="comma-separated workspace dir(s); overrides config verify_workspace. "
                         "Each is verified in turn.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    raw = json.loads(cfg_path.read_text())
    config = load_config(cfg_path)  # TaskConfig (thresholds/caps); ignores verifier_* keys
    repo_root = Path(__file__).resolve().parent.parent

    ws_names = [w.strip() for w in (args.workspace or raw["verify_workspace"]).split(",") if w.strip()]

    provider_name = raw.get("verifier_provider", config.provider)
    model = raw.get("verifier_model", config.model)
    effort = raw.get("verifier_reasoning_effort", config.reasoning_effort)
    log.info("building provider=%s model=%s effort=%s (uses .token_cache.json)", provider_name, model, effort)
    try:
        provider = get_provider(provider_name, api_key=None)
    except Exception as e:
        log.error("failed to build provider %s: %s", provider_name, e)
        return 1

    data_path = config.data_path
    if not Path(data_path).exists():
        log.warning("data_path %s does not exist — notebooks that load data will fail", data_path)
    nb_run = str(repo_root / "scripts" / "nb_run.py")
    python_exe = sys.executable  # the p312 interpreter (has torch/nbformat)

    all_outcomes: dict[str, object] = {}
    for wsname in ws_names:
        workspace = (repo_root / wsname).resolve() if not Path(wsname).is_absolute() else Path(wsname)
        log.info("=== VERIFYING WORKSPACE: %s ===", wsname)
        if not (workspace / "experiments.db").exists():
            log.error("workspace %s has no experiments.db — skipping", workspace)
            all_outcomes[wsname] = "SKIPPED (no experiments.db)"; continue
        if not (workspace / "adapter").exists():
            log.error("workspace %s has no adapter/ — skipping", workspace)
            all_outcomes[wsname] = "SKIPPED (no adapter)"; continue
        try:
            adapter = load_adapter(workspace / "adapter")
        except Exception as e:
            log.error("adapter load failed for %s: %s — skipping", workspace, e)
            all_outcomes[wsname] = f"SKIPPED (adapter err: {type(e).__name__})"; continue
        (workspace / "verify").mkdir(parents=True, exist_ok=True)
        log_path = workspace / "verify" / f"verifier_run_{datetime.now():%Y%m%d_%H%M%S}.log"
        try:
            with open(log_path, "w") as lf:
                verifier = Verifier(
                    provider=provider, model=model, reasoning_effort=effort, config=config,
                    workspace=str(workspace), data_path=data_path, adapter=adapter,
                    event_callback=make_event_callback(lf), nb_run_path=nb_run, python_exe=python_exe,
                    max_candidates=int(raw.get("verifier_max_candidates", 2)),
                    max_rounds=int(raw.get("verifier_max_rounds", 3)),
                    notebook_timeout=int(raw.get("verifier_notebook_timeout", 1800)),
                    steering=raw.get("verifier_steering", ""),
                    watchdog_interval=int(raw.get("verifier_watchdog_interval", 0)),
                    worker_model=raw.get("verifier_worker_model", ""),
                    critic_model=raw.get("verifier_critic_model", ""),
                    userrep_model=raw.get("verifier_userrep_model", ""),
                )
                all_outcomes[wsname] = verifier.run()
        except Exception as e:
            log.exception("verifier crashed on workspace %s", wsname)
            all_outcomes[wsname] = f"CRASHED ({type(e).__name__}: {e})"
        print(f"\n=== {wsname} OUTCOMES ===")
        oc = all_outcomes[wsname]
        for slug, outcome in (oc.items() if isinstance(oc, dict) else [("", oc)]):
            print(f"  {slug}: {outcome}")

    print("\n=== ALL WORKSPACES DONE ===")
    for ws, oc in all_outcomes.items():
        print(f"  {ws}: {oc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
