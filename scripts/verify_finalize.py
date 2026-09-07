"""Generate the verifier's final per-workspace reports for ALREADY-COMPLETED workspace(s),
WITHOUT re-running verification.

Produces, per workspace, the two deliverables the verifier now emits at end of `run()`:
  - verify/feedback_to_system.md  (machine-to-machine: Conductor / strategist / workers)
  - verify/report_for_human.md    (plain-language, first-principles: the user)

Outcomes are reconstructed from each candidate's verify/<slug>/STATE.json (same as run()'s
resume path), so this only synthesizes — it never re-verifies.

Usage (from repo root, so .token_cache.json is found):
  PYTHONPATH=src <p312-python> scripts/verify_finalize.py \
      --config data/verify_etfflow_o48_config.json \
      --workspace workspace_etfflow_grok,workspace_etfflow_g55,workspace_etfflow_o48
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from alpha_lab.adapter_loader import load_adapter
from alpha_lab.client import get_provider
from alpha_lab.config import load_config
from alpha_lab.verifier import Verifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("verify_finalize")


def _cb(ev):
    try:
        if isinstance(ev, str):
            line = ev
        else:
            line = " ".join(str(getattr(ev, a, "")) for a in ("type", "name", "status", "detail")
                            if getattr(ev, a, None))
        print(("[finalize] " + line)[:300], flush=True)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workspace", default=None,
                    help="comma-separated workspace dir(s); overrides config verify_workspace")
    args = ap.parse_args()

    raw = json.loads(Path(args.config).read_text())
    config = load_config(Path(args.config))
    repo_root = Path(__file__).resolve().parent.parent
    ws_names = [w.strip() for w in (args.workspace or raw["verify_workspace"]).split(",") if w.strip()]

    provider_name = raw.get("verifier_provider", config.provider)
    model = raw.get("verifier_model", config.model)
    effort = raw.get("verifier_reasoning_effort", config.reasoning_effort)
    log.info("building provider=%s model=%s effort=%s", provider_name, model, effort)
    provider = get_provider(provider_name, api_key=None)
    nb_run = str(repo_root / "scripts" / "nb_run.py")

    for wsname in ws_names:
        workspace = (repo_root / wsname).resolve() if not Path(wsname).is_absolute() else Path(wsname)
        log.info("=== FINALIZING WORKSPACE: %s ===", wsname)
        if not (workspace / "verify").exists():
            log.error("workspace %s has no verify/ — skipping", workspace)
            continue
        try:
            adapter = load_adapter(workspace / "adapter")
        except Exception as e:
            log.error("adapter load failed for %s: %s — skipping", workspace, e)
            continue
        v = Verifier(
            provider=provider, model=model, reasoning_effort=effort, config=config,
            workspace=str(workspace), data_path=config.data_path, adapter=adapter,
            event_callback=_cb, nb_run_path=nb_run, python_exe=sys.executable,
            max_candidates=int(raw.get("verifier_max_candidates", 4)),
            max_rounds=int(raw.get("verifier_max_rounds", 2)),
            notebook_timeout=int(raw.get("verifier_notebook_timeout", 1800)),
            steering=raw.get("verifier_steering", ""),
            watchdog_interval=int(raw.get("verifier_watchdog_interval", 0)),
            worker_model=raw.get("verifier_worker_model", ""),
            critic_model=raw.get("verifier_critic_model", ""),
            userrep_model=raw.get("verifier_userrep_model", ""),
        )
        # reconstruct outcomes from candidate STATE.json (same as run()'s resume path)
        results: dict[str, str] = {}
        for d in sorted(v.verify_dir.iterdir()):
            if d.is_dir():
                st = v._load_state(d)
                if st.get("outcome"):
                    results[d.name] = st["outcome"]
        log.info("%s: reconstructed %d outcome(s): %s", wsname, len(results), results)
        try:
            v._write_final_reports(results)
            log.info("=== %s: wrote verify/feedback_to_system.md + verify/report_for_human.md ===", wsname)
        except Exception as e:
            log.exception("%s: final report generation failed: %s", wsname, e)

    print("\n=== FINALIZE DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
