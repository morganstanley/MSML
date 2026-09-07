"""Validate an evidence-contract run directory BEFORE it reaches the corpus.

For people bringing their own harness: point this at one run directory laid
out per the evidence contract and it says exactly what parses, what is
missing, and which measurement families will light up downstream — at the
moment the submission is being built, not at index time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# optional referee inputs, per experiment: any one of these makes the run
# independently re-scorable by the matching referee kind
REFEREE_INPUTS = ["referee_predictions.csv", "curves.json",
                  "training_curves.json", "metrics_detailed.json",
                  "kernel_report.json"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="runcmp validate-submission",
        description="check an evidence-contract run directory")
    ap.add_argument("--run", required=True, type=Path,
                    help="one run directory (holds experiments/<name>/"
                         "results/metrics.json)")
    args = ap.parse_args(argv)
    run: Path = args.run
    exp_root = run / "experiments"
    if not exp_root.is_dir():
        print(f"INVALID: {run} has no experiments/ directory — the contract "
              "layout is experiments/<name>/results/metrics.json")
        return 1

    ok = 0
    referee_ready = 0
    problems: list[str] = []
    for exp in sorted(p for p in exp_root.iterdir() if p.is_dir()):
        mpath = exp / "results" / "metrics.json"
        if not mpath.is_file():
            problems.append(f"{exp.name}: no results/metrics.json")
            continue
        try:
            metrics = json.loads(mpath.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"{exp.name}: metrics.json does not parse "
                            f"({exc})")
            continue
        numeric = {k: v for k, v in (metrics or {}).items()
                   if isinstance(v, (int, float))}
        if not numeric:
            problems.append(f"{exp.name}: metrics.json has no numeric "
                            "fields — nothing to score")
            continue
        ok += 1
        extras = [n for n in REFEREE_INPUTS
                  if (exp / "results" / n).is_file()]
        referee_ready += bool(extras)
        print(f"  {exp.name}: {len(numeric)} numeric metric(s) "
              f"({', '.join(sorted(numeric)[:4])}"
              + ("…" if len(numeric) > 4 else "") + ")"
              + (f"; referee inputs: {', '.join(extras)}" if extras else ""))

    for p in problems:
        print(f"  PROBLEM: {p}")

    # run-level extras decide which measurement families light up
    has_events = (run / "events.jsonl").is_file()
    has_logs = any((run / "logs").glob("*.jsonl")) \
        if (run / "logs").is_dir() else False
    print(f"\nsummary for {run.name}:")
    print(f"  scorable experiments: {ok}"
          + (f" ({len(problems)} problem(s))" if problems else ""))
    print("  independent re-scoring (referee): "
          + (f"YES — {referee_ready} experiment(s) carry referee inputs"
             if referee_ready else
             "NO — add one referee input per experiment "
             f"({', '.join(REFEREE_INPUTS[:3])}, …)"))
    print("  timeline metrics: " + ("YES (events.jsonl)" if has_events
                                    else "NO (no events.jsonl)"))
    print("  token/cost ledger: " + ("YES (logs/*.jsonl transcripts)"
                                     if has_logs else
                                     "NO (no logs/*.jsonl transcripts — "
                                     "absence is recorded as absence, "
                                     "never as zero)"))
    print("validate-submission: " + ("OK" if ok else "INVALID"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
