"""Snapshot of every run under a root: state, board, freshness, best-so-far.

One pass, no polling loop — rerun it when you want a fresh view (or wrap it
in your terminal's own repeat command). Read-only over the same layouts the
indexer understands, so anything `index` would register shows up here while
it is still running.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import time
from pathlib import Path

from alpha_lab.benchmarks.runcmp.corpus import build_registry


def _best_metric(db_path: str | None) -> str:
    """The most common numeric key in results_json, with its extremes."""
    if not db_path or not Path(db_path).is_file():
        return "—"
    try:
        c = sqlite3.connect(db_path)
        vals: dict[str, list[float]] = collections.defaultdict(list)
        for (rj,) in c.execute("select results_json from experiments "
                               "where results_json is not null"):
            try:
                r = json.loads(rj)
            except json.JSONDecodeError:
                continue
            for k, v in (r or {}).items():
                if isinstance(v, (int, float)):
                    vals[k].append(float(v))
    except sqlite3.Error:
        return "—"
    if not vals:
        return "—"
    key = max(vals, key=lambda k: len(vals[k]))
    xs = vals[key]
    return f"{key} max={max(xs):.4g} min={min(xs):.4g} (n={len(xs)})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="runcmp watch",
        description="one snapshot of every run under a root (read-only)")
    ap.add_argument("--root", required=True, type=Path, action="append")
    args = ap.parse_args(argv)

    records = []
    skipped: list[Path] = []
    for root in args.root:
        records.extend(build_registry(root, skipped_archived=skipped))
    if skipped:
        print(f"archived attempts ignored: {len(skipped)}")
    if not records:
        print("no runs found")
        return 1
    now = time.time()
    for r in sorted(records, key=lambda x: x.label):
        age = "—"
        if r.run_log_path and Path(r.run_log_path).is_file():
            mins = (now - Path(r.run_log_path).stat().st_mtime) / 60
            age = f"{mins:.0f}m ago"
        state = r.run_state.upper() if r.run_state == "in_flight" \
            else r.run_state
        counts = ", ".join(f"{k}={v}" for k, v in
                           sorted((r.status_counts or {}).items())) or "no rows"
        print(f"{r.label}\n"
              f"    state={state}  log={age}  board: {counts}\n"
              f"    best: {_best_metric(r.db_path)}")
    live = sum(1 for r in records if r.run_state == "in_flight")
    print(f"\n{len(records)} run(s), {live} still in flight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
