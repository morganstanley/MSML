#!/usr/bin/env python
"""Alpha-Lab Benchmark Runner.

Runs the LLM Speedrun benchmark N times against a frozen Phase 2 harness
and adapter, collects results, and optionally updates the LEADERBOARD.md.

Usage:
    python benchmarking/run_benchmark.py                    # interactive
    python benchmarking/run_benchmark.py --name "my-method" # non-interactive
    python benchmarking/run_benchmark.py --collect-only      # just re-collect results from existing runs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
REFERENCE_DIR = BENCHMARK_DIR / "reference"
LEADERBOARD_PATH = BENCHMARK_DIR / "LEADERBOARD.md"
CONFIG_PATH = BENCHMARK_DIR / "benchmark_config.json"
RUNS_DIR = BENCHMARK_DIR / "runs"

# Use the repo's own run.py
RUN_SCRIPT = REPO_ROOT / "run.py"

# Python executable — auto-detect from config or use current interpreter
_PYTHON = None


def get_python() -> str:
    global _PYTHON
    if _PYTHON is not None:
        return _PYTHON

    # Check config for explicit python_executable
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    explicit = (cfg.get("pipeline", {}).get("phase3", {})
                .get("python_executable"))
    if explicit and os.path.isfile(explicit):
        _PYTHON = explicit
        return _PYTHON

    _PYTHON = sys.executable
    return _PYTHON


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_workspace(workspace: Path) -> None:
    """Create a fresh workspace seeded with the frozen reference artifacts."""
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    # Copy adapter (Phase 0 will detect manifest.json and skip customization)
    shutil.copytree(REFERENCE_DIR / "adapter", workspace / "adapter")

    # Copy harness (Phase 2 framework — frozen)
    shutil.copytree(REFERENCE_DIR / "harness", workspace / "harness")

    # Create empty dirs the pipeline expects
    (workspace / "harness" / "results").mkdir(exist_ok=True)
    (workspace / "harness" / "cache").mkdir(exist_ok=True)


def _run_pipeline(workspace: Path, run_log: Path) -> int:
    """Run the full pipeline for one benchmark run. Returns exit code."""
    python = get_python()
    cmd = [
        python, str(RUN_SCRIPT),
        "--config", str(CONFIG_PATH),
        "--workspace", str(workspace),
    ]
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log:     {run_log}")

    with open(run_log, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
        proc.wait()
    return proc.returncode


def _extract_results(workspace: Path) -> dict | None:
    """Extract benchmark results from a completed run's experiments.db."""
    db_path = workspace / "experiments.db"
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Total experiments
    total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    # Experiments that reached done/analyzed
    completed = conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE status IN ('done', 'analyzed')"
    ).fetchone()[0]

    # Extract best val_bpb (the adapter uses best_val_bpb as the key, but
    # also check val_bpb for robustness)
    rows = conn.execute(
        "SELECT results_json FROM experiments ORDER BY created_at ASC"
    ).fetchall()
    conn.close()

    best_bpb = float("inf")
    valid_count = 0
    all_bpbs = []

    for row in rows:
        rj = row["results_json"]
        if not rj:
            continue
        try:
            data = json.loads(rj)
        except (json.JSONDecodeError, TypeError):
            continue

        # Try both keys
        bpb = data.get("best_val_bpb") or data.get("val_bpb")
        if bpb is not None and isinstance(bpb, (int, float)) and bpb < 50:
            valid_count += 1
            all_bpbs.append(bpb)
            if bpb < best_bpb:
                best_bpb = bpb

    # Wall clock from first to last experiment
    conn2 = sqlite3.connect(str(db_path))
    times = conn2.execute(
        "SELECT MIN(created_at) as first_t, MAX(finished_at) as last_t "
        "FROM experiments"
    ).fetchone()
    conn2.close()

    wall_clock_hours = None
    if times[0] and times[1]:
        try:
            t0 = datetime.fromisoformat(times[0])
            t1 = datetime.fromisoformat(times[1])
            wall_clock_hours = (t1 - t0).total_seconds() / 3600
        except (ValueError, TypeError):
            pass

    if best_bpb == float("inf"):
        best_bpb = None

    return {
        "total_experiments": total,
        "completed_experiments": completed,
        "valid_results": valid_count,
        "success_rate": round(valid_count / total * 100, 1) if total > 0 else 0,
        "best_val_bpb": round(best_bpb, 4) if best_bpb is not None else None,
        "all_bpbs": sorted(all_bpbs),
        "wall_clock_hours": round(wall_clock_hours, 2) if wall_clock_hours else None,
    }


def collect_run_results(method_dir: Path) -> list[dict]:
    """Collect results from all run_* subdirectories of a method."""
    results = []
    for run_dir in sorted(method_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            r = _extract_results(run_dir)
            if r is not None:
                r["run_name"] = run_dir.name
                results.append(r)
    return results


def summarize_results(results: list[dict]) -> dict:
    """Compute aggregate stats across multiple runs."""
    bpbs = [r["best_val_bpb"] for r in results if r["best_val_bpb"] is not None]
    success_rates = [r["success_rate"] for r in results]
    wall_clocks = [r["wall_clock_hours"] for r in results
                   if r["wall_clock_hours"] is not None]

    summary = {
        "n_runs": len(results),
        "n_valid_runs": len(bpbs),
        "runs": results,
    }

    if bpbs:
        mean_bpb = sum(bpbs) / len(bpbs)
        summary["mean_val_bpb"] = round(mean_bpb, 4)
        summary["best_val_bpb"] = round(min(bpbs), 4)
        if len(bpbs) > 1:
            variance = sum((x - mean_bpb) ** 2 for x in bpbs) / (len(bpbs) - 1)
            summary["std_val_bpb"] = round(variance ** 0.5, 4)
        else:
            summary["std_val_bpb"] = 0.0

    if success_rates:
        summary["mean_success_rate"] = round(
            sum(success_rates) / len(success_rates), 1
        )

    if wall_clocks:
        summary["mean_wall_clock_hours"] = round(
            sum(wall_clocks) / len(wall_clocks), 2
        )

    return summary


# ---------------------------------------------------------------------------
# Leaderboard management
# ---------------------------------------------------------------------------

# Markers that delimit the results table in LEADERBOARD.md
TABLE_START = "<!-- RESULTS_TABLE_START -->"
TABLE_END = "<!-- RESULTS_TABLE_END -->"


def _parse_leaderboard_table(content: str) -> list[dict]:
    """Parse existing leaderboard entries from markdown table."""
    entries = []
    in_table = False
    for line in content.splitlines():
        if TABLE_START in line:
            in_table = True
            continue
        if TABLE_END in line:
            break
        if not in_table:
            continue
        # Skip header and separator rows
        if line.startswith("| Rank") or line.startswith("|---") or not line.strip():
            continue
        if not line.startswith("|"):
            continue

        cols = [c.strip() for c in line.split("|")]
        # cols[0] is empty (before first |), cols[-1] is empty (after last |)
        cols = [c for c in cols if c]

        if len(cols) >= 7:
            # Try to parse mean_bpb
            try:
                bpb_str = cols[2].split("±")[0].strip().replace("**", "")
                mean_bpb = float(bpb_str) if bpb_str != "N/A" else None
            except (ValueError, IndexError):
                mean_bpb = None

            entries.append({
                "rank": cols[0].strip(),
                "method": cols[1].strip().replace("**", ""),
                "mean_bpb_str": cols[2].strip(),
                "mean_bpb": mean_bpb,
                "best_bpb_str": cols[3].strip(),
                "success_rate_str": cols[4].strip(),
                "n_runs_str": cols[5].strip(),
                "date_str": cols[6].strip(),
            })

    return entries


def _format_table_row(
    rank: int,
    method: str,
    summary: dict,
    date_str: str,
    bold: bool = False,
) -> str:
    """Format a single leaderboard row."""
    mean_bpb = summary.get("mean_val_bpb")
    std_bpb = summary.get("std_val_bpb")
    best_bpb = summary.get("best_val_bpb")
    sr = summary.get("mean_success_rate")
    n_runs = summary.get("n_runs", 0)

    if mean_bpb is not None and std_bpb is not None:
        bpb_str = f"{mean_bpb:.4f} ± {std_bpb:.4f}"
    elif mean_bpb is not None:
        bpb_str = f"{mean_bpb:.4f}"
    else:
        bpb_str = "N/A"

    best_str = f"{best_bpb:.4f}" if best_bpb is not None else "N/A"
    sr_str = f"{sr:.0f}%" if sr is not None else "N/A"
    runs_str = f"{n_runs}"

    if bold:
        method = f"**{method}**"
        bpb_str = f"**{bpb_str}**"

    return f"| {rank} | {method} | {bpb_str} | {best_str} | {sr_str} | {runs_str} | {date_str} |"


def update_leaderboard(method: str, summary: dict) -> None:
    """Insert or update a method's entry in LEADERBOARD.md, ranked by mean_val_bpb."""
    content = LEADERBOARD_PATH.read_text()
    entries = _parse_leaderboard_table(content)

    date_str = datetime.now().strftime("%Y-%m-%d")
    new_mean = summary.get("mean_val_bpb")

    # Remove existing entry for this method (case-insensitive)
    entries = [e for e in entries if e["method"].lower() != method.lower()]

    # Build combined list with the new entry
    new_entry = {
        "method": method,
        "mean_bpb": new_mean,
        "summary": summary,
        "date_str": date_str,
    }

    all_entries = entries + [new_entry]
    # Sort: valid bpb first (ascending), then N/A entries
    all_entries.sort(
        key=lambda e: (
            0 if e.get("mean_bpb") or e.get("summary", {}).get("mean_val_bpb") else 1,
            e.get("mean_bpb") or e.get("summary", {}).get("mean_val_bpb") or 999,
        )
    )

    # Rebuild table
    table_lines = [
        "| Rank | Method | Mean val_bpb | Best val_bpb | Success Rate | Runs | Date |",
        "|------|--------|-------------|-------------|-------------|------|------|",
    ]

    for i, entry in enumerate(all_entries):
        rank = i + 1
        is_new = "summary" in entry and entry["method"] == method

        if is_new:
            row = _format_table_row(
                rank, method, entry["summary"], date_str, bold=True,
            )
        else:
            # Re-emit existing row with updated rank
            row = (
                f"| {rank} | {entry['method']} | {entry['mean_bpb_str']} "
                f"| {entry['best_bpb_str']} | {entry['success_rate_str']} "
                f"| {entry['n_runs_str']} | {entry['date_str']} |"
            )

        table_lines.append(row)

    new_table = "\n".join(table_lines)

    # Replace table in document
    pattern = re.compile(
        re.escape(TABLE_START) + r".*?" + re.escape(TABLE_END),
        re.DOTALL,
    )
    new_content = pattern.sub(
        TABLE_START + "\n" + new_table + "\n" + TABLE_END,
        content,
    )

    LEADERBOARD_PATH.write_text(new_content)
    print(f"\n  Updated {LEADERBOARD_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Alpha-Lab LLM Speedrun benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python benchmarking/run_benchmark.py --name "context-memory-v1"
              python benchmarking/run_benchmark.py --name "baseline" --runs 5
              python benchmarking/run_benchmark.py --collect-only --name "context-memory-v1"
        """),
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for this method/branch (used in leaderboard). "
             "Prompted interactively if not provided.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3)",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip running; just collect results from existing runs and "
             "optionally update the leaderboard.",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Don't prompt to update the leaderboard.",
    )
    args = parser.parse_args()

    # --- Method name ---
    method = args.name
    if method is None:
        method = input("Enter a name for this method/branch: ").strip()
        if not method:
            print("Error: method name is required.")
            sys.exit(1)

    # Sanitize for directory name
    dir_name = re.sub(r"[^\w\-.]", "_", method)
    method_dir = RUNS_DIR / dir_name

    print(f"\n{'='*60}")
    print(f"  Alpha-Lab Benchmark: LLM Speedrun")
    print(f"  Method:     {method}")
    print(f"  Runs:       {args.runs}")
    print(f"  Workspace:  {method_dir}")
    print(f"{'='*60}\n")

    # --- Validate reference exists ---
    if not (REFERENCE_DIR / "adapter" / "manifest.json").exists():
        print("Error: reference adapter not found at "
              f"{REFERENCE_DIR / 'adapter' / 'manifest.json'}")
        print("The benchmarking/reference/ directory must contain the frozen "
              "Phase 2 harness and adapter.")
        sys.exit(1)

    if not (REFERENCE_DIR / "harness" / "runner.py").exists():
        print("Error: reference harness not found at "
              f"{REFERENCE_DIR / 'harness' / 'runner.py'}")
        sys.exit(1)

    # --- Run or collect ---
    if not args.collect_only:
        method_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, args.runs + 1):
            run_dir = method_dir / f"run_{i}"
            run_log = method_dir / f"run_{i}.log"

            if run_dir.exists() and (run_dir / "experiments.db").exists():
                print(f"\n--- Run {i}/{args.runs}: already exists, skipping ---")
                continue

            print(f"\n--- Run {i}/{args.runs} ---")
            print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            _setup_workspace(run_dir)
            exit_code = _run_pipeline(run_dir, run_log)

            if exit_code != 0:
                print(f"  WARNING: run {i} exited with code {exit_code}")
            else:
                print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Collect results ---
    print(f"\n{'='*60}")
    print("  Collecting results...")
    print(f"{'='*60}\n")

    results = collect_run_results(method_dir)
    if not results:
        print("No completed runs found. Check logs in:")
        print(f"  {method_dir}")
        sys.exit(1)

    summary = summarize_results(results)

    # Save raw results JSON
    results_json_path = method_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump({"method": method, "summary": summary}, f, indent=2)

    # --- Display results ---
    print(f"  Method:        {method}")
    print(f"  Runs:          {summary['n_runs']} total, "
          f"{summary['n_valid_runs']} with valid results")

    if summary.get("mean_val_bpb") is not None:
        bpb_str = f"{summary['mean_val_bpb']:.4f}"
        if summary.get("std_val_bpb"):
            bpb_str += f" ± {summary['std_val_bpb']:.4f}"
        print(f"  Mean val_bpb:  {bpb_str}")
        print(f"  Best val_bpb:  {summary['best_val_bpb']:.4f}")
    else:
        print("  Mean val_bpb:  N/A (no valid results)")

    if summary.get("mean_success_rate") is not None:
        print(f"  Success rate:  {summary['mean_success_rate']:.0f}%")
    if summary.get("mean_wall_clock_hours") is not None:
        print(f"  Wall clock:    {summary['mean_wall_clock_hours']:.1f}h (mean)")

    print(f"\n  Per-run breakdown:")
    for r in results:
        bpb = f"{r['best_val_bpb']:.4f}" if r["best_val_bpb"] else "N/A"
        print(f"    {r['run_name']}: best_val_bpb={bpb}, "
              f"success={r['success_rate']}%, "
              f"experiments={r['valid_results']}/{r['total_experiments']}")

    print(f"\n  Raw results saved to: {results_json_path}")

    # --- Leaderboard update ---
    if args.no_update:
        return

    if not LEADERBOARD_PATH.exists():
        print("\n  LEADERBOARD.md not found — skipping update.")
        return

    # Show where this would land
    content = LEADERBOARD_PATH.read_text()
    entries = _parse_leaderboard_table(content)
    new_mean = summary.get("mean_val_bpb")

    if new_mean is not None:
        rank = 1
        for e in entries:
            if e["mean_bpb"] is not None and e["mean_bpb"] < new_mean:
                rank += 1
        print(f"\n  Leaderboard placement: #{rank} "
              f"(out of {len(entries)} existing entries)")
    else:
        print("\n  No valid results to add to leaderboard.")
        return

    answer = input("\n  Update LEADERBOARD.md? [y/N] ").strip().lower()
    if answer in ("y", "yes"):
        update_leaderboard(method, summary)
        print("  Done! Review the changes and commit when ready.")
    else:
        print("  Skipped. You can update later with --collect-only.")


if __name__ == "__main__":
    main()
