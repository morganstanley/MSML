"""Generate a cross-run markdown report from bench_summary.json artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alpha_lab.benchmarks.manifest import BENCHMARK_MANIFEST_NAME, RUN_MANIFEST_NAME
from alpha_lab.client import get_client

LOGGER = logging.getLogger(__name__)

SUMMARY_FILENAME = "bench_summary.json"


def _load_run(run_dir: Path) -> dict[str, Any]:
    """Load a run directory: manifest + per-workspace bench_summary.json.

    Args:
        run_dir: Path to the run output directory.

    Returns:
        Dict with id, dir, manifest, started_at, workspaces.

    Raises:
        NotADirectoryError: If run_dir does not exist.
        FileNotFoundError: If any workspace is missing bench_summary.json.
        ValueError: If no benchmark workspaces are found.
    """
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory does not exist: {run_dir}")

    manifest: dict[str, Any] = {}
    manifest_path = run_dir / RUN_MANIFEST_NAME
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    workspaces: dict[str, dict[str, Any]] = {}
    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir() or not (subdir / BENCHMARK_MANIFEST_NAME).exists():
            continue
        summary_path = subdir / SUMMARY_FILENAME
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing {SUMMARY_FILENAME} for workspace {subdir.name!r} "
                f"in run {run_dir}. Run summarize_workspace.py first."
            )
        try:
            workspaces[subdir.name] = json.loads(summary_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed {SUMMARY_FILENAME} in workspace {subdir.name!r}: {exc}"
            ) from exc

    if not workspaces:
        raise ValueError(f"No benchmark workspaces found in run directory: {run_dir}")

    return {
        "id": run_dir.name,
        "dir": run_dir,
        "manifest": manifest,
        "started_at": manifest.get("started_at"),
        "workspaces": workspaces,
    }


def _rank(values: list[float | None], direction: str) -> list[int]:
    """Rank values (1 = best, respecting direction). None values rank last.

    Args:
        values: Metric values, one per run. None = missing.
        direction: "maximize" or "minimize".

    Returns:
        Integer ranks in the same order as values.
    """
    reverse = direction != "minimize"
    present = sorted(
        ((v, i) for i, v in enumerate(values) if v is not None),
        key=lambda t: t[0],
        reverse=reverse,
    )
    default_rank = len(present) + 1 if present else len(values) + 1
    ranks = [default_rank] * len(values)
    for rank, (_, i) in enumerate(present, 1):
        ranks[i] = rank
    return ranks


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format a markdown table from headers and rows."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    def _fmt(cells: list[str]) -> str:
        return "| " + " | ".join(
            str(c).ljust(col_widths[i]) for i, c in enumerate(cells)
        ) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    return "\n".join([_fmt(headers), sep] + [_fmt(r) for r in rows])


def _generate_prose(
    runs: list[dict[str, Any]],
    workspace_names: list[str],
    model: str,
) -> dict[str, str]:
    """Generate one narrative paragraph per run using the LLM.

    Args:
        runs: Loaded run dicts (each with id, mean_rank, workspaces).
        workspace_names: Workspaces to include (intersection across runs).
        model: LLM model identifier.

    Returns:
        Dict mapping run_id -> narrative paragraph.
    """
    system = (
        "You are a research analyst. Given per-workspace benchmark summaries for a "
        "single run, write 1-2 paragraphs summarizing the run's performance and key "
        "findings. Be concise and factual. No markdown formatting."
    )

    client = get_client()
    results: dict[str, str] = {}

    for run in runs:
        workspace_lines = []
        for ws_name in workspace_names:
            summary = run["workspaces"].get(ws_name, {})
            narrative = summary.get("narrative", "")
            obs = summary.get("noteworthy_observations", [])
            obs_text = "; ".join(o.get("description", "") for o in obs)
            line = f"Workspace {ws_name}: {narrative}"
            if obs_text:
                line += f" Notable: {obs_text}"
            workspace_lines.append(line)

        user_content = (
            f"Run: {run['id']}\n"
            f"Mean rank: {run['mean_rank']:.2f}\n\n"
            + "\n\n".join(workspace_lines)
        )

        resp = client.responses.create(
            model=model,
            instructions=system,
            input=[{"role": "user", "content": user_content}],
            reasoning={"effort": "medium"},
            store=False,
        )

        text = ""
        for item in resp.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        text += content.text

        results[run["id"]] = text.strip()

    return results


def _render_report(
    *,
    runs: list[dict[str, Any]],
    workspace_names: list[str],
    metric_groups: list[dict[str, Any]],
    prose: dict[str, str],
    generated_at: str,
) -> str:
    """Render the markdown report string.

    Args:
        runs: Loaded run dicts.
        workspace_names: All workspace names included in the report.
        metric_groups: Per-metric ranking data. Each dict has keys:
            metric_name, metric_direction, workspace_names, ranks, mean_ranks.
        prose: run_id -> narrative paragraph from LLM.
        generated_at: ISO-style timestamp string.

    Returns:
        Complete markdown report string.
    """
    run_ids = [r["id"] for r in runs]
    lines: list[str] = []

    metric_summary = ", ".join(
        f"{g['metric_name']} ({g['metric_direction']})"
        for g in metric_groups
    )
    lines += [
        "# Benchmark Report",
        "",
        f"**Generated:** {generated_at}",
        f"**Metrics:** {metric_summary}",
        f"**Runs:** {len(runs)}",
        f"**Workspaces:** {len(workspace_names)}",
        "",
    ]

    # Resource consumption
    all_statuses: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for ws_name in workspace_names:
            ws = run["workspaces"].get(ws_name, {})
            for s in ws.get("experiment_counts", {}):
                if s not in seen:
                    all_statuses.append(s)
                    seen.add(s)
    all_statuses.sort()

    lines.append("## Resource Consumption")
    lines.append("")
    res_headers = ["Run", "Started At", "Total Experiments"] + all_statuses
    res_rows = []
    for run in runs:
        status_totals: dict[str, int] = {s: 0 for s in all_statuses}
        total = 0
        for ws_name in workspace_names:
            ws = run["workspaces"].get(ws_name, {})
            for s, cnt in ws.get("experiment_counts", {}).items():
                status_totals[s] = status_totals.get(s, 0) + cnt
                total += cnt
        res_rows.append(
            [run["id"], run["started_at"] or "", str(total)]
            + [str(status_totals[s]) for s in all_statuses]
        )
    lines.append(_format_table(res_headers, res_rows))
    lines.append("")

    # Per-metric rank tables
    for group in metric_groups:
        name = group["metric_name"]
        direction = group["metric_direction"]
        g_ws = group["workspace_names"]
        g_ranks = group["ranks"]
        g_mean = group["mean_ranks"]

        lines.append(f"## Relative Performance: {name} ({direction})")
        lines.append("")
        lines.append(f"Rank 1 = best ({direction}). Lower mean rank = better overall.")
        lines.append("")

        rank_headers = ["Workspace"] + run_ids
        rank_rows = [
            [ws] + [str(g_ranks[ws][j]) for j in range(len(runs))]
            for ws in g_ws
        ]
        rank_rows.append(
            ["**Mean Rank**"] + [f"{mr:.2f}" for mr in g_mean]
        )
        lines.append(_format_table(rank_headers, rank_rows))
        lines.append("")

    # Per-run narrative
    lines.append("## Per-Run Narrative")
    lines.append("")
    for run in runs:
        rid = run["id"]
        lines.append(f"### {rid}")
        lines.append("")
        if rid in prose:
            lines.append(prose[rid])
        else:
            for ws_name in workspace_names:
                summary = run["workspaces"].get(ws_name, {})
                narrative = summary.get("narrative", "")
                if narrative:
                    lines.append(f"**{ws_name}:** {narrative}")
                    lines.append("")
        lines.append("")

    return "\n".join(lines)


def create_report(
    run_dirs: list[Path],
    output: Path,
    *,
    metrics: list[str] | None = None,
    model: str | None = None,
    no_prose: bool = False,
) -> None:
    """Create a cross-run markdown report from bench_summary.json artifacts.

    Args:
        run_dirs: Paths to run output directories.
        output: Output markdown file path.
        metrics: Metric names to include. None includes all metrics.
        model: LLM model for prose generation. None skips prose.
        no_prose: Suppress prose generation even if model is set.

    Raises:
        NotADirectoryError: If a run directory does not exist.
        FileNotFoundError: If any workspace is missing bench_summary.json.
        ValueError: If no workspaces are common across all runs, or if
            requested metrics match no workspaces.
    """
    runs = [_load_run(d.resolve()) for d in run_dirs]

    workspace_sets = [frozenset(r["workspaces"]) for r in runs]
    common = frozenset.intersection(*workspace_sets)
    all_names = frozenset.union(*workspace_sets)
    omitted = all_names - common
    if omitted:
        for ws_name in sorted(omitted):
            absent_from = sorted(
                r["id"] for r in runs if ws_name not in r["workspaces"]
            )
            LOGGER.warning(
                "Workspace %r absent from run(s) %s -- omitted from report.",
                ws_name,
                ", ".join(absent_from),
            )

    if not common:
        raise ValueError("No workspaces are common across all runs. Cannot generate report.")

    intersection_names = sorted(common)

    # Discover (metric_name, metric_direction) per workspace from first run.
    metric_map: dict[str, str] = {}
    ws_metric: dict[str, str | None] = {}
    for ws_name in intersection_names:
        ws_summary = runs[0]["workspaces"][ws_name]
        name = ws_summary.get("metric_name")
        direction = ws_summary.get("metric_direction", "maximize")
        ws_metric[ws_name] = name
        if name and name not in metric_map:
            metric_map[name] = direction

    if metrics is not None:
        unknown = sorted(set(metrics) - set(metric_map))
        if unknown:
            raise ValueError(
                f"Requested metric(s) not found in any workspace: "
                f"{', '.join(unknown)}. "
                f"Available: {', '.join(sorted(metric_map))}"
            )
        metric_map = {k: v for k, v in metric_map.items() if k in metrics}

    if not metric_map:
        raise ValueError(
            "No metrics found in workspace summaries. "
            "Check that bench_summary.json files have metric_name set."
        )

    # Build per-metric groups.
    metric_groups: list[dict[str, Any]] = []
    included_workspaces: set[str] = set()
    for m_name in sorted(metric_map):
        m_direction = metric_map[m_name]
        group_ws = sorted(
            ws for ws in intersection_names if ws_metric[ws] == m_name
        )
        if not group_ws:
            continue

        ranks: dict[str, list[int]] = {}
        for ws_name in group_ws:
            values = [
                r["workspaces"][ws_name].get("best_metric_value") for r in runs
            ]
            ranks[ws_name] = _rank(values, m_direction)

        mean_ranks: list[float] = []
        for j in range(len(runs)):
            ws_ranks = [ranks[ws][j] for ws in group_ws]
            mr = sum(ws_ranks) / len(ws_ranks) if ws_ranks else float("nan")
            mean_ranks.append(mr)

        metric_groups.append({
            "metric_name": m_name,
            "metric_direction": m_direction,
            "workspace_names": group_ws,
            "ranks": ranks,
            "mean_ranks": mean_ranks,
        })
        included_workspaces.update(group_ws)

    workspace_names = (
        sorted(included_workspaces) if metrics is not None else intersection_names
    )

    # Global mean rank for prose generation.
    for j, run in enumerate(runs):
        all_ws_ranks: list[int] = []
        for group in metric_groups:
            for ws in group["workspace_names"]:
                all_ws_ranks.append(group["ranks"][ws][j])
        run["mean_rank"] = (
            sum(all_ws_ranks) / len(all_ws_ranks)
            if all_ws_ranks
            else float("nan")
        )

    prose: dict[str, str] = {}
    if model and not no_prose:
        LOGGER.info("Generating per-run narrative with %s ...", model)
        prose = _generate_prose(runs, workspace_names, model)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report_text = _render_report(
        runs=runs,
        workspace_names=workspace_names,
        metric_groups=metric_groups,
        prose=prose,
        generated_at=generated_at,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report_text)
    LOGGER.info("Report written to %s", output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a cross-run benchmark report."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        type=Path,
        metavar="RUN_DIR",
        help="One or more run output directories containing bench_summary.json artifacts.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output markdown file path.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        metavar="METRIC",
        help="Metric name(s) to include in rank tables. Default: all metrics.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model for per-run prose generation. Omit to skip prose entirely.",
    )
    parser.add_argument(
        "--no-prose",
        action="store_true",
        dest="no_prose",
        help="Skip per-run narrative generation even if --model is set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    try:
        create_report(
            [p.resolve() for p in args.runs],
            args.output.resolve(),
            metrics=args.metrics,
            model=args.model,
            no_prose=args.no_prose,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
