"""Summarize a completed Alpha Lab workspace into bench_summary.json."""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from alpha_lab.client import get_client

LOGGER = logging.getLogger(__name__)

SUMMARY_FILENAME = "bench_summary.json"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_TOP_K = 5
_MAX_ARTIFACT_CHARS = 3000
_MAX_DEBRIEF_CHARS = 1500


def _read_text(path: Path, max_chars: int = _MAX_ARTIFACT_CHARS) -> str | None:
    """Read a text file, truncated to max_chars. Returns None if missing."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n...[truncated at {max_chars} chars]"
        return text
    except (FileNotFoundError, OSError):
        return None


def _load_adapter_manifest(workspace: Path) -> dict[str, Any]:
    """Load adapter/manifest.json. Returns {} if missing or malformed."""
    path = workspace / "adapter" / "manifest.json"
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def _latest_milestone(workspace: Path) -> str | None:
    """Return content of the latest output/04_milestone_*.md, or None."""
    output_dir = workspace / "output"
    if not output_dir.is_dir():
        return None
    milestones = sorted(output_dir.glob("04_milestone_*.md"))
    return _read_text(milestones[-1]) if milestones else None


def _load_db_info(
    workspace: Path,
    extract_key: str | None,
    metric_direction: str,
    top_k: int,
) -> dict[str, Any]:
    """Query experiments.db for counts, best metric, runtime, and top-K candidates.

    Returns a dict with keys: experiment_counts, best_metric_value,
    best_experiment, runtime_seconds, top_k_raw (list of {name, metric,
    debrief_path}).
    """
    db_path = workspace / "experiments.db"
    if not db_path.exists():
        return {
            "experiment_counts": {},
            "best_metric_value": None,
            "best_experiment": None,
            "runtime_seconds": None,
            "top_k_raw": [],
        }

    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        status_rows = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM experiments GROUP BY status"
        ).fetchall()
        counts = {r["status"]: r["cnt"] for r in status_rows}

        time_row = conn.execute(
            "SELECT MIN(created_at) AS first, MAX(finished_at) AS last FROM experiments"
        ).fetchone()
        runtime_seconds: int | None = None
        if time_row and time_row["first"] and time_row["last"]:
            runtime_seconds = max(0, int(time_row["last"] - time_row["first"]))

        all_rows = conn.execute(
            "SELECT name, results_json, debrief_path FROM experiments ORDER BY created_at ASC"
        ).fetchall()
    finally:
        conn.close()

    candidates: list[tuple[float, str, str | None]] = []
    for row in all_rows:
        if not row["results_json"] or not extract_key:
            continue
        try:
            results = json.loads(row["results_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(results, dict) or results.get("smoke_test"):
            continue
        value = results.get(extract_key)
        if not isinstance(value, (int, float)):
            continue
        candidates.append((float(value), row["name"], row["debrief_path"]))

    reverse = metric_direction != "minimize"
    candidates.sort(key=lambda t: t[0], reverse=reverse)

    top_k_raw = [
        {"name": name, "metric": val, "debrief_path": dp}
        for val, name, dp in candidates[:top_k]
    ]
    best_value = candidates[0][0] if candidates else None
    best_name = candidates[0][1] if candidates else None

    return {
        "experiment_counts": counts,
        "best_metric_value": best_value,
        "best_experiment": best_name,
        "runtime_seconds": runtime_seconds,
        "top_k_raw": top_k_raw,
    }


def _resolve_debrief_path(debrief_path: str | None, workspace: Path) -> Path | None:
    """Resolve a debrief path (absolute or relative to workspace)."""
    if not debrief_path:
        return None
    p = Path(debrief_path)
    if not p.is_absolute():
        p = workspace / p
    return p


def _build_prompt(
    workspace: Path,
    adapter_manifest: dict[str, Any],
    top_k_raw: list[dict[str, Any]],
) -> str:
    """Assemble the summarization prompt from workspace artifacts."""
    parts: list[str] = []

    def _section(title: str, content: str | None) -> None:
        if content:
            parts.append(f"## {title}\n\n{content}")

    _section("Adapter manifest", json.dumps(adapter_manifest, indent=2))
    _section("Output index", _read_text(workspace / "output" / "index.md"))
    _section("Latest milestone report", _latest_milestone(workspace))
    _section("Reports overview", _read_text(workspace / "reports" / "overview.md"))
    _section("Playbook", _read_text(workspace / "playbook.md"))

    for entry in top_k_raw:
        dp = _resolve_debrief_path(entry.get("debrief_path"), workspace)
        if dp:
            debrief = _read_text(dp, max_chars=_MAX_DEBRIEF_CHARS)
            if debrief:
                _section(
                    f"Debrief: {entry['name']} (metric={entry['metric']})", debrief
                )

    return "\n\n---\n\n".join(parts) if parts else "(no workspace artifacts found)"


def _strip_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _call_llm(prompt: str, model: str) -> dict[str, Any]:
    """Call the LLM and return parsed JSON output.

    Args:
        prompt: User-facing prompt assembled from workspace artifacts.
        model: Model identifier.

    Returns:
        Parsed JSON dict from the model response.

    Raises:
        json.JSONDecodeError: If the model returns invalid JSON.
    """
    system = (
        "You are a research summarizer. Given workspace artifacts from an Alpha Lab "
        "research run, produce a JSON object with exactly these fields:\n"
        "- errors_summary (str | null): clusters of common failure modes, or null if none\n"
        "- noteworthy_observations (list): each item is "
        "{category, workspace_ids, description}. "
        "category is one of: rank_outlier, common_failure, variance_hotspot. "
        "workspace_ids is [] for a single-workspace summary.\n"
        "- narrative (str): one concise paragraph summarizing this run\n"
        "- top_k_config_summaries (list of str): one config summary sentence per "
        "top-K experiment listed, in order\n"
        "Respond ONLY with valid JSON. No markdown fences, no extra text."
    )

    client = get_client()
    resp = client.responses.create(
        model=model,
        instructions=system,
        input=[{"role": "user", "content": prompt}],
        reasoning={"effort": "medium"},
        store=False,
    )

    text = ""
    for item in resp.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    text += content.text

    return json.loads(_strip_fences(text))


def summarize(
    workspace: Path,
    *,
    model: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
    overwrite: bool = False,
) -> Path:
    """Summarize a workspace and write bench_summary.json.

    Args:
        workspace: Path to the workspace directory.
        model: LLM model for summarization.
        top_k: Number of top experiments to include.
        overwrite: Replace existing bench_summary.json if present.

    Returns:
        Path to the written bench_summary.json.

    Raises:
        FileExistsError: If bench_summary.json exists and overwrite is False.
        json.JSONDecodeError: If the LLM returns invalid JSON.
    """
    output_path = workspace / SUMMARY_FILENAME
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it."
        )

    adapter_manifest = _load_adapter_manifest(workspace)
    metric_info = adapter_manifest.get("metric", {})
    metric_name: str | None = metric_info.get("primary_metric")
    metric_direction: str = metric_info.get("direction", "maximize")
    extract_key: str | None = metric_info.get("extract_key") or metric_name

    db_info = _load_db_info(workspace, extract_key, metric_direction, top_k)
    top_k_raw: list[dict[str, Any]] = db_info.pop("top_k_raw")

    prompt = _build_prompt(workspace, adapter_manifest, top_k_raw)

    LOGGER.info("Summarizing %s with %s ...", workspace.name, model)
    llm_out = _call_llm(prompt, model)

    config_summaries: list[str] = llm_out.get("top_k_config_summaries", [])
    top_k_experiments = []
    for i, entry in enumerate(top_k_raw):
        dp = _resolve_debrief_path(entry.get("debrief_path"), workspace)
        debrief_excerpt = ""
        if dp:
            raw = _read_text(dp, max_chars=500)
            debrief_excerpt = raw or ""
        top_k_experiments.append({
            "name": entry["name"],
            "metric": entry["metric"],
            "config_summary": config_summaries[i] if i < len(config_summaries) else "",
            "debrief_excerpt": debrief_excerpt,
        })

    summary: dict[str, Any] = {
        "metric_name": metric_name,
        "metric_direction": metric_direction,
        "best_metric_value": db_info["best_metric_value"],
        "best_experiment": db_info["best_experiment"],
        "experiment_counts": db_info["experiment_counts"],
        "top_k_experiments": top_k_experiments,
        "runtime_seconds": db_info["runtime_seconds"],
        "errors_summary": llm_out.get("errors_summary"),
        "noteworthy_observations": llm_out.get("noteworthy_observations", []),
        "narrative": llm_out.get("narrative", ""),
    }

    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Wrote %s", output_path)
    return output_path


def _resolve_workspaces(patterns: list[str]) -> list[Path]:
    """Expand a list of paths/globs into resolved workspace directories.

    Args:
        patterns: Mix of literal paths and glob patterns (absolute or relative).

    Returns:
        Deduplicated sorted list of resolved directory paths.

    Raises:
        SystemExit: If any pattern matches no directories.
    """
    seen: set[Path] = set()
    result: list[Path] = []
    for pattern in patterns:
        if Path(pattern).is_absolute():
            matches = [
                Path(p).resolve() for p in glob.glob(pattern) if Path(p).is_dir()
            ]
        else:
            matches = [p.resolve() for p in Path(".").glob(pattern) if p.is_dir()]
        if not matches:
            literal = Path(pattern).resolve()
            if literal.is_dir():
                matches = [literal]
            else:
                LOGGER.error("No workspace directories matched: %s", pattern)
                raise SystemExit(2)
        for p in sorted(matches):
            if p not in seen:
                seen.add(p)
                result.append(p)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed Alpha Lab workspaces into bench_summary.json."
    )
    parser.add_argument(
        "--workspaces",
        nargs="+",
        required=True,
        metavar="WORKSPACE",
        help="Workspace directories to summarize. Glob patterns are supported.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model for summarization (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        dest="top_k",
        help=f"Number of top experiments to include (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        dest="num_workers",
        help="Number of parallel workers (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing bench_summary.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    if args.num_workers < 1:
        LOGGER.error("--num-workers must be at least 1")
        return 2

    try:
        workspaces = _resolve_workspaces(args.workspaces)
    except SystemExit as e:
        return int(e.code)

    kwargs = dict(model=args.model, top_k=args.top_k, overwrite=args.overwrite)

    def _run(workspace: Path) -> tuple[Path, Exception | None]:
        try:
            summarize(workspace, **kwargs)
            return workspace, None
        except (FileExistsError, json.JSONDecodeError) as exc:
            return workspace, exc

    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for workspace, exc in executor.map(_run, workspaces):
            if exc is not None:
                LOGGER.error("%s: %s", workspace.name, exc)
                failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
