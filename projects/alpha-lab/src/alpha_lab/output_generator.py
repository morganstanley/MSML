"""Generate polished user-facing documents in workspace/output/.

Deterministic Python — no LLM calls. Reads existing workspace artifacts
(data_report/, backtest/, plots/, reports/) and formats readable markdown.
All methods are idempotent (overwrites on re-run) and handle missing files
gracefully (skip sections, log warnings).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger("alpha_lab.output_generator")


class OutputGenerator:
    """Builds curated markdown documents in ``{workspace}/output/``."""

    def __init__(self, workspace: str | Path, adapter: Any = None) -> None:
        self.workspace = Path(workspace)
        self.output = self.workspace / "output"
        self.output_plots = self.output / "plots"
        self.adapter = adapter

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        self.output_plots.mkdir(parents=True, exist_ok=True)

    def _read(self, rel: str) -> str | None:
        """Read a workspace-relative file, returning None if missing."""
        p = self.workspace / rel
        if not p.exists():
            logger.warning("Missing file: %s", p)
            return None
        try:
            return p.read_text()
        except Exception as e:
            logger.warning("Cannot read %s: %s", p, e)
            return None

    def _copy_plots(self, src_dir: Path, dest_subdir: str) -> list[str]:
        """Copy image files from *src_dir* into output/plots/*dest_subdir*.

        Returns list of output-relative paths (for markdown links).
        """
        if not src_dir.is_dir():
            return []
        dest = self.output_plots / dest_subdir
        dest.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".svg"):
                shutil.copy2(f, dest / f.name)
                copied.append(f"plots/{dest_subdir}/{f.name}")
        return copied

    @staticmethod
    def _get_metric(row: dict, candidates: list[str]) -> object:
        """Return the first non-empty value from *candidates* keys in *row*."""
        for key in candidates:
            val = row.get(key)
            if val is not None and str(val).strip() != "":
                return val
        return None

    def _write_doc(self, name: str, content: str) -> Path:
        self._ensure_dirs()
        p = self.output / name
        p.write_text(content)
        logger.info("Wrote %s", p)
        return p

    # ------------------------------------------------------------------
    # Phase 1: Data Exploration Summary
    # ------------------------------------------------------------------

    def generate_phase1_summary(self) -> Path | None:
        """Produce ``output/01_data_exploration.md``."""
        findings = self._read("data_report/findings.md")
        schema = self._read("data_report/schema.md")
        learnings = self._read("learnings.md")

        if not findings and not schema and not learnings:
            logger.warning("No Phase 1 artifacts found — skipping summary")
            return None

        # Copy exploration plots
        plot_links = self._copy_plots(self.workspace / "plots", "exploration")
        # Exclude backtest sub-dir (Phase 2)
        plot_links = [p for p in plot_links if "/backtest/" not in p]

        sections: list[str] = []
        sections.append("# Data Exploration Summary\n")

        # Executive summary — extract first paragraph of findings
        if findings:
            exec_lines = []
            for line in findings.split("\n"):
                if line.startswith("## Executive") or line.startswith("## Data quality"):
                    if exec_lines:
                        break
                if exec_lines or line.startswith("## Executive"):
                    exec_lines.append(line)
            if exec_lines:
                sections.append("\n".join(exec_lines))
                sections.append("")

        # Dataset overview from schema
        if schema:
            sections.append("## Dataset Overview\n")
            sections.append(schema)
            sections.append("")

        # Full findings
        if findings:
            sections.append("## Key Findings\n")
            sections.append(findings)
            sections.append("")

        # Learnings highlights
        if learnings:
            sections.append("## Accumulated Learnings\n")
            sections.append(learnings)
            sections.append("")

        # Plot gallery
        if plot_links:
            sections.append("## Plots\n")
            for link in plot_links:
                label = Path(link).stem.replace("_", " ").title()
                sections.append(f"### {label}\n")
                sections.append(f"![{label}]({link})\n")

        return self._write_doc("01_data_exploration.md", "\n".join(sections))

    # ------------------------------------------------------------------
    # Phase 2: Methodology
    # ------------------------------------------------------------------

    def generate_methodology_doc(self) -> Path | None:
        """Produce ``output/02_backtest_methodology.md``."""
        framework_dir = "backtest"
        if self.adapter is not None:
            framework_dir = self.adapter.experiment.framework_dir
        engine_src = self._read(f"{framework_dir}/engine.py")
        baselines_src = self._read(f"{framework_dir}/baselines.py")
        review_file = "review.md"
        if self.adapter is not None:
            review_file = self.adapter.phase2_review_file
        review = self._read(f"{framework_dir}/{review_file}")
        run_src = self._read(f"{framework_dir}/run_backtest.py")

        if not engine_src:
            logger.warning("No framework engine found — skipping methodology doc")
            return None

        fw_desc = "Backtest"
        if self.adapter is not None:
            fw_desc = self.adapter.phase2_framework_description.title()

        sections: list[str] = []
        sections.append(f"# {fw_desc} Methodology\n")

        # Intro
        sections.append("## Overview\n")
        sections.append(
            "This document explains how the walk-forward backtesting framework works, "
            "what safeguards prevent lookahead bias, and what baseline strategies are "
            "included. All results in this project are generated by this framework.\n"
        )

        # Walk-forward explanation
        sections.append("## Walk-Forward Backtesting\n")
        sections.append(
            "Walk-forward (also called rolling-origin or expanding-window) backtesting "
            "is the gold standard for evaluating time-series models. Instead of a single "
            "train/test split, the data is divided into multiple sequential folds:\n\n"
            "1. **Train** on data up to time *t*\n"
            "2. **Test** on the next window of data (e.g. the following year)\n"
            "3. **Slide forward** and repeat\n\n"
            "This mirrors how a model would be used in production — it never sees "
            "future data during training.\n"
        )

        # Extract parameters from run_backtest.py argparse defaults
        params = self._extract_backtest_params(run_src)
        if params:
            sections.append("### Default Parameters\n")
            sections.append("| Parameter | Value | Description |")
            sections.append("|-----------|-------|-------------|")
            for name, val, desc in params:
                sections.append(f"| {name} | {val} | {desc} |")
            sections.append("")

        # Lookahead prevention
        sections.append("## Preventing Lookahead Bias\n")
        sections.append(
            "The framework includes several safeguards:\n\n"
            "- **Strictly chronological splits** — data is never shuffled\n"
            "- **Embargo period** — an optional gap between the end of training "
            "and the start of testing, preventing leakage from overlapping labels\n"
            "- **Horizon-aware purging** — when the label is defined as "
            "`y[t] = value[t+h]`, the last *h* rows of each training window are "
            "purged to prevent label overlap with the test window\n"
            "- **Per-split refitting** — each strategy is cloned and refit on each "
            "fold's training data (no state leaks between folds)\n"
            "- **Out-of-sample metrics only** — performance is computed exclusively "
            "on test-fold predictions\n"
        )

        # Baseline strategies
        if baselines_src:
            sections.append("## Baseline Strategies\n")
            sections.append(
                "Baselines provide a floor that any ML model must beat to be useful. "
                "The following are included:\n"
            )
            strategies = self._extract_strategy_docs(baselines_src)
            for name, doc in strategies:
                sections.append(f"### {name}\n")
                sections.append(f"{doc}\n")

        # Metrics
        sections.append("## Metrics\n")
        sections.append(
            "Each backtest produces both **forecasting** and **trading** metrics:\n\n"
            "**Forecasting:** MAE, RMSE, R², directional accuracy\n\n"
            "**Trading:** Sharpe ratio, Sortino ratio, maximum drawdown, total return, "
            "average turnover (all computed on next-day attributed returns to avoid "
            "overlapping-horizon compounding artifacts).\n"
        )

        # Independent review
        if review:
            sections.append("## Independent Code Review\n")
            # Extract the verdict
            verdict = "PASS" if "PASS" in review else "NEEDS FIXES"
            sections.append(
                f"An independent critic agent reviewed the entire backtest framework "
                f"for lookahead bias, data leakage, and correctness. "
                f"**Verdict: {verdict}.**\n"
            )
            # Include the summary section
            summary_match = re.search(
                r"## Summary of leakage-safety design\n(.*?)(?=\n## |\Z)",
                review,
                re.DOTALL,
            )
            if summary_match:
                sections.append(summary_match.group(1).strip())
                sections.append("")

        return self._write_doc("02_backtest_methodology.md", "\n".join(sections))

    def _extract_backtest_params(self, run_src: str | None) -> list[tuple[str, str, str]]:
        """Extract argparse defaults from run_backtest.py source."""
        if not run_src:
            return []
        params = []
        for match in re.finditer(
            r'add_argument\("--([^"]+)".*?default=([^,\)]+)',
            run_src,
        ):
            arg_name = match.group(1).replace("-", "_")
            default = match.group(2).strip()
            # Clean up expressions like "365 * 5"
            try:
                display_val = str(eval(default))  # noqa: S307 — trusted source
            except Exception:
                display_val = default
            params.append((arg_name, display_val, ""))
        return params

    def _extract_strategy_docs(self, src: str) -> list[tuple[str, str]]:
        """Extract strategy class names and docstrings from baselines.py."""
        results = []
        for match in re.finditer(
            r'class\s+(\w+)\(Strategy\):\s*\n\s*"""(.*?)"""',
            src,
            re.DOTALL,
        ):
            name = match.group(1)
            doc = match.group(2).strip().split("\n")[0]  # first line
            results.append((name, doc))
        # Also catch classes with `name: str = "..."` but no docstring
        for match in re.finditer(
            r'class\s+(\w+)\(Strategy\):\s*\n\s*name:\s*str\s*=\s*"([^"]+)"',
            src,
        ):
            cls_name = match.group(1)
            if not any(n == cls_name for n, _ in results):
                results.append((cls_name, f"Baseline strategy: {match.group(2)}"))
        return results

    # ------------------------------------------------------------------
    # Phase 2: Baseline Results
    # ------------------------------------------------------------------

    # Known metric column names (no structural columns like country/strategy)
    _KNOWN_METRICS = {
        "mae", "rmse", "r2", "sharpe", "sharpe_next_day",
        "max_dd", "max_drawdown", "max_drawdown_next_day",
        "total_return", "total_return_next_day",
        "sortino", "sortino_next_day",
        "ann_return", "ann_vol", "avg_turnover", "total_cost",
    }

    def _looks_like_metrics(self, columns: set[str]) -> bool:
        """Return True if *columns* look like baseline metrics output."""
        # Require strategy column + at least 1 recognized metric column.
        # Recognized metrics = _KNOWN_METRICS plus the adapter's primary metric
        # (so non-time-series adapters like wall_clock_seconds match too).
        has_strategy = "strategy" in columns
        known = set(self._KNOWN_METRICS)
        if self.adapter is not None:
            if self.adapter.metric.primary_metric:
                known.add(self.adapter.metric.primary_metric)
        metric_hits = columns & known
        return has_strategy and len(metric_hits) >= 1

    def _try_load_csv(self, path: Path) -> list[dict] | None:
        try:
            rows = list(csv.DictReader(path.open()))
            if rows and self._looks_like_metrics(set(rows[0].keys())):
                logger.info("Loaded baseline metrics from %s", path)
                return rows
        except Exception:
            pass
        return None

    def _try_load_parquet(self, path: Path) -> list[dict] | None:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            if self._looks_like_metrics(set(df.columns)):
                logger.info("Loaded baseline metrics from %s", path)
                return df.to_dict("records")
        except Exception:
            pass
        return None

    def _load_baseline_metrics(self) -> list[dict] | None:
        """Find and load baseline metrics from the workspace.

        Strategy:
        1. Check the canonical path (written by pipeline post-step)
        2. Check well-known locations from previous runs
        3. Scoped discovery: recurse only under likely artifact directories
           (reports/, plots/, backtest/, output/) instead of globbing the
           whole workspace — the broad glob was hanging on NFS when data/
           was large.
        """
        # 1. Canonical path (deterministic post-Phase-2 output)
        canonical = self.workspace / "output" / "baseline_metrics.csv"
        result = self._try_load_csv(canonical)
        if result:
            return result

        # 2. Well-known locations
        well_known_csv = [
            self.workspace / "plots" / "backtest" / "metrics_summary.csv",
            self.workspace / "backtest" / "metrics_summary.csv",
        ]
        for p in well_known_csv:
            result = self._try_load_csv(p)
            if result:
                return result

        well_known_pq = [
            self.workspace / "output" / "backtest_metrics.parquet",
            self.workspace / "backtest_metrics.parquet",
        ]
        for p in well_known_pq:
            result = self._try_load_parquet(p)
            if result:
                return result

        # 3. Scoped discovery — search likely subdirectories only
        # (avoid data/ which can be huge on NFS and cause hangs)
        logger.info("Well-known metric paths not found, searching scoped directories...")
        search_dirs = [
            self.workspace / "reports",
            self.workspace / "plots",
            self.workspace / "backtest",
            self.workspace / "output",
        ]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for csv_path in sorted(search_dir.rglob("*.csv")):
                if "experiment" in str(csv_path) or "__pycache__" in str(csv_path):
                    continue
                result = self._try_load_csv(csv_path)
                if result:
                    return result
            for pq_path in sorted(search_dir.rglob("*.parquet")):
                if "experiment" in str(pq_path) or "__pycache__" in str(pq_path):
                    continue
                result = self._try_load_parquet(pq_path)
                if result:
                    return result

        logger.warning("No baseline metrics found in workspace")
        return None

    def generate_baseline_results(self) -> Path | None:
        """Produce ``output/03_baseline_results.md``."""
        rows = self._load_baseline_metrics()
        if not rows:
            logger.warning("No baseline metrics found — skipping baseline results")
            return None

        # Copy backtest plots
        plot_links = self._copy_plots(
            self.workspace / "plots" / "backtest", "backtest"
        )

        sections: list[str] = []
        sections.append("# Baseline Results\n")
        sections.append(
            "Performance of baseline strategies evaluated via walk-forward "
            "backtesting. These numbers represent the floor that any ML/DL model "
            "must beat.\n"
        )

        # Key metrics columns — try multiple column name variants
        # (the Phase 2 agent may use different names across runs)
        metric_cols = [
            (["mae"], "MAE", ".4f"),
            (["rmse"], "RMSE", ".4f"),
            (["r2"], "R²", ".4f"),
            (["sharpe", "sharpe_next_day"], "Sharpe", ".4f"),
            (["max_dd", "max_drawdown_next_day", "max_drawdown"], "Max DD", ".4f"),
            (["total_return", "total_return_next_day"], "Total Return", ".4f"),
        ]
        # Extend with the adapter's primary metric if it isn't already in the
        # hardcoded time-series list. Non-time-series adapters (nanogpt
        # wall_clock_seconds, cuda_kernel throughput_gflops, etc.) otherwise
        # never appear in the summary table.
        if self.adapter is not None and self.adapter.metric.primary_metric:
            pm = self.adapter.metric.primary_metric
            _existing_aliases = {alias for aliases, _, _ in metric_cols for alias in aliases}
            if pm not in _existing_aliases:
                metric_cols.append(([pm], self.adapter.metric.display_name or pm, ".4f"))

        # Auto-detect grouping columns: any non-metric column with >1 distinct value.
        # Presence checks use `col in row` rather than truthy .get(col) — a legitimate
        # grouping value like 0 (e.g. seed=0, horizon=0) or False would otherwise be
        # silently dropped, collapsing groups or producing phantom "unknown" rows.
        _metric_keys = set()
        for aliases, _, _ in metric_cols:
            _metric_keys.update(aliases)
        _meta_keys = {"strategy", "model", "name"}  # not grouping dims
        _all_keys: set[str] = set()
        for row in rows:
            _all_keys.update(row.keys())
        _group_cols: list[str] = []
        for col in sorted(_all_keys - _metric_keys - _meta_keys):
            distinct = set(
                str(row[col]) for row in rows if col in row and row[col] is not None
            )
            if 1 < len(distinct) < len(rows):
                _group_cols.append(col)

        def _group_key(row: dict) -> str:
            parts = [
                f"{col}={row[col]}" for col in _group_cols
                if col in row and row[col] is not None
            ]
            return " ".join(parts) if parts else ""

        groups: dict[str, list[dict]] = {}
        for row in rows:
            key = _group_key(row)
            groups.setdefault(key, []).append(row)

        # Summary table
        sections.append("## Summary Table\n")
        show_group_col = len(groups) > 1
        if show_group_col:
            header = "| Group | Strategy | " + " | ".join(m[1] for m in metric_cols) + " |"
            sep = "|-------|----------|" + "|".join("------:" for _ in metric_cols) + "|"
        else:
            header = "| Strategy | " + " | ".join(m[1] for m in metric_cols) + " |"
            sep = "|----------|" + "|".join("------:" for _ in metric_cols) + "|"
        sections.append(header)
        sections.append(sep)

        for group_key in sorted(groups):
            for row in groups[group_key]:
                vals = []
                for candidates, _, fmt in metric_cols:
                    raw = self._get_metric(row, candidates)
                    if raw is not None:
                        try:
                            vals.append(f"{float(raw):{fmt}}")
                        except (ValueError, TypeError):
                            vals.append(str(raw))
                    else:
                        vals.append("—")
                if show_group_col:
                    line = f"| {group_key} | {row.get('strategy', '?')} | " + " | ".join(vals) + " |"
                else:
                    line = f"| {row.get('strategy', '?')} | " + " | ".join(vals) + " |"
                sections.append(line)
        sections.append("")

        # Key observations
        sections.append("## Key Observations\n")

        # Determine primary metric from adapter (fall back to sharpe)
        _bl_metric = "sharpe"
        _bl_metric_display = "Sharpe"
        _bl_direction = "maximize"
        if self.adapter is not None:
            _bl_metric = self.adapter.metric.primary_metric
            _bl_metric_display = self.adapter.metric.display_name
            _bl_direction = self.adapter.metric.direction

        # Build list of column name variants for the primary metric.
        # Baseline CSVs sometimes suffix metrics with "_next_day" (a time-series
        # convention); strip/add that suffix so either variant is picked up
        # without needing a sharpe-specific special case.
        _bl_keys: list[str] = [_bl_metric]
        if _bl_metric.endswith("_next_day"):
            base = _bl_metric[: -len("_next_day")]
            if base and base not in _bl_keys:
                _bl_keys.append(base)
        else:
            next_day = f"{_bl_metric}_next_day"
            if next_day not in _bl_keys:
                _bl_keys.append(next_day)

        # Find best strategy per group by primary metric
        best_by_group: list[tuple[str, str, float]] = []
        for group_key, crows in sorted(groups.items()):
            best_row = None
            best_val = float("inf") if _bl_direction == "minimize" else float("-inf")
            for row in crows:
                s = self._get_metric(row, _bl_keys)
                try:
                    s = float(s)
                    if math.isnan(s) or math.isinf(s):
                        continue
                    is_better = s < best_val if _bl_direction == "minimize" else s > best_val
                    if is_better:
                        best_val = s
                        best_row = row
                except (ValueError, TypeError):
                    continue
            if best_row:
                label = group_key if group_key else "Overall"
                best_by_group.append((label, best_row.get("strategy", "?"), best_val))

        if best_by_group:
            group_label = "per group" if show_group_col else "(overall)"
            sections.append(f"**Best baseline by {_bl_metric_display} {group_label}:**\n")
            for label, strat, val in best_by_group:
                sections.append(f"- **{label}**: {strat} ({_bl_metric_display} {val:.4f})")
            sections.append("")

        # Overall observations
        all_vals = []
        for row in rows:
            s = self._get_metric(row, _bl_keys)
            try:
                v = float(s)
                if not math.isnan(v) and not math.isinf(v):
                    all_vals.append(v)
            except (ValueError, TypeError):
                pass
        if all_vals:
            sections.append(
                f"- {_bl_metric_display} across all baselines ranges from "
                f"{min(all_vals):.4f} to {max(all_vals):.4f}\n"
            )

        # Plots
        if plot_links:
            sections.append("## Plots\n")
            for link in plot_links:
                label = Path(link).stem.replace("_", " ").title()
                sections.append(f"### {label}\n")
                sections.append(f"![{label}]({link})\n")

        return self._write_doc("03_baseline_results.md", "\n".join(sections))

    # ------------------------------------------------------------------
    # Phase 3: Milestone Reports
    # ------------------------------------------------------------------

    def copy_milestone_report(self, milestone_number: int) -> Path | None:
        """Copy ``reports/milestone_NNN/report.md`` → ``output/04_milestone_NNN.md``.

        Also copies associated plots and rewrites image links.
        """
        # Try both naming conventions
        for dirname in (
            f"milestone_{milestone_number:03d}",
            f"milestone_{milestone_number}",
            str(milestone_number),
        ):
            src_dir = self.workspace / "reports" / dirname
            if src_dir.is_dir():
                break
        else:
            logger.warning(
                "Milestone report directory not found for #%d", milestone_number
            )
            return None

        report_path = src_dir / "report.md"
        if not report_path.exists():
            logger.warning("No report.md in %s", src_dir)
            return None

        content = report_path.read_text()

        # Copy plots from the milestone's plots/ subdir
        plots_src = src_dir / "plots"
        dest_subdir = f"milestone_{milestone_number:03d}"
        if plots_src.is_dir():
            self._copy_plots(plots_src, dest_subdir)
            # Rewrite relative image links
            content = re.sub(
                r"!\[([^\]]*)\]\(plots/([^)]+)\)",
                rf"![\1](plots/{dest_subdir}/\2)",
                content,
            )

        out_name = f"04_milestone_{milestone_number:03d}.md"
        return self._write_doc(out_name, content)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def generate_index(self) -> Path | None:
        """Produce ``output/index.md`` — table of contents."""
        self._ensure_dirs()

        docs = sorted(
            f for f in self.output.iterdir()
            if f.suffix == ".md" and f.name != "index.md"
        )
        if not docs:
            logger.warning("No documents in output/ — skipping index")
            return None

        sections: list[str] = []
        sections.append("# Analysis Output\n")
        sections.append(
            "This directory contains polished summaries of each analysis phase. "
            "Documents are generated automatically from workspace artifacts.\n"
        )
        sections.append("## Table of Contents\n")

        for doc in docs:
            # Read first heading
            title = doc.stem.replace("_", " ").title()
            try:
                first_line = doc.read_text().split("\n")[0]
                if first_line.startswith("# "):
                    title = first_line[2:].strip()
            except Exception:
                pass
            sections.append(f"- [{title}]({doc.name})")

        sections.append("")

        # Note about plots — only count actual image files, not empty subdirs
        # that _copy_plots may have created for absent sources.
        has_plots = False
        try:
            if self.output_plots.exists():
                plot_exts = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
                has_plots = any(
                    p.is_file() and p.suffix.lower() in plot_exts
                    for p in self.output_plots.rglob("*")
                )
        except Exception:
            pass
        if has_plots:
            sections.append(
                "\n*All referenced plots are copied into `plots/` for "
                "self-contained viewing.*\n"
            )

        return self._write_doc("index.md", "\n".join(sections))

    # ------------------------------------------------------------------
    # Status Report (structured JSON, no file write)
    # ------------------------------------------------------------------

    def generate_status_report(self) -> dict[str, Any]:
        """Return a structured status report as a dict.

        Reads baseline metrics from output/baseline_metrics.csv and experiment
        results from the SQLite database.  Returns a dict suitable for JSON
        serialisation and direct rendering by the frontend.
        """
        report: dict[str, Any] = {}

        # --- Problem description (from data_report) ---
        findings = self._read("data_report/findings.md")
        schema = self._read("data_report/schema.md")
        learnings = self._read("learnings.md")

        problem: dict[str, Any] = {}
        # Extract first paragraph from findings as executive summary
        if findings:
            for line in findings.split("\n"):
                if line.startswith("## Executive"):
                    continue
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    problem["summary"] = stripped
                    break
        if schema:
            # Count countries and date range from schema text
            problem["schema_snippet"] = schema[:500]
        if learnings:
            problem["has_learnings"] = True

        # Try to extract dataset info from data exploration doc
        exploration = self._read("output/01_data_exploration.md")
        if exploration:
            problem["exploration_available"] = True

        report["problem"] = problem

        # --- Baseline results ---
        rows = self._load_baseline_metrics()
        baselines: dict[str, Any] = {"available": rows is not None}
        if rows:
            baselines["total_rows"] = len(rows)

            # Determine primary metric from adapter
            _kj_metric = "sharpe"
            _kj_direction = "maximize"
            if self.adapter is not None:
                _kj_metric = self.adapter.metric.primary_metric
                _kj_direction = self.adapter.metric.direction
            _kj_keys: list[str] = [_kj_metric]
            if _kj_metric.endswith("_next_day"):
                base = _kj_metric[: -len("_next_day")]
                if base and base not in _kj_keys:
                    _kj_keys.append(base)
            else:
                next_day = f"{_kj_metric}_next_day"
                if next_day not in _kj_keys:
                    _kj_keys.append(next_day)

            baselines["primary_metric"] = _kj_metric

            # Auto-detect grouping columns (non-metric, non-strategy,
            # with >1 distinct value — works for any domain).
            # Exclude known metric columns (not all numeric columns), so
            # naturally-numeric dims like horizon / seed / batch_size still
            # get detected as groups.
            _all_cols: set[str] = set()
            for r in rows:
                _all_cols.update(r.keys())
            _meta_cols = {"strategy", "model", "name"}
            _metric_cols: set[str] = set(self._KNOWN_METRICS) | set(_kj_keys)
            _group_cols: list[str] = []
            # Presence checks use `col in r` rather than truthy .get(col) —
            # otherwise legitimate grouping values like 0 / False are dropped,
            # collapsing groups or producing missing group_dims entries.
            for col in sorted(_all_cols - _metric_cols - _meta_cols):
                distinct = set(
                    str(r[col]) for r in rows if col in r and r[col] is not None
                )
                if 1 < len(distinct) < len(rows):
                    _group_cols.append(col)

            # Report discovered grouping columns under a stable schema key
            # (avoid incorrect/naive pluralization like "countrys").
            baselines["group_dims"] = {}
            for col in _group_cols:
                vals = sorted(set(
                    str(r[col]) for r in rows if col in r and r[col] is not None
                ))
                baselines["group_dims"][col] = vals

            strategies = sorted(set(r.get("strategy", "") for r in rows))
            baselines["strategies"] = strategies

            # Best baseline per group (by primary metric)
            def _group_key(row: dict) -> tuple:
                return tuple(str(row.get(c, "")) for c in _group_cols)

            groups: dict[tuple, list[dict]] = {}
            for r in rows:
                groups.setdefault(_group_key(r), []).append(r)

            best_per_group: list[dict[str, Any]] = []
            for gk, g_rows in groups.items():
                best_row = None
                best_val = float("inf") if _kj_direction == "minimize" else float("-inf")
                for r in g_rows:
                    s = self._get_metric(r, _kj_keys)
                    try:
                        sv = float(s)
                        if math.isnan(sv) or math.isinf(sv):
                            continue
                        is_better = sv < best_val if _kj_direction == "minimize" else sv > best_val
                        if is_better:
                            best_val = sv
                            best_row = r
                    except (ValueError, TypeError):
                        continue
                if best_row:
                    entry: dict[str, Any] = {
                        "strategy": best_row.get("strategy", "?"),
                        _kj_metric: round(best_val, 4),
                    }
                    for col, val in zip(_group_cols, gk):
                        entry[col] = val
                    best_per_group.append(entry)
            baselines["best_per_group"] = best_per_group

            # Average primary metric per strategy across all rows
            strat_vals: dict[str, list[float]] = {}
            for r in rows:
                strat = r.get("strategy", "?")
                s = self._get_metric(r, _kj_keys)
                try:
                    sv = float(s)
                    if not math.isnan(sv) and not math.isinf(sv):
                        strat_vals.setdefault(strat, []).append(sv)
                except (ValueError, TypeError):
                    pass
            baselines[f"avg_{_kj_metric}_by_strategy"] = {
                s: round(sum(vals) / len(vals), 4)
                for s, vals in strat_vals.items() if vals
            }

        report["baselines"] = baselines

        # --- Experiment results ---
        experiments: dict[str, Any] = {"available": False}
        db = self._open_experiment_db()
        if db:
            try:
                board = db.board_summary()
                experiments["available"] = True
                experiments["board"] = board
                experiments["total"] = sum(board.values())

                all_exps = db.list_all()

                # Top models by primary metric
                _primary = "sharpe"
                _direction = "maximize"
                if self.adapter is not None:
                    _primary = self.adapter.metric.primary_metric
                    _direction = self.adapter.metric.direction

                scored: list[dict[str, Any]] = []
                for exp in all_exps:
                    if not exp.results_json:
                        continue
                    try:
                        results = json.loads(exp.results_json)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if not isinstance(results, dict):
                        continue
                    primary_val = results.get(_primary)
                    if primary_val is None:
                        continue
                    try:
                        primary_float = float(primary_val)
                        if math.isnan(primary_float) or math.isinf(primary_float):
                            continue
                        # Sanity cap for ratio metrics (sharpe, sortino) — not for
                        # absolute metrics (throughput, wall_clock_seconds)
                        if _primary in ("sharpe", "sortino") and abs(primary_float) > 100:
                            continue
                    except (ValueError, TypeError):
                        primary_float = None

                    entry: dict[str, Any] = {
                        "name": exp.name,
                        "status": exp.status,
                        "primary_metric": _primary,
                        _primary: round(primary_float, 4) if primary_float is not None else None,
                        "slurm_job_id": exp.slurm_job_id,
                    }
                    # Include secondary metrics if available
                    for k, v in results.items():
                        if k != _primary and k not in entry:
                            entry[k] = self._safe_float(v)
                    scored.append(entry)

                # Sort by primary metric
                def _sort_key(x: dict) -> float:
                    v = x.get(_primary)
                    if v is None:
                        return float("inf") if _direction == "minimize" else float("-inf")
                    return v

                scored.sort(key=_sort_key, reverse=(_direction == "maximize"))
                experiments["top_models"] = scored[:10]
                experiments["all_scored_count"] = len(scored)

                # Failures
                failed = [
                    {"name": e.name, "error": (e.error or "")[:200]}
                    for e in all_exps
                    if e.status in ("slurm_failed", "error") or (
                        e.error and e.status == "analyzed"
                    )
                ]
                experiments["failures"] = failed

                # Running SLURM jobs
                running = [
                    {
                        "name": e.name,
                        "slurm_job_id": e.slurm_job_id,
                    }
                    for e in all_exps
                    if e.status == "running" and e.slurm_job_id
                ]
                experiments["running_slurm"] = running

            except Exception as e:
                logger.error("Failed to read experiment DB: %s", e)
                experiments["error"] = str(e)

        report["experiments"] = experiments

        # --- Comparison: top models vs baselines ---
        comparison: list[dict[str, Any]] = []
        if rows and experiments.get("top_models"):
            # Build baseline column-name variants (strip/add _next_day suffix),
            # generalizing beyond the sharpe special case.
            _baseline_keys: list[str] = [_primary]
            if _primary.endswith("_next_day"):
                base = _primary[: -len("_next_day")]
                if base and base not in _baseline_keys:
                    _baseline_keys.append(base)
            else:
                next_day = f"{_primary}_next_day"
                if next_day not in _baseline_keys:
                    _baseline_keys.append(next_day)
            all_baseline_vals = []
            for r in rows:
                s = self._get_metric(r, _baseline_keys)
                try:
                    v = float(s)
                    if not math.isnan(v) and not math.isinf(v):
                        all_baseline_vals.append(v)
                except (ValueError, TypeError):
                    pass
            # Use None (not 0) when no usable baselines — 0 is a plausible real
            # value for many metrics, so falling back to it produced spurious
            # beats_best / beats_avg comparisons.
            if all_baseline_vals:
                best_baseline_val: float | None = (
                    min(all_baseline_vals) if _direction == "minimize"
                    else max(all_baseline_vals)
                )
                avg_baseline_val: float | None = sum(all_baseline_vals) / len(all_baseline_vals)
            else:
                best_baseline_val = None
                avg_baseline_val = None

            for model in experiments["top_models"][:5]:
                model_primary = model.get(_primary)
                if model_primary is None:
                    continue
                entry: dict[str, Any] = {
                    "name": model["name"],
                    "model_primary_metric": model_primary,
                    "primary_metric_name": _primary,
                }
                if best_baseline_val is not None:
                    entry["best_baseline"] = round(best_baseline_val, 4)
                    entry["beats_best_baseline"] = (
                        model_primary < best_baseline_val if _direction == "minimize"
                        else model_primary > best_baseline_val
                    )
                if avg_baseline_val is not None:
                    entry["avg_baseline"] = round(avg_baseline_val, 4)
                    entry["beats_avg_baseline"] = (
                        model_primary < avg_baseline_val if _direction == "minimize"
                        else model_primary > avg_baseline_val
                    )
                comparison.append(entry)

        report["comparison"] = comparison

        # Timestamp
        report["generated_at"] = time.time()

        return report

    @staticmethod
    def _safe_float(val: Any) -> float | None:
        if val is None:
            return None
        try:
            v = float(val)
            # Reject NaN/Inf explicitly so they don't leak into JSON/reports;
            # abs(nan)<1e10 happens to evaluate false, but isfinite is what
            # we actually mean here.
            if not math.isfinite(v):
                return None
            return round(v, 4) if abs(v) < 1e10 else None
        except (ValueError, TypeError):
            return None

    def _open_experiment_db(self, db_path: str | None = None) -> Any:
        """Open the experiment database if it exists."""
        if db_path is None:
            db_path = str(self.workspace / "experiments.db")
        if not os.path.exists(db_path):
            return None
        try:
            from alpha_lab.experiment_db import ExperimentDB
            return ExperimentDB(db_path)
        except Exception as e:
            logger.warning("Cannot open experiment DB: %s", e)
            return None
