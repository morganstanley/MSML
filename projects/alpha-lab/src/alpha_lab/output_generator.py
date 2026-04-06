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
import os
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
        # Match add_argument lines with defaults
        descs = {
            "horizon": "Forecast horizon in days",
            "n_lags": "Number of lag features",
            "initial_train": "Initial training window (days)",
            "test_size": "Test window size (days)",
            "step_size": "Step size between folds (days)",
            "embargo": "Embargo gap between train and test (days)",
            "transaction_cost": "Cost per unit turnover",
            "periods_per_year": "Trading days per year (for annualization)",
        }
        for match in re.finditer(
            r'add_argument\("--([^"]+)".*?default=([^,\)]+)',
            run_src,
        ):
            arg_name = match.group(1).replace("-", "_")
            default = match.group(2).strip()
            if arg_name in descs:
                # Clean up expressions like "365 * 5"
                try:
                    display_val = str(eval(default))  # noqa: S307 — trusted source
                except Exception:
                    display_val = default
                params.append((arg_name, display_val, descs[arg_name]))
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

    # Columns that indicate a file contains backtest metrics
    _METRICS_FINGERPRINT = {"mae", "rmse", "sharpe", "country", "strategy"}

    def _looks_like_metrics(self, columns: set[str]) -> bool:
        """Return True if *columns* look like baseline metrics output."""
        # If adapter is available, check for its primary/secondary metrics
        if self.adapter is not None:
            known = {self.adapter.metric.primary_metric} | set(
                self.adapter.metric.secondary_metrics
            )
            if known & columns:
                return True
        # Fallback heuristic for time_series (no adapter)
        has_keys = {"country", "strategy"}.issubset(columns)
        metric_hits = columns & {
            "mae", "rmse", "r2", "sharpe", "sharpe_next_day",
            "max_dd", "max_drawdown", "max_drawdown_next_day",
            "total_return", "total_return_next_day", "sortino",
            "sortino_next_day",
        }
        return has_keys and len(metric_hits) >= 2

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
        3. Broad discovery: glob for any CSV/Parquet with metrics-like columns
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

        # 3. Broad discovery — search the whole workspace
        logger.info("Well-known metric paths not found, searching workspace...")
        for csv_path in sorted(self.workspace.rglob("*.csv")):
            if "experiment" in str(csv_path) or "__pycache__" in str(csv_path):
                continue
            result = self._try_load_csv(csv_path)
            if result:
                return result

        for pq_path in sorted(self.workspace.rglob("*.parquet")):
            if "experiment" in str(pq_path) or "__pycache__" in str(pq_path):
                continue
            result = self._try_load_parquet(pq_path)
            if result:
                return result

        logger.warning("No baseline metrics found anywhere in workspace")
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

        # Detect if data has a horizon column (multi-horizon runs)
        has_horizon = any("horizon" in row for row in rows)

        # Group by country (and horizon if present)
        groups: dict[str, list[dict]] = {}
        for row in rows:
            key = str(row.get("country", ""))
            if has_horizon:
                key = f"{row.get('country', '')} (h={row.get('horizon', '?')})"
            groups.setdefault(key, []).append(row)

        # Summary table
        sections.append("## Summary Table\n")
        header = "| Country | Strategy | " + " | ".join(m[1] for m in metric_cols) + " |"
        sep = "|---------|----------|" + "|".join("------:" for _ in metric_cols) + "|"
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
                line = f"| {group_key} | {row.get('strategy', '?')} | " + " | ".join(vals) + " |"
                sections.append(line)
        sections.append("")

        # Key observations
        sections.append("## Key Observations\n")

        # Find best strategy per country by Sharpe
        sharpe_keys = ["sharpe", "sharpe_next_day"]
        best_by_country: list[tuple[str, str, float]] = []
        for group_key, crows in sorted(groups.items()):
            best_row = None
            best_sharpe = -999.0
            for row in crows:
                s = self._get_metric(row, sharpe_keys)
                try:
                    s = float(s)
                    if s > best_sharpe:
                        best_sharpe = s
                        best_row = row
                except (ValueError, TypeError):
                    continue
            if best_row:
                best_by_country.append((group_key, best_row.get("strategy", "?"), best_sharpe))

        if best_by_country:
            sections.append("**Best baseline by Sharpe per country:**\n")
            for label, strat, sharpe in best_by_country:
                sections.append(f"- **{label}**: {strat} (Sharpe {sharpe:.4f})")
            sections.append("")

        # Overall observations
        all_sharpes = []
        for row in rows:
            s = self._get_metric(row, sharpe_keys)
            try:
                all_sharpes.append(float(s))
            except (ValueError, TypeError):
                pass
        if all_sharpes:
            sections.append(
                f"- Sharpe ratios across all baselines range from "
                f"{min(all_sharpes):.4f} to {max(all_sharpes):.4f}\n"
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

        # Note about plots
        if self.output_plots.exists() and any(self.output_plots.rglob("*")):
            sections.append(
                "\n*All referenced plots are copied into `plots/` for "
                "self-contained viewing.*\n"
            )

        return self._write_doc("index.md", "\n".join(sections))

    # ------------------------------------------------------------------
    # Status Report (structured JSON, no file write)
    # ------------------------------------------------------------------

    def generate_status_report(self, db_path: str | None = None) -> dict[str, Any]:
        """Return a structured status report as a dict.

        Reads baseline metrics from output/baseline_metrics.csv and experiment
        results from the SQLite database.  Returns a dict suitable for JSON
        serialisation and direct rendering by the frontend.
        """
        report: dict[str, Any] = {}

        # --- Metric config (from adapter) ---
        _primary = "sharpe"
        _direction = "maximize"
        _display_name = "Sharpe"
        if self.adapter is not None:
            _primary = self.adapter.metric.primary_metric
            _direction = self.adapter.metric.direction
            _display_name = self.adapter.metric.display_name
        report["metric_config"] = {
            "primary_metric": _primary,
            "direction": _direction,
            "display_name": _display_name,
        }

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

            # Group by horizon
            horizons = sorted(set(
                int(r["horizon"]) for r in rows if "horizon" in r and r["horizon"]
            )) if any("horizon" in r for r in rows) else []
            baselines["horizons"] = horizons

            strategies = sorted(set(r.get("strategy", "") for r in rows))
            baselines["strategies"] = strategies

            countries = sorted(set(r.get("country", "") for r in rows))
            baselines["countries"] = countries

            # Best baseline per horizon (by primary metric)
            primary_keys = [_primary]
            _better = (lambda a, b: a > b) if _direction == "maximize" else (lambda a, b: a < b)
            _worst = -999.0 if _direction == "maximize" else 999.0
            best_per_horizon: list[dict[str, Any]] = []
            for h in (horizons or [None]):
                h_rows = [r for r in rows if (h is None or str(r.get("horizon")) == str(h))]
                if not h_rows:
                    continue
                best_row = None
                best_val = _worst
                for r in h_rows:
                    s = self._get_metric(r, primary_keys)
                    try:
                        sv = float(s)
                        if _better(sv, best_val):
                            best_val = sv
                            best_row = r
                    except (ValueError, TypeError):
                        continue
                if best_row:
                    entry: dict[str, Any] = {}
                    if h is not None:
                        entry["horizon"] = h
                    if best_row.get("strategy"):
                        entry["strategy"] = best_row["strategy"]
                    if best_row.get("country"):
                        entry["country"] = best_row["country"]
                    entry[_primary] = round(best_val, 4)
                    # Include other numeric values from the row
                    for k, v in best_row.items():
                        if k not in entry and k not in ("horizon", "strategy", "country"):
                            fv = self._safe_float(v)
                            if fv is not None:
                                entry[k] = fv
                    best_per_horizon.append(entry)
            baselines["best_per_horizon"] = best_per_horizon

            # Average primary metric per strategy across all rows
            strat_values: dict[str, list[float]] = {}
            for r in rows:
                strat = r.get("strategy", "?")
                s = self._get_metric(r, primary_keys)
                try:
                    strat_values.setdefault(strat, []).append(float(s))
                except (ValueError, TypeError):
                    pass
            baselines["avg_primary_by_strategy"] = {
                s: round(sum(vals) / len(vals), 4)
                for s, vals in strat_values.items() if vals
            }

        report["baselines"] = baselines

        # --- Experiment results ---
        experiments: dict[str, Any] = {"available": False}
        db = self._open_experiment_db(db_path)
        if db:
            try:
                board = db.board_summary()
                experiments["available"] = True
                experiments["board"] = board
                experiments["total"] = sum(board.values())

                all_exps = db.list_all()

                # Top models by primary metric (uses _primary/_direction from top)

                scored: list[dict[str, Any]] = []
                for exp in all_exps:
                    if not exp.results_json:
                        continue
                    try:
                        results = json.loads(exp.results_json)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    primary_val = results.get(_primary)
                    if primary_val is None:
                        continue
                    try:
                        primary_float = float(primary_val)
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
            all_baseline_vals = []
            for r in rows:
                s = self._get_metric(r, [_primary])
                try:
                    all_baseline_vals.append(float(s))
                except (ValueError, TypeError):
                    pass

            if all_baseline_vals:
                best_baseline = (max if _direction == "maximize" else min)(all_baseline_vals)
                avg_baseline = sum(all_baseline_vals) / len(all_baseline_vals)
            else:
                best_baseline = 0
                avg_baseline = 0

            for model in experiments["top_models"][:5]:
                model_val = model.get(_primary)
                if model_val is not None:
                    if _direction == "maximize":
                        beats = model_val > best_baseline
                    else:
                        beats = model_val < best_baseline
                    comparison.append({
                        "name": model["name"],
                        "model_primary": model_val,
                        "best_baseline": round(best_baseline, 4),
                        "avg_baseline": round(avg_baseline, 4),
                        "beats_best_baseline": beats,
                    })

        report["comparison"] = comparison

        # Timestamp
        import time
        report["generated_at"] = time.time()

        return report

    @staticmethod
    def _safe_float(val: Any) -> float | None:
        if val is None:
            return None
        try:
            v = float(val)
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
