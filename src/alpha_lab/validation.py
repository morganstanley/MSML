"""System-side validation for experiment reality checks (used by the ``reality_check`` tool).

This module is called by the *system* (step 9 in the worker prompt) to validate
an experiment against real data before GPU submission.  It reads config.yaml and
infers everything automatically.

Workers also have ``experiment_validation.py`` (step 5) — a set of explicit-arg
helpers they call *inside* ``run_experiment.py`` during development.  Time estimation
is delegated to that module so both paths use the same heuristics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# pandas and yaml are lazy-imported inside the validation entry points so
# importing this module (e.g. for reality_check) doesn't fail with
# ImportError in environments that don't ship the optional runtime deps.
# The entry points surface the missing dep as a ValidationReport error.


@dataclass
class ValidationReport:
    """Generic validation results — works for any ML task."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    estimated_runtime_seconds: float | None = None
    time_limit_seconds: float | None = None

    @property
    def passed(self) -> bool:
        """Validation passes if there are no blocking errors."""
        return len(self.errors) == 0

    @property
    def timing_ok(self) -> bool:
        """Check if estimated runtime fits within time limit."""
        if self.estimated_runtime_seconds is None or self.time_limit_seconds is None:
            return True  # Can't check, assume OK
        return self.estimated_runtime_seconds <= self.time_limit_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "estimated_runtime_seconds": self.estimated_runtime_seconds,
            "time_limit_seconds": self.time_limit_seconds,
            "timing_ok": self.timing_ok,
        }

    def format(self) -> str:
        """Format report as readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("VALIDATION REPORT")
        lines.append("=" * 60)

        if self.passed:
            lines.append("✓ PASSED (no blocking errors)")
        else:
            lines.append("✗ FAILED (blocking errors found)")

        if self.errors:
            lines.append("\n❌ ERRORS (must fix):")
            for i, err in enumerate(self.errors, 1):
                lines.append(f"  {i}. {err}")

        if self.warnings:
            lines.append("\n⚠️  WARNINGS (should review):")
            for i, warn in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warn}")

        if self.estimated_runtime_seconds is not None:
            hours = int(self.estimated_runtime_seconds // 3600)
            minutes = int((self.estimated_runtime_seconds % 3600) // 60)
            seconds = int(self.estimated_runtime_seconds % 60)
            runtime_str = f"{hours}h {minutes}m {seconds}s"
            lines.append(f"\n⏱️  Estimated runtime: {runtime_str}")

            if self.time_limit_seconds is not None:
                limit_hours = int(self.time_limit_seconds // 3600)
                limit_str = f"{limit_hours}h"
                if self.timing_ok:
                    lines.append(f"   ✓ Fits within time limit ({limit_str})")
                else:
                    lines.append(f"   ✗ EXCEEDS time limit ({limit_str})")

        lines.append("=" * 60)
        return "\n".join(lines)


def _collect_file_paths(
    config: dict[str, Any], workspace: Path
) -> list[tuple[str, Path]]:
    """Collect all file paths referenced in a config, regardless of key name.

    Scans top-level keys, ``data.*``, and ``paths.*`` for string values that
    look like file paths (.parquet, .csv, or containing '/').  Returns a list
    of ``(raw_value, resolved_path)`` tuples.  Absolute paths are kept as-is;
    relative paths are resolved against *workspace*.
    """
    candidates: list[str] = []

    def _scan(d: dict) -> None:
        for v in d.values():
            if isinstance(v, str) and (
                v.endswith(".parquet") or v.endswith(".csv") or "/" in v
            ):
                candidates.append(v)

    _scan(config)
    for section in ("data", "paths"):
        sub = config.get(section)
        if isinstance(sub, dict):
            _scan(sub)

    seen: set[str] = set()
    results: list[tuple[str, Path]] = []
    for raw in candidates:
        if raw in seen:
            continue
        seen.add(raw)
        p = Path(raw)
        resolved = p if p.is_absolute() else workspace / p
        results.append((raw, resolved))
    return results


def run_reality_check(
    experiment_dir: Path,
    workspace: Path,
    time_limit_seconds: float | None = None,
) -> ValidationReport:
    """
    Run a reality check on a small slice of real data.

    This is a principle-based validation framework:
    - Checks data availability
    - Validates train/test separation is possible
    - Estimates timing feasibility
    - Calls experiment's domain-specific validation if available

    Args:
        experiment_dir: Path to experiment directory (e.g., experiments/{name}/)
        workspace: Path to workspace root
        time_limit_seconds: Time limit for the full job (e.g., 21600 = 6 hours)

    Returns:
        ValidationReport with errors, warnings, and timing estimate
    """
    report = ValidationReport(time_limit_seconds=time_limit_seconds)

    # Lazy imports — surface missing runtime deps as a validation error
    # instead of an ImportError at module load time.
    try:
        import pandas as pd
        import yaml
    except ImportError as e:
        report.errors.append(f"Reality check requires pandas and pyyaml: {e}")
        return report

    # Load config
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        report.errors.append(f"config.yaml not found at {config_path}")
        return report

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        report.errors.append(f"Failed to load config.yaml: {e}")
        return report

    # yaml.safe_load returns None for empty files and non-dict for scalar/list
    # documents; subsequent .get(...) calls would blow up on either.
    if config is None:
        report.errors.append(f"config.yaml is empty: {config_path}")
        return report
    if not isinstance(config, dict):
        report.errors.append(
            f"config.yaml must be a mapping, got {type(config).__name__}"
        )
        return report

    # 1. Check data availability
    # Collect ALL file paths from config — workers use many different key names
    # (data_path, panel_path, ibes_path, rt_path, etf_path, paths.ibes, etc.)
    data_config = config.get("data", {}) if isinstance(config.get("data"), dict) else {}

    all_data_paths = _collect_file_paths(config, workspace)

    # The "panel path" is the processed data file the experiment actually trains on.
    # Column checks only apply to this file — raw source files (ibes_path, rt_path, etc.)
    # have different columns that the experiment transforms during processing.
    panel_path_str = (
        data_config.get("data_path")
        or config.get("data_path")
        or data_config.get("panel_path")
        or config.get("panel_path")
    )

    # Extract column names from config — check multiple locations workers use
    columns_config = config.get("columns", {}) if isinstance(config.get("columns"), dict) else {}
    date_col = (
        data_config.get("date_col")
        or columns_config.get("date")
        or data_config.get("date")
    )
    id_col = (
        data_config.get("id_col")
        or data_config.get("ticker_col")
        or columns_config.get("id")
    )

    df_sample = None
    total_rows = 0

    if all_data_paths:
        # Check every referenced file exists
        for rel_path, full_path in all_data_paths:
            if not full_path.exists():
                report.errors.append(
                    f"Data file not found: {rel_path} "
                    f"(resolved to {full_path})"
                )

        if report.errors:
            return report  # Can't proceed with missing files

        # Pick the file to sample for deeper checks:
        # Prefer the explicit panel path; fall back to first parquet found.
        sample_path = None
        if panel_path_str:
            p = Path(panel_path_str)
            sample_path = p if p.is_absolute() else workspace / p
        else:
            for _, full_path in all_data_paths:
                if full_path.suffix == ".parquet":
                    sample_path = full_path
                    break

        if sample_path and sample_path.exists():
            try:
                suffix = sample_path.suffix.lower()
                if suffix == ".parquet":
                    import pyarrow.parquet as pq

                    pf = pq.ParquetFile(sample_path)
                    if pf.metadata.num_rows == 0:
                        report.errors.append(f"Data file {sample_path.name} is empty")
                        return report

                    # Read first row group only (typically 100k–1M rows)
                    table = pf.read_row_group(0)
                    df_sample = table.to_pandas()
                    if len(df_sample) > 5000:
                        df_sample = df_sample.head(5000)

                    total_rows = pf.metadata.num_rows
                    all_columns = set(table.column_names)
                elif suffix == ".csv":
                    # CSVs can't be sliced by row group; read a bounded slice
                    # via nrows so we never pull the entire file into memory.
                    df_sample = pd.read_csv(sample_path, nrows=5000)
                    all_columns = set(df_sample.columns)
                    # total_rows is unknown without a full scan; leave as the
                    # sample size so downstream estimates fall back to lower
                    # bounds rather than guessing wrong.
                    total_rows = len(df_sample)
                    if total_rows == 0:
                        report.errors.append(f"Data file {sample_path.name} is empty")
                        return report
                else:
                    report.warnings.append(
                        f"Skipping deep data checks — unsupported sample file "
                        f"type {suffix or '(none)'} for {sample_path.name}"
                    )
                    all_columns = set()

                # Column checks only apply to the explicit panel file,
                # not raw source files whose columns get transformed.
                if panel_path_str and all_columns:
                    if date_col and date_col not in all_columns:
                        report.errors.append(
                            f"Data missing required date column: {date_col}"
                        )
                    if id_col and id_col not in all_columns:
                        report.errors.append(
                            f"Data missing required id/ticker column: {id_col}"
                        )

            except Exception as e:
                report.errors.append(f"Failed to read data file {sample_path}: {e}")
                return report
    else:
        report.warnings.append("No data file paths found in config; skipping data checks")

    # 2. Generic principle: Validate train/test separation is feasible
    validation_config = config.get("validation", {}) if isinstance(config.get("validation"), dict) else {}
    train_days = validation_config.get("train_days", 0)
    test_days = validation_config.get("test_days", 0)
    embargo_days = validation_config.get("embargo_days", 0)
    label_horizon_days = validation_config.get("label_horizon_days", 0)

    if train_days == 0 or test_days == 0:
        report.warnings.append(
            "train_days or test_days not specified; skipping split validation"
        )
    else:
        min_required_days = train_days + test_days + embargo_days + label_horizon_days

        # Check if data covers the requested date range
        if df_sample is not None and date_col and date_col in df_sample.columns:
            start_date = data_config.get("start")
            end_date = data_config.get("end")

            if start_date and end_date:
                try:
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    requested_days = (end_dt - start_dt).days

                    if requested_days < min_required_days:
                        report.errors.append(
                            f"Requested date range ({start_date} to {end_date}) "
                            f"has {requested_days} days, but train/test/embargo/label "
                            f"requires at least {min_required_days} days"
                        )

                    # Check data actually exists in this range
                    df_dates = pd.to_datetime(df_sample[date_col]).unique()
                    df_dates_sorted = sorted(df_dates)
                    actual_start = df_dates_sorted[0] if len(df_dates_sorted) > 0 else None
                    actual_end = df_dates_sorted[-1] if len(df_dates_sorted) > 0 else None

                    if actual_start and start_dt < actual_start:
                        report.warnings.append(
                            f"Config start date {start_date} is before data start "
                            f"{actual_start.date()}; walk-forward will start from "
                            f"first available data"
                        )

                    if actual_end and end_dt > actual_end:
                        report.warnings.append(
                            f"Config end date {end_date} is after data end "
                            f"{actual_end.date()}; will use available data only"
                        )

                    # Check for liquidity/universe filters that might reduce usable data
                    universe_config = config.get("universe", {}) if isinstance(config.get("universe"), dict) else {}
                    if universe_config.get("use_liq_ok"):
                        # Check if liq_ok column exists and has a warm-up period
                        if "liq_ok" in df_sample.columns:
                            first_liquid_date = None
                            df_sample_sorted = df_sample.sort_values(date_col)
                            for date_val in df_sample_sorted[date_col].unique():
                                date_df = df_sample_sorted[df_sample_sorted[date_col] == date_val]
                                if date_df["liq_ok"].sum() > 0:
                                    first_liquid_date = date_val
                                    break

                            if first_liquid_date:
                                first_liquid_dt = pd.to_datetime(first_liquid_date)
                                if first_liquid_dt > start_dt:
                                    gap_days = (first_liquid_dt - start_dt).days
                                    report.warnings.append(
                                        f"Liquidity filter (liq_ok) has {gap_days}-day "
                                        f"warm-up period; first usable date is "
                                        f"{first_liquid_dt.date()}, not {start_date}. "
                                        f"This may reduce OOS window length."
                                    )

                except Exception as e:
                    report.warnings.append(
                        f"Could not validate date range: {e}"
                    )

    # 3. Generic safety check: Look for obvious leakage patterns
    # Check if features include forward returns (common leakage source)
    features_config = config.get("features", {}) if isinstance(config.get("features"), dict) else {}
    feature_cols = []

    # Extract feature columns from various config formats
    if "exog_reals" in features_config:
        feature_cols.extend(features_config["exog_reals"])
    if "feature_cols" in features_config:
        feature_cols.extend(features_config["feature_cols"])
    if "columns" in features_config:
        feature_cols.extend(features_config["columns"])

    # Check for forward-looking patterns
    leaking_features = [
        f for f in feature_cols
        if isinstance(f, str) and (
            f.startswith("fwd_ret") or
            f.startswith("future_") or
            "forward" in f.lower()
        )
    ]

    if leaking_features:
        report.errors.append(
            f"Potential data leakage: Features include forward-looking columns: "
            f"{leaking_features}. Forward returns should be labels, not features. "
            f"If this is intentional (e.g., predicting residuals), document it clearly "
            f"and implement validate_configuration() to suppress this error."
        )

    # Check if allow_return_features=True is used (safety guard override)
    # Scan run_experiment.py for this pattern
    run_experiment_path = experiment_dir / "run_experiment.py"
    if run_experiment_path.exists():
        try:
            run_experiment_code = run_experiment_path.read_text()
            if "allow_return_features=True" in run_experiment_code:
                report.warnings.append(
                    "SAFETY OVERRIDE DETECTED: allow_return_features=True found in "
                    "run_experiment.py. This disables the backtest framework's return "
                    "feature guard. Ensure forward returns are NOT in feature list, "
                    "or document why this override is necessary."
                )
        except Exception:
            pass

    # 4. Domain-specific validation is deliberately NOT executed here.
    # Experiment code (`experiments/<name>/strategy.py`) is LLM-generated and
    # must not be imported in-process on the dispatcher/system runtime, since
    # that would give untrusted code arbitrary execution before any worker
    # sandbox is applied. If domain-specific `validate_configuration` is
    # needed, it should run inside the worker subprocess that the dispatcher
    # already launches for the experiment (which has its own sandboxing and
    # time limits), not as part of the system-side reality check.

    # 5. Generic principle: Estimate timing feasibility via throughput test
    # Run a micro-benchmark: 10 rows, 1 split, 1 epoch
    if not report.errors:  # Only if no blocking errors so far
        try:
            timing_estimate = estimate_runtime(
                experiment_dir,
                workspace,
                config,
                df_sample,
                total_rows=total_rows or None,
            )
            report.estimated_runtime_seconds = timing_estimate
        except Exception as e:
            report.warnings.append(f"Could not estimate runtime: {e}")

    return report


def _extract_model_type(config: dict[str, Any]) -> str:
    """Extract model type from config, trying multiple key paths workers use."""
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict):
        for key in ("model_type", "type", "library"):
            val = model_cfg.get(key, "")
            if val:
                return str(val).lower()
    # Fall back to top-level keys
    for key in ("model_type", "library"):
        val = config.get(key, "")
        if val:
            return str(val).lower()
    return "unknown"


def _extract_training_params(config: dict[str, Any]) -> dict[str, int]:
    """Extract training parameters from config, trying multiple key paths."""
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    training_cfg = config.get("training", {}) if isinstance(config.get("training"), dict) else {}

    epochs = (
        training_cfg.get("epochs")
        or training_cfg.get("max_epochs")
        or model_cfg.get("epochs")
        or config.get("epochs")
        or 10
    )
    batch_size = (
        training_cfg.get("batch_size")
        or model_cfg.get("batch_size")
        or config.get("batch_size")
        or 512
    )
    return {"epochs": int(epochs), "batch_size": int(batch_size)}


def estimate_runtime(
    experiment_dir: Path,
    workspace: Path,
    config: dict[str, Any],
    df_sample: pd.DataFrame | None,
    total_rows: int | None = None,
) -> float | None:
    """Estimate full experiment runtime.

    Delegates to ``experiment_validation.estimate_training_time`` so both
    worker-side (step 5) and system-side (step 9) use identical heuristics.
    """
    try:
        from alpha_lab.experiment_validation import estimate_training_time

        if total_rows is None:
            if df_sample is None or len(df_sample) == 0:
                return None
            total_rows = len(df_sample)

        model_type = _extract_model_type(config)
        params = _extract_training_params(config)

        # Determine library for the estimator
        model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
        library = model_cfg.get("library", "")
        if not library:
            # Infer from model type
            dl_keywords = {"transformer", "lstm", "gru", "tft", "patchtst", "timesnet",
                           "deepar", "nhits", "tcn", "mlp", "autoformer"}
            library = "pytorch" if any(k in model_type for k in dl_keywords) else "lightgbm"

        # Extract feature count from config
        features_cfg = config.get("features", {})
        n_features = 0
        for key in ("exog_reals", "feature_cols", "columns"):
            cols = features_cfg.get(key, [])
            if isinstance(cols, list):
                n_features = max(n_features, len(cols))
        n_features = n_features or 50  # conservative default

        result = estimate_training_time(
            model_type=model_type,
            library=library,
            n_samples=total_rows,
            n_features=n_features,
            n_epochs=params["epochs"],
            batch_size=params["batch_size"],
        )
        return result.get("estimated_seconds")

    except Exception:
        return None


def save_validation_report(report: ValidationReport, experiment_dir: Path) -> None:
    """Save validation report to experiment directory."""
    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    report_path = results_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Save as human-readable text
    report_txt_path = results_dir / "validation_report.txt"
    with open(report_txt_path, "w") as f:
        f.write(report.format())
