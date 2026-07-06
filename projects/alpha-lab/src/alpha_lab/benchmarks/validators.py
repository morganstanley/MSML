"""Post-run workspace validation for benchmark runners.

After a workspace has been run through Alpha Lab, ``validate_workspace``
walks the resulting filesystem and checks that every artifact the pipeline
was supposed to produce is actually there and well-formed.

This catches the failure mode where ``python run.py`` exits 0 but skipped
a phase, produced an empty ``learnings.md``, or left the experiment DB
without any completed rows — silent regressions that exit code alone
won't surface.

The validator is phase-aware: it reads ``config.json``'s ``pipeline.phases``
and only asserts the artifacts owned by phases that were configured to run.
Phase 2 file paths come from the adapter manifest (``experiment.framework_dir``,
``experiment.framework_files``, and ``phase2_review_file``) — they are not
hardcoded here, since they vary per adapter.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path


# Substring scanned for in JSONL logs; the agent emits this exact prefix on
# tool dispatch failures (see src/alpha_lab/agent.py::_handle_tool_calls).
TOOL_ERROR_MARKER = "[TOOL ERROR]"

# Terminal-success statuses in ExperimentDB's kanban schema
# (see KANBAN_COLUMNS in src/alpha_lab/experiment_db.py). An experiment is
# considered fully done once it reaches ``done`` or ``analyzed``.
EXPERIMENT_COMPLETED_STATUSES = ("done", "analyzed")

# Statuses indicating an experiment is still being worked on. ``finished`` is
# included because the analyzer may have left it pending retry — rows in that
# state when the dispatcher exits are stuck mid-pipeline.
EXPERIMENT_IN_FLIGHT_STATUSES = (
    "to_implement", "implemented", "checked", "queued", "running", "finished",
)


@dataclass
class Check:
    """One assertion in the validation result."""

    name: str
    ok: bool
    detail: str = ""


@dataclass
class ValidationResult:
    """Structured validator output ready to be stamped into the manifest."""

    ok: bool
    checks: list[Check] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}


def validate_workspace(workspace: Path | str) -> ValidationResult:
    """Run every applicable artifact check on ``workspace``.

    Returns a :class:`ValidationResult` whose ``ok`` is True only if every
    check passed. Missing ``config.json`` is itself a failure — without it
    we can't decide which phase artifacts to expect.
    """
    ws = Path(workspace)
    checks: list[Check] = []

    config_path = ws / "config.json"
    if not config_path.is_file():
        checks.append(Check(
            name="config.json present",
            ok=False,
            detail=f"missing at {config_path}",
        ))
        return ValidationResult(ok=False, checks=checks)
    checks.append(Check(name="config.json present", ok=True))

    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        checks.append(Check(name="config.json parses", ok=False, detail=str(e)))
        return ValidationResult(ok=False, checks=checks)
    checks.append(Check(name="config.json parses", ok=True))

    phases = _resolve_phases(config)

    # Phase 0 always runs implicitly (adapter resolution) when any later
    # phase needs it; we always assert the adapter is well-formed.
    adapter_checks, manifest = _check_adapter(ws)
    checks.extend(adapter_checks)

    if "phase1" in phases:
        checks.extend(_check_phase1(ws))
    if "phase2" in phases:
        # Only run Phase 2 checks if the adapter manifest was usable;
        # otherwise we can't resolve framework_dir / review_file paths and
        # the adapter failure has already been recorded above.
        if manifest is not None:
            checks.extend(_check_phase2(ws, manifest))
    if "phase3" in phases:
        checks.extend(_check_phase3(ws))

    checks.extend(_check_logs(ws))

    return ValidationResult(
        ok=all(c.ok for c in checks),
        checks=checks,
    )


# ---------------------------------------------------------------------------
# Per-phase checks
# ---------------------------------------------------------------------------


def _resolve_phases(config: dict) -> set[str]:
    """Resolve which pipeline phases the run was configured to execute.

    Mirrors ``TaskConfig.pipeline.phases`` defaulting to ``["phase1"]``
    when missing/empty — otherwise a config that relies on the default
    would skip Phase 1 artifact checks and report a false success.
    """
    pipeline = config.get("pipeline") or {}
    phases = pipeline.get("phases")
    if not phases:
        return {"phase1"}
    return {str(p) for p in phases}


def _check_adapter(ws: Path) -> tuple[list[Check], dict | None]:
    """Return (checks, parsed manifest dict or None on failure).

    The parsed manifest is forwarded to ``_check_phase2`` so the framework
    directory and review filename can be resolved from the adapter rather
    than hardcoded.

    Defensive: ``is_file`` (not just ``exists``) so a *directory* named
    ``manifest.json`` is treated as missing instead of crashing; broad
    exception handling on read so transient IO failures surface as a
    failed check rather than an uncaught exception in the runner.
    """
    out: list[Check] = []
    manifest_path = ws / "adapter" / "manifest.json"
    out.append(Check(name="adapter/manifest.json exists", ok=manifest_path.is_file()))
    if not manifest_path.is_file():
        return out, None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        out.append(Check(name="adapter manifest parses", ok=False, detail=str(e)))
        return out, None
    out.append(Check(name="adapter manifest parses", ok=True))

    # See save_adapter() in src/alpha_lab/adapter_loader.py for the schema:
    # domain_name + metric{primary_metric,...} + experiment{framework_dir,...}.
    required = {"domain_name", "metric", "experiment"}
    missing = sorted(required - set(manifest))
    out.append(Check(
        name="adapter manifest has required keys",
        ok=not missing,
        detail=f"missing: {missing}" if missing else "domain_name + metric + experiment",
    ))

    metric = manifest.get("metric") or {}
    out.append(Check(
        name="adapter metric has primary_metric",
        ok=bool(metric.get("primary_metric")),
        detail=f"primary_metric={metric.get('primary_metric')!r}",
    ))

    return out, manifest


def _check_phase1(ws: Path) -> list[Check]:
    out: list[Check] = []
    learnings = ws / "learnings.md"
    # Semantic emptiness: a whitespace-only file fails too (matches
    # detect_phase1_complete() in src/alpha_lab/pipeline.py).
    learnings_ok = False
    detail = "missing"
    if learnings.exists():
        text = learnings.read_text()
        learnings_ok = bool(text.strip())
        detail = f"{len(text.strip())} non-whitespace chars"
    out.append(Check(name="learnings.md non-empty", ok=learnings_ok, detail=detail))

    data_report = ws / "data_report"
    md_files = sorted(data_report.glob("*.md")) if data_report.is_dir() else []
    out.append(Check(
        name="data_report/*.md produced",
        ok=len(md_files) >= 1,
        detail=f"{len(md_files)} file(s)",
    ))
    return out


def _check_phase2(ws: Path, manifest: dict) -> list[Check]:
    """Validate Phase 2 artifacts.

    ``manifest`` is required (not Optional) — Phase 2 paths come straight
    from it (``experiment.framework_dir`` / ``experiment.framework_files``
    / ``phase2_review_file``).  The caller (``validate_workspace``) must
    only invoke this once the adapter manifest has been confirmed
    well-formed; passing ``None`` here is a programming error and we'd
    rather it raise than silently fall back to hardcoded defaults.
    """
    out: list[Check] = []

    experiment = manifest.get("experiment") or {}
    framework_dir_name = experiment.get("framework_dir") or "backtest"
    framework_files = list(experiment.get("framework_files") or [])
    review_file_name = manifest.get("phase2_review_file") or "review.md"

    # Reject manifest values that would point outside the workspace
    # (absolute path or ``..`` traversal). A malicious or buggy adapter
    # could otherwise make the validator read arbitrary files on the host.
    for label, value in (
        ("framework_dir", framework_dir_name),
        ("phase2_review_file", review_file_name),
    ):
        if not _is_safe_relative(value):
            out.append(Check(
                name=f"adapter {label} is workspace-relative",
                ok=False,
                detail=f"refused unsafe path {value!r}",
            ))
            return out
    safe_framework_files = []
    for f in framework_files:
        if _is_safe_relative(f):
            safe_framework_files.append(f)
        else:
            out.append(Check(
                name="adapter framework_files entry is workspace-relative",
                ok=False,
                detail=f"refused unsafe path {f!r}",
            ))
            return out
    framework_files = safe_framework_files

    framework = ws / framework_dir_name
    out.append(Check(
        name=f"{framework_dir_name}/ exists",
        ok=framework.is_dir(),
        detail=f"adapter framework_dir={framework_dir_name!r}",
    ))
    if not framework.is_dir():
        return out

    # Verify the files the adapter says should exist (more accurate than an
    # arbitrary "at least 2 .py files" count).  If the adapter didn't declare
    # ``framework_files``, fall back to a lenient ">= 1 .py" check.
    if framework_files:
        missing = [f for f in framework_files if not (framework / f).exists()]
        out.append(Check(
            name=f"{framework_dir_name} has adapter-declared files",
            ok=not missing,
            detail=(
                f"missing: {missing}" if missing
                else f"all {len(framework_files)} files present"
            ),
        ))
    else:
        py_files = sorted(framework.glob("*.py"))
        out.append(Check(
            name=f"{framework_dir_name} has at least one Python file",
            ok=len(py_files) >= 1,
            detail=f"{len(py_files)} .py files",
        ))

    test_files = sorted(framework.glob("tests/test_*.py"))
    out.append(Check(
        name=f"{framework_dir_name}/tests/test_*.py present",
        ok=len(test_files) >= 1,
        detail=f"{len(test_files)} test file(s)",
    ))

    review = framework / review_file_name
    out.append(Check(
        name=f"{framework_dir_name}/{review_file_name} exists",
        ok=review.exists(),
    ))
    if review.exists():
        # Reuse the pipeline's verdict extractor so the validator can't
        # disagree with the supervisor about whether a Phase 2 review.md
        # says PASS.  Lazy-import to avoid pulling in the agent loop's
        # OTel dependencies just to validate a workspace.
        from alpha_lab.pipeline import _extract_verdict

        verdict = _extract_verdict(review.read_text())
        out.append(Check(
            name=f"{framework_dir_name}/{review_file_name} verdict is PASS",
            ok=verdict == "PASS",
            detail=verdict if verdict != "PASS" else "",
        ))
    return out


def _is_safe_relative(value: str) -> bool:
    """True if ``value`` is a workspace-relative path with no ``..`` traversal.

    Absolute paths and any segment equal to ``..`` are rejected.  Empty
    strings are also rejected as nonsensical here.
    """
    if not isinstance(value, str) or not value:
        return False
    p = Path(value)
    if p.is_absolute():
        return False
    if any(part == ".." for part in p.parts):
        return False
    return True


def _check_phase3(ws: Path) -> list[Check]:
    out: list[Check] = []
    db_path = ws / "experiments.db"
    out.append(Check(name="experiments.db exists", ok=db_path.exists()))
    if not db_path.exists():
        return out

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            cur.execute("SELECT status, COUNT(*) FROM experiments GROUP BY status")
            counts: dict[str, int] = {row[0]: row[1] for row in cur.fetchall()}
        finally:
            conn.close()
    except sqlite3.Error as e:
        out.append(Check(name="experiments.db queryable", ok=False, detail=str(e)))
        return out

    total = sum(counts.values())
    completed = sum(counts.get(s, 0) for s in EXPERIMENT_COMPLETED_STATUSES)
    in_flight = sum(counts.get(s, 0) for s in EXPERIMENT_IN_FLIGHT_STATUSES)

    out.append(Check(
        name="experiments.db has rows",
        ok=total >= 1,
        detail=f"{total} total: {dict(sorted(counts.items()))}",
    ))
    out.append(Check(
        name="at least one experiment reached done/analyzed",
        ok=completed >= 1,
        detail=f"{completed} in {EXPERIMENT_COMPLETED_STATUSES}",
    ))
    out.append(Check(
        name="no experiments stuck mid-pipeline",
        ok=in_flight == 0,
        detail=f"{in_flight} still in {EXPERIMENT_IN_FLIGHT_STATUSES}" if in_flight else "",
    ))
    return out


def _check_logs(ws: Path) -> list[Check]:
    out: list[Check] = []
    logs_dir = ws / "logs"
    if not logs_dir.is_dir():
        out.append(Check(name="logs/ exists", ok=False))
        return out
    out.append(Check(name="logs/ exists", ok=True))

    jsonl_files = sorted(logs_dir.glob("*.jsonl"))
    out.append(Check(
        name="logs/*.jsonl produced",
        ok=len(jsonl_files) >= 1,
        detail=f"{len(jsonl_files)} file(s)",
    ))

    # Stream each file line-by-line and stop at the first marker hit per file.
    # JSONL logs can grow to hundreds of MB on long Phase 3 runs; reading them
    # all into memory would be wasteful.  Failure to open / read a file is
    # recorded per-file as its own failed Check so the error is visible —
    # don't silently treat an unreadable log as "no errors found".
    offenders: list[str] = []
    for f in jsonl_files:
        try:
            with f.open("r", errors="replace") as fh:
                contains = any(TOOL_ERROR_MARKER in line for line in fh)
        except OSError as e:
            out.append(Check(
                name=f"log readable: {f.name}",
                ok=False,
                detail=f"{type(e).__name__}: {e}",
            ))
            continue
        if contains:
            offenders.append(f.name)
    out.append(Check(
        name=f"no {TOOL_ERROR_MARKER} strings in logs",
        ok=not offenders,
        detail=f"errors in: {offenders}" if offenders else "",
    ))
    return out
