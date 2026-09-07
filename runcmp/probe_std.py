"""Standard probe library — shared metric definitions for investigators.

Two investigators fact-checked cleanly while disagreeing on derived numbers
("pytest runs": 41 vs 129), because each re-derived quantities with its own
ad-hoc definitions and the fact-checker only verifies findings against the
investigator's own probes. This module is the fix: the deterministic layer
owns the definitions, and probes import them instead of re-deriving.

Usage inside a probe (RUNCMP_CORPUS / RUNCMP_PACKS env vars are set):

    from runcmp import probe_std as std
    p = std.pack("domain4_traffic/domain4/msml")
    std.stage_decomposition("domain4_traffic/domain4/msml")
    std.shell_pattern_counts("domain4_traffic/domain4/msml", role="implement")

Counting definitions are explicit in the names: a *mention* is a pattern
match anywhere in a command; an *invocation* is the pattern in command
position. Never report a bare "pytest runs" — say which.
"""

from __future__ import annotations

import glob as globmod
import json
import os
import re
from pathlib import Path

# Canonical shell-command classifier (from the d4 stage investigation).
# Pattern semantics: MENTION = re.search over the whole command string.
STANDARD_SHELL_PATTERNS: dict[str, str] = {
    "pytest_mention": r"pytest",
    "smoke_mention": r"(?i)smoke",
    "compile_mention": r"py_compile|compileall",
    "shell_syntax_check": r"bash\s+-n",
    "run_experiment_mention": r"run_experiment\.py",
    "checkpoint_reload_mention": r"(?i)reload|best_model|checkpoint",
    "environment_probe": r"pip |pip3|command -v|which python|python3 -V|site-packages|nvidia-smi",
    "filesystem_inventory": r"(^|[;&|\n ])(find|ls|pwd|wc)\b",
    "write_heredoc": r"cat\s*>|cat\s*>>",
    "git_check": r"git (diff|status)",
}

# INVOCATION = the tool in command position (start of command or after a
# separator), not merely mentioned (e.g. inside a grep or a path).
STANDARD_INVOCATION_PATTERNS: dict[str, str] = {
    "pytest_invocation": r"(^|[;&|\n]\s*)(python3?\s+-m\s+)?pytest\b",
    "pip_invocation": r"(^|[;&|\n]\s*)pip3?\b",
    "nvidia_smi_invocation": r"(^|[;&|\n]\s*)nvidia-smi\b",
}


def _corpus() -> dict:
    return json.load(open(os.environ["RUNCMP_CORPUS"]))


def _packs_dir() -> Path:
    return Path(os.environ["RUNCMP_PACKS"])


def pack(label: str) -> dict:
    """Load a run's evidence pack by its corpus label."""
    fname = label.replace("/", "__") + ".json"
    return json.load(open(_packs_dir() / fname))


def run_record(label: str) -> dict:
    """The corpus registry record for a run (framework, workspace, ...)."""
    for r in _corpus()["runs"]:
        if r["label"] == label:
            return r
    raise KeyError(label)


def definitions() -> dict[str, str]:
    """The deterministic bench's metric id -> definition text."""
    from runcmp.bench import METRICS

    return {m["id"]: m["definition"] for m in METRICS}


def samples(label: str, key: str | None = None):
    """THE raw distribution arrays for one run, straight from its pack.

    ``key`` in {request_bytes, tool_result_bytes, llm_gap_seconds,
    tool_gap_seconds, session_minutes} returns that array; ``None`` returns
    the whole samples dict. Use this instead of ad-hoc extraction: a
    reviewer once mangled these arrays into constants in its own probe and
    then blamed a "preserved sampling limitation" that did not exist
    (2026-08-02). If an array looks degenerate, re-read the pack file
    directly before claiming the corpus is limited.
    """
    s = (pack(label).get("agent_logs") or {}).get("samples") or {}
    return s.get(key) if key else s


def trajectory(label: str) -> list[dict]:
    """THE per-run scored trajectory, straight from the pack.

    Each element: {"n": attempt number, "name": experiment, "value":
    self-reported metric, "best_so_far": running champion}. Use this for
    every progression/attempt-spread/ladder chart — a reviewer once
    plotted garbled values (0.07-0.16 where the pack says 0.022) as
    "self-reported RMSE" after a hand-rolled join (2026-08-02).
    """
    return (pack(label).get("experiments") or {}).get("trajectory") or []


def metrics(label: str) -> dict:
    """THE registry values for one run: metric id -> value, from bench.json.

    These are the numbers bench.md prints — already computed, zero
    recomputation here. Any quantity with an id in ``definitions()`` must be
    QUOTED from this mapping; re-deriving a registry-covered quantity with
    ad-hoc probe code produced rival "same-named" numbers in two reports of
    the same corpus (queue-wait medians, 2026-08-02) and is forbidden. If
    you believe a registry value is wrong, report the discrepancy as a
    finding — never silently substitute your own.
    """
    bench_path = Path(os.environ["RUNCMP_TABLES"]).parent / "bench.json"
    bench = json.load(open(bench_path))
    run = bench["runs"].get(label)
    if run is None:
        raise KeyError(
            f"{label!r} not in bench.json ({sorted(bench['runs'])[:3]}...)")
    return run["metrics"]


def stage_decomposition(label: str) -> dict | None:
    """The bench-standard per-stage sessions/calls/tokens accounting.

    Computed by the same bench code that renders bench.md's stage table —
    THE definition of stage costs; do not re-derive session or call
    boundaries in ad-hoc probe code.
    """
    from runcmp.bench import efficiency_decomposition

    return efficiency_decomposition(pack(label))


def shell_commands(label: str, role: str = "implement") -> list[str]:
    """Every shell_exec command string from a run's worker logs for a role."""
    ws = run_record(label)["workspace"]
    commands: list[str] = []
    for f in sorted(globmod.glob(f"{ws}/logs/worker*{role}*.jsonl")):
        for line in open(f, errors="ignore"):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if x.get("type") == "tool_call" and x.get("name") == "shell_exec":
                try:
                    commands.append(json.loads(x["arguments"]).get("command", ""))
                except (ValueError, TypeError):
                    continue
    return commands


def shell_pattern_counts(
    label: str, role: str = "implement", patterns: dict[str, str] | None = None
) -> dict[str, int]:
    """Standard pattern counts over a role's shell commands.

    Includes both mention- and invocation-style counts so reports can never
    conflate them; total command count under key ``shell_calls``.
    """
    cmds = shell_commands(label, role)
    pats = dict(STANDARD_SHELL_PATTERNS)
    pats.update(STANDARD_INVOCATION_PATTERNS)
    if patterns:
        pats.update(patterns)
    out = {"shell_calls": len(cmds)}
    for name, pat in pats.items():
        rx = re.compile(pat)
        out[name] = sum(bool(rx.search(c)) for c in cmds)
    return out
