"""Filesystem layout for the Conductor's meta-state.

All Conductor outputs live under ``<workspace>/meta/``. Other agents
(strategist, workers, reporter, supervisor, phase 1/2 builders) read
``meta/directives.md`` and ``meta/annotations.json`` at the top of every
turn — those are the cross-agent communication channel. The user's
write-only channel into the system is ``meta/instructions/from_user.md``;
the Conductor's write-only-to-the-user channel is ``meta/notes_to_user.md``.

Conventions enforced by this module:

* All paths are derived from ``workspace`` — no hard-coded prefixes.
* :func:`ensure_meta_layout` is idempotent and never overwrites existing
  files. It only creates missing directories and writes dummy content for
  files that don't exist yet, so the user can re-run pipelines on the
  same workspace without losing state.
* ``meta/instructions/from_user.md`` is auto-created on first run with a
  short comment that doubles as documentation for the user.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("alpha_lab.meta_layout")


# ---------------------------------------------------------------------------
# Path helpers — all pure, all derived from workspace.
# ---------------------------------------------------------------------------

def meta_dir(workspace: str | Path) -> Path:
    return Path(workspace) / "meta"


def directives_path(workspace: str | Path) -> Path:
    return meta_dir(workspace) / "directives.md"


def annotations_path(workspace: str | Path) -> Path:
    return meta_dir(workspace) / "annotations.json"


def notes_to_user_path(workspace: str | Path) -> Path:
    return meta_dir(workspace) / "notes_to_user.md"


def notes_inbox_path(workspace: str | Path) -> Path:
    """Combined inbox where strategist and workers leave notes for the
    Conductor. Single file partitioned by header (e.g. ``## strategist —
    2026-05-09 14:30``) — simpler than per-role files."""
    return meta_dir(workspace) / "notes_inbox.md"


def meta_log_jsonl_path(workspace: str | Path) -> Path:
    return meta_dir(workspace) / "meta_log.jsonl"


def meta_log_md_path(workspace: str | Path) -> Path:
    return meta_dir(workspace) / "meta_log.md"


def instructions_dir(workspace: str | Path) -> Path:
    return meta_dir(workspace) / "instructions"


def from_user_path(workspace: str | Path) -> Path:
    return instructions_dir(workspace) / "from_user.md"


def ack_path(workspace: str | Path) -> Path:
    return instructions_dir(workspace) / "ack.md"


def last_seen_path(workspace: str | Path) -> Path:
    """Conductor bookkeeping — hash of the last from_user.md content the
    conductor saw. Lets it diff against the current content each turn."""
    return instructions_dir(workspace) / ".last_seen"


def throttle_path(workspace: str | Path) -> Path:
    return meta_dir(workspace) / "throttle.json"


def scratch_dir(workspace: str | Path) -> Path:
    """Where the conductor writes one-shot Python analysis scripts. Each
    script gets a timestamp-prefixed filename so the audit trail is preserved."""
    return meta_dir(workspace) / "scratch"


def backups_dir(workspace: str | Path) -> Path:
    """Pre-overwrite backups. The conductor's ``delete_path`` and phase
    rewind tools always copy here before destructive operations."""
    return meta_dir(workspace) / "backups"


def directive_acks_path(workspace: str | Path) -> Path:
    """Append-only log of directive acknowledgements. When a strategist
    or worker acts on a one-shot or per-experiment directive, it appends
    a line here so that later turns / other workers know not to repeat
    the work. Each line is JSON: ``{ts, directive_id, actor_role,
    actor_id, action}``."""
    return meta_dir(workspace) / "directive_acks.jsonl"


def directive_retirements_path(workspace: str | Path) -> Path:
    """Append-only log of directives the Conductor has explicitly retired.

    Each line is JSON: ``{ts, directive_id, reason}``. Readers
    (``directives_for_role``, the digest builder) filter out any
    directive id appearing here so a retired directive stops being
    injected into downstream agents' prompts.

    Retirement is one-way and permanent: once a directive is retired,
    issuing the *same* directive again requires a new id. This forces
    the Conductor to own the lifecycle of every in-force directive — if
    a directive isn't applicable anymore, retire it; never let it just
    sit unread."""
    return meta_dir(workspace) / "directive_retirements.jsonl"


def token_usage_path(workspace: str | Path) -> Path:
    """Centralized per-API-call token usage log.

    Every LLM API response (across all providers and all agent roles —
    openai, grok, bedrock; strategist, workers, conductor, supervisor,
    reporter, phase agents) appends one JSON line here. Each line has
    ``ts``, ``log_name`` (the agent role), ``provider``, ``model``,
    ``input_tokens``, ``output_tokens``, ``reasoning_tokens``, and
    ``total_tokens``. The file is append-only across the run; aggregation
    is done by reading and summing.
    """
    return meta_dir(workspace) / "token_usage.jsonl"


def record_token_usage(
    workspace: str | Path,
    *,
    log_name: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    usage_raw: dict | None = None,
) -> None:
    """Append one usage entry to ``meta/token_usage.jsonl``.

    Best-effort: failures (missing dir, disk full, etc.) log a warning but
    never raise — token tracking is observability, never a hard dependency
    of the main run.
    """
    import time as _time
    path = token_usage_path(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": _time.time(),
            "log_name": log_name,
            "provider": provider,
            "model": model,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "reasoning_tokens": int(reasoning_tokens or 0),
            "cache_read_input_tokens": int(cache_read_tokens or 0),
            "cache_write_input_tokens": int(cache_write_tokens or 0),
            "total_tokens": int(input_tokens or 0) + int(output_tokens or 0),
            # Provider usage object verbatim. Kept because "input_tokens"
            # means different things per provider (Anthropic/Bedrock exclude
            # cache reads from it, OpenAI includes them), and because the
            # cache breakdown -- 5-minute vs 1-hour ephemeral entries -- only
            # exists here. Cost cannot be recomputed later without it.
            "usage_raw": usage_raw or {},
        }
        # Open in append mode — POSIX guarantees small atomic appends so
        # multiple agent threads writing concurrently don't tear lines.
        with open(path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as e:
        logger.warning("token_usage write failed (%s): %s", path, e)


def token_usage_summary_path(workspace: str | Path) -> Path:
    """Aggregated counterpart of ``meta/token_usage.jsonl``.

    Rewritten (not appended) on each refresh — one JSONL line per agent
    role plus a final ``GRAND_TOTAL`` line. Lets the user inspect the
    cumulative tally via ``cat meta/token_usage_summary.jsonl`` without
    needing to re-read and aggregate the per-call log themselves.
    """
    return meta_dir(workspace) / "token_usage_summary.jsonl"


def _role_bucket(log_name: str) -> str:
    """Coarse role bucket for a per-call ``log_name``.

    Mirrors the bucketing used by the one-off tally script for the prior
    runs (so cross-run tallies and live summaries use the same labels).
    """
    if log_name == "strategist":
        return "strategist"
    if log_name == "conversation":
        return "phase1_explore"
    if log_name.startswith("conductor"):
        return "conductor"
    if log_name.startswith("supervisor"):
        return "supervisor"
    if log_name.startswith("reporter"):
        return "reporter"
    if log_name.startswith("phase0_customize"):
        return "phase0_customize"
    if log_name.startswith("phase2_builder"):
        return "phase2_builder"
    if log_name.startswith("phase2_critic"):
        return "phase2_critic"
    if log_name.startswith("phase2_tester"):
        return "phase2_tester"
    if log_name.startswith("worker_") and "_implement_" in log_name:
        return "worker_implement"
    if log_name.startswith("worker_") and "_analyze_" in log_name:
        return "worker_analyze"
    if log_name.startswith("worker_") and "_fix_" in log_name:
        return "worker_fix"
    return "other"


def refresh_token_usage_summary(workspace: str | Path) -> None:
    """Recompute ``meta/token_usage_summary.jsonl`` from the per-call log.

    Reads every line of ``meta/token_usage.jsonl``, aggregates by role
    bucket, and writes one JSONL line per role plus a final
    ``GRAND_TOTAL`` row. Atomic tmp+rename so concurrent readers always
    see consistent state.

    Best-effort: returns silently on read/parse/write errors so token
    accounting never blocks the dispatcher's hot path.
    """
    import time as _time
    src = token_usage_path(workspace)
    if not src.exists():
        return
    buckets: dict[str, dict[str, int]] = {}
    try:
        with open(src, errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                role = _role_bucket(d.get("log_name", "") or "")
                b = buckets.setdefault(role, {
                    "turns": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_write_input_tokens": 0,
                })
                b["turns"] += 1
                b["input_tokens"]     += int(d.get("input_tokens") or 0)
                b["output_tokens"]    += int(d.get("output_tokens") or 0)
                b["reasoning_tokens"] += int(d.get("reasoning_tokens") or 0)
                b["cache_read_input_tokens"]  += int(d.get("cache_read_input_tokens") or 0)
                b["cache_write_input_tokens"] += int(d.get("cache_write_input_tokens") or 0)
    except OSError as e:
        logger.warning("token_usage_summary read failed (%s): %s", src, e)
        return

    grand = {"turns": 0, "input_tokens": 0, "output_tokens": 0,
             "reasoning_tokens": 0, "cache_read_input_tokens": 0,
             "cache_write_input_tokens": 0}
    for b in buckets.values():
        for k in grand:
            grand[k] += b[k]

    dst = token_usage_summary_path(workspace)
    ts = _time.time()
    lines_out: list[str] = []
    # Per-role lines, descending by input_tokens (largest cost first).
    for role, b in sorted(buckets.items(), key=lambda kv: -kv[1]["input_tokens"]):
        rec = {
            "role": role,
            "turns": b["turns"],
            "input_tokens": b["input_tokens"],
            "output_tokens": b["output_tokens"],
            "reasoning_tokens": b["reasoning_tokens"],
            "cache_read_input_tokens": b["cache_read_input_tokens"],
            "cache_write_input_tokens": b["cache_write_input_tokens"],
            "total_tokens": b["input_tokens"] + b["output_tokens"],
            "last_updated_ts": ts,
        }
        lines_out.append(json.dumps(rec))
    # Grand total at the bottom so a tail/cat naturally shows it last.
    lines_out.append(json.dumps({
        "role": "GRAND_TOTAL",
        "turns": grand["turns"],
        "input_tokens": grand["input_tokens"],
        "output_tokens": grand["output_tokens"],
        "reasoning_tokens": grand["reasoning_tokens"],
        "cache_read_input_tokens": grand["cache_read_input_tokens"],
        "cache_write_input_tokens": grand["cache_write_input_tokens"],
        "total_tokens": grand["input_tokens"] + grand["output_tokens"],
        "last_updated_ts": ts,
    }))

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text("\n".join(lines_out) + "\n")
        tmp.replace(dst)
    except OSError as e:
        logger.warning("token_usage_summary write failed (%s): %s", dst, e)


def run_state_path(workspace: str | Path) -> Path:
    """JSON file persisting Phase-3 run-level state across process restarts.

    Schema: ``{"dispatcher_start_ts": <unix float>, ...}``.

    The dispatcher writes ``dispatcher_start_ts`` once on the first
    ``_init_log`` of a process; on restart the existing value is preserved
    (so the Conductor's ``min_runtime_hours`` floor measures wall-clock from
    the original boot, not from the most recent restart). The
    ``request_run_end`` Conductor tool reads this to enforce its floor.
    """
    return meta_dir(workspace) / "run_state.json"


def read_dispatcher_start_ts(workspace: str | Path) -> float | None:
    """Return the dispatcher's first-boot wall-clock timestamp, or None if
    the run-state file has not been written yet.

    Tolerant of missing / corrupt JSON — used by the Conductor's
    ``request_run_end`` tool and by retrospective audits."""
    p = run_state_path(workspace)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    ts = raw.get("dispatcher_start_ts")
    if isinstance(ts, (int, float)) and ts > 0:
        return float(ts)
    return None


def write_dispatcher_start_ts(workspace: str | Path, ts: float) -> None:
    """Persist the dispatcher's first-boot timestamp. Idempotent: if the
    file already has a valid ts, do not overwrite (so restarts don't reset
    the wall-clock floor)."""
    existing = read_dispatcher_start_ts(workspace)
    if existing is not None:
        return
    md = meta_dir(workspace)
    md.mkdir(parents=True, exist_ok=True)
    p = run_state_path(workspace)
    try:
        p.write_text(json.dumps({"dispatcher_start_ts": ts}) + "\n")
    except OSError as e:
        logger.warning("Failed to write %s: %s", p, e)


def steer_state_path(workspace: str | Path) -> Path:
    """JSON file persisting per-phase steer audit state across process
    restarts. Schema: ``{phaseN: {"hash": "<sha1>", "ts": <unix>}}``.

    Used by ``run.py`` to skip an upstream-phase Conductor turn when
    the corresponding phase's artifacts are bit-identical to the last
    audited state — prevents the expensive opus-at-max-reasoning
    no-action turn from firing on every restart (it costs ~$10/turn
    and adds 5-15 min of wall-clock latency for zero new information)."""
    return meta_dir(workspace) / "steer_state.json"


# ---------------------------------------------------------------------------
# Defaults written by ensure_meta_layout for first-time runs.
# ---------------------------------------------------------------------------

DEFAULT_DIRECTIVES_HEADER = """\
# Conductor directives

Conductor-issued directives appear below this header, most recent at the top.
Each directive is for a specific role (`strategist`, `worker`, `reporter`,
`supervisor`, or `all`) and other agents read this file at the top of their
next turn.

The Conductor maintains this file. Do not edit it by hand — the Conductor
overwrites it each turn from its in-memory plan. Use
`meta/instructions/from_user.md` if you want to influence Conductor decisions.

(no directives yet)
"""

DEFAULT_ANNOTATIONS = "{}\n"  # exp_id (str) -> label

DEFAULT_NOTES_TO_USER_HEADER = """\
# Conductor notes to user

The Conductor appends here. The user is not expected to read these promptly —
they are a record of what the Conductor observed and concluded over the run.
Reverse-chronological; most recent at the top of each session block.

"""

DEFAULT_NOTES_INBOX_HEADER = """\
# Notes inbox (other agents → Conductor)

Strategist and workers append here when they want to flag something to the
Conductor. The Conductor reads this at the top of every turn. Each entry
should start with a `## <role> — <iso timestamp>` header.

"""

DEFAULT_FROM_USER = """\
# Instructions from user to Conductor

# This is the ONLY place to give the Conductor instructions for this run.
# Write your guidance here in plain text — replace this comment block, or
# add free-form text below it. The Conductor reads this file at the top of
# every turn and treats whatever you write as authoritative.
#
# Use it for:
#   - run-level baseline guidance written before launch (e.g.
#     exploration / exploitation policy, mechanism classes to prefer or
#     avoid, evaluation slices to prioritize, leakage rules)
#   - mid-run nudges (e.g. "please prioritize cold-client experiments",
#     "the goal has shifted to won-only P@5", "undo the phase 2 rewind")
#
# Empty file = no user instructions. The Conductor will fall back to its
# built-in defaults and the run's `description` and `target` fields.
#
# The system never waits for you — write here at any time, the Conductor
# picks it up on its next turn (after each milestone, plus a slow timer).
"""

DEFAULT_ACK = """\
# Conductor acknowledgements of user instructions

The Conductor appends here when it has read and translated a new user
instruction from `from_user.md`. Most recent at the top.

"""

DEFAULT_THROTTLE = '{"gpu": "none", "cpu": "none"}\n'


# ---------------------------------------------------------------------------
# Bootstrap.
# ---------------------------------------------------------------------------

def ensure_meta_layout(workspace: str | Path) -> Path:
    """Create the ``meta/`` directory tree and dummy files if missing.

    Idempotent. Existing files are never overwritten — re-running on a
    populated workspace preserves all state. Returns the path to ``meta/``
    so callers can chain.
    """
    md = meta_dir(workspace)
    md.mkdir(parents=True, exist_ok=True)

    instructions_dir(workspace).mkdir(parents=True, exist_ok=True)
    scratch_dir(workspace).mkdir(parents=True, exist_ok=True)
    backups_dir(workspace).mkdir(parents=True, exist_ok=True)

    # Files: only write defaults if missing. Never overwrite.
    _write_if_missing(directives_path(workspace), DEFAULT_DIRECTIVES_HEADER)
    _write_if_missing(annotations_path(workspace), DEFAULT_ANNOTATIONS)
    _write_if_missing(notes_to_user_path(workspace), DEFAULT_NOTES_TO_USER_HEADER)
    _write_if_missing(notes_inbox_path(workspace), DEFAULT_NOTES_INBOX_HEADER)
    _write_if_missing(meta_log_jsonl_path(workspace), "")
    _write_if_missing(meta_log_md_path(workspace), "# Conductor decision log\n\n(no decisions yet)\n")
    _write_if_missing(directive_acks_path(workspace), "")
    _write_if_missing(directive_retirements_path(workspace), "")
    _write_if_missing(from_user_path(workspace), DEFAULT_FROM_USER)
    _write_if_missing(ack_path(workspace), DEFAULT_ACK)
    _write_if_missing(throttle_path(workspace), DEFAULT_THROTTLE)

    return md


def _write_if_missing(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` only if the file does not exist.
    Never truncates or overwrites an existing file."""
    if path.exists():
        return
    try:
        path.write_text(content)
    except OSError as e:
        logger.warning("Failed to bootstrap %s: %s", path, e)


# ---------------------------------------------------------------------------
# Throttle reader — used by dispatcher in the assign / submit hot path. Must
# be tolerant of a missing or corrupt file (system reverts to no throttling).
# ---------------------------------------------------------------------------

VALID_THROTTLE_LEVELS = ("none", "slow", "halt-new")


def read_throttle(workspace: str | Path) -> dict[str, str]:
    """Return current throttle state ``{"gpu": <level>, "cpu": <level>}``.

    Levels: ``none`` (default), ``slow`` (halve new-submit capacity, rounded
    down to >=1), ``halt-new`` (let in-flight finish but block new launches).

    Tolerant of:

    * Missing file → defaults to ``{"gpu": "none", "cpu": "none"}``.
    * Corrupt JSON → defaults; warning logged but no exception.
    * Unknown levels → coerced to ``none`` so a typo from the conductor does
      not silently halt the pipeline.
    """
    p = throttle_path(workspace)
    default = {"gpu": "none", "cpu": "none"}
    if not p.exists():
        return default
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        logger.warning("read_throttle: failed to parse %s (%s); defaulting", p, e)
        return default
    if not isinstance(raw, dict):
        return default
    out = dict(default)
    for k in ("gpu", "cpu"):
        v = raw.get(k, "none")
        if isinstance(v, str) and v in VALID_THROTTLE_LEVELS:
            out[k] = v
    return out
