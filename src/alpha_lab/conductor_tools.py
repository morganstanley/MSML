"""Implementation helpers for the Conductor's tools.

The actual tool dispatch lives in ``tools.py``; this module contains the
side-effecting helpers (file appends, JSON merges, backups, log digesting)
that the dispatch entries delegate to. Splitting them out keeps ``tools.py``
focused on schemas and dispatch and lets us unit-test the helpers without
constructing an AgentLoop.

All helpers are tolerant of missing meta/ files — they will create what
they need lazily so the system continues operating even if the user (or a
prior crash) wiped the meta/ directory.

None of these helpers wait, block, or alert. If they hit an error they
either log a warning and return a sensible default, or raise — the caller
in ``tools.py`` is responsible for converting raises into ``[ERROR] …``
strings the agent sees.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from alpha_lab import meta_layout

logger = logging.getLogger("alpha_lab.conductor_tools")


# Bound the per-call payload sizes so a single tool call cannot exhaust the
# agent's context. These are deliberately generous; the agent's prompt also
# tells it to read deliberately. ``tool_output_max_chars`` in the agent loop
# applies on top of these as a final cap.
MAX_REASON_CHARS = 2_000
MAX_EVIDENCE_CHARS = 8_000
MAX_DIRECTIVE_CHARS = 4_000
MAX_NOTE_CHARS = 4_000
MAX_DIGEST_CHARS = 12_000
DEFAULT_META_LOG_LAST_N = 20
DEFAULT_PEEK_LINES = 200


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically via tmp-then-rename.

    Without this, files like ``meta/directives.md`` and
    ``meta/annotations.json`` (whose writes are read-modify-write under
    the hood) can be seen mid-write by a concurrent reader — e.g. a
    strategist turn reading directives while the Conductor's daemon
    thread is rewriting them. POSIX ``rename`` on the same filesystem
    is atomic, so a reader either sees the old content or the new one,
    never a torn intermediate."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


VALID_DECISION_TYPES = (
    "park",
    "unpark",
    "set_priority",
    "clear_experiment_block",
    "annotate",
    "directive",
    "retire_directive",
    "note_to_user",
    "ack_user_instruction",
    "throttle",
    "kill",
    "delete",
    "backup",
    "phase_rewind",
    "no_action",
    "retrospective",
)


VALID_ANNOTATION_LABELS = (
    "champion",
    "control",
    "challenger",
    "exploration",
    "exploitation",
    "ensemble-candidate",
    "home-run-attempt",
    "quarantined",
    "quarantined_leakage",
    "quarantined_invalid_split",
    "quarantined_zombie",
)


VALID_DIRECTIVE_TARGETS = (
    "strategist", "worker", "reporter", "supervisor",
    # Phase 2 framework-building agents. The Conductor can target a
    # specific step (e.g. "the tester must add a leakage assertion
    # before the harness is accepted") in addition to broad-strokes
    # "all" or per-role directives.
    "builder", "critic", "tester",
    "all",
)


# ---------------------------------------------------------------------------
# meta_log: the audit trail.
# ---------------------------------------------------------------------------


def meta_log_entry(
    decision_type: str,
    *,
    target: object = None,
    reason: str = "",
    evidence: str = "",
    prior_decision_id: int | None = None,
    self_check: str = "",
) -> dict[str, Any]:
    """Build a single meta_log entry dict.

    The Conductor's audit log lives in ``meta/meta_log.jsonl``. Each entry
    is one decision and is intentionally thin — narrative belongs in the
    ``reason`` field, not in the structure.
    """
    if decision_type not in VALID_DECISION_TYPES:
        # Don't reject unknown types — the meta_log is append-only and we'd
        # rather record the strange entry than refuse to log. Just warn.
        logger.warning(
            "meta_log_entry: unrecognized decision_type %r — recording anyway",
            decision_type,
        )
    return {
        "ts": time.time(),
        "decision_type": decision_type,
        "target": target,
        "reason": (reason or "")[:MAX_REASON_CHARS],
        "evidence": (evidence or "")[:MAX_EVIDENCE_CHARS],
        "prior_decision_id": prior_decision_id,
        "self_check": (self_check or "")[:MAX_REASON_CHARS],
    }


def meta_log_append(workspace: str | Path, entry: dict[str, Any]) -> None:
    """Append entry to meta/meta_log.jsonl. Best-effort: warns on OSError
    instead of raising so a transient FS hiccup does not crash the
    Conductor's turn."""
    meta_layout.ensure_meta_layout(workspace)
    path = meta_layout.meta_log_jsonl_path(workspace)
    try:
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        logger.warning("meta_log_append: failed to write to %s: %s", path, e)


def meta_log_read(
    workspace: str | Path,
    last_n: int = DEFAULT_META_LOG_LAST_N,
    sample_older: bool = False,
) -> list[dict[str, Any]]:
    """Read the meta_log.

    Returns the most recent ``last_n`` entries. If ``sample_older=True``,
    additionally returns up to 10 stratified-sampled entries from earlier
    in the log (this powers the Conductor's retrospective audit step).

    Bounded: ``last_n`` is clamped to [1, 500] and the sampled older set
    is at most 10 entries.
    """
    last_n = max(1, min(int(last_n), 500))
    path = meta_layout.meta_log_jsonl_path(workspace)
    if not path.exists():
        return []
    try:
        lines = path.read_text().splitlines()
    except OSError as e:
        logger.warning("meta_log_read: %s — returning []", e)
        return []
    parsed: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed.append(json.loads(line))
        except ValueError:
            continue  # Skip malformed lines; never crash the read

    if not parsed:
        return []
    if len(parsed) <= last_n:
        return parsed

    recent = parsed[-last_n:]
    if not sample_older:
        return recent

    older = parsed[:-last_n]
    n_samples = min(10, len(older))
    if n_samples == 0:
        return recent
    # Stratified sample: pick evenly-spaced indices.
    step = max(1, len(older) // n_samples)
    sampled = [older[i] for i in range(0, len(older), step)][:n_samples]
    # Mark sampled entries so the agent can tell them apart from contiguous
    # recent history without having to compare timestamps.
    for s in sampled:
        s["_sampled"] = True
    return sampled + recent


def meta_log_render_md(workspace: str | Path, last_n: int = 200) -> None:
    """Re-render meta_log.md as a human-readable digest of the JSONL.

    The user reads this file — keep it tight. One line per entry, reverse
    chronological, header at top.
    """
    meta_layout.ensure_meta_layout(workspace)
    entries = meta_log_read(workspace, last_n=last_n, sample_older=False)
    lines = [
        "# Conductor decision log",
        "",
        f"_Last {len(entries)} entries (most recent first). Full history in `meta_log.jsonl`._",
        "",
    ]
    for e in reversed(entries):
        ts = e.get("ts", 0)
        when = _dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")
        dt = e.get("decision_type", "?")
        target = e.get("target", "")
        reason = e.get("reason", "").replace("\n", " ").strip()
        if len(reason) > 200:
            reason = reason[:197] + "..."
        target_str = f" {target}" if target not in (None, "") else ""
        lines.append(f"- `{when}` **{dt}**{target_str} — {reason}")
    md_path = meta_layout.meta_log_md_path(workspace)
    try:
        _atomic_write_text(md_path, "\n".join(lines) + "\n")
    except OSError as e:
        logger.warning("meta_log_render_md: failed to write %s: %s", md_path, e)


# ---------------------------------------------------------------------------
# Annotations: leaderboard labels keyed by experiment id.
# ---------------------------------------------------------------------------


def _read_annotations_raw(workspace: str | Path) -> dict[str, Any]:
    """Read raw annotations.json. Each value is either a plain string label
    (legacy flat format) or a dict ``{"label": str, "reason": str, "ts": float}``
    (current rich format). Callers should normalize via
    ``read_annotations`` (labels only) or ``read_annotation_details``
    (full records)."""
    p = meta_layout.annotations_path(workspace)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def read_annotations(workspace: str | Path) -> dict[str, str]:
    """Return ``{exp_id: label}``. Tolerates both legacy flat strings and
    the rich shape; callers that want the reason should use
    ``read_annotation_details``."""
    raw = _read_annotations_raw(workspace)
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            out[str(k)] = v
        elif isinstance(v, dict):
            lbl = v.get("label")
            if isinstance(lbl, str):
                out[str(k)] = lbl
    return out


def read_annotation_details(workspace: str | Path) -> dict[str, dict[str, Any]]:
    """Return ``{exp_id: {"label": str, "reason": str, "ts": float}}``.

    Each record always carries a ``label`` key; ``reason`` and ``ts``
    default to ``""`` and ``0.0`` if the on-disk entry was the legacy
    flat-string form (where only the label was stored)."""
    raw = _read_annotations_raw(workspace)
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        key = str(k)
        if isinstance(v, str):
            out[key] = {"label": v, "reason": "", "ts": 0.0}
        elif isinstance(v, dict):
            lbl = v.get("label")
            if isinstance(lbl, str):
                out[key] = {
                    "label": lbl,
                    "reason": str(v.get("reason", "") or ""),
                    "ts": float(v.get("ts", 0.0) or 0.0),
                }
    return out


def set_annotation(
    workspace: str | Path,
    exp_id: int,
    label: str,
    reason: str = "",
) -> None:
    """Set the annotation for one experiment id. Other annotations untouched.

    The optional ``reason`` is stored alongside the label so consumers
    (strategist, worker, board view) can surface the *why* without
    grepping ``meta_log.jsonl``."""
    meta_layout.ensure_meta_layout(workspace)
    raw = _read_annotations_raw(workspace)
    raw[str(int(exp_id))] = {
        "label": label,
        "reason": (reason or "").strip(),
        "ts": time.time(),
    }
    p = meta_layout.annotations_path(workspace)
    try:
        _atomic_write_text(p, json.dumps(raw, indent=2) + "\n")
    except OSError as e:
        logger.warning("set_annotation: failed to write %s: %s", p, e)


def clear_annotation(workspace: str | Path, exp_id: int) -> None:
    raw = _read_annotations_raw(workspace)
    raw.pop(str(int(exp_id)), None)
    p = meta_layout.annotations_path(workspace)
    try:
        _atomic_write_text(p, json.dumps(raw, indent=2) + "\n")
    except OSError as e:
        logger.warning("clear_annotation: failed to write %s: %s", p, e)


# ---------------------------------------------------------------------------
# Directives: appended (most recent at top) to meta/directives.md.
#
# Each directive has a *scope* that tells other agents how to treat it:
#
#   - ``standing``       — applies indefinitely to every action of the target
#                          role. The dominant shape: "include cold-client
#                          slice in every debrief", "always cite sample size
#                          on metrics." No ack is expected; many same-role
#                          agents act on it independently and that's correct.
#
#   - ``one-shot``       — the FIRST agent of the target role to honor it
#                          claims it via ack_directive(); subsequent same-role
#                          agents see the ack and skip. Used for discrete
#                          tasks that should happen exactly once across all
#                          same-role actors: "propose 3 new experiments",
#                          "write a leaderboard summary CSV".
#
#   - ``per-experiment:<id>`` — tied to a specific experiment id. Only the
#                          worker assigned to that experiment acts; everyone
#                          else ignores. "For experiment #45, redo the
#                          analysis with the fixed metric extraction."
#
# Each directive gets a deterministic id (``d-<unix-ts>-<short-random>``)
# written into the header so the ack log can reference it.
# ---------------------------------------------------------------------------


VALID_DIRECTIVE_SCOPES = ("standing", "one-shot")  # plus "per-experiment:<id>"


def _new_directive_id() -> str:
    """Unique id readable in the markdown header. We don't need
    cryptographic uniqueness — the timestamp prefix is enough to keep
    things ordered, and 6 random hex chars make accidental collisions
    vanishingly unlikely without locking."""
    import secrets
    ts = int(time.time())
    suffix = secrets.token_hex(3)  # 6 hex chars
    return f"d-{ts}-{suffix}"


def _is_valid_scope(scope: str) -> bool:
    if scope in VALID_DIRECTIVE_SCOPES:
        return True
    if scope.startswith("per-experiment:"):
        rest = scope[len("per-experiment:"):]
        return rest.isdigit() and int(rest) > 0
    return False


def append_directive(
    workspace: str | Path,
    target_role: str,
    message: str,
    reason: str = "",
    scope: str = "standing",
) -> str:
    """Prepend a directive entry under the targeted role. The file is
    structured as a single markdown document the conductor maintains.

    Returns the directive's auto-generated id so the caller can include
    it in the meta_log entry. Other agents discover the same id by
    parsing ``meta/directives.md`` themselves.
    """
    meta_layout.ensure_meta_layout(workspace)
    target_role = target_role if target_role in VALID_DIRECTIVE_TARGETS else "all"
    message = (message or "")[:MAX_DIRECTIVE_CHARS]
    when = _dt.datetime.now().isoformat(timespec="seconds")
    if not _is_valid_scope(scope):
        logger.warning(
            "append_directive: unrecognized scope %r — defaulting to standing",
            scope,
        )
        scope = "standing"

    directive_id = _new_directive_id()
    p = meta_layout.directives_path(workspace)
    existing = ""
    if p.exists():
        try:
            existing = p.read_text()
        except OSError:
            existing = ""
    # Header is a single line so it's cheap to parse with a regex:
    # "## directive <id> <role> scope=<scope> <iso-timestamp>"
    block = (
        f"## directive {directive_id}  {target_role}  scope={scope}  {when}\n"
        f"{message.strip()}\n"
    )
    if reason:
        block += f"\n_reason: {reason.strip()}_\n"
    # Preserve the file header (everything up to the first '## directive'
    # line) and prepend the new directive after it. When the file has only
    # the bootstrap placeholder ('(no directives yet)') and no real
    # directive marker yet, ``_split_header`` returns the whole content as
    # ``header`` with empty ``sep``/``rest`` — so strip the placeholder
    # from ``header``, not ``rest``.
    header, sep, rest = _split_header(existing, marker_prefix="## directive")
    if "(no directives yet)" in header:
        header = header.replace("(no directives yet)", "").rstrip() + "\n"
    if not sep and "(no directives yet)" in rest:
        # Defensive: future placement of the placeholder below an existing
        # directive header would land in rest. Keep the original strip too.
        rest = rest.replace("(no directives yet)", "").rstrip() + "\n"
    new_content = (header.rstrip() + "\n\n" if header.strip() else "") + block + "\n" + rest.lstrip()
    try:
        _atomic_write_text(p, new_content)
    except OSError as e:
        logger.warning("append_directive: failed to write %s: %s", p, e)
    return directive_id


# ---------------------------------------------------------------------------
# Parsing + reading directives. Strategist and workers call this at the top
# of every turn to filter out one-shot directives that have already been
# acked by an agent of their role, and to pick out per-experiment directives
# that target their assigned experiment.
# ---------------------------------------------------------------------------


# Matches the directive header line:
#   ## directive d-1778383200-a4f7  strategist  scope=one-shot  2026-05-10T16:00:00
_DIRECTIVE_HEADER_RE = __import__("re").compile(
    r"^##\s+directive\s+(\S+)\s+(\S+)\s+scope=(\S+)\s+(\S+)\s*$"
)


def parse_directives(workspace: str | Path) -> list[dict[str, Any]]:
    """Parse meta/directives.md into a list of {id, role, scope, timestamp,
    body} dicts. Most-recent first.

    Returns []
      - if the file doesn't exist,
      - if there are no directive blocks yet (the boilerplate header is
        present but nothing has been issued).
    Lines that don't match the header regex are skipped silently — old
    directive blocks from prior runs that used a different header format
    just don't appear as parseable directives. That's deliberate; the new
    scope/ack mechanism only applies to directives issued under the new
    format.
    """
    p = meta_layout.directives_path(workspace)
    if not p.exists():
        return []
    try:
        text = p.read_text()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        m = _DIRECTIVE_HEADER_RE.match(line)
        if m:
            if current is not None:
                out.append(current)
            current = {
                "id": m.group(1),
                "role": m.group(2),
                "scope": m.group(3),
                "timestamp": m.group(4),
                "body": "",
            }
            continue
        if current is None:
            continue
        # Body lines accumulate until the next header.
        if line.startswith("## ") and not line.startswith("## directive "):
            # Some other markdown header (e.g. boilerplate footer) — end
            # the directive's body.
            out.append(current)
            current = None
            continue
        current["body"] += line + "\n"
    if current is not None:
        out.append(current)
    return out


def read_directive_acks(workspace: str | Path) -> list[dict[str, Any]]:
    """Read meta/directive_acks.jsonl. Each entry is one agent's claim
    that it acted on a directive."""
    p = meta_layout.directive_acks_path(workspace)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue  # skip malformed lines; never crash the read
    except OSError:
        pass
    return out


def append_directive_ack(
    workspace: str | Path,
    directive_id: str,
    actor_role: str,
    actor_id: str,
    action: str,
) -> None:
    """Append a single ack line. Best-effort — warns on OSError instead
    of raising so a transient FS hiccup doesn't crash the agent's turn."""
    meta_layout.ensure_meta_layout(workspace)
    entry = {
        "ts": time.time(),
        "directive_id": directive_id,
        "actor_role": actor_role,
        "actor_id": actor_id,
        "action": (action or "")[:MAX_REASON_CHARS],
    }
    p = meta_layout.directive_acks_path(workspace)
    try:
        with open(p, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        logger.warning("append_directive_ack: failed to write %s: %s", p, e)


def read_directive_retirements(workspace: str | Path) -> list[dict[str, Any]]:
    """Read meta/directive_retirements.jsonl. Each line is
    ``{ts, directive_id, reason}`` — the Conductor's explicit decision
    that a directive is no longer in force."""
    p = meta_layout.directive_retirements_path(workspace)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue  # skip malformed lines; never crash the read
    except OSError:
        pass
    return out


def retired_directive_ids(workspace: str | Path) -> set[str]:
    return {
        str(r.get("directive_id", ""))
        for r in read_directive_retirements(workspace)
        if r.get("directive_id")
    }


def append_directive_retirement(
    workspace: str | Path,
    directive_id: str,
    reason: str,
) -> None:
    """Append a retirement entry. Idempotent in spirit — appending the
    same directive id twice is harmless (readers dedupe via set), but a
    second retirement *should* carry a different reason to record why
    the Conductor came back to it."""
    meta_layout.ensure_meta_layout(workspace)
    entry = {
        "ts": time.time(),
        "directive_id": directive_id,
        "reason": (reason or "").strip()[:MAX_REASON_CHARS],
    }
    p = meta_layout.directive_retirements_path(workspace)
    try:
        with open(p, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        logger.warning("append_directive_retirement: failed to write %s: %s", p, e)


def directives_for_role(
    workspace: str | Path,
    role: str,
    experiment_id: int | None = None,
) -> list[dict[str, Any]]:
    """Return the active directives the given role should follow on its
    current turn. Drops:

    * directives targeted at other roles (and not ``all``);
    * directives the Conductor has explicitly retired
      (``meta/directive_retirements.jsonl``);
    * one-shot directives that any same-role actor has already acked;
    * per-experiment directives whose experiment id doesn't match the
      caller's ``experiment_id`` (or any if ``experiment_id`` is None
      and the directive specifies one).

    Standing directives are always included unless retired.
    """
    directives = parse_directives(workspace)
    retired = retired_directive_ids(workspace)
    acks = read_directive_acks(workspace)
    acked_by_role: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for a in acks:
        key = (a.get("directive_id", ""), a.get("actor_role", ""))
        acked_by_role.setdefault(key, []).append(a)

    active: list[dict[str, Any]] = []
    for d in directives:
        # Retired directives are off the air for everyone.
        if d["id"] in retired:
            continue
        # Role gate. "all" targets everyone; otherwise must match.
        if d["role"] not in ("all", role):
            continue
        scope = d["scope"]
        if scope == "standing":
            active.append(d)
            continue
        if scope == "one-shot":
            # Any actor of the same role acking is enough to claim it.
            if acked_by_role.get((d["id"], role)):
                continue
            active.append(d)
            continue
        if scope.startswith("per-experiment:"):
            try:
                target_exp = int(scope.split(":", 1)[1])
            except (ValueError, IndexError):
                continue
            if experiment_id is None or target_exp != experiment_id:
                continue
            # Same one-shot semantics within the experiment.
            if acked_by_role.get((d["id"], role)):
                continue
            active.append(d)
            continue
        # Unrecognized scope — include conservatively as if standing.
        active.append(d)
    return active


def render_directives_for_prompt(
    directives: list[dict[str, Any]],
    acks: list[dict[str, Any]] | None = None,
) -> str:
    """Render the active directives into a markdown block suitable for
    injecting into an agent's prompt context. Also includes a short
    tail of recent acks so the agent can see what same-role peers have
    already done (useful even for standing directives — "another worker
    just included the cold-client slice in #178's debrief")."""
    if not directives:
        body = "(no active directives that apply to you right now)\n"
    else:
        parts = []
        for d in directives:
            scope_label = d["scope"]
            ack_note = ""
            if scope_label == "one-shot":
                ack_note = (
                    "  _Scope: one-shot — once you act on this, call "
                    "`ack_directive` with its id; later same-role turns "
                    "will then see it as claimed and skip._\n"
                )
            elif scope_label.startswith("per-experiment:"):
                ack_note = (
                    "  _Scope: per-experiment — applies only to the "
                    "experiment id named in the scope, and only once._\n"
                )
            parts.append(
                f"### directive {d['id']}  ({d['role']}, scope={d['scope']}, {d['timestamp']})\n"
                f"{d['body'].strip()}\n"
                f"{ack_note}"
            )
        body = "\n".join(parts)
    out = "## Conductor directives (active for you right now)\n\n" + body
    if acks:
        recent = acks[-5:]
        lines = ["", "### Recent same-type peer activity (ack tail)", ""]
        for a in recent:
            when = _dt.datetime.fromtimestamp(a.get("ts", 0)).isoformat(timespec="seconds")
            lines.append(
                f"- `{when}` {a.get('actor_role','?')}/{a.get('actor_id','?')} "
                f"acked **{a.get('directive_id','?')}**: "
                f"{(a.get('action') or '')[:160]}"
            )
        out += "\n" + "\n".join(lines) + "\n"
    return out


ESCALATE_AFTER_ZERO_ACK_TURNS = 3


def directive_uptake_escalation(
    workspace: str | Path,
    active: list[dict[str, Any]],
    acks: list[dict[str, Any]],
    actor_role: str = "strategist",
) -> str | None:
    """Deterministic escalation when directives go unacknowledged.

    A 9-run audit (2026-08-08) found the conductor issuing 8–27 directives
    per run while some models never acknowledged one; every zero-ack run
    lost its pair. The directive channel must not fail silently: after
    ``ESCALATE_AFTER_ZERO_ACK_TURNS`` consecutive turns with active
    directives and zero recorded acknowledgements, return a banner the
    caller puts at the TOP of the agent's context, demanding explicit
    ``ack_directive`` calls this turn. Counting is per role and intended
    for the single-threaded strategist seat (worker turns are many and
    concurrent; counting them would race). Any recorded ack resets the
    counter."""
    p = meta_layout.meta_dir(workspace) / ".directive_uptake.json"
    try:
        state = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        state = {}
    n = int(state.get(actor_role, 0))
    if not active or acks:
        if n:
            state[actor_role] = 0
            _atomic_write_text(p, json.dumps(state))
        return None
    n += 1
    state[actor_role] = n
    _atomic_write_text(p, json.dumps(state))
    if n < ESCALATE_AFTER_ZERO_ACK_TURNS:
        return None
    logger.warning(
        "directive uptake escalation: %d active directive(s), zero acks "
        "for %d consecutive %s turns", len(active), n, actor_role)
    return (
        "## ⚠ ESCALATION: Conductor directives are being ignored\n\n"
        f"{len(active)} directive(s) are active for your role and ZERO "
        f"acknowledgements have been recorded across {n} consecutive "
        "turns. Directives are the user's steering channel — mandatory, "
        "not advisory. THIS TURN, before other work: read every directive "
        "in the directives section, act on each applicable one, and call "
        "`ack_directive` with the directive id and what you did. If a "
        "directive does not apply to your current work, ack it with the "
        "reason it does not apply.\n"
    )


def _split_header(text: str, marker_prefix: str) -> tuple[str, str, str]:
    """Split a markdown document into (header, separator, rest) where the
    header is everything before the first line starting with marker_prefix.

    Returns ("", "", text) if the marker isn't present.
    """
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(marker_prefix):
            return "".join(lines[:i]), line, "".join(lines[i:])
    return text, "", ""


# ---------------------------------------------------------------------------
# Notes to/from user.
# ---------------------------------------------------------------------------


def append_note_to_user(workspace: str | Path, message: str) -> None:
    """Append (most recent at top, reverse chronological session blocks)
    a Conductor note to ``meta/notes_to_user.md``.

    The conductor LLM often starts its message with its own ``## <date>``
    header (free-form, e.g. ``## 2026-05-11 ~08:22 — Post-M5 audit``). If
    we naively prepend our auto ISO timestamp, the output gets two ``## ``
    headers in a row. Strip one leading ``## `` line from the message so
    the auto ISO header (precise + sortable) is the only top-line marker.
    """
    meta_layout.ensure_meta_layout(workspace)
    message = (message or "")[:MAX_NOTE_CHARS]
    when = _dt.datetime.now().isoformat(timespec="seconds")
    p = meta_layout.notes_to_user_path(workspace)
    existing = ""
    if p.exists():
        try:
            existing = p.read_text()
        except OSError:
            existing = ""
    stripped = message.strip()
    if stripped.startswith("## "):
        # Drop the model's own date header line (keeps body intact).
        nl = stripped.find("\n")
        stripped = stripped[nl + 1:].lstrip("\n") if nl != -1 else ""
    block = f"## {when}\n{stripped}\n\n"
    header, sep, rest = _split_header(existing, marker_prefix="## ")
    new_content = header + block + (sep + rest if sep else "")
    try:
        _atomic_write_text(p, new_content)
    except OSError as e:
        logger.warning("append_note_to_user: failed to write %s: %s", p, e)


def append_note_to_conductor(
    workspace: str | Path,
    sender_role: str,
    message: str,
) -> None:
    """Strategist/worker → Conductor message. Single combined inbox file
    partitioned by sender header."""
    meta_layout.ensure_meta_layout(workspace)
    message = (message or "")[:MAX_NOTE_CHARS]
    when = _dt.datetime.now().isoformat(timespec="seconds")
    p = meta_layout.notes_inbox_path(workspace)
    block = f"## {sender_role} — {when}\n{message.strip()}\n\n"
    try:
        with open(p, "a") as f:
            f.write(block)
    except OSError as e:
        logger.warning("append_note_to_conductor: failed to write %s: %s", p, e)


def read_notes_inbox(workspace: str | Path, max_chars: int = MAX_DIGEST_CHARS) -> str:
    p = meta_layout.notes_inbox_path(workspace)
    if not p.exists():
        return ""
    try:
        text = p.read_text()
    except OSError:
        return ""
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


# ---------------------------------------------------------------------------
# User instructions (from_user.md ↔ ack.md, with .last_seen bookkeeping).
# ---------------------------------------------------------------------------


def read_from_user_diff(workspace: str | Path) -> tuple[str, bool]:
    """Return (current_content, is_new) where is_new is True when
    from_user.md has changed since the conductor last saw it."""
    meta_layout.ensure_meta_layout(workspace)
    fu = meta_layout.from_user_path(workspace)
    ls = meta_layout.last_seen_path(workspace)
    try:
        current = fu.read_text() if fu.exists() else ""
    except OSError:
        current = ""
    try:
        last = ls.read_text() if ls.exists() else ""
    except OSError:
        last = ""
    return current, current != last


def mark_user_instructions_seen(workspace: str | Path) -> None:
    """Persist the current from_user.md content as 'last seen'. The
    conductor calls this after acknowledging an update."""
    meta_layout.ensure_meta_layout(workspace)
    fu = meta_layout.from_user_path(workspace)
    ls = meta_layout.last_seen_path(workspace)
    try:
        ls.write_text(fu.read_text() if fu.exists() else "")
    except OSError as e:
        logger.warning("mark_user_instructions_seen: failed to write %s: %s", ls, e)


def append_ack(workspace: str | Path, message: str) -> None:
    """Append an acknowledgement entry. Most recent at top."""
    meta_layout.ensure_meta_layout(workspace)
    message = (message or "")[:MAX_NOTE_CHARS]
    when = _dt.datetime.now().isoformat(timespec="seconds")
    p = meta_layout.ack_path(workspace)
    existing = p.read_text() if p.exists() else ""
    header, sep, rest = _split_header(existing, marker_prefix="## ")
    block = f"## {when}\n{message.strip()}\n\n"
    new_content = header + block + (sep + rest if sep else "")
    try:
        _atomic_write_text(p, new_content)
    except OSError as e:
        logger.warning("append_ack: failed to write %s: %s", p, e)


# ---------------------------------------------------------------------------
# Throttle.
# ---------------------------------------------------------------------------


def set_throttle_state(
    workspace: str | Path,
    *,
    gpu: str | None = None,
    cpu: str | None = None,
) -> dict[str, str]:
    """Update throttle.json. Pass only the dimension you want to change;
    the other is preserved. Unknown level strings are coerced to 'none'."""
    meta_layout.ensure_meta_layout(workspace)
    current = meta_layout.read_throttle(workspace)
    if gpu is not None:
        current["gpu"] = gpu if gpu in meta_layout.VALID_THROTTLE_LEVELS else "none"
    if cpu is not None:
        current["cpu"] = cpu if cpu in meta_layout.VALID_THROTTLE_LEVELS else "none"
    p = meta_layout.throttle_path(workspace)
    try:
        _atomic_write_text(p, json.dumps(current) + "\n")
    except OSError as e:
        logger.warning("set_throttle_state: failed to write %s: %s", p, e)
    return current


# ---------------------------------------------------------------------------
# Backups and deletes (always backup before delete).
# ---------------------------------------------------------------------------


def _new_backup_dir(workspace: str | Path, label: str = "") -> Path:
    """Allocate a fresh, timestamped backup subdirectory. Label is for
    human readability; the timestamp guarantees uniqueness."""
    meta_layout.ensure_meta_layout(workspace)
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    name = f"{ts}_{label}".rstrip("_") if label else ts
    # If a backup at this same timestamp already exists (rare collision in a
    # tight test loop), append a counter.
    base = meta_layout.backups_dir(workspace) / name
    i = 0
    p = base
    while p.exists():
        i += 1
        p = base.with_name(f"{base.name}_{i}")
    p.mkdir(parents=True, exist_ok=False)
    return p


def backup_workspace_path(
    workspace: str | Path, rel_path: str, label: str = ""
) -> Path:
    """Copy ``<workspace>/<rel_path>`` to a fresh backup subdir. Returns
    the destination path. The source is left untouched.

    ``rel_path`` must stay inside the workspace — absolute paths or
    ``..`` traversal raise ValueError so a malicious agent cannot use this
    tool to read or copy paths outside its sandbox."""
    src = _safe_workspace_path(workspace, rel_path)
    if not src.exists():
        raise FileNotFoundError(f"backup_path: source {src} does not exist")
    dest_dir = _new_backup_dir(workspace, label=Path(rel_path).name)
    dest = dest_dir / src.name
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
    return dest


def safe_delete_with_backup(
    workspace: str | Path,
    rel_path: str,
    label: str = "",
    adapter: Any | None = None,
) -> Path:
    """Backup then delete. Returns the backup path so the user can restore.

    Refuses to delete certain protected paths (the meta/ tree itself,
    experiments.db, the adapter, the adapter's declared framework_dir,
    data_report, scripts, playbook/learnings). The Conductor's prompt
    further constrains usage to stale caches the agent has proved are
    unneeded; this function is the second-line guard."""
    src = _safe_workspace_path(workspace, rel_path)
    if not src.exists():
        raise FileNotFoundError(f"delete_path: source {src} does not exist")
    if _is_protected(workspace, src, adapter=adapter):
        raise PermissionError(
            f"delete_path: {rel_path} is protected (meta/, db, adapter, "
            f"adapter framework dir, data_report, scripts, etc.)"
        )
    backup = backup_workspace_path(workspace, rel_path, label=label or "predelete")
    if src.is_dir():
        shutil.rmtree(src)
    else:
        src.unlink()
    return backup


# Universal protected prefixes — these have fixed names across all
# adapters and the system relies on them. Adapter-specific dirs
# (notably the framework_dir, whose name varies — backtest / harness /
# kernels / speedrun / etc.) are added dynamically at protection-check
# time by ``_is_protected`` when an adapter is passed in.
_PROTECTED_PREFIXES = (
    "meta",
    "adapter",
    "data_report",
    "scripts",
)
_PROTECTED_FILES = (
    "experiments.db",
    "experiments.db-wal",
    "experiments.db-shm",
    "playbook.md",
    "learnings.md",
)


def _is_protected(
    workspace: str | Path,
    target: Path,
    adapter: Any | None = None,
) -> bool:
    """True if ``target`` is a workspace path the Conductor must not
    delete: the workspace root, anything outside it, the meta/ tree,
    the adapter dir, ``data_report``, ``scripts``, key DB/playbook
    files, or — when ``adapter`` is provided — the adapter's declared
    framework_dir (its name varies by adapter, so we resolve it from
    the adapter rather than hardcoding common values; any future
    adapter with a new framework_dir name gets protection automatically)."""
    try:
        rel = target.resolve().relative_to(Path(workspace).resolve())
    except ValueError:
        return True  # Outside workspace = always protected
    parts = rel.parts
    if not parts:
        return True  # workspace root itself
    if parts[0] in _PROTECTED_PREFIXES:
        return True
    if adapter is not None:
        try:
            fw = getattr(getattr(adapter, "experiment", None), "framework_dir", None)
        except Exception:
            fw = None
        if fw and parts[0] == fw:
            return True
    if str(rel) in _PROTECTED_FILES:
        return True
    return False


def _safe_workspace_path(workspace: str | Path, rel_path: str) -> Path:
    """Resolve rel_path within workspace. Reject absolute paths and
    ``..``-style escapes; the resolved path must stay inside workspace."""
    if os.path.isabs(rel_path):
        raise ValueError(f"path must be workspace-relative, got absolute: {rel_path}")
    workspace = Path(workspace).resolve()
    candidate = (workspace / rel_path).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError:
        raise ValueError(f"path escapes workspace: {rel_path}")
    return candidate


# ---------------------------------------------------------------------------
# Phase rewind: writes a marker the dispatcher acts on, after backing up
# the relevant phase's artifacts. The actual phase reset happens in
# pipeline.py / dispatcher.py in response to the marker — keeping the tool
# itself thin and the destructive logic in one place.
# ---------------------------------------------------------------------------


PHASE_REWIND_MARKER = "phase_rewind_pending.json"
RUN_END_MARKER = "run_end_pending.json"
VERIFY_REQUEST_MARKER = "verify_request.json"


def request_verification(workspace, candidate: str = "", steering: str = "",
                         reason: str = "", priority: str = "") -> dict:
    """Commission an independent verification. The Conductor is BOTH user and system here:
    it writes the verifier's steering to ``verify/from_user.md`` (as the user — what to verify
    and what it cares about), and the dispatcher surfaces the verifier's reports back into the
    Conductor's context (as the system). This writes a marker the dispatcher consumes to spawn
    the verifier in a background thread (mirrors request_run_end / request_phase_rewind).

    A named ``candidate`` is a PRIORITY PIN: the verifier verifies it as its NEXT candidate
    (jumping its adaptive selection order) instead of the arg being ignored. ``priority`` (e.g.
    "high"/"urgent") is the urgency signal. The pin is a single overwritten file, so the latest
    request wins — NOT strict FIFO; an urgent request jumps ahead of the adaptive order. It still
    waits for any in-flight candidate to finish (no mid-candidate preemption)."""
    import json as _json
    import time as _time
    from pathlib import Path as _Path
    meta_layout.ensure_meta_layout(workspace)
    vdir = _Path(workspace) / "verify"
    vdir.mkdir(parents=True, exist_ok=True)
    if steering and steering.strip():  # Conductor steers the verifier as "the user"
        (vdir / "from_user.md").write_text(steering.strip() + "\n")
    _pin_path = vdir / "priority_pin.json"
    if candidate and candidate.strip() and candidate.strip().upper() != "NONE":
        # Priority pin honored by Verifier._select_next_candidate as the next pick.
        _pin_path.write_text(_json.dumps(
            {"candidate": candidate.strip(), "priority": (priority or "").strip(), "ts": _time.time()}))
    else:
        # NONE/empty -> clear any stale pin so the latest request wins (adaptive pick), not FIFO.
        try:
            _pin_path.unlink()
        except OSError:
            pass
    payload = {"candidate": candidate, "reason": reason, "priority": priority, "ts": _time.time()}
    (meta_layout.meta_dir(workspace) / VERIFY_REQUEST_MARKER).write_text(_json.dumps(payload))
    return payload


def request_run_end(
    workspace: str | Path,
    reason: str,
    evidence: str,
    *,
    dispatcher_start_ts: float,
    analyzed_count: int,
    min_runtime_hours: float,
    min_analyzed_before_end: int,
    allow_end: bool,
) -> dict[str, Any]:
    """Conductor-driven graceful end-of-run request.

    Writes ``meta/run_end_pending.json`` which the dispatcher reads at the
    top of each main-loop iteration. On seeing the marker the dispatcher
    stops admitting new submissions, lets in-flight experiments finish,
    and exits cleanly after one last milestone report.

    Returns the marker dict. Refuses (raises ``ValueError``) when any
    floor is not met:

      * ``allow_end`` is False  — explicit kill switch in the run config.
      * Wall-clock from ``dispatcher_start_ts`` < ``min_runtime_hours``.
      * ``analyzed_count < min_analyzed_before_end``.

    Floors live on TaskConfig at the top level (parallel to the
    conductor_* knobs). Defensive against missing values: the caller is
    expected to pass them through from the dispatcher's TaskConfig.
    """
    if not allow_end:
        raise ValueError(
            "request_run_end refused: allow_conductor_end_run=false in the "
            "task config. The Conductor is not permitted to end this run; "
            "the dispatcher decides when to stop."
        )
    elapsed_h = (time.time() - dispatcher_start_ts) / 3600.0
    if elapsed_h < min_runtime_hours:
        raise ValueError(
            f"request_run_end refused: only {elapsed_h:.1f}h elapsed, "
            f"min_runtime_hours={min_runtime_hours:.1f}. The run is too "
            f"young to end; if you genuinely have evidence the run is "
            f"exhausted, write a note_to_user explaining what you saw "
            f"and let the dispatcher continue."
        )
    if analyzed_count < min_analyzed_before_end:
        raise ValueError(
            f"request_run_end refused: only {analyzed_count} experiments "
            f"have reached `analyzed`/`done`, "
            f"min_analyzed_before_end={min_analyzed_before_end}. The run "
            f"has not produced enough data to defensibly end."
        )
    meta_layout.ensure_meta_layout(workspace)
    marker_payload = {
        "ts": time.time(),
        "reason": (reason or "")[:MAX_REASON_CHARS],
        "evidence": (evidence or "")[:MAX_EVIDENCE_CHARS],
        "elapsed_hours_at_request": elapsed_h,
        "analyzed_at_request": analyzed_count,
    }
    marker_path = meta_layout.meta_dir(workspace) / RUN_END_MARKER
    _atomic_write_text(marker_path, json.dumps(marker_payload, indent=2) + "\n")
    return marker_payload


def request_phase_rewind(
    workspace: str | Path,
    target_phase: str,
    reason: str,
    evidence: str,
    adapter: Any | None = None,
) -> dict[str, Any]:
    """Back up the phase's artifacts and write a marker the dispatcher
    consumes. Returns the marker dict.

    target_phase ∈ {phase0, phase1, phase2}. (Phase 3 rewinds are not
    requested — Phase 3 is the experiment loop itself; the conductor
    parks experiments individually instead of restarting the loop.)

    ``adapter`` (optional) lets the backup capture the
    adapter-declared framework dir whose name varies by adapter; if
    not passed, only the universally-named artifacts are backed up.
    """
    if target_phase not in ("phase0", "phase1", "phase2"):
        raise ValueError(
            f"request_phase_rewind: target_phase must be phase0/1/2, got {target_phase!r}"
        )
    meta_layout.ensure_meta_layout(workspace)
    backup_root = _new_backup_dir(workspace, label=f"rewind_{target_phase}")

    # Always back up the adapter — every phase rewind needs it (Phase 0
    # rewrites it; Phase 1/2 rewinds may need to re-customize parts).
    src_adapter = Path(workspace) / "adapter"
    if src_adapter.is_dir():
        shutil.copytree(src_adapter, backup_root / "adapter")
    # Phase 1 produces learnings/data_report; Phase 2 produces the
    # adapter-declared framework dir. Back up whichever exists so the
    # rewound agents see the prior state.
    if target_phase in ("phase1", "phase2"):
        for sub in ("learnings.md", "data_report", "scripts", "notes", "plots"):
            p = Path(workspace) / sub
            if p.exists():
                if p.is_dir():
                    shutil.copytree(p, backup_root / sub)
                else:
                    shutil.copy2(p, backup_root / sub)
    if target_phase == "phase2":
        # Adapter declares the framework dir name. When the caller
        # passed an adapter we resolve it directly; otherwise the
        # framework dir backup is skipped (caller's responsibility to
        # pass adapter when phase 2 rewind matters).
        fw_dir = None
        if adapter is not None:
            fw_dir = getattr(getattr(adapter, "experiment", None), "framework_dir", None)
        if fw_dir:
            p = Path(workspace) / fw_dir
            if p.exists() and p.is_dir():
                shutil.copytree(p, backup_root / fw_dir, dirs_exist_ok=True)

    marker_payload = {
        "ts": time.time(),
        "target_phase": target_phase,
        "reason": (reason or "")[:MAX_REASON_CHARS],
        "evidence": (evidence or "")[:MAX_EVIDENCE_CHARS],
        "backup_dir": str(backup_root),
    }
    marker_path = meta_layout.meta_dir(workspace) / PHASE_REWIND_MARKER
    _atomic_write_text(marker_path, json.dumps(marker_payload, indent=2) + "\n")
    return marker_payload


# ---------------------------------------------------------------------------
# Read system load / peek experiment log.
# ---------------------------------------------------------------------------


def read_system_load(workspace: str | Path) -> str:
    """One-line system-load summary the Conductor uses to decide throttling."""
    parts: list[str] = []
    # CPU load (1m,5m,15m)
    try:
        load = os.getloadavg()
        parts.append(f"cpu_load 1m/5m/15m {load[0]:.1f}/{load[1]:.1f}/{load[2]:.1f}")
    except OSError:
        parts.append("cpu_load unavailable")
    # Disk free at workspace
    try:
        st = shutil.disk_usage(workspace)
        parts.append(f"disk_free_GB {st.free / 1e9:.0f} of {st.total / 1e9:.0f}")
    except OSError:
        pass
    # GPU summary via nvidia-smi (best-effort; works on the dispatcher host
    # only — that is the host the agent's shell_exec lives on, so this is
    # the right reading).
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout:
            gpu_lines = []
            for ln in out.stdout.strip().splitlines():
                fields = [f.strip() for f in ln.split(",")]
                if len(fields) >= 4:
                    idx, util, mem_u, mem_t = fields[:4]
                    gpu_lines.append(f"gpu{idx}={util}% {mem_u}MiB/{mem_t}MiB")
            if gpu_lines:
                parts.append(" ".join(gpu_lines))
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        parts.append("gpu unavailable")
    # Throttle state in case the agent forgot it
    th = meta_layout.read_throttle(workspace)
    parts.append(f"throttle gpu={th['gpu']} cpu={th['cpu']}")
    return " | ".join(parts)


def peek_experiment_log(
    workspace: str | Path, exp_name: str, last_n_lines: int = DEFAULT_PEEK_LINES
) -> str:
    """Read the last N lines of the experiment subprocess output log.

    Looks for the local executor's stdout dump (``experiments/<name>/local_job.out``
    is the conventional path; falls back to slurm output and a generic glob).
    """
    last_n_lines = max(1, min(int(last_n_lines), 2000))
    candidates = [
        Path(workspace) / "experiments" / exp_name / "local_job.out",
        Path(workspace) / "experiments" / exp_name / "slurm.out",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            try:
                content = p.read_text(errors="replace")
            except OSError:
                continue
            tail = content.splitlines()[-last_n_lines:]
            return "\n".join(tail)
    # Glob fallback — pick the most recent file under the experiment dir.
    exp_dir = Path(workspace) / "experiments" / exp_name
    if exp_dir.is_dir():
        files = sorted(
            (p for p in exp_dir.rglob("*") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in files:
            if p.suffix in (".out", ".log", ".txt"):
                try:
                    content = p.read_text(errors="replace")
                except OSError:
                    continue
                tail = content.splitlines()[-last_n_lines:]
                return f"# from {p.relative_to(workspace)}\n" + "\n".join(tail)
    return f"[no log found for experiment {exp_name!r}]"


# ---------------------------------------------------------------------------
# Read experiment: bounded summary that lets the Conductor decide whether
# to dive deeper via read_file calls on the returned paths.
# ---------------------------------------------------------------------------


def read_experiment_summary(workspace: str | Path, db: Any, exp_id: int) -> dict[str, Any]:
    """Return a bounded summary of one experiment.

    Includes: id, name, status, worker_id, parked_at, annotation, truncated
    description, truncated hypothesis, config keys+truncated values, headline
    metrics, error excerpt, and computed paths to the experiment's
    ``run_experiment.py``, debrief, and results dir. **Does not** include
    the full debrief or full code — the Conductor fetches those via
    ``read_file`` only when warranted, to keep per-call payloads small.
    """
    exp = db.get(exp_id) if db is not None else None
    if exp is None:
        return {"error": f"experiment {exp_id} not found"}

    annotations = read_annotations(workspace)
    annotation = annotations.get(str(exp_id), "")

    # Bounded extraction
    description = (exp.description or "")[:800]
    hypothesis = (exp.hypothesis or "")[:800]
    error = (exp.error or "")[:500]

    config_summary: dict[str, Any] = {}
    try:
        cfg = json.loads(exp.config_json or "{}")
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                s = json.dumps(v) if not isinstance(v, str) else v
                if len(s) > 200:
                    s = s[:200] + "..."
                config_summary[k] = s
    except ValueError:
        config_summary = {"_raw": (exp.config_json or "")[:500]}

    metrics_summary: dict[str, Any] = {}
    try:
        results = json.loads(exp.results_json or "{}")
        if isinstance(results, dict):
            # Headline + a few secondary, ordered to keep size predictable
            for k in list(results.keys())[:8]:
                v = results[k]
                if isinstance(v, (int, float, str, bool)) or v is None:
                    metrics_summary[k] = v
    except ValueError:
        pass

    exp_dir = Path(workspace) / "experiments" / exp.name
    paths = {
        "experiment_dir": str(exp_dir),
        "run_experiment_py": str(exp_dir / "run_experiment.py"),
        "results_dir": str(exp_dir / "results"),
        "metrics_json": str(exp_dir / "results" / "metrics.json"),
        "debrief": _existing_path(
            [
                exp_dir / "debrief.md",
                exp_dir / "analysis.md",
                exp_dir / "results" / "analysis.md",
            ]
        ),
    }
    flags = {
        "has_run_py": (exp_dir / "run_experiment.py").exists(),
        "has_metrics": (exp_dir / "results" / "metrics.json").exists(),
        "has_debrief": paths["debrief"] is not None,
    }

    # Inline up to ~4k chars of the debrief content when one exists, so the
    # Conductor has the analyzer's reasoning available without a separate
    # read_file call. Path is still returned alongside; the Conductor can
    # fetch the full file if it wants more than the head+tail excerpt.
    debrief_excerpt: str | None = None
    if paths["debrief"] is not None:
        try:
            text = Path(paths["debrief"]).read_text(encoding="utf-8", errors="replace")
            max_chars = 4_000
            if len(text) <= max_chars:
                debrief_excerpt = text
            else:
                head = max_chars // 2
                tail = max_chars - head
                trimmed = len(text) - head - tail
                debrief_excerpt = (
                    text[:head]
                    + f"\n\n[...trimmed {trimmed} chars from middle of debrief; call read_file on paths.debrief for full content...]\n\n"
                    + text[-tail:]
                )
        except OSError:
            debrief_excerpt = None

    return {
        "id": exp.id,
        "name": exp.name,
        "status": exp.status,
        "worker_id": exp.worker_id,
        "slurm_job_id": exp.slurm_job_id,
        "priority": exp.priority,
        "parked_at": exp.parked_at,
        "annotation": annotation,
        "description": description,
        "hypothesis": hypothesis,
        "config_summary": config_summary,
        "metrics_summary": metrics_summary,
        "error": error,
        "fix_attempts": exp.fix_attempts,
        "created_at": exp.created_at,
        "updated_at": exp.updated_at,
        "started_at": exp.started_at,
        "finished_at": exp.finished_at,
        "paths": paths,
        "flags": flags,
        "debrief_excerpt": debrief_excerpt,
    }


def _existing_path(candidates: list[Path]) -> str | None:
    for p in candidates:
        if p.exists():
            return str(p)
    return None
