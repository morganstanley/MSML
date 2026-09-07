"""Layer 1 — build a per-run evidence pack from post-run artifacts.

One evidence pack (a single JSON file) per run, assembled in one streamed
pass over each large artifact. Performance contract, per file kind:

- ``events.jsonl`` (0.3–0.6 GB per run): ``api_request`` / ``agent_text``
  lines dominate the bytes; they are counted and size-sampled but never
  JSON-parsed here.
- per-agent transcripts (``logs/*.jsonl``, ``.gz`` accepted): every line's
  type is sniffed from its head; ``api_request`` lines ARE parsed (they
  carry the model identity and payload composition), everything else is
  parsed only for the small event types.
- ``experiments.db`` / contract ``metrics.json`` files: read fully (small).

Every section of the pack carries a ``sources`` list naming the files it was
built from, so findings built on the pack can be fact-checked back to disk.
Packs are versioned (``PACK_SCHEMA``); the extract stage re-extracts any
cached pack written by an older extractor or truncated by a killed worker.
Two run layouts are supported natively — in-house workspaces (experiments
database + transcripts) and external evidence-contract submissions
(``experiments/<name>/results/metrics.json``); both feed one shared rollup
so every downstream metric means the same thing for every harness.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from alpha_lab.benchmarks.runcmp.corpus import (
    INFLIGHT_STATUSES,
    RunRecord,
    load_registry,
)

# ---------------------------------------------------------------------------
# Shared normalization helpers
# ---------------------------------------------------------------------------

_ERROR_PREFIXES = ("[error]", "tool execution failed", "error executing tool")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_PATH_RE = re.compile(r"(/[\w.\-]+)+")
_HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b")

_VALIDATION_ID_KEYS = (
    "validation_content_sha256",
    "validation_fingerprint",
    "validation_sha256",
    "val_content_sha256",
    "split_fingerprint",
    # external evidence-contract runs tag every score with the frozen slice
    # it was computed on (same-slice = comparable; different = not)
    "val_slice_id",
)

_HTTP_RE = re.compile(
    r'HTTP Request: \w+ (?P<url>\S+) "HTTP/[\d.]+ (?P<code>\d{3})'
)
_EXIT_RE = re.compile(r"=== \S+ EXIT code=(?P<code>-?\d+)")
_LAUNCH_RE = re.compile(r"^launch (?P<dt>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_CLOCK_MIN_RE = re.compile(r"^(\d{2}:\d{2})")


def normalize_signature(text: str, limit: int = 240) -> str:
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    first = _PATH_RE.sub("<path>", first)
    first = _HEX_RE.sub("<hex>", first)
    first = _NUM_RE.sub("<n>", first)
    return first[:limit]


def tool_failure_signature(output: str) -> str | None:
    """A tool result counts as a failure only on standardized markers."""
    head = (output or "").lstrip()[:80].lower()
    if any(head.startswith(p) for p in _ERROR_PREFIXES):
        return normalize_signature(output)
    return None


def _usage_tokens(usage: dict) -> dict[str, int]:
    """Four usage shapes exist in run event logs: the OpenAI Responses dump
    (nested ``input_tokens_details``/``output_tokens_details``), the
    agent-loop typed subset written for providers without a raw dump
    (flat ``cache_read_input_tokens``/``reasoning_tokens`` — Bedrock, grok),
    the raw Anthropic Messages dump (``cache_creation_input_tokens``;
    ``input_tokens`` there EXCLUDES cache reads/writes by provider
    semantics — preserved verbatim, normalization is token_accounting's
    job), and the raw Chat Completions dump (``prompt_tokens``/
    ``completion_tokens`` — lab-gateway/vLLM models; found all-zero-parsed
    on the 2026-08-05 d7 corpus for 6 of 10 runs). Read all; nested wins
    when present. Also accept Bedrock Converse camelCase keys
    (``cacheReadInputTokens``) in case a raw Converse usage dump lands in
    the events."""
    if not isinstance(usage, dict):
        return {}
    ind = usage.get("input_tokens_details") or {}
    outd = usage.get("output_tokens_details") or {}
    # Chat Completions nested detail blocks (OpenAI chat API & vLLM/LiteLLM)
    pd = usage.get("prompt_tokens_details") or {}
    cd = usage.get("completion_tokens_details") or {}
    return {
        "input": int(usage.get("input_tokens")
                     or usage.get("inputTokens")
                     or usage.get("prompt_tokens") or 0),
        "output": int(usage.get("output_tokens")
                      or usage.get("outputTokens")
                      or usage.get("completion_tokens") or 0),
        "cache_read": int(ind.get("cached_tokens")
                          or usage.get("cache_read_input_tokens")
                          or usage.get("cacheReadInputTokens")
                          or pd.get("cached_tokens") or 0),
        "cache_write": int(ind.get("cache_write_tokens")
                           or usage.get("cache_write_input_tokens")
                           or usage.get("cacheWriteInputTokens")
                           or usage.get("cache_creation_input_tokens") or 0),
        "reasoning": int(outd.get("reasoning_tokens")
                         or usage.get("reasoning_tokens")
                         or cd.get("reasoning_tokens") or 0),
    }


def _acc(dst: dict[str, int], add: dict[str, int]) -> None:
    for k, v in add.items():
        dst[k] = dst.get(k, 0) + v


# ---------------------------------------------------------------------------
# events.jsonl
# ---------------------------------------------------------------------------

# Bulky types never parsed; everything else is.
_SKIP_TYPES = ('"api_request"', '"agent_text"')

_DATETIME_SNIFF = re.compile(rb'"datetime": "([^"]+)"')

# Errors meaning the harness stopped the job, not that the job misbehaved.
_LIMIT_KILL_RE = re.compile(
    r"\b(TIMEOUT|timed out|time limit|wall.?clock limit|killed by (?:the )?"
    r"(?:harness|dispatcher)|OOM|out of memory)\b", re.I)
_TYPE_SNIFF = re.compile(rb'"type":\s*"([a-z_]+)"')


def last_attempt_start(events_path: Path, run_tag: str) -> str | None:
    """Timestamp of the newest ``run_start`` marker for this run.

    Several runs share one event log, and a run restarted from scratch
    writes a second ``run_start`` under the same name. Everything before the
    newest marker belongs to the attempt that was abandoned; counting it
    would inflate that run's calls, tokens and cost.
    """
    marker = b'"run": "' + run_tag.encode() + b'"'
    newest = None
    with _open_log(events_path) as fh:
        for raw in fh:
            if b'"run_start"' not in raw[:60] or marker not in raw:
                continue
            try:
                dt = json.loads(raw).get("datetime")
            except (ValueError, UnicodeDecodeError):
                continue
            if dt and (newest is None or dt > newest):
                newest = dt
    return newest


def parse_events(events_path: Path, run_tag: str | None = None,
                 since_iso: str | None = None) -> dict:
    tokens: dict[str, int] = {}
    api_calls = 0
    api_request_lines = 0
    tool_calls: Counter = Counter()
    tool_failures: Counter = Counter()
    failure_signatures: Counter = Counter()
    tool_failure_signatures: dict[str, Counter] = defaultdict(Counter)
    tool_failure_examples: dict[str, str] = {}
    error_events = 0
    status_error_signatures: Counter = Counter()
    experiment_transitions: Counter = Counter()
    phase_records: list[dict] = []
    ts_min = ts_max = None
    phase_first_ts: dict[str, float] = {}
    api_points: list[tuple[float, int, int]] = []  # (ts, in, out) for phase attribution
    json_errors = 0
    run_start: dict = {}

    # One event log per task directory holds every run in it, so a run's
    # own lines have to be selected by name; without this every run in the
    # directory reports the same totals.
    tag_marker = b'"run": "' + run_tag.encode() + b'"' if run_tag else None
    since_marker = since_iso.encode() if since_iso else None
    skipped_other_run = skipped_earlier_attempt = 0

    saw_any_run_key = False

    with _open_log(events_path) as fh:
        for raw in fh:
            if tag_marker is not None and tag_marker not in raw:
                if not saw_any_run_key and b'"run": "' in raw:
                    saw_any_run_key = True
                skipped_other_run += 1
                continue
            if since_marker is not None:
                dm = _DATETIME_SNIFF.search(raw)
                if dm and dm.group(1) < since_marker:
                    skipped_earlier_attempt += 1
                    continue
            m = _TYPE_SNIFF.search(raw[:120])
            etype = m.group(1).decode() if m else ""
            if etype in ("api_request", "agent_text"):
                if etype == "api_request":
                    api_request_lines += 1
                continue
            try:
                ev = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                json_errors += 1
                continue
            ts = ev.get("timestamp")
            if isinstance(ts, (int, float)):
                ts_min = ts if ts_min is None else min(ts_min, ts)
                ts_max = ts if ts_max is None else max(ts_max, ts)
            if etype == "run_start":
                run_start = {
                    "config": ev.get("config"),
                    "workspace": ev.get("workspace"),
                    "datetime": ev.get("datetime"),
                }
            elif etype == "api_response":
                api_calls += 1
                u = _usage_tokens(ev.get("usage") or {})
                _acc(tokens, u)
                if isinstance(ts, (int, float)):
                    api_points.append((ts, u.get("input", 0), u.get("output", 0)))
            elif etype == "tool_call":
                tool_calls[str(ev.get("name") or "?")] += 1
            elif etype == "tool_result":
                name = str(ev.get("name") or "?")
                sig = tool_failure_signature(str(ev.get("output") or ""))
                if sig:
                    tool_failures[name] += 1
                    failure_signatures[sig] += 1
                    tool_failure_signatures[name][sig] += 1
                    tool_failure_examples.setdefault(name, sig)
            elif etype == "error":
                error_events += 1
                failure_signatures[
                    normalize_signature(str(ev.get("detail") or ev.get("error") or ""))
                ] += 1
            elif etype == "status":
                if ev.get("status") == "error":
                    status_error_signatures[
                        normalize_signature(str(ev.get("detail") or ""))
                    ] += 1
            elif etype == "experiment":
                experiment_transitions[str(ev.get("status") or "?")] += 1
            elif etype == "phase":
                phase = str(ev.get("phase") or "?")
                if isinstance(ts, (int, float)) and phase not in phase_first_ts:
                    phase_first_ts[phase] = ts
                phase_records.append(
                    {
                        "ts": ts,
                        "phase": phase,
                        "step": ev.get("step"),
                        "iteration": ev.get("iteration"),
                        "status": ev.get("status"),
                        "detail": str(ev.get("detail") or "")[:200],
                    }
                )

    # Contiguous phase windows: first event of each phase to the next phase's
    # first event (last phase runs to the end of the stream).
    ordered = sorted(phase_first_ts.items(), key=lambda kv: kv[1])
    windows: dict[str, dict] = {}
    for i, (phase, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else (ts_max or start)
        windows[phase] = {"start": start, "end": end, "seconds": round(end - start, 1)}
    phase_activity: dict[str, dict] = {
        p: {"api_calls": 0, "input": 0, "output": 0} for p in windows
    }
    unassigned = {"api_calls": 0, "input": 0, "output": 0}
    for ts, tin, tout in api_points:
        for phase, w in windows.items():
            if w["start"] <= ts < w["end"] or (
                phase == ordered[-1][0] and ts >= w["start"]
            ):
                act = phase_activity[phase]
                act["api_calls"] += 1
                act["input"] += tin
                act["output"] += tout
                break
        else:
            unassigned["api_calls"] += 1
            unassigned["input"] += tin
            unassigned["output"] += tout

    if (tag_marker is not None and skipped_other_run > 0
            and not saw_any_run_key):
        # Lines were dropped by the tag filter, yet none of the dropped lines
        # carries a `"run":` key at all — this is the single-run event-log
        # format that tags only its run_start line (shared multi-run logs tag
        # every line, so a foreign run's lines would have shown a key).
        # Zeroed counts here would be a parser artifact, not a world fact;
        # parse untagged and say so.
        result = parse_events(events_path, run_tag=None, since_iso=since_iso)
        result["run_tag_filter"] = run_tag
        result["run_tag_absent_in_events"] = True
        return result

    return {
        "sources": [str(events_path)],
        "run_tag_filter": run_tag,
        "attempt_start": since_iso,
        "lines_other_runs_skipped": skipped_other_run,
        "lines_earlier_attempt_skipped": skipped_earlier_attempt,
        "run_start": run_start,
        "api_calls": api_calls,
        "api_request_lines": api_request_lines,
        "tokens": tokens,
        "wall_seconds": round((ts_max - ts_min), 1) if ts_min is not None else None,
        "ts_min": ts_min,
        "ts_max": ts_max,
        "tools": dict(tool_calls.most_common()),
        "tool_failures": dict(tool_failures.most_common()),
        "tool_failure_signatures": {
            name: dict(sigs.most_common(15))
            for name, sigs in tool_failure_signatures.items()
        },
        "tool_failure_examples": tool_failure_examples,
        "failure_signatures": dict(failure_signatures.most_common(60)),
        "status_error_signatures": dict(status_error_signatures.most_common(30)),
        "error_events": error_events,
        "experiment_transitions": dict(experiment_transitions),
        "phase_windows": windows,
        "phase_activity": phase_activity,
        "unassigned_activity": unassigned,
        "phase_records": phase_records[-600:],
        "json_errors": json_errors,
    }


# ---------------------------------------------------------------------------
# ws/logs/*.jsonl (per-agent logs)
# ---------------------------------------------------------------------------

_ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("conductor", "conductor"),
    # verifier must precede the worker patterns: verifier transcripts are
    # named verifier_worker_* / verifier_select_* / verifier_arbiter_*, and
    # the bare "worker" pattern would claim them (2026-08-06: checked every
    # log filename in the corpus — none contains "verifier" except the
    # verifier's own, so this steals nothing back).
    ("verifier", "verifier"),
    # Real worker transcripts are named worker_worker_0_implement_<slug>;
    # the previous patterns ("worker_implement", "worker_analyze") were
    # dead substrings that never matched a real filename, so every worker
    # sub-role collapsed into "worker" — and msml's handoff transcripts
    # (whose own system prompt says "You are the User's Proxy") silently
    # contaminated the worker ledger. Found 2026-08-13 by the
    # seat-taxonomy study (plan_deep_notes/SEAT_TAXONOMY_EVIDENCE.md).
    # handoff must precede the bare "worker" pattern.
    ("handoff", "handoff"),
    ("worker_implement", "_implement"),
    ("worker_analyze", "_analyze"),
    ("worker_fix", "_fix_"),
    ("worker", "worker"),
    ("strategist", "strategist"),
    ("dispatcher", "dispatcher"),
    ("supervisor", "supervisor"),
    ("builder", "phase2_builder"),
    ("critic", "phase2_critic"),
    ("tester", "phase2_tester"),
    ("fixer", "fixer"),
    ("reporter", "reporter"),
    ("phase0", "phase0"),
    ("phase1", "phase1"),
    ("pipeline", "pipeline"),
    # "conversation" is agent.py's *default* transcript filename, not a role:
    # the only agent launched without an explicit log_name is the main
    # exploration loop (run.py), i.e. the Phase-1 explorer. Emit a
    # self-describing seat so reports never print the bare filename.
    ("phase1_explorer", "conversation"),
)


def _artifact_class(path: str) -> str | None:
    """Which written-product family a read targets, or None for code/data.

    Families: debrief (per-experiment write-ups), learnings (the accumulated
    lessons file), report (final/phase reports, notes, output documents),
    plan (plans/agendas/todo files). Deliberately coarse — the question is
    whether the written products get read at all, and by which seats."""
    low = path.lower()
    if not low.endswith((".md", ".txt", ".rst")):
        return None
    name = low.rsplit("/", 1)[-1]
    if "debrief" in name:
        return "debrief"
    if "learnings" in name:
        return "learnings"
    if re.search(r"plan|todo|agenda", name):
        return "plan"
    if ("/reports/" in low or "/notes/" in low or "/output/" in low
            or "/phase1/" in low or name.startswith(("report", "readme"))
            or "report" in name):
        return "report"
    return None


def role_for_log(name: str) -> str:
    lowered = name.lower()
    for role, pat in _ROLE_PATTERNS:
        if pat in lowered:
            return role
    return "other"


def _open_log(path: Path):
    """Binary line stream over a transcript, gzip-compressed or not.

    Finished runs get their larger transcripts compressed in place
    (observed 2026-08-06: a July run's conductor/worker logs and its
    token ledger were .jsonl.gz on disk); a plain-open reader silently
    drops those seats from the evidence.
    """
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _resolve_maybe_gz(path: Path) -> Path:
    """The registry records paths as they were at index time; a later
    compress-in-place sweep renames them to .gz (15 of the corpus's event
    logs, counted 2026-08-06). Follow the rename instead of crashing."""
    if not path.is_file() and path.with_name(path.name + ".gz").is_file():
        return path.with_name(path.name + ".gz")
    return path


_COMPOSITION_KEYS = ("instructions", "tool_schemas", "tool_results",
                     "tool_args", "thinking", "images", "assistant_text",
                     "user_text", "other")


def _subsample(vals: list, ndigits: int = 0, cap: int = 400) -> list:
    """Evenly subsample to ``cap`` values, rounding for pack compactness."""
    if not vals:
        return []
    step = max(1, len(vals) // cap)
    out = vals[::step][:cap]
    return [round(v, ndigits) if ndigits else int(round(v)) for v in out]


# Per-line envelope of an api_request event: constant "type" key plus a
# timestamp that changes every line. Content comparison starts after it.
_REQ_ENVELOPE_RE = re.compile(rb'^\{"type": "api_request", "timestamp": [0-9.]+,\s*')


def _common_prefix_len(a: bytes, b: bytes) -> int:
    """Length of the shared leading bytes of ``a`` and ``b``.

    Binary search with zero-copy memoryview compares — consecutive requests
    can be hundreds of KB and one run has thousands of pairs, so a per-byte
    Python loop is too slow.
    """
    va, vb = memoryview(a), memoryview(b)
    n = min(len(a), len(b))
    step = 65536
    i = 0
    while i < n and va[i:i + step] == vb[i:i + step]:
        i += step
    if i >= n:
        return n
    lo, hi = i, min(i + step, n)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if va[lo:mid] == vb[lo:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _add_block_composition(block: Any, role: str, comp: dict) -> None:
    """Classify one content block/item (Anthropic blocks, OpenAI Responses
    items, or chat-completions content parts) into billed-payload
    components, adding its serialized chars."""
    if not isinstance(block, dict):
        comp["other"] += len(str(block))
        return
    bt = str(block.get("type") or "")
    if bt in ("text", "input_text", "output_text"):
        key = "assistant_text" if role == "assistant" else "user_text"
        comp[key] += len(str(block.get("text") or ""))
    elif bt in ("thinking", "redacted_thinking", "reasoning"):
        comp["thinking"] += len(json.dumps(block, default=str))
    elif bt in ("tool_result", "function_call_output"):
        comp["tool_results"] += len(json.dumps(
            block.get("content", block.get("output")), default=str))
    elif bt in ("tool_use", "function_call"):
        comp["tool_args"] += len(json.dumps(
            block.get("input", block.get("arguments")), default=str)) \
            + len(str(block.get("name") or ""))
    elif bt in ("image", "input_image", "image_url") or "image_url" in block:
        comp["images"] += len(json.dumps(block, default=str))
    elif bt == "message":
        inner = block.get("content")
        if isinstance(inner, str):
            key = ("assistant_text" if block.get("role") == "assistant"
                   else "user_text")
            comp[key] += len(inner)
        else:
            for b in inner or []:
                _add_block_composition(b, str(block.get("role") or ""), comp)
    else:
        comp["other"] += len(json.dumps(block, default=str))


def _add_request_composition(ev: dict, comp: dict) -> None:
    """Accumulate one api_request's payload into per-component char counts.

    This is what the model was actually sent (and, without history caching,
    re-billed) — the deterministic basis for explaining cost gaps instead of
    leaving each evaluator to re-derive it differently.
    """
    comp["instructions"] += len(str(ev.get("instructions") or ""))
    comp["tool_schemas"] += len(json.dumps(ev.get("tools") or [], default=str))
    inp = ev.get("input")
    if isinstance(inp, str):
        comp["user_text"] += len(inp)
        return
    for msg in inp or []:
        if not isinstance(msg, dict):
            comp["other"] += len(str(msg))
            continue
        role = str(msg.get("role") or "")
        if not role and msg.get("type"):
            # OpenAI Responses top-level item (reasoning / function_call /
            # ...). A ZDR reasoning item carries EMPTY summary/content arrays
            # plus an opaque encrypted_content blob — the content walk below
            # iterates the empty array and counts nothing, silently dropping
            # 12-15% of a deep request's bytes (d5 sol runs, 2026-08-07).
            _add_block_composition(msg, role, comp)
            continue
        # Chat-completions dialect (lab endpoints: kimi/glm): tool output is a
        # role="tool" message with plain string content, arguments ride in a
        # "tool_calls" key, and replayed reasoning in "reasoning_content" —
        # none of which exist as typed blocks, so without these branches the
        # bytes fold into user_text/other and the tool_results/tool_args/
        # thinking/images fields sit at 0 for every lab-endpoint run.
        if msg.get("reasoning_content"):
            comp["thinking"] += len(str(msg["reasoning_content"]))
        for tc in msg.get("tool_calls") or []:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            comp["tool_args"] += (len(str(fn.get("name") or ""))
                                  + len(str(fn.get("arguments") or "")))
        c = msg.get("content")
        if role == "tool":
            comp["tool_results"] += (len(c) if isinstance(c, str)
                                     else len(json.dumps(c, default=str))
                                     if c is not None else 0)
        elif isinstance(c, str):
            comp["assistant_text" if role == "assistant" else "user_text"] += len(c)
        elif c is not None:
            for b in c or []:
                _add_block_composition(b, role, comp)
        elif not (msg.get("tool_calls") or msg.get("reasoning_content")):
            _add_block_composition(msg, role, comp)


def parse_agent_logs(logs_dir: Path) -> dict:
    roles: dict[str, dict] = defaultdict(
        lambda: {
            "files": 0,
            "api_calls": 0,
            "tokens": {},
            "models": {},
            "tool_calls": 0,
            "tool_failures": 0,
            "sessions_ended_clean": 0,
            "prompt_chars": [],
            "turns": [],
        }
    )
    tools: Counter = Counter()
    tool_fail: Counter = Counter()
    # Web searches are the exploration phases' main outward move; keep the
    # query text so a review can judge WHAT was searched, not only how often
    # (user ask 2026-08-11: phases 0-1 were invisible beyond hours + clicks)
    web_searches: list[dict] = []
    # Who READS the written products: every read/grep tool call names its
    # target; classifying those paths answers "which reports actually get
    # read, and by whom" (user ask 2026-08-11) — the causal half of report
    # quality, since debriefs/learnings are read mid-run by deciding seats.
    artifact_reads: Counter = Counter()
    per_file: list[dict] = []
    composition = {k: 0 for k in _COMPOSITION_KEYS}
    # Reasoning round-trip census (byte scan, no extra JSON parse): a model
    # that PRODUCES reasoning in responses but never RECEIVES it back in
    # requests is running blind to its own traces — the defect class that
    # hid GLM/deepseek's missing replay until 2026-08-07. Markers cover all
    # three dialects: reasoning_content (chat/kimi), type:thinking
    # (Anthropic), encrypted_content (OpenAI ZDR items).
    roundtrip = {"requests_total": 0, "requests_with_reasoning": 0,
                 "responses_total": 0, "responses_with_reasoning": 0}
    session_first: list[int] = []
    session_last: list[int] = []
    largest_request = 0
    fresh_growth: list[float] = []
    cache_growth: list[float] = []
    # One fixed replay definition for the whole pipeline (2026-07-31: two
    # reviewer probes measured "replay" with different definitions and
    # published 92.96% vs 97.51% for the same run under one name).
    replay_shares: list[float] = []
    # Raw distribution samples (bounded at pack time). Reports are required
    # to show distributions, not just medians — these give every reviewer
    # the same raw material so per-probe re-derivations stop diverging.
    s_req_bytes: list[int] = []
    s_tool_result_bytes: list[int] = []
    s_llm_gap: list[float] = []
    s_tool_gap: list[float] = []
    s_session_min: list[float] = []
    files: list[Path] = []
    if logs_dir.is_dir():
        files = sorted(logs_dir.glob("*.jsonl"))
        plain = {p.name for p in files}
        # compressed-in-place transcripts; a .gz twin of a still-present
        # plain file would double-count, so the plain one wins
        files += sorted(p for p in logs_dir.glob("*.jsonl.gz")
                        if p.name[:-3] not in plain)
    for path in files:
        role = role_for_log(path.name)
        r = roles[role]
        r["files"] += 1
        calls = tcalls = tfails = invocations = 0
        # Seat outcome, read from the seat's OWN transcript. The token ledger
        # only records calls that RETURNED, so a seat killed before its first
        # response leaves no ledger trace at all — reading seat outcomes from
        # the ledger silently converts "destroyed by the service" into "the
        # model chose not to run this seat" (2026-07-30: six verifier seats in
        # d2_o5_cond died on capacity refusals and I reported them as a
        # behavioral difference between models).
        capacity_refusals = 0
        seat_errors = 0
        toks: dict[str, int] = {}
        # who sat in this seat: model named on each logged request (every
        # seat can run a different LLM; 2026-08-06 the answer to "which
        # model steered this run?" was not recoverable from the viewer)
        fmodels: Counter = Counter()
        prompt_chars = None
        last_tool = ""
        input_first = input_last = None
        input_sum = 0
        input_seq: list[int] = []
        cache_seq: list[int] = []
        req_sizes: list[int] = []
        prev_req: bytes | None = None
        last_req_ts = last_tool_ts = None
        first_ts = last_ts = None
        with _open_log(path) as fh:
            for raw in fh:
                m = _TYPE_SNIFF.search(raw[:120])
                etype = m.group(1).decode() if m else ""
                if etype == "api_request":
                    req_sizes.append(len(raw))
                    s_req_bytes.append(len(raw))
                    roundtrip["requests_total"] += 1
                    if (b'"reasoning_content"' in raw
                            or b'"type": "thinking"' in raw
                            or b'"encrypted_content"' in raw):
                        roundtrip["requests_with_reasoning"] += 1
                    # Strip the per-line envelope (type + timestamp) before
                    # comparing: the timestamp differs on every line, so a
                    # raw-line prefix would measure timestamp divergence
                    # (measured: 0.0003 median) instead of content replay.
                    env = _REQ_ENVELOPE_RE.match(raw)
                    body = raw[env.end():] if env else raw
                    if prev_req is not None and len(body) > 0:
                        replay_shares.append(
                            _common_prefix_len(prev_req, body) / len(body))
                    prev_req = body
                    try:
                        ev = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    ts = ev.get("timestamp")
                    if isinstance(ts, (int, float)):
                        last_req_ts = ts
                        first_ts = first_ts if first_ts is not None else ts
                        last_ts = ts
                    if prompt_chars is None:
                        prompt_chars = len(str(ev.get("instructions") or ""))
                    if ev.get("model"):
                        fmodels[str(ev["model"])] += 1
                    _add_request_composition(ev, composition)
                    continue
                if etype == "error":
                    seat_errors += 1
                    if b"overloaded_error" in raw:
                        capacity_refusals += 1
                    continue
                if etype not in ("api_response", "tool_call", "tool_result",
                                 "status"):
                    continue
                if b"overloaded_error" in raw:
                    capacity_refusals += 1
                try:
                    ev = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                ts = ev.get("timestamp")
                if isinstance(ts, (int, float)):
                    first_ts = first_ts if first_ts is not None else ts
                    last_ts = ts
                if etype == "status":
                    if ev.get("status") == "starting":
                        invocations += 1
                elif etype == "api_response":
                    calls += 1
                    roundtrip["responses_total"] += 1
                    if (b'"reasoning_content"' in raw
                            or b'"type": "thinking"' in raw
                            or b'"encrypted_content"' in raw):
                        roundtrip["responses_with_reasoning"] += 1
                    if (isinstance(ts, (int, float))
                            and last_req_ts is not None
                            and 0 <= ts - last_req_ts < 3600):
                        s_llm_gap.append(ts - last_req_ts)
                    last_req_ts = None
                    usage = _usage_tokens(ev.get("usage") or {})
                    _acc(toks, usage)
                    inp = usage.get("input", 0)
                    if input_first is None:
                        input_first = inp
                    input_last = inp
                    input_sum += inp
                    if len(input_seq) < 300:
                        input_seq.append(inp)
                    if len(cache_seq) < 300:
                        cache_seq.append(usage.get("cache_read", 0))
                elif etype == "tool_call":
                    tcalls += 1
                    if isinstance(ts, (int, float)):
                        last_tool_ts = ts
                    last_tool = str(ev.get("name") or "?")
                    tools[f"{role}:{last_tool}"] += 1
                    if last_tool in ("read_file", "grep_file", "cat"):
                        a_ = (ev.get("args") or ev.get("arguments")
                              or ev.get("input") or {})
                        if isinstance(a_, str):
                            try:
                                a_ = json.loads(a_)
                            except (json.JSONDecodeError, TypeError):
                                a_ = {}
                        path_ = str((a_ or {}).get("path")
                                    or (a_ or {}).get("file") or "")
                        cls_ = _artifact_class(path_)
                        if cls_:
                            artifact_reads[f"{role}:{cls_}"] += 1
                    if ("web_search" in last_tool
                            and len(web_searches) < 300):
                        args_ = (ev.get("args") or ev.get("arguments")
                                 or ev.get("input") or {})
                        if isinstance(args_, str):
                            try:
                                args_ = json.loads(args_)
                            except (json.JSONDecodeError, TypeError):
                                args_ = {"raw": args_[:200]}
                        web_searches.append({
                            "seat": role,
                            "query": str((args_ or {}).get("query")
                                         or (args_ or {}).get("q")
                                         or (args_ or {}).get("raw")
                                         or "")[:300]})
                elif etype == "tool_result":
                    s_tool_result_bytes.append(len(raw))
                    if (isinstance(ts, (int, float))
                            and last_tool_ts is not None
                            and 0 <= ts - last_tool_ts < 3600):
                        s_tool_gap.append(ts - last_tool_ts)
                    last_tool_ts = None
                    name = str(ev.get("name") or last_tool or "?")
                    if tool_failure_signature(str(ev.get("output") or "")):
                        tfails += 1
                        tool_fail[f"{role}:{name}"] += 1
        r["api_calls"] += calls
        _acc(r["tokens"], toks)
        for mdl, n in fmodels.items():
            r["models"][mdl] = r["models"].get(mdl, 0) + n
        r["tool_calls"] += tcalls
        r["tool_failures"] += tfails
        r["invocations"] = r.get("invocations", 0) + invocations
        if prompt_chars is not None:
            r["prompt_chars"].append(prompt_chars)
        r["turns"].append(calls)
        if last_tool and re.search(r"report|finish|complete|submit|done", last_tool):
            r["sessions_ended_clean"] += 1
        if req_sizes:
            session_first.append(req_sizes[0])
            session_last.append(req_sizes[-1])
            largest_request = max(largest_request, max(req_sizes))
        if len(input_seq) >= 6:
            fresh_growth.append(
                sum(input_seq[-3:]) / 3 - sum(input_seq[:3]) / 3)
            cache_growth.append(
                sum(cache_seq[-3:]) / 3 - sum(cache_seq[:3]) / 3)
        if first_ts is not None and last_ts is not None and last_ts > first_ts:
            s_session_min.append((last_ts - first_ts) / 60)
        per_file.append(
            {
                "file": path.name,
                "role": role,
                "api_calls": calls,
                "tokens": toks,
                "models": dict(fmodels),
                "tool_calls": tcalls,
                "tool_failures": tfails,
                "invocations": invocations,
                "input_first": input_first,
                "input_last": input_last,
                "input_sum": input_sum,
                "input_seq": input_seq[:300],
                "request_chars_first": req_sizes[0] if req_sizes else None,
                "request_chars_last": req_sizes[-1] if req_sizes else None,
                "request_chars_max": max(req_sizes) if req_sizes else None,
                # Seat outcome from this seat's own transcript (never the
                # ledger): a seat that issued requests but never got a
                # response was killed, not skipped.
                "requests_issued": len(req_sizes),
                "capacity_refusals": capacity_refusals,
                "errors": seat_errors,
                "died_before_first_response": bool(req_sizes) and calls == 0,
            }
        )
    for r in roles.values():
        r["prompt_chars"] = sorted(r["prompt_chars"])
        r["turns"] = sorted(r["turns"])

    def _median(vals: list) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        n = len(s)
        return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)

    med_first = _median(session_first)
    med_last = _median(session_last)
    return {
        "sources": [str(logs_dir)],
        "roles": {k: dict(v) for k, v in sorted(roles.items())},
        "tools_by_role": dict(tools.most_common()),
        "tool_failures_by_role": dict(tool_fail.most_common()),
        "web_searches": web_searches,
        "artifact_reads": dict(artifact_reads.most_common()),
        "files": per_file,
        # Billed-payload decomposition, summed over every request every agent
        # sent — the canonical basis for explaining any cost/volume gap
        # (definition: serialized chars per component; sessions = one jsonl).
        "request_composition_chars": composition,
        "reasoning_roundtrip": roundtrip,
        # The pipeline's ONE replay definition. Any report language about
        # "replay" or "unchanged prefix" must cite this measure.
        # Bounded raw samples for distribution charts (hist/density/box in
        # reports). Evenly subsampled to <=400 values each; units in names.
        "samples": {
            "definition": ("evenly subsampled raw values from this run's "
                           "agent transcripts; llm_gap = api_request to next "
                           "api_response in one transcript, tool_gap = "
                           "tool_call to next tool_result, session_minutes = "
                           "one value per agent transcript file"),
            "request_bytes": _subsample(s_req_bytes),
            "tool_result_bytes": _subsample(s_tool_result_bytes),
            "llm_gap_seconds": _subsample(s_llm_gap, 3),
            "tool_gap_seconds": _subsample(s_tool_gap, 3),
            "session_minutes": _subsample(s_session_min, 2),
        },
        "replay": {
            "definition": ("shared leading bytes of consecutive api_request "
                           "lines within one agent jsonl (after stripping "
                           "each line's type/timestamp envelope), divided by "
                           "the current line's bytes; 1.0 = the previous "
                           "request reappears verbatim as a prefix"),
            "consecutive_pairs": len(replay_shares),
            "median_prefix_share": (round(_median(replay_shares), 4)
                                    if replay_shares else None),
            "p90_prefix_share": (round(sorted(replay_shares)[
                                     int(0.9 * (len(replay_shares) - 1))], 4)
                                 if replay_shares else None),
        },
        "request_stats": {
            "definition": ("per session (one agent jsonl): serialized chars "
                           "of the first/last api_request line; growth = "
                           "median last / median first"),
            "sessions_measured": len(session_first),
            "session_first_chars_median": med_first,
            "session_last_chars_median": med_last,
            "session_growth_ratio": (round(med_last / med_first, 2)
                                     if med_first and med_last else None),
            "largest_request_chars": largest_request or None,
        },
        "cache_pattern": {
            "definition": ("per session with >=6 calls: mean of last 3 minus "
                           "mean of first 3, for fresh input tokens and for "
                           "cache-read tokens; medians across sessions. "
                           "Growing fresh with flat cache-read = the "
                           "conversation history is NOT being cached"),
            "fresh_input_growth_median": _median(fresh_growth),
            "cache_read_growth_median": _median(cache_growth),
        },
        "seat_outcomes": _seat_outcomes(per_file),
    }


def _seat_outcomes(per_file: list[dict]) -> dict:
    """Which agent sessions ran, and which were killed before answering.

    Read from each seat's own transcript. ``died_before_first_response``
    means the seat issued at least one request and never received a
    response — the service refused or the connection failed until retries
    ran out. That is a service outcome, NOT a decision by the model, and it
    is invisible in the token ledger (no response = no ledger row).
    """
    verifier = [f for f in per_file if str(f.get("file", "")).startswith("verifier")]

    def _tally(rows: list[dict]) -> dict:
        started = len(rows)
        answered = sum(1 for f in rows if (f.get("api_calls") or 0) > 0)
        died = sum(1 for f in rows if f.get("died_before_first_response"))
        return {
            "seats_started": started,
            "seats_answered": answered,
            "seats_died_before_first_response": died,
            "completion_fraction": (round(answered / started, 4)
                                    if started else None),
            "capacity_refusals": sum(f.get("capacity_refusals") or 0
                                     for f in rows),
        }

    out = {
        "definition": ("per agent session (one jsonl): a seat 'answered' if "
                       "it received >=1 api_response; 'died_before_first_"
                       "response' if it issued a request and got none. "
                       "capacity_refusals counts 529 overloaded_error "
                       "responses from the serving side"),
        "all_seats": _tally(per_file),
    }
    # Every agent type gets the same breakdown — conductor, strategist,
    # workers, phase agents, supervisor, reporter, verifier. A capacity
    # failure that lands on the conductor (steering) or the strategist
    # (proposals) distorts a run just as badly as one that lands on the
    # verifier, and grouping only the verifier would have hidden those.
    by_role: dict[str, list[dict]] = defaultdict(list)
    for f in per_file:
        by_role[str(f.get("role") or "other")].append(f)
    out["by_role"] = {}
    for role, rows in sorted(by_role.items()):
        t = _tally(rows)
        dead = sorted(str(f.get("file", ""))[:-6] for f in rows
                      if f.get("died_before_first_response"))
        if dead:
            t["dead_seat_names"] = dead[:20]
        out["by_role"][role] = t
    if verifier:
        out["verifier"] = _tally(verifier)
        out["verifier"]["dead_seat_names"] = sorted(
            str(f.get("file", ""))[:-6] for f in verifier
            if f.get("died_before_first_response"))
    return out


# ---------------------------------------------------------------------------
# run.log
# ---------------------------------------------------------------------------

def parse_run_log(path: Path) -> dict:
    endpoints: dict[str, dict] = defaultdict(
        lambda: {"requests": 0, "codes": Counter(), "per_minute": Counter()}
    )
    launches: list[str] = []
    exit_codes: list[int] = []
    tracebacks = 0
    retries = 0
    # Error lines, normalized and counted, so infrastructure failures are
    # visible in the pack instead of only in the raw log.
    error_signatures: Counter = Counter()
    phase2_abortions = 0
    dispatcher_crashes = 0
    strategist_stalls = 0
    agent_stops = 0
    # Console logs get compressed in place like every other finished-run
    # artifact; a plain text open on a .gz would "parse" gzip bytes into
    # zero counts instead of failing.
    with io.TextIOWrapper(_open_log(path), errors="ignore") as fh:
        for line in fh:
            m = _HTTP_RE.search(line)
            if m:
                endpoint = m.group("url").rstrip('/').rsplit("/", 1)[-1]
                e = endpoints[endpoint]
                e["requests"] += 1
                e["codes"][m.group("code")] += 1
                cm = _CLOCK_MIN_RE.match(line)
                if cm:
                    e["per_minute"][cm.group(1)] += 1
                continue
            if " ERROR " in line or line.startswith(("mlflow.exceptions.",
                    "OSError:", "RuntimeError:", "requests.exceptions.")):
                sig = normalize_signature(line.strip())[:160]
                if sig:
                    error_signatures[sig] += 1
            if line.startswith("launch "):
                launches.append(line.strip()[:200])
                # A launch marker starts a fresh generation: the counters
                # describe the final (clean) generation only, so an aborted
                # pre-start sharing the log file must not charge the run.
                # Logs without markers (other launchers) keep cumulative
                # counts across their appended invocations, as before.
                endpoints.clear()
                tracebacks = retries = phase2_abortions = 0
                dispatcher_crashes = strategist_stalls = agent_stops = 0
            elif "EXIT code=" in line:
                em = _EXIT_RE.search(line)
                if em:
                    exit_codes.append(int(em.group("code")))
            elif line.startswith("Traceback (most recent call last):"):
                tracebacks += 1
            elif "Retrying request" in line or "Retrying in" in line:
                retries += 1
            elif "Max fix iterations reached" in line and "Phase 2" in line:
                phase2_abortions += 1
            elif "Phase 3 dispatcher crashed unexpectedly" in line:
                dispatcher_crashes += 1
            elif "Strategist stall:" in line:
                strategist_stalls += 1
            elif "Agent stopped unexpectedly" in line:
                agent_stops += 1
    summary = {}
    for name, e in endpoints.items():
        peak = e["per_minute"].most_common(1)
        summary[name] = {
            "requests": e["requests"],
            "codes": dict(e["codes"]),
            "rate_limited": e["codes"].get("429", 0),
            "peak_per_minute": peak[0][1] if peak else 0,
        }
    return {
        "error_signatures": dict(error_signatures.most_common(25)),
        "sources": [str(path)],
        "launches": launches,
        "exit_codes": exit_codes,
        "final_exit_code": exit_codes[-1] if exit_codes else None,
        "http_endpoints": summary,
        "retries": retries,
        "tracebacks": tracebacks,
        "phase2_abortions": phase2_abortions,
        "dispatcher_crashes": dispatcher_crashes,
        "strategist_stalls": strategist_stalls,
        "agent_stops": agent_stops,
    }


# ---------------------------------------------------------------------------
# experiments.db + per-experiment result files
# ---------------------------------------------------------------------------

_REALIZATION_MARKER_RE = re.compile(r"(job.*\.out$|_job\.out$|exit_code)")
_PRIMARY_RESULT_RE = re.compile(r"(^metrics\.json$|predictions.*\.npz$|forecast.*\.npz$)")
_REALIZATION_GAP_S = 300
_SMOKE_PATH_RE = re.compile(r"smoke")


def scan_realizations(exp_dir: Path, started_at, finished_at) -> dict | None:
    """Physical execution evidence inside one experiment directory.

    A DB row records one lifecycle, but the directory keeps every launch:
    execution-completion markers (``*job*.out``, ``*exit_code``) cluster into
    *realizations* (>5 min apart). Extra ``metrics.json`` generations outside
    ``results/`` are earlier replaced outputs. Post-finish writes to primary
    data files (metrics/predictions/forecasts) are mutation-after-completion;
    analysis additions (plots, summaries) are deliberately not counted.
    Failed pre-start launches contribute their last output line as a
    classifiable failure signature.
    """
    if not exp_dir.is_dir():
        return None
    marker_times: list[float] = []
    failed_launch_files: list[Path] = []
    extra_metrics = 0
    primary_mutations: list[str] = []
    for root, dirs, files in os.walk(exp_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        rel_root = os.path.relpath(root, exp_dir)
        for fname in files:
            path = Path(root) / fname
            rel = fname if rel_root == "." else f"{rel_root}/{fname}"
            try:
                st = path.stat()
            except OSError:
                continue
            if _REALIZATION_MARKER_RE.search(fname) and not _SMOKE_PATH_RE.search(rel):
                marker_times.append(st.st_mtime)
                if (
                    fname.endswith(".out")
                    and isinstance(started_at, (int, float))
                    and st.st_mtime < started_at - 60
                    and st.st_size < 100_000
                ):
                    failed_launch_files.append(path)
            if fname == "metrics.json" and not _SMOKE_PATH_RE.search(rel):
                parts = rel.split("/")
                if not (len(parts) == 2 and parts[0] == "results"):
                    extra_metrics += 1
            if (
                rel.startswith("results/")
                and _PRIMARY_RESULT_RE.search(fname)
                and not _SMOKE_PATH_RE.search(rel)
                and isinstance(finished_at, (int, float))
                and st.st_mtime > finished_at + 60
            ):
                primary_mutations.append(rel)
    realizations = 0
    last = None
    for t in sorted(marker_times):
        if last is None or t - last > _REALIZATION_GAP_S:
            realizations += 1
        last = t
    launch_failure_signatures: Counter = Counter()
    error_line_re = re.compile(r"(error|exception|traceback|failed)", re.I)
    for path in failed_launch_files[:20]:
        try:
            with open(path, "rb") as fh:
                tail = fh.read(100_000).decode(errors="replace")
        except OSError:
            continue
        lines = [l.strip() for l in tail.splitlines() if l.strip()]
        hit = None
        for i, line in enumerate(lines):
            if error_line_re.search(line):
                hit = line
                # An "XyzError:" line usually carries its message on the
                # following line; join them for a meaningful signature.
                if line.rstrip().endswith(":") and i + 1 < len(lines):
                    hit = f"{line} {lines[i + 1]}"
        if hit:
            launch_failure_signatures[normalize_signature(hit)] += 1
    return {
        "realizations": realizations,
        "extra_metrics_generations": extra_metrics,
        "primary_results_mutated_after_finish": primary_mutations[:6],
        "launch_failure_signatures": dict(launch_failure_signatures.most_common(5)),
    }


_CODE_FILE_CAP = 200            # .py files scanned per experiment dir
_CODE_FILE_MAX_BYTES = 1_000_000
_CODE_SKIP_DIRS = {"__pycache__", ".git", "venv", ".venv", "node_modules",
                   "checkpoints", "wandb"}
_CTRL_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With)


def scan_code_metrics(exp_dir: Path) -> dict | None:
    """Deterministic code-quality metrics for the Python an experiment shipped.

    Line counts never fail; AST metrics (functions, lengths, branches,
    nesting) are per-file and a file that does not parse is *counted* as a
    parse failure rather than silently skipped — "no functions" and
    "unparseable" must stay distinguishable. Pure function of the files on
    disk: no model, no sampling, no environment dependence.
    """
    if not exp_dir.is_dir():
        return None
    py_files: list[Path] = []
    for root, dirs, files in os.walk(exp_dir):
        dirs[:] = [d for d in dirs if d not in _CODE_SKIP_DIRS]
        for f in sorted(files):
            if f.endswith(".py"):
                py_files.append(Path(root) / f)
        if len(py_files) >= _CODE_FILE_CAP:
            py_files = py_files[:_CODE_FILE_CAP]
            break
    if not py_files:
        return None
    total = code = comment = blank = 0
    n_funcs = 0
    func_lens: list[int] = []
    branches = 0
    max_nesting = 0
    parse_failures = 0
    for path in sorted(py_files):
        try:
            if path.stat().st_size > _CODE_FILE_MAX_BYTES:
                continue
            src = path.read_text(errors="replace")
        except OSError:
            continue
        for line in src.splitlines():
            total += 1
            s = line.strip()
            if not s:
                blank += 1
            elif s.startswith("#"):
                comment += 1
            else:
                code += 1
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError, RecursionError):
            parse_failures += 1
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_funcs += 1
                end = getattr(node, "end_lineno", None)
                if isinstance(end, int):
                    func_lens.append(end - node.lineno + 1)
            if isinstance(node, _CTRL_NODES + (ast.BoolOp, ast.IfExp,
                                               ast.ExceptHandler,
                                               ast.comprehension)):
                branches += 1
        # Max control-flow nesting depth, iteratively (no recursion limit).
        stack: list[tuple[ast.AST, int]] = [(tree, 0)]
        while stack:
            node, d = stack.pop()
            nd = d + 1 if isinstance(node, _CTRL_NODES) else d
            if nd > max_nesting:
                max_nesting = nd
            stack.extend((child, nd) for child in ast.iter_child_nodes(node))
    denom = code + comment
    return {
        "py_files": len(py_files),
        "total_lines": total,
        "code_lines": code,
        "comment_lines": comment,
        "blank_lines": blank,
        "comment_share": round(comment / denom, 4) if denom else None,
        "functions": n_funcs,
        "mean_function_lines": (
            round(sum(func_lens) / len(func_lens), 1) if func_lens else None
        ),
        "max_function_lines": max(func_lens) if func_lens else None,
        "branch_nodes": branches,
        "branches_per_100_code_lines": (
            round(100.0 * branches / code, 2) if code else None
        ),
        "max_control_nesting": max_nesting,
        "ast_parse_failures": parse_failures,
    }


def _metric_config(workspace: Path) -> dict:
    manifest = workspace / "adapter" / "manifest.json"
    out = {"metric_key": "metric", "direction": "maximize", "source": None}
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text())
            metric = data.get("metric") or {}
            out["metric_key"] = (
                metric.get("extract_key") or metric.get("primary_metric") or "metric"
            )
            out["direction"] = metric.get("direction") or "maximize"
            out["source"] = str(manifest)
        except (json.JSONDecodeError, OSError):
            pass
    return out


def read_experiments(db_path: Path, workspace: Path) -> dict:
    mc = _metric_config(workspace)
    metric_key, direction = mc["metric_key"], mc["direction"]
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM experiments ORDER BY id")]
    columns = [r[1] for r in con.execute("PRAGMA table_info(experiments)")]
    con.close()

    experiments = []
    missing_result_files = 0
    db_file_metric_mismatches = 0
    results_replaced_after_finish = 0
    for row in rows:
        results = {}
        rj = row.get("results_json")
        if rj:
            try:
                results = json.loads(rj)
            except (json.JSONDecodeError, TypeError):
                results = {}
        file_metrics = {}
        file_mtime = None
        name = str(row.get("name") or "")
        fpath = workspace / "experiments" / name / "results" / "metrics.json"
        if fpath.is_file():
            try:
                file_mtime = fpath.stat().st_mtime
                file_metrics = json.loads(fpath.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        elif rj and name:
            missing_result_files += 1
        db_metric = results.get(metric_key)
        file_metric = file_metrics.get(metric_key)
        metric_mismatch = (
            isinstance(db_metric, (int, float))
            and isinstance(file_metric, (int, float))
            and not isinstance(db_metric, bool)
            and not isinstance(file_metric, bool)
            and not math.isclose(float(db_metric), float(file_metric),
                                 rel_tol=1e-9, abs_tol=1e-12)
        )
        if metric_mismatch:
            db_file_metric_mismatches += 1
        replaced_after_finish = (
            file_mtime is not None
            and isinstance(row.get("finished_at"), (int, float))
            and file_mtime > float(row["finished_at"]) + 300
        )
        if replaced_after_finish:
            results_replaced_after_finish += 1
        merged = {**results, **file_metrics}  # file wins
        metric = merged.get(metric_key)
        if not isinstance(metric, (int, float)) or isinstance(metric, bool):
            metric = None
        validation_id = None
        for key in _VALIDATION_ID_KEYS:
            v = merged.get(key)
            if v:
                validation_id = f"{key}:{v}"
                break
        flags = {
            k: merged.get(k)
            for k in ("valid_score", "ranking_protocol_valid", "valid", "smoke_test",
                      "smoke", "partial", "param_count", "wall_clock_seconds",
                      "device")
            if k in merged
        }
        started, finished = row.get("started_at"), row.get("finished_at")
        duration = None
        if isinstance(started, (int, float)) and isinstance(finished, (int, float)):
            if finished >= started:
                duration = round(finished - started, 1)
        realization = scan_realizations(
            workspace / "experiments" / name, started, finished
        ) if name else None
        code_metrics = scan_code_metrics(
            workspace / "experiments" / name
        ) if name else None
        experiments.append(
            {
                "id": row.get("id"),
                "name": name,
                "status": row.get("status"),
                "parent_id": row.get("parent_id"),
                "priority": row.get("priority"),
                "parked_at": row.get("parked_at"),
                "created_at": row.get("created_at"),
                "started_at": started,
                "finished_at": finished,
                "duration_seconds": duration,
                "fix_attempts": row.get("fix_attempts"),
                "error": normalize_signature(str(row.get("error") or ""))
                if row.get("error")
                else None,
                "metric": metric,
                "validation_id": validation_id,
                "flags": flags,
                # A run the harness cut off at its wall-clock limit never
                # reached the point where it writes its evidence files. That
                # is not the same as a run that finished and ignored the
                # requirement, and the two must not be counted together.
                "cut_off_by_limit": bool(
                    _LIMIT_KILL_RE.search(str(row.get("error") or ""))
                ),
                "metric_mismatch": bool(metric_mismatch),
                "replaced_after_finish": bool(replaced_after_finish),
                "realization": realization,
                "code": code_metrics,
                "hypothesis": str(row.get("hypothesis") or "")[:300],
            }
        )

    return _aggregate_experiments(
        experiments, metric_key=metric_key, direction=direction,
        sources=[str(db_path), str(workspace / "experiments")],
        columns=columns,
        missing_result_files=missing_result_files,
        db_file_metric_mismatches=db_file_metric_mismatches,
        results_replaced_after_finish=results_replaced_after_finish,
    )


def _aggregate_experiments(experiments: list[dict], *, metric_key: str,
                           direction: str, sources: list[str],
                           columns: list[str],
                           missing_result_files: int = 0,
                           db_file_metric_mismatches: int = 0,
                           results_replaced_after_finish: int = 0) -> dict:
    """Shared rollup over experiment rows — one implementation whether the
    rows came from an experiments.db (msml/cond) or from an external
    harness's evidence-contract layout, so every downstream metric means
    the same thing for every harness."""
    scored = [
        e
        for e in experiments
        if e["metric"] is not None
        and not e["flags"].get("smoke_test")
        and not e["flags"].get("smoke")
    ]
    minimize = direction == "minimize"
    order = sorted(
        scored, key=lambda e: (e["finished_at"] or e["created_at"] or 0)
    )
    best = None
    trajectory = []
    improvements = 0
    for i, e in enumerate(order, 1):
        is_better = best is None or (
            e["metric"] < best["metric"] if minimize else e["metric"] > best["metric"]
        )
        if is_better:
            best = e
            improvements += 1
        trajectory.append(
            {"n": i, "name": e["name"], "value": e["metric"],
             "best_so_far": best["metric"]}
        )
    first_created = min(
        (e["created_at"] for e in experiments if isinstance(e["created_at"], (int, float))),
        default=None,
    )
    time_to_best = None
    if best and isinstance(best.get("finished_at"), (int, float)) and first_created:
        time_to_best = round(best["finished_at"] - first_created, 1)

    validation_census: dict[str, dict] = {}
    for e in scored:
        vid = e["validation_id"] or "unverified"
        c = validation_census.setdefault(vid, {"count": 0, "best": None, "best_name": None})
        c["count"] += 1
        if c["best"] is None or (
            e["metric"] < c["best"] if minimize else e["metric"] > c["best"]
        ):
            c["best"] = e["metric"]
            c["best_name"] = e["name"]

    code_rows = [e for e in experiments if e.get("code")]

    def _code_arr(key: str, ndigits: int = 0) -> list:
        vals = [e["code"][key] for e in code_rows if e["code"].get(key) is not None]
        return _subsample(vals, ndigits=ndigits)

    code_metrics_rollup = None
    if code_rows:
        tot_code = sum(e["code"]["code_lines"] for e in code_rows)
        tot_comment = sum(e["code"]["comment_lines"] for e in code_rows)
        code_metrics_rollup = {
            "experiments_with_code": len(code_rows),
            "total_py_files": sum(e["code"]["py_files"] for e in code_rows),
            "total_lines": sum(e["code"]["total_lines"] for e in code_rows),
            "total_code_lines": tot_code,
            "total_comment_lines": tot_comment,
            "comment_share_overall": (
                round(tot_comment / (tot_code + tot_comment), 4)
                if (tot_code + tot_comment) else None
            ),
            "ast_parse_failures": sum(
                e["code"]["ast_parse_failures"] for e in code_rows
            ),
            # Raw per-experiment arrays (evenly subsampled at the shared
            # cap) so code quality is plottable as distributions, same as
            # durations and request sizes.
            "samples": {
                "code_lines": _code_arr("code_lines"),
                "comment_share": _code_arr("comment_share", ndigits=4),
                "functions": _code_arr("functions"),
                "mean_function_lines": _code_arr("mean_function_lines", ndigits=1),
                "max_control_nesting": _code_arr("max_control_nesting"),
                "branches_per_100_code_lines": _code_arr(
                    "branches_per_100_code_lines", ndigits=2
                ),
            },
        }

    status_counts = Counter(e["status"] for e in experiments)
    exec_failures = sum(
        1 for e in experiments if e["error"] and e["metric"] is None
    )
    negative_results = sum(
        1 for e in experiments if e["error"] and e["metric"] is not None
    )
    return {
        "sources": sources,
        "metric_key": metric_key,
        "direction": direction,
        "columns": columns,
        "total": len(experiments),
        "status_counts": dict(status_counts),
        "inflight_rows": sum(
            v for k, v in status_counts.items() if k in INFLIGHT_STATUSES
        ),
        "scored": len(scored),
        "best": {
            "name": best["name"], "value": best["metric"], "id": best["id"]
        }
        if best
        else None,
        "improvements": improvements,
        "experiments_to_best": next(
            (t["n"] for t in trajectory if best and t["value"] == best["metric"]),
            None,
        ),
        "time_to_best_seconds": time_to_best,
        "trajectory": trajectory,
        "validation_census": validation_census,
        "code_metrics": code_metrics_rollup,
        "execution_failures": exec_failures,
        "negative_results": negative_results,
        "fix_attempts_total": sum(int(e["fix_attempts"] or 0) for e in experiments),
        "missing_result_files": missing_result_files,
        "db_file_metric_mismatches": db_file_metric_mismatches,
        "results_replaced_after_finish": results_replaced_after_finish,
        "multi_realization_experiments": sum(
            1 for e in experiments
            if (e.get("realization") or {}).get("realizations", 0) > 1
        ),
        "untracked_multi_realizations": sum(
            1 for e in experiments
            if (e.get("realization") or {}).get("realizations", 0) > 1
            and not int(e.get("fix_attempts") or 0)
        ),
        "primary_mutations_after_finish": sum(
            1 for e in experiments
            if (e.get("realization") or {}).get("primary_results_mutated_after_finish")
        ),
        "launch_failure_signatures": dict(sum(
            (Counter((e.get("realization") or {})
                     .get("launch_failure_signatures") or {})
             for e in experiments), Counter()
        ).most_common(15)),
        "parked_rows": sum(1 for e in experiments if e.get("parked_at")),
        "experiments": experiments,
    }


def _norm_key(k: str) -> str:
    return "".join(ch for ch in k.lower() if ch.isalnum())


def _contract_metric_value(metrics: dict, metric_key: str):
    """The self-reported value for the domain metric, tolerating each
    harness's own spelling of the key (observed corpus: ``rmse`` vs
    ``overall_rmse`` vs ``holdout_log_loss`` for logloss). Normalized
    exact match first, then suffix, then prefix; ties resolve
    alphabetically so re-extraction is deterministic."""
    want = _norm_key(metric_key)
    if not want:
        return None, None
    cands = {k: v for k, v in metrics.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for test in (lambda k: _norm_key(k) == want,
                 lambda k: _norm_key(k).endswith(want),
                 lambda k: _norm_key(k).startswith(want)):
        hits = sorted(k for k in cands if test(k))
        if hits:
            return cands[hits[0]], hits[0]
    return None, None


def read_contract_experiments(workspace: Path, domain: str) -> dict | None:
    """Experiments from an evidence-contract run (external harness).

    A contract run keeps no experiments.db, adapter, or agent transcripts —
    just ``experiments/<name>/results/metrics.json`` plus optional code and
    referee artifacts. This builds the same experiment rows the db reader
    builds, then feeds the shared rollup, so every downstream chart and
    metric means the same thing for every harness. Metric name and
    direction come from the lineup (the benchmark-domain registry), never
    from the harness. Timing is honest-coarse: the only per-experiment
    timestamp a bare contract carries is the evidence file's own mtime.
    """
    exp_root = workspace / "experiments"
    if not exp_root.is_dir():
        return None
    from alpha_lab.benchmarks.runcmp.lineup import load_lineup

    metric_key, direction = "", "maximize"
    try:
        for entry in load_lineup():
            if entry.get("id") == domain:
                m = entry.get("metric") or {}
                metric_key = str(m.get("name") or "")
                direction = ("minimize" if m.get("lower_is_better")
                             else "maximize")
                break
    except (OSError, json.JSONDecodeError):
        pass

    rows = []
    for d in sorted((p for p in exp_root.iterdir() if p.is_dir()),
                    key=lambda p: p.name):
        mpath = d / "results" / "metrics.json"
        merged: dict = {}
        mtime = None
        if mpath.is_file():
            try:
                merged = json.loads(mpath.read_text())
                mtime = mpath.stat().st_mtime
            except (json.JSONDecodeError, OSError):
                merged = {}
        metric, matched_key = _contract_metric_value(merged, metric_key)
        validation_id = None
        for key in _VALIDATION_ID_KEYS:
            if merged.get(key):
                validation_id = f"{key}:{merged[key]}"
                break
        rows.append({
            "id": None,  # assigned after mtime sort
            "name": d.name,
            "status": "done" if merged else "no_evidence",
            "parent_id": None, "priority": None, "parked_at": None,
            "created_at": None, "started_at": None,
            "finished_at": mtime,
            "duration_seconds": None,
            "fix_attempts": None,
            "error": None if merged else "results/metrics.json missing or unparseable",
            "metric": metric,
            "metric_source_key": matched_key,
            "validation_id": validation_id,
            "flags": {k: merged.get(k) for k in (
                "valid_score", "ranking_protocol_valid", "valid", "smoke_test",
                "smoke", "partial", "param_count", "wall_clock_seconds",
                "device") if k in merged},
            "cut_off_by_limit": False,
            "metric_mismatch": False,
            "replaced_after_finish": False,
            "realization": None,
            "code": scan_code_metrics(d),
            # contract convention: an "approach" field is the harness's own
            # one-paragraph statement of what the experiment tried
            "hypothesis": str(merged.get("approach") or "")[:300],
        })
    if not rows:
        return None
    rows.sort(key=lambda r: (r["finished_at"] is None, r["finished_at"] or 0,
                             r["name"]))
    for i, r in enumerate(rows, 1):
        r["id"] = i
    return _aggregate_experiments(
        rows, metric_key=metric_key, direction=direction,
        sources=[str(exp_root)], columns=[],
    )


# ---------------------------------------------------------------------------
# Conductor meta/ (cond runs only)
# ---------------------------------------------------------------------------

def read_conductor_meta(meta_dir: Path) -> dict | None:
    if not meta_dir.is_dir():
        return None
    out: dict = {"sources": [str(meta_dir)]}
    log = meta_dir / "meta_log.jsonl"
    decisions: Counter = Counter()
    targets: Counter = Counter()
    evidence_nonempty = self_check_nonempty = total = 0
    first_ts = last_ts = None
    if log.is_file():
        with open(log, errors="ignore") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                decisions[str(rec.get("decision_type") or "?")] += 1
                if rec.get("target"):
                    targets[str(rec["target"])] += 1
                if str(rec.get("evidence") or "").strip():
                    evidence_nonempty += 1
                if str(rec.get("self_check") or "").strip():
                    self_check_nonempty += 1
                ts = rec.get("ts")
                if isinstance(ts, (int, float)):
                    first_ts = ts if first_ts is None else min(first_ts, ts)
                    last_ts = ts if last_ts is None else max(last_ts, ts)
    out["decisions_total"] = total
    out["decision_types"] = dict(decisions.most_common())
    out["targets"] = dict(targets.most_common())
    out["evidence_nonempty"] = evidence_nonempty
    out["self_check_nonempty"] = self_check_nonempty
    out["first_ts"] = first_ts
    out["last_ts"] = last_ts

    ann = meta_dir / "annotations.json"
    if ann.is_file():
        try:
            data = json.loads(ann.read_text())
            norm = {
                str(k): (v if isinstance(v, str) else (v or {}).get("label", "?"))
                for k, v in data.items()
            }
            labels = Counter(norm.values())
            out["annotations"] = dict(labels.most_common())
            out["annotations_map"] = dict(list(norm.items())[:60])
        except (json.JSONDecodeError, OSError):
            out["annotations"] = {"__error__": 1}
    for name, key in (
        ("directive_acks.jsonl", "directive_acks"),
        ("directive_retirements.jsonl", "directive_retirements"),
        ("token_usage.jsonl", "token_usage_lines"),
    ):
        p = meta_dir / name
        if not p.is_file():
            p = meta_dir / (name + ".gz")
        if p.is_file():
            with _open_log(p) as fh:
                out[key] = sum(1 for _ in fh)
    out["scratch_scripts"] = (
        len(list((meta_dir / "scratch").glob("*.py")))
        if (meta_dir / "scratch").is_dir()
        else 0
    )
    out["phase_rewinds"] = len(list(meta_dir.glob("phase_rewind*")))
    out["has_verify_request"] = (meta_dir / "verify_request.json").is_file()
    notes = meta_dir / "notes_to_user.md"
    out["notes_to_user_chars"] = notes.stat().st_size if notes.is_file() else 0
    directives = meta_dir / "directives.md"
    out["directives_chars"] = directives.stat().st_size if directives.is_file() else 0
    return out


def read_seat_ledger(meta_dir: Path) -> dict | None:
    """Who actually sat in each seat: provider+model per role, counted from
    the run's own token ledger (every returned call is one line carrying
    log_name, provider, model). This is the observed assignment — config
    declares intent, and framework defaults or fallbacks can silently
    substitute a different model (2026-08-06: one July run's conductor ran
    claude-opus-4-7 while every other seat ran gpt-5.6-sol, and no viewer
    surface showed it). Absent where the framework keeps no ledger (msml
    layout) — there the per-role ``models`` counts in agent_logs are the
    observed source instead.
    """
    path = meta_dir / "token_usage.jsonl"
    if not path.is_file():
        path = meta_dir / "token_usage.jsonl.gz"
    if not path.is_file():
        return None
    seats: dict[str, Counter] = defaultdict(Counter)
    with _open_log(path) as fh:
        for raw in fh:
            try:
                rec = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            role = role_for_log(str(rec.get("log_name") or "?"))
            pair = f"{rec.get('provider') or '?'}:{rec.get('model') or '?'}"
            seats[role][pair] += 1
    if not seats:
        return None
    return {
        "sources": [str(path)],
        "seats": {k: dict(v.most_common()) for k, v in sorted(seats.items())},
    }


# ---------------------------------------------------------------------------
# Durable memory (msml layout)
# ---------------------------------------------------------------------------

def read_memory(workspace: Path) -> dict | None:
    records_dir = workspace / ".alpha_lab" / "memory" / "records"
    if not records_dir.is_dir():
        return None
    hashes: Counter = Counter()
    count = 0
    for p in records_dir.glob("*.json"):
        count += 1
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        payload = {
            k: data.get(k)
            for k in ("kind", "summary", "content", "tags", "agent", "sources")
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        hashes[digest] += 1
    duplicates = sum(v - 1 for v in hashes.values() if v > 1)
    return {
        "sources": [str(records_dir)],
        "records": count,
        "duplicate_records": duplicates,
        "duplicate_fraction": round(duplicates / count, 3) if count else 0.0,
    }


# ---------------------------------------------------------------------------
# Workspace inventory (light)
# ---------------------------------------------------------------------------

_PRUNE = {
    "data", "datasets", "models", "checkpoints", "backups", "__pycache__",
    ".git", ".cache", "cache", "caches", "tokenizers", "runs",
    "lightning_logs", "catboost_info", "wandb", "scratch_data",
}
_VERIFY_HINTS = ("reality_check", "canonical_audit", "causal_evidence",
                 "frozen_equivalence")
_KEYWORDS = {
    "assertions": ("assert ", "raise assertionerror"),
    "validation": ("validation", "val_"),
    "verification": ("verify", "reality_check", "audit"),
    "dedup_leakage": ("dedup", "duplicate", "leakage"),
    "hashing": ("sha256", "fingerprint", "blake2"),
}


def inventory_workspace(workspace: Path) -> dict:
    classes: Counter = Counter()
    for root, dirs, files in os.walk(workspace):
        rel_root = os.path.relpath(root, workspace)
        dirs[:] = [d for d in dirs if d not in _PRUNE]
        parts = rel_root.split(os.sep)
        for fname in files:
            rel = os.path.join(rel_root, fname) if rel_root != "." else fname
            low = rel.lower()
            if fname == "metrics.json" and "results" in parts:
                classes["experiment_results"] += 1
            elif fname == "debrief.md":
                classes["debriefs"] += 1
            elif parts[0] in ("verify", "verification") or any(
                h in low for h in _VERIFY_HINTS
            ):
                classes["verification"] += 1
            elif parts[0] == "harness" and fname.endswith(".py"):
                classes["harness_source"] += 1
            elif parts[0] == "logs" and fname.endswith(".jsonl"):
                classes["agent_logs"] += 1
            elif parts[0] in ("output", "reports", "notes"):
                classes["reports_notes"] += 1
            elif fname.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
                classes["plots"] += 1

    keyword_profile: dict[str, dict[str, int]] = {}
    for area in ("adapter", "harness"):
        base = workspace / area
        if not base.is_dir():
            continue
        counts = dict.fromkeys(_KEYWORDS, 0)
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in (".py", ".md", ".json"):
                continue
            try:
                if p.stat().st_size > 2_000_000:
                    continue
                text = p.read_text(errors="ignore").lower()
            except OSError:
                continue
            for key, pats in _KEYWORDS.items():
                counts[key] += sum(text.count(pat) for pat in pats)
        keyword_profile[area] = counts
    # Early-phase products (user ask 2026-08-11): sizes and crude structure
    # of the durable phase-0/1 artifacts, layout-aware like the framework
    # detection — cond keeps learnings.md at the workspace root, msml keeps
    # agenda.md and a phase1/ directory; absent files stay absent (0 bytes
    # means an empty file, missing keys mean the layout has no such thing).
    def _fsize(path: Path) -> int:
        try:
            return path.stat().st_size if path.is_file() else 0
        except OSError:
            return 0

    def _bullets(path: Path) -> int:
        if not path.is_file():
            return 0
        n = 0
        try:
            for line in path.read_text(errors="replace").splitlines():
                s = line.lstrip()
                if s.startswith(("- ", "* ")) or re.match(r"\d+[.)] ", s):
                    n += 1
        except OSError:
            return 0
        return n

    # Report census (user ask 2026-08-11): the run's written products,
    # measured — bytes, markdown tables, numeric tokens, image references —
    # over the report/notes/output documents and the per-experiment
    # debriefs. "Best" stays a reviewer judgment; these are the volume and
    # structure facts that judgment can be checked against.
    def _doc_stats(paths: list[Path]) -> dict:
        bytes_ = tables = numbers = images = 0
        for f in paths:
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            bytes_ += len(text)
            tables += sum(1 for ln in text.splitlines()
                          if ln.lstrip().startswith("|") and "|" in ln[3:])
            numbers += len(re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text))
            images += len(re.findall(r"!\[|\.png\)|<img ", text))
        return {"files": len(paths), "bytes": bytes_, "table_rows": tables,
                "numbers": numbers, "images": images}

    report_docs: list[Path] = []
    debrief_docs: list[Path] = []
    for sub in ("reports", "notes", "output"):
        d = workspace / sub
        if d.is_dir():
            report_docs += [f for f in d.rglob("*")
                            if f.is_file() and f.suffix in (".md", ".txt")]
    for f in workspace.rglob("debrief.md"):
        debrief_docs.append(f)
        if len(debrief_docs) >= 400:
            break

    p1 = workspace / "phase1"
    p1_files = [f for f in p1.rglob("*") if f.is_file()] if p1.is_dir() else []
    plans = [f for f in workspace.glob("*.md")
             if re.search(r"plan|todo|agenda", f.name, re.IGNORECASE)]
    exploration = {
        "learnings_bytes": _fsize(workspace / "learnings.md"),
        "learnings_bullets": _bullets(workspace / "learnings.md"),
        "plan_files": {f.name: _fsize(f) for f in plans[:12]},
        "plan_bytes": sum(_fsize(f) for f in plans),
        "phase1_dir_files": len(p1_files),
        "phase1_dir_bytes": sum(_fsize(f) for f in p1_files),
    }
    reports = {"final": _doc_stats(report_docs),
               "debriefs": _doc_stats(debrief_docs)}
    return {
        "sources": [str(workspace)],
        "classes": dict(classes),
        "keyword_profile": keyword_profile,
        "exploration": exploration,
        "reports": reports,
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_FIELDS = (
    "provider", "model", "reasoning_effort", "domain", "target",
)


def read_config(record: RunRecord, run_start_config: str | None) -> dict:
    candidates = []
    if run_start_config:
        candidates.append(Path(run_start_config))
    ws = Path(record.workspace)
    candidates += [ws / ".alpha_lab" / "config.json", ws / "config.json"]
    for path in candidates:
        try:
            if path.is_file():
                data = json.loads(path.read_text())
                out = {k: data.get(k) for k in _CONFIG_FIELDS}
                p3 = ((data.get("pipeline") or {}).get("phase3") or {})
                out["phase3"] = {
                    k: p3.get(k)
                    for k in (
                        "max_experiments", "worker_count", "gpu_ids",
                        "time_limit_seconds", "max_fix_iterations",
                        "no_conductor", "cpu_enabled",
                    )
                }
                out["max_fix_iterations"] = data.get("max_fix_iterations") or p3.get(
                    "max_fix_iterations"
                )
                # declared per-seat pins (conductor_model, verifier_*_model,
                # web_search_model, ...) — intent, to hold against the
                # observed seat ledger
                out["seat_pins"] = {
                    k: data[k] for k in sorted(data)
                    if k not in ("provider", "model")
                    and isinstance(data.get(k), str)
                    and ("model" in k or "provider" in k)
                }
                out["source"] = str(path)
                return out
        except (json.JSONDecodeError, OSError):
            continue
    return {"source": None}


# ---------------------------------------------------------------------------
# Pack assembly
# ---------------------------------------------------------------------------

# Bumped whenever the extractor learns new fields, so cached packs from an
# older extractor re-extract instead of silently serving stale evidence
# (2026-08-06: pack-2 adds seat ledgers, gz transcripts, verifier seats).
PACK_SCHEMA = "runcmp-pack-7"


def build_pack(record: RunRecord) -> dict:
    ws = Path(record.workspace)
    t0 = time.time()
    run_tag = ws.name
    events = None
    if record.events_path:
        ev_path = _resolve_maybe_gz(Path(record.events_path))
        events = parse_events(ev_path, run_tag=run_tag,
                              since_iso=last_attempt_start(ev_path, run_tag))
    agent_logs = parse_agent_logs(ws / "logs")
    run_log = (parse_run_log(_resolve_maybe_gz(Path(record.run_log_path)))
               if record.run_log_path else None)
    exps = (
        read_experiments(Path(record.db_path), ws) if record.db_path
        else read_contract_experiments(ws, record.domain)
    )
    pack = {
        "schema": PACK_SCHEMA,
        "run": {
            **{
                k: getattr(record, k)
                for k in (
                    "label", "framework", "domain", "era", "run_dir",
                    "workspace", "completeness", "framework_evidence",
                    "pair_key", "model", "run_state",
                )
            },
        },
        "config": read_config(
            record, (events or {}).get("run_start", {}).get("config")
        ),
        "experiments": exps,
        "events": events,
        "agent_logs": agent_logs,
        "run_log": run_log,
        "conductor": read_conductor_meta(ws / "meta"),
        "seats": read_seat_ledger(ws / "meta"),
        "memory": read_memory(ws),
        "inventory": inventory_workspace(ws),
        "adapter_drift": read_adapter_drift(ws, (events or {}).get("ts_min")),
        "extract_seconds": None,
    }
    pack["extract_seconds"] = round(time.time() - t0, 1)
    return pack


def read_adapter_drift(ws: Path, run_start_ts: float | None) -> dict:
    """Which of the run's own instruction files changed, and when.

    A run rewrites its own adapter prompts at phase 0 and, through its
    supervisor, sometimes mid-run. A mid-run rewrite permanently changes the
    per-task behavior of every later agent (2026-07-29 d2_o5_cond: a 5.4KB
    implement-prompt patch at hour 5.7 preceded 4.4x more verification
    commands per task), so it must be visible as data, not archaeology.
    """
    adapter = ws / "adapter"
    if not adapter.is_dir():
        return {}
    files = {}
    midrun = 0
    for p in sorted(adapter.glob("*.md")):
        try:
            st = p.stat()
        except OSError:
            continue
        hours = (round((st.st_mtime - run_start_ts) / 3600, 2)
                 if run_start_ts else None)
        patched_midrun = bool(hours is not None and hours > 1.0)
        midrun += patched_midrun
        files[p.name] = {"bytes": st.st_size,
                         "hours_after_run_start": hours,
                         "patched_midrun": patched_midrun}
    return {"files": files, "files_patched_midrun": midrun}


def _extract_one(args: tuple[dict, str]) -> str:
    row, out_dir = args
    record = RunRecord(**row)
    pack = build_pack(record)
    safe = record.label.replace("/", "__").replace("#", "_")
    out_path = Path(out_dir) / f"{safe}.json"
    out_path.write_text(json.dumps(pack, indent=1, default=str) + "\n")
    return (
        f"{record.label}: {pack['extract_seconds']}s"
        f" tokens={((pack.get('events') or {}).get('tokens') or {}).get('input', 0):,}"
    )


def _pack_current(path: Path) -> bool:
    """A cached pack counts only if THIS extractor version wrote it, whole.

    An existence-only check silently serves stale evidence after the
    extractor learns new fields (2026-08-06: a whole republish ran with
    packs missing the seat ledgers). The schema tag is the first key
    build_pack writes, so the head settles the version; a worker killed
    mid-write leaves a truncated file, so the tail must close the object.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
            if f'"schema": "{PACK_SCHEMA}"'.encode() not in head:
                return False
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 8))
            return b"}" in fh.read()
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract per-run evidence packs")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--labels", nargs="*", help="restrict to these labels")
    ap.add_argument("--all", action="store_true",
                    help="include incomplete runs (default: complete only)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    records = load_registry(args.corpus)
    if args.labels:
        records = [r for r in records if r.label in set(args.labels)]
    elif not args.all:
        records = [r for r in records if r.completeness == "complete"]
    args.out.mkdir(parents=True, exist_ok=True)
    # The packs directory belongs to this corpus: a pack whose run is no
    # longer registered is a stale derivative (a narrowed re-index would
    # otherwise trip every downstream packs==corpus assertion). Prune
    # loudly — packs regenerate deterministically from the runs.
    expected = {r.label.replace("/", "__").replace("#", "_") + ".json"
                for r in load_registry(args.corpus)}
    if not args.labels:
        for stray in sorted(p for p in args.out.glob("*.json")
                            if p.name not in expected):
            stray.unlink()
            print(f"pruned stale pack (run not in corpus): {stray.name}")
    todo = []
    for r in records:
        safe = r.label.replace("/", "__").replace("#", "_")
        if not args.force and _pack_current(args.out / f"{safe}.json"):
            print(f"skip (cached): {r.label}")
            continue
        todo.append((r.__dict__, str(args.out)))
    if not todo:
        print("nothing to do")
        return 0
    print(f"extracting {len(todo)} runs with {args.workers} workers...")
    if args.workers <= 1:
        for item in todo:
            print(" ", _extract_one(item))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for msg in pool.map(_extract_one, todo):
                print(" ", msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
