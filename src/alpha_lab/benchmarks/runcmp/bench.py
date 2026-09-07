"""Deterministic benchmark layer: a versioned metric registry over evidence packs.

Everything here is a pure function of the extracted packs (plus optional
referee.json and small bounded workspace scans), so a re-run over the same
corpus reproduces the same numbers. Output: ``bench.json`` (machine) and
``bench.md`` (numbers-first scorecards + classified diagnostics). No winners
are opined; directions are declared per metric and diagnosis classes come from
the versioned ``rules.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from alpha_lab.benchmarks.runcmp.corpus import load_registry
from alpha_lab.benchmarks.runcmp.tabulate import cost_usd, load_packs

BENCH_VERSION = "5"


def _exploration(p: dict) -> dict:
    return (p.get("inventory") or {}).get("exploration") or {}


def _websearches(p: dict, early_only: bool) -> int | None:
    tb = (p.get("agent_logs") or {}).get("tools_by_role")
    if not isinstance(tb, dict):
        return None
    early = ("phase0", "phase1_explorer", "phase1")
    total = 0
    for key, n in tb.items():
        role, _, tool = key.partition(":")
        if "web_search" not in tool:
            continue
        if early_only and role not in early:
            continue
        total += int(n or 0)
    return total


def _reports(p: dict, kind: str) -> dict:
    return ((p.get("inventory") or {}).get("reports") or {}).get(kind) or {}


def _artifact_reads(p: dict, cls: str) -> int | None:
    ar = (p.get("agent_logs") or {}).get("artifact_reads")
    if not isinstance(ar, dict):
        return None
    return sum(int(n or 0) for key, n in ar.items()
               if key.endswith(":" + cls))


def _supervisor_sessions(p: dict) -> int | None:
    so = (p.get("agent_logs") or {}).get("seat_outcomes") or {}
    rec = (so.get("by_role") or {}).get("supervisor")
    return rec.get("seats_started") if isinstance(rec, dict) else None


# Append-only union of every metric id ever registered. enforce_manifest()
# hard-fails if the live registry is missing anything from this union —
# dropping a metric is a crash, never a silent drift (user ruling
# 2026-08-02: no metric is ever removed from the hard data).
_MANIFEST_PATH = Path(__file__).parent / "metrics_manifest.json"


def enforce_manifest(metrics: list[dict],
                     manifest_path: Path = _MANIFEST_PATH) -> None:
    ids = [m["id"] for m in metrics]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise RuntimeError(
            f"metric registry has duplicate ids (later entries silently "
            f"shadow earlier ones): {dupes}"
        )
    current = {m["id"] for m in metrics}
    recorded: set[str] = set()
    if manifest_path.exists():
        recorded = set(json.loads(manifest_path.read_text())["metric_ids"])
    missing = sorted(recorded - current)
    if missing:
        raise RuntimeError(
            "metric registry lost previously-registered metrics "
            f"(append-only manifest {manifest_path.name}): {missing}"
        )
    union = sorted(recorded | current)
    if set(union) != recorded:
        try:
            manifest_path.write_text(json.dumps(
                {"comment": "Append-only union of every metric id ever "
                            "registered. bench refuses to run if any entry "
                            "is missing from the live registry.",
                 "metric_ids": union}, indent=1) + "\n")
        except OSError as exc:  # the check must kill; the append is best-effort
            print(f"[bench] WARNING: could not update {manifest_path}: {exc}",
                  file=sys.stderr)

# ---------------------------------------------------------------------------
# Metric registry — id, title, unit, direction, definition, compute(pack, ctx)
# direction: "lower" | "higher" | "info"
# ---------------------------------------------------------------------------


def _tokens(pack):
    return ((pack.get("events") or {}).get("tokens")) or {}


def _exps(pack):
    return pack.get("experiments") or {}


def _total_tokens(pack):
    t = _tokens(pack)
    return int(t.get("input", 0)) + int(t.get("output", 0))


def _per_scored(pack, value):
    s = _exps(pack).get("scored") or 0
    return round(value / s, 0) if s and value is not None else None


def _comp_share(pack, key):
    comp = (pack.get("agent_logs") or {}).get("request_composition_chars") or {}
    total = sum(v for v in comp.values() if isinstance(v, (int, float)))
    v = comp.get(key)
    return round(v / total, 4) if total and v is not None else None


def _roundtrip_share(pack, side):
    rt = (pack.get("agent_logs") or {}).get("reasoning_roundtrip") or {}
    total = rt.get(f"{side}_total")
    with_r = rt.get(f"{side}_with_reasoning")
    return round(with_r / total, 4) if total and with_r is not None else None


def _memory(pack):
    ev = pack.get("events") or {}
    calls = sum(v for k, v in (ev.get("tools") or {}).items()
                if k.startswith("memory_"))
    fails = sum(v for k, v in (ev.get("tool_failures") or {}).items()
                if k.startswith("memory_"))
    return calls, fails


def _role_tokens(pack, roles):
    total = 0
    for role, r in ((pack.get("agent_logs") or {}).get("roles") or {}).items():
        if role in roles:
            t = r.get("tokens") or {}
            total += int(t.get("input", 0)) + int(t.get("output", 0))
    return total


def _phase_hours(pack, phase):
    w = ((pack.get("events") or {}).get("phase_windows")) or {}
    if phase not in w:
        return None
    return round(w[phase]["seconds"] / 3600, 2)


def _tail_after_last_improvement(pack):
    traj = _exps(pack).get("trajectory") or []
    if not traj:
        return None
    last_improve = 0
    best = None
    for t in traj:
        if best is None or t["best_so_far"] != best:
            best = t["best_so_far"]
            last_improve = t["n"]
    return len(traj) - last_improve


def _late_gain_fraction(pack):
    """Fraction of total best-so-far improvement earned in the last 25% of
    scored experiments."""
    traj = _exps(pack).get("trajectory") or []
    if len(traj) < 4:
        return None
    first, final = traj[0]["best_so_far"], traj[-1]["best_so_far"]
    total_gain = abs(final - first)
    if total_gain == 0:
        return 0.0
    cut = traj[max(0, int(len(traj) * 0.75) - 1)]["best_so_far"]
    return round(abs(final - cut) / total_gain, 3)


def _validation_coverage(pack):
    census = _exps(pack).get("validation_census") or {}
    scored = _exps(pack).get("scored") or 0
    if not scored:
        return None
    with_id = sum(v["count"] for k, v in census.items() if k != "unverified")
    return round(with_id / scored, 3)


def _champion_annotation_correct(pack):
    """Does the Conductor's 'champion' annotation point at the true best?"""
    cond = pack.get("conductor") or {}
    amap = cond.get("annotations_map") or {}
    champs = [k for k, v in amap.items() if v == "champion"]
    best = _exps(pack).get("best") or {}
    if not champs or best.get("id") is None:
        return None
    return 1.0 if str(best["id"]) in champs else 0.0


def _report_covers_best(pack):
    """Does any report/output markdown mention the final best experiment?"""
    best = (_exps(pack).get("best") or {}).get("name")
    if not best:
        return None
    ws = Path(pack["run"]["workspace"])
    found_any = False
    for sub in ("reports", "output", "notes"):
        base = ws / sub
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.md"))[:200]:
            try:
                if p.stat().st_size > 4_000_000:
                    continue
                found_any = True
                if best in p.read_text(errors="ignore"):
                    return 1.0
            except OSError:
                continue
    return 0.0 if found_any else None


def _read_failures(pack):
    sigs = ((pack.get("events") or {}).get("tool_failure_signatures")) or {}
    total = 0
    for tool, hist in sigs.items():
        for sig, count in hist.items():
            if "file not found" in sig.lower() or "image not found" in sig.lower():
                total += count
    return total


def _http_429(pack):
    rl = pack.get("run_log") or {}
    return sum(v.get("rate_limited", 0)
               for v in (rl.get("http_endpoints") or {}).values())


def _samples(pack, key):
    vals = ((pack.get("agent_logs") or {}).get("samples") or {}).get(key)
    return sorted(v for v in (vals or []) if isinstance(v, (int, float)))


def _pct(vals, q):
    if not vals:
        return None
    idx = min(int(q * (len(vals) - 1) + 0.5), len(vals) - 1)
    return round(vals[idx], 4)


def _sample_median(pack, key):
    return _pct(_samples(pack, key), 0.5)


def _sample_p90(pack, key):
    return _pct(_samples(pack, key), 0.9)


def _durations_minutes(pack):
    return sorted(
        e["duration_seconds"] / 60.0
        for e in (_exps(pack).get("experiments") or [])
        if isinstance(e.get("duration_seconds"), (int, float))
    )


def _refinement_stats(pack):
    """(scored parent-child pairs, child wins) over explicit parent links."""
    rows = _exps(pack).get("experiments") or []
    by_id = {e["id"]: e for e in rows if e.get("id") is not None}
    minimize = _exps(pack).get("direction") == "minimize"
    pairs = wins = 0
    for e in rows:
        parent = by_id.get(e.get("parent_id"))
        if not parent or e.get("metric") is None or parent.get("metric") is None:
            continue
        pairs += 1
        better = (e["metric"] < parent["metric"] if minimize
                  else e["metric"] > parent["metric"])
        wins += 1 if better else 0
    return pairs, wins


def _decision_types(pack):
    con = pack.get("conductor")
    return (con or {}).get("decision_types") if con else None


def _inventory_classes(pack):
    return ((pack.get("inventory") or {}).get("classes")) or None


def _code_rollup(pack):
    return _exps(pack).get("code_metrics")


def _code_median(pack, key):
    vals = sorted(
        e["code"][key]
        for e in (_exps(pack).get("experiments") or [])
        if e.get("code") and e["code"].get(key) is not None
    )
    if not vals:
        return None
    mid = len(vals) // 2
    med = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
    return round(med, 4)


METRICS: list[dict] = [
    # --- lifecycle ---
    dict(id="lifecycle.scored_fraction", unit="ratio", direction="higher",
         definition="scored experiments / total DB rows",
         compute=lambda p, c: round((_exps(p).get("scored") or 0) /
                                    max(_exps(p).get("total") or 0, 1), 3)
         if _exps(p).get("total") else None),
    dict(id="lifecycle.inflight_rows_at_end", unit="rows", direction="lower",
         definition="DB rows left in non-terminal statuses",
         compute=lambda p, c: _exps(p).get("inflight_rows")),
    # ---- exploration: the phase-0/1 products and moves (2026-08-11: the
    # early phases were invisible beyond hours and tool clicks; these read
    # the pack fields extract now carries — inventory.exploration and the
    # per-seat web-search ledger). direction="info": more exploration is
    # not better by fiat; the cross-run boards let depth be charted against
    # final quality, which is the actual question.
    dict(id="exploration.web_searches", unit="count", direction="info",
         definition="web_search tool calls, all seats (transcripts)",
         compute=lambda p, c: _websearches(p, early_only=False)),
    dict(id="exploration.web_searches_early", unit="count", direction="info",
         definition="web_search calls by the phase-0/1 seats",
         compute=lambda p, c: _websearches(p, early_only=True)),
    dict(id="exploration.learnings_bytes", unit="bytes", direction="info",
         definition="size of the run's accumulated learnings file",
         compute=lambda p, c: _exploration(p).get("learnings_bytes")),
    dict(id="exploration.learnings_bullets", unit="count", direction="info",
         definition="bulleted/numbered items in the learnings file",
         compute=lambda p, c: _exploration(p).get("learnings_bullets")),
    dict(id="exploration.plan_bytes", unit="bytes", direction="info",
         definition="workspace-root plan/todo/agenda file bytes",
         compute=lambda p, c: _exploration(p).get("plan_bytes")),
    dict(id="exploration.phase1_product_bytes", unit="bytes", direction="info",
         definition="bytes under the phase1/ products directory",
         compute=lambda p, c: _exploration(p).get("phase1_dir_bytes")),
    dict(id="exploration.plots", unit="count", direction="info",
         definition="image files anywhere in the workspace",
         compute=lambda p, c: ((p.get("inventory") or {}).get("classes")
                               or {}).get("plots")),
    # ---- reporting: the run's written products and their readership
    # (2026-08-11). Volume/structure are facts; "best report" stays the
    # reviewers' judgment, checkable against these. Readership counts come
    # from the transcripts' read/grep calls — the causal half: debriefs and
    # learnings are read mid-run by the seats still making decisions.
    dict(id="reporting.report_bytes", unit="bytes", direction="info",
         definition="bytes across reports/notes/output documents",
         compute=lambda p, c: _reports(p, "final").get("bytes")),
    dict(id="reporting.report_table_rows", unit="count", direction="info",
         definition="markdown table rows in the report documents",
         compute=lambda p, c: _reports(p, "final").get("table_rows")),
    dict(id="reporting.report_numbers", unit="count", direction="info",
         definition="numeric tokens in the report documents",
         compute=lambda p, c: _reports(p, "final").get("numbers")),
    dict(id="reporting.report_images", unit="count", direction="info",
         definition="image references in the report documents",
         compute=lambda p, c: _reports(p, "final").get("images")),
    dict(id="reporting.debrief_bytes", unit="bytes", direction="info",
         definition="bytes across per-experiment debriefs",
         compute=lambda p, c: _reports(p, "debriefs").get("bytes")),
    dict(id="reporting.debrief_reads", unit="count", direction="info",
         definition="read/grep calls that opened a debrief (any seat)",
         compute=lambda p, c: _artifact_reads(p, "debrief")),
    dict(id="reporting.learnings_reads", unit="count", direction="info",
         definition="read/grep calls that opened the learnings file",
         compute=lambda p, c: _artifact_reads(p, "learnings")),
    dict(id="reporting.report_reads", unit="count", direction="info",
         definition="read/grep calls that opened a report/notes document",
         compute=lambda p, c: _artifact_reads(p, "report")),
    dict(id="governance.supervisor_sessions", unit="count", direction="info",
         definition="supervisor seat sessions started (its transcripts)",
         compute=lambda p, c: _supervisor_sessions(p)),
    dict(id="lifecycle.wall_hours", unit="h", direction="info",
         definition="event-stream first to last timestamp",
         compute=lambda p, c: round(((p.get("events") or {}).get("wall_seconds")
                                     or 0) / 3600, 2) or None),
    dict(id="lifecycle.phase3_hours", unit="h", direction="info",
         definition="phase3 event window",
         compute=lambda p, c: _phase_hours(p, "phase3")),
    dict(id="lifecycle.dispatcher_crashes", unit="count", direction="lower",
         definition="run.log 'Phase 3 dispatcher crashed unexpectedly' lines",
         compute=lambda p, c: (p.get("run_log") or {}).get("dispatcher_crashes")),
    dict(id="lifecycle.tracebacks", unit="count", direction="lower",
         definition="run.log Python traceback headers",
         compute=lambda p, c: (p.get("run_log") or {}).get("tracebacks")),
    dict(id="lifecycle.strategist_calls_per_scored", unit="calls", direction="lower",
         definition="strategist-role API calls / scored experiments",
         compute=lambda p, c: _per_scored(
             p, ((p.get("agent_logs") or {}).get("roles") or {})
             .get("strategist", {}).get("api_calls"))),
    # --- context replay (the pipeline's ONE replay definition; packs
    #     agent_logs.replay — shared leading bytes of consecutive requests
    #     within one agent transcript / current request bytes) ---
    dict(id="context.replay_median_prefix_share", unit="ratio", direction="lower",
         definition="median shared-leading-bytes of consecutive api_request "
                     "lines / current line bytes (1.0 = full verbatim replay)",
         compute=lambda p, c: ((p.get("agent_logs") or {}).get("replay") or {})
         .get("median_prefix_share")),
    dict(id="context.replay_p90_prefix_share", unit="ratio", direction="lower",
         definition="p90 of the same per-pair prefix share",
         compute=lambda p, c: ((p.get("agent_logs") or {}).get("replay") or {})
         .get("p90_prefix_share")),
    dict(id="lifecycle.completion_attempt_failures", unit="count", direction="lower",
         definition="failed complete_research tool calls (poll churn)",
         compute=lambda p, c: ((p.get("events") or {}).get("tool_failures")
                               or {}).get("complete_research", 0)),
    dict(id="lifecycle.final_exit_code", unit="code", direction="info",
         definition="last outer EXIT marker in run.log",
         compute=lambda p, c: (p.get("run_log") or {}).get("final_exit_code")),
    dict(id="lifecycle.runlog_error_lines", unit="count", direction="lower",
         definition="ERROR/exception lines in run.log (normalized signatures)",
         compute=lambda p, c: sum(((p.get("run_log") or {})
                                   .get("error_signatures") or {}).values()) or None),
    dict(id="lifecycle.infrastructure_error_lines", unit="count", direction="info",
         definition="run.log error lines naming credentials/TLS/certificates or "
                     "logging-backend failures — environment interference, not "
                     "framework or model faults",
         compute=lambda p, c: sum(v for k, v in ((p.get("run_log") or {})
                                  .get("error_signatures") or {}).items()
                                  if any(w in k.lower() for w in
                                         ("tls", "certificat", "mlflow",
                                          "credential", "kerberos"))) or None),
    # --- integrity: does the reported number match the preserved file, and
    # was anything rewritten after the experiment finished? These are the
    # deterministic detectors for a self-reported score that its own
    # artifacts do not reproduce (observed once in this corpus).
    dict(id="integrity.metric_mismatches", unit="count", direction="lower",
         definition="experiments whose DB metric disagrees with the metric "
                     "in their preserved results file",
         compute=lambda p, c: sum(1 for e in (_exps(p).get("experiments") or [])
                                  if e.get("metric_mismatch")) or None),
    dict(id="integrity.results_replaced_after_finish", unit="count", direction="lower",
         definition="experiments whose results file was rewritten after the "
                     "experiment reached a terminal state",
         compute=lambda p, c: _exps(p).get("results_replaced_after_finish") or None),
    dict(id="integrity.db_file_metric_mismatches", unit="count", direction="lower",
         definition="rows where DB metric != file metric (rel_tol 1e-9); "
                    "0 means checked-and-clean, not unknown",
         compute=lambda p, c: _exps(p).get("db_file_metric_mismatches")),
    # --- lineage and queue dynamics ---
    dict(id="search.variant_rows", unit="count", direction="info",
         definition="experiments carrying a parent link (refinements of an "
                     "earlier experiment rather than fresh starts)",
         compute=lambda p, c: sum(1 for e in (_exps(p).get("experiments") or [])
                                  if e.get("parent_id") is not None) or None),
    dict(id="search.parked_rows", unit="count", direction="info",
         definition="experiments soft-cancelled by parking",
         compute=lambda p, c: _exps(p).get("parked_rows") or None),
    dict(id="search.cpu_flagged_experiments", unit="count", direction="info",
         definition="experiments whose preserved flags name a cpu device",
         compute=lambda p, c: sum(1 for e in (_exps(p).get("experiments") or [])
                                  if "cpu" in str((e.get("flags") or {})
                                                  .get("device", "")).lower()) or None),
    dict(id="search.partial_result_experiments", unit="count", direction="lower",
         definition="experiments whose preserved flags mark the result partial",
         compute=lambda p, c: sum(1 for e in (_exps(p).get("experiments") or [])
                                  if (e.get("flags") or {}).get("partial")) or None),
    dict(id="search.median_queue_wait_minutes", unit="min", direction="info",
         definition="median minutes between an experiment's creation and its "
                     "start (scheduler latency)",
         compute=lambda p, c: (lambda ws: round(sorted(ws)[len(ws)//2]/60, 1)
                               if ws else None)(
             [float(e["started_at"]) - float(e["created_at"])
              for e in (_exps(p).get("experiments") or [])
              if e.get("started_at") and e.get("created_at")
              and float(e["started_at"]) >= float(e["created_at"])])),
    dict(id="search.prioritized_rows", unit="count", direction="info",
         definition="experiments carrying a non-default queue priority "
                     "(conductor prioritization in use)",
         compute=lambda p, c: sum(1 for e in (_exps(p).get("experiments") or [])
                                  if e.get("priority") not in (None, 0)) or None),
    # --- reliability: API-level and event-level disruption ---
    dict(id="reliability.api_error_events", unit="count", direction="lower",
         definition="agent-visible API errors in the event stream (retried "
                     "provider failures: overloads, dropped connections)",
         compute=lambda p, c: sum(((p.get("events") or {})
                                   .get("status_error_signatures") or {}).values()) or None),
    dict(id="reliability.failure_signature_events", unit="count", direction="lower",
         definition="total failure-signature occurrences in the event stream",
         compute=lambda p, c: sum(((p.get("events") or {})
                                   .get("failure_signatures") or {}).values()) or None),
    dict(id="reliability.error_events", unit="count", direction="lower",
         definition="error-type events in the event stream",
         compute=lambda p, c: (p.get("events") or {}).get("error_events") or None),
    dict(id="lifecycle.phase2_abortions", unit="count", direction="lower",
         definition="run.log 'Max fix iterations reached' phase-2 abort lines",
         compute=lambda p, c: (p.get("run_log") or {}).get("phase2_abortions") or None),
    # Self-hardening visibility: a run whose supervisor rewrites its own
    # worker instructions mid-run permanently changes its per-task work
    # (2026-07-29 d2_o5_cond: a 5.4KB implement-prompt patch at hour 5.7
    # preceded 4.4x more verification commands per task; no metric surfaced
    # it, so three reviewer generations attributed the cost to the model).
    dict(id="lifecycle.adapter_patch_calls", unit="count", direction="info",
         definition="patch_adapter_file tool calls across the whole run "
                    "(phase-0 customization + any mid-run supervisor patches)",
         compute=lambda p, c: ((p.get("events") or {}).get("tools") or {})
         .get("patch_adapter_file") or None),
    # Service-side capacity refusals and the seats they killed. Without these
    # rows an arm whose verification stage was destroyed by 529s reads as an
    # arm that chose to verify less (2026-07-30 incident). Broken out per run;
    # the model is a column in this table, so per-model reading is direct.
    dict(id="reliability.capacity_refusals", unit="count", direction="lower",
         definition="529 overloaded_error responses from the serving side, "
                    "counted across every agent transcript in the run",
         compute=lambda p, c: (((p.get("agent_logs") or {})
                                .get("seat_outcomes") or {})
                               .get("all_seats") or {}).get("capacity_refusals")),
    dict(id="reliability.seats_died_before_first_response", unit="count",
         direction="lower",
         definition="agent sessions that issued a request and never received "
                    "a response (killed by refusals/connection failures); "
                    "invisible in the token ledger",
         compute=lambda p, c: (((p.get("agent_logs") or {})
                                .get("seat_outcomes") or {})
                               .get("all_seats") or {})
         .get("seats_died_before_first_response")),
    dict(id="verifier.seats_started", unit="count", direction="info",
         definition="verifier agent sessions created in this run",
         compute=lambda p, c: (((p.get("agent_logs") or {})
                                .get("seat_outcomes") or {})
                               .get("verifier") or {}).get("seats_started")),
    dict(id="verifier.completion_fraction", unit="ratio", direction="higher",
         definition="verifier seats that received at least one response / "
                    "verifier seats started; <1 means part of the "
                    "verification stage was killed, not skipped by choice",
         compute=lambda p, c: (((p.get("agent_logs") or {})
                                .get("seat_outcomes") or {})
                               .get("verifier") or {}).get("completion_fraction")),
    dict(id="verifier.seats_died_before_first_response", unit="count",
         direction="lower",
         definition="verifier seats killed before their first response",
         compute=lambda p, c: (((p.get("agent_logs") or {})
                                .get("seat_outcomes") or {})
                               .get("verifier") or {})
         .get("seats_died_before_first_response")),
    dict(id="reliability.roles_with_killed_seats", unit="count",
         direction="lower",
         definition="how many distinct agent roles (conductor, strategist, "
                    "worker, phase agents, supervisor, reporter, verifier) "
                    "lost at least one session before its first response",
         compute=lambda p, c: sum(
             1 for r in ((((p.get("agent_logs") or {})
                           .get("seat_outcomes") or {}).get("by_role")) or {}).values()
             if (r or {}).get("seats_died_before_first_response"))),
    # Numeric only — bench rollups sum/compare across runs, so a text-valued
    # metric raises TypeError (fixed once already today). The role NAMES live
    # in the per-seat survival table in tables.md.
    dict(id="reliability.max_killed_seats_in_one_role", unit="count",
         direction="lower",
         definition="largest number of sessions any single agent role lost "
                    "before its first response",
         compute=lambda p, c: max(
             [(r or {}).get("seats_died_before_first_response") or 0
              for r in ((((p.get("agent_logs") or {})
                          .get("seat_outcomes") or {}).get("by_role")) or {}).values()]
             or [0]) or None),
    dict(id="lifecycle.adapter_files_patched_midrun", unit="count",
         direction="info",
         definition="adapter prompt files modified more than 1h after run "
                    "start — the run's supervisor rewrote its own working "
                    "rules; every later agent behaves differently",
         compute=lambda p, c: (p.get("adapter_drift") or {})
         .get("files_patched_midrun")),
    dict(id="context.session_growth_ratio", unit="ratio", direction="info",
         definition="median last-request chars / median first-request chars "
                    "per agent session (one jsonl = one session)",
         compute=lambda p, c: ((p.get("agent_logs") or {})
                               .get("request_stats") or {})
         .get("session_growth_ratio")),
    dict(id="context.images_share", unit="ratio", direction="info",
         definition="image chars / total request chars, summed over every "
                    "request sent (plots ride in history for the session)",
         compute=lambda p, c: _comp_share(p, "images")),
    dict(id="context.tool_results_share", unit="ratio", direction="info",
         definition="tool-output chars / total request chars over every "
                    "request sent",
         compute=lambda p, c: _comp_share(p, "tool_results")),
    dict(id="context.thinking_share", unit="ratio", direction="info",
         definition="retained reasoning chars / total request chars over "
                    "every request sent",
         compute=lambda p, c: _comp_share(p, "thinking")),
    dict(id="context.reasoning_produced_share", unit="ratio", direction="info",
         definition="fraction of API responses carrying a reasoning trace "
                    "(reasoning_content / thinking block / encrypted item)",
         compute=lambda p, c: _roundtrip_share(p, "responses")),
    dict(id="context.reasoning_returned_share", unit="ratio", direction="info",
         definition="fraction of API requests carrying reasoning back to the "
                    "model; ~0 while produced_share is high = the model runs "
                    "blind to its own past reasoning (the GLM/deepseek defect "
                    "class found 2026-08-07)",
         compute=lambda p, c: _roundtrip_share(p, "requests")),
    dict(id="context.fresh_growth_per_session", unit="tokens",
         direction="info",
         definition="median (mean of last 3 − mean of first 3) fresh input "
                    "tokens per session; large positive with flat cache-read "
                    "growth = history is NOT cached and cost grows with "
                    "session depth squared",
         compute=lambda p, c: ((p.get("agent_logs") or {})
                               .get("cache_pattern") or {})
         .get("fresh_input_growth_median")),
    dict(id="context.cache_read_growth_per_session", unit="tokens",
         direction="info",
         definition="median cache-read token growth per session (see "
                    "context.fresh_growth_per_session)",
         compute=lambda p, c: ((p.get("agent_logs") or {})
                               .get("cache_pattern") or {})
         .get("cache_read_growth_median")),
    dict(id="lifecycle.strategist_stalls", unit="count", direction="lower",
         definition="run.log strategist-stall detections",
         compute=lambda p, c: (p.get("run_log") or {}).get("strategist_stalls") or None),
    # --- gateway load and server-side failures ---
    dict(id="http.requests_total", unit="count", direction="info",
         definition="HTTP requests in run.log across all endpoints",
         compute=lambda p, c: sum(v.get("requests", 0) for v in
                                  ((p.get("run_log") or {})
                                   .get("http_endpoints") or {}).values()) or None),
    dict(id="http.5xx_responses", unit="count", direction="lower",
         definition="HTTP 5xx responses across all endpoints",
         compute=lambda p, c: sum(n for v in ((p.get("run_log") or {})
                                  .get("http_endpoints") or {}).values()
                                  for code, n in (v.get("codes") or {}).items()
                                  if str(code).startswith("5")) or None),
    dict(id="http.peak_requests_per_minute", unit="req/min", direction="info",
         definition="highest per-minute request rate on any endpoint",
         compute=lambda p, c: max((v.get("peak_per_minute", 0) for v in
                                   ((p.get("run_log") or {})
                                    .get("http_endpoints") or {}).values()),
                                  default=0) or None),
    # --- agent-session hygiene ---
    dict(id="agents.sessions_ended_clean_total", unit="count", direction="info",
         definition="agent sessions that ended cleanly, summed over roles",
         compute=lambda p, c: sum((r.get("sessions_ended_clean") or 0) for r in
                                  ((p.get("agent_logs") or {})
                                   .get("roles") or {}).values()) or None),
    dict(id="agents.turns_total", unit="count", direction="info",
         definition="agent turns, summed over roles",
         compute=lambda p, c: sum(
             (len(t) if isinstance(t := (r.get("turns") or 0), list) else t)
             for r in ((p.get("agent_logs") or {})
                       .get("roles") or {}).values()) or None),
    dict(id="lifecycle.agents_stopped_unexpectedly", unit="count", direction="lower",
         definition="run.log 'Agent stopped unexpectedly' lines — an agent "
                     "turn that died without finishing its work",
         compute=lambda p, c: (p.get("run_log") or {}).get("agent_stops") or None),
    dict(id="lifecycle.experiments_cut_off_by_limit", unit="count", direction="info",
         definition="experiment rows whose recorded error names a harness stop "
                     "(timeout / time limit / OOM) — distinct from failures",
         compute=lambda p, c: sum(1 for e in (_exps(p).get("experiments") or [])
                                  if e.get("cut_off_by_limit")) or None),
    # --- search ---
    dict(id="search.scored", unit="count", direction="info",
         definition="experiments with a finite primary metric (non-smoke)",
         compute=lambda p, c: _exps(p).get("scored")),
    dict(id="search.improvements", unit="count", direction="info",
         definition="best-so-far improvements over the run",
         compute=lambda p, c: _exps(p).get("improvements")),
    dict(id="search.experiments_to_best", unit="count", direction="info",
         definition="scored-experiment index of the final best",
         compute=lambda p, c: _exps(p).get("experiments_to_best")),
    dict(id="search.time_to_best_hours", unit="h", direction="info",
         definition="first experiment created to best experiment finished",
         compute=lambda p, c: round((_exps(p).get("time_to_best_seconds") or 0)
                                    / 3600, 2)
         if _exps(p).get("time_to_best_seconds") else None),
    dict(id="search.tail_after_last_improvement", unit="experiments",
         direction="info",
         definition="scored experiments after the last best-so-far improvement",
         compute=lambda p, c: _tail_after_last_improvement(p)),
    dict(id="search.late_gain_fraction", unit="ratio", direction="info",
         definition="share of total improvement earned in the last 25% of "
                    "scored experiments",
         compute=lambda p, c: _late_gain_fraction(p)),
    # --- quality / integrity ---
    dict(id="quality.best_value", unit="metric", direction="info",
         definition="best self-reported primary metric (NOT cross-framework "
                    "comparable; see referee.*)",
         compute=lambda p, c: (_exps(p).get("best") or {}).get("value")),
    dict(id="quality.validation_identity_coverage", unit="ratio",
         direction="higher",
         definition="scored rows carrying a machine-readable validation "
                    "identity / scored rows",
         compute=lambda p, c: _validation_coverage(p)),
    dict(id="integrity.missing_result_files", unit="count", direction="lower",
         definition="DB rows with results_json but no results/metrics.json",
         compute=lambda p, c: _exps(p).get("missing_result_files")),
    dict(id="integrity.multi_realization_experiments", unit="count",
         direction="info",
         definition="experiments whose directory holds execution-completion "
                    "markers (*job*.out / *exit_code) clustered >5min apart — "
                    "the row was executed more than once",
         compute=lambda p, c: _exps(p).get("multi_realization_experiments")),
    dict(id="integrity.untracked_multi_realizations", unit="count",
         direction="lower",
         definition="multi-realization experiments whose DB row records "
                    "fix_attempts=0 — re-executions invisible to the tracking "
                    "layer (tracked repairs are excluded)",
         compute=lambda p, c: _exps(p).get("untracked_multi_realizations")),
    dict(id="integrity.primary_mutations_after_finish", unit="count",
         direction="lower",
         definition="experiments whose primary data files (results metrics/"
                    "predictions/forecasts) were written >60s after "
                    "finished_at; analysis additions (plots, summaries) are "
                    "not counted",
         compute=lambda p, c: _exps(p).get("primary_mutations_after_finish")),
    dict(id="integrity.champion_annotation_correct", unit="bool",
         direction="higher",
         definition="Conductor 'champion' annotation matches the true best "
                    "row (N/A without annotations)",
         compute=lambda p, c: _champion_annotation_correct(p)),
    dict(id="integrity.final_report_covers_best", unit="bool", direction="higher",
         definition="any reports/output/notes markdown mentions the final "
                    "best experiment's name",
         compute=lambda p, c: _report_covers_best(p)),
    # --- efficiency ---
    dict(id="efficiency.total_tokens_m", unit="Mtok", direction="info",
         definition="event-stream input+output tokens, millions",
         compute=lambda p, c: round(_total_tokens(p) / 1e6, 1)),
    dict(id="efficiency.tokens_per_scored", unit="tok", direction="lower",
         definition="(input+output tokens) / scored experiments",
         compute=lambda p, c: _per_scored(p, _total_tokens(p))),
    dict(id="efficiency.core_tokens_per_scored", unit="tok", direction="lower",
         definition="tokens of worker+strategist+fixer roles / scored — "
                    "meta (conductor/verifier/supervisor/reporter) excluded, "
                    "so frameworks are compared on the research core",
         compute=lambda p, c: _per_scored(
             p, _role_tokens(p, {"worker", "worker_implement",
                                 "worker_analyze", "strategist", "fixer"}))),
    dict(id="efficiency.meta_role_share", unit="ratio", direction="info",
         definition="conductor+verifier+supervisor tokens / total role tokens",
         compute=lambda p, c: round(
             _role_tokens(p, {"conductor", "verifier", "supervisor"})
             / max(_role_tokens(p, set(((p.get("agent_logs") or {})
                                        .get("roles") or {}).keys())), 1), 3)),
    dict(id="efficiency.reporter_share", unit="ratio", direction="info",
         definition="reporter tokens / total role tokens",
         compute=lambda p, c: round(
             _role_tokens(p, {"reporter"})
             / max(_role_tokens(p, set(((p.get("agent_logs") or {})
                                        .get("roles") or {}).keys())), 1), 3)),
    dict(id="efficiency.cost_usd", unit="USD", direction="info",
         definition="fresh/cache-read/cache-write/output at real per-model "
                    "vendor rates (tabulate.MODEL_RATES), provider counting "
                    "semantics normalized",
         compute=lambda p, c: cost_usd(
             _tokens(p), (p.get("run") or {}).get("model") or "")),
    # The two numbers that expose history-caching failures and session-depth
    # cost blowups (2026-07-29: opus-5's 2-3x deeper sessions billed 5-10x
    # because the whole history re-billed fresh each turn — quadratic in
    # depth; no scorecard row surfaced it, so three reviewer generations
    # missed it).
    dict(id="efficiency.fresh_tokens_per_call", unit="tokens", direction="lower",
         definition="fresh (uncached) input tokens per API call, provider "
                    "counting semantics normalized (OpenAI input includes "
                    "cache reads; Anthropic/lab input is fresh-only); grows "
                    "with session depth when history is not cached",
         compute=lambda p, c: (
             round(max(_tokens(p).get("input", 0)
                       - (_tokens(p).get("cache_read", 0)
                          if str((p.get("run") or {}).get("model") or "")
                          .lower().startswith("gpt") else 0), 0) / ev, 0)
             if (ev := (p.get("events") or {}).get("api_calls") or 0) else None)),
    dict(id="efficiency.cache_hit_fraction", unit="ratio", direction="higher",
         definition="cache_read / (cache_read + fresh_input), fresh "
                    "normalized per provider semantics; low values on "
                    "long-session runs mean history re-bills every turn",
         compute=lambda p, c: (
             round(cr / (cr + fresh), 4)
             if (cr := _tokens(p).get("cache_read", 0) or 0) is not None
             and (fresh := max(
                 (_tokens(p).get("input", 0) or 0)
                 - (cr if str((p.get("run") or {}).get("model") or "")
                    .lower().startswith("gpt") else 0), 0)) is not None
             and (cr + fresh) > 0 else None)),
    dict(id="efficiency.worker_prompt_chars_median", unit="chars",
         direction="info",
         definition="median first-request system-prompt size across worker "
                    "sessions",
         compute=lambda p, c: (lambda pc: int(statistics.median(pc)) if pc else None)(
             ((p.get("agent_logs") or {}).get("roles") or {})
             .get("worker", {}).get("prompt_chars") or [])),
    dict(id="efficiency.cache_read_fraction", unit="ratio", direction="higher",
         definition="cache_read tokens / input tokens",
         compute=lambda p, c: round(int(_tokens(p).get("cache_read", 0)) /
                                    max(int(_tokens(p).get("input", 0)), 1), 3)),
    # --- tools / reliability ---
    dict(id="tools.calls", unit="count", direction="info",
         definition="tool_call events in the run stream",
         compute=lambda p, c: sum(((p.get("events") or {}).get("tools")
                                   or {}).values())),
    dict(id="tools.failure_rate", unit="ratio", direction="lower",
         definition="tool results starting with a standardized error marker / "
                    "tool calls",
         compute=lambda p, c: round(
             sum(((p.get("events") or {}).get("tool_failures") or {}).values())
             / max(sum(((p.get("events") or {}).get("tools") or {}).values()), 1),
             4)),
    dict(id="tools.read_failures", unit="count", direction="lower",
         definition="'file/image not found' tool failures (path-contract "
                    "mismatches)",
         compute=lambda p, c: _read_failures(p)),
    dict(id="reliability.execution_failures", unit="count", direction="lower",
         definition="DB rows with an error and no metric",
         compute=lambda p, c: _exps(p).get("execution_failures")),
    dict(id="reliability.fix_attempts", unit="count", direction="info",
         definition="sum of fix_attempts over all rows",
         compute=lambda p, c: _exps(p).get("fix_attempts_total")),
    dict(id="memory.calls", unit="count", direction="info",
         definition="memory_* tool calls",
         compute=lambda p, c: _memory(p)[0] or None),
    dict(id="memory.failure_rate", unit="ratio", direction="lower",
         definition="failed memory_* calls / memory_* calls",
         compute=lambda p, c: round(_memory(p)[1] / _memory(p)[0], 4)
         if _memory(p)[0] else None),
    dict(id="memory.records", unit="count", direction="info",
         definition="durable memory records on disk",
         compute=lambda p, c: (p.get("memory") or {}).get("records")),
    dict(id="memory.duplicate_fraction", unit="ratio", direction="lower",
         definition="duplicate durable records / records",
         compute=lambda p, c: (p.get("memory") or {}).get("duplicate_fraction")),
    dict(id="http.rate_limited", unit="count", direction="lower",
         definition="HTTP 429 responses in run.log",
         compute=lambda p, c: _http_429(p)),
    dict(id="http.retries", unit="count", direction="lower",
         definition="client retry messages in run.log",
         compute=lambda p, c: (p.get("run_log") or {}).get("retries")),
    # --- governance (N/A where the framework has no meta layer) ---
    dict(id="governance.decisions", unit="count", direction="info",
         definition="Conductor meta_log decisions",
         compute=lambda p, c: (p.get("conductor") or {}).get("decisions_total")),
    dict(id="governance.directive_ack_rate", unit="ratio", direction="higher",
         definition="directive acknowledgements / directives issued",
         compute=lambda p, c: (
             round((p.get("conductor") or {}).get("directive_acks", 0) /
                   max((p.get("conductor") or {}).get("decision_types", {})
                       .get("directive", 0), 1), 3)
             if (p.get("conductor") or {}).get("decision_types", {})
             .get("directive") else None)),
    dict(id="governance.selfcheck_rate", unit="ratio", direction="higher",
         definition="decisions with non-empty self_check / decisions",
         compute=lambda p, c: (
             round((p.get("conductor") or {}).get("self_check_nonempty", 0) /
                   max((p.get("conductor") or {}).get("decisions_total", 0), 1), 3)
             if (p.get("conductor") or {}).get("decisions_total") else None)),
    dict(id="governance.phase_rewinds", unit="count", direction="info",
         definition="Conductor phase rewinds",
         compute=lambda p, c: (p.get("conductor") or {}).get("phase_rewinds")),
    # --- referee (from referee.json when present) ---
    dict(id="referee.artifact_coverage", unit="ratio", direction="higher",
         definition="scored experiments with loadable prediction artifacts / "
                    "scored",
         compute=lambda p, c: ((c.get("referee_runs") or {})
                               .get(p["run"]["label"]) or {})
         .get("artifact_coverage")),
    dict(id="referee.self_report_reproduced", unit="count", direction="higher",
         definition="experiments whose self-reported metric is reproduced by "
                    "the referee evaluator within 5% on some stored array",
         compute=lambda p, c: ((c.get("referee_runs") or {})
                               .get(p["run"]["label"]) or {})
         .get("self_report_reproduced")),
    # --- code quality (deterministic scan of each experiment's shipped .py
    # files; extract.py scan_code_metrics — line counts + AST functions,
    # branches, nesting; user ruling 2026-08-02: these are permanent) ---
    dict(id="code.total_lines", unit="lines", direction="info",
         definition="python lines summed over all experiment dirs",
         compute=lambda p, c: (_code_rollup(p) or {}).get("total_lines")),
    dict(id="code.total_py_files", unit="files", direction="info",
         definition=".py files summed over all experiment dirs",
         compute=lambda p, c: (_code_rollup(p) or {}).get("total_py_files")),
    dict(id="code.comment_share_overall", unit="ratio", direction="info",
         definition="comment lines / (code + comment lines) over all "
                    "experiment code",
         compute=lambda p, c: (_code_rollup(p) or {}).get("comment_share_overall")),
    dict(id="code.median_code_lines_per_experiment", unit="lines", direction="info",
         definition="median non-blank non-comment lines per experiment",
         compute=lambda p, c: _code_median(p, "code_lines")),
    dict(id="code.median_functions_per_experiment", unit="count", direction="info",
         definition="median function/method definitions per experiment",
         compute=lambda p, c: _code_median(p, "functions")),
    dict(id="code.median_mean_function_lines", unit="lines", direction="info",
         definition="median over experiments of mean function length",
         compute=lambda p, c: _code_median(p, "mean_function_lines")),
    dict(id="code.median_branches_per_100_code_lines", unit="ratio", direction="info",
         definition="median branch-node density (If/For/While/Try/With/"
                    "BoolOp/IfExp/except/comprehension per 100 code lines)",
         compute=lambda p, c: _code_median(p, "branches_per_100_code_lines")),
    dict(id="code.median_max_control_nesting", unit="depth", direction="info",
         definition="median over experiments of deepest control-flow nesting",
         compute=lambda p, c: _code_median(p, "max_control_nesting")),
    dict(id="code.ast_parse_failures", unit="count", direction="lower",
         definition="shipped .py files that do not parse",
         compute=lambda p, c: (_code_rollup(p) or {}).get("ast_parse_failures")),
    # --- distribution scalars from the raw sample arrays (the arrays
    # themselves stay in packs for charting; these are the registry's
    # always-computed summaries — user ruling 2026-08-02: every sensible
    # metric, registered permanently) ---
    dict(id="context.median_request_bytes", unit="bytes", direction="info",
         definition="median serialized api_request bytes (raw sample array)",
         compute=lambda p, c: _sample_median(p, "request_bytes")),
    dict(id="context.p90_request_bytes", unit="bytes", direction="info",
         definition="p90 serialized api_request bytes",
         compute=lambda p, c: _sample_p90(p, "request_bytes")),
    dict(id="context.median_tool_result_bytes", unit="bytes", direction="info",
         definition="median tool-result payload bytes",
         compute=lambda p, c: _sample_median(p, "tool_result_bytes")),
    dict(id="speed.median_llm_gap_seconds", unit="s", direction="lower",
         definition="median gap between consecutive model calls in one "
                    "transcript (model turnaround incl. streaming)",
         compute=lambda p, c: _sample_median(p, "llm_gap_seconds")),
    dict(id="speed.p90_llm_gap_seconds", unit="s", direction="lower",
         definition="p90 of the same per-call gap",
         compute=lambda p, c: _sample_p90(p, "llm_gap_seconds")),
    dict(id="speed.median_tool_gap_seconds", unit="s", direction="lower",
         definition="median tool execution gap",
         compute=lambda p, c: _sample_median(p, "tool_gap_seconds")),
    dict(id="speed.p90_tool_gap_seconds", unit="s", direction="lower",
         definition="p90 tool execution gap",
         compute=lambda p, c: _sample_p90(p, "tool_gap_seconds")),
    dict(id="agents.median_session_minutes", unit="min", direction="info",
         definition="median agent transcript wall-clock length",
         compute=lambda p, c: _sample_median(p, "session_minutes")),
    dict(id="agents.p90_session_minutes", unit="min", direction="info",
         definition="p90 agent transcript wall-clock length",
         compute=lambda p, c: _sample_p90(p, "session_minutes")),
    dict(id="agents.transcripts_total", unit="count", direction="info",
         definition="agent transcript files in the run",
         compute=lambda p, c: len((p.get("agent_logs") or {}).get("files") or [])
         or None),
    dict(id="agents.seats_total", unit="count", direction="info",
         definition="distinct agent seats (transcript role names)",
         compute=lambda p, c: len((p.get("agent_logs") or {}).get("roles") or {})
         or None),
    # --- tool-call shape ---
    dict(id="tools.calls_per_api_call", unit="ratio", direction="info",
         definition="tool calls / model calls (batching factor)",
         compute=lambda p, c: (
             round(sum(((p.get("events") or {}).get("tools") or {}).values()) /
                   ((p.get("events") or {}).get("api_calls") or 0), 2)
             if (p.get("events") or {}).get("api_calls") else None)),
    dict(id="tools.distinct_tools_used", unit="count", direction="info",
         definition="distinct tool names invoked at least once",
         compute=lambda p, c: len(((p.get("events") or {}).get("tools") or {}))
         or None),
    # --- phase clocks (phase3 was registered; 1 and 2 were not) ---
    dict(id="lifecycle.phase1_hours", unit="h", direction="info",
         definition="phase1 event window",
         compute=lambda p, c: _phase_hours(p, "phase1")),
    dict(id="lifecycle.phase2_hours", unit="h", direction="info",
         definition="phase2 event window",
         compute=lambda p, c: _phase_hours(p, "phase2")),
    dict(id="lifecycle.run_launches", unit="count", direction="info",
         definition="outer launch lines in run.log (restarts show here)",
         compute=lambda p, c: len((p.get("run_log") or {}).get("launches") or [])
         or None),
    # --- experiment outcomes not previously registered ---
    dict(id="reliability.negative_results", unit="count", direction="info",
         definition="experiments that errored yet still carry a metric",
         compute=lambda p, c: _exps(p).get("negative_results")),
    dict(id="reliability.launch_failure_events", unit="count", direction="lower",
         definition="experiment launch-failure signature events",
         compute=lambda p, c: sum((_exps(p).get("launch_failure_signatures")
                                   or {}).values()) or None),
    dict(id="search.median_experiment_duration_minutes", unit="min",
         direction="info",
         definition="median started-to-finished minutes over experiments "
                    "with recorded timestamps",
         compute=lambda p, c: _pct(_durations_minutes(p), 0.5)),
    dict(id="search.total_experiment_compute_hours", unit="h", direction="info",
         definition="summed experiment durations",
         compute=lambda p, c: (
             round(sum(_durations_minutes(p)) / 60, 2)
             if _durations_minutes(p) else None)),
    dict(id="search.refinement_scored_pairs", unit="count", direction="info",
         definition="explicit parent-child pairs where both carry a metric",
         compute=lambda p, c: _refinement_stats(p)[0] or None),
    dict(id="search.refinement_win_fraction", unit="ratio", direction="info",
         definition="scored children beating their parent / scored pairs",
         compute=lambda p, c: (
             round(_refinement_stats(p)[1] / _refinement_stats(p)[0], 3)
             if _refinement_stats(p)[0] else None)),
    # --- conductor decision mix (counts; None where the seat doesn't exist) ---
    dict(id="governance.parks", unit="count", direction="info",
         definition="conductor park decisions",
         compute=lambda p, c: (_decision_types(p) or {}).get("park", 0)
         if _decision_types(p) is not None else None),
    dict(id="governance.kills", unit="count", direction="info",
         definition="conductor kill decisions",
         compute=lambda p, c: (_decision_types(p) or {}).get("kill", 0)
         if _decision_types(p) is not None else None),
    dict(id="governance.priority_changes", unit="count", direction="info",
         definition="conductor set_priority decisions",
         compute=lambda p, c: (_decision_types(p) or {}).get("set_priority", 0)
         if _decision_types(p) is not None else None),
    dict(id="governance.directives_issued", unit="count", direction="info",
         definition="conductor directive decisions",
         compute=lambda p, c: (_decision_types(p) or {}).get("directive", 0)
         if _decision_types(p) is not None else None),
    dict(id="governance.directive_retirements", unit="count", direction="info",
         definition="retire_directive events (repeats included)",
         compute=lambda p, c: (p.get("conductor") or {}).get(
             "directive_retirements")),
    # --- workspace artifact inventory ---
    dict(id="artifacts.total_files", unit="count", direction="info",
         definition="classified files in the workspace inventory",
         compute=lambda p, c: sum((_inventory_classes(p) or {}).values())
         or None),
    dict(id="artifacts.verification_files", unit="count", direction="info",
         definition="inventory files classified as verification artifacts",
         compute=lambda p, c: (_inventory_classes(p) or {}).get("verification")),
    dict(id="artifacts.reports_notes_files", unit="count", direction="info",
         definition="inventory files classified as reports/notes",
         compute=lambda p, c: (_inventory_classes(p) or {}).get("reports_notes")),
]


# ---------------------------------------------------------------------------
# Efficiency decomposition — resolves *where* token-cost gaps come from.
# tokens/scored factors exactly into (sessions/scored) x (calls/session) x
# (tokens/call), computed per worker stage + strategist, with closure checks.
# ---------------------------------------------------------------------------

_CORE_ROLES = {"worker", "worker_implement", "worker_analyze", "strategist",
               "fixer"}
_STAGE_RE = re.compile(r"(implement|analyze|handoff|fix)")


def efficiency_decomposition(pack: dict) -> dict | None:
    exps = _exps(pack)
    scored = exps.get("scored") or 0
    files = (pack.get("agent_logs") or {}).get("files") or []
    if not scored or not files:
        return None
    stages: dict[str, dict] = {}
    for f in files:
        role = f.get("role") or ""
        if role not in _CORE_ROLES or not f.get("api_calls"):
            continue
        if role == "strategist":
            stage = "strategist"
        else:
            m = _STAGE_RE.search(f.get("file") or "")
            stage = f"worker_{m.group(1)}" if m else "worker_other"
        s = stages.setdefault(stage, {"sessions": 0, "calls": 0,
                                      "input": 0, "output": 0,
                                      "end_ctx": [], "invocations": 0})
        s["sessions"] += 1
        s["calls"] += int(f.get("api_calls") or 0)
        s["invocations"] += int(f.get("invocations") or 0)
        if f.get("input_last"):
            s["end_ctx"].append(int(f["input_last"]))
        t = f.get("tokens") or {}
        s["input"] += int(t.get("input") or 0)
        s["output"] += int(t.get("output") or 0)
    out = {}
    total_per_scored = 0
    for stage, s in sorted(stages.items()):
        tokens = s["input"] + s["output"]
        per_scored = round(tokens / scored)
        total_per_scored += per_scored
        out[stage] = {
            "sessions": s["sessions"],
            "sessions_per_scored": round(s["sessions"] / scored, 2),
            "calls_per_session": round(s["calls"] / max(s["sessions"], 1), 1),
            "tokens_per_call": round(tokens / max(s["calls"], 1)),
            "tokens_per_scored": per_scored,
            "end_context_median": (
                int(statistics.median(s["end_ctx"])) if s["end_ctx"] else None),
            "invocations": s["invocations"],
        }
    # strategist wake rate per phase3 hour (interval-driven wakeups).
    w = (((pack.get("events") or {}).get("phase_windows")) or {}).get("phase3") or {}
    p3h = (w.get("seconds") or 0) / 3600
    strat = stages.get("strategist", {})
    return {
        "scored": scored,
        "stages": out,
        "core_tokens_per_scored_sum": total_per_scored,
        "strategist_calls_per_phase3_hour": (
            round(strat.get("calls", 0) / p3h, 1) if p3h else None),
    }


def _stage_of(f: dict) -> str | None:
    role = f.get("role") or ""
    if role not in _CORE_ROLES or not f.get("api_calls"):
        return None
    if role == "strategist":
        return "strategist"
    m = _STAGE_RE.search(f.get("file") or "")
    return f"worker_{m.group(1)}" if m else "worker_other"


def gap_attribution(left_pack: dict, right_pack: dict) -> dict | None:
    """Additive attribution of the core tokens/scored gap to named causes.

    Worker stages present on both sides split into (A) tokens the right side
    billed on turns beyond the left side's mean session depth — computed from
    the actual per-turn input sequences — and (B) the remainder: heavier
    context at the same turn depth. The strategist splits into invocation
    count vs per-invocation cost. One-sided stages carry their whole delta.
    Causes sum to the core gap exactly (closure is reported).
    """
    sides = []
    for pack in (left_pack, right_pack):
        scored = (_exps(pack) or {}).get("scored") or 0
        if not scored:
            return None
        stages: dict[str, list[dict]] = {}
        for f in (pack.get("agent_logs") or {}).get("files") or []:
            stage = _stage_of(f)
            if stage:
                stages.setdefault(stage, []).append(f)
        sides.append((stages, scored))
    (ls, l_scored), (rs, r_scored) = sides

    def stage_tokens_per_scored(files, scored):
        total = sum(
            int((f.get("tokens") or {}).get("input") or 0)
            + int((f.get("tokens") or {}).get("output") or 0)
            for f in files
        )
        return total / scored

    causes: dict[str, float] = {}
    for stage in sorted(set(ls) | set(rs)):
        lf, rf = ls.get(stage), rs.get(stage)
        if lf and rf:
            gap = (stage_tokens_per_scored(rf, r_scored)
                   - stage_tokens_per_scored(lf, l_scored))
            if stage == "strategist":
                li = sum(int(f.get("invocations") or 0) for f in lf) or 1
                ri = sum(int(f.get("invocations") or 0) for f in rf) or 1
                l_tot = stage_tokens_per_scored(lf, l_scored) * l_scored
                r_tot = stage_tokens_per_scored(rf, r_scored) * r_scored
                per_inv = ((l_tot / li) + (r_tot / ri)) / 2
                count_effect = (ri / r_scored - li / l_scored) * per_inv
                causes["strategist: invocations per scored experiment"] = count_effect
                causes["strategist: cost per invocation"] = gap - count_effect
            else:
                n_l = [len(f.get("input_seq") or []) for f in lf]
                depth = round(sum(n_l) / max(len(n_l), 1))
                extra = sum(
                    sum((f.get("input_seq") or [])[depth:]) for f in rf
                ) / r_scored
                causes[f"{stage}: turns beyond depth {depth}"] = extra
                causes[f"{stage}: heavier context at same depth"] = gap - extra
        elif rf:
            causes[f"{stage}: stage exists only on right"] = (
                stage_tokens_per_scored(rf, r_scored))
        elif lf:
            causes[f"{stage}: stage exists only on left"] = (
                -stage_tokens_per_scored(lf, l_scored))
    total_gap = (
        sum(stage_tokens_per_scored(f, r_scored) for f in rs.values())
        - sum(stage_tokens_per_scored(f, l_scored) for f in ls.values())
    )
    return {
        "core_gap_tokens_per_scored": round(total_gap),
        "causes": {k: round(v) for k, v in causes.items()},
        "closure": round(sum(causes.values()) / total_gap, 4) if total_gap else None,
    }


def render_decomposition(pairs: list[tuple[str, str, str]],
                         runs: dict,
                         attributions: dict | None = None) -> list[str]:
    lines = ["## Efficiency decomposition (core research roles)", "",
             "tokens/scored = sessions/scored x calls/session x tokens/call, "
             "per stage. This is the *mechanism* behind any tokens-per-"
             "experiment gap; the stage sums close against "
             "`efficiency.core_tokens_per_scored`.", ""]
    for name, left, right in pairs:
        dl = (runs.get(left) or {}).get("efficiency_decomposition")
        dr = (runs.get(right) or {}).get("efficiency_decomposition")
        if not dl or not dr:
            continue
        lines.append(f"### {name}")
        lines.append("")
        lname = left.rsplit("/", 1)[-1][:22]
        rname = right.rsplit("/", 1)[-1][:22]
        lines.append(f"| stage | {lname}: sess/scored · "
                     "calls/sess · tok/call · **tok/scored** | "
                     f"{rname}: same | {rname}/{lname} tok/scored |")
        lines.append("|---|---|---|---|")
        for stage in sorted(set(dl["stages"]) | set(dr["stages"])):
            a, b = dl["stages"].get(stage), dr["stages"].get(stage)
            def cell(s):
                if not s:
                    return "—"
                return (f"{s['sessions_per_scored']} · {s['calls_per_session']}"
                        f" · {s['tokens_per_call']:,} · "
                        f"**{s['tokens_per_scored']:,}**")
            ratio = (round(b["tokens_per_scored"] / a["tokens_per_scored"], 2)
                     if a and b and a["tokens_per_scored"] else "—")
            lines.append(f"| {stage} | {cell(a)} | {cell(b)} | {ratio} |")
        lines.append(
            f"| **total** | **{dl['core_tokens_per_scored_sum']:,}** | "
            f"**{dr['core_tokens_per_scored_sum']:,}** | "
            f"**{round(dr['core_tokens_per_scored_sum'] / max(dl['core_tokens_per_scored_sum'], 1), 2)}** |")
        sw_l = dl.get("strategist_calls_per_phase3_hour")
        sw_r = dr.get("strategist_calls_per_phase3_hour")
        si_l = (dl["stages"].get("strategist") or {}).get("invocations")
        si_r = (dr["stages"].get("strategist") or {}).get("invocations")
        lines.append("")
        lines.append(
            f"Strategist: L {si_l} invocations / {sw_l} calls per phase3 hour "
            f"vs R {si_r} invocations / {sw_r} calls per phase3 hour. "
            f"Per-scored invocation cost: L "
            f"{round((si_l or 0) / max(dl['scored'], 1), 1)} vs R "
            f"{round((si_r or 0) / max(dr['scored'], 1), 1)}.")
        lines.append("")
        att = (attributions or {}).get(name)
        if att and att.get("core_gap_tokens_per_scored"):
            gap = att["core_gap_tokens_per_scored"]
            lines.append(f"**Gap attribution** (core gap {gap:,} tokens/scored; "
                         f"causes sum to {att['closure']:.1%}):")
            lines.append("")
            lines.append("| cause | tokens/scored | share of gap |")
            lines.append("|---|---|---|")
            for cause, v in sorted(att["causes"].items(),
                                   key=lambda kv: -abs(kv[1])):
                lines.append(f"| {cause} | {v:,} | {v / gap:.1%} |")
            lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Diagnostics classification
# ---------------------------------------------------------------------------


def load_rules(path: Path) -> dict:
    return json.loads(path.read_text())


def classify(tool: str, signature: str, rules: dict) -> tuple[str, str]:
    t, s = tool.lower(), signature.lower()
    for rule in rules["rules"]:
        if rule.get("tool_prefix") and not t.startswith(rule["tool_prefix"]):
            continue
        if rule["match"] in s:
            return rule["class"], rule.get("note", "")
    return "unknown", ""


def diagnostics(pack: dict, rules: dict) -> list[dict]:
    sigs = ((pack.get("events") or {}).get("tool_failure_signatures")) or {}
    rows = []
    for tool, hist in sigs.items():
        for sig, count in hist.items():
            cls, note = classify(tool, sig, rules)
            rows.append({"tool": tool, "signature": sig, "count": count,
                         "class": cls, "note": note})
    # Failed-launch outputs are not tool events; they surface here so retry
    # storms are counted and classified like everything else.
    for sig, count in (_exps(pack).get("launch_failure_signatures") or {}).items():
        cls, note = classify("launch", sig, rules)
        rows.append({"tool": "launch", "signature": sig, "count": count,
                     "class": cls, "note": note})
    rows.sort(key=lambda r: -r["count"])
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.4g}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


# ---------------------------------------------------------------------------
# Seat attribution and computed verdicts — the REPORT, not a human, declares
# which run (and model) wins, overall and per pipeline seat. Winner picking is
# mechanical: only metrics with a declared direction vote; ties and absent
# values abstain; info metrics never vote.
# ---------------------------------------------------------------------------

SEAT_BY_METRIC: dict[str, str] = {
    "lifecycle.strategist_calls_per_scored": "strategist",
    "search.scored": "strategist",
    "search.improvements": "strategist",
    "search.experiments_to_best": "strategist",
    "search.time_to_best_hours": "strategist",
    "search.tail_after_last_improvement": "strategist",
    "search.late_gain_fraction": "strategist",
    "tools.calls": "workers",
    "tools.failure_rate": "workers",
    "tools.read_failures": "workers",
    "reliability.execution_failures": "workers",
    "reliability.fix_attempts": "workers",
    "efficiency.worker_prompt_chars_median": "workers",
    "integrity.missing_result_files": "framework (phase 2)",
    "integrity.db_file_metric_mismatches": "framework (phase 2)",
    "integrity.multi_realization_experiments": "framework (phase 2)",
    "integrity.untracked_multi_realizations": "framework (phase 2)",
    "integrity.primary_mutations_after_finish": "framework (phase 2)",
    "quality.validation_identity_coverage": "framework (phase 2)",
    "referee.artifact_coverage": "framework (phase 2)",
    "referee.self_report_reproduced": "framework (phase 2)",
    "governance.decisions": "conductor",
    "governance.directive_ack_rate": "conductor",
    "governance.selfcheck_rate": "conductor",
    "governance.phase_rewinds": "conductor",
    "integrity.champion_annotation_correct": "conductor",
    "integrity.final_report_covers_best": "reporter",
    "efficiency.reporter_share": "reporter",
    "memory.calls": "memory",
    "memory.failure_rate": "memory",
    "memory.records": "memory",
    "memory.duplicate_fraction": "memory",
}
_OUTCOME_SEAT = "run outcome (whole stack)"

_SEAT_ORDER = ["strategist", "workers", "framework (phase 2)", "conductor",
               "reporter", "memory", _OUTCOME_SEAT]


def _dedupe_pairs(pairs: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Drop pairs whose unordered run set repeats (pair_key auto-pairs and
    explicit --pair specs can name the same two runs)."""
    seen: set[frozenset] = set()
    out = []
    for name, a, b in pairs:
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, a, b))
    return out


def pair_verdict(runs: dict, left: str, right: str) -> dict:
    """Mechanical winner table for one pair: per-metric votes, per-seat and
    overall tallies. Pure function of the computed metrics."""
    votes: list[dict] = []
    tallies: dict[str, dict[str, int]] = {}
    overall = {left: 0, right: 0}
    for m in METRICS:
        if m["direction"] not in ("higher", "lower"):
            continue
        va = runs[left]["metrics"].get(m["id"])
        vb = runs[right]["metrics"].get(m["id"])
        if va is None or vb is None or va == vb:
            continue
        better_left = (va > vb) if m["direction"] == "higher" else (va < vb)
        winner = left if better_left else right
        seat = SEAT_BY_METRIC.get(m["id"], _OUTCOME_SEAT)
        votes.append({"metric": m["id"], "direction": m["direction"],
                      "seat": seat, "left": va, "right": vb, "winner": winner})
        t = tallies.setdefault(seat, {left: 0, right: 0})
        t[winner] += 1
        overall[winner] += 1
    return {"left": left, "right": right, "votes": votes,
            "seat_tallies": tallies, "overall": overall}


def _verdict_word(wins_a: int, wins_b: int, a: str, b: str) -> str:
    if wins_a == wins_b:
        return "split"
    return a if wins_a > wins_b else b


def render_verdicts(pairs: list[tuple[str, str, str]], runs: dict) -> list[str]:
    lines = ["## Computed verdict — which run wins, per seat", ""]
    lines.append(
        "Mechanical and deterministic: every metric with a declared direction "
        "(↑/↓) votes for the run on its better side; ties and absent values "
        "abstain; info (·) metrics never vote. Seats aggregate their metrics' "
        "votes. When the pair shares a framework and differs in model, this "
        "is a model comparison and the verdict names the model.")
    lines.append("")
    if not pairs:
        lines.append("(no pairs)")
        lines.append("")
        return lines
    for name, a, b in pairs:
        if a not in runs or b not in runs:
            continue
        v = pair_verdict(runs, a, b)
        ma, mb = runs[a].get("model") or "?", runs[b].get("model") or "?"
        tag = {a: f"{a} ({ma})", b: f"{b} ({mb})"} if ma != mb else               {a: a, b: b}
        lines.append(f"### {name}: `{tag[a]}` vs `{tag[b]}`")
        lines.append("")
        lines.append(f"| seat | `{a}` wins | `{b}` wins | verdict |")
        lines.append("|---|---|---|---|")
        for seat in _SEAT_ORDER:
            t = v["seat_tallies"].get(seat)
            if not t:
                lines.append(f"| {seat} | 0 | 0 | no directional evidence |")
                continue
            w = _verdict_word(t[a], t[b], tag[a], tag[b])
            lines.append(f"| {seat} | {t[a]} | {t[b]} | **{w}** |")
        oa, ob = v["overall"][a], v["overall"][b]
        lines.append(f"| **overall** | **{oa}** | **{ob}** | "
                     f"**{_verdict_word(oa, ob, tag[a], tag[b])}** |")
        lines.append("")
        lines.append("Per-metric votes:")
        lines.append("")
        lines.append("| metric | seat | " + f"`{a}` | `{b}` | winner |")
        lines.append("|---|---|---|---|---|")
        for vote in v["votes"]:
            arrow = {"lower": "↓", "higher": "↑"}[vote["direction"]]
            lines.append(
                f"| {vote['metric']} ({arrow}) | {vote['seat']} | "
                f"{_fmt(vote['left'])} | {_fmt(vote['right'])} | "
                f"`{vote['winner']}` |")
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Per-experiment table — every experiment row preserved in the pack, with
# best-effort token attribution (sum of agent-log files whose filename embeds
# the experiment name; the run's estimated cost is apportioned by token share).
# ---------------------------------------------------------------------------


def experiment_table(pack: dict) -> list[dict]:
    exps = (_exps(pack) or {}).get("experiments") or []
    files = (pack.get("agent_logs") or {}).get("files") or []
    total_tokens = 0
    file_tokens: list[tuple[str, int]] = []
    for f in files:
        tk = f.get("tokens") or {}
        n = (tk.get("input") or 0) + (tk.get("output") or 0)
        total_tokens += n
        file_tokens.append((f.get("file") or "", n))
    cost = None
    for m in METRICS:
        if m["id"] == "efficiency.cost_usd":
            try:
                cost = m["compute"](pack, {})
            except Exception:  # noqa: BLE001
                cost = None
    rows = []
    for e in exps:
        name = e.get("name") or ""
        tokens = sum(n for fname, n in file_tokens if name and name in fname)
        row = {
            "id": e.get("id"),
            "name": name,
            "status": e.get("status"),
            "metric": e.get("metric"),
            "duration_seconds": e.get("duration_seconds"),
            "fix_attempts": e.get("fix_attempts"),
            "error": (str(e.get("error"))[:60] if e.get("error") else None),
            "tokens_attributed": tokens or None,
            "est_cost_usd": (round(cost * tokens / total_tokens, 2)
                             if cost and tokens and total_tokens else None),
        }
        rows.append(row)
    return rows


def render_experiment_tables(runs: dict) -> list[str]:
    lines = ["## Results per experiment", ""]
    lines.append(
        "Every experiment row preserved from the run DB. `tokens` is the sum "
        "of agent-log files whose filename embeds the experiment name (worker "
        "and verifier sessions); `est $` apportions the run's estimated cost "
        "by that token share. Experiments whose sessions cannot be matched by "
        "name show —.")
    lines.append("")
    for label in sorted(runs):
        rows = runs[label].get("experiment_table") or []
        if not rows:
            continue
        lines.append(f"### {label} ({len(rows)} experiments)")
        lines.append("")
        lines.append("| id | experiment | status | metric | dur (s) | fixes "
                     "| tokens | est $ | error |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['id']} | `{(r['name'] or '')[:44]}` | {r['status']} | "
                f"{_fmt(r['metric'])} | {_fmt(r['duration_seconds'])} | "
                f"{_fmt(r['fix_attempts'])} | {_fmt(r['tokens_attributed'])} "
                f"| {_fmt(r['est_cost_usd'])} | "
                f"{('`' + r['error'] + '`') if r['error'] else ''} |")
        lines.append("")
    return lines


def _default_pairs(runs: dict) -> list[tuple[str, str, str]]:
    """Pairs implied by pair_key: any key with exactly two runs.

    Cross-framework keys keep the historical cond-left/msml-right column
    order; same-framework keys (e.g. a model A/B on one framework) order by
    label. Keys with more or fewer than two runs pair nothing.
    """
    by_key: dict[str, list[str]] = {}
    for label, r in runs.items():
        if r.get("pair_key"):
            by_key.setdefault(r["pair_key"], []).append(label)
    pairs: list[tuple[str, str, str]] = []
    for key, labels in sorted(by_key.items()):
        if len(labels) != 2:
            continue
        a, b = sorted(labels)
        sides = {runs[l]["framework"]: l for l in (a, b)}
        if len(sides) == 2:
            fws = sorted(sides)
            a, b = sides[fws[0]], sides[fws[1]]
        pairs.append((key, a, b))
    return pairs


def render_md(bench: dict, referee: dict | None,
              pairs: list[tuple[str, str, str]] | None = None) -> str:
    lines = [
        "# Framework bench — deterministic scorecard",
        "",
        f"bench_version {bench['bench_version']}, rules_version "
        f"{bench['rules_version']}, generated {bench['generated_at']}. "
        "Every number is a pure function of the preserved run artifacts; "
        "re-running reproduces it. Directions are declared per metric; "
        "diagnosis classes come from the versioned rules file.",
        "",
        "Terms: a **forecast origin** is the timestamp a forecast is "
        "launched from; **shared origins** are the origins both runs of a "
        "pair preserved predictions for; a **validation identity** is one "
        "frozen origin list used as a common test — scores compare only "
        "within one identity, and the referee's declared winner per pair "
        "governs. Lab-hosted models (glm, kimi) have vendor cost $0 and "
        "are excluded from dollar comparisons by construction.",
        "",
    ]
    # Referee first: it is the only cross-framework quality evidence.
    lines.append("## Model quality under one referee evaluator")
    lines.append("")
    ref_runs = (referee or {}).get("runs") or {}
    if ref_runs:
        # One N-way leaderboard per domain: every rankable run, sorted by
        # its best referee-scored experiment (the same sweep the viewer
        # uses). Runs the referee marked unrankable, and self-verification-
        # only domains (returns series), stay OFF the table with the
        # reason stated — their numbers never join it.
        by_dom: dict[str, list[str]] = {}
        for label, rr in ref_runs.items():
            by_dom.setdefault(rr.get("domain") or "?", []).append(label)
        for domain in sorted(by_dom):
            kinds = {ref_runs[l].get("referee_kind") for l in by_dom[domain]}
            if kinds <= {"returns_parquet"}:
                continue  # self-verification only, no cross-run ruler
            lower = True
            for l in by_dom[domain]:
                d = (bench["runs"].get(l) or {}).get("direction")
                if d:
                    lower = d == "minimize"
            rows, unrankable = [], []
            for label in sorted(by_dom[domain]):
                rr = ref_runs[label]
                if rr.get("rankable") is False:
                    unrankable.append((label, rr.get("rankable_reason", "")))
                    continue
                best, best_exp = None, ""
                for e in (rr.get("experiments") or []):
                    for key in ("recomputed", "official_referee_score",
                                "referee_score"):
                        v = e.get(key)
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            if best is None or (v < best if lower else v > best):
                                best, best_exp = v, e.get("experiment", "")
                            break
                if best is not None:
                    rows.append((best, label, best_exp, rr))
            if not rows and not unrankable:
                continue
            lines.append(f"### {domain} — referee leaderboard "
                         f"({'lower' if lower else 'higher'} wins)")
            lines.append("")
            if rows:
                rows.sort(key=lambda t: t[0], reverse=not lower)
                # Self-report domains (bits-per-byte and kin) may rank runs
                # whose validation identities are unique per run — the
                # referee presumes a unique identity is the task's own
                # split described in the run's own words. When that
                # presumption is load-bearing, the leaderboard must say so
                # on its face rather than let a reader treat the ordering
                # as measured on one ruler.
                idents: set[str] = set()
                for _b, _l, _e, rr in rows:
                    idents.update(rr.get("validation_identities") or [])
                if len(idents) > 1:
                    lines.append(
                        f"NOTE: these {len(rows)} ranked runs report "
                        f"{len(idents)} distinct validation identities; "
                        "the ordering presumes each run's own identity is "
                        "the task's split. Treat close orderings as "
                        "indicative, not decided — per-run identities are "
                        "in referee.json.")
                    lines.append("")
                lines.append("| rank | run | framework | best experiment | "
                             "referee score | scored | coverage |")
                lines.append("|---|---|---|---|---|---|---|")
                for i, (best, label, best_exp, rr) in enumerate(rows, 1):
                    lines.append(
                        f"| {i} | `{label}` | {rr.get('framework', '')} | "
                        f"`{best_exp[:44]}` | {round(best, 6)} | "
                        f"{rr.get('scored_experiments', '—')} | "
                        f"{rr.get('artifact_coverage', '—')} |")
                lines.append("")
            for label, reason in unrankable:
                lines.append(f"- `{label}` — referee-verified but NOT on "
                             "this leaderboard: "
                             f"{reason or 'no rankable scores'}")
                lines.append("")
    if referee and referee.get("pairs"):
        for p in referee["pairs"]:
            if p.get("comparable"):
                b = p["best"]
                # Shared-origin pairs carry origin/truth-pool stats;
                # shared-truth pairs (classification tables, kernel benches)
                # share truth by construction and have neither — render the
                # fields each kind actually publishes (2026-08-03: the first
                # d5/d6 pairing crashed the chain here).
                if "shared_origins" in p:
                    ctx = (f"{p['shared_origins']} shared origins "
                           f"[{p['shared_origin_range'][0]}–"
                           f"{p['shared_origin_range'][1]}], "
                           f"truth verified identical (conflicts="
                           f"{p['truth_pool_conflicts']}); ")
                else:
                    ctx = "shared truth by construction; "
                lines.append(
                    f"- **{p['pair']}** — {ctx}board "
                    f"{p['experiments_on_board']['left']}+"
                    f"{p['experiments_on_board']['right']} experiments. "
                    f"Best {p['left']}: `{b['left']['experiment']}` "
                    f"{p['metric']}={b['left']['referee_score']}; "
                    f"best {p['right']}: `{b['right']['experiment']}` "
                    f"{p['metric']}={b['right']['referee_score']}. "
                    f"**Winner: {p[p['winner']]}**"
                )
                lines.append("")
                lines.append("| rank | run | experiment | array | referee "
                             f"{p['metric']} | self-report reproduced |")
                lines.append("|---|---|---|---|---|---|")
                for i, r in enumerate(p["leaderboard"], 1):
                    ident = r.get("self_report_identified")
                    lines.append(
                        f"| {i} | {r['run']} | `{r['experiment'][:44]}` | "
                        f"{r.get('array', '—')} | {r['referee_score']} | "
                        f"{'yes' if ident else '—' if ident is None else 'no'}"
                        " |")
                lines.append("")
            else:
                lines.append(f"- **{p['pair']}** — not comparable: "
                             f"{p.get('reason', 'one side has no '
                                     'referee-scored artifacts')}")
    elif not ref_runs:
        lines.append("(no referee results)")
    lines.append("")

    # Metric matrix per domain.
    runs = bench["runs"]
    by_domain: dict[str, list[str]] = {}
    for label, r in runs.items():
        by_domain.setdefault(r["domain"], []).append(label)
    for domain in sorted(by_domain):
        labels = sorted(by_domain[domain],
                        key=lambda l: (runs[l]["framework"], l))
        lines.append(f"## {domain}")
        lines.append("")
        header = "| metric (direction) | " + " | ".join(
            f"{runs[l]['framework']}<br>`{l[:32]}`" for l in labels
        ) + " |"
        lines.append(header)
        lines.append("|---" * (len(labels) + 1) + "|")
        for m in METRICS:
            vals = [runs[l]["metrics"].get(m["id"]) for l in labels]
            # All-None rows still render: an invisible metric is
            # indistinguishable from a dropped one (user ruling 2026-08-02).
            arrow = {"lower": "↓", "higher": "↑", "info": "·"}[m["direction"]]
            lines.append(f"| {m['id']} ({arrow}) | "
                         + " | ".join(_fmt(v) for v in vals) + " |")
        lines.append("")

    # Computed verdicts (per seat) for every pair.
    lines.extend(render_verdicts(
        _dedupe_pairs(_default_pairs(runs) + list(pairs or [])), runs))

    # Framework rollups.
    lines.append("## Framework rollups (medians across runs)")
    lines.append("")
    fws = sorted({r["framework"] for r in runs.values()})
    lines.append("| metric (direction) | " + " | ".join(fws) + " |")
    lines.append("|---" * (len(fws) + 1) + "|")
    for m in METRICS:
        row = []
        for fw in fws:
            vals = [r["metrics"].get(m["id"]) for r in runs.values()
                    if r["framework"] == fw
                    and r["metrics"].get(m["id"]) is not None]
            row.append(_fmt(statistics.median(vals)) if vals else "—")
        # Rendered even when every cell is "—": see the per-domain table.
        arrow = {"lower": "↓", "higher": "↑", "info": "·"}[m["direction"]]
        lines.append(f"| {m['id']} ({arrow}) | " + " | ".join(row) + " |")
    lines.append("")

    # Efficiency decomposition per pair.
    all_pairs = _dedupe_pairs(_default_pairs(runs) + list(pairs or []))
    lines.extend(render_decomposition(all_pairs, runs,
                                      bench.get("attributions")))

    # Diagnostics.
    lines.append("## Diagnostics — failure signatures, counted and classified")
    lines.append("")
    lines.append("Classes: **bug** = component violates its own contract; "
                 "**design** = correctly implemented behavior whose cost "
                 "follows from a design choice; **policy** = guard working as "
                 "intended (not a failure); **external** = remote service "
                 "fault; **unknown** = unclassified (extend rules.json).")
    lines.append("")
    for label in sorted(runs):
        diag = runs[label]["diagnostics"]
        if not diag:
            continue
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| count | class | tool | signature |")
        lines.append("|---|---|---|---|")
        for d in diag[:20]:
            lines.append(f"| {d['count']} | {d['class']} | {d['tool']} | "
                         f"`{d['signature'][:110]}` |")
        lines.append("")
    # Class totals per framework.
    lines.append("### Failure-class totals per framework")
    lines.append("")
    totals: dict[str, dict[str, int]] = {}
    for r in runs.values():
        t = totals.setdefault(r["framework"], {})
        for d in r["diagnostics"]:
            t[d["class"]] = t.get(d["class"], 0) + d["count"]
    classes = sorted({c for t in totals.values() for c in t})
    lines.append("| framework | " + " | ".join(classes) + " |")
    lines.append("|---" * (len(classes) + 1) + "|")
    for fw in sorted(totals):
        lines.append(f"| {fw} | " + " | ".join(
            _fmt(totals[fw].get(c, 0)) for c in classes) + " |")
    lines.append("")

    # Per-experiment results, one table per run.
    lines.extend(render_experiment_tables(runs))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_bench(corpus_path: Path, packs_dir: Path, out_dir: Path,
              rules_path: Path, referee_path: Path | None,
              pairs: list[tuple[str, str, str]] | None = None) -> dict:
    from datetime import UTC, datetime

    enforce_manifest(METRICS)
    registry = {r.label: r for r in load_registry(corpus_path)}
    packs = load_packs(packs_dir)
    rules = load_rules(rules_path)
    referee = None
    if referee_path and referee_path.is_file():
        referee = json.loads(referee_path.read_text())
    ctx = {"referee_runs": (referee or {}).get("runs", {})}

    runs = {}
    for label, pack in packs.items():
        rec = registry.get(label)
        metrics = {}
        for m in METRICS:
            try:
                metrics[m["id"]] = m["compute"](pack, ctx)
            except Exception as exc:  # noqa: BLE001 — a metric must never kill the bench
                metrics[m["id"]] = None
                metrics[f"{m['id']}.__error__"] = f"{type(exc).__name__}: {exc}"
        runs[label] = {
            "framework": pack["run"]["framework"],
            "domain": pack["run"]["domain"],
            "era": pack["run"]["era"],
            "model": (pack.get("config") or {}).get("model") or "",
            "direction": ((pack.get("experiments") or {}).get("direction")
                          or ""),
            "pair_key": (rec.pair_key if rec else ""),
            "metrics": metrics,
            "efficiency_decomposition": efficiency_decomposition(pack),
            "diagnostics": diagnostics(pack, rules),
            "experiment_table": experiment_table(pack),
        }
    attributions = {}
    for name, left, right in _dedupe_pairs(_default_pairs(runs)
                                           + list(pairs or [])):
        if left in packs and right in packs:
            att = gap_attribution(packs[left], packs[right])
            if att:
                attributions[name] = att
    bench = {
        "schema": "runcmp-bench-1",
        "bench_version": BENCH_VERSION,
        "rules_version": rules["rules_version"],
        "generated_at": datetime.now(UTC).isoformat(),
        "metric_definitions": [
            {k: v for k, v in m.items() if k != "compute"} for m in METRICS
        ],
        "runs": runs,
        "attributions": attributions,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(bench, indent=1) + "\n")
    md_text = render_md(bench, referee, pairs)
    # Completeness postcondition: every registered metric must be visible in
    # the rendered scorecard. A renderer regression that hides a metric is a
    # crash, not a quiet omission.
    unrendered = sorted(m["id"] for m in METRICS if m["id"] not in md_text)
    if unrendered:
        raise RuntimeError(f"bench.md is missing registered metrics: "
                           f"{unrendered}")
    (out_dir / "bench.md").write_text(md_text)
    return bench


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic bench over packs")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--packs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rules", type=Path,
                    default=Path(__file__).parent / "rules.json")
    ap.add_argument("--referee", type=Path, default=None,
                    help="referee.json from the referee stage (optional)")
    ap.add_argument("--pair", action="append", default=[],
                    help="extra pair for the decomposition section, "
                         "NAME=LEFT_LABEL:RIGHT_LABEL (repeatable)")
    args = ap.parse_args(argv)
    pairs = []
    for spec in args.pair:
        name, rest = spec.split("=", 1)
        left, right = rest.split(":", 1)
        pairs.append((name, left, right))
    bench = run_bench(args.corpus, args.packs, args.out, args.rules,
                      args.referee, pairs)
    n_metrics = len(bench["metric_definitions"])
    print(f"bench: {len(bench['runs'])} runs x {n_metrics} metrics -> "
          f"{args.out / 'bench.md'}")
    errors = [
        (label, k) for label, r in bench["runs"].items()
        for k in r["metrics"] if k.endswith(".__error__")
    ]
    for label, k in errors:
        print(f"  metric error: {label} {k}: {bench['runs'][label]['metrics'][k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
