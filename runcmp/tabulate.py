"""Layer 2 — deterministic pair tables and corpus aggregates.

Arithmetic only: no winners are declared here. The output (``tables.md`` +
``tables.json``) is the shared quantitative ground both the investigator and
the fact-checker work from.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from runcmp.corpus import load_registry

# USD per 1M tokens — real vendor prices, verified 2026-07-29 against the
# official pricing pages (platform.claude.com/docs/en/about-claude/pricing;
# developers.openai.com/api/docs/models): Opus 4.7 / 4.8 / 5 share one price;
# cache_write is the 1-hour-TTL rate because provider_anthropic.py requests
# {"type": "ephemeral", "ttl": "1h"}. gpt-5.6 and its -sol variant share one
# price. Lab-hosted models (glm/kimi) are free.
MODEL_RATES = {
    "claude-opus": {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 10.0},
    "gpt-5.6": {"input": 5.0, "output": 30.0, "cache_read": 0.5, "cache_write": 0.0},
    "glm": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0},
    "kimi": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0},
    # Lab-hosted, unbilled (verified live 2026-08-05: the serving gateway's
    # cost headers price these at 0.0).
    "deepseek": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0},
    "gemma": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0},
}
# Unknown models get Opus pricing (the most expensive in use) so a naming
# drift inflates a cost rather than silently zeroing it.
DEFAULT_RATES = MODEL_RATES["claude-opus"]


def rates_for(model: str) -> dict:
    m = (model or "").lower()
    for prefix, rates in MODEL_RATES.items():
        if prefix in m:
            return rates
    return DEFAULT_RATES


def _seat(pack: dict, group: str, key: str):
    """One seat-outcome number (see extract._seat_outcomes for definitions)."""
    return ((((pack.get("agent_logs") or {}).get("seat_outcomes") or {})
             .get(group) or {}).get(key))


def _seat_pair(pack: dict) -> str | None:
    """`started / answered` for the verifier — a gap means seats were killed
    before replying, which is a service outcome, not a model's choice."""
    started = _seat(pack, "verifier", "seats_started")
    answered = _seat(pack, "verifier", "seats_answered")
    if started is None:
        return None
    return f"{started} / {answered}"


def _comp_mix(pack: dict) -> str | None:
    """Compact payload-mix string: % of all sent request chars that were
    tool outputs / images / retained reasoning / tool arguments."""
    comp = (pack.get("agent_logs") or {}).get("request_composition_chars") or {}
    total = sum(v for v in comp.values() if isinstance(v, (int, float)))
    if not total:
        return None
    pct = lambda k: round(100 * (comp.get(k) or 0) / total, 1)  # noqa: E731
    return (f"{pct('tool_results')}/{pct('images')}/"
            f"{pct('thinking')}/{pct('tool_args')}")


def load_packs(packs_dir: Path) -> dict[str, dict]:
    packs = {}
    for p in sorted(packs_dir.glob("*.json")):
        pack = json.loads(p.read_text())
        packs[pack["run"]["label"]] = pack
    return packs


def cost_usd(tokens: dict, model: str = "") -> float:
    """Cost at real per-model rates, provider counting semantics normalized.

    OpenAI ``input_tokens`` INCLUDE cache reads (fresh = input - cache_read);
    Anthropic and the lab chat endpoints report fresh-only input. The model
    name selects both the rates and the semantics.
    """
    r = rates_for(model)
    inp = int(tokens.get("input", 0))
    cr = int(tokens.get("cache_read", 0))
    cw = int(tokens.get("cache_write", 0))
    out = int(tokens.get("output", 0))
    openai_counters = (model or "").lower().startswith("gpt")
    fresh = max(inp - cr, 0) if openai_counters else inp
    return round(
        (fresh * r["input"] + cr * r["cache_read"]
         + cw * r["cache_write"] + out * r["output"]) / 1e6,
        2,
    )


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _memory_stats(pack: dict) -> tuple[int, int]:
    ev = pack.get("events") or {}
    calls = sum(v for k, v in (ev.get("tools") or {}).items() if k.startswith("memory_"))
    fails = sum(
        v for k, v in (ev.get("tool_failures") or {}).items() if k.startswith("memory_")
    )
    return calls, fails


def _shared_validation(left: dict, right: dict) -> dict:
    lc = ((left.get("experiments") or {}).get("validation_census")) or {}
    rc = ((right.get("experiments") or {}).get("validation_census")) or {}
    shared = sorted((set(lc) & set(rc)) - {"unverified"})
    return {
        "left_identities": len([k for k in lc if k != "unverified"]),
        "right_identities": len([k for k in rc if k != "unverified"]),
        "shared": shared,
        "shared_best": {
            vid: {"left": lc[vid]["best"], "right": rc[vid]["best"]} for vid in shared
        },
    }


def pair_table(name: str, left: dict, right: dict) -> tuple[str, dict]:
    """Markdown table + machine-readable row dict for one cond/msml pair."""
    ll, rl = left["run"]["label"], right["run"]["label"]
    le, re_ = left.get("experiments") or {}, right.get("experiments") or {}
    lev, rev = left.get("events") or {}, right.get("events") or {}
    lt, rt = lev.get("tokens") or {}, rev.get("tokens") or {}
    lrl, rrl = left.get("run_log") or {}, right.get("run_log") or {}
    lmc, lmf = _memory_stats(left)
    rmc, rmf = _memory_stats(right)
    shared_val = _shared_validation(left, right)

    def per_scored(total, exps):
        s = exps.get("scored") or 0
        return round(total / s, 0) if s else None

    rows = [
        ("metric", f"{le.get('metric_key')} ({le.get('direction')})",
         f"{re_.get('metric_key')} ({re_.get('direction')})"),
        ("best value", (le.get("best") or {}).get("value"),
         (re_.get("best") or {}).get("value")),
        ("best experiment", (le.get("best") or {}).get("name"),
         (re_.get("best") or {}).get("name")),
        ("scored / total rows", f"{le.get('scored')}/{le.get('total')}",
         f"{re_.get('scored')}/{re_.get('total')}"),
        ("self-declared validation-set labels (runs' own metadata; "
         "0 = unlabeled, NOT proof of incomparability — referee measures)",
         shared_val["left_identities"], shared_val["right_identities"]),
        ("improvements", le.get("improvements"), re_.get("improvements")),
        ("experiments to best", le.get("experiments_to_best"),
         re_.get("experiments_to_best")),
        ("time to best (h)",
         round((le.get("time_to_best_seconds") or 0) / 3600, 2)
         if le.get("time_to_best_seconds") else None,
         round((re_.get("time_to_best_seconds") or 0) / 3600, 2)
         if re_.get("time_to_best_seconds") else None),
        ("wall clock (h)",
         round((lev.get("wall_seconds") or 0) / 3600, 2),
         round((rev.get("wall_seconds") or 0) / 3600, 2)),
        ("LLM calls", lev.get("api_calls"), rev.get("api_calls")),
        ("total tokens (in+out)",
         int(lt.get("input", 0)) + int(lt.get("output", 0)),
         int(rt.get("input", 0)) + int(rt.get("output", 0))),
        ("cache-read tokens", lt.get("cache_read"), rt.get("cache_read")),
        ("est. cost (USD)",
         cost_usd(lt, (left.get("run") or {}).get("model") or ""),
         cost_usd(rt, (right.get("run") or {}).get("model") or "")),
        ("request payload mix (out/img/think/args %, all requests)",
         _comp_mix(left), _comp_mix(right)),
        ("session growth (median last/first request chars)",
         ((left.get("agent_logs") or {}).get("request_stats") or {}).get("session_growth_ratio"),
         ((right.get("agent_logs") or {}).get("request_stats") or {}).get("session_growth_ratio")),
        ("fresh-input growth per session (tokens; flat cache-read = "
         "history uncached)",
         ((left.get("agent_logs") or {}).get("cache_pattern") or {}).get("fresh_input_growth_median"),
         ((right.get("agent_logs") or {}).get("cache_pattern") or {}).get("fresh_input_growth_median")),
        ("adapter files patched mid-run",
         (left.get("adapter_drift") or {}).get("files_patched_midrun"),
         (right.get("adapter_drift") or {}).get("files_patched_midrun")),
        ("service capacity refusals (529)",
         _seat(left, "all_seats", "capacity_refusals"),
         _seat(right, "all_seats", "capacity_refusals")),
        ("agent seats killed before any reply",
         _seat(left, "all_seats", "seats_died_before_first_response"),
         _seat(right, "all_seats", "seats_died_before_first_response")),
        ("verifier seats started / answered",
         _seat_pair(left), _seat_pair(right)),
        ("tokens per scored exp",
         per_scored(int(lt.get("input", 0)) + int(lt.get("output", 0)), le),
         per_scored(int(rt.get("input", 0)) + int(rt.get("output", 0)), re_)),
        ("execution failures", le.get("execution_failures"),
         re_.get("execution_failures")),
        ("negative results", le.get("negative_results"), re_.get("negative_results")),
        ("in-flight rows at end", le.get("inflight_rows"), re_.get("inflight_rows")),
        ("fix attempts total", le.get("fix_attempts_total"),
         re_.get("fix_attempts_total")),
        ("memory tool calls", lmc, rmc),
        ("memory tool failures", lmf, rmf),
        ("HTTP 429s",
         sum(v.get("rate_limited", 0) for v in (lrl.get("http_endpoints") or {}).values()),
         sum(v.get("rate_limited", 0) for v in (rrl.get("http_endpoints") or {}).values())),
        ("tracebacks in run.log", lrl.get("tracebacks"), rrl.get("tracebacks")),
        ("dispatcher crashes", lrl.get("dispatcher_crashes"),
         rrl.get("dispatcher_crashes")),
        ("final exit code", lrl.get("final_exit_code"), rrl.get("final_exit_code")),
        ("verification artifacts",
         (left.get("inventory") or {}).get("classes", {}).get("verification", 0),
         (right.get("inventory") or {}).get("classes", {}).get("verification", 0)),
    ]
    lines = [f"### {name}", "",
             f"| measure | {ll} | {rl} |", "|---|---|---|"]
    for label, lv, rv in rows:
        nd = 4 if label == "best value" else 2
        lines.append(f"| {label} | {_fmt(lv, nd)} | {_fmt(rv, nd)} |")
    lines.append("")
    if shared_val["shared"]:
        lines.append(
            f"Shared validation identities: {len(shared_val['shared'])} — "
            "directly comparable bests: "
            + "; ".join(
                f"`{vid[:44]}…` {ll}={_fmt(v['left'], 4)} vs {rl}={_fmt(v['right'], 4)}"
                for vid, v in shared_val["shared_best"].items()
            )
        )
    else:
        lines.append(
            "The two runs' preserved files declare no common validation-set "
            "label, so their SELF-REPORTED bests are not compared in this "
            "table. Declared labels are not the last word: the referee "
            "measures the actual overlap of preserved predictions and truth "
            "data (see referee.json) — where it declares a pair winner, that "
            "measured verdict governs, regardless of this caption."
        )
    lines.append("")

    machine = {
        "pair": name,
        "left": ll,
        "right": rl,
        "rows": {label: {"left": lv, "right": rv} for label, lv, rv in rows},
        "shared_validation": shared_val,
    }
    return "\n".join(lines), machine


def phase_table(packs: dict[str, dict], labels: list[str]) -> str:
    lines = ["### Phase wall-clock minutes (event windows)", ""]
    phases = ["phase0", "phase1", "phase2", "phase3"]
    lines.append("| run | " + " | ".join(phases) + " |")
    lines.append("|---|" + "---|" * len(phases))
    for label in labels:
        w = ((packs[label].get("events") or {}).get("phase_windows")) or {}
        cells = [
            _fmt(round(w[p]["seconds"] / 60, 1)) if p in w else "—" for p in phases
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    # full per-seat tool usage, so every role:tool count is in the tables
    lines.append("### Tools by seat (every invoked tool, count)")
    lines.append("")
    for label in labels:
        tb = ((packs[label].get("agent_logs") or {})
              .get("tools_by_role")) or {}
        by_role: dict[str, list] = {}
        for key, n in tb.items():
            role, _, tool = key.partition(":")
            by_role.setdefault(role, []).append((tool, n))
        lines.append(f"**{label}**")
        for role in sorted(by_role):
            items = sorted(by_role[role], key=lambda kv: -kv[1])
            lines.append(f"- {role}: " + ", ".join(f"{t} ×{n}" for t, n in items))
        lines.append("")
    # named API-level errors (request ids stripped by normalization upstream)
    lines.append("### API error lines seen by agents (named, top 3)")
    lines.append("")
    import re as _re
    for label in labels:
        sigs = ((packs[label].get("events") or {})
                .get("status_error_signatures")) or {}
        merged: dict[str, int] = {}
        for k, v in sigs.items():
            k = _re.sub(r"req_[A-Za-z0-9]+", "req_<id>", k)[:120]
            merged[k] = merged.get(k, 0) + v
        tops = sorted(merged.items(), key=lambda kv: -kv[1])[:3]
        if tops:
            lines.append(f"- **{label}**: " + "; ".join(f"{k} ×{v}" for k, v in tops))
    lines.append("")
    return "\n".join(lines)



def role_activity_table(packs: dict[str, dict], labels: list[str]) -> str:
    """Per-seat activity and hygiene: calls, turns, tool use, tool failures,
    clean session endings. The measured basis for any per-seat judgment."""
    roles: set[str] = set()
    for label in labels:
        roles |= set(((packs[label].get("agent_logs") or {}).get("roles")) or {})
    lines = ["### Per-seat activity (api_calls / turns / calls_per_session / "
             "tool_calls / tool_failures / sessions_ended_clean)", ""]
    lines.append("| role | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for role in sorted(roles):
        cells = []
        for label in labels:
            r = (((packs[label].get("agent_logs") or {}).get("roles")) or {}).get(role)
            if not r:
                cells.append("—")
                continue
            turns = r.get("turns")
            if isinstance(turns, list):
                turns = len(turns)
            # Session depth is the cost lever for uncached-history providers
            # (each turn re-bills the whole history: cost grows with depth^2).
            n_sessions = r.get("sessions") or r.get("files") or 0
            calls = r.get("api_calls", 0)
            depth = (f"{calls / n_sessions:.1f}" if n_sessions else "—")
            cells.append(f"{calls} / {turns or 0} / {depth} / "
                         f"{r.get('tool_calls', 0)} / {r.get('tool_failures', 0)} / "
                         f"{r.get('sessions_ended_clean', 0)}")
        lines.append(f"| {role} | " + " | ".join(cells) + " |")
    lines.append("")

    # Seat survival per role: sessions started / answered / killed before any
    # reply, plus capacity refusals. A role with started > answered lost work
    # to the serving side — never to a model's choice.
    lines.append("### Per-seat survival (sessions started / answered / killed "
                 "before any reply / 529 capacity refusals)")
    lines.append("")
    lines.append("| role | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for role in sorted(roles):
        cells = []
        for label in labels:
            r = ((((packs[label].get("agent_logs") or {})
                   .get("seat_outcomes") or {}).get("by_role") or {})
                 .get(role))
            if not r:
                cells.append("—")
                continue
            cells.append(
                f"{r.get('seats_started', 0)} / {r.get('seats_answered', 0)} / "
                f"{r.get('seats_died_before_first_response', 0)} / "
                f"{r.get('capacity_refusals', 0)}")
        lines.append(f"| {role} | " + " | ".join(cells) + " |")
    lines.append("")
    # per-seat tool failures, named
    lines.append("### Tool failures by seat (top lines)")
    lines.append("")
    for label in labels:
        fb = ((packs[label].get("agent_logs") or {})
              .get("tool_failures_by_role")) or {}
        tops = sorted(fb.items(), key=lambda kv: -kv[1])[:6]
        if tops:
            lines.append(f"- **{label}**: " + "; ".join(
                f"{k} ×{v}" for k, v in tops))
    lines.append("")
    return "\n".join(lines)

def role_table(packs: dict[str, dict], labels: list[str]) -> str:
    roles: set[str] = set()
    for label in labels:
        roles |= set(((packs[label].get("agent_logs") or {}).get("roles")) or {})
    roles = sorted(roles)
    lines = ["### Tokens by role (input+output, millions)", ""]
    lines.append("| role | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for role in roles:
        cells = []
        for label in labels:
            r = (((packs[label].get("agent_logs") or {}).get("roles")) or {}).get(role)
            if not r:
                cells.append("—")
            else:
                t = r.get("tokens") or {}
                cells.append(
                    _fmt(round((int(t.get("input", 0)) + int(t.get("output", 0))) / 1e6, 1), 1)
                )
        lines.append(f"| {role} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def corpus_aggregates(registry, packs: dict[str, dict]) -> tuple[str, dict]:
    lines = ["## Corpus aggregates", ""]
    # Attrition: every discovered run, by framework.
    att: dict[str, dict] = {}
    for rec in registry:
        a = att.setdefault(rec.framework, {"attempts": 0, "complete": 0})
        a["attempts"] += 1
        if rec.completeness == "complete":
            a["complete"] += 1
    lines.append("### Attrition (all discovered run attempts)")
    lines.append("")
    lines.append("| framework | attempts | reached a scored Phase 3 | rate |")
    lines.append("|---|---|---|---|")
    for fw, a in sorted(att.items()):
        rate = a["complete"] / a["attempts"] if a["attempts"] else 0
        lines.append(
            f"| {fw} | {a['attempts']} | {a['complete']} | {rate:.0%} |"
        )
    lines.append("")
    lines.append(
        "Attempts counted from every discovered workspace, including retries "
        "and deliberately-stopped runs; see corpus.json for per-run detail."
    )
    lines.append("")

    # Per-framework distributions over extracted packs.
    per_fw: dict[str, dict[str, list]] = {}
    for label, pack in packs.items():
        fw = pack["run"]["framework"]
        e = pack.get("experiments") or {}
        ev = pack.get("events") or {}
        t = ev.get("tokens") or {}
        scored = e.get("scored") or 0
        d = per_fw.setdefault(fw, {
            "tokens_per_scored": [], "scored_frac": [], "memory_fail_rate": [],
            "fix_attempts": [], "tracebacks": [],
        })
        if scored:
            d["tokens_per_scored"].append(
                (int(t.get("input", 0)) + int(t.get("output", 0))) / scored
            )
        if e.get("total"):
            d["scored_frac"].append(scored / e["total"])
        mc, mf = _memory_stats(pack)
        if mc:
            d["memory_fail_rate"].append(mf / mc)
        d["fix_attempts"].append(e.get("fix_attempts_total") or 0)
        d["tracebacks"].append((pack.get("run_log") or {}).get("tracebacks") or 0)
    lines.append("### Per-framework distributions (extracted runs)")
    lines.append("")
    fw_cols = sorted(per_fw)
    lines.append("| measure | " + " | ".join(fw_cols) + " |")
    lines.append("|---" * (len(fw_cols) + 1) + "|")

    def med(fw, key, pct=False, nd=0):
        vals = per_fw.get(fw, {}).get(key) or []
        if not vals:
            return "—"
        m = statistics.median(vals)
        return f"{m:.0%}" if pct else _fmt(round(m, nd) if nd else int(m))

    for label, key, pct, nd in (
        ("median tokens per scored exp", "tokens_per_scored", False, 0),
        ("median scored fraction", "scored_frac", True, 0),
        ("median memory-tool failure rate", "memory_fail_rate", True, 0),
        ("median fix attempts per run", "fix_attempts", False, 0),
        ("median tracebacks per run", "tracebacks", False, 0),
    ):
        lines.append(
            f"| {label} | "
            + " | ".join(med(fw, key, pct, nd) for fw in fw_cols) + " |"
        )
    lines.append("")
    machine = {"attrition": att, "per_framework": {
        fw: {k: v for k, v in d.items()} for fw, d in per_fw.items()
    }}
    return "\n".join(lines), machine


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic pair/corpus tables")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--packs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--pair", action="append", default=[],
        help="extra pair as NAME=LEFT_LABEL:RIGHT_LABEL (repeatable)",
    )
    args = ap.parse_args(argv)

    registry = load_registry(args.corpus)
    packs = load_packs(args.packs)

    pairs: list[tuple[str, str, str]] = []
    by_pair: dict[str, dict[str, str]] = {}
    for rec in registry:
        if rec.pair_key and rec.label in packs:
            by_pair.setdefault(rec.pair_key, {})[rec.framework] = rec.label
    for key, sides in sorted(by_pair.items()):
        fws = sorted(sides)
        if len(fws) == 2:
            pairs.append((key, sides[fws[0]], sides[fws[1]]))
    for spec in args.pair:
        name, rest = spec.split("=", 1)
        left, right = rest.split(":", 1)
        pairs.append((name, left, right))

    md = ["# runcmp deterministic tables", "",
          "Terms: a **forecast origin** is the timestamp a forecast is "
          "launched from (one origin = every sensor x every horizon step). "
          "**Shared origins** are the origins both runs of a pair preserved "
          "predictions for — cross-run scores are computed only on those. "
          "A **validation identity** is one frozen list of shared origins "
          "used as a common test; scores are comparable only within one "
          "identity. The referee's declared winner per pair is computed on "
          "the identity it marked comparable. Lab-hosted models (glm, kimi) "
          "run at vendor cost $0 and are excluded from dollar comparisons "
          "by construction.", ""]
    machine = {"pairs": [], "corpus": {}}
    for name, left_label, right_label in pairs:
        if left_label not in packs or right_label not in packs:
            md.append(f"### {name}\n\n(missing pack: {left_label} or {right_label})\n")
            continue
        text, m = pair_table(name, packs[left_label], packs[right_label])
        md.append(text)
        md.append(phase_table(packs, [left_label, right_label]))
        md.append(role_table(packs, [left_label, right_label]))
        md.append(role_activity_table(packs, [left_label, right_label]))
        machine["pairs"].append(m)
    corpus_md, corpus_machine = corpus_aggregates(registry, packs)
    md.append(corpus_md)
    machine["corpus"] = corpus_machine

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables.md").write_text("\n".join(md) + "\n")
    (args.out / "tables.json").write_text(
        json.dumps(machine, indent=1, default=str) + "\n"
    )
    print(f"wrote {args.out / 'tables.md'} ({len(machine['pairs'])} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
