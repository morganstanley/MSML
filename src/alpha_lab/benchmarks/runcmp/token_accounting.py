"""Deterministic per-run, per-model token accounting.

Merges two telemetry sources with explicitly different semantics:

1. **Workspace events** (pack ``events.tokens``): every API call the run
   logged, all providers mixed. For OpenAI calls ``input`` INCLUDES cached
   reads (``cache_read`` is the real cached subset). For Bedrock calls
   ``input`` is FRESH-ONLY and — until the 2026-07-26 provider fix —
   ``cache_read``/``cache_write`` were logged as hardcoded zeros.
2. **MLflow raw Bedrock traces** (``mlflow_usage_by_model.json``, built from
   ``mlflow.bedrock.autolog`` spans): authoritative per-model Bedrock usage
   including ``cacheReadInputTokens``/``cacheWriteInputTokens``. Only exists
   for runs launched with ``--mlflow``.

Output rows per run: one row per Bedrock model (from MLflow), plus one
derived OpenAI-side residual row (events totals minus the Bedrock share)
when both sources exist, or a single events row otherwise. Bedrock runs
with no MLflow record get ``cache_read = unknown`` — never zero.

Cost is a stated convention, not billing truth: fresh input 2.5, cache read
0.25, output 10.0 per 1M tokens; Bedrock cache WRITE 5.0 per 1M (2x fresh —
1h-TTL cachePoint premium); OpenAI cache writes bill inside normal input.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Real per-model vendor rates live in one place: tabulate.MODEL_RATES
# (verified 2026-07-29 against platform.claude.com and developers.openai.com;
# opus 4.7/4.8/5 share one price, cache-write is the 1h-TTL rate the
# providers request, gpt-5.6(-sol) shares one price, lab-hosted glm/kimi are
# free).
from alpha_lab.benchmarks.runcmp.tabulate import MODEL_RATES, rates_for

ACCOUNTING_VERSION = "2"


def _safe(label: str) -> str:
    return label.replace("/", "__").replace("#", "_")


def _cost(fresh: float, cache_read: float, output: float,
          cache_write: float = 0.0, model: str = "") -> float:
    r = rates_for(model)
    return round((fresh * r["input"] + cache_read * r["cache_read"]
                  + output * r["output"]
                  + cache_write * r["cache_write"]) / 1e6, 2)


def _counterfactual_cached_input_cost(
        pack: dict, model: str) -> tuple[float, float] | None:
    """Input cost had conversation history been cached (labeled counterfactual).

    The pre-fix harness never marked history cacheable, so every call
    re-billed the whole session fresh. This computes, from each session's
    REAL per-call fresh-input sequence, what the same traffic would have
    cost with the moving cache breakpoint (shipped 2026-07-29): each call's
    new suffix written once at the cache-write rate, the prior history read
    at the cache-read rate. Sequences are capped at 300 calls per session in
    the packs; the per-session ratio is applied to the session's full input
    total, which keeps the estimate honest for deeper sessions. This is an
    ESTIMATE for fair comparison — the real bills are the other column.
    """
    r = rates_for(model)
    files = (pack.get("agent_logs") or {}).get("files") or []
    actual_cost = cf_cost = 0.0
    for f in files:
        seq = f.get("input_seq") or []
        total = f.get("input_sum") or 0
        if not seq or not total:
            continue
        s = sum(seq)
        if not s:
            continue
        writes = sum(max(seq[i] - (seq[i - 1] if i else 0), 0)
                     for i in range(len(seq)))
        reads = sum(seq[i - 1] for i in range(1, len(seq)))
        scale = total / s
        actual_cost += total * r["input"]
        cf_cost += (writes * scale * r["cache_write"]
                    + reads * scale * r["cache_read"])
    if actual_cost == 0:
        return None
    return round(cf_cost / 1e6, 2), round(actual_cost / 1e6, 2)


def _ledger_by_model(workspace: str) -> dict[str, dict]:
    """Per-model token sums from a run's own ledger (meta/token_usage.jsonl).

    The ledger has one line per API call with provider, model, and all four
    counters — the only per-model source for runs that mix models (cond's
    conductor runs opus-4-7 while the main agents run the configured model).
    Counters keep each provider's native semantics: Anthropic input is
    fresh-only; OpenAI input includes cache reads.
    """
    path = Path(workspace) / "meta" / "token_usage.jsonl"
    if not path.is_file():
        return {}
    agg: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(rec.get("model") or "unknown")
            a = agg.setdefault(key, {"provider": str(rec.get("provider") or ""),
                                     "calls": 0, "input": 0, "output": 0,
                                     "cache_read": 0, "cache_write": 0})
            a["calls"] += 1
            a["input"] += int(rec.get("input_tokens") or 0)
            a["output"] += int(rec.get("output_tokens") or 0)
            a["cache_read"] += int(rec.get("cache_read_input_tokens") or 0)
            a["cache_write"] += int(rec.get("cache_write_input_tokens") or 0)
    return agg


def build(corpus_path: Path, packs_dir: Path, recovery_path: Path | None,
          mapping: dict[str, str], out_dir: Path) -> dict:
    corpus = json.loads(corpus_path.read_text())
    runs = corpus.get("runs", corpus if isinstance(corpus, list) else [])
    recovery = (json.loads(recovery_path.read_text())
                if recovery_path and recovery_path.is_file() else {})

    result: dict = {"version": ACCOUNTING_VERSION,
                    "rates_per_1M_usd": MODEL_RATES,
                    "runs": {}}
    for r in runs:
        label = r["label"]
        run_model = str(r.get("model") or "")
        pack_file = packs_dir / f"{_safe(label)}.json"
        ev = {}
        pack: dict = {}
        if pack_file.is_file():
            pack = json.loads(pack_file.read_text())
            ev = (pack.get("events", {}) or {}).get("tokens", {}) or {}
        ev_row = {k: int(ev.get(k) or 0)
                  for k in ("input", "output", "cache_read", "cache_write",
                            "reasoning")}
        entry: dict = {"events_totals_all_providers": ev_row, "models": {}}

        # Per-model ledger (cond runs): the exact source, split by model.
        ledger = _ledger_by_model(str(r.get("workspace") or ""))
        if ledger:
            led_in = led_out = 0
            for model_id, m in sorted(ledger.items()):
                openai_counters = (m["provider"] or "").lower() in ("openai", "grok")
                fresh = (max(m["input"] - m["cache_read"], 0)
                         if openai_counters else m["input"])
                entry["models"][model_id] = {
                    "provider": m["provider"], "source": "token_ledger",
                    "calls": m["calls"],
                    "fresh_input": fresh, "cache_read": m["cache_read"],
                    "cache_write": m["cache_write"], "output": m["output"],
                    "prompt_total": fresh + m["cache_read"] + m["cache_write"],
                    "cost_usd": _cost(fresh, m["cache_read"], m["output"],
                                      m["cache_write"], model_id),
                }
                led_in += m["input"]
                led_out += m["output"]
            entry["ledger_vs_events_delta"] = {
                "input": ev_row["input"] - led_in,
                "output": ev_row["output"] - led_out,
                "note": ("sanity cross-check: run ledger totals vs the "
                         "event-stream totals; small deltas are calls that "
                         "crashed between the API return and one of the two "
                         "recorders"),
            }
            if run_model.lower().startswith("claude"):
                cf = _counterfactual_cached_input_cost(pack, run_model)
                if cf:
                    entry["input_cost_if_history_cached_usd"] = cf[0]
                    entry["input_cost_actual_usd"] = cf[1]
            result["runs"][label] = entry
            continue

        exp_key = mapping.get(label)
        rec = recovery.get(exp_key or "", {})
        models = (rec or {}).get("models", {})
        bed_in = bed_out = 0
        for model_id, m in sorted(models.items()):
            fresh = int(m.get("inputTokens") or 0)
            cread = int(m.get("cacheReadInputTokens") or 0)
            cwrite = int(m.get("cacheWriteInputTokens") or 0)
            outp = int(m.get("outputTokens") or 0)
            bed_in += fresh
            bed_out += outp
            entry["models"][model_id] = {
                "provider": "bedrock", "source": "mlflow_raw",
                "calls": int(m.get("calls") or 0),
                "fresh_input": fresh, "cache_read": cread,
                "cache_write": cwrite, "output": outp,
                "prompt_total": fresh + cread + cwrite,
                "cost_usd": _cost(fresh, cread, outp, cwrite, model_id),
            }
        if models:
            # OpenAI-side residual: events count every provider's calls;
            # Bedrock's events input is fresh-only, same basis as MLflow's
            # inputTokens, so the subtraction is semantically consistent.
            res_in = ev_row["input"] - bed_in
            res_out = ev_row["output"] - bed_out
            if res_in > 0 or res_out > 0:
                entry["models"]["openai_residual"] = {
                    "provider": "openai", "source": "events_minus_bedrock",
                    "fresh_input": max(res_in - ev_row["cache_read"], 0),
                    "cache_read": ev_row["cache_read"],
                    "cache_write": ev_row["cache_write"],
                    "output": max(res_out, 0),
                    "prompt_total": max(res_in, 0),
                    "cost_usd": _cost(
                        max(res_in - ev_row["cache_read"], 0),
                        ev_row["cache_read"], max(res_out, 0),
                        model="gpt-5.6"),
                    "note": "residual; OpenAI input INCLUDES cached reads",
                }
        else:
            # Provider from the run's own recorded model, never from label
            # guessing: the native-run labels say "o48"/"o5", not "opus",
            # so the old guess dropped every native Claude run into the
            # OpenAI branch — wrong counting semantics and wrong prices.
            model = (r.get("model") or "").lower()
            if r.get("provider"):
                provider = r["provider"]
            elif model.startswith("claude"):
                provider = "anthropic"
            elif re.search(r"_o(48|5)(_|$)|opus", label):
                provider = "anthropic"
            else:
                provider = "openai"
            if provider == "anthropic":
                entry["models"]["anthropic_events"] = {
                    "provider": "anthropic", "source": "events",
                    "fresh_input": ev_row["input"],
                    "cache_read": ev_row["cache_read"],
                    "cache_write": ev_row["cache_write"],
                    "output": ev_row["output"],
                    "prompt_total": (ev_row["input"] + ev_row["cache_read"]
                                     + ev_row["cache_write"]),
                    "cost_usd": _cost(
                        ev_row["input"], ev_row["cache_read"],
                        ev_row["output"], ev_row["cache_write"],
                        model=run_model),
                    "note": ("native Anthropic: events input is FRESH-ONLY; "
                             "cache counters are recorded per call. Includes "
                             "a small OpenAI-proxied web-search share at the "
                             "same convention."),
                }
            elif provider == "bedrock":
                entry["models"]["bedrock_events_only"] = {
                    "provider": "bedrock", "source": "events_only",
                    "fresh_input": ev_row["input"],
                    "cache_read": "unknown",
                    "cache_write": "unknown",
                    "output": ev_row["output"],
                    "prompt_total": f">={ev_row['input']}",
                    "cost_usd_floor": _cost(
                        ev_row["input"], 0, ev_row["output"],
                        model=run_model),
                    "note": ("no MLflow record (run launched without "
                             "--mlflow); Bedrock cache telemetry was "
                             "dropped pre-fix — cache_read is UNKNOWN, "
                             "not zero; cost is a floor"),
                }
            else:
                entry["models"]["openai_events"] = {
                    "provider": "openai", "source": "events",
                    "fresh_input": ev_row["input"] - ev_row["cache_read"],
                    "cache_read": ev_row["cache_read"],
                    "cache_write": ev_row["cache_write"],
                    "output": ev_row["output"],
                    "prompt_total": ev_row["input"],
                    "cost_usd": _cost(
                        ev_row["input"] - ev_row["cache_read"],
                        ev_row["cache_read"], ev_row["output"],
                        model=run_model),
                }
        if run_model.lower().startswith("claude"):
            cf = _counterfactual_cached_input_cost(pack, run_model)
            if cf:
                entry["input_cost_if_history_cached_usd"] = cf[0]
                entry["input_cost_actual_usd"] = cf[1]
        result["runs"][label] = entry

    md = ["# Token accounting (deterministic; real per-model vendor rates)", "",
          "Rates per 1M tokens (verified 2026-07-29, vendor pricing pages): "
          + "; ".join(
              f"{name}: in ${r['input']}, out ${r['output']}, "
              f"cache-read ${r['cache_read']}, cache-write ${r['cache_write']} (1h TTL)"
              for name, r in MODEL_RATES.items())
          + ". Anthropic/lab input counts EXCLUDE cache reads; OpenAI "
            "input_tokens INCLUDE them — rows below are normalized "
            "(fresh_input never contains cached reads). cond runs are split "
            "per model from the run's own token ledger; msml runs are "
            "single-model event totals.",
          "",
          "**Lab-hosted models (glm, kimi) run at vendor cost $0** — the lab "
          "serves them on its own hardware, no per-token bill exists, and "
          "every dollar column below therefore EXCLUDES them from cost "
          "comparisons by construction. Their token volumes are still real "
          "and comparable.",
          ""]
    for label, entry in result["runs"].items():
        md.append(f"### {label}")
        md.append("| model (source) | calls | fresh_input | cache_read | "
                  "cache_write | output | prompt_total | cost USD |")
        md.append("|---|---|---|---|---|---|---|---|")
        for mid, m in entry["models"].items():
            cost = m.get("cost_usd", m.get("cost_usd_floor"))
            floor = "cost_usd_floor" in m
            md.append(
                f"| {mid} ({m['source']}) | {m.get('calls', '—')} | "
                f"{m['fresh_input']:,} | "
                f"{m['cache_read'] if isinstance(m['cache_read'], str) else format(m['cache_read'], ',')} | "
                f"{m['cache_write'] if isinstance(m['cache_write'], str) else format(m['cache_write'], ',')} | "
                f"{m['output']:,} | {m['prompt_total'] if isinstance(m['prompt_total'], str) else format(m['prompt_total'], ',')} | "
                f"{'>=' if floor else ''}{cost} |")
            if m.get("note"):
                md.append(f"|  | | | | | | | _{m['note']}_ |")
        # The counterfactual line exists to flag the (since fixed)
        # history-recaching harness defect. On post-fix runs actual and
        # counterfactual agree to pennies, and printing the old blame line
        # sent reviewers chasing a bug that no longer exists (user order
        # 2026-08-04: excise it). Values stay in token_accounting.json
        # unconditionally; the markdown narrates only a MATERIAL gap.
        cf = entry.get("input_cost_if_history_cached_usd")
        actual = entry.get("input_cost_actual_usd")
        if (cf is not None and actual is not None
                and float(actual) - float(cf) > 1.0):
            md.append(
                f"|  | | | | | | | _input cost had history been cached "
                f"(counterfactual ESTIMATE for fair comparison): "
                f"${cf} vs actual ${actual} — a history-recaching harness "
                f"regression billed the difference; use the counterfactual "
                f"for model-vs-model cost_ |")
        ev = entry["events_totals_all_providers"]
        md.append(f"| _events total (mixed semantics)_ | — | "
                  f"{ev['input']:,} | {ev['cache_read']:,} | "
                  f"{ev['cache_write']:,} | {ev['output']:,} | — | — |")
        md.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "token_accounting.json").write_text(
        json.dumps(result, indent=2))
    (out_dir / "token_accounting.md").write_text("\n".join(md))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="runcmp token-accounting",
        description="Per-run, per-model token and cost ledger")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--packs", type=Path, required=True)
    ap.add_argument("--recovery", type=Path, default=None,
                    help="mlflow_usage_by_model.json")
    ap.add_argument("--map", action="append", default=[],
                    help="LABEL=RECOVERY_KEY (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    mapping = dict(spec.split("=", 1) for spec in args.map)
    result = build(args.corpus, args.packs, args.recovery, mapping, args.out)
    print(f"token accounting: {len(result['runs'])} runs -> "
          f"{args.out}/token_accounting.md")
    return 0


if __name__ == "__main__":
    main()
