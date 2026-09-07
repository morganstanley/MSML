"""The change gate: prove a harness change helps and hurts nothing.

The paved path for proposing a harness change (a PR): run the benchmark
campaign with the OLD harness and with the PROPOSED harness, then

    python -m alpha_lab.benchmarks.runcmp gate \\
        --corpus ... --bench ... --referee ... \\
        --baseline era=july_baseline --candidate era=myfix_0807 \\
        --out gate_out/

The gate pairs the two sides per (task, model) cell and evaluates a
versioned, declarative policy (``gate_policy.json`` beside this module is
the default; ``--policy`` overrides):

- **guards** — the candidate may not be worse than the baseline by more
  than each guard's tolerance (final score under the independent referee,
  wall clock, cost, failures, code quality, ... — all data, not code);
- **improvements** — at least one improvement metric must be better than
  the baseline by at least its minimum gain, somewhere.

The output is a SCREEN, not the decision. Humans get the evidence in a
common, easy-to-interpret form (``gate.md``; and via ``publish --gate``
one native MLflow run whose charts are the per-cell deltas and whose
description is the full table) and decide for themselves — the change is
multidimensional and single runs carry wide uncertainty, so a mechanical
verdict would be false confidence. The mechanical policy reading (PASS /
FAIL) is still computed for CI (exit code 0/1) and stated on the page as
what it is: a screen. The stage also writes ``mission.md`` so LLM
reviewers can be run over the same evidence — their mission REQUIRES an
explicit argued PASS/FAIL recommendation (several reviewers, comparable
positions), which is where opinionated judgment belongs.

Selectors are ``key=value`` pairs over the corpus registry fields
(``era=...``, ``framework=...``, ``model=...``; comma for AND, repeatable
flag for OR). Sides are compared cell-by-cell: same task, same model.
Model-mismatched comparisons are refused unless ``--allow-model-mismatch``
(a model change is a different experiment, not a harness change).

Everything here is deterministic; re-running reproduces the verdict.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

from alpha_lab.benchmarks.runcmp.corpus import RunRecord, load_registry

DEFAULT_POLICY = Path(__file__).with_name("gate_policy.json")

# Signed convention used everywhere below: positive pct/abs = candidate is
# WORSE than baseline, negative = better. One sign rule, stated once, so
# tables and charts cannot flip meaning between metrics of opposite
# directions.
_WORSE_SIGN_NOTE = ("positive = candidate worse than baseline, "
                    "negative = candidate better")


def parse_selector(specs: list[str]) -> list[dict[str, str]]:
    """``era=x,model=y`` -> [{"era": "x", "model": "y"}]; repeated flags OR."""
    out = []
    for spec in specs:
        clause: dict[str, str] = {}
        for part in spec.split(","):
            key, _, val = part.partition("=")
            if not val:
                raise ValueError(
                    f"selector part {part!r} is not key=value "
                    "(keys: era, framework, model, domain, label)")
            if key not in ("era", "framework", "model", "domain", "label"):
                raise ValueError(f"unknown selector key {key!r}")
            clause[key] = val
        out.append(clause)
    return out


def select_runs(records: list[RunRecord],
                selector: list[dict[str, str]]) -> list[RunRecord]:
    picked = []
    for r in records:
        for clause in selector:
            if all(fnmatch.fnmatch(str(getattr(r, k)), v)
                   for k, v in clause.items()):
                picked.append(r)
                break
    return picked


def _referee_best(referee: dict, label: str, lower: bool) -> float | None:
    """The run's best referee-scored experiment — the same sweep the viewer
    uses, honoring the referee's rankability verdict."""
    rr = (referee.get("runs") or {}).get(label) or {}
    if rr.get("rankable") is False:
        return None
    best = None
    for e in (rr.get("experiments") or []):
        for key in ("recomputed", "official_referee_score", "referee_score"):
            v = e.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if best is None or (v < best if lower else v > best):
                    best = v
                break
    return best


def _domain_directions(bench: dict, referee: dict) -> dict[str, bool]:
    """domain -> lower_is_better, from bench run directions with referee
    pair records as fallback."""
    out: dict[str, bool] = {}
    for run in (bench.get("runs") or {}).values():
        d = run.get("direction")
        if d and run.get("domain") not in out:
            out[run["domain"]] = d == "minimize"
    for p in (referee.get("pairs") or []):
        left = p.get("left") or ""
        dom = left.split("/")[1] if "/" in left else ""
        if dom and dom not in out and "lower_is_better" in p:
            out[dom] = bool(p["lower_is_better"])
    return out


def _value(entry: dict, source: str, mid: str, bench_run: dict,
           referee: dict, label: str, lower: bool):
    if source == "referee":
        return _referee_best(referee, label, lower)
    v = (bench_run.get("metrics") or {}).get(mid)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) \
        else None


def _worseness(base: float, cand: float, lower_is_better: bool
               ) -> tuple[float, float | None]:
    """(worse_abs, worse_pct) under the one sign rule. pct is None when the
    baseline is zero (division would manufacture a number)."""
    worse_abs = (cand - base) if lower_is_better else (base - cand)
    worse_pct = (worse_abs / abs(base) * 100.0) if base else None
    return worse_abs, worse_pct


def _pick_cell_run(runs: list[RunRecord], referee: dict,
                   lower_by_dom: dict[str, bool]) -> RunRecord:
    """One run represents a side in a cell: the referee-best one (falls back
    to most terminal rows). Count is reported so best-of-N is visible."""
    def key(r: RunRecord):
        lower = lower_by_dom.get(r.domain, True)
        best = _referee_best(referee, r.label, lower)
        unscored = best is None
        score = (best if best is not None else 0.0)
        return (unscored, score if lower else -score, -r.db_terminal_rows)
    return sorted(runs, key=key)[0]


def evaluate(records: list[RunRecord], bench: dict, referee: dict,
             policy: dict, baseline_sel: list[dict], candidate_sel: list[dict],
             allow_model_mismatch: bool = False) -> dict:
    known_bench_ids = set()
    for run in (bench.get("runs") or {}).values():
        known_bench_ids |= set(run.get("metrics") or {})
    for rule in policy.get("guards", []) + policy.get("improvements", []):
        if rule.get("source") == "bench" and rule["id"] not in known_bench_ids:
            raise SystemExit(
                f"gate policy names unknown bench metric {rule['id']!r} — "
                "fix the policy (ids are bench.json's metrics keys)")

    base_runs = select_runs(records, baseline_sel)
    cand_runs = select_runs(records, candidate_sel)
    if not base_runs or not cand_runs:
        raise SystemExit(
            f"selector matched no runs (baseline: {len(base_runs)}, "
            f"candidate: {len(cand_runs)}) — check era/framework/model "
            "values against `runcmp index` output")
    lower_by_dom = _domain_directions(bench, referee)

    def cells(runs: list[RunRecord]) -> dict[tuple[str, str], list[RunRecord]]:
        out: dict[tuple[str, str], list[RunRecord]] = {}
        for r in runs:
            out.setdefault((r.domain, r.model), []).append(r)
        return out

    bcells, ccells = cells(base_runs), cells(cand_runs)
    if not allow_model_mismatch:
        shared = set(bcells) & set(ccells)
    else:
        # collapse the model axis: pair per domain, models noted
        def collapse(cs):
            out: dict[tuple[str, str], list[RunRecord]] = {}
            for (dom, _m), rs in cs.items():
                out.setdefault((dom, "*"), []).extend(rs)
            return out
        bcells, ccells = collapse(bcells), collapse(ccells)
        shared = set(bcells) & set(ccells)

    result: dict = {
        "schema": "runcmp-gate-1",
        "policy_version": policy.get("policy_version"),
        "sign_convention": _WORSE_SIGN_NOTE,
        "baseline": {"selector": baseline_sel,
                     "runs": sorted(r.label for r in base_runs)},
        "candidate": {"selector": candidate_sel,
                      "runs": sorted(r.label for r in cand_runs)},
        "cells": [], "coverage": {"baseline_only": [], "candidate_only": []},
        "violations": [], "improvements_shown": [], "not_evaluable": [],
    }
    for key in sorted(set(bcells) - set(ccells)):
        result["coverage"]["baseline_only"].append(list(key))
    for key in sorted(set(ccells) - set(bcells)):
        result["coverage"]["candidate_only"].append(list(key))

    bench_runs = bench.get("runs") or {}
    for dom, model in sorted(shared):
        lower = lower_by_dom.get(dom, True)
        b = _pick_cell_run(bcells[(dom, model)], referee, lower_by_dom)
        c = _pick_cell_run(ccells[(dom, model)], referee, lower_by_dom)
        cell: dict = {
            "domain": dom, "model": model,
            "baseline_run": b.label, "candidate_run": c.label,
            "baseline_n": len(bcells[(dom, model)]),
            "candidate_n": len(ccells[(dom, model)]),
            "lower_is_better": lower, "guards": [], "improvements": [],
        }
        for rule in policy.get("guards", []):
            rule_lower = (lower if rule.get("source") == "referee"
                          else rule.get("direction", "lower") == "lower")
            bv = _value(rule, rule["source"], rule["id"],
                        bench_runs.get(b.label) or {}, referee, b.label,
                        lower)
            cv = _value(rule, rule["source"], rule["id"],
                        bench_runs.get(c.label) or {}, referee, c.label,
                        lower)
            row = {"id": rule["id"], "label": rule.get("label", rule["id"]),
                   "baseline": bv, "candidate": cv}
            if bv is None or cv is None:
                row["status"] = ("missing:" + rule.get("on_missing", "skip"))
                if rule.get("on_missing", "skip") == "fail":
                    result["violations"].append(
                        {**row, "domain": dom, "model": model,
                         "reason": "guard not evaluable (missing value) "
                                   "and policy says on_missing=fail"})
                else:
                    result["not_evaluable"].append(
                        {**row, "domain": dom, "model": model})
            else:
                worse_abs, worse_pct = _worseness(bv, cv, rule_lower)
                row["worse_abs"] = worse_abs
                row["worse_pct"] = worse_pct
                tol_pct = rule.get("tolerance_pct")
                tol_abs = rule.get("tolerance_abs")
                violated = False
                if tol_abs is not None:
                    violated = worse_abs > tol_abs
                elif tol_pct is not None:
                    violated = (worse_pct is not None
                                and worse_pct > tol_pct)
                row["status"] = "VIOLATED" if violated else "ok"
                if violated:
                    # state the violation in the SAME unit the tolerance is
                    # written in — an absolute-count rule quoted in percent
                    # reads as a different rule than the policy declares
                    if tol_abs is not None:
                        reason = (f"worse by {worse_abs:g} "
                                  f"(tolerance {tol_abs:g})")
                    else:
                        reason = (f"worse by {worse_pct:.2f}% "
                                  f"(tolerance {tol_pct}%)")
                    result["violations"].append(
                        {**row, "domain": dom, "model": model,
                         "reason": reason})
            cell["guards"].append(row)
        for rule in policy.get("improvements", []):
            rule_lower = (lower if rule.get("source") == "referee"
                          else rule.get("direction", "lower") == "lower")
            bv = _value(rule, rule["source"], rule["id"],
                        bench_runs.get(b.label) or {}, referee, b.label,
                        lower)
            cv = _value(rule, rule["source"], rule["id"],
                        bench_runs.get(c.label) or {}, referee, c.label,
                        lower)
            row = {"id": rule["id"], "label": rule.get("label", rule["id"]),
                   "baseline": bv, "candidate": cv}
            if bv is not None and cv is not None:
                worse_abs, worse_pct = _worseness(bv, cv, rule_lower)
                row["worse_abs"], row["worse_pct"] = worse_abs, worse_pct
                gain_pct = rule.get("min_gain_pct")
                gain_abs = rule.get("min_gain_abs")
                shown = False
                if gain_abs is not None:
                    shown = -worse_abs >= gain_abs
                elif gain_pct is not None:
                    shown = (worse_pct is not None
                             and -worse_pct >= gain_pct)
                row["improved"] = shown
                if shown:
                    result["improvements_shown"].append(
                        {**row, "domain": dom, "model": model})
            cell["improvements"].append(row)
        result["cells"].append(cell)

    missing_rule = policy.get("missing_candidate_cell", "fail")
    for dom, model in result["coverage"]["baseline_only"]:
        if missing_rule == "fail":
            result["violations"].append({
                "id": "coverage", "label": "task coverage", "domain": dom,
                "model": model,
                "reason": "the candidate fields no run on a (task, model) "
                          "cell the baseline covers"})
    passed = not result["violations"]
    if policy.get("require_improvement", True):
        passed = passed and bool(result["improvements_shown"])
        if not result["improvements_shown"]:
            result["no_improvement"] = (
                "no improvement metric cleared its minimum gain on any "
                "cell — the policy requires the change to demonstrably "
                "improve something")
    result["verdict"] = "PASS" if passed else "FAIL"
    if not result["cells"]:
        result["verdict"] = "FAIL"
        result["violations"].append({
            "id": "coverage", "label": "task coverage",
            "reason": "no (task, model) cell exists on BOTH sides — "
                      "nothing was comparable"})
    return result


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def render_md(g: dict) -> str:
    nflag = len(g["violations"])
    nimp = len(g["improvements_shown"])
    lines = [
        f"# Change screen — {nflag} policy flag(s), "
        f"{nimp} demonstrated improvement(s)",
        "",
        f"Candidate `{g['candidate']['selector']}` vs baseline "
        f"`{g['baseline']['selector']}` — policy version "
        f"{g['policy_version']}. Sign convention: {g['sign_convention']}.",
        "",
        f"The mechanical policy reading is **{g['verdict']}** (the CI exit "
        "code) — a screen, not the decision. The decision belongs to the "
        "human reading this table and to the LLM reviewers run over the "
        "same evidence (each is required to state an explicit "
        "`Recommendation: PASS/FAIL` with its argument).",
        "",
    ]
    if g["violations"]:
        lines.append("## Policy flags (each trips the mechanical screen; confirm or overrule with evidence)")
        lines.append("")
        for v in g["violations"]:
            lines.append(f"- **{v['label']}** on {v.get('domain', '?')}"
                         f" [{v.get('model') or 'any model'}]: {v['reason']}")
        lines.append("")
    if g.get("no_improvement"):
        lines.append(f"## No improvement demonstrated\n\n{g['no_improvement']}\n")
    if g["improvements_shown"]:
        lines.append("## Improvements demonstrated")
        lines.append("")
        for v in g["improvements_shown"]:
            better = (-v["worse_pct"] if v.get("worse_pct") is not None
                      else -v["worse_abs"])
            unit = "%" if v.get("worse_pct") is not None else ""
            lines.append(f"- **{v['label']}** on {v['domain']}"
                         f" [{v.get('model') or 'any model'}]: better by "
                         f"{better:.2f}{unit} "
                         f"({_fmt(v['baseline'])} → {_fmt(v['candidate'])})")
        lines.append("")
    for cell in g["cells"]:
        lines.append(f"## {cell['domain']} [{cell['model'] or 'model n/a'}] — "
                     f"candidate `{cell['candidate_run']}` "
                     f"(best of {cell['candidate_n']}) vs baseline "
                     f"`{cell['baseline_run']}` (best of {cell['baseline_n']})")
        lines.append("")
        lines.append("| check | baseline | candidate | candidate worse by | status |")
        lines.append("|---|---|---|---|---|")
        for row in cell["guards"]:
            worse = (f"{row['worse_pct']:.2f}%" if row.get("worse_pct")
                     is not None else _fmt(row.get("worse_abs")))
            lines.append(f"| guard: {row['label']} | {_fmt(row['baseline'])} "
                         f"| {_fmt(row['candidate'])} | {worse} "
                         f"| {row['status']} |")
        for row in cell["improvements"]:
            worse = (f"{row['worse_pct']:.2f}%" if row.get("worse_pct")
                     is not None else _fmt(row.get("worse_abs")))
            mark = "IMPROVED" if row.get("improved") else ""
            lines.append(f"| improvement: {row['label']} "
                         f"| {_fmt(row['baseline'])} | {_fmt(row['candidate'])} "
                         f"| {worse} | {mark} |")
        lines.append("")
    cov = g["coverage"]
    if cov["baseline_only"] or cov["candidate_only"]:
        lines.append("## Coverage differences")
        lines.append("")
        for dom, model in cov["baseline_only"]:
            lines.append(f"- baseline-only cell: {dom} [{model or '?'}]")
        for dom, model in cov["candidate_only"]:
            lines.append(f"- candidate-only cell: {dom} [{model or '?'}] "
                         "(new coverage — reported, never penalized)")
        lines.append("")
    if g["not_evaluable"]:
        lines.append("## Not evaluable (missing on one side; policy says skip)")
        lines.append("")
        for row in g["not_evaluable"]:
            lines.append(f"- {row['label']} on {row['domain']} "
                         f"[{row.get('model') or '?'}] (baseline "
                         f"{_fmt(row['baseline'])}, candidate "
                         f"{_fmt(row['candidate'])})")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="runcmp gate",
        description="Change gate: candidate harness vs baseline, "
                    "policy-judged (exit 0 = PASS)")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--referee", required=True, type=Path)
    ap.add_argument("--baseline", action="append", required=True,
                    metavar="KEY=VALUE[,KEY=VALUE]",
                    help="selector for baseline runs (era=/framework=/"
                         "model=/domain=/label=; * wildcards; repeat to OR)")
    ap.add_argument("--candidate", action="append", required=True,
                    metavar="KEY=VALUE[,KEY=VALUE]")
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--allow-model-mismatch", action="store_true",
                    help="pair cells per task even when models differ "
                         "(a model change is normally a different "
                         "experiment, not a harness change)")
    ap.add_argument("--out", required=True, type=Path,
                    help="directory for gate.json + gate.md")
    args = ap.parse_args(argv)

    records = load_registry(args.corpus)
    bench = json.loads(args.bench.read_text())
    referee = json.loads(args.referee.read_text())
    policy = json.loads(args.policy.read_text())
    result = evaluate(records, bench, referee, policy,
                      parse_selector(args.baseline),
                      parse_selector(args.candidate),
                      allow_model_mismatch=args.allow_model_mismatch)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate.json").write_text(json.dumps(result, indent=1) + "\n")
    md = render_md(result)
    (args.out / "gate.md").write_text(md)
    # The reviewers' mission: the auto-derived corpus mission with the
    # gate focus (explicit PASS/FAIL recommendation required), plus the
    # screen itself as evidence context.
    from alpha_lab.benchmarks.runcmp.mission import build_mission
    mission_text = build_mission(args.corpus, focus="gate") + (
        "\n\n## The change screen (deterministic, judge it — do not "
        "just repeat it)\n\n" + md)
    (args.out / "mission.md").write_text(mission_text)
    print(f"gate screen: {len(result['violations'])} flag(s), "
          f"{len(result['improvements_shown'])} improvement(s), "
          f"{len(result['cells'])} cell(s) compared; mechanical reading "
          f"{result['verdict']} -> {args.out}/gate.md (+ mission.md for "
          "LLM reviewers)")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
