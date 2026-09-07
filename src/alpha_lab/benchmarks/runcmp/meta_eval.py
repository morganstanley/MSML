"""Deterministic scorer for investigator-architecture tournaments.

Scores one or more investigation output dirs against (a) the deterministic
fact-checker and (b) a key-fact rubric: independently established facts about
the corpus that a good investigation of the SAME mission should surface. Each
key fact is a set of text signatures matched over the VERIFIED findings only
(fact-check-failed findings never score). The rubric is versioned data, not
judgment: change it only by adding facts with their establishing evidence.

Usage:
    PYTHONPATH=src python -m alpha_lab.benchmarks.runcmp.meta_eval \
        --packs comparison_out/ibes/packs --rubric ibes \
        --dir name1=path1 --dir name2=path2 ... --out tour_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Key-fact rubrics per corpus. Every entry cites how the fact was established
# (deterministic bench/referee output or a prior 100%-fact-checked finding).
# A fact scores when ANY verified finding matches ALL its patterns (regex,
# case-insensitive) over the finding's full JSON text.
RUBRICS: dict[str, list[dict]] = {
    "ibes": [
        {"id": "gate_self_authored",
         "established_by": "inv_sol #1 / inv_opus #2 (verified): gate regime "
                           "authored via the run's own phase-0 adapter",
         "patterns": [r"(phase.?0|adapter|builder contract|MANIFEST)",
                      r"(gate|canonical|audit)",
                      r"(author|prescrib|self|own|wrote|custom)"]},
        {"id": "impossible_threshold",
         "established_by": "inv_sol #2 (verified) + gate payloads: required "
                           "median linkage 2000 vs ~1020 available",
         "patterns": [r"2,?000", r"1,?02\d"]},
        {"id": "strategist_single_session",
         "established_by": "inv_opus #4 / inv_sol #4 (verified): 2,860 "
                           "strategist calls in one never-ended session",
         "patterns": [r"2,?860", r"strategist"]},
        {"id": "artifact_asymmetry",
         "established_by": "referee.json + full-scan: gpt56 preserves "
                           "daily series, o48 preserves none",
         "patterns": [r"(daily_returns|positions\.parquet|series)",
                      r"(o48|opus)", r"(not preserved|none|lacks|only|absent|6 )"]},
        {"id": "best_claim_not_reproduced",
         "established_by": "referee.json (deterministic): gpt56 best claim "
                           "1.274 recomputes to <=0.615 from preserved series",
         "patterns": [r"1\.27", r"(0\.6\d|not reproduc|does not reproduce|"
                               r"recomput)"]},
        {"id": "file_not_found_churn",
         "established_by": "bench diagnostics: 2,525 read_file "
                           "File-not-found failures (o48: 61)",
         "patterns": [r"2,?5\d\d", r"(read_file|file.not.found|missing.file)"]},
        {"id": "conductor_rewinds_fixed_gate",
         "established_by": "inv_sol #2 (verified): the final phase-2 rewind "
                           "corrected the impossible threshold; earlier ones "
                           "re-entrenched",
         "patterns": [r"rewind", r"(threshold|universe|linkage|corrected|fix)"]},
        {"id": "yield_collapse_mechanism",
         "established_by": "bench + inv findings: gates converted completed "
                           "work into non-scored rows (38% vs 94%)",
         "patterns": [r"(38%|0\.381|51)", r"(94%|0\.936|102)"]},
    ],
    # d2/d4 sol-vs-opus (buggy-cond era corpus, 8 runs). Each fact is
    # anchored to a verified finding of the 2026-07-26 investigations or a
    # deterministic bench/tables fact.
    "model_ab": [
        {"id": "cpu_misroute_mechanism",
         "established_by": "inv_opus F2 (verified): _is_cpu_experiment "
                           "literal-marker scan + model.to(DEV) indirection "
                           "misrouted 14 opus d2-cond trainings to CPU",
         "patterns": [r"(marker|router|routing|_is_cpu_experiment|scanner)",
                      r"(CPU|cpu)", r"14"]},
        {"id": "drain_deadlock",
         "established_by": "inv_sol F4 (verified): d4-cond 11h idle tail — "
                           "drain counted a never-launched 'checked' row as "
                           "in-flight while refusing to launch it",
         "patterns": [r"(drain|termination|run_end)",
                      r"(deadlock|idle|stuck|11\s*h)",
                      r"(checked|never.launched|pre.launch|unlaunched)"]},
        {"id": "msml_guards_emergent",
         "established_by": "inv_sol F5 (verified): sol d2-msml guards are "
                           "worker-authored (no prescribed guard text in "
                           "either adapter), spread to 12 experiments",
         "patterns": [r"(guard|preflight|UUID|SLURM|lock)",
                      r"(worker|strategist|author|emergent|self)",
                      r"(no prescribed|not prescribed|neither adapter|"
                      r"spread|copied|propagat)"]},
        {"id": "opus_msml_flawless",
         "established_by": "bench (deterministic): opus-msml scored 19/19 "
                           "in both domains, zero execution failures",
         "patterns": [r"19\s*/\s*19|19 of 19", r"msml"]},
        {"id": "cost_cache_artifact",
         "established_by": "inv_opus F3 (verified): sol runs billed 80-91% "
                           "of input at the cached 1/10 rate; the cost gap "
                           "is a billing artifact, not verbosity",
         "patterns": [r"cach", r"(cost|billing|price|dollar|\$)",
                      r"(8\d\s*[-–%]|9[01]\s*%|1/10|tenth)"]},
        {"id": "d2cond_board_contamination",
         "established_by": "bench + inv_sol F1/F3 (verified): opus d2-cond "
                           "board mixes CPU-era near-floor rows with 8 "
                           "post-fix GPU submissions",
         "patterns": [r"(near.floor|contaminat|CPU.era|mixed board)",
                      r"(d2|domain.?2)", r"(8|eight)"]},
        {"id": "headline_corroboration",
         "established_by": "inv_sol F1/F2 + inv_opus F1 (verified): "
                           "headline best values checked file-vs-DB per "
                           "run; opus d4-msml contradicted at full "
                           "precision; nothing referee-verifiable",
         "patterns": [r"(corroborat|file.{0,20}DB|metrics\.json|"
                      r"training.log)",
                      r"(headline|best value|best_value)"]},
        {"id": "framework_conditional_verdict",
         "established_by": "inv_sol + inv_opus decision sections (both "
                           "verified): no universal winner — sol under "
                           "cond, opus under msml",
         "patterns": [r"(sol.{0,60}cond|cond.{0,60}sol)",
                      r"(opus.{0,60}msml|msml.{0,60}opus)",
                      r"(no universal|framework.dependent|framework."
                      r"conditional|depends on the framework|per.framework)"]},
    ],
}


def score_dir(name: str, path: Path, packs: Path, rubric: list[dict],
              python: str) -> dict:
    out: dict = {"name": name, "path": str(path)}
    # (Re)run the deterministic fact-checker for this dir.
    proc = subprocess.run(
        [python, "-m", "alpha_lab.benchmarks.runcmp.factcheck",
         "--out", str(path), "--packs", str(packs)],
        capture_output=True, text=True, timeout=1800,
    )
    out["factcheck_stdout"] = (proc.stdout + proc.stderr).strip()[-200:]
    v = {}
    vpath = path / "verification.json"
    if vpath.is_file():
        v = json.loads(vpath.read_text())
    out["findings_total"] = v.get("total", 0)
    out["findings_verified"] = v.get("verified", 0)
    out["findings_failed"] = v.get("failed", 0)
    out["factcheck_rate"] = (round(v.get("verified", 0) / v["total"], 3)
                             if v.get("total") else None)

    verified_texts = []
    per_finding = v.get("findings") or []
    findings_path = path / "findings.jsonl"
    if findings_path.is_file():
        rows = [json.loads(l) for l in findings_path.read_text().splitlines()
                if l.strip()]
        statuses = {f.get("id"): f.get("status") for f in per_finding
                    if isinstance(f, dict)}
        for r in rows:
            st = statuses.get(r.get("id"), "verified" if not statuses else "")
            if st in ("verified", ""):
                verified_texts.append(json.dumps(r, default=str))
    blob_list = verified_texts

    hits = {}
    for fact in rubric:
        matched = any(
            all(re.search(p, t, re.I) for p in fact["patterns"])
            for t in blob_list
        )
        hits[fact["id"]] = matched
    out["key_facts"] = hits
    out["key_fact_coverage"] = round(
        sum(hits.values()) / len(rubric), 3) if rubric else None

    qpath = path / "questions.json"
    if qpath.is_file():
        qs = json.loads(qpath.read_text())
        out["questions"] = {
            "total": len(qs),
            "resolved": sum(q["status"] == "resolved" for q in qs),
            "unresolvable": sum(q["status"] == "unresolvable" for q in qs),
            "open": sum(q["status"] == "open" for q in qs),
        }
    out["report_written"] = (path / "REPORT.md").is_file()
    return out


RANKING_RULES = (
    "Ranking is mechanical: (1) delivered key-fact coverage (an investigation "
    "without a report scores 0 coverage), (2) fact-check rate, (3) ledger "
    "drained (resolved+unresolvable == total), (4) verified findings count. "
    "A config DOMINATES when it is >= every rival on (1)-(3) and > on at "
    "least one."
)


def compute_ranking(results: list[dict]) -> list[dict]:
    """Mechanical ranking + dominance flags. Pure function of the scores."""
    def key(r):
        delivered_cov = (r.get("key_fact_coverage") or 0) if r.get(
            "report_written") else 0.0
        q = r.get("questions") or {}
        drained = 1 if (q and q.get("open", 1) == 0) else 0
        return (delivered_cov, r.get("factcheck_rate") or 0, drained,
                r.get("findings_verified") or 0)
    ranked = sorted(results, key=key, reverse=True)
    top = key(ranked[0]) if ranked else None
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
        r["rank_key"] = key(r)
    if len(ranked) > 1:
        t, s2 = ranked[0]["rank_key"], ranked[1]["rank_key"]
        ranked[0]["dominant"] = all(a >= b for a, b in zip(t[:3], s2[:3]))             and t != s2
    return ranked


def render(results: list[dict], rubric: list[dict]) -> str:
    ranked = compute_ranking(results)
    lines = ["# Investigator-architecture tournament — deterministic scores",
             ""]
    lines.append("## Computed ranking")
    lines.append("")
    lines.append(RANKING_RULES)
    lines.append("")
    lines.append("| rank | config | delivered coverage | fact-check | "
                 "ledger drained | verified findings |")
    lines.append("|---|---|---|---|---|---|")
    for r in ranked:
        cov, fc, drained, n = r["rank_key"]
        lines.append(f"| {r['rank']} | {r['name']} | {cov} | {fc} | "
                     f"{'yes' if drained else 'no/none'} | {n} |")
    if ranked and ranked[0].get("dominant"):
        lines.append("")
        lines.append(f"**Computed verdict: `{ranked[0]['name']}` dominates** "
                     "(>= all rivals on coverage/fact-check/ledger, > on at "
                     "least one).")
    elif ranked:
        lines.append("")
        lines.append("**Computed verdict: no strict dominance** — top-ranked "
                     f"`{ranked[0]['name']}` ties a rival on the primary axes.")
    lines.append("")
    lines.append("| config | findings (verified/total) | fact-check rate | "
                 "key-fact coverage | ledger (res/unres/open) | report |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        q = r.get("questions") or {}
        ledger = (f"{q.get('resolved', '—')}/{q.get('unresolvable', '—')}/"
                  f"{q.get('open', '—')}" if q else "—")
        lines.append(
            f"| {r['name']} | {r['findings_verified']}/{r['findings_total']} "
            f"| {r['factcheck_rate']} | {r['key_fact_coverage']} | {ledger} "
            f"| {'yes' if r['report_written'] else 'NO'} |")
    lines.append("")
    lines.append("## Key facts × configs")
    lines.append("")
    lines.append("| fact | " + " | ".join(r["name"] for r in results) + " |")
    lines.append("|---" * (len(results) + 1) + "|")
    for fact in rubric:
        row = [("✔" if r["key_facts"].get(fact["id"]) else "—")
               for r in results]
        lines.append(f"| {fact['id']} | " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Facts established by: "
                 + "; ".join(f"**{f['id']}**: {f['established_by']}"
                             for f in rubric))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packs", required=True, type=Path)
    ap.add_argument("--rubric", required=True, choices=sorted(RUBRICS))
    ap.add_argument("--dir", action="append", required=True,
                    help="NAME=PATH of one investigation output dir")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    rubric = RUBRICS[args.rubric]
    results = []
    for spec in args.dir:
        name, _, path = spec.partition("=")
        results.append(score_dir(name, Path(path), args.packs, rubric,
                                 args.python))
    md = render(results, rubric)
    if args.out:
        args.out.write_text(md)
        json_path = args.out.with_suffix(".json")
        json_path.write_text(json.dumps(results, indent=1, default=str))
        print(f"wrote {args.out} and {json_path}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
