"""Auto-generated investigation missions (`runcmp mission`).

A mission tells the investigator two things the system prompt cannot know:
what THIS investigation must decide, and what THIS corpus contains. Both
are derivable: the decisions come from a named focus preset, and the corpus
inventory comes from the same deterministic artifacts the investigator will
read (`corpus.json`, `referee.json`, the scorecard files). Hand-writing
that inventory is how stale corpus descriptions — and confidently wrong
reports — happen, so the investigator stages generate their mission by
default and hand-written missions are the exception (see
missions/TEMPLATE.md for when one is warranted).

Facts the registry cannot know (a config knob one wave carried, a serving
incident) enter through repeatable ``--note`` flags and are printed under
an "Operator notes" heading, marked as declared rather than derived.

The generated text is always written next to the investigation output as
``mission.md`` so a human can read exactly what scoped the report, and
rerun or edit it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Focus presets: the decision set an investigation must deliver. Static and
# harness-agnostic by design — corpus facts are injected separately, and the
# evidence rules live in the investigator system prompt, not here.
# ---------------------------------------------------------------------------

_FOCUS_HARNESS = """\
## 1. The decisions this report must deliver

1. **Harness** — which harness should a team use, per task and overall,
   and which of each losing harness's mechanisms are worth copying, fixing,
   or deleting. Use same-model pairs as replication: a difference that
   flips with the model is a model interaction, not a harness property,
   and must be labeled as such.
2. **Model** — which model should drive a whole run, and which model per
   agent seat. Behaviors observed in a run are properties of the model
   that produced them, not transplantable parts.
3. **Combination** — which harness-and-model pairing to actually run
   today, with the margin (decisive / narrow / coin-flip) and what would
   change the answer. Where the corpus spans several tasks, say whether
   one combination wins everywhere or the answer is task-dependent — and
   if it is task-dependent, name what drives the flip, with evidence.
4. **Capability** — what each model is genuinely good at and what each
   harness elicits, judged by reading the work products, not only the
   scores.

Close with a ranked **"what to change first"** list per harness, each item
tied to cited evidence.
"""

_FOCUS_VARIABILITY = """\
## 1. The decisions this report must deliver

This corpus contains deliberate repeats: the same harness+model+task cell
run more than once (the replication groups are enumerated below). The
report answers, for someone deciding how much to trust any single run:

1. **How large is run-to-run variation?** Per replication group: the
   spread in referee-scored quality (only within one validation identity —
   never across), in scored yield, in search shape (experiments proposed,
   champion changes, time to best), and in champion family (did repeats
   land on the same kind of solution?). Distinguish sampling noise from
   drift with a named cause (config delta, serving conditions, code
   version) — operator notes below declare the known deltas.
2. **Do the headline verdicts survive variability?** For every pair
   verdict the deterministic layer declares, state whether the margin
   exceeds the observed repeat spread for that cell. A verdict inside the
   spread is a coin-flip and must be reported as such.
3. **Which harness is more repeatable, and why?** Mechanisms with
   evidence: what in each harness amplifies or damps run-to-run variance
   (verification gates, queue policy, retry behavior, prompt generation)?
4. **What should change** to reduce harmful variance (or exploit useful
   exploration variance), ranked, per harness, each item cited.

A single-run cell cannot support variance claims; use it only as context
and say so where it appears.
"""

_FOCUS_GATE = """\
## 1. The decision this report must deliver

A harness change is proposed: candidate runs against baseline runs (the
change screen below names the sides, the paired cells, and what the
deterministic policy flagged). Your deliverable is an explicit, argued
**recommendation** on the change. The FIRST line of your report must be
exactly `Recommendation: PASS` or `Recommendation: FAIL`, followed by a
confidence grade (decisive / comfortable / narrow / coin-flip). Do not
hedge into "unresolved" — several reviewers run this same mission
precisely so that positions can be compared; a refusal to decide carries
no information.

1. **Did the change hurt anything that matters?** Judge every dimension
   the screen flagged AND what the screen cannot see (read the work
   products, the failure records, the trajectories). A screen flag is a
   mechanical reading, not a verdict: confirm it with mechanism evidence
   or overrule it with cause (a wall-clock regression traced to an
   external outage is weather, not the change) — either way, show the
   evidence.
2. **Did the change deliver the improvement it claims?** Quantify it,
   under the same-ledger rules; separate real gains from run-to-run
   noise wherever repeats exist.
3. **The recommendation, argued.** Tie it to the guard dimensions the
   policy names (score, time, cost, failures, code quality) plus anything
   material the policy missed; state exactly which single piece of
   evidence carries the most weight, and what result would flip your
   recommendation.
"""

_FOCUS_TREATMENT = """\
## 1. The decisions this report must deliver

This corpus contains a deliberate treatment: some runs' models receive
their own prior reasoning back on every request ("replayed"), others ran
blind to it (operator notes below name which eras/models are treated and
which model is the always-replayed drift control). The MODELS and the
treatment are the subject of this report; harness comparison is context
only.

1. **Per treated model, with vs without reasoning replay.** For every
   treated model, compare its replayed runs against the same cell's blind
   runs: referee quality sized against that cell's replication noise and
   against the untreated control's same-period drift; search shape
   (best-so-far trajectories, attempts to best, dead ends); reliability
   (failures, recoveries); latency and token cost per call and per scored
   experiment. State per model: better, worse, or not measurable — and
   what would settle it.
2. **Which model to use WITH replay available, and which WITHOUT.** Does
   the model ranking change between the two serving modes? A model that
   only wins when replayed is a different recommendation from one that
   wins regardless.
3. **How replay changes the research behavior, from the transcripts and
   work products.** Iteration depth, self-correction after failures,
   whether hypotheses build on earlier reasoning, debrief quality — quote
   the artifacts; do not infer behavior from scores alone.
4. **Confounds, weighed explicitly.** Serving incidents concentrated on
   treated runs, endpoint differences between eras, era drift measured on
   the control — say what is attributable to the treatment and what is
   not, and grade every conclusion accordingly.

Coverage mandate: every treated cell in the corpus — each treated model
on each task under each harness where both a replayed and a blind run
exist — gets examined by name; cells where the comparison is impossible
are named as such with the reason. The untreated control model is
examined on the same tasks for the same periods. No cell may appear only
"where interesting".

Dimension sweep (mandatory): for the treated models, walk every registry
metric family and every evidence-pack section (agent logs, request
composition, replay shares, tool failures, seats, run log) and state per
dimension whether replay changed it, with evidence, or that it shows no
material difference.

Close with, in order: a MANDATORY per-model x per-task verdict table
(with vs without replay: better / worse / not measurable, each graded
against the control); findings with the mechanism behind each;
corrections and retractions; verdict robustness (which single piece of
evidence carries the most weight, and what would flip each verdict);
what stays unresolved with the cheapest follow-up campaign that would
settle it; and a closing self-audit. Charts appear where sections need
them (per-cell before/after slopes, trajectories, latency and cost
distributions); census-like tables go to an appendix. Depth beats
brevity throughout.
"""

_FOCUS_UNION = """\
## 1. The decisions this report must deliver — the UNION of everything

This is the campaign's single all-encompassing report. Its section list
is the UNION of every section theme the per-battery reports have carried
(derived from their actual headers), at full depth. Nothing below may be
dropped or compressed into a table row; each numbered family gets its own
full section with evidence, and families 3-6 (harness, model,
combination, repeatability) are answered PER TASK as well as overall
(one subsection per task).

1. **Headline verdicts** — the answers in one page up front, each with
   its margin grade and what would change it.
2. **Corpus and coverage** — what the corpus contains and what is
   missing: cells never run, group validity, censoring (operator stops,
   non-finished runs), denominators. Discussion here; the run-by-run
   registry tables go to the appendix.
3. **Harness** — which harness per task and overall, calibrated against
   each task's replication noise; architectural verdicts: keep, copy,
   fix, delete — ranked, cited; where each harness is genuinely better
   and where it is broken; what each harness elicits from the same model;
   and a governance-machinery deep-dive: what the conductor, verifier,
   queue, throttles, routing, memory and handoff each contributed, and
   what the other harness does at the same decision points.
4. **Model** — which model for a whole run per task and overall, with a
   tier ranking; which model per agent seat; capability read from the
   work products (quote debriefs/playbooks/analysis, not only scores).
5. **Combination** — the harness+model pairing to run today per task and
   overall, ranked over every cell present, with margins, a decision
   matrix for what the reader is optimizing, and what would flip each
   answer.
6. **Repeatability** — run-to-run spread per replication group; do the
   pair verdicts survive repeat spread; search-shape and champion-family
   spread; which harness is more repeatable and by what mechanism.
7. **Treatment (reasoning replay)** — per treated model, with vs without
   replay against the untreated control's drift: quality, search shape,
   behavior read from transcripts, latency/cost; which model to use WITH
   replay and which WITHOUT; the reasoning round-trip census.
8. **Reliability** — external interference vs harness defects vs model
   defects, each named with log signatures and counts; what each cost;
   the shape of the runs (deaths, retries, relaunches, exits).
9. **Cost** — tokens/dollars/wall-clock per scored experiment, per seat,
   per task; where the money goes (token split by seat); cache semantics
   spelled out (reported vs engine-internal); the cost/quality trade-off
   charted, honestly (absent ledgers are absent, not zero).
10. **Search dynamics and lineage** — full best-so-far progressions per
    task (referee units where comparable; own-identity clearly labeled
    otherwise); experiment lineage (parent/variant chains, champion
    families — did searches build or thrash?); duration/session/queue
    distributions and yield-time trade-offs.
11. **Context engineering and tool calling** — payload composition, size
    and growth, repetition and re-reading, history replay shares,
    summarization/caching behavior, connected to outcomes; tool-usage and
    tool-failure census per seat and per model — every invoked tool, its
    failures, and a verdict — with the dominant failure classes and their
    mechanisms.
12. **Behavior provenance** — where each important behavior came from:
    prescribed by prompts/adapters vs model-emergent vs harness-forced;
    an adapter patch audit (what was patched mid-run, by whom, with what
    behavioral consequence).
13. **Code and written artifacts** — read the code, don't just count it:
    experiment code volume/quality per run and model; analysis scripts,
    debriefs, playbooks, verification notebooks and whether their
    reproductions matched the claimed numbers (who mandates them, who
    writes them, are they good — quote examples); defects the runs
    exposed in each HARNESS's own code (launchers, request/transcript
    assembly, metric extraction, path contracts) with log signatures;
    code-level keep/fix items per harness, ranked.
14. **Measurement hygiene and anomalies** — denominator, token, and
    tool-accounting traps found while investigating; every place the
    registry and your own recomputation disagree, both values on the
    record; a numerical-oddities register where every large unexplained
    gap is chased to its mechanism or explicitly left open.
15. **Campaign design integrity** — what in the campaign's own design
    limits its conclusions (eras as uncontrolled variable, missing
    repeats, identity discipline), and the cheapest redesign that fixes
    it.

Close with, in order: **Findings, with the mechanism behind each**
(numbered, plain language); ranked **"what to change first"** lists for
each harness plus infrastructure; **corrections and retractions**;
**verdict robustness** — stress tests: could another reviewer land
elsewhere, and which single piece of evidence carries the most weight per
verdict; **what stays unresolved and why**, each item with what would
settle it; and a **closing self-audit** of this report's own weakest
claims.

Structural requirements: registry/census tables to the APPENDIX, never
the body; every chart family the earlier reports used (progressions,
distributions, trade-off scatters, pair dumbbells, fault heatmaps,
slopes/spans) appears where its section needs it. Depth beats brevity
everywhere: if a theme was worth a page in a per-battery report, it is
worth at least that here.

Dimension sweep (mandatory): EVERY dimension must be examined across the
full MODEL x DOMAIN x HARNESS cube — every model on every task under
every harness where a run exists gets examined in the relevant sections,
and every absent cell is named as absent; no task, model, or harness may
appear only "where interesting". Additionally walk every dimension the
deterministic layer measures — the full registry metric list
(get_tables / bench.json) family by family and every section of the
evidence packs (events, agent logs, run log, seats, memory, inventory,
adapter drift); for each dimension either report what it shows with
evidence or state explicitly that it shows no material difference. No
dimension may be silently skipped; if the iteration budget forces
triage, list the dimensions left unexamined by name in the unresolved
section.
"""

FOCI: dict[str, str] = {
    "harness": _FOCUS_HARNESS,
    "variability": _FOCUS_VARIABILITY,
    "gate": _FOCUS_GATE,
    "treatment": _FOCUS_TREATMENT,
    "union": _FOCUS_UNION,
}


# ---------------------------------------------------------------------------
# Corpus inventory
# ---------------------------------------------------------------------------

def _lineup_index() -> dict[str, dict]:
    path = Path(__file__).with_name("lineup.json")
    try:
        data = json.loads(path.read_text())
    except OSError:
        return {}
    return {d["id"]: d for d in data.get("domains", []) if isinstance(d, dict)}


def _fmt_metric(entry: dict) -> str:
    m = entry.get("metric") or {}
    name = m.get("name", "?")
    direction = "lower is better" if m.get("lower_is_better") else "higher is better"
    return f"{name} ({direction})"


def _corpus_section(corpus: dict, referee: dict | None) -> str:
    runs = corpus.get("runs", [])
    lineup = _lineup_index()
    ref_runs: dict[str, dict] = {}
    ref_pairs: list[dict] = []
    if referee:
        rr = referee.get("runs")
        if isinstance(rr, dict):
            ref_runs = rr
        elif isinstance(rr, list):
            ref_runs = {r.get("label"): r for r in rr if isinstance(r, dict)}
        ref_pairs = referee.get("pairs", []) or []

    harnesses = sorted({r.get("framework", "?") for r in runs})
    models = sorted({str(r.get("model", "?")) for r in runs})
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_domain[r.get("domain", "?")].append(r)

    out: list[str] = []
    out.append("## 2. The corpus, exactly (generated from the registry)")
    out.append("")
    out.append(
        f"{len(runs)} runs; harnesses: {', '.join(harnesses)}; models: "
        f"{', '.join(models)}; tasks: {len(by_domain)}. The registry "
        "(list_runs) is authoritative; this section summarizes it."
    )

    for dom in sorted(by_domain):
        drs = by_domain[dom]
        lu = lineup.get(dom, {})
        title = lu.get("title", dom)
        out.append("")
        out.append(f"### Task `{dom}` — {title}")
        if lu:
            out.append(f"Quality metric: {_fmt_metric(lu)}.")
        # per-run inventory
        out.append("")
        out.append("| run label | harness | model | state | terminal rows |")
        out.append("|---|---|---|---|---|")
        for r in sorted(drs, key=lambda x: x.get("label", "")):
            out.append(
                f"| `{r.get('label')}` | {r.get('framework')} "
                f"| {r.get('model', '?')} | {r.get('run_state', '?')} "
                f"| {r.get('db_terminal_rows', r.get('db_rows', '?'))} |"
            )
        not_finished = [r for r in drs if r.get("run_state") != "finished"]
        if not_finished:
            out.append("")
            out.append(
                "Runs NOT in a finished state (report them as present, draw "
                "verdicts only from finished runs): "
                + ", ".join(f"`{r.get('label')}` ({r.get('run_state')})"
                            for r in not_finished)
            )
        # replication groups
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in drs:
            groups[(r.get("framework"), str(r.get("model")))].append(r)
        reps = {k: v for k, v in groups.items() if len(v) > 1}
        if reps:
            out.append("")
            out.append(
                "Replication groups (same harness + model + task, run more "
                "than once — the run-to-run variance evidence):"
            )
            for (fw, model), members in sorted(reps.items()):
                members = sorted(members, key=lambda x: (x.get("era", ""),
                                                         x.get("label", "")))
                out.append(
                    f"- {fw} + {model}: {len(members)} runs — "
                    + ", ".join(f"`{m.get('label')}`" for m in members)
                )
        # rankability, quoted from the referee (never re-derived here)
        if ref_runs:
            dom_refs = [ref_runs[l] for l in ref_runs
                        if isinstance(ref_runs[l], dict)
                        and ref_runs[l].get("domain") == dom]
            if dom_refs:
                kinds = sorted({d.get("referee_kind", "?") for d in dom_refs})
                unrankable = [d for d in dom_refs if not d.get("rankable", True)]
                out.append("")
                out.append(
                    f"Referee kind: {', '.join(kinds)}. The referee marks "
                    f"{len(dom_refs) - len(unrankable)} of {len(dom_refs)} "
                    "scored runs rankable (its flags are authoritative; "
                    "the per-domain leaderboard in bench.md applies them)."
                )
                for d in unrankable:
                    reason = d.get("rankable_reason")
                    out.append(
                        f"  - `{d.get('label')}` is verified but UNRANKED"
                        + (f": {reason}" if reason else
                           " (see referee.json for the recorded reason).")
                    )
        # pair verdict caveats: non-comparable pair records carry no run
        # labels, so attribute them by matching member labels where present
        # and fall back to the single-domain case.
        def _pair_in_dom(p: dict) -> bool:
            best = p.get("best") or {}
            sides = [str((best.get(s) or {}).get("run", ""))
                     for s in ("left", "right")]
            if any(sides):
                return any(f"/{dom}/" in s or s.startswith(f"{dom}/")
                           or f"{dom}" in s for s in sides if s)
            return len(by_domain) == 1
        dom_bad = [p for p in ref_pairs
                   if isinstance(p, dict) and not p.get("comparable", True)
                   and _pair_in_dom(p)]
        if dom_bad:
            reason = next((p.get("reason") for p in dom_bad
                           if p.get("reason")), None)
            out.append("")
            out.append(
                f"{len(dom_bad)} declared pair(s) on this task are "
                "NON-COMPARABLE per the referee"
                + (f', recorded reason: "{reason}"' if reason else "")
                + " — never present the two sides' raw scores as a decided "
                "ranking; use per-run corroboration plus the lower "
                "evidence tiers, and say so."
            )
    return "\n".join(out)


def _scorecards_section(campaign_dir: Path) -> str:
    known = [
        ("tables.md", "deterministic pair tables + corpus aggregates"),
        ("bench.md", "metric registry scorecard; opens with the per-task "
                     "referee leaderboards"),
        ("referee.json", "independent re-scoring (the only cross-harness "
                         "quality evidence)"),
        ("token_accounting.md", "normalized tokens/cost per run (absent "
                                "ledgers stay absent — never a zero)"),
    ]
    lines = ["## 3. Scorecards available", ""]
    found = False
    for name, what in known:
        if (campaign_dir / name).exists():
            lines.append(f"- `{name}` — {what}")
            found = True
    if not found:
        lines.append(
            "- none found next to corpus.json yet — run the deterministic "
            "chain first; investigations without scorecards lose their "
            "tier-1/tier-2 anchors."
        )
    return "\n".join(lines)


_STANDING_CHECKS = """\
## Standing sanity checks (every focus)

- **Early phases must earn their hours.** Every pack carries the
  exploration ledger (``inventory.exploration``: learnings/plan sizes and
  structure, phase-1 product bytes; ``agent_logs.web_searches``: every
  web-search query with its seat). Judge whether phase 0/1 work connected
  to outcomes — searched facts that reached the framework or experiments,
  learnings that later experiments used — or was dead weight; grade it,
  with citations, instead of ignoring it.
- **Written products must find readers.** Packs carry a report census
  (``inventory.reports``: bytes/tables/numbers/images for the report
  documents and the debriefs) and a readership ledger
  (``agent_logs.artifact_reads``: which seats opened debriefs, learnings,
  plans, reports). Judge report quality against those facts — and flag
  written products nobody ever read: a debrief no seat opens is process
  cost, not knowledge transfer.
- **Trace round-trip.** Every pack carries
  `agent_logs.reasoning_roundtrip` (requests/responses totals and how many
  of each carried a reasoning trace). A run whose responses produce
  reasoning while its requests never return it is running blind to its own
  past thinking — a harness or serving defect, not a model property. Flag
  it as a finding; do not explain it away as instrumentation. (This exact
  defect hid in every GLM and deepseek run before 2026-08-07 behind an
  extractor gap that reported zeros for all lab models.)
"""


def build_mission(corpus_path: Path, focus: str = "harness",
                  notes: list[str] | None = None) -> str:
    """Compose a mission from the deterministic artifacts.

    ``corpus_path`` is the campaign's corpus.json; referee.json and the
    scorecards are discovered next to it.
    """
    if focus not in FOCI:
        raise ValueError(
            f"unknown focus '{focus}' — available: {', '.join(sorted(FOCI))}")
    corpus = json.loads(Path(corpus_path).read_text())
    campaign_dir = Path(corpus_path).parent
    referee = None
    ref_path = campaign_dir / "referee.json"
    if ref_path.exists():
        referee = json.loads(ref_path.read_text())

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    head = (
        f"# Investigation mission — focus: {focus} (auto-generated)\n\n"
        f"Generated by `runcmp mission` on {stamp} from "
        f"`{Path(corpus_path).name}`"
        + (", `referee.json`" if referee else "")
        + " and the scorecard files. Regenerate rather than edit; operator "
          "facts belong in `--note` flags.\n"
    )
    parts = [head, FOCI[focus], _corpus_section(corpus, referee),
             _scorecards_section(campaign_dir), _STANDING_CHECKS]
    if notes:
        parts.append(
            "## 4. Operator notes (declared, not derived — verify against "
            "run artifacts where possible)\n\n"
            + "\n".join(f"- {n}" for n in notes)
        )
    return "\n\n".join(parts) + "\n"


def expand_notes(notes: list[str] | None) -> list[str]:
    """Expand ``@path`` notes to file contents (one note per non-blank line).

    Shell-quoting operator notes through nested `bash -c` strings silently
    killed reviewer launches TWICE on 2026-08-07/08 (an apostrophe inside a
    single-quoted --note terminated the quote and the unit died at spawn
    with no log). A file sidesteps quoting entirely.
    """
    out: list[str] = []
    for n in notes or []:
        if n.startswith("@"):
            for line in Path(n[1:]).read_text().splitlines():
                if line.strip():
                    out.append(line.strip())
        else:
            out.append(n)
    return out


def resolve_mission_arg(mission_arg: str, corpus_path: Path,
                        out_dir: Path, notes: list[str] | None = None) -> str:
    """Shared by the investigator stages.

    ``auto`` / ``auto:<focus>`` -> generated (and written to
    ``<out>/mission.md``); ``@path`` -> file contents; ``none`` -> empty;
    anything else -> literal mission text. Whatever text is used is written
    to ``<out>/mission.md`` so the scope of every report is auditable.
    """
    text: str
    if mission_arg in ("", "auto") or mission_arg.startswith("auto:"):
        focus = mission_arg.partition(":")[2] or "harness"
        text = build_mission(corpus_path, focus=focus, notes=notes)
    elif mission_arg == "none":
        text = ""
    elif mission_arg.startswith("@"):
        text = Path(mission_arg[1:]).read_text()
    else:
        text = mission_arg
    if text:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "mission.md").write_text(text)
    return text


def check_mission(path: Path, corpus_path: Path) -> int:
    """Lint a hand-written mission before it commissions a review.

    The worst review batch on record passed every mechanical gate and was
    still unusable because its hand-written mission never spelled the
    sections out. This checks the two things a mission must get right to be
    enforceable: numbered ``N. **Theme**`` sections (the stub check and the
    report critic key off them) and run labels that actually exist in the
    corpus it will run against (splicing an old mission's corpus
    description forward is the classic way to break this).
    """
    import re

    from runcmp.corpus import load_registry
    text = path.read_text(errors="replace")
    themes = re.findall(r"^\s*\d+\.\s+\*\*(.+?)\*\*", text, re.MULTILINE)
    rc = 0
    if themes:
        print(f"themes: {len(themes)} numbered section themes parsed")
        for t in themes:
            print(f"  - {t}")
    else:
        print("FAIL: no `N. **Theme**` numbered sections found — the "
              "deterministic stub check and the report critic's section "
              "audit both key off that exact form; a mission without it "
              "cannot be enforced")
        rc = 1
    labels = {r.label for r in load_registry(corpus_path)}
    mentioned = set(re.findall(
        r"\b[\w.-]+/[\w.-]+/[\w.-]+/[\w.-]+\b", text))
    unknown = sorted(m for m in mentioned
                     if m not in labels and not m.startswith(("http", "/")))
    if unknown:
        print(f"WARNING: {len(unknown)} label-like reference(s) not in this "
              "corpus (a spliced-forward corpus description?): "
              + ", ".join(unknown[:6]))
    known = sorted(m for m in mentioned if m in labels)
    if known:
        print(f"labels verified in corpus: {len(known)}")
    print("mission check: " + ("OK" if rc == 0 else "FAIL"))
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="runcmp mission",
        description="Generate an investigation mission from the "
                    "deterministic artifacts (preview / edit / reuse).")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--focus", default="harness",
                    help=f"decision preset: {', '.join(sorted(FOCI))}")
    ap.add_argument("--note", action="append", default=[],
                    help="operator-declared fact the registry cannot know "
                         "(repeatable)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the mission here (default: print to stdout)")
    ap.add_argument("--check", type=Path, default=None,
                    help="lint a hand-written mission file against this "
                         "corpus instead of generating one")
    args = ap.parse_args(argv)
    if args.check is not None:
        return check_mission(args.check, args.corpus)
    # @file notes expand here exactly as in the investigator stages — the
    # CLI shipped without this and a hand-assembled mission carried a
    # literal "@path" string where the operator facts belonged (found by a
    # cold-start user, 2026-08-11)
    text = build_mission(args.corpus, focus=args.focus,
                         notes=expand_notes(args.note))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"mission ({args.focus}) -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())


def refresh_reports_index(campaign_dir: Path) -> Path | None:
    """Write ``REPORTS.md`` at the campaign's top level: one line per
    investigation report found in the campaign's subdirectories, so a human
    browsing the directory finds the written reports without knowing the
    layout convention. Idempotent; called by the investigator stages after
    a report lands (and safe to call any time).
    """
    campaign_dir = Path(campaign_dir)
    entries = []
    for rp in sorted(campaign_dir.glob("*/REPORT.md")):
        title = ""
        try:
            for line in rp.read_text().splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
        except OSError:
            continue
        verified = ""
        vpath = rp.parent / "verification.json"
        if vpath.exists():
            try:
                v = json.loads(vpath.read_text())
                t = v.get("total")
                ok = v.get("verified")
                if t is not None:
                    verified = f" — findings verified {ok}/{t}"
            except (OSError, json.JSONDecodeError):
                pass
        entries.append(
            f"- [{rp.parent.name}]({rp.parent.name}/REPORT.md) — "
            f"{title or 'report'}{verified} "
            f"(+ [HTML]({rp.parent.name}/REPORT.html), findings.md, "
            f"mission.md)")
    if not entries:
        return None
    out = campaign_dir / "REPORTS.md"
    out.write_text(
        "# Investigation reports in this campaign\n\n"
        "Deterministic scorecards live at this level (`bench.md`, "
        "`tables.md`, `token_accounting.md`, `referee.json`); each written "
        "investigation lives in its own subdirectory:\n\n"
        + "\n".join(entries) + "\n")
    return out
