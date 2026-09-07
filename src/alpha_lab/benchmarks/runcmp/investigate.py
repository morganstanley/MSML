"""Layer 3 — agentic investigator over the extracted corpus.

A single LLM agent (one writer) drives read-only tools over the evidence
packs, raw logs, and experiment databases, records findings with
machine-checkable evidence references, and writes REPORT.md. Deterministic
code sits below it (extraction) and above it (fact-check); the middle is
judgment, on purpose. The prompt states the mission and the evidence rules —
it does not impose a rubric or fixed axes.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from alpha_lab.benchmarks.runcmp.corpus import load_registry

MAX_ITERATIONS = 150
# Per-run head-room, mirroring the team variant. The flat 150 was calibrated
# to a 2-run corpus. Measured failure (2026-07-31, 8-run corpus): the solo
# investigator spent 115 of ~150 iterations on probes, recorded 13 findings,
# and hit the cap with no report written — while the same configuration had
# finished a smaller corpus at iteration 104. The team variants already scale
# with corpus size; the solo path did not, and was the only one that failed.
ITER_PER_EXTRA_RUN = 12
# Extra iterations per mission decision family beyond the historical 4
# (mirrors the team reviewer's ITER_PER_EXTRA_QUESTION).
ITER_PER_EXTRA_FAMILY = 8
# Stop investigating and start composing when this many iterations remain, so
# the report is always written rather than lost at the cap.
# 20 sufficed for prose-and-tables reports; the chart/distribution
# requirements (12 forms, raw-sample distributions, marginals) put real work
# into composition, and tonight's rounds finished at 150-204 of 318 anyway —
# budgets are not the binding constraint, compose head-room is.
COMPOSE_RESERVE = 32
# 20k showed an investigator 6% of a 75-run corpus's tables per call
# (2026-08-09); 48k keeps iterations meaningful on wide corpora while
# staying far under the model context.
MAX_TOOL_OUTPUT = 48_000
HISTORY_CHAR_BUDGET = 700_000
KEEP_RECENT_ITEMS = 30



def _chart_census(md: str) -> dict:
    """Count chart blocks, forms, and distinct distribution forms in a
    report draft — the same arithmetic factcheck.py prints post-hoc, applied
    at write time so a floor failure is caught while it can still be fixed."""
    blocks = re.findall(r"```chart[ \t]*\n(.*?)```", md, re.DOTALL)
    kinds: dict[str, int] = {}
    for b in blocks:
        m = re.search(r"^\s*type:\s*(\w+)", b, re.MULTILINE)
        k = (m.group(1).lower() if m else "bars")
        kinds[k] = kinds.get(k, 0) + 1
    dist = sum(1 for k in ("hist", "density", "box") if kinds.get(k))
    if any(re.search(r"^\s*band ", b, re.MULTILINE) for b in blocks):
        dist += 1
    if any(re.search(r"^\s*type:\s*scatter", b, re.MULTILINE)
           and re.search(r"^\s*marginals:\s*true", b, re.MULTILINE)
           for b in blocks):
        dist += 1
    return {"total": len(blocks), "kinds": kinds, "dist_forms": dist}



# The exact chart-block grammar the renderer accepts. This is the single
# source of truth for the writer: it is appended to the system prompt, and
# the write gate refuses any block the renderer cannot parse. The grammar
# was previously specified only in hand-written mission files; when a
# generated mission omitted it, the writer improvised a YAML dialect and
# every chart in two reports shipped as raw fence text (2026-08-07).
CHART_SYNTAX = """\
## Chart blocks — EXACT grammar (the renderer accepts nothing else)

Charts are fenced ```chart blocks. Every block: a `type:` line (one of the
12 below, exact spelling) and a `title:` line (say which direction is
better). Data rows are PIPE-SEPARATED lines — never YAML lists, never
JSON arrays, never `values: [...]`.

- bars — ranked list: one row per bar, `label | value [| note]`
- line — series over x: `x:` and `y:` axis-label lines, then one
  `series NAME: x1,y1; x2,y2; ...` line per series. Options:
  `style: step`, `fill: true`; quantile band: `band NAME: x,lo,mid,hi; ...`
- scatter — `x:`/`y:` labels, rows `point LABEL | x | y [| group]`;
  add `marginals: true` at 10+ points
- stacked — composition: rows `row LABEL | name=value | name=value ...`
- grouped — side-by-side bars: rows `row LABEL | name=v | name=v ...`
- dumbbell — two dots per label: rows `row LABEL | name=v | name=v`
- slope — two-state change: `x: before -> after`, rows `slope LABEL | v1 | v2`
- heatmap — `cols: A | B | ...`, rows `row LABEL | v | v | ...`
- spans — time windows: rows `span LABEL | start | end [| group]`.
  start/end are NUMERIC ONLY (e.g. hours since run start: 0, 5.2, 18.8);
  clock strings like `17:31:41` do not parse, and clock time cannot
  represent a window that crosses midnight anyway
- Every chart row must be a REAL data point. Never add commentary,
  placeholder, or "note" rows inside a chart block (a leaked
  "duplicate label guard — see row above" row shipped as a fake bar,
  2026-08-07); annotations belong in the caption text under the chart
- box — five-number rows: `box LABEL | min | q1 | median | q3 | max`
- hist — raw values, renderer bins: `series NAME: v1, v2, v3, ...`
- density — ridgeline strips, same raw-value series lines as hist

A submitted draft is REFUSED if any chart block fails to parse under this
grammar, with the failing block named. Never draw ASCII art.
"""

# The investigator's standing orders. Harness-agnostic on purpose: the
# corpus may hold two in-house frameworks, or many, or external harnesses
# that submitted evidence-contract runs — the actual roster is injected at
# run time from the corpus registry (see _corpus_roster), never hardcoded.
SYSTEM_PROMPT = """\
You are the post-hoc investigator for a comparison of multi-agent research
harnesses that ran the same benchmark tasks. The corpus registry (list_runs)
is the authoritative roster of harnesses, models, and tasks — start there;
do not assume any particular harnesses exist.

Everything the harnesses did is preserved in post-run artifacts, already
indexed for you. Gather CHART-GRADE data as you investigate, not at the end:
your report is required to present distributions (packs carry raw sample
arrays under ``agent_logs.samples``; experiments carry per-row ``duration``),
per-run best-so-far progressions over experiment order or wall-clock, and
two-quantity trade-off points. When a probe surfaces a quantity you will
chart, print the full series/array in that probe's output so it is quotable
at compose time. Your job is the question the run logs exist to answer,
expressed STRICTLY in the mission's stated decision space:

- **Framework comparison** (different frameworks, same model): which
  architectural choices are good, which are bad, and which cannot be resolved
  from this corpus — so the next iteration of each framework knows what to
  keep, copy, or delete.
- **Model comparison** (same framework, different model): the ONLY available
  decisions are which model to use for the whole run, or which model per
  agent seat (strategist, workers, builder/critic, conductor, ...).
  Behaviors observed in a run — artifact habits, self-authored gates,
  launchers, search styles — are emergent properties of the model that
  produced them, NOT components that can be transplanted between runs.
  Never recommend keeping/copying/mixing pieces of one model's run into
  another's; express every verdict as a model-choice (overall or per seat)
  with its evidence.

## Evidence rules (non-negotiable)

1. Every number you state must come from a tool result in this session.
2. Any claim about a *population* (all experiments of a run, all runs of a
   framework) must come from `run_python` code that processed every item, or
   from the evidence packs / tables which already did. Never generalize from
   one or two examples you happened to read.
3. Causal claims need mechanism evidence (the failing call, the directive that
   changed behavior, the config delta) — otherwise say "unresolved" and state
   what evidence would settle it. Correlation across 5 pairs is a hint, not a
   conclusion.
3b. "Unresolved" is a LAST RESORT, not an exit. Before writing it you must
   (a) directly examine the seat's/dimension's own outputs in every relevant
   run — a seat can be judged on the quality of what it produced even when
   run outcomes are confounded; outcome-confounding alone is NEVER
   sufficient grounds; (b) attempt a graded verdict ("sol, moderate
   confidence" / "leans opus, weak") — a calibrated lean with its evidence
   is worth more than a refusal; (c) if still unresolved, the row must list
   which artifacts you read and why direct quality judgment is impossible
   (e.g. the seat produced no artifacts in one run). A report where more
   than 1-2 rows are unresolved means you stopped digging too early.
4. Best-metric values across frameworks are NOT directly comparable unless the
   two sides share a validation identity (the tables mark this per pair).
   In-run process measures (reliability, efficiency, discipline) ARE comparable.
5. When you find evidence *against* an earlier finding of yours, record the
   correction explicitly. Retractions are wins, not embarrassments.
6. One finished run per pair per side: treat single-pair differences as
   observations; give weight to patterns that recur across pairs/domains.
6b. **The grid may be incomplete — that is data, not an obstacle.** The
   corpus is whatever the registry contains: domains may have a single
   run, or runs on only one side, or sides with different models. Start
   by printing the full run inventory and name every missing permutation
   explicitly (a missing cell is a reportable fact about the campaign).
   Then scale each verdict to the evidence that exists: pair verdicts
   only where a same-model pair exists; a cross-framework observation on
   mismatched models must be labeled model-confounded; a single-cell
   domain contributes within-run results and process evidence only.
   Never let an incomplete grid silently shrink a section — the section
   states what the corpus can and cannot support and reports what is
   there.
7. **Standard quantities come from the standard library, never re-derived.**
   Inside `run_python`, `from alpha_lab.benchmarks.runcmp import probe_std as std`
   gives you the deterministic layer's own definitions: `std.pack(label)`,
   `std.metrics(label)` (THE registry values bench.md prints, id -> value),
   `std.samples(label, key)` (THE raw distribution arrays — never
   re-extract them by hand),
   `std.trajectory(label)` (THE scored attempt/best-so-far series for
   every progression chart — never a hand-rolled join),
   `std.stage_decomposition(label)` (THE session/call/token stage accounting
   bench.md prints), `std.shell_pattern_counts(label, role=...)` (canonical
   mention- vs invocation-counting — never report a bare "pytest runs"),
   `std.definitions()` (every bench metric's definition). Ad-hoc derivation
   is allowed only for quantities the library lacks, and the finding must
   then state the definition used. Two investigators once both passed
   fact-check while disagreeing 3x on "pytest runs" because each invented
   its own counting — that is what this rule prevents.
7b. **NEVER PRESENT A SUBSET (hard rule).** You have full discretion to
   investigate deeper and to compute any quantity your own way — different
   populations, different definitions, sharper cuts are welcome and often
   the point. What is forbidden is leaving the reader with LESS than the
   whole picture:
   - When your computed quantity overlaps one the registry publishes
     (an id in `std.definitions()`), the registry value from
     `std.metrics(label)` appears ALONGSIDE yours, with your population/
     definition stated — the reader gets both, never only your version.
     (Two reports of this corpus printed different "median queue wait"
     values, each alone, from different row populations — 2026-08-02.
     Either number alone is a subset; both together are information.)
   - Any table, chart, or aggregate over runs/experiments covers the FULL
     population in scope, or states exactly what was excluded and why.
     A silently truncated series, a chart of 8 of 16 runs without saying
     so, a "representative" sample standing in for a census — all subsets.
   - If you believe a registry value is wrong or misleading, that is a
     finding: quote both numbers and name the definitional difference.
8. **Agent behavior is never spontaneous — trace it to its source.** Any
   behavioral difference you report (a verification habit, a guard, a
   proposal style, a tool-usage pattern) is unfinished until you identify
   its origin layer, quoting the artifact:
   - prescribed: prompt/template text (the run's `adapter/*.md`, agent
     registry prompts) or framework code/contract — quote the instruction;
   - emergent: the first occurrence (which agent, which experiment,
     which turn) and the propagation chain (copied code, debriefs,
     memory/playbook entries that fossilized it);
   - or explicitly unresolved, with the artifacts you checked.
   A report that says WHAT agents did differently without WHERE it came
   from is a symptom description, not a finding.

## Findings

Record findings as you go with `record_finding`. A finding must carry
machine-checkable evidence references (the tool result you derived it from):

- {"pack": "<run label>", "path": "experiments.best.value", "value": 0.8369}
- {"table": "<pair name>", "row": "memory tool failures", "side": "right", "value": 186}
- {"probe": "probe_003", "contains": "some literal string from its output"}
- {"file": "<absolute path>", "contains": "literal string"}  (optional contains)

References are resolved the moment you call `record_finding`; a finding with
any unresolvable reference is rejected with the mismatch detail so you can fix
it. Copy `contains` strings character-for-character from the tool output —
cite only lines a probe actually printed, never aggregates you computed
yourself (print those from a probe first, then cite the probe). Pack paths
follow the pack's real schema (e.g. `config.phase3.max_experiments`,
`experiments.best.value`); when unsure, `get_evidence` the section first.

Scope each finding honestly (one run / one pair / framework-wide) and, for
architectural judgments, name the choice being judged and your assessment in
your own words (good / bad / mixed / unresolved — with the tradeoff stated).

## Deliverable

When the investigation is complete, call `write_report` with REPORT.md:
plain language a person can read without knowing the internals, leading with
the architectural verdicts and what should change in each framework, findings
with their evidence, corrections/retractions, and what stays unresolved and
why. Then you are done. Work through the corpus systematically before writing:
the deterministic tables first (they frame the landscape), then the anomalies
and mechanisms behind the deltas that matter.
"""

SYSTEM_PROMPT += "\n" + CHART_SYNTAX

TOOLS = [
    {
        "type": "function",
        "name": "list_runs",
        "description": "Corpus registry: every discovered run with framework, domain, completeness, row counts, pairing.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_tables",
        "description": "The deterministic pair/corpus tables (markdown). Start here.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "get_evidence",
        "description": (
            "One section of a run's evidence pack. Sections: run, config, "
            "experiments (summary, no per-experiment list), experiments_full, "
            "events, agent_logs, run_log, conductor, memory, inventory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "section": {"type": "string"},
            },
            "required": ["label", "section"],
        },
    },
    {
        "type": "function",
        "name": "query_db",
        "description": "Read-only SQL against a run's experiments.db (SELECT only, max 200 rows).",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "sql": {"type": "string"},
            },
            "required": ["label", "sql"],
        },
    },
    {
        "type": "function",
        "name": "search_log",
        "description": (
            "Regex search over a run's raw artifacts. which: 'run.log', "
            "'events' (large, ~10s), or a glob under the workspace like "
            "'logs/*.jsonl' or 'meta/meta_log.jsonl'. Returns up to max_hits "
            "matching lines (each truncated)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "which": {"type": "string"},
                "pattern": {"type": "string"},
                "max_hits": {"type": "integer"},
            },
            "required": ["label", "which", "pattern"],
        },
    },
    {
        "type": "function",
        "name": "read_artifact",
        "description": (
            "Read lines from a file inside a run directory (bounded; max 400 "
            "lines per call). relpath is relative to the run dir (the "
            "workspace parent), e.g. 'ws/learnings.md', 'ws/meta/directives.md', "
            "'ws/experiments/<name>/debrief.md'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "relpath": {"type": "string"},
                "start_line": {"type": "integer"},
                "num_lines": {"type": "integer"},
            },
            "required": ["label", "relpath"],
        },
    },
    {
        "type": "function",
        "name": "run_python",
        "description": (
            "Run a python script over the corpus (120s timeout). Env vars "
            "RUNCMP_CORPUS (corpus.json), RUNCMP_PACKS (packs dir), "
            "RUNCMP_TABLES (tables.json) are set. Standard quantities: import "
            "alpha_lab.benchmarks.runcmp.probe_std (see evidence rule 7). "
            "Read-only discipline: do "
            "not write into run directories. stdout becomes citable probe "
            "output (probe_NNN)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"},
                           "note": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "type": "function",
        "name": "record_finding",
        "description": "Record one finding with machine-checkable evidence references.",
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "scope": {"type": "string"},
                "assessment": {"type": "string"},
                "choice": {"type": "string",
                           "description": "the decision judged, in the "
                                          "mission's decision space: an "
                                          "architectural choice (framework "
                                          "comparison) or a model/seat choice "
                                          "(model comparison)"},
                "evidence": {"type": "array", "items": {"type": "object"}},
                "notes": {"type": "string"},
            },
            "required": ["claim", "scope", "evidence"],
        },
    },
    {
        "type": "function",
        "name": "write_report",
        "description": "Write the final REPORT.md and finish the investigation.",
        "parameters": {
            "type": "object",
            "properties": {"markdown": {"type": "string"}},
            "required": ["markdown"],
        },
    },
]


# Evidence-integrity markers prepended to probe outputs that must not be cited.
# Measured incident (2026-07-29 final_grid): an investigator typed statistics
# into bare print() calls — probe code with no data access — labeled the output
# "full scan", and published a 12-row table whose growth figures were 20x off
# ground truth. The reference validator only checks that a cited string exists
# in a probe output, so manufactured output passed as "verified". These markers
# make such probes visibly and machine-checkably inadmissible.
REPRINT_PROBE_MARKER = (
    "[REPRINT-ONLY PROBE — this script reads no data; its output is typed, "
    "not measured, and is NOT admissible as evidence. Re-run as a probe that "
    "computes the values from the corpus/packs/run files.]"
)
STALE_PROBE_MARKER = (
    "[STALE-SOURCED PROBE — this script reads archived prior-generation "
    "reviewer artifacts; numbers from there are NOT admissible as evidence "
    "for this corpus. Compute fresh values from the corpus/packs/run files.]"
)

_PROBE_READ_RE = re.compile(
    r"\bopen\(|\bPath\(|glob\.|json\.load|read_text|read_bytes|os\.walk|"
    r"np\.load|pd\.read|sqlite3|iterdir|os\.listdir|np\.fromfile|environ\["
    # probe_std is this package's own evidence reader: a probe delegating to
    # it IS reading data. Without this token such probes were stamped
    # reprint-only, failed verification, and forced redundant raw-JSON
    # re-probes of already-measured values (external reader, 2026-08-11).
    r"|\bprobe_std\b"
)
_PROBE_STALE_RE = re.compile(
    r"pretables_|prebench_|aborted_|noincident_|badaccounting_|"
    r"/probes/|REPORT\.md|investigator_log|critic_log|findings\.jsonl"
)


def record_session(out_dir: Path, **fields) -> None:
    """Append this session's identity to ``<out>/sessions.jsonl``.

    Reviews once carried no record of which model wrote them; a later
    re-write then had to guess the writer — and a guess once silently
    switched a review's configuration to a different model (2026-08-10).
    Append-only, one line per session; ``rereview`` reads the last line.
    Never fatal: a review must not die because its provenance line could
    not be written.
    """
    try:
        with open(out_dir / "sessions.jsonl", "a") as fh:
            fh.write(json.dumps({"ts": time.time(), **fields},
                                default=str) + "\n")
    except OSError:
        pass


def _classify_probe_code(code: str) -> str | None:
    """Return an inadmissibility marker for probes that manufacture evidence.

    A probe whose code (comments and string literals stripped) contains no
    data-reading primitive can only print what was typed into it. A probe
    whose raw code references archived reviewer generations or reviewer
    output artifacts launders prior sessions' numbers. Real measurements
    return ``None``.
    """
    if _PROBE_STALE_RE.search(code):
        return STALE_PROBE_MARKER
    stripped_lines = []
    for ln in code.splitlines():
        ln = ln.split("#", 1)[0]
        ln = re.sub(r"'''.*?'''|\"\"\".*?\"\"\"", "''", ln)
        ln = re.sub(r"'[^']*'|\"[^\"]*\"", "''", ln)
        stripped_lines.append(ln)
    if not _PROBE_READ_RE.search("\n".join(stripped_lines)):
        return REPRINT_PROBE_MARKER
    return None


class InvestigatorSession:
    """Tool implementations bound to one corpus + output directory."""

    def __init__(self, corpus_path: Path, packs_dir: Path, out_dir: Path):
        self.records = {r.label: r for r in load_registry(corpus_path)}
        # Resolved to absolute: probes run with cwd=probes/, where a relative
        # script/corpus/packs path re-resolves inside that cwd and doubles up
        # (observed live: every probe of an investigator session failed with
        # ".../probes/comparison_out/.../probe_001.py: No such file").
        self.packs_dir = Path(packs_dir).resolve()
        self.out_dir = Path(out_dir).resolve()
        self.corpus_path = Path(corpus_path).resolve()
        # Continue numbering after any probes already on disk. A session that
        # reuses a review directory (recompose) used to restart at probe_001 and
        # OVERWRITE the very probe files the frozen findings cite — measured
        # 2026-08-10: a recomposed review went from 15/15 verified to 17/19
        # failed because its evidence had been clobbered by its own re-writer.
        # Probes are append-only per directory, never reused.
        existing = [int(m.group(1)) for m in
                    (re.match(r"probe_(\d+)\.", f.name)
                     for f in (self.out_dir / "probes").glob("probe_*"))
                    if m]
        self.probe_count = max(existing, default=0)
        self.findings = 0
        self.report_written = False
        self._own_usage: list[dict] = []   # this review's own model calls
        (out_dir / "probes").mkdir(parents=True, exist_ok=True)
        self.findings_path = out_dir / "findings.jsonl"
        # tabulate writes tables.md/.json next to corpus.json; investigations
        # usually run with --out a subdirectory of that root, so resolve the
        # deterministic tables from the corpus root first, out_dir as fallback
        # (the original layout where --out was the corpus root itself).
        self.tables_dir = (
            self.corpus_path.parent
            if (self.corpus_path.parent / "tables.md").is_file()
            else self.out_dir
        )

    # -- helpers ---------------------------------------------------------
    def _pack(self, label: str) -> dict | None:
        safe = label.replace("/", "__").replace("#", "_")
        p = self.packs_dir / f"{safe}.json"
        if not p.is_file():
            return None
        return json.loads(p.read_text())

    def _run_dir(self, label: str) -> Path | None:
        rec = self.records.get(label)
        return Path(rec.run_dir) if rec else None

    # -- tools -----------------------------------------------------------
    def list_runs(self) -> str:
        rows = []
        for r in self.records.values():
            rows.append(
                f"{r.label} | fw={r.framework} dom={r.domain} "
                f"complete={r.completeness} rows={r.db_rows} pair={r.pair_key or '-'}"
            )
        return "\n".join(rows)

    def get_tables(self) -> str:
        p = self.tables_dir / "tables.md"
        return p.read_text() if p.is_file() else "[ERROR] tables.md not built"

    def get_evidence(self, label: str, section: str) -> str:
        pack = self._pack(label)
        if pack is None:
            return f"[ERROR] no pack for {label!r} (labels: {list(self.records)[:8]}...)"
        if section == "experiments":
            data = dict(pack.get("experiments") or {})
            data.pop("experiments", None)
        elif section == "experiments_full":
            data = (pack.get("experiments") or {}).get("experiments")
        elif section == "events":
            data = dict(pack.get("events") or {})
            data.pop("phase_records", None)
        else:
            data = pack.get(section)
        if data is None:
            return f"[ERROR] no section {section!r} in pack for {label}"
        return json.dumps(data, indent=1, default=str)

    def query_db(self, label: str, sql: str) -> str:
        rec = self.records.get(label)
        if rec is None or not rec.db_path:
            return f"[ERROR] no database for {label!r}"
        if not re.match(r"\s*(select|with)\b", sql, re.I):
            return "[ERROR] SELECT/WITH queries only"
        try:
            con = sqlite3.connect(f"file:{rec.db_path}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            rows = con.execute(sql).fetchmany(200)
            out = [dict(r) for r in rows]
            con.close()
            return json.dumps(out, indent=1, default=str)
        except sqlite3.Error as exc:
            return f"[ERROR] {exc}"

    def search_log(self, label: str, which: str, pattern: str,
                   max_hits: int = 40) -> str:
        rec = self.records.get(label)
        if rec is None:
            return f"[ERROR] unknown label {label!r}"
        max_hits = min(int(max_hits or 40), 200)
        try:
            rx = re.compile(pattern.encode(), re.I)
        except re.error as exc:
            return f"[ERROR] bad regex: {exc}"
        if which == "run.log":
            paths = [Path(rec.run_log_path)] if rec.run_log_path else []
        elif which == "events":
            paths = [Path(rec.events_path)] if rec.events_path else []
        else:
            if ".." in which or which.startswith("/"):
                return "[ERROR] glob must be relative, no '..'"
            paths = sorted(Path(rec.workspace).glob(which))[:80]
        if not paths:
            return f"[ERROR] nothing to search for which={which!r}"
        hits: list[str] = []
        for path in paths:
            if not path.is_file():
                continue
            try:
                with open(path, "rb") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            text = line.decode(errors="replace").strip()[:300]
                            hits.append(f"{path.name}:{i}: {text}")
                            if len(hits) >= max_hits:
                                return "\n".join(hits) + "\n[truncated at max_hits]"
            except OSError as exc:
                hits.append(f"[ERROR] {path.name}: {exc}")
        return "\n".join(hits) if hits else "(no matches)"

    def read_artifact(self, label: str, relpath: str, start_line: int = 1,
                      num_lines: int = 120) -> str:
        run_dir = self._run_dir(label)
        if run_dir is None:
            return f"[ERROR] unknown label {label!r}"
        target = (run_dir / relpath).resolve()
        if not str(target).startswith(str(run_dir.resolve()) + "/"):
            return "[ERROR] path escapes the run directory"
        if not target.is_file():
            return f"[ERROR] not a file: {relpath}"
        num_lines = min(int(num_lines or 120), 400)
        start_line = max(int(start_line or 1), 1)
        out = []
        with open(target, errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if i < start_line:
                    continue
                if i >= start_line + num_lines:
                    out.append(f"[...truncated at line {i}]")
                    break
                out.append(f"{i}\t{line.rstrip()[:500]}")
        return "\n".join(out) if out else "(empty range)"

    def note_own_usage(self, seat: str, model: str, provider,
                       response) -> None:
        """Record one of THIS review's own model calls, for the cost footer.

        Fresh-input semantics differ by provider: Anthropic/Bedrock report
        input_tokens EXCLUDING cache reads, the OpenAI-shaped providers
        INCLUDE them — normalized here so the footer's arithmetic is one
        rule. Bookkeeping must never break a review: any failure is dropped.
        """
        try:
            inp = int(getattr(response, "input_tokens", 0) or 0)
            cr = int(getattr(response, "cache_read_input_tokens", 0) or 0)
            cw = int(getattr(response, "cache_write_input_tokens", 0) or 0)
            out = int(getattr(response, "output_tokens", 0) or 0)
            pname = type(provider).__name__.lower()
            fresh = inp if ("anthropic" in pname or "bedrock" in pname) \
                else max(inp - cr, 0)
            rec = {"seat": seat, "model": model, "fresh_input": fresh,
                   "cache_read": cr, "cache_write": cw, "output": out}
            self._own_usage.append(rec)
            with open(self.out_dir / "usage.jsonl", "a") as fh:
                fh.write(json.dumps({"ts": time.time(), **rec}) + "\n")
        except Exception:  # noqa: BLE001 — never let accounting kill a review
            pass

    def _production_cost_footer(self) -> str:
        """The bill for making THIS report, appended at publication.

        Token counts are the session's own API responses (usage.jsonl);
        prices come from the same rates table token accounting uses.
        Bounded by PRODUCTION_COST_MARKER so the fact-checker exempts these
        numbers from the body audit. Empty when nothing was recorded (old
        sessions, unit tests)."""
        entries = getattr(self, "_own_usage", None) or []
        if not entries:
            return ""
        from alpha_lab.benchmarks.runcmp.render_html import (
            PRODUCTION_COST_MARKER)
        from alpha_lab.benchmarks.runcmp.tabulate import rates_for
        by: dict[tuple, dict] = {}
        for e in entries:
            d = by.setdefault((e["seat"], e["model"]),
                              {"calls": 0, "fresh_input": 0, "cache_read": 0,
                               "cache_write": 0, "output": 0})
            d["calls"] += 1
            for f in ("fresh_input", "cache_read", "cache_write", "output"):
                d[f] += e[f]
        total = 0.0
        calls = t_in = t_cached = t_out = 0
        bits = []
        for (seat, model), d in sorted(by.items()):
            r = rates_for(model)
            cost = (d["fresh_input"] * r["input"]
                    + d["cache_read"] * r["cache_read"]
                    + d["cache_write"] * r["cache_write"]
                    + d["output"] * r["output"]) / 1e6
            total += cost
            calls += d["calls"]
            t_in += d["fresh_input"] + d["cache_read"]
            t_cached += d["cache_read"]
            t_out += d["output"]
            bits.append(f"{seat} ${cost:,.2f} ({d['calls']} calls, {model})")
        return (PRODUCTION_COST_MARKER + "\n\n---\n\n"
                f"*Production cost of this report: **${total:,.2f}** — "
                f"{calls} model calls, {t_in/1e6:.1f}M tokens in "
                f"({t_cached/1e6:.1f}M from cache), {t_out/1e6:.2f}M out. "
                "By seat: " + "; ".join(bits) + ". Token counts are this "
                "session's own API usage (usage.jsonl beside this report); "
                "prices are the package's list-price table (lab-hosted "
                "models are priced 0). The deterministic stages and the "
                "fact-check run without models.*")

    def run_python(self, code: str, note: str = "") -> str:
        self.probe_count += 1
        probe_dir = self.out_dir / "probes"
        # Probe files are immutable once written: findings cite them by name,
        # so writing over one silently invalidates recorded evidence. The
        # counter starts past everything on disk (see __init__); this guard
        # holds even if that ever regresses.
        while ((probe_dir / f"probe_{self.probe_count:03d}.py").exists()
               or (probe_dir / f"probe_{self.probe_count:03d}.out").exists()):
            self.probe_count += 1
        name = f"probe_{self.probe_count:03d}"
        script = probe_dir / f"{name}.py"
        header = f"# note: {note}\n" if note else ""
        script.write_text(header + code)
        inadmissible = _classify_probe_code(code)
        env = dict(
            RUNCMP_CORPUS=str(self.corpus_path),
            RUNCMP_PACKS=str(self.packs_dir),
            RUNCMP_TABLES=str(self.tables_dir / "tables.json"),
            PATH="/usr/bin:/bin",
            HOME=str(probe_dir),
        )
        import os as _os
        if _os.environ.get("PYTHONPATH"):
            env["PYTHONPATH"] = _os.environ["PYTHONPATH"]
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=120,
                cwd=str(probe_dir), env=env,
            )
            output = proc.stdout[:MAX_TOOL_OUTPUT]
            if proc.returncode != 0:
                output += f"\n[stderr]\n{proc.stderr[-4000:]}"
        except subprocess.TimeoutExpired:
            output = "[ERROR] probe timed out after 120s"
        if inadmissible:
            output = inadmissible + "\n" + output
        (probe_dir / f"{name}.out").write_text(output)
        return f"[{name}]\n{output}"

    def record_finding(self, **kwargs) -> str:
        evidence = kwargs.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            return "[ERROR] finding needs a non-empty evidence list"
        # Validate every reference NOW, while the tool outputs are still in
        # context — a reference reconstructed from memory instead of copied
        # verbatim is rejected with the exact mismatch so it can be fixed.
        from alpha_lab.benchmarks.runcmp.factcheck import FactChecker

        checker = FactChecker(self.out_dir, self.packs_dir)
        problems = []
        for i, ref in enumerate(evidence):
            if not isinstance(ref, dict):
                problems.append(f"evidence[{i}] is not an object")
                continue
            ok, detail = checker.check_reference(ref)
            if not ok:
                problems.append(f"evidence[{i}] {json.dumps(ref)[:160]} — {detail}")
        if problems:
            return (
                "[ERROR] finding NOT recorded; fix these references and "
                "re-record (copy `contains` strings character-for-character "
                "from the tool output; cite only lines a probe actually "
                "printed):\n" + "\n".join(problems)
            )
        self.findings += 1
        rec = {"id": self.findings, "ts": time.time(), **kwargs}
        with open(self.findings_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return f"finding #{self.findings} recorded (all references verified)"

    def _registry_echo_problems(self, markdown: str) -> list[str]:
        """Deterministic never-a-subset check (user rulings 2026-08-02).

        Two reports in a row printed self-anchored 'time to best' / 'wall
        hours' values with no registry value alongside — prompt-level rules
        did not bind the critic-less configuration, so this gate does. Any
        line naming an overlap-prone registry family must carry mostly
        numbers that exist in bench.json for that family (a compliant
        both-numbers presentation passes automatically).
        """
        fam_patterns = {
            "search.time_to_best_hours": r"time[\s_-]?to[\s_-]?best",
            "lifecycle.wall_hours": r"wall[\s_-]?(clock[\s_-]?)?hours?",
            "search.median_queue_wait_minutes": r"queue[\s_-]?wait",
        }
        bench_path = Path(self.tables_dir) / "bench.json"
        if not bench_path.exists():
            return []
        bench = json.loads(bench_path.read_text())
        published: set[float] = set()
        for run in bench.get("runs", {}).values():
            for mid in fam_patterns:
                v = run.get("metrics", {}).get(mid)
                if isinstance(v, (int, float)):
                    published.add(round(float(v), 2))
        if not published:
            return []
        num_re = re.compile(r"\b\d+(?:\.\d+)?\b")
        hits = total = 0
        in_matched_table = False
        for line in markdown.splitlines():
            is_table_row = line.lstrip().startswith("|")
            matched = any(re.search(p, line, re.IGNORECASE)
                          for p in fam_patterns.values())
            # A table header naming the quantity taints every row of that
            # table: the numbers live in rows that don't repeat the phrase.
            if is_table_row and matched:
                in_matched_table = True
            elif not is_table_row:
                in_matched_table = False
            if not (matched or (is_table_row and in_matched_table)):
                continue
            for tok in num_re.findall(line):
                val = float(tok)
                if not (0.01 <= val <= 10000):
                    continue
                total += 1
                if any(abs(val - p) <= 0.006 for p in published):
                    hits += 1
        if total >= 4 and hits / total < 0.5:
            # Make compliance copy-paste: print the registry's own values.
            # (A model once responded to the bare rule by deleting the
            # quantities from the report instead of quoting the registry —
            # dropping variables is the original sin this gate exists for.)
            listing = []
            for label, run in sorted(bench.get("runs", {}).items()):
                short = label.split("/")[-1]
                vals = ", ".join(
                    f"{mid.split('.')[-1]}={run['metrics'][mid]}"
                    for mid in fam_patterns
                    if isinstance(run.get("metrics", {}).get(mid), (int, float))
                )
                if vals:
                    listing.append(f"{short}: {vals}")
            return [
                f"registry echo check failed: lines naming time-to-best / "
                f"wall hours / queue wait carry {total} numbers but only "
                f"{hits} match bench.json's published values. These "
                f"quantities MUST appear, and MUST use the registry values "
                f"below (you may show your own-anchor number IN ADDITION, "
                f"definition stated — never alone, and never delete the "
                f"quantity). Registry values (time_to_best_hours, "
                f"wall_hours, median_queue_wait_minutes): "
                + "; ".join(listing)
            ]
        return []

    def _progression_series_problems(self, markdown: str) -> list[str]:
        """Deterministic trajectory-fidelity gate (2026-08-03). Three
        observer reports in a row printed progression series contradicting
        the packs; the last ran the correct std.trajectory probe and then
        wrote different numbers into its charts, and the body-number audit
        bins chart payloads into their own uncounted bucket, so nothing
        fired. Prompt rules did not bind; this gate does: every x,value
        series must be quotable from an admissible source at its printed
        precision."""
        from alpha_lab.benchmarks.runcmp import factcheck as _fc
        det, clean, _dirty = _fc.load_admissible_sources(self.out_dir,
                                                         self.packs_dir)
        audit = _fc.audit_progression_series(markdown, det, clean)
        if not audit["series_flagged"]:
            return []
        details = "; ".join(
            f"series {f['series']!r} in chart {f['chart']!r}: "
            f"{f['unmatched']} of {f['distinct_values']} distinct values "
            "match no probe output or corpus file (e.g. "
            + ", ".join(f["examples"]) + ")"
            for f in audit["flags"][:6])
        return [
            "progression-series check failed: " + details
            + ". Chart series are DATA, not prose: print the authoritative "
              "series in a probe (std.trajectory(label) is the standard "
              "source for scored-attempt series) and paste those exact "
              "values into the chart — never retype from memory, "
              "interpolate, or reconstruct them."]

    def _referee_attribution_problems(self, markdown: str) -> list[str]:
        """Deterministic verbatim-attribution gate (2026-08-03): a report
        printed two names in a 'referee-selected champion' table that are
        not referee.json's selections — one retyped from memory, one lifted
        from a non-champion leaderboard row. Values and winners were right;
        only the strings drifted, which no numeric gate can see."""
        from alpha_lab.benchmarks.runcmp import factcheck as _fc
        referee_path = Path(self.tables_dir) / "referee.json"
        if not referee_path.is_file():
            return []
        referee = json.loads(referee_path.read_text())
        audit = _fc.audit_referee_attributions(markdown, referee)
        if not audit["violations"]:
            return []
        best = sorted(_fc._collect_best_names(referee))

        def _describe(v):
            if v.get("where") == "missing-referee-value":
                return (f"the row naming `{v['name']}` does not carry its "
                        f"referee score {v.get('expected')} — a different "
                        "number in the referee's column is a substituted "
                        "attribution (you may show your own number too, "
                        "labeled, but the referee's value must be there)")
            return (f"`{v['name']}` is presented as a referee selection "
                    f"but is not in referee.json's best set")
        return [
            "referee-attribution check failed: "
            + "; ".join(_describe(v) for v in audit["violations"][:6])
            + ". Referee names AND values are QUOTES: copy them from "
              "referee.json (pairs[].best.left/right.experiment + "
              ".referee_score), never retype or substitute. The referee's "
              "declared best names are: "
            + ", ".join(f"`{n}`" for n in best)]

    def write_report(self, markdown: str) -> str:
        # Chart-floor gate (user ruling 2026-08-01: most numerical data must
        # be presented pictorially, diversity proportionate to the data). A
        # draft under the floors is refused with an actionable census, up to
        # three times; the writer revises and resubmits. After three refusals
        # the report is accepted with a visible warning — never lost.
        census = _chart_census(markdown)
        problems = []
        from alpha_lab.benchmarks.runcmp.render_html import chart_errors
        renderer_errors = chart_errors(markdown)
        if renderer_errors:
            problems.append(
                f"{len(renderer_errors)} chart block(s) will NOT render: "
                + " || ".join(renderer_errors[:5])
                + " — rewrite them in the exact pipe-row grammar from your "
                  "instructions (section 'Chart blocks — EXACT grammar')")
        problems.extend(self._registry_echo_problems(markdown))
        problems.extend(self._progression_series_problems(markdown))
        problems.extend(self._referee_attribution_problems(markdown))
        if census["total"] < 12:
            problems.append(f"only {census['total']} charts (>=12 required)")
        if len(census["kinds"]) < 6:
            problems.append(f"only {len(census['kinds'])} chart forms "
                            "(>=6 required)")
        if census["dist_forms"] < 4:
            problems.append(f"only {census['dist_forms']} distribution forms "
                            "(>=4 required among hist / density / box / band "
                            "/ scatter with marginals: true)")
        refusals = getattr(self, "_report_refusals", 0)
        # Anti-evasion: remember the most substantial draft seen. A model
        # once responded to refusals by writing its content to a side file
        # and submitting a 4-line pointer stub, which the warn-accept path
        # then published (one solo review, 2026-08-02). The warn-accept path now
        # publishes the best stored draft, never a later smaller one.
        best = getattr(self, "_best_draft", None)
        if best is None or (census["total"], len(markdown)) >= (
                _chart_census(best)["total"], len(best)):
            self._best_draft = markdown
        if problems and refusals < 3:
            self._report_refusals = refusals + 1
            return (f"[REPORT REFUSED {self._report_refusals}/3] "
                    + "; ".join(problems)
                    + ". Revise and resubmit the FULL report — keep all "
                    "existing content, add the missing pictorial forms from "
                    "data already in your draft or probe outputs: raw "
                    "distribution arrays are in pack agent_logs.samples "
                    "(request_bytes, tool_result_bytes, llm_gap_seconds, "
                    "tool_gap_seconds, session_minutes), per-experiment "
                    "durations in the pack experiment lists, attempt spreads "
                    "as line + band, marginals: true on any 10+-point "
                    "scatter. Writing the report anywhere other than "
                    "write_report does not count: only the draft submitted "
                    "here is audited and published.")
        if problems:
            kept = self._best_draft
            if len(kept) > len(markdown):
                markdown = kept
            # Never publish dead chart fences, even on the warn-accept path:
            # strip any block the renderer rejects (the floors are advisory
            # after three refusals; raw fence text on a page never is).
            from alpha_lab.benchmarks.runcmp.render_html import (
                strip_unrenderable_charts)
            markdown, n_dead = strip_unrenderable_charts(markdown)
            if n_dead:
                problems.append(f"{n_dead} unrenderable chart block(s) "
                                "stripped at publication")
                problems = (self._registry_echo_problems(markdown)
                            + self._progression_series_problems(markdown)
                            + self._referee_attribution_problems(markdown)
                            ) or problems
            markdown = ("> Chart-floor warning (accepted after 3 refusals): "
                        + "; ".join(problems) + "\n\n" + markdown)
        # Both formats, always: the markdown is what factcheck.py audits, the
        # HTML is what a human opens. Written by one call so they cannot drift.
        from alpha_lab.benchmarks.runcmp.render_html import write_report_pair

        footer = self._production_cost_footer()
        if footer and "<!-- production-cost -->" not in markdown:
            markdown = markdown.rstrip() + "\n\n" + footer + "\n"
        _, html_path = write_report_pair(self.out_dir, markdown)
        self.report_written = True
        return (f"REPORT.md written ({len(markdown):,} chars); "
                f"{html_path.name} rendered alongside it")

    def dispatch(self, name: str, args: dict) -> str:
        try:
            fn = getattr(self, name, None)
            if fn is None:
                return f"[ERROR] unknown tool {name!r}"
            return str(fn(**args))
        except TypeError as exc:
            return f"[ERROR] bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 — surfaced to the agent
            return f"[ERROR] {type(exc).__name__}: {exc}"


def _estimate_chars(history: list[dict]) -> int:
    return sum(len(json.dumps(item, default=str)) for item in history)


def _compact_history(history: list[dict]) -> None:
    """Blank out old tool outputs when the transcript gets heavy."""
    if _estimate_chars(history) < HISTORY_CHAR_BUDGET:
        return
    cutoff = len(history) - KEEP_RECENT_ITEMS
    for item in history[:cutoff]:
        if item.get("type") == "function_call_output" and len(
            str(item.get("output") or "")
        ) > 500:
            item["output"] = "[output dropped to save context — re-run the tool if needed]"
        # Submitted drafts dominate a long revision loop: every bounced
        # write_report keeps its full markdown in the call ARGUMENTS, which
        # the output-only rule above never touches — 23 bounced drafts of
        # ~60-80k chars drowned a writer's context entirely (one re-writing
        # session, 2026-08-10). The draft itself is safe on disk under
        # report_drafts/; the replacement stays valid JSON because provider
        # translators may parse arguments when rebuilding history.
        elif item.get("type") == "function_call" and len(
            str(item.get("arguments") or "")
        ) > 4000:
            item["arguments"] = json.dumps(
                {"dropped": "arguments dropped to save context — submitted "
                            "report drafts are kept under report_drafts/"})


def run_investigation(
    corpus_path: Path,
    packs_dir: Path,
    out_dir: Path,
    provider_name: str,
    model: str,
    reasoning_effort: str = "high",
    mission: str = "",
) -> bool:
    from alpha_lab.client import get_provider

    provider = get_provider(provider_name)
    record_session(out_dir, stage="investigate", provider=provider_name,
                   model=model, reasoning_effort=reasoning_effort)
    session = InvestigatorSession(corpus_path, packs_dir, out_dir)
    history: list[dict] = []
    transcript = open(out_dir / "investigator_log.jsonl", "a")

    def log(kind: str, payload) -> None:
        transcript.write(json.dumps({"ts": time.time(), kind: payload},
                                    default=str)[:40_000] + "\n")
        transcript.flush()

    # The prompt is harness-agnostic; the actual roster is injected here
    # from the registry so the investigator never inherits a stale world.
    from collections import Counter
    fw = Counter(r.framework for r in session.records.values())
    dom = Counter(r.domain for r in session.records.values())
    roster = ("Corpus roster (from the registry): "
              + ", ".join(f"{n} run(s) from harness '{f}'"
                          for f, n in sorted(fw.items()))
              + " across tasks "
              + ", ".join(f"{d} ({n})" for d, n in sorted(dom.items()))
              + ".")
    opening = (roster
               + "\nBegin. Start from list_runs and get_tables, then "
                 "investigate.")
    if mission:
        opening += f"\n\n## Mission scope for this investigation\n{mission}"
    history.extend(provider.build_user_items(opening))
    # Budget scales with corpus size AND mission size: a bigger corpus
    # needs proportionally more probing, and a mission with more decision
    # families needs more room per family (2026-08-08: a 15-family union
    # mission ran on the 4-family allowance and composed thin). Families
    # are the mission's numbered bold items.
    n_runs = len(session.records)
    n_families = len(re.findall(r"^\s*\d+\.\s+\*\*", mission or "",
                                re.MULTILINE))
    budget = (MAX_ITERATIONS + ITER_PER_EXTRA_RUN * max(0, n_runs - 2)
              + ITER_PER_EXTRA_FAMILY * max(0, n_families - 4))
    print(f"iteration budget: {budget} for {n_runs} runs / "
          f"{max(n_families, 4)} mission families "
          f"(compose reserve {COMPOSE_RESERVE})", flush=True)
    composing = False
    for iteration in range(budget):
        # Force composition before the cap so the report is never lost.
        if not composing and iteration >= budget - COMPOSE_RESERVE:
            composing = True
            history.extend(provider.build_user_items(
                f"ITERATION BUDGET NOTICE: {budget - iteration} iterations "
                "remain. Stop investigating now and compose the report from "
                "what you already have. Call write_report before the budget "
                "is exhausted — an unwritten report loses all of this work. "
                "Report checklist before you write: (1) chart variety — if "
                "the mission defines chart types, research progress gets a "
                "`line` chart over experiments or wall-clock per task "
                "(`style: step` for best-so-far), composition gets "
                "`stacked`, trade-offs get `scatter` (with `marginals: "
                "true` at 10+ points), pair margins suit `dumbbell`, "
                "timelines/stalls suit `spans`, matrices suit `heatmap`, "
                "and `bars` only for ranked lists; pick what fits, but an "
                "all-one-type report is defective. (2) Distributions: at least FOUR distinct distribution forms (hist / density / box / band / marginal scatter) at this corpus size, and any "
                "quantity you summarized by median/mean with 10+ values "
                "behind it appears at least once as `hist`, `density`, or "
                "`box` where it carries a verdict — packs carry raw arrays "
                "in agent_logs.samples. (3) Every anomaly you present must "
                "end in a mechanism, a named missing artifact, or an "
                "explicit open question. (4) Define every term of art at "
                "first use. (5) REGISTRY CROSS-CHECK, row by row: any table "
                "column or chart named like a registry metric (time to "
                "best, wall hours, queue wait, replay share, failure rate, "
                "cost, code metrics) must carry the std.metrics(label) "
                "value — or show BOTH your derived number and the registry "
                "value with your definition stated. An own-derived value "
                "standing alone where the registry publishes the same "
                "quantity makes the report fail review (rule 7b)."))
            log("budget_notice", {"iteration": iteration, "budget": budget})
        response = None
        for attempt in range(4):
            try:
                text = ""
                for event in provider.stream_response(
                    model=model,
                    system=SYSTEM_PROMPT,
                    history=history,
                    tools=TOOLS,
                    reasoning_effort=reasoning_effort,
                ):
                    if event.type == "text_delta":
                        text += event.delta
                    elif event.type == "done":
                        response = event.response
                break
            except Exception as exc:  # noqa: BLE001 — retried
                wait = 15 * (attempt + 1)
                print(f"  API error ({exc}); retry in {wait}s", flush=True)
                log("api_error", str(exc))
                time.sleep(wait)
        if response is None:
            print("  giving up: API unavailable")
            transcript.close()
            return False
        session.note_own_usage("investigator", model, provider, response)
        provider.append_response_to_history(history, response)
        if response.text:
            log("text", response.text)
            print(f"[{iteration}] {response.text[:300]}", flush=True)
        if not response.tool_calls:
            history.extend(provider.build_user_items(
                "Continue with tool calls, or finish with write_report."
            ))
            continue
        outputs = []
        for tc in response.tool_calls:
            args = {}
            try:
                args = json.loads(tc.arguments) if tc.arguments else {}
            except json.JSONDecodeError:
                pass
            result = session.dispatch(tc.name, args)
            log("tool", {"name": tc.name, "args": args, "result": result[:2000]})
            print(f"[{iteration}] {tc.name}({str(args)[:120]}) -> {len(result)} chars",
                  flush=True)
            outputs.append({"call_id": tc.call_id,
                            "output": result[:MAX_TOOL_OUTPUT]})
        history.extend(provider.build_tool_result_items(outputs))
        _compact_history(history)
        if session.report_written:
            print(f"done: {session.findings} findings, report written")
            transcript.close()
            return True
    print("iteration cap reached without write_report")
    transcript.close()
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Agentic corpus investigation")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--packs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--mission", default="auto",
                    help="investigation scope: 'auto' (default — generated "
                         "from the corpus + referee; 'auto:<focus>' picks a "
                         "decision preset, see `runcmp mission --help`), "
                         "'@/path/to/file', 'none', or inline text")
    ap.add_argument("--note", action="append", default=[],
                    help="operator-declared corpus fact for auto missions "
                         "(repeatable)")
    args = ap.parse_args(argv)
    from alpha_lab.benchmarks.runcmp.mission import resolve_mission_arg
    from alpha_lab.benchmarks.runcmp.mission import expand_notes
    mission = resolve_mission_arg(args.mission, args.corpus, args.out,
                                  notes=expand_notes(args.note))
    ok = run_investigation(
        args.corpus, args.packs, args.out, args.provider, args.model,
        args.reasoning_effort, mission=mission,
    )
    if ok:
        from alpha_lab.benchmarks.runcmp.mission import refresh_reports_index
        refresh_reports_index(args.out.parent)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
