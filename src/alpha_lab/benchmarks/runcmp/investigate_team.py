"""Multi-role (optionally multi-model) investigation: planner -> executor <-> critic.

The single-model investigator (investigate.py) conflates three jobs: deciding
what to look for, gathering evidence, and judging whether a drafted finding is
actually supported. The meta-benchmarks showed different models are strong at
different jobs (sol: auditing/provenance; opus: synthesis), so this module
splits the jobs into roles with a per-role model assignment:

- planner   (optional): one call — turns the mission + deterministic tables
             into a prioritized evidence plan the executor works through.
- executor  (required): the existing tool loop; does all evidence work.
- critic    (optional): gates every record_finding. A draft that passes the
             deterministic reference check still needs the critic's accept;
             revise/reject verdicts flow back to the executor as tool output.

Every role accepts any provider/model, and all roles may be the SAME model —
the architecture must stand on its own even when no model diversity exists.
Role assignment comes from --role NAME=PROVIDER:MODEL (repeatable); unassigned
roles inherit --provider/--model. --role planner=none / critic=none disables a
role (executor-only equals the classic single-agent investigator).

Outputs are a superset of investigate.py's: findings.jsonl (with a
critic_verdict field when the critic is on), plan.md, critic_log.jsonl,
investigator_log.jsonl, REPORT.md — so factcheck.py runs unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from alpha_lab.benchmarks.runcmp.investigate import (
    MAX_ITERATIONS,
    MAX_TOOL_OUTPUT,
    SYSTEM_PROMPT,
    TOOLS,
    InvestigatorSession,
    _compact_history,
)

MAX_CRITIQUES_PER_CLAIM = 3
# Global churn budget: total revise verdicts across all claims. Beyond this
# the critic is restricted to accept/reject (t3 spent 26 revision round-trips
# and hit the iteration cap without ever attempting its report — the budget
# must stay finite). Measured defect the other way (2026-07-29 final_grid,
# all-opus team): 12 was exhausted by the first twelve drafts, converting the
# 13 later genuine "revise" verdicts into automatic claim-closures — the run
# recorded 1 finding from 25 distinct claims while its critic accepted only
# one outright. 30 keeps the ceiling meaningful without deciding most claims
# by arithmetic.
MAX_TOTAL_REVISIONS = 30
# Budget scales with corpus x mission size. Measured defect (2026-07-26
# model_ab lineup): the flat 150-iteration budget is calibrated to a 2-run
# pair; on the 8-run corpus every team config collapsed (acceptance rates
# 34%->11%, two forced flushes) because critics demanded wider evidence and
# executors ran out of iterations. Solos on the same corpus finished with
# budget to spare only because nobody taxed their per-claim iterations.
ITER_PER_EXTRA_RUN = 12
ITER_PER_EXTRA_QUESTION = 8
REFUNDS_PER_EXTRA_RUN = 4
# Critic-rejection refunds must scale with the mission too: a 15-family
# mission produces ~3x the findings of a 4-family one, and a refund pool
# sized for 4 families silently taxes late findings' rejections against
# the main budget (found 2026-08-09, same shape as the budget freeze).
REFUNDS_PER_EXTRA_QUESTION = 3
# When this many iterations remain, investigation stops and composition
# starts — the forced flush produced finding dumps with no decision section
# and no mandatory tables (mission-compliance failures) because the executor
# was never ordered to stop digging and compose.
COMPOSE_RESERVE = 25
# Critic bounces refund executor iterations (bounded): the critic exists to
# raise finding quality, not to tax investigation breadth. Measured in rounds
# 1-2: every team config grazed MAX_ITERATIONS while solos finished with a
# third of the budget to spare; t6's 38 bounces were exactly its DNF margin.
MAX_BOUNCE_REFUNDS = 40

QUESTION_TOOLS = [
    {"type": "function", "name": "list_questions",
     "description": "The open-questions ledger: every question with its "
                    "status (open/resolved/unresolvable). The report cannot "
                    "be written while any question is open.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "add_question",
     "description": "Register a new question the investigation must resolve "
                    "(from the plan, or discovered mid-investigation).",
     "parameters": {"type": "object", "properties": {
         "text": {"type": "string"}}, "required": ["text"]}},
    {"type": "function", "name": "resolve_question",
     "description": "Mark a ledger question resolved, citing the finding ids "
                    "that answer it.",
     "parameters": {"type": "object", "properties": {
         "question_id": {"type": "integer"},
         "resolution": {"type": "string",
                        "description": "one-paragraph answer"},
         "finding_ids": {"type": "array", "items": {"type": "integer"},
                         "description": "recorded findings that carry the "
                                        "evidence (may be empty only for "
                                        "trivial factual lookups)"}},
         "required": ["question_id", "resolution"]}},
    {"type": "function", "name": "mark_unresolvable",
     "description": "Declare a ledger question unresolvable with a concrete "
                    "reason (e.g. the artifact was not preserved). The critic "
                    "reviews this; lazy bail-outs are rejected.",
     "parameters": {"type": "object", "properties": {
         "question_id": {"type": "integer"},
         "reason": {"type": "string"}}, "required": ["question_id", "reason"]}},
]

REPORT_CRITIC_SYSTEM = """\
You are the CRITIC reviewing the FINAL REPORT of an evidence-driven
investigation, against the mission that commissioned it. Until now you have
only gated individual findings; this is the deliverable, and it is the only
thing the reader sees.

The mission lists the sections the report must deliver and how deeply. Check
delivery, not style:
- Does every section the mission demands exist AND answer, with evidence?
  A heading followed by one line per case is a placeholder, not an answer —
  the mission's words for this are "nothing may be dropped or compressed
  into a table row".
- Where the mission says answer per task / per model / per seat, is that done,
  or is one aggregate sentence standing in for the breakdown?
- Are the verdicts graded (how strong, what would change them) rather than
  asserted?
- Are the report's numbers carried with their provenance (a probe, a finding,
  a quoted line), and do the stated margins get compared against the noise
  the mission supplies?
- Does the report state what it could not settle, and what would settle it?

Do NOT ask for prose polish, reordering, or extra charts; do not ask for new
evidence-gathering that the report's own findings do not already support. Ask
only for content the executor can write from evidence it already has.

Reply with STRICT JSON only:
{"verdict": "accept"|"revise", "score": <0-100>, "reasons": "<short>",
 "required_changes": "<numbered, specific: which section, what must be added>"}

`score` is how much of the mission this draft delivers, 0-100, judged on
coverage and evidence — NOT on length. A padded draft scores lower than a
compact one that answers everything. Score each draft on its own merits; the
best-scoring draft is what gets published if the revision budget runs out, so
an honest number matters more than a generous one.

Accept a report that answers the mission, even if terse in places. Use
"revise" when whole demanded sections are stubs, when a per-task or per-seat
breakdown the mission required is missing, or when verdicts arrive ungraded.
There is no "reject": the report is published either way, so a revise must be
worth one more drafting round."""

PLANNER_SYSTEM = """\
You are the PLANNER of an evidence-driven investigation of autonomous research
runs. You get the mission and the deterministic comparison tables. Produce a
prioritized evidence plan for the executor: the 5-10 most decision-relevant
questions, and for each, WHERE the evidence lives (which pack sections, DB
queries, log searches, or python probes over preserved artifacts) and what a
sufficient answer looks like. Flag the traps: denominator effects, tool counts
that conflate mentions with invocations, claims that need provenance (worker
behavior is never spontaneous), and self-reported metrics that should be
recomputed from preserved artifacts rather than trusted. Be concrete and
short; the executor has full tool access and the same mission text."""

CRITIC_SYSTEM = """\
You are the CRITIC gating which findings enter the record of an evidence-driven
investigation. You get one draft finding (claim, scope, choice, assessment,
evidence references whose quoted `contains` strings were already verified to
exist character-for-character in tool outputs) plus the mission. Your job is
adversarial review of SUPPORT, not style:
- Does the quoted evidence actually establish the claim, or only something
  weaker/adjacent? Name the gap.
- Is any causal or provenance step asserted without a reference?
- Are numbers used with the right denominators and definitions?
- Is the claim over-general (population words backed by samples)?
You also receive the full (bounded) output of every probe the draft cites.
If support the draft lacks is sitting in a cited probe's output, do NOT
declare the point untested and do NOT reject: use "revise" and name the
exact line the executor should quote. Missing evidence that you can see is
a copy instruction, not a gap.
A draft that presents an anomaly — two numbers pointing opposite ways, a
paradox, an unexplained gap — and stops at naming it is a symptom, not a
finding. Use "revise" and require one of: (a) the mechanism that produces
the anomaly, with evidence; (b) the named preserved artifact that would
decide it but is absent; or (c) an explicit statement that the question is
open. "Semantics differ" or "caching differs" is the beginning of an
explanation, not the end: the revision must name the two semantics and
reconcile one number under each.
Reply with STRICT JSON only:
{"verdict": "accept"|"revise"|"reject", "reasons": "<short>",
 "required_changes": "<what the executor must add/narrow; empty for accept>"}
Accept findings that are narrow and fully supported even if modest. Use
"revise" (naming the exact missing piece) when the direction is right but the
support falls short — census and provenance claims especially deserve a revise
demanding stated definitions, not a reject over definitional quibbles.
Numbers derived by simple arithmetic from quoted references — deltas, ratios,
percentages, orderings among quoted values — need no quote of their own:
check the arithmetic yourself and accept if it holds; demand a quote only for
a raw value that has no quoted source. When the claim's direction is verified
and the only gap is a quote the executor could copy from a probe it already
ran, prefer accept (stating the caveat in reasons) over revise: measured on
this pipeline, executors abandon most revise verdicts rather than loop, so a
revise usually costs the record a true finding, not a round-trip.
Reserve "reject" for claims that are wrong, unfixable, or would need evidence
that does not exist. Do not reward volume; do not destroy true findings on
style grounds.
Scale your demands to the CLAIM'S scope, not the corpus size: a claim scoped
to one run needs that one run's evidence; only claims using population words
(all/every/none/each/never) warrant full-population demands. On a many-run
corpus, demanding cross-run censuses for single-run claims burns the whole
investigation's budget on one finding — that is a critic failure, not rigor.
The executor's iterations are a shared, finite resource you are spending.
Treat "unresolved" verdicts inside draft findings with the same rigor as
claims: bounce them unless the finding shows the seat's/dimension's own
outputs were directly examined and a graded verdict was attempted —
outcome-confounding alone never justifies "unresolved".
NEVER A SUBSET: the executor has full discretion to compute quantities its
own way — that is investigation, not a violation. What you gate is
completeness of presentation: (a) a draft whose own-derived value overlaps
a registry-published quantity (probe_std `std.metrics(label)`) must show
the registry value alongside its own, with its population/definition
stated — "revise" if the registry number is absent, never for the rival
existing; (b) a draft aggregating over runs/experiments must cover the
full population in scope or state its exclusions — "revise" a silent
truncation or a "representative" sample standing in for a census. Two
reports of this corpus once shipped rival "median queue wait" values,
each ALONE, from different row populations (2026-08-02) — either number
alone is a subset; both together are information.
INFRASTRUCTURE-BLAME CLAIMS: a draft asserting the corpus, packs, or
instrumentation are limited/degenerate (constant arrays, missing fields,
"sampling limitation") must show the raw pack file read DIRECTLY
(json.load of the pack, the exact field printed) — a probe printing
constants or zeros is first evidence about the probe, not the corpus.
"Revise" any such claim whose evidence is the reviewer's own extraction
rather than the raw file: one report charted 400 copies of one number
twice and blamed a nonexistent "preserved sampling limitation" while the
pack held 400 distinct values (2026-08-02).
PROGRESSION SERIES: attempt-spread / best-so-far / improvement-ladder
charts must carry the pack trajectory (probe_std ``std.trajectory``) or
referee per-claim values, and the draft must show the probe that printed
them. "Revise" any progression whose values come from a hand-rolled join:
one report charted 0.07-0.16 as "self-reported RMSE" where the pack
trajectory says 0.022-0.024, and built its band and ladders on the
garbled series (2026-08-02).
REFEREE ATTRIBUTIONS: anything presented as the referee's selection —
winner, champion experiment name, recomputed value — must match
referee.json verbatim; a reviewer's own re-scoring join labeled
"referee-selected" is a fabricated attribution ("revise", require the
verbatim referee selection shown next to the reviewer's own). One report
shipped 6 of 8 champion names that contradicted referee.json under a
"referee-selected" column header (2026-08-02)."""


def _one_call(provider, model: str, system: str, user_text: str,
              reasoning_effort: str, retries: int = 4,
              usage_sink=None) -> str:
    """One non-tool LLM call via the provider protocol, with bounded retries.

    ``usage_sink(response)`` is called on completion so the session can add
    the call to its own production-cost ledger (every report ends with its
    bill; user order 2026-08-11)."""
    for attempt in range(retries):
        try:
            history = provider.build_user_items(user_text)
            text = ""
            for event in provider.stream_response(
                model=model, system=system, history=history, tools=[],
                reasoning_effort=reasoning_effort,
            ):
                if event.type == "text_delta":
                    text += event.delta
                elif event.type == "done":
                    if usage_sink is not None and event.response is not None:
                        usage_sink(event.response)
                    if not text:
                        text = event.response.text or ""
            return text
        except Exception as exc:  # noqa: BLE001 — retried
            time.sleep(10 * (attempt + 1))
            last = exc
    raise RuntimeError(f"role call failed after {retries} attempts: {last}")


class TeamSession(InvestigatorSession):
    """InvestigatorSession whose record_finding is critic-gated."""

    def __init__(self, corpus_path: Path, packs_dir: Path, out_dir: Path,
                 critic=None):
        super().__init__(corpus_path, packs_dir, out_dir)
        self._critic = critic  # None → behave exactly like the base class
        self._critique_counts: dict[str, int] = {}
        self._total_revisions = 0
        self._critic_log = open(self.out_dir / "critic_log.jsonl", "a")
        self._questions: dict[int, dict] = {}
        self._question_seq = 0

    # ---- open-questions ledger ------------------------------------------

    def seed_questions(self, texts: list[str]) -> None:
        for t in texts:
            self.add_question(text=t)

    def add_question(self, text: str) -> str:
        self._question_seq += 1
        self._questions[self._question_seq] = {
            "id": self._question_seq, "text": str(text)[:600],
            "status": "open", "note": ""}
        self._write_ledger()
        return f"question #{self._question_seq} registered"

    def list_questions(self) -> str:
        if not self._questions:
            return "(ledger empty)"
        return "\n".join(
            f"#{q['id']} [{q['status']}] {q['text']}"
            + (f" — {q['note']}" if q["note"] else "")
            for q in self._questions.values())

    def resolve_question(self, question_id: int, resolution: str,
                         finding_ids: list | None = None) -> str:
        q = self._questions.get(int(question_id))
        if q is None:
            return f"[ERROR] no question #{question_id}"
        known = set(range(1, self.findings + 1))
        bad = [f for f in (finding_ids or []) if int(f) not in known]
        if bad:
            return f"[ERROR] finding ids {bad} do not exist (recorded: {sorted(known)})"
        q["status"] = "resolved"
        q["note"] = (f"findings {sorted(int(f) for f in (finding_ids or []))}: "
                     if finding_ids else "") + str(resolution)[:500]
        self._write_ledger()
        return f"question #{question_id} resolved"

    def mark_unresolvable(self, question_id: int, reason: str) -> str:
        q = self._questions.get(int(question_id))
        if q is None:
            return f"[ERROR] no question #{question_id}"
        if self._critic is not None:
            verdict_raw = _one_call(
                self._critic["provider"], self._critic["model"], CRITIC_SYSTEM,
                "A question in an evidence investigation is being declared "
                "UNRESOLVABLE. Accept only if the reason shows the evidence "
                "genuinely cannot exist in the preserved artifacts (not that "
                "it was merely hard to find). STRICT JSON verdict as usual.\n\n"
                f"## Question\n{q['text']}\n\n## Claimed reason\n{reason}",
                self._critic["reasoning_effort"],
                usage_sink=lambda r: self.note_own_usage("critic", self._critic["model"], self._critic["provider"], r))
            try:
                verdict = json.loads(
                    verdict_raw[verdict_raw.index("{"):verdict_raw.rindex("}") + 1])
            except (ValueError, json.JSONDecodeError):
                verdict = {"verdict": "accept", "reasons": "unparseable verdict"}
            self._log_critic({"ts": time.time(), "unresolvable": q["id"],
                              "reason": reason, "verdict": verdict})
            if str(verdict.get("verdict", "")).lower() != "accept":
                return ("[CRITIC] not accepted as unresolvable: "
                        + str(verdict.get("reasons", ""))[:300]
                        + " — keep digging or resolve it.")
        q["status"] = "unresolvable"
        q["note"] = str(reason)[:500]
        self._write_ledger()
        return f"question #{question_id} marked unresolvable"

    def _write_ledger(self) -> None:
        (self.out_dir / "questions.json").write_text(
            json.dumps(list(self._questions.values()), indent=1))

    # The report critic keeps sending a draft back while the critic still says
    # revise AND there is budget to revise with. A flat two-bounce cap was the
    # first attempt and it failed exactly as a fail-open gate does: one review
    # published with eight stub sections after the critic had named three of
    # them twice, while 880 of its 1,114 iterations were still unspent
    # (2026-08-10). So the floor stays at 2, and beyond that the gate holds as
    # long as iterations remain and the report phase is inside its wall-clock
    # budget. Two guards keep it finite: the executor's own iteration budget,
    # which the loop reports into `iterations_left`, and REPORT_PHASE_SECONDS.
    MAX_REPORT_REVISIONS = 2          # floor: always allowed this many
    REPORT_PHASE_SECONDS = 2 * 60 * 60
    REPORT_REVISION_RESERVE = 12      # iterations kept back per further bounce

    def write_report(self, markdown: str) -> str:
        open_qs = [q for q in self._questions.values() if q["status"] == "open"]
        if open_qs:
            return ("[ERROR] report refused: " + str(len(open_qs))
                    + " ledger questions still open: "
                    + "; ".join(f"#{q['id']} {q['text'][:80]}" for q in open_qs[:6])
                    + " — resolve_question or mark_unresolvable each first.")
        bounce = self._report_review(markdown)
        if bounce:
            return bounce
        # the gate has stopped bouncing: publish the best-scored draft, which
        # may be an earlier round rather than this submission
        markdown = self._pick_publishable(markdown)
        self._write_ledger()
        markdown = self._chart_editor_pass(markdown)
        return super().write_report(markdown)

    # ---------------------------------------------------------- report review
    def _mission_sections(self) -> list[str]:
        """The numbered section themes the mission demands, e.g. 'Cost'.

        The generated missions list them as `N. **Theme** — ...`; a hand-written
        mission that does not use that form yields an empty list and the
        deterministic stub check is simply skipped.
        """
        mission = (self._critic or {}).get("mission") or ""
        return re.findall(r"^\s*\d+\.\s+\*\*(.+?)\*\*", mission, re.MULTILINE)

    @staticmethod
    def _match_section(theme: str, headings: list[str]) -> str | None:
        """The draft heading that answers a mission theme, or None.

        Deliberately loose. Writers retitle: a mission theme "Treatment
        (reasoning replay)" appears as "7. Treatment: reasoning replay",
        "Behavior provenance" as "Behaviour provenance", "Measurement hygiene
        and anomalies" as "Measurement hygiene and the numerical-oddities
        registry". An exact-substring test called all three missing and would
        have sent three good reports back for nothing (measured on the six
        published reports, 2026-08-10).
        """
        import difflib

        def toks(text: str) -> set[str]:
            words = re.findall(r"[a-z]{3,}", text.lower())
            stop = {"and", "the", "per", "with", "for", "its", "from", "what",
                    "each", "into", "not", "one", "how", "why", "them"}
            return {w.rstrip("s") for w in words if w not in stop}

        want = toks(theme)
        best, best_score = None, 0.0
        for h in headings:
            hv = toks(re.sub(r"^\d+[.)]\s*", "", h))
            overlap = len(want & hv) / max(1, len(want))
            fuzzy = difflib.SequenceMatcher(None, theme.lower(), h.lower()).ratio()
            score = max(overlap, fuzzy)
            if score > best_score:
                best, best_score = h, score
        return best if best_score >= 0.5 else None

    @classmethod
    def _stub_sections(cls, markdown: str, themes: list[str]) -> list[str]:
        """Mission themes whose section in the draft is a placeholder.

        A section counts as delivered if it carries either real bulk or real
        citations; the pair matters because a short section that quotes three
        probes is an answer, while 500 characters of uncited one-liners per
        task is what the mission forbids ("compressed into a table row").

        A theme with no matching heading is reported as *possibly* missing and
        left to the critic — this check never asserts absence on its own, because
        heading wording is the writer's choice.
        """
        sections: dict[str, str] = {}
        cur = ""
        for line in markdown.splitlines():
            if line.startswith("## "):
                cur = line[3:].strip()
                sections[cur] = ""
            elif cur:
                sections[cur] += line + "\n"
        stubs = []
        for theme in themes:
            head = cls._match_section(theme, list(sections))
            if head is None:
                # Absence is NOT asserted here. Mission themes range from one
                # word ("Harness") to a whole sentence, and writers retitle
                # freely; on the nine published reports this branch produced
                # false "missing section" flags for two reports that answer
                # every theme. The critic reads the mission and the full draft
                # and decides absence; this check only measures what it can see.
                continue
            body = sections[head]
            cites = len(re.findall(r"probe_\d+|\[Finding\s*\d+|Finding\s+\d+",
                                   body, re.IGNORECASE))
            if len(body) < 900 and cites < 2:
                stubs.append(f"{theme} (section {head!r}): {len(body)} chars, "
                             f"{cites} evidence references — a stub, not an answer")
        return stubs

    def _may_bounce_again(self) -> bool:
        """Past the floor, is there budget left to spend on another revision?

        Two conditions, both required: iterations remain (the loop keeps
        `iterations_left` current; when it is unknown the answer is no, so an
        un-wired caller can never loop forever), and the report phase is still
        inside its wall-clock budget measured from the first review.
        """
        left = getattr(self, "iterations_left", None)
        if not isinstance(left, int) or left <= self.REPORT_REVISION_RESERVE:
            return False
        started = getattr(self, "_report_phase_start", None)
        if started and (time.time() - started) > self.REPORT_PHASE_SECONDS:
            self._log_critic({"ts": time.time(),
                              "report_review": "time budget reached — publishing",
                              "elapsed_s": round(time.time() - started),
                              "rounds": getattr(self, "_report_reviews", 0)})
            return False
        return True

    def _draft_defects(self, markdown: str) -> int:
        """Deterministic hard violations in a draft, counted at review time.

        The critic's score is judgment; these are facts: registry echo,
        progression-series fidelity, referee attribution, chart grammar.
        Counted per candidate so publication can rank by them — a stored
        round with zero violations must never lose to a one-point-higher
        round carrying four, which is what a score-only selector does.
        Any audit that cannot run counts nothing: a broken audit must not
        cost the campaign its report.
        """
        from alpha_lab.benchmarks.runcmp.render_html import chart_errors
        n = 0
        for check in (self._registry_echo_problems,
                      self._progression_series_problems,
                      self._referee_attribution_problems,
                      chart_errors):
            try:
                n += len(check(markdown))
            except Exception:  # noqa: BLE001 — audit failure ≠ draft defect
                pass
        return n

    def _keep_draft(self, markdown: str, verdict: str, reasons: str,
                    score: float | None = None, defects: int = 0) -> None:
        """Store every submitted draft with the verdict it earned.

        Kept so the terminal round has candidates to choose between. Nothing
        here compares LENGTH: a longer draft is not a better one, and a rule
        that published the biggest would just reward padding. The ranking
        used is fewest hard DEFECTS, then the critic's own SCORE (see
        _candidate_rank).
        """
        base = getattr(self, "out_dir", None) or getattr(self, "_run_dir", None)
        if base is None:          # nothing to store into: keep no candidates
            return
        d = Path(base) / "report_drafts"
        d.mkdir(parents=True, exist_ok=True)
        n = getattr(self, "_report_reviews", 0)
        path = d / f"round_{n:02d}.md"
        path.write_text(markdown)
        cands = getattr(self, "_report_candidates", [])
        cands.append({"round": n, "path": str(path), "verdict": verdict,
                      "score": score, "defects": defects,
                      "reasons": reasons[:500]})
        self._report_candidates = cands

    def _pick_publishable(self, current: str) -> str:
        """Which draft ships when the gate stops bouncing.

        Ranking: fewest hard defects first, then the critic's score, then
        the later round (it had the earlier feedback). Score alone is not
        enough: a writer that fixed four referee-attribution violations in
        round 4 (scored 88) must not have the stale round 3 (scored 89)
        reintroduced over it — the deterministic audits downstream would
        bounce or banner the very defect the writer already removed. Size
        is never consulted: a longer draft is not a better one, and a
        biggest-wins rule would reward padding.

        What this also makes impossible is a failure that actually shipped:
        one review's critic saw drafts of 63,934 and 59,582 characters, and a
        third submission published unreviewed at 22,154 because the cap had
        fired (2026-08-10). An unscored draft cannot win by arriving last.
        """
        cands = [c for c in getattr(self, "_report_candidates", [])
                 if isinstance(c.get("score"), (int, float))]
        if not cands:
            return current
        best = max(cands, key=_candidate_rank)
        current_round = getattr(self, "_report_reviews", 0)
        if best["round"] >= current_round:
            return current
        self._log_critic({
            "ts": time.time(),
            "publish_choice": "fewest defects, then best score",
            "round": best["round"], "score": best["score"],
            "defects": best.get("defects", 0),
            "current_round": current_round,
            "scores": {c["round"]: c["score"] for c in cands},
            "defect_counts": {c["round"]: c.get("defects", 0) for c in cands}})
        return Path(best["path"]).read_text()

    def _report_review(self, markdown: str) -> str | None:
        """Critic pass over the whole draft; returns tool output to bounce it.

        Returns None to let the draft through. Any failure of this machinery
        lets the draft through too: a review that cannot run must never cost
        the campaign its report.
        """
        if self._critic is None:
            return None
        done = getattr(self, "_report_reviews", 0)
        if done >= self.MAX_REPORT_REVISIONS and not self._may_bounce_again():
            return None
        stubs = self._stub_sections(markdown, self._mission_sections())
        try:
            raw = _one_call(
                self._critic["provider"], self._critic["model"],
                REPORT_CRITIC_SYSTEM,
                "## Mission\n" + ((self._critic.get("mission") or "")[:60_000])
                + ("\n\n## Sections a deterministic check already flags as stubs\n"
                   + "\n".join("- " + s for s in stubs) if stubs else "")
                + "\n\n## The draft report\n" + markdown[:400_000],
                self._critic["reasoning_effort"],
                usage_sink=lambda r: self.note_own_usage("report critic", self._critic["model"], self._critic["provider"], r))
            verdict = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        except Exception as exc:  # noqa: BLE001
            self._log_critic({"ts": time.time(), "report_review": "call failed",
                              "error": str(exc)[:300], "stub_sections": stubs})
            return None
        self._report_reviews = done + 1
        if not getattr(self, "_report_phase_start", None):
            self._report_phase_start = time.time()
        try:
            score = float(verdict.get("score"))
        except (TypeError, ValueError):
            score = None
        defects = self._draft_defects(markdown)
        self._keep_draft(markdown, str(verdict.get("verdict", "")).lower(),
                         str(verdict.get("reasons", "")), score, defects)
        self._log_critic({"ts": time.time(), "report_review": verdict.get("verdict"),
                          "round": self._report_reviews, "score": score,
                          "defects": defects,
                          "stub_sections": stubs,
                          "reasons": str(verdict.get("reasons"))[:2000],
                          "required_changes": str(verdict.get("required_changes"))[:4000],
                          "draft_chars": len(markdown)})
        if str(verdict.get("verdict", "accept")).lower() != "revise":
            return None
        return ("[REPORT SENT BACK BY THE CRITIC "
                f"{self._report_reviews}/{self.MAX_REPORT_REVISIONS}] "
                + str(verdict.get("reasons", ""))[:1500]
                + "\nRequired changes:\n"
                + str(verdict.get("required_changes", ""))[:4000]
                + ("\nDeterministic stub check:\n"
                   + "\n".join("- " + s for s in stubs) if stubs else "")
                + (f"\nDeterministic audits count {defects} hard violation(s) "
                   "in this draft (registry echo / progression series / "
                   "referee attribution / chart grammar). Publication ranks "
                   "drafts by FEWEST violations before score — fix these "
                   "first, or a lower-scored clean draft will beat this one."
                   if defects else "")
                + "\n\nRevise and resubmit the FULL report through write_report: "
                  "keep every existing section and number, and expand what is "
                  "named above from evidence you already gathered. Writing "
                  "anywhere other than write_report does not count.")

    # Visualization meta-pass (user mandate 2026-08-01): the gating model
    # reads the accepted draft and improves its pictorial presentation —
    # converting number-dense tables to the right chart forms, diversifying
    # distribution displays, adding marginals — under a mechanically enforced
    # constraint: it may only re-present numbers already in the draft.
    _EDIT_NUM_RE = _num_re = None  # set lazily below

    def _chart_editor_pass(self, markdown: str) -> str:
        if self._critic is None or getattr(self, "_chart_edited", False):
            return markdown
        # Full-rewrite safety bound: the editing model's 64k-token output
        # ceiling comfortably carries ~200k chars; the old 70k skip made
        # the editor silently bypass exactly the deep reports that need
        # chart polish most (2026-08-08).
        if len(markdown) > 200_000:
            return markdown
        self._chart_edited = True
        import re as _re2
        from alpha_lab.benchmarks.runcmp.investigate import (
            _chart_census, CHART_SYNTAX)
        from alpha_lab.benchmarks.runcmp.render_html import (
            _CHART_TYPES, chart_errors)
        before = _chart_census(markdown)
        # The editor used to be told to produce forms by name ("band") without
        # being given the renderer's type table, so it emitted `type: band` and
        # `type: bar`; one unrenderable block discarded the whole improved draft
        # (2026-08-10: five reports lost the pass, every other check passing).
        # It now gets the same grammar the writer gets, and the legal spellings
        # enumerated from the renderer itself.
        grammar = (
            "\n\nThe ONLY legal values of `type:` are: "
            + ", ".join(sorted(_CHART_TYPES))
            + ". `band` is NOT a type — it is a row inside a `line` chart "
              "(`band NAME: x,lo,mid,hi; ...`), and a marginal scatter is "
              "`type: scatter` with `marginals: true`. Use those two for the "
              "distribution-form requirement, never a `type:` of your own "
              "invention.\n\n" + CHART_SYNTAX)
        editor_system = (
                "You are the visualization editor for an evidence report. "
                "Improve how its numerical data is presented pictorially: "
                "convert number-dense tables or prose into fenced ```chart "
                "blocks where a chart carries the point better; upgrade "
                "wrongly-chosen forms (ranked list -> bars, evolution -> "
                "line/step, composition -> stacked, two quantities -> "
                "scatter with marginals: true at 10+ points, distributions "
                "-> hist/density/box/band); diversify distribution forms — "
                "the revised report must contain AT LEAST FOUR distinct "
                "distribution forms among hist / density / box / band / "
                "scatter-with-marginals, built from values already present. "
                "Maximize usefulness, diversity, appropriateness. HARD "
                "RULES: every numeric value in any chart you add or change "
                "must appear verbatim elsewhere in the draft; never invent, "
                "re-derive, or round numbers; never change prose claims, "
                "findings, verdicts, tables, or provenance lines; keep all "
                "existing content. Return ONLY the complete revised "
                "markdown." + grammar)

        # Two attempts. An unrenderable block is a mistake the editor is told
        # to fix — with the renderer's own error text and the legal type list —
        # rather than something this pipeline silently repairs or silently
        # throws the whole improved draft away for (2026-08-10: five reports
        # lost their chart pass to one bad `type:`, cause unlogged).
        edited = ""
        prev_errors: list[str] = []   # attempt 2 quotes attempt 1's failures
        for attempt in (1, 2):
            ask = markdown if attempt == 1 else (
                markdown + "\n\n<!-- Your previous revision was rejected: "
                + "; ".join(prev_errors[:6])
                + ". Fix exactly those blocks — use only the legal `type:` "
                  "values listed in your instructions — and return the full "
                  "revised markdown again. -->")
            try:
                edited_raw = _one_call(
                    self._critic["provider"], self._critic["model"],
                    editor_system, ask, self._critic["reasoning_effort"],
                    usage_sink=lambda r: self.note_own_usage("chart editor", self._critic["model"], self._critic["provider"], r))
            except Exception as exc:
                self._log_critic({"ts": time.time(),
                                  "chart_editor": "call failed",
                                  "attempt": attempt, "error": str(exc)[:300]})
                return markdown
            edited = edited_raw.strip()
            if edited.startswith("```markdown"):
                edited = edited[len("```markdown"):].strip()
                if edited.endswith("```"):
                    edited = edited[:-3].strip()
            prev_errors = chart_errors(edited)
            if not prev_errors:
                break
            self._log_critic({"ts": time.time(), "chart_editor": "retrying",
                              "attempt": attempt,
                              "render_errors": prev_errors[:5]})
        # A chart series like "101,103,105" is three values, but a naive
        # thousands-separator regex reads it as one token "101103105" and
        # brands every legitimate editor chart as fabrication (caught by the
        # harness test before production). A token is covered if its joined
        # form OR every >=3-significant-char comma-part exists in the draft.
        num_re = _re2.compile(r"\d[\d,]*\.?\d*")
        def _norm(s: str) -> str:
            # A trailing period is sentence punctuation, not a decimal
            # (live test: "21,081." produced a phantom part "081.").
            return s.replace(",", "").rstrip(".")
        def _sig(s: str) -> bool:
            return len(s.replace(".", "")) >= 3
        def _forms(text: str) -> tuple[set[str], set[str]]:
            joined, parts = set(), set()
            for v in num_re.findall(text):
                j = _norm(v)
                if _sig(j):
                    joined.add(j)
                for p in v.split(","):
                    p = p.rstrip(".")
                    if _sig(p):
                        parts.add(p)
            return joined, parts
        dj, dp = _forms(markdown)
        new_numbers = set()
        for v in num_re.findall(edited):
            j = _norm(v)
            if not _sig(j):
                continue
            covered = j in dj or all((not _sig(p.rstrip("."))) or
                                     p.rstrip(".") in dp
                                     for p in v.split(","))
            if not covered:
                new_numbers.add(j)
        after = _chart_census(edited)
        # The editor's output must RENDER: an injected block the renderer
        # rejects ships as raw fence text (measured 2026-08-07: the editor
        # added 6 blocks to an accepted draft; dead bars/spans reached the
        # published page).
        render_errors = chart_errors(edited)
        ok = (not new_numbers
              and len(edited) >= 0.9 * len(markdown)
              and after["total"] >= before["total"]
              and after["dist_forms"] >= before["dist_forms"]
              and not render_errors)
        # Every discard used to be logged as the bare word "discarded", so a
        # pass that failed on ONE bad chart block looked identical to one that
        # fabricated numbers (2026-08-10: five reports, cause invisible).
        why = []
        if new_numbers: why.append("new numbers")
        if len(edited) < 0.9 * len(markdown): why.append("shorter than 90%")
        if after["total"] < before["total"]: why.append("fewer charts")
        if after["dist_forms"] < before["dist_forms"]: why.append("fewer distribution forms")
        if render_errors: why.append(f"{len(render_errors)} unrenderable block(s)")
        self._log_critic({
            "ts": time.time(), "chart_editor": "applied" if ok else "discarded",
            "discard_reasons": why, "render_errors": render_errors[:5],
            "census_before": before, "census_after": after,
            "new_numbers": sorted(new_numbers)[:20],
            "len_before": len(markdown), "len_after": len(edited)})
        return edited if ok else markdown

    def _log_critic(self, payload: dict) -> None:
        self._critic_log.write(json.dumps(payload, default=str)[:40_000] + "\n")
        self._critic_log.flush()

    def _cited_probe_outputs(self, pending: dict, per_probe: int = 6000,
                             total: int = 24000) -> str:
        """Bounded full outputs of every probe the draft cites, so the critic
        can point at support the executor forgot to quote instead of ruling
        the point untested (2026-07-31: a report shipped 'not expanded here'
        cells whose values sat verbatim in a cited probe's output)."""
        names: list[str] = []
        for ref in pending.get("evidence") or []:
            name = str((ref or {}).get("probe") or "")
            if name and name not in names:
                names.append(name)
        chunks: list[str] = []
        used = 0
        for name in names:
            path = self.out_dir / "probes" / f"{name}.out"
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            snippet = text[:per_probe]
            if used + len(snippet) > total:
                snippet = snippet[: max(0, total - used)]
            if not snippet:
                break
            used += len(snippet)
            chunks.append(f"### {name}.out (first {len(snippet)} chars)\n"
                          + snippet)
        if not chunks:
            return ""
        return ("\n\n## Cited probes' output (bounded; use for "
                "revise-with-pointer, never as a reason to reject)\n"
                + "\n\n".join(chunks))

    def record_finding(self, **kwargs) -> str:
        if self._critic is None:
            return super().record_finding(**kwargs)
        # Deterministic reference validation first — same contract as the
        # single-agent path; the critic only ever sees valid drafts.
        claim_key = hashlib.sha1(
            str(kwargs.get("claim", ""))[:400].encode()).hexdigest()[:12]
        base_result = super().record_finding(**kwargs)
        if base_result.startswith("[ERROR]"):
            return base_result
        # The base class appended the finding; pull it back out pending verdict.
        lines = self.findings_path.read_text().splitlines()
        pending = json.loads(lines[-1])
        self.findings_path.write_text("\n".join(lines[:-1])
                                      + ("\n" if len(lines) > 1 else ""))
        self.findings -= 1

        verdict_raw = _one_call(
            self._critic["provider"], self._critic["model"], CRITIC_SYSTEM,
            "## Mission\n" + self._critic["mission"][:20000]
            + "\n\n## Draft finding (references already string-verified)\n"
            + json.dumps({k: v for k, v in pending.items()
                          if k not in ("id", "ts")}, indent=1, default=str)[:14000]
            + self._cited_probe_outputs(pending),
            self._critic["reasoning_effort"],
            usage_sink=lambda r: self.note_own_usage(
                "critic", self._critic["model"], self._critic["provider"], r),
        )
        try:
            start = verdict_raw.index("{")
            end = verdict_raw.rindex("}") + 1
            verdict = json.loads(verdict_raw[start:end])
        except (ValueError, json.JSONDecodeError):
            verdict = {"verdict": "accept",
                       "reasons": "critic returned unparseable verdict; "
                                  "fail-open to not lose validated evidence",
                       "required_changes": ""}
        self._log_critic({"ts": time.time(), "claim_key": claim_key,
                          "finding": pending, "verdict": verdict})

        v = str(verdict.get("verdict", "accept")).lower()
        if v == "accept":
            self.findings += 1
            pending["id"] = self.findings
            pending["critic_verdict"] = verdict
            with open(self.findings_path, "a") as fh:
                fh.write(json.dumps(pending, default=str) + "\n")
            return (f"finding #{self.findings} recorded "
                    f"(references verified; critic accepted: "
                    f"{str(verdict.get('reasons', ''))[:200]})")
        n = self._critique_counts.get(claim_key, 0) + 1
        self._critique_counts[claim_key] = n
        self._total_revisions += 1
        # The churn ceiling scales with the mission: a 15-family ledger
        # produces ~3x the findings of the 4-family missions this constant
        # was calibrated on (2026-08-09, same shape as the budget freeze).
        revisions_ceiling = (MAX_TOTAL_REVISIONS
                             + 2 * max(0, len(self._questions) - 4))
        if self._total_revisions > revisions_ceiling and v == "revise":
            # Churn budget exhausted: downgrade revise to reject so the
            # executor moves on instead of orbiting the critic.
            v = "reject"
        if v == "reject" or n >= MAX_CRITIQUES_PER_CLAIM:
            return ("[CRITIC] finding NOT recorded (this claim is closed): "
                    f"{str(verdict.get('reasons', ''))[:300]} — do not "
                    "re-submit this text. The underlying QUESTION stays open: "
                    "if the fact matters, build a NARROWER claim on different "
                    "or stronger evidence and record that instead.")
        return ("[CRITIC] finding NOT recorded — revise and re-record. "
                f"Reasons: {str(verdict.get('reasons', ''))[:300]} "
                f"Required changes: {str(verdict.get('required_changes', ''))[:400]} "
                "Re-submit THIS claim as your next action: add the missing "
                "quotes (re-print the lines with a probe if needed) and call "
                "record_finding again with the same claim text. A revised "
                "re-submission is far cheaper than drafting a new claim — "
                "do not move on and abandon a finding the critic called "
                "directionally right.")


def run_team_investigation(
    corpus_path: Path,
    packs_dir: Path,
    out_dir: Path,
    roles: dict[str, dict | None],
    mission: str = "",
) -> bool:
    from alpha_lab.client import get_provider

    out_dir.mkdir(parents=True, exist_ok=True)
    executor = roles["executor"]
    assert executor, "executor role is required"
    exec_provider = get_provider(executor["provider"])
    executor = {**executor, "provider": exec_provider}

    critic = roles.get("critic")
    if critic:
        critic = {**critic, "provider": get_provider(critic["provider"]),
                  "mission": mission}

    # the session exists before the planner call so the planner's own
    # tokens land in the production-cost ledger like every other seat's
    session = TeamSession(corpus_path, packs_dir, out_dir, critic=critic)
    plan_text = ""
    planner = roles.get("planner")
    if planner:
        planner = {**planner, "provider": get_provider(planner["provider"])}
        try:
            tables = session.dispatch("get_tables", {})
        except Exception:  # noqa: BLE001
            tables = ""
        plan_text = _one_call(
            planner["provider"], planner["model"], PLANNER_SYSTEM,
            "## Mission\n" + mission + "\n\n## Deterministic tables\n"
            # 20k hid 94% of a 75-run corpus's tables from the planner
            # (2026-08-08); 120k fits comfortably in the planner context.
            + str(tables)[:120_000],
            planner["reasoning_effort"],
            usage_sink=lambda r: session.note_own_usage(
                "planner", planner["model"], planner["provider"], r),
        )
        (out_dir / "plan.md").write_text(plan_text)
    # Seed the ledger from the mission's question headings (## Q1 — ...).
    seeds = [ln.lstrip("# ").strip() for ln in mission.splitlines()
             if ln.startswith("## Q")]
    session.seed_questions(seeds)
    provider = executor["provider"]
    model = executor["model"]
    reasoning_effort = executor["reasoning_effort"]
    history: list[dict] = []
    transcript = open(out_dir / "investigator_log.jsonl", "a")

    def log(kind: str, payload) -> None:
        transcript.write(json.dumps({"ts": time.time(), kind: payload},
                                    default=str)[:40_000] + "\n")
        transcript.flush()

    opening = "Begin. Start from list_runs and get_tables, then investigate."
    if mission:
        opening += f"\n\n## Mission scope for this investigation\n{mission}"
    if plan_text:
        opening += ("\n\n## Evidence plan from the planning stage (follow "
                    "unless the evidence contradicts it)\n" + plan_text)
    if critic:
        opening += ("\n\nNote: a critic gates record_finding. Rejections are "
                    "final for that claim; revisions state exactly what to "
                    "add. Narrow, fully-supported findings pass; volume does "
                    "not.")
    opening += ("\n\nAn open-questions ledger is enforced: the mission's "
                "questions are pre-registered; register the plan's extra "
                "questions with add_question; check list_questions before "
                "finishing — write_report refuses while any question is "
                "open. Resolve with citations, or mark_unresolvable with a "
                "concrete why-the-evidence-cannot-exist reason.")
    history.extend(provider.build_user_items(opening))

    bounce_refunds = 0
    # Corpus/mission-scaled budget (see constants above): a 2-run pair keeps
    # the historical 150; the 8-run corpus gets the head-room the teams
    # measurably lacked. The budget is RECOMPUTED as the ledger grows: the
    # planner registers most questions via add_question after start, and a
    # start-frozen budget starved a 14-question union mission with a
    # 4-question allowance (measured 2026-08-08: every big-mission report
    # came back thin because compose was forced at the 4-question budget).
    n_runs = len(session.records)

    def _current_budget() -> int:
        n_questions = max(len(session._questions), 4)
        return (MAX_ITERATIONS
                + ITER_PER_EXTRA_RUN * max(0, n_runs - 2)
                + ITER_PER_EXTRA_QUESTION * (n_questions - 4))

    def _current_max_refunds() -> int:
        n_questions = max(len(session._questions), 4)
        return (MAX_BOUNCE_REFUNDS
                + REFUNDS_PER_EXTRA_RUN * max(0, n_runs - 2)
                + REFUNDS_PER_EXTRA_QUESTION * (n_questions - 4))

    budget = _current_budget()
    max_refunds = _current_max_refunds()
    print(f"iteration budget: {budget} (+<= {max_refunds} bounce refunds) "
          f"for {n_runs} runs / {max(len(session._questions), 4)} questions",
          flush=True)
    composing = False
    iteration = -1
    while True:
        new_budget = _current_budget()
        if new_budget != budget:
            print(f"iteration budget rescaled: {budget} -> {new_budget} "
                  f"({len(session._questions)} questions on the ledger)",
                  flush=True)
            budget = new_budget
        max_refunds = _current_max_refunds()
        if iteration >= budget - 1 + min(bounce_refunds, max_refunds):
            break
        iteration += 1
        response = None
        for attempt in range(4):
            try:
                for event in provider.stream_response(
                    model=model, system=SYSTEM_PROMPT, history=history,
                    tools=TOOLS + QUESTION_TOOLS,
                    reasoning_effort=reasoning_effort,
                ):
                    if event.type == "done":
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
        session.note_own_usage("executor", model, provider, response)
        provider.append_response_to_history(history, response)
        if response.text:
            log("text", response.text)
            print(f"[{iteration}] {response.text[:300]}", flush=True)
        if not response.tool_calls:
            history.extend(provider.build_user_items(
                "Continue with tool calls, or finish with write_report."))
            continue
        outputs = []
        for tc in response.tool_calls:
            args = {}
            try:
                args = json.loads(tc.arguments) if tc.arguments else {}
            except json.JSONDecodeError:
                pass
            result = session.dispatch(tc.name, args)
            if tc.name == "record_finding" and result.startswith("[CRITIC]"):
                bounce_refunds += 1
            log("tool", {"name": tc.name, "args": args, "result": result[:2000]})
            print(f"[{iteration}] {tc.name}({str(args)[:120]}) -> "
                  f"{len(result)} chars", flush=True)
            outputs.append({"call_id": tc.call_id,
                            "output": result[:MAX_TOOL_OUTPUT]})
        history.extend(provider.build_tool_result_items(outputs))
        _compact_history(history)
        remaining = (budget + min(bounce_refunds, max_refunds)
                     - iteration - 1)
        # the report gate spends this: it keeps bouncing a bad draft only while
        # there is budget left to revise with (user ruling 2026-08-10)
        session.iterations_left = remaining
        if remaining <= COMPOSE_RESERVE and not composing:
            # Hard stop on investigation: everything left is for composing
            # the real report. The forced flush below stays as last resort,
            # but a composed report with explicit unresolved rows must be
            # reachable from here by construction.
            composing = True
            open_qs = sum(1 for q in session._questions.values()
                          if q["status"] == "open")
            history.extend(provider.build_user_items(
                f"[budget — COMPOSE NOW] {remaining} iterations remain. STOP "
                "investigating: do not open new evidence threads. In order: "
                f"(1) close the {open_qs} open ledger questions "
                "(resolve_question with finding citations, or "
                "mark_unresolvable with the concrete missing-evidence "
                "reason); (2) call write_report with the FULL composed "
                "report, including every mandatory table the mission "
                "specifies — rows you could not resolve get an explicit "
                "'unresolved' entry with the confounder named. A finding "
                "dump without the mandatory tables fails the mission."))
        elif remaining in (10, 5):
            history.extend(provider.build_user_items(
                f"[budget] {remaining} iterations remain. write_report NOW "
                "with what you have — an unreported investigation scores "
                "zero."))
        if session.report_written:
            print(f"done: {session.findings} findings, report written")
            transcript.close()
            return True
    # Iteration cap reached without a report. First choice: a draft the critic
    # already scored — real composed work always beats an auto-assembled flush.
    if publish_best_draft(session):
        print("iteration cap: published the best-scored submitted draft")
        transcript.close()
        return True
    # No scored draft: force-flush one. A weak report beats a silent DNF —
    # the scoreboard rule "unreported scores zero" must be unreachable by
    # construction.
    open_qs = [q for q in session._questions.values() if q["status"] == "open"]
    findings = []
    if session.findings_path.is_file():
        findings = [json.loads(l) for l in
                    session.findings_path.read_text().splitlines() if l.strip()]
    lines = ["# Investigation report (FORCED FLUSH at iteration cap)", "",
             "The executor exhausted its iteration budget before writing a "
             "report. Everything below is auto-assembled from the recorded, "
             "reference-verified findings and the question ledger.", ""]
    for f in findings:
        lines.append(f"## Finding #{f.get('id')}")
        lines.append(str(f.get("claim", "")))
        lines.append("")
    if open_qs:
        lines.append("## UNRESOLVED questions (budget exhausted)")
        for q in open_qs:
            lines.append(f"- #{q['id']} {q['text']}")
        lines.append("")
    session._questions and session._write_ledger()
    # The flush has no charts; with a fresh refusal counter the chart floors
    # would REFUSE it and this path would print success while writing nothing.
    # Floors warn, never refuse, when no writer is left to revise.
    session._report_refusals = 3
    if getattr(session, "_best_draft", None) is None:
        session._best_draft = "\n".join(lines)
    InvestigatorSession.write_report(session, "\n".join(lines))
    print(f"forced report flush: {len(findings)} findings, "
          f"{len(open_qs)} unresolved questions")
    transcript.close()
    return True


def _candidate_rank(c: dict) -> tuple:
    """Publication order for stored drafts, used by every selector.

    Fewest hard defects dominates (they are facts, not judgment); the
    critic's score breaks ties; a later round wins equal score, since it
    had the earlier feedback.
    """
    return (-c.get("defects", 0), c["score"], c["round"])


def publish_best_draft(session) -> bool:
    """Terminal salvage: ship the best-scored draft already submitted.

    The report gate only publishes through a write_report call, so a writer
    that becomes unable to call it (context drowned, API dead, iteration cap)
    used to strand every scored draft on disk: one re-writing session ended
    with 23 critic-scored drafts — best 85/100 — and no report (2026-08-10).
    Both loops call this at exhaustion; it returns True iff a report shipped.
    """
    cands = [c for c in getattr(session, "_report_candidates", [])
             if isinstance(c.get("score"), (int, float))]
    if not cands:
        return False
    best = max(cands, key=_candidate_rank)
    md = Path(best["path"]).read_text()
    session._log_critic({
        "ts": time.time(), "publish_choice": "salvage at budget exhaustion",
        "round": best["round"], "score": best["score"],
        "defects": best.get("defects", 0),
        "scores": {c["round"]: c["score"] for c in cands},
        "defect_counts": {c["round"]: c.get("defects", 0) for c in cands}})
    md = session._chart_editor_pass(md)
    # Floors must warn, never refuse: there is no writer left to revise, so a
    # refusal here would mean no report at all — the outcome this exists to
    # make unreachable.
    session._report_refusals = 3
    session._best_draft = md
    session._questions and session._write_ledger()
    InvestigatorSession.write_report(session, md)
    return bool(getattr(session, "report_written", False))


def _parse_role(spec: str, default_provider: str, default_model: str,
                default_effort: str) -> dict | None:
    if spec == "none":
        return None
    if ":" in spec:
        prov, model = spec.split(":", 1)
    else:
        prov, model = default_provider, spec
    return {"provider": prov, "model": model,
            "reasoning_effort": default_effort}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Multi-role (multi-model) corpus investigation")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--packs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--provider", default="openai",
                    help="default provider for unassigned roles")
    ap.add_argument("--model", default="gpt-5.6-sol",
                    help="default model for unassigned roles")
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--role", action="append", default=[],
                    help="NAME=PROVIDER:MODEL or NAME=none; NAME in "
                         "planner|executor|critic (repeatable)")
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

    default = {"provider": args.provider, "model": args.model,
               "reasoning_effort": args.reasoning_effort}
    roles: dict[str, dict | None] = {
        "planner": dict(default), "executor": dict(default),
        "critic": dict(default),
    }
    for spec in args.role:
        name, _, value = spec.partition("=")
        if name not in roles:
            raise SystemExit(f"unknown role {name!r}")
        roles[name] = _parse_role(value, args.provider, args.model,
                                  args.reasoning_effort)

    from alpha_lab.benchmarks.runcmp.investigate import record_session
    executor = roles.get("executor") or {}
    record_session(args.out, stage="investigate-team",
                   provider=executor.get("provider"),
                   model=executor.get("model"),
                   reasoning_effort=executor.get("reasoning_effort"),
                   roles={n: (f"{r['provider']}:{r['model']}" if r else "none")
                          for n, r in roles.items()})
    ok = run_team_investigation(args.corpus, args.packs, args.out, roles,
                                mission)
    if ok:
        from alpha_lab.benchmarks.runcmp.mission import refresh_reports_index
        refresh_reports_index(args.out.parent)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
