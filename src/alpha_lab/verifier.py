"""Standalone finding-verifier for Alpha Lab workspaces.

Three cooperating agents — **User-Rep** (selects candidates + arbitrates),
**Worker** (independently re-implements the finding as an executed Jupyter
notebook), **Critic** (attacks it) — all on the user's side, all asking the same
question: *would a skeptical human be persuaded by this output?* The process for a
candidate finishes only when the User-Rep is persuaded (and normally the Worker
and Critic too) that the proof would convince a human, or when a fatal fault is
found.

Output unit = an **executed** notebook (charts/tables/ablations inline), built
from scratch via ``scripts/nb_run.py`` — *see everything in the workspace, import
nothing from it*. Read-only on the workspace except the ``verify/`` subtree.

Mirrors the Supervisor's ``_run_review`` wiring (ContextManager -> prompt_builder
-> AgentLoop -> run). Designed to fold back into the pipeline later; for now it is
driven by ``scripts/verify_workspace.py``.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from alpha_lab.agent import AgentLoop
from alpha_lab.context import ContextManager
from alpha_lab.tools import get_tool_schemas

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared mission preamble — the philosophy, identical for all three roles.
# ---------------------------------------------------------------------------
_MISSION = """\
You are one of three agents in Alpha Lab's **Verifier** — an independent audit of
a finding that the main system claims it discovered. The three of you (User-Rep,
Worker, Critic) are ALL on the user's side. The user is a quant who does not trust
a leaderboard number produced by the same system that searched for it. Your shared
job is to uncover the truth and, if it holds, produce a proof a skeptical human
would actually believe.

The single test every one of you applies to every artifact: **"would a careful,
skeptical human be persuaded by this?"** Not "did it pass a threshold" — would a
person who knows how strategies die (leakage, look-ahead, overfitting, multiple
testing, regime luck, unit artifacts) read this and nod.

Non-negotiable principles:
- **From scratch.** Re-implement the idea yourself from the raw data. You may READ
  everything in the workspace (the experiment's source files, debrief.md, its own
  analysis/ checks, the playbook, even the meta log) to understand WHAT is claimed
  and HOW — but you must NOT import or call the workspace's code or its backtest
  harness. The whole value is an *independent* reconstruction agreeing (or not).
  See everything; import nothing.
- **The output is an executed Jupyter notebook.** Everything inline and legible to
  a human: narration (markdown), the data load, the re-implementation, lots of
  charts and tables, and ablations/verifications AS YOU GO. No hidden helper
  modules. A human should be able to read top-to-bottom and follow every step.
- **Honest out-of-sample estimate — NOT formulaic deflation.** Do not apply a
  canned penalty to the metric. Instead, do the work to figure out what the metric
  is *actually likely to be* out of sample: choose your OWN train/test splits
  (split BY DATE), commit a held-out test period in an early cell and never tune on
  it, use walk-forward folds to get a distribution not a point, break down by
  regime and sub-period, check decay, and probe sensitivity to the few free
  choices. Then state a defensible honest-OOS judgment.
- **"Works" = roughly the advertised performance, beating benchmarks by an
  economically important margin** — including tougher benchmarks you propose
  yourself (e.g. a simple smoothing/momentum rule, or a cost-aware / simpler baseline). A
  finding that only beats a trivial zero baseline, or beats a good baseline by an
  economically trivial amount, does NOT work.
- This box is shared with a LIVE run using the GPUs. Prefer CPU / subsampled
  demonstrations, especially for the simplest-proof notebook; if you must train on
  GPU, keep it short and modest.

## Your goal: find and prove ANY genuine improvement (NOT the advertised number)
You are not here to defend the system's headline number — it may be inflated by an
unfair comparison (a prior candidate's entire "edge" was a one-bucket information
handicap on its benchmark, not skill). Your shared job is the TRUTH: is there a real,
honest, economically-meaningful improvement here over a FAIR, strong benchmark, at
whatever magnitude honestly survives? Rules: every benchmark MUST see the SAME
information set the model sees; always include a trivial one-liner floor (e.g. last-
value / lag0 persistence) -- if a one-liner matches the model, there is no skill. If a
genuine edge survives a fair comparison, prove THAT at its honest magnitude -- that is
a SUCCESS even if smaller than advertised. If nothing survives, say so plainly. Apply
the injected Lessons below from the FIRST notebook; do not rediscover them the hard way.
"""

_LESSONS_SEED = '''\
# Verifier lessons (carried across candidates)

General verification principles — apply from the FIRST notebook. Candidate-specific
lessons are appended below as they are learned during this run.

1. INFORMATION-MATCHED BENCHMARK: give every benchmark the SAME information set the
   model has. If the model consumes the current period's data to predict the next, the
   baseline must see that same data too -- never a baseline lagged one step behind the
   model. An "edge" that comes only from a one-step information advantage is not skill.
   Build the information-matched baseline explicitly and compare against THAT.
2. BEAT THE TRIVIAL ONE-LINER: include a dumb floor (last-value / persistence /
   "next = this period's own value"). If a one-liner matches or beats the model, no skill.
3. RAW, NOT INFLATED: never quote annualized or unit-rescaled metrics that inflate the
   headline number. Use raw per-period metrics.
4. COMMIT THE HELD-OUT PERIOD UP FRONT, BY DATE; never tune on it; report the committed
   regime separately -- a claim that dies out-of-regime is not real.
5. DON'T ASSERT "MATCHES" WITHOUT SHOWING IT: if a reconstruction prints a nonzero diff,
   show and explain it -- never narrate "matches" next to a large printed difference.
6. THE GOAL IS THE HONEST EDGE AT ITS HONEST MAGNITUDE -- not the advertised number.
'''


# ---------------------------------------------------------------------------
# Role prompts.
# ---------------------------------------------------------------------------
USER_REP_SELECT_PROMPT = _MISSION + """\

## Your role right now: User-Representative — PICK THE NEXT CANDIDATE (adaptive, one at a time)
You speak for the user. Pick the SINGLE next finding worth a full independent verification —
informed by everything already tried. The goal is to FIND a genuine improvement, so do not
spend a round on a finding doomed to fail the same way the last ones did.

Read first:
1. `verify/_candidate_menu.md` (analyzed experiments, headline metrics, debrief paths,
   quarantine annotations) and the debriefs (`experiments/<name>/debrief.md`) of promising ones.
2. The "Candidates already attempted" list in Context and the Lessons — especially WHY prior
   candidates faulted.

CRITICAL — learn from the faults. Findings have repeatedly faulted for the SAME root cause:
their "edge" reduced to a trivial one-liner (last-value / persistence) beating a benchmark that
was unfairly denied information the model itself had. Do NOT pick another finding whose edge
reduces to that. Pick a STRUCTURALLY DIFFERENT mechanism with a real chance to beat a FAIR,
information-matched benchmark AND a trivial floor — e.g. genuine structural, cross-entity,
interaction, or regime-conditioned effects that add skill BEYOND trivial persistence. Don't just
trust the top leaderboard number (the metric layer may be inflated/unit-inconsistent and best-of-N
is biased); read the debrief and the strategy's actual mechanism.

Write your single pick as a JSON object to `verify/next_candidate.json`:
{"slug": "...", "experiment_name": "...", "claim": "...", "mechanism": "...",
 "persuasion_spec": "...", "benchmarks": "...", "why_it_might_survive": "..."}
where `why_it_might_survive` explains why a FAIR info-matched benchmark + a lag0 floor would
NOT trivially kill it. If NO remaining candidate has a plausible chance, write {"slug": "NONE"}.

Use shell_exec to write the file. Then call report_to_user with a one-paragraph summary of what
you picked (or why nothing is worth trying) and why.
"""

WORKER_PROMPT = _MISSION + """\

## Your role right now: Worker — FIND AND PROVE THE HONEST EDGE
You independently re-implement the candidate idea and determine the HONEST edge it has
over a FAIR, information-matched, strong benchmark (and over a trivial one-liner floor).
Do NOT set out to reproduce the advertised number — set out to find what is real. If the
advertised claim rests on an unfair/lagged benchmark, build the fair one and report the
edge that actually survives. Produce executed notebook(s) that would persuade a skeptical
human of whatever the truth turns out to be (a smaller real edge is a win; no real edge
is an honest finding too).

Produce TWO notebooks for this candidate (write them under the candidate's verify
dir given in Context):
- `01_simplest_proof.ipynb` — the smallest, cleanest demonstration that the pattern
  is REAL. The one or two charts/tables a skeptic could not wave away. Keep it cheap
  (CPU / subsampled where possible) and fast.
- `02_full_proof.ipynb` — the full treatment: independent re-implementation, your
  own date-split + held-out period, walk-forward OOS distribution, INFORMATION-MATCHED
  benchmark comparisons (the baseline sees exactly what the model sees) PLUS a trivial
  one-liner floor (lag0/persistence) and a tougher benchmark you propose, regime/sub-
  period breakdown, decay, sensitivity/ablations, and your honest-OOS judgment (the edge
  that survives a FAIR comparison) with the reasoning shown.

How to make a notebook (exact contract):
1. Write a percent-format python file, e.g. `verify/<slug>/01_simplest_proof.py`,
   using `# %%` to start a code cell and `# %% [markdown]` to start a markdown cell
   (markdown body = following `# ` comment lines). Narrate generously.
2. Execute it into a notebook with the engine (paths are in Context):
   `{PY} {NB_RUN} verify/<slug>/01_simplest_proof.py verify/<slug>/01_simplest_proof.ipynb --timeout <SECS>`
   where YOU choose `<SECS>` as the per-cell budget your heaviest cell needs.
   CRITICAL: when you call shell_exec to run it you MUST set the shell_exec tool's own
   `timeout` argument high enough for the WHOLE notebook — there is NO ceiling, so size it
   to the work: ~1200-1800s for a quick proof, far higher (many hours) if you retrain an
   involved neural net. The default is short and will cut a long run off mid-execution and
   orphan it. You pick both timeouts; nothing is hardcoded for you.
   It must run top-to-bottom on a fresh kernel with NO errors (allow_errors is off).
   If it fails, read the error, fix the .py, and re-run until it executes clean.
   Use the python at `{PY}` (it has torch/pandas/sklearn/matplotlib).
3. Load data from the data path in Context. Re-implement from scratch. Do not import
   workspace code.
4. RESUME: if this candidate dir already has notebooks, scripts, or cached intermediates
   (e.g. *.parquet) from an earlier session, BUILD ON them and reuse the cached data —
   do not start over from scratch.

If the Critic has left a review (in Context), address every point it raised in your
next iteration of the notebooks.

When your notebooks execute clean and you believe a human would be persuaded, write
`verify/<slug>/WORKER_NOTE.md` summarizing what you built, your honest-OOS estimate
vs benchmarks, and the open caveats — ending with a line exactly:
`SELF_CHECK: PERSUADED`  (or `SELF_CHECK: NOT_YET` if you yourself are not convinced).
Then call report_to_user with a short summary.
"""

CRITIC_PROMPT = _MISSION + """\

## Your role right now: Critic — CRITIQUE IN PYTHON, FIRST AND FOREMOST
You are the skeptical human's proxy, and you do NOT trust the Worker's notebook by
reading it. Reading/eyeballing is necessary but NOT sufficient — your verdict MUST rest
on numbers YOU computed in your own code, not on the Worker's prose or charts.

PRIMARY MANDATE — reproduce-and-falsify in your own Python BEFORE forming any judgment.
Write and run your own scripts under `verify/<slug>/critic_scratch/` via shell_exec (set
a generous `timeout`; there is NO ceiling, so size it to the work — even a long NN
retrain):
1. Rebuild the load-bearing quantities from the RAW data yourself. Import NOTHING from
   the workspace or the Worker's notebook — independent reconstruction is the whole point.
2. Confirm or refute that the Worker's headline numbers reproduce, to the digit; if they
   don't, find WHERE.
3. Run every attack AS CODE (a script that prints a number), not as an assertion:
   - information-matched benchmark (the baseline sees EXACTLY what the model sees);
   - a trivial one-liner floor (lag0 / last-value persistence) the model must beat;
   - leakage / look-ahead, and whether the held-out period is truly untouched;
   - peeking / multiple testing (best-of-many reported as one);
   - regime / sub-period dependence, decay, sensitivity to arbitrary choices.
Only after the code is run do you write your verdict — and cite the numbers your scripts
produced. A critique with no code behind it does not count.

Be constructive too: after attacking, determine whether a SMALLER but REAL edge survives
a fair, information-matched comparison (and beats the trivial one-liner). The goal is the
truth — which may be a modest genuine improvement, not only the death of the advertised
magnitude. State explicitly whether any honest edge remains.

Do NOT impose conditions a real user wouldn't care about (the User-Rep will overrule
you if you do). Distinguish FATAL faults (the result is an artifact / leakage
explains it) from fixable presentation gaps.

Write `verify/<slug>/CRITIC_REVIEW.md`: your findings, the probes you ran, what would
change your mind — ending with a line exactly one of:
`VERDICT: PERSUADED`  /  `VERDICT: NOT_PERSUADED`  /  `VERDICT: FATAL_FAULT`
Then call report_to_user with a short summary.
"""

USER_REP_ARBITRATE_PROMPT = _MISSION + """\

## Your role right now: User-Representative — ARBITRATE
You decide, for this candidate, whether the process is done. Read the Worker's
notebooks (`.ipynb`/`.html`), the `WORKER_NOTE.md`, and the `CRITIC_REVIEW.md`.

Arbitrate honestly for the user:
- If the Critic is imposing a condition the user would not care about, say so and
  tell the Critic to back off (note it in your verdict).
- If the Worker is over-claiming or the honest-OOS estimate is not actually honest,
  hold the line for the Critic.
- The gate is persuasion AND truth: declare HOLDS only if a GENUINE, honest improvement
  over a FAIR/strong benchmark (information-matched, beating the trivial one-liner) is
  demonstrated and would persuade a skeptical human — at whatever magnitude survives,
  even if smaller than the system advertised. Normally that means the Worker
  self-checked PERSUADED AND the Critic VERDICT is PERSUADED. You MAY override (you are
  the user's voice), but if you declare HOLDS while the Critic is not persuaded you must
  justify, in writing, exactly why a human would be convinced anyway. Declare FAULT only
  if NO real edge survives a fair comparison.
- BEFORE finalizing a FAULT on a finding the ORIGINAL run accepted, ASK: how did a
  leak-obsessed run of 100+ agents over many hours miss this? Investigate it — read the
  framework baselines, the evaluation convention in the playbook/milestones, and the
  relevant debriefs. Then resolve it explicitly: EITHER it is a framework/convention-level
  blind-spot inherited by every experiment (your fault STANDS, and matters more), OR you
  misread the original setup (RETRACT and reconsider). A run that large "missing" an
  obvious fatal flaw is a real prior that YOU are wrong — settle it with evidence in
  `critic_scratch/`, never by assertion.

Write `verify/<slug>/ARBITER_VERDICT.md` with your reasoning and which notebook is
the headline proof — ending with a line exactly one of:
`VERDICT: HOLDS`     (done — a genuine honest improvement over a FAIR/strong benchmark is
                      verified and believable, at its honest magnitude even if smaller
                      than advertised)
`VERDICT: CONTINUE`  (a real edge may exist but the proof needs a fair-benchmark rebuild
                      or fix — give the Worker specific guidance for another round)
`VERDICT: FAULT`     (no real improvement survives a fair, information-matched comparison
                      — say whether the finding is an artifact/false or just not provable)
Also append a line `LESSON: <one short failure mode or standard future candidates should
apply>` (or `LESSON: none`) so later candidates benefit from what you learned here.
You speak for the user: honor any steering in `from_user.md` (shown above), and append a short,
PLAIN-LANGUAGE note for them to `verify/notes_to_user.md` — NO framework jargon (the user does not
know the framework's quant jargon; say it in everyday words): your verdict, the one-line
why, and — if it FAULTed on something the user might choose to waive (e.g. transaction costs or
tradeability of the signal) — say that explicitly so they can steer the next candidate. They read
it when they choose; never wait for them. (Your verdict + LESSON are ALSO surfaced to the system
the instant you decide.)
Then call report_to_user with a short summary and, if CONTINUE, the guidance.
"""


# ---------------------------------------------------------------------------
# Final per-workspace reports (the deliverable the user reads + the system ingests).
# Two grounded synthesis passes over THIS workspace's verdicts/reviews/lessons.
# ---------------------------------------------------------------------------
SYSTEM_FEEDBACK_PROMPT = _MISSION + """\

## Your role right now: write FEEDBACK_TO_SYSTEM (machine-to-machine)
You are NOT building a notebook this time — you are writing the verifier's consolidated
verdict on THIS workspace, addressed to Alpha Lab's OWN agents: the **Conductor** first (the
meta-agent that steers the run), then the **Strategist** and **Workers** if they read it.
Write dense, precise, technical prose. Domain and quant jargon is fine and PREFERRED here (the
task's own metrics, leakage/benchmark terms, walk-forward, information-matched benchmark,
permutation null, static vs dynamic effects, etc.) — these readers know it. Optimize for an LLM acting on it.

Read ALL the evidence listed in Context FIRST (every candidate's `ARBITER_VERDICT.md`,
`CRITIC_REVIEW.md`, `WORKER_NOTE.md`; `lessons.md`; `notes_to_user.md`; the menu; and the
select-log reasoning for skipped/NONE candidates). Ground every claim in those files — cite the
candidate and the ACTUAL numbers your colleagues computed; invent nothing. If something wasn't
checked, say so.

Write `verify/feedback_to_system.md` with these sections:
1. **Bottom line** — did ANY finding deliver a genuine, honest, economically-meaningful edge
   over a FAIR, information-matched benchmark? State the magnitude that honestly survives (may
   be "none"; may be "a real but sub-bar ~X residual"). No hedging.
2. **Per finding verified** — for each candidate: claimed mechanism; what the independent
   reconstruction found (key numbers; did the headline reproduce to the digit?); the **ROOT
   CAUSE** of the fault, named precisely (information-handicap benchmark / loses-to-a-trivial-floor
   on walk-forward (single-split selection) / static effect mistaken for dynamic skill / confound
   or known exposure mistaken for alpha / seed-luck & multiple-testing / structure a permutation-null
   matches / unit-inflation / cost-fragility, etc.); and any honest residual.
3. **Systemic blind spots** — the FRAMEWORK/CONVENTION-level failure modes (not per-experiment)
   that recur and silently inflate many leaderboard entries. Name each and list which experiments
   it touched.
4. **REMEDIES — actionable, paired to each fault/blind spot.** Be specific and implementable:
   name the component (e.g. `backtest/baselines.py`, the eval/metrics convention, the playbook
   gate, the Strategist's proposal criteria, the Conductor's annotation/parking policy) and exactly
   what to add/change — e.g. "add the information-matched baseline and a trivial last-value/persistence
   floor; gate every leaderboard entry on beating BOTH on walk-forward, not a single split"; "report
   raw per-period metrics, never annualized/unit-rescaled ones"; "require a seed-ensemble + permutation
   null before status='analyzed'"; "decompose any apparent skill into a static/known-exposure part vs
   the dynamic part before crediting it". EVERY fault gets a concrete remedy.
5. **Do next / do NOT re-explore** — guidance the Conductor can turn into directives/parking:
   which mechanism-classes are dead here and why; which honest residual (if any) merits a
   properly-benchmarked follow-up.

Write the file with shell_exec (a quoted heredoc, `cat > verify/feedback_to_system.md <<'MD' ...
MD`, so nothing is shell-expanded). Then call report_to_user with a one-line confirmation. Do not
dumb anything down — this one is for the machine.
"""

HUMAN_FEEDBACK_PROMPT = _MISSION + """\

## Your role right now: write REPORT_FOR_HUMAN (plain language, first principles)
You are NOT building a notebook this time — you are writing the report the USER reads. The user
is sharp but does NOT live inside this framework's vocabulary. The running notes have been full
of quant/ML jargon (the task's own metrics, leakage/benchmark terms, information-matched baselines,
walk-forward, annualized metrics, static vs dynamic effects) that is impenetrable from outside. Your job is the OPPOSITE of the system
report: explain what was studied, found, and concluded **from first principles, in plain English**,
as if to a smart colleague who has never heard the framework's terms.

Hard rules for THIS report:
- **No unexplained jargon.** If a term is unavoidable, define it in plain words the first time, in
  one short clause (e.g. "a 'lag0' rule — the dumbest possible guess: assume next period just
  repeats this period's value"). Prefer the plain phrasing outright. Translate, don't quote.
- For each finding, walk the reader through: **what the system claimed it found** (what real-world
  pattern, why it would make money — in everyday words), **what we did to check it** (we rebuilt it
  ourselves from the raw data and compared it against fair, simple alternatives and out-of-time
  periods), **what turned out to be true**, and **why the claim broke**, in a sentence or two a
  non-specialist can follow.
- Quantify in intuitive terms: "the elaborate model did no better than a one-line rule of thumb";
  "the apparent profit was really just betting the market goes up, in a year it went up"; "the
  signal only looked good because it was the luckiest of many random tries"; "the edge was real but
  so small that normal trading costs would wipe it out." Use a number only with its meaning attached.
- Be honest and concrete. If a small real edge survived, say exactly what it was, how small, and why
  it isn't (yet) worth trading.

Read ALL the evidence in Context first (the per-candidate verdicts/reviews, lessons, notes) so the
plain-English account is accurate — then translate it.

Write `verify/report_for_human.md`: a one-paragraph **plain-English bottom line**; then, per finding,
a short **what was claimed / what we found / why it broke** entry in everyday language; then a **what
this means** closing — the few simple, recurring reasons these findings didn't hold, and what that
says about the data itself. Write the file with shell_exec (a quoted heredoc). Then call
report_to_user with a one-line confirmation.
"""


WATCHDOG_PROMPT = _MISSION + """\

## Your role right now: User-Representative WATCHDOG (timer check-in — the verifier's conductor)
You are the User-Rep, woken on a timer to make sure the Worker/Critic for the CURRENT candidate are
not grinding down a dead end. Light hand: you supervise, you do NOT do the verification yourself, and
you intervene ONLY when warranted (this mirrors how the main pipeline's Conductor watches Phase 3).

A progress digest for the active candidate is in Context (its dir, files built so far + their
freshness, seconds-since-last-change, the current step). Read it, glance at the newest notebook/scratch
or the agent log if useful, and honor the user's steering in from_user.md. Judge ONE thing: is the
active agent making genuine progress, or is it (a) STALLED (frozen / no new output for a long time),
(b) LOOPING (re-running the same failing step), or (c) OFF-TRACK (chasing something the user's steering
did not ask for)?

- ON TRACK -> do nothing but append one short reassuring line to `verify/notes_to_user.md` (so the user
  knows you checked), then report_to_user "on track". Do NOT micromanage a healthy agent.
- DEAD-ENDING -> write a SHORT, concrete course-correction to `verify/<slug>/WATCHDOG_DIRECTIVE.md` (the
  agent reads it at the top of its next turn): say what to STOP and what to do INSTEAD, honoring the
  user's steering. Also append a plain-language line to `verify/notes_to_user.md` telling the user what
  you saw and did.
- HUNG SUBPROCESS (a notebook/training clearly stuck — running far too long with no new output) -> in
  addition, write one identifying substring (the candidate slug, or the stuck script name) to
  `verify/<slug>/WATCHDOG_KILL.txt`; the coordinator will terminate that job so the agent gets unstuck.
  Only for a genuinely hung job, never a slow-but-progressing one.

Use shell_exec (read-only inspection + writing those small files via a quoted heredoc). Then call
report_to_user with a one-line verdict: on-track / corrected / killed-hung-job.
"""


def _extract_verdict(path: Path) -> str | None:
    """Return the last `VERDICT: TOKEN` (or `SELF_CHECK: TOKEN`) in a file."""
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    m = re.findall(r"(?:VERDICT|SELF_CHECK):\s*([A-Z_]+)", text)
    return m[-1].upper() if m else None


class Verifier:
    """Drives candidate selection and the per-candidate worker/critic/arbiter loop."""

    def __init__(
        self,
        *,
        provider: Any,
        model: str,
        reasoning_effort: str,
        config: Any,
        workspace: str,
        data_path: str,
        adapter: Any,
        event_callback: Callable[[Any], None],
        nb_run_path: str,
        python_exe: str,
        max_candidates: int = 3,
        max_rounds: int = 4,
        notebook_timeout: int = 1800,
        steering: str = "",
        watchdog_interval: int = 0,
        worker_model: str = "",
        critic_model: str = "",
        userrep_model: str = "",
    ) -> None:
        self.provider = provider
        self.model = model
        # Per-role models (Worker / Critic / User-Rep); "" -> inherit the main model.
        # Top-level config can set all three independently (mirrors conductor_model). The
        # User-Rep model covers select + arbitrate + watchdog + the final reports.
        self.worker_model = worker_model or model
        self.critic_model = critic_model or model
        self.userrep_model = userrep_model or model
        self.effort = reasoning_effort
        self.config = config
        self.workspace = str(workspace)
        self.data_path = data_path
        self.adapter = adapter
        self.event_callback = event_callback
        self.nb_run_path = nb_run_path
        self.python_exe = python_exe
        self.max_candidates = max_candidates
        self.max_rounds = max_rounds
        self.notebook_timeout = notebook_timeout
        self.steering = steering
        self.watchdog_interval = int(watchdog_interval or 0)
        self.verify_dir = Path(self.workspace) / "verify"
        # Watchdog (User-Rep timer supervisor) state — mirrors the dispatcher's Conductor
        # scheduling (_state_lock / _conductor_running / _last_conductor_time / _conductor_thread).
        self._state_lock = threading.Lock()
        self._watchdog_running = False
        # Run-end wind-down: when set, the candidate loop finishes the
        # candidate currently in flight (its verdict must be RECORDED even
        # though the run is over and nothing will act on it) and starts no
        # new candidates. Set by the dispatcher's stop() drain.
        self._wind_down = threading.Event()
        self._last_watchdog_time = 0.0
        self._watchdog_thread: threading.Thread | None = None

    def _model_for(self, log_name: str) -> str:
        """Per-role model from the agent's log_name `verifier_<role>_<slug>...`: Worker / Critic /
        User-Rep (the last covers select, arbitrate, watchdog, and the final reports). Each defaults
        to the main model; the top-level config can set all three independently. Match the ROLE token
        ONLY — not a substring of the whole name, since a candidate slug can itself contain
        'worker'/'critic' and would misroute that candidate's arbiter/watchdog to the wrong model."""
        parts = log_name.split("_", 2)
        role = parts[1] if len(parts) > 1 and parts[0] == "verifier" else ""
        if role == "worker":
            return self.worker_model
        if role == "critic":
            return self.critic_model
        return self.userrep_model

    # -- agent runner (mirrors Supervisor._run_review) ----------------------
    def _run_agent(self, system_prompt: str, initial_message: str,
                   tools: list[dict], log_name: str, extra_context: str = "") -> str:
        m = self._model_for(log_name)  # per-role model (worker/critic/user-rep) by log-name
        context = ContextManager(
            provider=self.provider,
            model=m,
            workspace=self.workspace,
            summarization_threshold_tokens=getattr(
                self.config, "context_summarization_threshold_tokens", 150_000),
            learnings_summary_threshold_tokens=getattr(
                self.config, "learnings_summary_threshold_tokens", 20_000),
        )
        shared = (
            f"\n## Context\n"
            f"- Workspace (read-only except `verify/`): `{self.workspace}`\n"
            f"- Raw data path (the data the system used): `{self.data_path}`\n"
            f"- Your output dir: `{self.verify_dir}`\n"
            f"- Notebook engine: run `{self.python_exe} {self.nb_run_path} IN.py OUT.ipynb "
            f"--timeout <SECS-you-choose>` AND set the shell_exec tool's `timeout` arg high "
            f"enough for the whole run (the default is short and will cut long notebooks off).\n"
            f"- Always use this python for code/notebooks: `{self.python_exe}`\n"
            f"- shell_exec runs with cwd = the workspace.\n"
            f"- ALWAYS pass an explicit `timeout` to shell_exec, sized to the work (large for an "
            f"NN retrain). NEVER run an unscoped find (`find /`, `find /ms|/v|...`) — it is refused; "
            f"scope every find to a specific directory (the paths you need are above).\n"
        )
        dk = self.verify_dir.parent / "adapter" / "domain_knowledge.md"
        if dk.exists():
            shared += ("\n## Domain knowledge (from this run's adapter — the task's own metrics, "
                       "baselines, and leakage traps; ground your checks in these, do not assume a domain)\n"
                       + dk.read_text(errors="replace")[:4000])
        lessons_file = self.verify_dir / "lessons.md"
        if lessons_file.exists():
            shared += ("\n## Lessons from prior verification (APPLY from the first notebook)\n"
                       + lessons_file.read_text(errors="replace")[:6000])
        fu = self.verify_dir / "from_user.md"
        if fu.exists():
            steer = "\n".join(ln for ln in fu.read_text(errors="replace").splitlines()
                              if ln.strip() and not ln.lstrip().startswith("#")).strip()
            if steer:
                shared += ("\n## User steering — `verify/from_user.md` (AUTHORITATIVE; the user may "
                           "edit it mid-run). Honor it; NEVER block or wait for the user.\n" + steer[:4000])
        if extra_context:
            shared += "\n" + extra_context

        def prompt_builder(workspace, learnings, config=None):
            return system_prompt + shared

        agent = AgentLoop(
            provider=self.provider,
            model=m,
            context=context,
            event_callback=self.event_callback,
            reasoning_effort=self.effort,
            config=self.config,
            tools=tools,
            prompt_builder=prompt_builder,
            log_name=log_name,
            min_report_attempts=1,
            db=None,
            adapter=self.adapter,
        )
        try:
            return agent.run(initial_message) or ""
        except Exception as e:  # never let one agent turn kill the run
            logger.exception("verifier agent %s crashed: %s", log_name, e)
            return f"[agent {log_name} crashed: {type(e).__name__}: {e}]"

    # -- candidate menu (orchestrator reads DB read-only; no live writes) ---
    def _build_menu(self) -> int:
        self.verify_dir.mkdir(parents=True, exist_ok=True)
        db_path = Path(self.workspace) / "experiments.db"
        ann_path = Path(self.workspace) / "meta" / "annotations.json"
        annotations: dict[str, Any] = {}
        if ann_path.exists():
            try:
                annotations = json.loads(ann_path.read_text())
            except Exception:
                annotations = {}
        lines = ["# Candidate menu (analyzed experiments)\n",
                 "Headline numbers are the SYSTEM's (possibly inflated/unit-mixed) — "
                 "use them only to triage, then read the debrief.\n"]
        n = 0
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                # 'done' is the terminal state AFTER 'analyzed' (kanban: analyzed -> done) and still
                # carries results_json. The trigger and the whole dispatcher count ("analyzed","done")
                # together, so the menu must too — otherwise an experiment promoted to 'done' is counted
                # toward the verify threshold yet silently absent from the candidate menu.
                "SELECT id,name,results_json FROM experiments "
                "WHERE status IN ('analyzed','done') AND results_json IS NOT NULL "
                "ORDER BY updated_at DESC"
            ).fetchall()
            con.close()
            for r in rows:
                try:
                    res = json.loads(r["results_json"] or "{}")
                except Exception:
                    res = {}
                metrics = {k: res[k] for k in list(res)[:8]
                           if isinstance(res.get(k), (int, float))}
                ann = annotations.get(str(r["id"]))
                ann_label = (ann.get("label") if isinstance(ann, dict) else ann) or ""
                deb = f"experiments/{r['name']}/debrief.md"
                lines.append(
                    f"- **#{r['id']} {r['name']}**{(' [' + ann_label + ']') if ann_label else ''}\n"
                    f"  metrics: {metrics}\n  debrief: `{deb}`")
                n += 1
        except Exception as e:
            lines.append(f"\n[menu build error: {type(e).__name__}: {e}]")
        (self.verify_dir / "_candidate_menu.md").write_text("\n".join(lines) + "\n")
        return n

    def _select_next_candidate(self, outcomes: dict) -> dict | None:
        """User-Rep picks ONE next candidate, informed by what already faulted (adaptive).

        Returns the candidate dict, or None when nothing remaining has a plausible chance.
        """
        existing = sorted(d.name for d in self.verify_dir.iterdir() if d.is_dir())
        rows = sorted(set(existing) | set(outcomes))
        hist = "\n".join(f"- {s}: {outcomes.get(s, 'attempted/in-progress')}" for s in rows) or "(none yet)"
        extra = ("## Candidates already attempted — do NOT re-pick these or the same "
                 f"mechanism-class:\n{hist}\n")
        # Conductor priority pin: if request_verification named a candidate, honor it as THIS turn's
        # pick (hard override of the adaptive choice) unless already attempted. Consumed once — single
        # overwritten file, so the latest pin wins, not FIFO.
        pin_path = self.verify_dir / "priority_pin.json"
        if pin_path.exists():
            try:
                pin = json.loads(pin_path.read_text())
                pin_path.unlink()  # consume once
                pin_cand = str(pin.get("candidate", "")).strip()
                pin_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", pin_cand)
                if pin_cand and pin_slug not in rows:
                    extra += (f"\n## PRIORITY PIN (Conductor, priority={pin.get('priority') or 'high'}) — "
                              f"OVERRIDES your adaptive choice this turn:\nSelect the experiment matching "
                              f"`{pin_cand}` as THIS turn's candidate and write it to next_candidate.json. "
                              f"Deviate only if it is already in the attempted list above or cannot be found "
                              f"in the menu.\n")
            except (OSError, ValueError):
                pass
        tools = get_tool_schemas(["read_file", "grep_file", "shell_exec", "report_to_user"])
        npath = self.verify_dir / "next_candidate.json"
        try:
            npath.unlink()  # verifier-owned scratch; clear so we detect a fresh pick
        except OSError:
            pass
        self._run_agent(USER_REP_SELECT_PROMPT,
                        "Pick the single next candidate with a real chance to survive a fair "
                        'benchmark, or write {"slug":"NONE"}. Go.',
                        tools, f"verifier_select_{len(outcomes) + 1}", extra_context=extra)
        if not npath.exists():
            logger.warning("user-rep wrote no next_candidate.json")
            return None
        try:
            cand = json.loads(npath.read_text())
        except Exception as e:
            logger.error("next_candidate.json parse failed: %s", e)
            return None
        if not isinstance(cand, dict) or not cand.get("slug") or str(cand.get("slug")).upper() == "NONE":
            return None
        return cand

    # -- watchdog: User-Rep timer supervisor (mirrors dispatcher Conductor scheduling) ----
    def _should_run_watchdog(self) -> bool:
        """Mirror of dispatcher._should_run_conductor: NOOP when off; at most one in flight;
        otherwise fire on the slow timer."""
        if self.watchdog_interval <= 0:
            return False
        with self._state_lock:
            if self._watchdog_running:
                return False
            return (time.time() - self._last_watchdog_time) >= self.watchdog_interval

    def _maybe_run_watchdog(self, slug: str, cdir: Path, step: str) -> None:
        """Mirror of dispatcher._maybe_run_conductor: spawn one watchdog check in a daemon
        thread, single in-flight, non-blocking."""
        if not self._should_run_watchdog():
            return
        with self._state_lock:
            self._watchdog_running = True
            self._last_watchdog_time = time.time()

        def _watchdog_thread() -> None:
            try:
                self._run_watchdog(slug, cdir, step)
            except Exception as e:
                logger.exception("verifier watchdog crashed: %s", e)
            finally:
                with self._state_lock:
                    self._watchdog_running = False
                    self._watchdog_thread = None

        t = threading.Thread(target=_watchdog_thread, daemon=True)
        with self._state_lock:
            self._watchdog_thread = t
        t.start()

    def _run_watchdog(self, slug: str, cdir: Path, step: str) -> None:
        """One watchdog turn: digest the candidate's live progress, then run the User-Rep
        watchdog agent (judges on-track/stalled/looping/off-track; writes a directive + user
        note + optional kill-marker). Concurrent with the active worker/critic turn."""
        digest, frozen_s = [], -1
        try:
            now = time.time()
            files = [f for f in cdir.glob("*") if f.is_file()]
            for f in sorted(files, key=lambda p: -p.stat().st_mtime)[:30]:
                digest.append(f"  {f.name}  ({int(now - f.stat().st_mtime)}s ago)")
            newest = max((f.stat().st_mtime for f in files), default=0)
            frozen_s = int(now - newest) if newest else -1
        except Exception:
            pass
        extra = (
            f"## Watchdog timer check — candidate `{slug}`, current step: {step}\n"
            f"- candidate dir: `verify/{slug}/` — write a directive (if needed) to "
            f"`verify/{slug}/WATCHDOG_DIRECTIVE.md`, a kill-marker to `verify/{slug}/WATCHDOG_KILL.txt`\n"
            f"- seconds since the newest artifact changed: {frozen_s}  (large => possibly frozen)\n"
            f"- candidate-dir contents (newest first, age):\n" + ("\n".join(digest) or "  (empty)") + "\n"
        )
        tools = get_tool_schemas(["read_file", "grep_file", "shell_exec", "report_to_user"])
        self._run_agent(
            WATCHDOG_PROMPT,
            f"Timer check on `{slug}` (step {step}). On track, or dead-ending? Act only if warranted. Go.",
            tools, f"verifier_watchdog_{slug}", extra_context=extra)

    def _consume_watchdog_markers(self, slug: str, cdir: Path) -> None:
        """Mirror of dispatcher._consume_kill_requests: if the watchdog left a kill-marker,
        terminate the matching hung notebook/training subprocess (never the orchestrator)."""
        marker = cdir / "WATCHDOG_KILL.txt"
        if not marker.exists():
            return
        try:
            tokens = [t.strip() for t in marker.read_text(errors="replace").splitlines() if t.strip()][:5]
            marker.unlink()
        except OSError:
            return
        if not tokens:
            return
        import os as _os, subprocess as _sp
        # The orchestrator the verifier runs inside (integrated: run.py/dispatcher; standalone:
        # verify_workspace) and its shell parent are NEVER killable. Skip by PID — matching by name is
        # unsafe because the integrated orchestrator is `run.py` and "run.py" is also a substring of the
        # verifier's own `nb_run.py` jobs, so a name-skip would also spare the very jobs we must kill.
        _self_pids = {_os.getpid(), _os.getppid()}
        try:
            ps = _sp.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return
        for line in ps.splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            pid, args = int(parts[0]), parts[1]
            if pid in _self_pids:
                continue
            if "verify_workspace" in args or "ps -eo" in args or "verifier_watchdog" in args:
                continue
            if not any(k in args for k in ("nb_run", "ipykernel", "python")):
                continue
            if not any(tok in args for tok in tokens):
                continue
            try:
                _sp.run(["kill", "-9", str(pid)], timeout=5)
                self._note_to_user(f"watchdog killed a hung job (pid {pid}) for `{slug}`")
                logger.info("watchdog killed hung pid %d for %s", pid, slug)
            except Exception:
                pass

    def _run_agent_supervised(self, system_prompt: str, initial_message: str, tools: list[dict],
                              log_name: str, extra_context: str = "", *,
                              slug: str = "", cdir: Path | None = None, step: str = "") -> str:
        """Run one agent turn under the watchdog. Mirrors the dispatcher: the agent runs in a
        daemon thread while this coordinator tick-loop spawns the timer watchdog (single in
        flight) and consumes its markers. Pure NOOP passthrough to _run_agent when the watchdog
        is off (watchdog_interval<=0), so default behavior is unchanged."""
        if self.watchdog_interval <= 0 or cdir is None:
            return self._run_agent(system_prompt, initial_message, tools, log_name, extra_context)
        with self._state_lock:  # restart timer at turn start -> first check after one full interval
            self._last_watchdog_time = time.time()
        holder: dict[str, str] = {}

        def _target() -> None:
            holder["out"] = self._run_agent(system_prompt, initial_message, tools, log_name, extra_context)

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        while t.is_alive():
            self._maybe_run_watchdog(slug, cdir, step)
            self._consume_watchdog_markers(slug, cdir)
            t.join(timeout=20)  # coordinator tick
        return holder.get("out", "")

    # -- per-candidate inner loop -------------------------------------------
    def _verify_candidate(self, cand: dict) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(cand.get("slug") or cand.get("experiment_name", "cand")))
        cdir = self.verify_dir / slug
        cdir.mkdir(parents=True, exist_ok=True)
        dossier = (
            f"## Candidate under verification: `{slug}`\n"
            f"- experiment: {cand.get('experiment_name')}\n"
            f"- claim: {cand.get('claim')}\n"
            f"- mechanism: {cand.get('mechanism')}\n"
            f"- persuasion spec (what a human needs to see): {cand.get('persuasion_spec')}\n"
            f"- benchmarks to beat: {cand.get('benchmarks')}\n"
            f"- write all artifacts under: `verify/{slug}/`\n"
        )
        worker_tools = get_tool_schemas(
            ["shell_exec", "read_file", "grep_file", "view_image", "report_to_user"])
        critic_tools = worker_tools
        arb_tools = worker_tools

        review = cdir / "CRITIC_REVIEW.md"
        arb = cdir / "ARBITER_VERDICT.md"
        steps = ["worker", "critic", "arbiter"]
        state = self._load_state(cdir)
        if state.get("outcome"):  # graceful resume: this candidate already finished
            logger.info("resume: candidate %s already done (%s)", slug, state["outcome"])
            return state["outcome"]
        rnd = int(state.get("round", 1) or 1)
        step = state.get("step", "worker")
        if step not in steps:
            step = "worker"
        outcome = f"INCONCLUSIVE (exhausted {self.max_rounds} rounds)"
        worker_self = critic_v = None
        while rnd <= self.max_rounds:
            self.event_callback_safe(f"candidate {slug}: round {rnd}/{self.max_rounds} (from {step})")
            prior = ""
            if rnd > 1 and review.exists():
                prior = f"\n## Critic's review to address\n" + review.read_text(errors="replace")[:8000]
            if rnd > 1 and arb.exists():
                prior += f"\n## User-Rep guidance\n" + arb.read_text(errors="replace")[:4000]
            wd = cdir / "WATCHDOG_DIRECTIVE.md"
            wd_ctx = (("\n## WATCHDOG directive (the User-Rep flagged a problem during a prior turn — "
                       "address it FIRST):\n" + wd.read_text(errors="replace")[:3000]) if wd.exists() else "")
            for s in steps[steps.index(step):]:
                # Mark the step as in flight BEFORE running it: a hard kill
                # mid-step then leaves a visible {round, step, outcome: null}
                # cut marker instead of no STATE.json at all (three d5
                # verification folders had nothing, 2026-08-07 — a
                # first-round worker cut writes no state otherwise).
                self._save_state(cdir, rnd, s, None)
                if s == "worker":
                    self._run_agent_supervised(
                        WORKER_PROMPT.format(PY=self.python_exe, NB_RUN=self.nb_run_path),
                        f"Build/iterate the proof notebooks for `{slug}` (reuse any prior work on disk). Go.",
                        worker_tools, f"verifier_worker_{slug}_r{rnd}",
                        extra_context=dossier + prior + wd_ctx, slug=slug, cdir=cdir, step="worker")
                    self._save_state(cdir, rnd, "critic", None)
                elif s == "critic":
                    self._run_agent_supervised(
                        CRITIC_PROMPT,
                        f"Critique the proof for `{slug}`. Run your own probes. Go.",
                        critic_tools, f"verifier_critic_{slug}_r{rnd}",
                        extra_context=dossier, slug=slug, cdir=cdir, step="critic")
                    self._save_state(cdir, rnd, "arbiter", None)
                else:
                    self._run_agent(USER_REP_ARBITRATE_PROMPT,
                                    f"Arbitrate `{slug}`: is a genuine improvement proven and persuasive? Go.",
                                    arb_tools, f"verifier_arbiter_{slug}_r{rnd}",
                                    extra_context=dossier)
            verdict = _extract_verdict(arb)
            worker_self = _extract_verdict(cdir / "WORKER_NOTE.md")
            critic_v = _extract_verdict(review)
            logger.info("candidate %s round %d -> arbiter=%s worker=%s critic=%s",
                        slug, rnd, verdict, worker_self, critic_v)
            if verdict == "HOLDS":
                outcome = f"HOLDS (round {rnd}; worker={worker_self}, critic={critic_v})"
                break
            if verdict == "FAULT":
                outcome = f"FAULT (round {rnd}; critic={critic_v})"
                break
            rnd += 1
            step = "worker"
        self._save_state(cdir, rnd, "done", outcome)
        self._carry_lesson(cdir, slug)
        self._surface_finding(slug, cdir, outcome, worker_self, critic_v)
        return outcome

    def _carry_lesson(self, cdir: Path, slug: str) -> None:
        """Append the arbiter's `LESSON:` line to verify/lessons.md for later candidates."""
        arbf = cdir / "ARBITER_VERDICT.md"
        if not arbf.exists():
            return
        for ln in arbf.read_text(errors="replace").splitlines():
            s = ln.strip()
            if s.upper().startswith("LESSON:"):
                lesson = s[len("LESSON:"):].strip()
                if lesson and lesson.lower() != "none":
                    try:
                        with open(self.verify_dir / "lessons.md", "a") as lf:
                            lf.write(f"- (from {slug}) {lesson}\n")
                    except Exception:
                        pass
                break

    def _surface_finding(self, slug: str, cdir: Path, outcome: str,
                         worker_self: str | None, critic_v: str | None) -> None:
        """Push a resolved finding to the SYSTEM the instant all three agents have weighed in —
        so feedback flows both ways in real time, not only at end-of-run. The user channel
        (notes_to_user.md) is written plainly by the arbiter itself; here we append the
        system-facing entry (agreement + remedy/lesson + pointers to the full diagnosis) to the
        live `verify/feedback_stream.md` (the Conductor/strategist read this mid-run; wired into
        the live pipeline it maps to meta/notes_inbox.md)."""
        import datetime
        arbf = cdir / "ARBITER_VERDICT.md"
        lesson = ""
        if arbf.exists():
            for ln in arbf.read_text(errors="replace").splitlines():
                if ln.strip().upper().startswith("LESSON:"):
                    lesson = ln.strip()[len("LESSON:"):].strip()
                    break
        entry = (f"\n## {datetime.datetime.now():%Y-%m-%d %H:%M} — {slug} -> {outcome}\n"
                 f"- agreement: worker={worker_self}, critic={critic_v} (arbiter decided)\n"
                 f"- remedy / lesson: {lesson or '(see verdict)'}\n"
                 f"- full diagnosis: `verify/{slug}/ARBITER_VERDICT.md` | critique: "
                 f"`verify/{slug}/CRITIC_REVIEW.md`\n")
        try:
            with open(self.verify_dir / "feedback_stream.md", "a") as f:
                f.write(entry)
        except Exception:
            pass
        self.event_callback_safe(f"finding surfaced (system+user): {slug} -> {outcome}")

    def _load_state(self, cdir: Path) -> dict:
        f = cdir / "STATE.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                pass
        return {"round": 1, "step": "worker", "outcome": None}

    def _save_state(self, cdir: Path, rnd: int, step: str, outcome) -> None:
        try:
            (cdir / "STATE.json").write_text(
                json.dumps({"round": rnd, "step": step, "outcome": outcome}))
        except Exception:
            pass

    def _init_user_channels(self) -> None:
        """Seed the user-facing channels (mirrors the Conductor): notes_to_user.md (the user
        reads when they like; never alerted) and from_user.md (the user steers; read at the top
        of every agent turn; never blocks). Also seeds the live system-facing findings stream.
        Any `steering` passed in (e.g. from config) is written as an ACTIVE first line of
        from_user.md so the agents honor it from candidate #1 (the user shouldn't have to wait
        for the first fault to discover they wanted to waive costs)."""
        fu = self.verify_dir / "from_user.md"
        if not fu.exists():
            seed = (self.steering.strip() + "\n\n") if self.steering.strip() else ""
            fu.write_text(
                seed +
                "# Steering for the verifier — edit anytime.\n"
                "# Read at the top of EVERY agent turn and treated as authoritative; the verifier\n"
                "# NEVER blocks or waits for you. Lines starting with # are ignored. Examples:\n"
                "#   verify <experiment_name> next\n"
                "#   stop after this candidate / stop now\n"
                "#   I don't care about transaction costs or tradeability of the signal\n"
                "#   be harsher about leakage / commit a specific held-out regime by date\n\n")
        notes = self.verify_dir / "notes_to_user.md"
        if not notes.exists():
            notes.write_text("# Verifier — notes to the user (running log; read whenever you like)\n\n")
        stream = self.verify_dir / "feedback_stream.md"
        if not stream.exists():
            stream.write_text(
                "# Verifier — live findings stream (system-facing).\n"
                "# Appended the moment a candidate resolves and all three agents have weighed in,\n"
                "# so feedback flows both ways in real time. The Conductor/strategist read this\n"
                "# mid-run; wired into the live pipeline it maps to meta/notes_inbox.md. The\n"
                "# end-of-run feedback_to_system.md consolidates this stream.\n")

    def _note_to_user(self, msg: str) -> None:
        import datetime
        try:
            with open(self.verify_dir / "notes_to_user.md", "a") as f:
                f.write(f"- {datetime.datetime.now():%Y-%m-%d %H:%M} — {msg}\n")
        except Exception:
            pass

    def event_callback_safe(self, msg: str) -> None:
        try:
            self.event_callback(msg)
        except Exception:
            pass

    # -- final per-workspace reports (the deliverable) ----------------------
    def _write_final_reports(self, results: dict) -> None:
        """Per-workspace final product. Two grounded synthesis passes over THIS workspace's
        evidence (verdicts/reviews/lessons/notes/select-logs):
          - verify/feedback_to_system.md : for the Conductor (then strategist/workers) —
            deep, actionable, machine-readable; framework jargon welcome.
          - verify/report_for_human.md   : first-principles, plain-language, for the user.
        Each report is written BY an agent that reads the actual evidence (nothing here is
        hand-authored by code), so the remedies/diagnoses are grounded, not paraphrased.
        """
        cand_dirs = sorted(d.name for d in self.verify_dir.iterdir()
                           if d.is_dir() and (d / "STATE.json").exists())
        ev = [
            "## Evidence to read FIRST (this workspace only) — ground every statement in it:",
            f"- Outcome per candidate (the verdicts): {json.dumps(results)}",
            "- `verify/_candidate_menu.md` (what findings existed), `verify/lessons.md` (failure "
            "modes already banked), `verify/notes_to_user.md` (the running log).",
            "- For EACH verified candidate, read its verdict + critique + worker note:",
        ]
        for s in cand_dirs:
            ev.append(f"  - `verify/{s}/ARBITER_VERDICT.md` , `verify/{s}/CRITIC_REVIEW.md` , "
                      f"`verify/{s}/WORKER_NOTE.md`  (+ its notebooks / `critic_scratch/` if useful)")
        ev.append("- The User-Rep's reasoning for candidates it SKIPPED / declared NONE is in "
                  "`logs/verifier_select_*.jsonl` (grep the last/biggest text block) — fold in the "
                  "surveyed-but-not-verified set where it matters (esp. WHY nothing was viable).")
        ev_ptr = "\n".join(ev)
        tools = get_tool_schemas(["read_file", "grep_file", "shell_exec", "report_to_user"])
        self.event_callback_safe("writing final reports (feedback_to_system + report_for_human)")
        self._run_agent(
            SYSTEM_FEEDBACK_PROMPT,
            "Read all the evidence, then write `verify/feedback_to_system.md`. Go.",
            tools, "verifier_report_system", extra_context=ev_ptr)
        self._run_agent(
            HUMAN_FEEDBACK_PROMPT,
            "Read all the evidence, then write `verify/report_for_human.md` in plain English. Go.",
            tools, "verifier_report_human", extra_context=ev_ptr)
        for fn in ("feedback_to_system.md", "report_for_human.md"):
            ok = (self.verify_dir / fn).exists()
            logger.info("final report %s: %s", fn, "written" if ok else "MISSING")
            if not ok:
                self._note_to_user(f"(verifier: {fn} was not written by the report agent)")

    # -- top level ----------------------------------------------------------
    def finish_current_and_stop(self) -> None:
        """Run-end wind-down (dispatcher stop() drain): complete the
        candidate currently in flight so its verdict is recorded — nobody
        is left to act on it, but the decision must survive — and start no
        further candidates."""
        self._wind_down.set()

    def run(self) -> dict:
        self.verify_dir.mkdir(parents=True, exist_ok=True)
        lessons_file = self.verify_dir / "lessons.md"
        if not lessons_file.exists():
            lessons_file.write_text(_LESSONS_SEED)
        self._init_user_channels()
        n = self._build_menu()
        logger.info("verifier: candidate menu has %d analyzed experiments", n)
        # Resume: load outcomes of already-resolved candidates from their STATE.json so a
        # restart doesn't redo finished work.
        results: dict[str, str] = {}
        for d in sorted(self.verify_dir.iterdir()):
            if d.is_dir():
                st = self._load_state(d)
                if st.get("outcome"):
                    results[d.name] = st["outcome"]
        if results:
            logger.info("resume: %d candidate(s) already resolved: %s", len(results), results)
            self._note_to_user("Resuming. Already decided: "
                               + "; ".join(f"{s} -> {o}" for s, o in results.items()))
        # Adaptive loop: pick the next candidate informed by prior faults (pivoting away
        # from doomed mechanism-classes) until we've tried max_candidates or the User-Rep
        # finds nothing else with a chance.
        while len(results) < self.max_candidates:
            if self._wind_down.is_set():
                logger.info("verifier: wind-down — run ended; verdicts for "
                            "in-flight work recorded, no new candidates")
                self._note_to_user(
                    "Run ended while the verification sweep was underway. "
                    "The candidate in flight was carried to its verdict "
                    "(recorded below and in its verify/ folder); no further "
                    "candidates were started.")
                break
            cand = self._select_next_candidate(results)
            if not cand:
                logger.info("verifier: User-Rep found no further candidate worth trying; stopping")
                self._note_to_user("User-Rep found no further candidate with a plausible chance of "
                                   "surviving a fair benchmark - stopping the sweep.")
                break
            slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(cand.get("slug") or cand.get("experiment_name", "cand")))
            if slug in results:
                logger.info("verifier: User-Rep re-picked already-tried %s; stopping to avoid a loop", slug)
                break
            self._note_to_user(f"Verifying next: **{slug}** (exp: {cand.get('experiment_name')}). "
                               f"Why it might survive a fair benchmark: {cand.get('why_it_might_survive', '?')}")
            logger.info("verifier: verifying candidate '%s'", slug)
            try:
                results[slug] = self._verify_candidate(cand)
            except Exception as e:
                logger.exception("candidate %s crashed", slug)
                results[slug] = f"ERROR: {type(e).__name__}: {e}"
            self._note_to_user(f"Candidate **{slug}** -> {results[slug]}")
        lines = ["# Verifier report\n"]
        for slug, outcome in results.items():
            lines.append(f"- **{slug}**: {outcome}  "
                         f"(see `verify/{re.sub(r'[^A-Za-z0-9_-]+','-',slug)}/`)")
        (self.verify_dir / "report.md").write_text("\n".join(lines) + "\n")
        try:
            self._write_final_reports(results)
        except Exception as e:
            logger.exception("final report generation failed: %s", e)
        logger.info("verifier: done. outcomes=%s", results)
        return results
