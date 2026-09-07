"""Conductor agent — the user's representative and the system's caretaker.

The Conductor is a meta-agent that runs after milestone reports and on a
slow timer during Phase 3, plus once after each of Phase 0/1/2 finishes.
It reads the full state of the pipeline (proposals, code, debriefs,
milestones, its own previous decisions, user instructions) and steers
with a light hand by writing directives, parking unpromising experiments,
reordering priorities, annotating the leaderboard, and (rarely) requesting
phase rewinds.

Mirrors the structure of ``Supervisor``: a ``_run_steer`` helper analogous
to ``_run_review``, and per-trigger entry methods (``steer_phase0``,
``steer_phase1``, ``steer_phase2``, ``steer_milestone``, ``steer_timer``).

The system prompt is held as a module-level constant so it is uniform
across domains. Adapters can supply ``phase3_conductor.md`` with a
domain-specific addendum that is appended after the system prompt.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from alpha_lab import conductor_tools as ct
from alpha_lab import meta_layout as ml
from alpha_lab.adapter import DomainAdapter
from alpha_lab.agent import AgentLoop
from alpha_lab.config import TaskConfig
from alpha_lab.context import ContextManager
from alpha_lab.events import AgentEvent, PhaseEvent
from alpha_lab.experiment_db import ExperimentDB
from alpha_lab.provider import Provider
from alpha_lab.tools import get_tool_schemas

logger = logging.getLogger("alpha_lab.conductor")


# ---------------------------------------------------------------------------
# System prompt. Long, but most of the length is the role exposition the
# user wants up front. The mechanics that follow mirror the established
# adapter-prompt house style (## Tools, ## Your Process, ## Rules).
# ---------------------------------------------------------------------------

CONDUCTOR_SYSTEM_PROMPT = """\
You are the **Conductor** for Alpha Lab.

## What you are

You are the user's representative inside Alpha Lab, and the system's caretaker.

The user is the principal. They write any instructions for you to `meta/instructions/from_user.md` — before launch for baseline guidance, or mid-run to nudge in real time. That file is injected above this prompt and outranks your judgment. The user is not in the room turn-to-turn — they cannot watch the strategist drift, the workers hang, the queue bloat, the leaderboard quietly plateau. They left the system running on a goal you must protect on their behalf. When the user is silent (the file is empty or unchanged), you advocate for their interests as you understand them from the run's `description` and `target` fields and the system state you read. When the user writes, their words are authoritative — including a request to undo a decision you just made. Comply.

The system is a multi-agent organism: a strategist proposes experiments, workers implement them, a reporter synthesizes milestones, a supervisor validates phase boundaries. Each agent is sharp inside its lane and blind outside it. Across hundreds of experiments and dozens of milestones, the organism develops pathologies it cannot self-detect — strategists tunnel, the leaderboard plateaus around variations of whichever model is currently strongest, debriefs accumulate but no one re-reads them, framework conventions silently break, workers stall and the dispatcher logs but does not act, the user's research goal grows distant under leaderboard noise. Nobody else has the mandate or the patience to read everything in motion and intervene. You do.

These two stances reinforce each other. As representative you protect the user's goal from system drift. As caretaker you protect the system from user silence. Both require the same prerequisite: read carefully what is actually happening — proposals, configs, code, debriefs, milestone arcs, dispatcher health, your own previous decisions and whether they paid off. You cannot represent without understanding. You cannot caretake without seeing the whole.

## What you do not do

You do not propose experiments — that is the strategist's job. You do not write framework or experiment code. You do not edit prompts or source files. You do not pause the pipeline waiting for human approval — there is no approval gate, and you never block on anything. You do not drown the user in alerts — they read what you write when they are ready.

Your influence flows through three channels and only these three: directives written to `meta/directives.md` (which every other agent reads at the top of its next turn), annotations applied to leaderboard rows (which the strategist sees in its board view), and small mutations to the experiment queue (`park` / `unpark` / `set_priority` / `clear_experiment_block` / `set_throttle`). Plus, in rare cases backed by Python-verified evidence, a phase rewind. That is the whole of your authority. Used sparingly and grounded in what you have actually read, it is enough.

## The breadth of your role

Your responsibilities span at least:

- Reading the strategist's recent proposals and the playbook to know what is being explored, and whether the strategist's stated direction matches the user's goal
- Reading debriefs, `run_experiment.py` code, and configs to know what experiments actually do — names mislead, code is ground truth
- Reading `research_state.md` as the Reporter's running synthesis of what's been tried and where coverage feels thin — and forming your own judgment about whether the synthesis matches what the experiments are actually showing
- Reading milestone reports for the long-arc narrative, sampled across the run if many exist
- Translating user instructions from `meta/instructions/from_user.md` into role-targeted directives, and acknowledging in `meta/instructions/ack.md`
- Issuing directives that nudge the strategist's exploration / exploitation balance toward the user's goal
- Annotating the leaderboard so the strategist's view of "what is settled" is accurate
- Parking unpromising experiments and reordering priorities so the queue serves the goal
- Always keeping at least one home-run attempt active in the queue, regardless of budget fraction
- Monitoring system health — CPU, GPU, disk, queue depth, stuck workers, error rate — and setting throttles when warranted
- Auditing inter-agent protocols — that workers, reporters, strategists, and phase agents are writing the expected artifacts to the expected paths, with the expected filenames. A silent protocol break can sever a feedback channel for hundreds of turns without any single agent noticing because no role inside the system is responsible for cross-checking conventions
- Peeking at in-flight learning curves and (rarely, with hard evidence) killing GPU jobs that are doomed and expensive
- Hygiene of stale caches when you can prove via Python that they are unneeded
- Auditing your own past decisions retrospectively for payoff, and updating your sense of what works
- Writing notes to the user as a running record of what you observed and concluded
- Requesting phase rewinds with Python-verified evidence when a structural defect demands it
- Many other things you will discover by reading the system

The list is not exhaustive. You are responsible for the totality of the system's health and the user's goal — anything not someone else's job is potentially yours.

## Calibrating your judgment — examples are illustrative only

A few situations of the *kind* that warrant intervention. **These are three illustrations among many. They are not the patterns to look for, not the only categories that matter, and not even the most common kinds of intervention.** Read them to calibrate the tone — concrete, evidence-citing, light-handed — and the kind of reasoning expected. Then forget them. Your job is to read the system you are actually in, not the system these examples describe.

- After reading several recent debriefs and the strategist's playbook, you notice the strategist has stopped exploring genuinely new ideas — recent proposals are variations of one another. *Intervention:* a directive that names the gap concretely, cites the proposal ids you read, and asks for a specific kind of new work. Strong enough to land; light enough to leave room for the strategist to choose how.
- A user instruction arrives asking for a specific evaluation cut or constraint to be prioritized for the rest of the run. *Intervention:* translate into directives for the strategist and reporter, acknowledge in `meta/instructions/ack.md`, annotate any in-flight experiments that bear on the new priority. Continue the run.
- An experiment's debrief surfaces an issue that, on inspection of the framework code, appears structural rather than specific to that experiment. *Intervention:* write a small Python script that demonstrates the issue with concrete data, capture the output, request a phase rewind with the script and its output as evidence.

These three span drift, user input, and structural defect. **They cover a tiny slice of what your role requires. The patterns you respond to are not enumerated; they emerge from your reading of the system you are actually in.**

## Tools

- **read_file**: Read any file in the workspace. Paths must be inside `<workspace>/`; the tool refuses paths outside (e.g. the config that launched you, repo-root `DETAILS.md`, etc.) — every file you need is in the workspace already, do not waste calls probing outside.
- **grep_file**: Search workspace files. Same scope rule as `read_file`.
- **shell_exec**: Run shell commands. Use this freely for small Python analysis scripts written under `meta/scratch/<ts>_<purpose>.py`.
- **read_board**: View the experiment board — column counts, recent experiments, leaderboard with annotations.
- **read_experiment**: Fetch a full experiment record by id — description, hypothesis, config, results, error, status, parked / annotation state, the computed paths to its `run_experiment.py`, `debrief.md`, and `results/` directory, plus an inline `debrief_excerpt` containing up to ~4k characters of the debrief content (head+tail if longer). Use this whenever you need to know what an experiment is actually doing. The excerpt is for quick orientation; for the full debrief or code, follow up with `read_file` on the corresponding path.
- **read_adapter**: Read the current workspace adapter files.
- **read_meta_log**: Read your own previous decisions (last N entries from `meta/meta_log.jsonl`). Pass `sample_older=true` to additionally get up to 10 stratified-sampled older entries for retrospective audit.
- **read_user_instructions**: Read `meta/instructions/from_user.md` and check if it has changed since you last marked it seen.
- **ack_user_instruction**: After reading a NEW user instruction, write to `meta/instructions/ack.md` restating in your own words what the user asked for AND the directives/throttles/annotations/rewinds you applied in response. The user reads ack.md to confirm you understood them — they do not block on it, but it is your only channel to close the loop on a user instruction.
- **read_system_load**: CPU load, free RAM, free disk, and GPU utilization summary.
- **peek_experiment_log**: Read the last N lines of an in-flight experiment subprocess log.
- **park_experiment**: Move an experiment to parked state — soft-cancel; the dispatcher skips parked rows.
- **unpark_experiment**: Restore a parked experiment to the active queue.
- **set_priority**: Override an experiment's queue priority (higher runs first).
- **clear_experiment_block**: Clear a STALE `blocked:` error on a ready row so the dispatcher can assign it again — no clone, no economics change. Use ONLY when a transient precondition a worker flagged `blocked:` has since resolved (e.g. the shared suite is green again) and the row is self-stranded: the dispatcher won't assign a blocked row, so the flag never clears itself, and the worker that could clear it is never assigned. Do NOT clear a permanent block like `blocked: superseded by #<id>`. Verify the block is genuinely stale (read the row's error and confirm its precondition now holds), record why in the meta log, and no-op if the row is not actually blocked.
- **annotate_experiment**: Apply a leaderboard label — champion / control / quarantined / exploration / exploitation / ensemble-candidate / home-run-attempt.
- **issue_directive**: Append a directive to `meta/directives.md` for a target role. Valid roles: `strategist`, `worker`, `reporter`, `supervisor`, `builder`, `critic`, `tester`, `all`. The Phase 2 framework roles (`builder` / `critic` / `tester`) let you target a specific step of the harness build loop (e.g. "the tester must add a leakage assertion before the harness is accepted"). The `scope` parameter controls fan-out among same-role agents:
  - `standing` (default) — applies to every action of the target role until you supersede it. Use for per-action policies ("include the cold-client slice in every debrief", "prefer minute-bar resampling over second-bar", "no log-returns without clipping").
  - `one-shot` — the first same-role agent that acts on it calls `ack_directive` with the directive's id; later same-role turns see it as claimed and skip. Use for tasks that must happen exactly once across all same-role actors ("propose 3 new exploration experiments now", "write a leaderboard CSV under `reports/`").
  - `per-experiment:<id>` — applies only when an agent of the target role is working on experiment `<id>`, and is claimed by the first one to ack. Use for surgical fixes ("rerun #178 with seed=42", "the strategist mis-specified the loss for #214 — switch to MSE in the implement step").
- **retire_directive**: Mark a directive as no longer in force. Directives NEVER expire on their own — you own the lifecycle. Every turn, the digest shows you the full in-force list under `## IN-FORCE DIRECTIVES`; for each one you must AFFIRM (no action, still applicable), REVISE (issue a new one + retire the old with reason="superseded by <new id>"), or RETIRE (with reason). A directive issued during Phase 1 keeps getting injected into Phase 3 workers' prompts as noise until you retire it. Don't let directives sit unread.
- **write_note_to_user**: Append to `meta/notes_to_user.md`. The user is not expected to read it; it is a record.
- **set_throttle**: Set system throttle for cpu or gpu — `none` / `slow` / `halt-new`. Slow halves new-submit capacity; halt-new lets in-flight finish but stops new launches.
- **kill_experiment**: Cancel an in-flight GPU experiment. Reserved for cases with hard evidence (see Rules).
- **delete_path**: Delete a workspace path. Always backs up to `meta/backups/<ts>/` first. Reserved for stale caches you can prove are unneeded.
- **backup_path**: Back up a path to `meta/backups/<ts>/` without deleting.
- **request_phase_rewind**: Request a rewind to an earlier phase. Executes immediately; backs up affected artifacts; future phase agents see your evidence and the prior artifacts.
- **request_verification**: Commission the independent finding-verifier on an actionable result. Naming a `candidate` PINS it as the verifier's NEXT pick (jumping its adaptive order); set `priority='high'` to flag urgency; pass candidate='NONE' to let it pick adaptively. Latest request wins (NOT FIFO); the pin runs after any in-flight candidate finishes (no mid-candidate preemption). You are BOTH user and system here: your `steering` becomes the verifier's `from_user.md` (what to verify + what you care about — costs, tradeability, which benchmark, etc.), and its findings come back to you in the `## Verifier findings` digest section. Propagating the verdict is part of the job: when a candidate resolves, fold the actionable conclusion — especially a FAULT and its remedy — into a directive (and/or a leaderboard annotation) so the strategist and workers stop building on what failed an independent check (they can also read `verify/feedback_to_system.md` directly, but you decide what is worth steering on). Runs in a background thread (does not block the run). The system also auto-commissions a verification once enough experiments are analyzed if you have not asked.
- **request_run_end**: Request the run end gracefully. Executes immediately, subject to config floors (minimum runtime hours, minimum analyzed-experiment count, `allow_conductor_end_run`). The dispatcher then stops admitting new submissions, lets in-flight experiments finish, generates one final milestone report, and exits. Use ONLY when you have Python-verified evidence the run is exhausted (broad coverage per `research_state.md`, no improvement in the primary metric for a long stretch, AND the user's `description`/`target` goal has been substantively satisfied). Same evidence discipline as `request_phase_rewind`: write a script under `meta/scratch/` that demonstrates the exhaustion, run it, paste script and output into evidence. The tool refuses when floors are unmet.
- **report_to_user**: Call when your turn is complete.

## Building Deep Understanding (your primary job)

You cannot steer the strategist effectively without understanding what it is doing. The leaderboard alone is not enough — names are misleading, and the same metric value can come from very different mechanisms. Each turn, build or refresh your understanding of:

- **What the strategist is currently trying.** Read the playbook and the strategist's recent proposals via `read_experiment` for the last ~20 ids and `read_file` for `playbook.md`. Note the stated hypotheses, the mechanism classes, the direction the strategist believes it is heading, and any patterns in what it has stopped trying.
- **What has been actually tried.** For the leaderboard top-10 and for at least the last 10 completed experiments, read the debrief and the strategy code. Names are misleading; the code is ground truth. Pay attention to the candidate generator, the loss function, the validation split, and any teacher / control fallback logic.
- **What has failed and why.** Skim a handful of cancelled or errored experiments per turn. Patterns in failures often reveal framework bugs, unrealistic assumptions in the strategist's hypotheses, or systematic gaps in the candidate pool.
- **What is missing.** Read `research_state.md` for the Reporter's running synthesis of what has been tried, then form your own read of where coverage feels thin. Don't force the experiments into rigid named categories — different vocabularies describe the same family, and that's fine.
- **System invariants and inter-agent protocols.** Watch for protocol breakage between roles. Workers should be writing `experiments/<name>/run_experiment.py`, `experiments/<name>/results/metrics.json`, and `experiments/<name>/debrief.md` (or `analysis.md`); the reporter writes to `reports/milestone_NNN/` with the filenames downstream code expects; the strategist updates `playbook.md`. Verify a sample each turn — does every recently-completed experiment have a `metrics.json` and a debrief? Are status transitions matching on-disk reality (a row in `status=running` should have its `run_experiment.py` on disk)? Are annotations and directives referencing real ids? When you find a break, write a directive — or, if the convention is structural to the framework code, request a phase rewind with Python-verified evidence.
- **Where the research path is heading.** Read the latest milestone report in full plus a sample of older milestones (every 1-in-5, or all if few) to recover the long-arc narrative. Milestones give synthesis; experiments give ground truth. You need both.

Sampling is the right discipline — there can be hundreds of experiments. Read deeply where it matters and skim the bulk. Use `grep_file` and `shell_exec` (with small Python scripts under `meta/scratch/`) to summarize cheaply when needed.

**A note on context budget.** You have substantial but not unlimited context. The Initial Additional Context above is your orientation — it tells you where to look without reading anything in depth. Use tool calls to dive deeper only where it matters: leaderboard top-N, recent novel proposals, anomalies, the user's stated priority areas. Reading 10–20 experiments at depth per turn is healthy; reading 100 is wasteful. When you find yourself wanting to read everything, write a small Python script under `meta/scratch/` that summarizes the bulk and read the script's output instead — that's both cheaper and more rigorous than serial `read_file` calls.

## Resource hygiene — stalled-job detection

A GPU experiment can squat on a slot without doing GPU work: the model is on device, CUDA memory is allocated, the dispatcher sees `proc.poll() is None`, but the python is grinding through single-threaded CPU data prep, stuck in a model bug, or hanging in a DataLoader. The dispatcher only sees "process alive" and waits. You can see more by reading live state — and free the slot when no one else will.

Each turn, sample live process state and cross-reference with running rows:

- Via `shell_exec`: `nvidia-smi pmon -c 1` (per-process SM utilization) and `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` (per-process GPU memory). A process holding nontrivial GPU memory while its `sm` column reads `-` or `0` is not computing on the GPU.
- Map PIDs to experiment names. The dispatcher's earlier "Submitted local job <id> on GPU N (PID <pid>)" log lines tie job IDs to PIDs; `ps -ef | grep run_local.sh` shows the live bash wrapper for each running experiment (its cwd is `experiments/<name>`). Cross-reference against `read_board`'s running list.
- For suspected stalls, `peek_experiment_log` for the tail of the experiment's `local_job.<job>.out`. A log that has stopped growing while the python process is alive is a stronger stall signal than either symptom alone.

Don't pick a fixed "stale" threshold. Some experiments legitimately spend hours in CPU pre-train work (large `TimeSeriesDataSet` construction, hierarchical index building, custom dataset iteration). Use judgment: an experiment that has written nothing past Lightning's CUDA-init line, shows zero GPU SM activity, and is in that state across two of your consecutive turns is a stall worth killing. Be more patient with experiments whose code shows heavy pre-train data prep is expected, and impatient with ones that have already produced Lightning's training-loop header (epoch / step lines should keep coming).

When you confirm a stall with evidence, call `kill_experiment` and record the pmon snapshot, the log tail, and a guess at the cause (common patterns: `num_workers=0` on a large dataset, a model `__init__` doing CPU work the worker LLM didn't realize was O(N²), or a deadlock in a custom dataloader). The pattern is often more valuable than the individual kill — if a particular config shape produces stalls repeatedly, issue a `worker` directive that narrows future implementations to avoid it.

## Your Process

1. **Read user instructions.** The prompt assembly above this section already injects the contents of `meta/instructions/from_user.md`. That is the sole user-input channel and it outranks your own defaults. If the user has issued problem-specific exploration / exploitation guidelines, throttle preferences, or revert requests, comply — they are authoritative. If the file is empty, fall back to your built-in defaults plus the run's `description` and `target` fields.
2. **Read and audit your own past decisions for payoff.** Pull your last ~20 decisions in full via `read_meta_log`. Then sample 5–10 older entries from across the run with `sample_older=true` — pick a mix: decisions you considered important at the time, decisions you reversed, decisions the user approved or modified, and decisions whose stated rationale should be testable in retrospect. For each sampled decision, ask: did the predicted outcome match what actually happened? Did parking experiment X save budget that produced better work, or did similar ideas just reappear under different names? Did your directive shift the strategist's behavior, or was it ignored? Is the annotation still accurate? Update your sense of which kinds of intervention pay off and which do not. Append a brief retrospective entry to your meta-log under `decision_type: retrospective` capturing patterns of effective intervention and patterns of wasted effort. The user reads this.
2a. **Review the in-force directive list.** The digest's `## IN-FORCE DIRECTIVES` section lists every directive currently being injected into downstream agents' prompts. For each one this turn, decide: AFFIRM (still applicable — no action), REVISE (issue a replacement with `issue_directive` and `retire_directive` the old one with reason="superseded by <new id>"), or RETIRE (`retire_directive` with the reason it no longer applies). Common triggers for retirement: phase has moved on (a Phase 1 "all"-scope directive after Phase 1 ends), mechanism abandoned, payoff already realized, framework change made the directive moot. Directives NEVER expire on their own; you own the lifecycle. Letting an inapplicable directive carry over is the same kind of inattention as silent no-action.
3. **Read incoming notes** at `meta/notes_inbox.md`. The strategist or workers may have asked for things — for example, the strategist may have written "I really think this idea will work; please don't park it before it has run unless you have learned something I haven't." Weigh those notes; give the originator benefit of the doubt unless you have learned something they didn't.
4. **Build deep understanding** as described in the section above. This is the bulk of your turn. Do not shortcut it.
5. **Read system load** with `read_system_load`, and sample per-process GPU state per the Resource-hygiene section above. Decide whether throttle changes or any `kill_experiment` calls are warranted.
6. **Decide and act.** Typical actions: park / unpark experiments based on whether their hypothesis still has a chance against the current control; set priorities to surface high-information experiments; annotate the leaderboard so the strategist's view of "what's settled" stays accurate; issue directives that reflect what you have learned about the research; translate user instructions into role-targeted directives and acknowledge in `meta/instructions/ack.md`; set throttles when CPU or GPU oversubscription is genuinely hurting throughput; write a note to the user when something is worth recording even if no action is needed.
7. **If you choose not to act, log it deliberately.** Many turns warrant no intervention — the system is healthy, the strategist is on track, no user instruction has arrived, no resource problem is visible. That is fine, but it must be a deliberate decision based on what you read this turn, not a default. Use `write_note_to_user` to record what you read and why no intervention was warranted (or, equivalently, an explicit `issue_directive(target_role='all', message='no-action; system healthy', ...)`). Silent no-action is indistinguishable from inattention; the user cannot tell whether you saw the system was healthy or did not bother to look.
8. **Log every mutation** with `reason` and `evidence` fields. Evidence = file paths plus excerpts (often pulled from `read_experiment` outputs you just read), or the script you ran and its output. The audit log shows the absence of evidence; do not skip it.
9. **Call `report_to_user` and STOP.** A Conductor turn must end with `report_to_user`. Aim for one decisive turn — read the digest, do your audit, take 1-5 actions (directives / annotations / parks / throttles / a note to the user), then call `report_to_user` with a 2-4 sentence summary of what you found and what you did. The next phase-boundary or milestone or timer trigger will fire a fresh turn; you do not need to "finish everything" in this single call. If you find yourself reading file after file beyond ~20 reads without converging on a decision, you are likely exploring rather than steering — stop reading and call `report_to_user` with what you have. The slow timer (default 30 min) will give you another turn if more is needed.

## Rules

- **NEVER WAIT.** Do not pause the system. If you find a problem, write a directive about it and keep going. There is no human acknowledgement for anything you do.
- **NEVER STEER WITHOUT UNDERSTANDING.** Do not park, prioritize, annotate, or issue directives without first reading the experiment(s), debrief(s), or strategy code in question. Decisions made off leaderboard numbers alone tend to oscillate and erode the strategist's trust in your steering.
- **WATCH INTER-AGENT PROTOCOLS.** Workers, reporters, strategists, and phase agents are supposed to produce specific artifacts at specific paths with specific names. A silent convention break — filename mismatch, directory drift, missing required artifact, status not matching on-disk state — can sever feedback channels for hundreds of turns. No agent inside the system reads across roles to catch these. You do. Verify a sample of protocols each turn.
- **SILENT NO-ACTION IS INATTENTION.** Every turn must produce some record — a mutation with reason and evidence, or a `write_note_to_user` message describing what you read and why no intervention was warranted. The user reads the log to know whether you are paying attention.
- **AUDIT YOUR OWN DECISIONS FOR PAYOFF, NOT JUST CONSISTENCY.** Self-audit is not only "am I about to contradict myself?" — it is "did my prior decisions actually serve the goal?" Sample older decisions every turn and update your model of which kinds of intervention work and which do not.
- **Soft mutations only.** Park, do not delete. Backup before any overwrite that could lose data. The user reverts decisions by writing to `meta/instructions/from_user.md` — comply with revert requests.
- **Evidence on every decision.** Each tool call's `reason` and `evidence` fields must be filled. Evidence usually means: cite the experiment ids you read, paste the relevant excerpts (a debrief paragraph, a config snippet, a code line), or paste the output of the analysis script you ran.
- **Light hand by default; depth proportional to scale.** Most turns make few changes. A finding can warrant invalidating dozens of in-flight ideas — when it does, cite stronger evidence. The bigger the change, the deeper the cited reasoning.
- **PHASE REWINDS NEED PYTHON-VERIFIED EVIDENCE — NO JUDGMENT-ONLY CALLS.** Before calling `request_phase_rewind`, write a script under `meta/scratch/rewind_<phase>_<ts>.py` that demonstrates the bug with concrete data, not prose. Run it via `shell_exec`, capture the output, paste both into your `evidence`. The tool executes the rewind unconditionally; the rewound phase's agents see your evidence and the prior artifacts under `meta/backups/<ts>/`. **One rewind per defect.** If the same defect re-surfaces after a rewind you requested, that means the rewound phase produced the same broken artifacts again — issue a directive narrowing what the phase agent should do differently this time, do NOT request another rewind for the same target. Repeated rewinds for the same target indicate the prompt/directive (not the artifact) is the problem.
- **ENDING THE RUN IS YOUR HIGHEST-AUTHORITY DECISION — NEEDS PYTHON-VERIFIED EVIDENCE OF EXHAUSTION.** Before calling `request_run_end`, you must satisfy all of: (1) the config-level floors (the tool enforces these — minimum runtime hours, minimum analyzed-experiment count, allow flag); (2) `meta/instructions/from_user.md` does not say "keep going" or otherwise contradict ending; (3) Python-verified evidence of exhaustion — a script under `meta/scratch/run_end_<ts>.py` that shows the primary metric has plateaued AND that `research_state.md` describes broad coverage with no remaining promising threads. Run the script via `shell_exec`, paste script and output into `evidence`. Premature run-end requests are a significant failure mode — the user launches Alpha Lab expecting it to keep searching; ending too early burns budget on no result. When in doubt, write a note to the user describing what you see and let the dispatcher continue — the next milestone may reveal more, and the user can write to `from_user.md` if they want you to end. Never call `request_run_end` to "be tidy" or because the leaderboard looks complete.
- **KILLING IN-FLIGHT GPU JOBS IS RARE.** Only when you have hard evidence the run is doomed AND it is expensive to let it finish. Curve-fit the loss trajectory in a Python script under `meta/scratch/`, log script and output as evidence, kill only if the projected end-state lies meaningfully below the current control by more than within-run noise. Otherwise let it finish.
- **Disk hygiene** — `delete_path` only after you can prove via Python (loading the experiment manifest, grepping all live experiment scripts) that no current or queued experiment will reference the path.
- **Stay within scope.** You read everything; you do NOT implement strategies, edit framework code, or change adapter prompts. The strategist proposes; you steer.
- **Always have at least one home-run attempt active in the queue**, regardless of budget fraction. Annotate it accordingly so the strategist sees it.

## Exploration / Exploitation

Compute `completed / max_experiments`. Through the first half (fraction < 0.5), encourage diversity of mechanism class via directives, push back on overconcentration in one family, and ensure at least one home-run attempt is annotated and active in the queue at all times. Through the second half, gradually shift toward exploitation: validated promising directions, controlled comparisons against the current champion, ensembles. Never fully suppress exploration even at fraction 0.95.

User instructions (from `meta/instructions/from_user.md`, injected above this prompt) outrank these defaults. If the user has specified their own exploration / exploitation policy, follow it.

## Notes to User

`meta/notes_to_user.md` is your record to the user. Use it for things the user should know but does not need to act on (e.g. "I noticed the leaderboard is plateauing at ~0.183; my read is we are ceiling-bound on the available teacher and only ensembling will move it"). Append-only, reverse-chronological. The user is not expected to read these promptly.

## Additional Context

The Additional Context section below is regenerated each turn. It includes: the latest milestone summary digest, your last ~20 decisions, the current `meta/instructions/from_user.md` content, the current `meta/notes_inbox.md` content, the current annotations, the budget fraction, and a compact strategist-proposal digest covering the last ~20 ids. Read it carefully before deciding.
"""


# Tools the Conductor receives. Order doesn't matter — the schemas come from
# the registry. Workers/strategist receive `note_to_conductor`; the Conductor
# does NOT (it has nothing to write to itself with).
CONDUCTOR_TOOLS: list[str] = [
    "read_file",
    "grep_file",
    "shell_exec",
    "read_board",
    "read_experiment",
    "read_adapter",
    "read_meta_log",
    "read_user_instructions",
    "ack_user_instruction",
    "read_system_load",
    "peek_experiment_log",
    "park_experiment",
    "unpark_experiment",
    "set_priority",
    "clear_experiment_block",
    "annotate_experiment",
    "issue_directive",
    "retire_directive",
    "write_note_to_user",
    "set_throttle",
    "kill_experiment",
    "delete_path",
    "backup_path",
    "request_phase_rewind",
    "request_run_end",
    "request_verification",
    "report_to_user",
]


# ---------------------------------------------------------------------------
# Initial Additional Context (the digest the Conductor's prompt-builder
# injects each turn). Bounded by character budgets per subsection so the
# total stays well under the model's context window with room for tool
# results to accumulate during the turn.
# ---------------------------------------------------------------------------

# Per-section caps. Generous but bounded. A run with 500 experiments and 100
# milestones will produce a digest comfortably under 5K tokens (~20K chars).
_DIGEST_BUDGETS = {
    "user_directives": 3_000,
    "system_pulse": 600,
    "leaderboard": 4_000,
    "annotations": 1_500,
    "activity": 1_000,
    "recent_decisions": 3_000,
    "inbox": 2_000,
    "milestone_excerpt": 4_000,
    "in_force_directives": 8_000,
}


def build_conductor_context(
    workspace: str | Path,
    db: ExperimentDB | None,
    metric_key: str = "sharpe",
    trigger: str = "milestone",
    metric_direction: str = "maximize",
) -> str:
    """Construct the Initial Additional Context string for one Conductor turn.

    Pure function in spirit (returns a string) but reads many files, so it
    cannot be made fully pure. Each subsection is independently testable.

    Bounded: total length stays under ~25K chars on a healthy run with up
    to ~500 experiments and ~100 milestones; per-subsection caps prevent
    a single subsection (e.g. a giant inbox) from blowing the budget.
    """
    sections: list[str] = []

    sections.append(_section_user_directives(workspace))
    sections.append(_section_system_pulse(workspace, db, trigger))
    sections.append(_section_in_force_directives(workspace))
    if db is not None:
        sections.append(
            _section_leaderboard(workspace, db, metric_key, metric_direction)
        )
    sections.append(_section_annotations(workspace))
    if db is not None:
        sections.append(_section_activity(db))
    sections.append(_section_recent_decisions(workspace))
    sections.append(_section_inbox(workspace))
    sections.append(_section_latest_milestone_excerpt(workspace))
    sections.append(_section_verifier_findings(workspace))

    return "\n\n".join(s for s in sections if s.strip())


def _section_verifier_findings(workspace: str | Path) -> str:
    """The verifier's live findings (verify/feedback_stream.md), surfaced so the Conductor — as
    the system — can fold them into directives/annotations. Bounded; empty if it hasn't run."""
    from pathlib import Path as _P
    stream = _P(workspace) / "verify" / "feedback_stream.md"
    if not stream.exists():
        return ""
    try:
        body = stream.read_text(errors="replace")
    except OSError:
        return ""
    tail = "\n".join(ln for ln in body.splitlines() if not ln.startswith("#"))[-4000:]
    if not tail.strip():
        return ""
    return ("## Verifier findings (from the independent verifier you commission via "
            "`request_verification`; evidence to fold into directives/annotations)\n" + tail)


def _strip_from_user_boilerplate(text: str) -> str:
    """Return the user-written content of ``from_user.md``, stripping
    only the bootstrap boilerplate.

    The file is bootstrapped with a multi-line `#`-commented block that
    explains how to use it. If we forward that raw, the model treats
    the boilerplate as "the user's instructions". Strategy:

    1. If the file exactly matches the default boilerplate → return ""
       (the user has not written anything).
    2. Otherwise strip any **leading** `#`-commented lines that match
       the boilerplate header verbatim (so users who add free text BELOW
       the default get clean instructions without the help block), but
       preserve everything else — including `#`-prefixed lines the user
       wrote intentionally (markdown headers, their own annotations).
    """
    if not text:
        return ""
    from alpha_lab.meta_layout import DEFAULT_FROM_USER

    if text.strip() == DEFAULT_FROM_USER.strip():
        return ""

    # Strip exact boilerplate header lines (and blank lines among them)
    # from the front, then return the remainder as-is.
    boilerplate_lines = {
        line.rstrip()
        for line in DEFAULT_FROM_USER.splitlines()
        if line.lstrip().startswith("#")
    }
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].rstrip()
        if s in boilerplate_lines or s == "":
            i += 1
            continue
        break
    return "\n".join(lines[i:]).strip()


def _section_user_directives(workspace: str | Path) -> str:
    """Single source of user input: meta/instructions/from_user.md.

    The user writes free-text directives there at any time — before launch
    for run-level baseline guidance, or mid-run to nudge in real time. The
    Conductor reads this on every turn and treats it as authoritative.
    Empty file means no user instructions; the Conductor falls back to its
    built-in defaults. The bootstrap-time comment block is stripped before
    we show the content to the model so it doesn't get treated as a real
    instruction.
    """
    fu, _is_new = ct.read_from_user_diff(workspace)
    fu_clean = _strip_from_user_boilerplate(fu or "")
    fu_block = fu_clean or "(empty — no user instructions)"
    body = (
        f"## USER INSTRUCTIONS\n\n"
        f"_From `meta/instructions/from_user.md` (the user writes here at any "
        f"time; you read it on every turn). Comment-only content is stripped — "
        f"if you see '(empty — no user instructions)' below, the user has not "
        f"written anything yet:_\n\n{fu_block}"
    )
    return body[: _DIGEST_BUDGETS["user_directives"]]


def _section_system_pulse(
    workspace: str | Path, db: ExperimentDB | None, trigger: str
) -> str:
    parts = [f"## SYSTEM PULSE", f"trigger: {trigger}"]
    if db is not None:
        summary = db.board_summary()
        completed = summary.get("done", 0) + summary.get("analyzed", 0)
        # The Conductor sees max via the agent system through TaskConfig
        # injection; here we report the raw counts instead.
        parts.append(
            f"counts: " + " | ".join(f"{k}={v}" for k, v in sorted(summary.items()))
        )
        parts.append(f"completed: {completed}")
    th = ml.read_throttle(workspace)
    parts.append(f"throttle: gpu={th['gpu']} cpu={th['cpu']}")
    parts.append("load: " + ct.read_system_load(workspace))
    return "\n".join(parts)[: _DIGEST_BUDGETS["system_pulse"]]


def _section_leaderboard(
    workspace: str | Path,
    db: ExperimentDB,
    metric_key: str,
    metric_direction: str = "maximize",
) -> str:
    rows = db.leaderboard(metric_key, top_n=30, direction=metric_direction)
    if not rows:
        return "## LEADERBOARD\n(empty)"
    annotations = ct.read_annotations(workspace)
    lines = ["## LEADERBOARD (top 30)", "", f"| id | name | {metric_key} | annotation |", "|---|---|---|---|"]
    for exp in rows:
        try:
            results = json.loads(exp.results_json or "{}")
            metric = results.get(metric_key, "?") if isinstance(results, dict) else "?"
        except ValueError:
            metric = "?"
        annotation = annotations.get(str(exp.id), "")
        name = exp.name[:60]
        lines.append(f"| #{exp.id} | {name} | {metric} | {annotation} |")
    text = "\n".join(lines)
    return text[: _DIGEST_BUDGETS["leaderboard"]]


def _section_in_force_directives(workspace: str | Path) -> str:
    """Surface every currently-in-force directive to the Conductor each
    turn so the Conductor owns the lifecycle. Nothing carries over
    silently: each turn the Conductor must look at this list and decide,
    per directive, whether to AFFIRM (no action), REVISE (issue a new
    one + ``retire_directive`` the old one), or RETIRE
    (``retire_directive``) it. Retired directives stop appearing here
    on the next turn."""
    directives = ct.parse_directives(workspace)
    retired = ct.retired_directive_ids(workspace)
    in_force = [d for d in directives if d["id"] not in retired]
    if not in_force:
        return (
            "## IN-FORCE DIRECTIVES (you must review each one this turn)\n"
            "(none — issue new directives via `issue_directive` if needed)"
        )
    # Group by role for readability; preserve recency (parse_directives
    # returns most-recent first because the file is written that way).
    lines = [
        "## IN-FORCE DIRECTIVES (you must review each one this turn)",
        "",
        "**Lifecycle rule.** Every directive listed below is currently being "
        "injected into the prompt of the agent whose role it targets. For "
        "each one, this turn you MUST either:",
        "  - **AFFIRM** (no action — still applicable, still being followed),",
        "  - **REVISE** (issue a new directive with `issue_directive`, then "
        "`retire_directive` the old one with reason='superseded by <new id>'), or",
        "  - **RETIRE** (`retire_directive` with the reason it no longer "
        "applies — e.g. phase moved on, mechanism abandoned, payoff "
        "already realized).",
        "",
        "Don't let directives sit unread. A directive issued during Phase 1 "
        "and targeting `all` will keep being injected into every Phase 3 "
        "worker's prompt as noise unless you retire it.",
        "",
    ]
    for d in in_force:
        body = (d.get("body") or "").strip()
        # Trim each body so the budget covers many directives, not one
        # giant one.
        if len(body) > 600:
            body = body[:600].rstrip() + " […]"
        lines.append(
            f"### {d['id']}  role={d['role']}  scope={d['scope']}  "
            f"issued={d['timestamp']}"
        )
        lines.append(body)
        lines.append("")
    return "\n".join(lines)[: _DIGEST_BUDGETS["in_force_directives"]]


def _section_annotations(workspace: str | Path) -> str:
    details = ct.read_annotation_details(workspace)
    if not details:
        return "## ANNOTATIONS\n(none)"
    # Top-line: counts by label (population-level summary).
    counts: dict[str, int] = {}
    for rec in details.values():
        lbl = rec.get("label", "")
        if lbl:
            counts[lbl] = counts.get(lbl, 0) + 1
    sections = ["## ANNOTATIONS (counts by label)"]
    sections.extend(
        f"- {label}: {n}"
        for label, n in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    # Most-recent-with-reason: helps the Conductor recall *why* it
    # labeled each row when planning the next turn. Sorted by ts
    # descending; rows without a reason (legacy entries) are skipped.
    with_reason = [
        (eid, rec) for eid, rec in details.items()
        if (rec.get("reason") or "").strip()
    ]
    with_reason.sort(key=lambda er: er[1].get("ts", 0.0), reverse=True)
    if with_reason:
        sections.append("")
        sections.append("**Recent annotations (with rationale):**")
        for eid, rec in with_reason[:10]:
            sections.append(
                f"- #{eid} [{rec.get('label','')}] — {rec.get('reason','')[:200]}"
            )
    return "\n".join(sections)[: _DIGEST_BUDGETS["annotations"]]


def _section_activity(db: ExperimentDB) -> str:
    """Recent activity: most recently updated rows (last 10)."""
    rows, _total = db.list_experiments(limit=10, offset=0)
    lines = ["## RECENT ACTIVITY (last 10 by updated_at)"]
    for exp in rows:
        when = _dt.datetime.fromtimestamp(exp.updated_at).isoformat(timespec="seconds")
        ann_marker = ""
        if exp.parked_at is not None:
            ann_marker = " [parked]"
        lines.append(
            f"- #{exp.id} {exp.name[:50]} | {exp.status} | updated {when}{ann_marker}"
        )
    body = "\n".join(lines)
    return body[: _DIGEST_BUDGETS["activity"]]


def _section_recent_decisions(workspace: str | Path) -> str:
    entries = ct.meta_log_read(workspace, last_n=5, sample_older=False)
    if not entries:
        return "## RECENT CONDUCTOR DECISIONS\n(none yet — first turn)"
    lines = ["## RECENT CONDUCTOR DECISIONS (last 5)"]
    for e in reversed(entries):
        when = _dt.datetime.fromtimestamp(e.get("ts", 0)).isoformat(timespec="seconds")
        dt = e.get("decision_type", "?")
        target = e.get("target", "")
        reason = (e.get("reason", "") or "").replace("\n", " ").strip()[:200]
        lines.append(f"- `{when}` **{dt}** {target} — {reason}")
    body = "\n".join(lines)
    return body[: _DIGEST_BUDGETS["recent_decisions"]]


def _section_inbox(workspace: str | Path) -> str:
    inbox = ct.read_notes_inbox(workspace, max_chars=_DIGEST_BUDGETS["inbox"])
    if not inbox.strip():
        return "## NOTES INBOX (other agents → you)\n(empty)"
    return f"## NOTES INBOX (other agents → you, most recent at end)\n\n{inbox}"


def _section_latest_milestone_excerpt(workspace: str | Path) -> str:
    """Find the latest milestone report and return an excerpt.

    Tolerant of two filename conventions seen in the wild (``report.md`` from
    the built-in adapters and ``milestone_report.md`` from some customized
    Phase-0 outputs). The Conductor needs to see whichever one exists.
    """
    reports_dir = Path(workspace) / "reports"
    if not reports_dir.is_dir():
        return "## LATEST MILESTONE\n(no reports directory yet)"
    milestone_dirs = sorted(
        (d for d in reports_dir.iterdir()
         if d.is_dir() and d.name.startswith("milestone_")),
        key=lambda d: d.name,
    )
    if not milestone_dirs:
        return "## LATEST MILESTONE\n(no milestones yet)"
    latest = milestone_dirs[-1]
    for fname in ("milestone_report.md", "report.md"):
        p = latest / fname
        if p.exists():
            try:
                content = p.read_text()
            except OSError:
                continue
            excerpt = content[: _DIGEST_BUDGETS["milestone_excerpt"]]
            return f"## LATEST MILESTONE: {latest.name}\n\n{excerpt}"
    return f"## LATEST MILESTONE: {latest.name}\n(report file not found)"


# ---------------------------------------------------------------------------
# Conductor class.
# ---------------------------------------------------------------------------


# Default initial-message templates per trigger. Brief — the system prompt
# carries the full guidance; the initial message just orients the agent
# to why it was called this turn.
_INITIAL_MESSAGES = {
    "milestone": (
        "A milestone report just finished. Review what changed since your last "
        "turn, audit your past decisions for payoff, build deep understanding of "
        "what the strategist is now doing, and steer with a light hand. Go."
    ),
    "timer": (
        "Slow-timer trigger. The system has been running for a while since "
        "your last turn. Determine which phase is active (read "
        "`logs/pipeline.jsonl` and the workspace state), then steer that "
        "phase: if Phase 1 is in flight, look at `scripts/`, `notes/`, and "
        "`learnings.md` (or lack thereof) and check whether the agent is "
        "making progress or stuck; if Phase 2, check the framework dir; "
        "if Phase 3, read the leaderboard and recent debriefs. Issue "
        "directives if you see drift or stuckness, write a note to the "
        "user about what you observed, or deliberately log no-action. Go."
    ),
    "phase0_done": (
        "Phase 0 (adapter customization) just finished. Review the customized "
        "adapter against the user's research goal — does the adapter actually "
        "express what the user wants? If misaligned, issue directives or, if "
        "structural, request a phase rewind with Python-verified evidence. Go."
    ),
    "phase1_done": (
        "Phase 1 (data exploration) just finished. Review `learnings.md`, "
        "`data_report/`, scripts, and plots. Look for leakage signs, sloppy "
        "splits, target definitions that don't match the user's goal, missing "
        "data quality checks. Issue directives for downstream phases or "
        "request a phase rewind if exploration is shallow or wrong-targeted. Go."
    ),
    "phase2_done": (
        "Phase 2 (framework / harness) just finished. Review the framework "
        "directory (its name varies by adapter — common values are `backtest/`, "
        "`harness/`, `framework/`; the adapter manifest at `adapter/manifest.json` "
        "names it under `experiment.framework_dir`), the tests inside it, and "
        "the critic's review file (often `<framework_dir>/review.md` or "
        "`harness/review.md`). Specifically scan for evaluation-protocol "
        "issues: train/val/test split conventions, target leakage, look-ahead, "
        "metric definitions. Issue directives or request a rewind if needed. Go."
    ),
}


def build_conductor(
    main_provider: "Provider",
    config: "TaskConfig",
    workspace: str,
    db: "ExperimentDB",
    adapter: "DomainAdapter | None",
    event_callback: Callable[[Any], None],
) -> "Conductor | None":
    """Construct a Conductor with the right provider, or return ``None``.

    Returns ``None`` if ``no_conductor=True`` or ``adapter is None``
    (both Phase 3 and the phase-boundary entry points treat None as
    "Conductor disabled"). Centralizes the provider-selection logic so
    ``run.py`` and ``Dispatcher`` don't drift.

    The Conductor can run on its own provider — defaults to Bedrock/opus
    regardless of what the rest of the pipeline uses. Only build a
    separate provider when the *provider type* differs (e.g. main is
    openai, conductor is bedrock); the same provider can serve multiple
    Claude/GPT models so there is no need to duplicate if the types match.
    """
    if config.pipeline.phase3.no_conductor or adapter is None:
        return None

    cond_provider_name = config.conductor_provider or config.provider
    if cond_provider_name and cond_provider_name != config.provider:
        from alpha_lab.client import get_provider
        try:
            conductor_provider = get_provider(cond_provider_name)
            logger.info(
                "Conductor will use a separate %s provider "
                "(main pipeline uses %s)",
                cond_provider_name, config.provider,
            )
        except Exception as e:
            # If the second provider can't be built (auth, network),
            # fall back to the main provider rather than crashing.
            logger.warning(
                "Could not build separate Conductor provider %s "
                "(%s); falling back to the main provider.",
                cond_provider_name, e,
            )
            conductor_provider = main_provider
    else:
        conductor_provider = main_provider

    return Conductor(
        provider=conductor_provider,
        config=config,
        workspace=workspace,
        db=db,
        adapter=adapter,
        event_callback=event_callback,
    )


class Conductor:
    """Meta-agent that steers the pipeline.

    Lifecycle: instantiated once near the top of the run (in ``run.py``)
    and reused across all phases. Triggered after each Phase 0/1/2
    completes (called from ``run.py``) and after each Phase 3 milestone,
    plus on a slow timer in Phase 3 (called from ``Dispatcher``). Each
    ``steer_*`` call runs a short-lived AgentLoop; state between turns
    lives in the ``meta/`` filesystem.
    """

    def __init__(
        self,
        provider: Provider,
        config: TaskConfig,
        workspace: str,
        db: ExperimentDB,
        adapter: DomainAdapter | None,
        event_callback: Callable[[AgentEvent], None],
    ) -> None:
        self.provider = provider
        self.config = config
        self.workspace = workspace
        self.db = db
        self.adapter = adapter
        self.event_callback = event_callback
        # Serializes ``_run_steer`` calls. The timer sidecar (started
        # during Phase 1/2) and the boundary call (e.g. ``steer_phase1``
        # at phase end) can otherwise race: the sidecar may still be
        # inside ``_run_steer`` when the with-block tears it down,
        # because the join() has a finite timeout. Two simultaneous
        # turns clobber each other's directive writes and confuse the
        # event log. Reentrant so a future caller invoking another
        # ``steer_*`` from inside a tool wouldn't deadlock.
        self._steer_lock = threading.RLock()
        ml.ensure_meta_layout(workspace)

    # ------------------------------------------------------------------
    # Public per-trigger entry points. Each is a one-liner that delegates
    # to _run_steer with the correct trigger string and log_name.
    # ------------------------------------------------------------------

    def steer_phase0(self) -> str:
        return self._run_steer("phase0_done", "conductor_phase0")

    def steer_phase1(self) -> str:
        return self._run_steer("phase1_done", "conductor_phase1")

    def steer_phase2(self) -> str:
        return self._run_steer("phase2_done", "conductor_phase2")

    def steer_milestone(self) -> str:
        return self._run_steer("milestone", "conductor_milestone")

    def steer_timer(self) -> str:
        return self._run_steer("timer", "conductor_timer")

    # ------------------------------------------------------------------
    # Core run helper. Mirrors Supervisor._run_review.
    # ------------------------------------------------------------------

    def _run_steer(self, trigger: str, log_name: str) -> str:
        with self._steer_lock:
            return self._run_steer_locked(trigger, log_name)

    def _run_steer_locked(self, trigger: str, log_name: str) -> str:
        if self.config.pipeline.phase3.no_conductor:
            # Defense-in-depth: even if the dispatcher somehow forgot to
            # gate on no_conductor, this method short-circuits silently.
            logger.debug("Conductor is no-op (no_conductor=True); skipping %s", trigger)
            return ""

        metric_key = (
            self.adapter.metric.primary_metric
            if self.adapter is not None
            else self.config.pipeline.phase3.convergence_metric or "sharpe"
        )
        metric_direction = (
            self.adapter.metric.direction
            if self.adapter is not None
            else "maximize"
        )

        digest = build_conductor_context(
            workspace=self.workspace,
            db=self.db,
            metric_key=metric_key,
            trigger=trigger,
            metric_direction=metric_direction,
        )

        # Adapter may supply a phase3_conductor.md addendum the customizer
        # wrote during Phase 0. Append after the system prompt — never
        # replace it.
        adapter_addendum = self._read_adapter_addendum()

        def prompt_builder(
            workspace: str | None,
            learnings: str | None,
            config: Any | None = None,
        ) -> str:
            parts = [CONDUCTOR_SYSTEM_PROMPT]
            if adapter_addendum:
                parts.append(
                    "## Adapter-customized addendum\n\n" + adapter_addendum.strip()
                )
            if workspace:
                parts.append(f"## Workspace\n`{workspace}`")
            parts.append("## Initial Additional Context\n\n" + digest)
            return "\n\n".join(parts)

        tools = get_tool_schemas(CONDUCTOR_TOOLS)

        # The Conductor uses its own model. If ``conductor_model`` is empty
        # the config asks us to inherit ``model`` (the rest of the pipeline's
        # model). Otherwise route to the Conductor-specific one — typically
        # opus, because the Conductor's job (deep cross-experiment audit,
        # retrospective evaluation, structured judgment) benefits from the
        # strongest available reasoning regardless of what the workers use.
        conductor_model = self.config.conductor_model or self.config.model

        context = ContextManager(
            provider=self.provider,
            model=conductor_model,
            workspace=self.workspace,
            summarization_threshold_tokens=self.config.context_summarization_threshold_tokens,
            learnings_summary_threshold_tokens=self.config.learnings_summary_threshold_tokens,
        )

        agent = AgentLoop(
            provider=self.provider,
            model=conductor_model,
            context=context,
            event_callback=self.event_callback,
            reasoning_effort=self.config.conductor_reasoning_effort,
            config=self.config,
            tools=tools,
            prompt_builder=prompt_builder,
            log_name=log_name,
            min_report_attempts=1,
            adapter=self.adapter,
            db=self.db,
        )

        # Phase label: use the trigger string so phase 0/1/2 boundary
        # turns aren't mislabeled. Timer triggers can fire during any
        # phase (sidecar in Phase 1/2 + dispatcher in Phase 3) so they
        # get a neutral "conductor" label rather than a wrong-phase
        # hardcode. Milestone is Phase 3-only and stays "phase3".
        if trigger.endswith("_done"):
            phase_label = trigger.replace("_done", "")
        elif trigger == "milestone":
            phase_label = "phase3"
        else:  # timer or anything else cross-phase
            phase_label = "conductor"
        self.event_callback(PhaseEvent(
            phase=phase_label, step="conductor", status="starting",
            detail=f"Conductor steer: {trigger}",
        ))

        initial_message = _INITIAL_MESSAGES.get(
            trigger, _INITIAL_MESSAGES["timer"]
        )

        try:
            report = agent.run(initial_message)
        except Exception as e:
            # Per design: Conductor crashes do NOT halt the dispatcher.
            # Log, emit, return empty so the dispatcher continues.
            logger.error("Conductor turn (%s) crashed: %s", trigger, e)
            self.event_callback(PhaseEvent(
                phase=phase_label, step="conductor", status="error",
                detail=f"Conductor turn crashed: {e}",
            ))
            return ""

        self.event_callback(PhaseEvent(
            phase=phase_label, step="conductor", status="completed",
            detail=f"Conductor steer complete: {trigger}",
        ))

        # Always re-render the human digest of meta_log so the user reading
        # meta_log.md sees the freshest state, even if the agent forgot to
        # re-render after some intermediate decision.
        ct.meta_log_render_md(self.workspace)
        return report or ""

    def _read_adapter_addendum(self) -> str:
        if self.adapter is None:
            return ""
        # The DomainAdapter loader populates ``prompts`` with whichever
        # phase keys exist on disk. ``phase3_conductor`` is optional —
        # adapters that don't ship one return an empty string here, and
        # the Conductor falls back to its built-in canonical prompt.
        return self.adapter.prompts.get("phase3_conductor", "") or ""
