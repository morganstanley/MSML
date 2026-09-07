# Overnight audit — verifier→alpha_lab integration + context.py fix

Watching: gpt-5.5 integrated run `workspace_etfflow_gpt55v` (run task `b62j42l7a`, monitor `b63cj5w22`).
Goal: adversarially audit everything I changed, find + fix real bugs, log each with evidence here for morning review.

Bug surface = my uncommitted changes: verifier.py, dispatcher.py (verifier glue), conductor_tools.py
(request_verification), tools.py (handler), conductor.py (CONDUCTOR_TOOLS + verifier-findings section),
config.py (verifier fields), context.py (parallel-batch orphan-backoff fix), scripts/verify_*.py.

Method: read each change, ask "how does this break", verify with a tool call, fix, re-verify.

---

## Bugs found & fixed

(appended below as I find them — each: location, what's wrong, why it bites, fix, verification)

### #0 (already fixed earlier this session) — context.py trim_history orphaned parallel tool-call batches
- **Where:** `src/alpha_lab/context.py` `trim_history` orphan-backoff.
- **Bug:** backoff skipped only tool-*results*; a mid-loop summarize splitting a parallel call batch
  `[call_A, call_B, out_A, out_B]` dropped `call_A` but kept `out_A` → OpenAI 400 "No tool call found
  for function call output" → crashed every long agent turn (4 distinct events in run #1, incl. the
  Phase 1 agent's final turn → no `data_report/*.md` → "Cannot run Phase 2" dead-end).
- **Fix:** added `_is_tool_call_item`; backoff now skips the whole tool block (calls + results).
- **Verified:** unit test — split landing among parallel outputs leaves no orphan; simple `[call,out]`
  case still intact. Live: Phase 0 conductor steer (first 400 culprit) completed clean on restart.
- **END-TO-END CONFIRMED (run #2):** Phase 1 completed 23:55 with **0 orphan 400s** (vs 4 in run #1)
  and 0 agent crashes; wrote all 4 `data_report/*.md` (findings/schema/statistics/deep_learning_blueprint);
  `detect_phase1_complete=True` → dead-end cleared, Phase 2 building.

### #1 (FOUND — fix proposed, awaiting user OK) — patch_adapter_file's `git add -A` stages the whole tree
- **Where:** `src/alpha_lab/tools.py:2239` — `patch_adapter_file` runs
  `git add -A && git commit -m 'checkpoint before supervisor patch' --allow-empty` before overwriting an adapter file.
- **Bug:** `git add -A` stages the ENTIRE working tree (source edits, configs, `run_gpt55v.out`, this audit
  file) into the checkpoint commit, under the repo's git identity — not just the `adapter/` dir it's protecting.
  Pollutes the branch (11 such commits today during run #1). Non-destructive (commits, never deletes) but noisy.
- **Origin (evidence):** present since ≥2026-04-15 (`8465462`); absent from `ms-tech/main` AND `origin/main`
  → local-branch-only, pre-this-session. Not from the verifier integration.
- **Fix (APPLIED + TESTED, user-approved):** `git add adapter/` would've been wrong — the adapter dir is gitignored,
  so git can't track it (proof: checkpoint `819c42f` actually committed `run_gpt55v.out` + `context.py`, NOT the
  adapter). Replaced the git checkpoint entirely with a plain file backup: `adapter/<file>` →
  `adapter/.backups/<file>.<ts>.bak` before overwrite. Zero git side effects. Verified: `patch_adapter_file` on a
  throwaway ws overwrites + preserves the original in `.backups/`, no git invoked; `tools.py` compiles; the only
  remaining "git add -A" hit is the explanatory comment. Live on next restart (running process still has old code).

### #2 (FIXED + verified) — verifier auto-trigger re-fired back-to-back instead of once-per-batch
- **Where:** `src/alpha_lab/dispatcher.py` `_should_run_verifier` auto branch.
- **Bug:** `analyzed >= verify_after_n and analyzed != ran_at`. `_maybe_run_verifier` runs every dispatcher
  loop; `analyzed` keeps growing as experiments finish during each (expensive) verifier run, so after the
  first fire it re-fired on EVERY later completion → the verifier (≤10 candidates × notebooks × 3 gpt-5.5
  models × 2 rounds) ran back-to-back indefinitely past the threshold, and starved the Conductor's explicit
  `request_verification` of the one-in-flight slot. Contradicts the "once if the Conductor hasn't asked" intent.
- **Fix:** fire once per fresh batch — `analyzed >= ran_at + verify_after_n` (first fire at ≥N). A Conductor
  request still fires immediately and updates ran_at, so auto won't pile on right after a manual run.
- **Verified:** sim over analyzed=1..45 — OLD fired 36× (10,11,…,45), NEW fires 4× (10,20,30,40); compiles.
  Live on next restart (verifier hasn't fired yet — Phase 3 not reached, so old behavior never bit).

### #3 (FIXED + verified) — watchdog kill-marker could terminate the integrated orchestrator
- **Where:** `src/alpha_lab/verifier.py` `_consume_watchdog_markers`.
- **Bug:** the kill loop skipped lines containing `verify_workspace` (the STANDALONE orchestrator) but not the
  INTEGRATED orchestrator `run.py`, which contains "python" and passes the nb_run/ipykernel/python filter. If
  the watchdog wrote a broad kill-token (WATCHDOG_PROMPT allows the candidate slug, which can derive from the
  workspace name) matching run.py's args, it would `kill -9` the main dispatcher — killing the whole run. The
  docstring claimed "(never the orchestrator)" but the code didn't guarantee it. Live risk (watchdog_interval=600).
- **Fix:** skip by PID (`os.getpid()` + `os.getppid()`), not name — a name-skip on "run.py" is unsafe because
  "run.py" is a substring of the verifier's own `nb_run.py` jobs (would spare the very jobs it must kill).
  Also parse pid/args robustly (skip the ps header line).
- **Verified:** sim — orchestrator PID never selected even with the token in its args; legit hung `nb_run.py`
  job IS killable; standalone-orchestrator + unrelated python spared. verifier.py compiles. Live on next restart.

### #4 (FIXED + verified) — verifier `_model_for` misroutes role models when a slug contains "worker"/"critic"
- **Where:** `src/alpha_lab/verifier.py` `_model_for`.
- **Bug:** `if "worker" in log_name` / `if "critic" in log_name` substring-matched the WHOLE log_name
  (`verifier_<role>_<slug>_r<n>`), so a candidate whose slug contains "worker"/"critic" routed its
  arbiter/watchdog (User-Rep) to the wrong role model. Harmless when the 3 role models are identical (this
  run: all gpt-5.5) but wrong for the "all 3 can differ" config you asked for.
- **Fix:** extract the ROLE token (`log_name.split("_",2)[1]`), match exactly; the slug (index 2) is ignored.
- **Verified:** all 7 real log_names route correctly; `verifier_arbiter_my-worker-exp` and
  `verifier_watchdog_my-critic-strat` now stay User-Rep. Compiles.

### #5 (FIXED + verified) — verifier candidate menu excluded terminal 'done' experiments
- **Where:** `src/alpha_lab/verifier.py` `_build_menu` SQL.
- **Bug:** menu queried `WHERE status='analyzed'`, but the kanban promotes `analyzed -> done` (terminal, still
  carrying results_json) and the trigger + whole dispatcher count `("analyzed","done")` together. An experiment
  promoted to 'done' is counted toward the verify threshold yet invisible to the menu the verifier reads — it
  could fire on a count it can't fully see, and can never verify a fully-completed ('done') experiment.
- **Fix:** `WHERE status IN ('analyzed','done') AND results_json IS NOT NULL`. (No parked_at filter — standalone
  verifier may open an older DB read-only where that column doesn't exist.)
- **Verified:** synthetic DB — menu returns analyzed+done-with-results, excludes running/null-results. Compiles.

### #6 (FIXED + verified, user-flagged) — verifier feedback undiscoverable by other agents + Conductor "spread" implicit
- **Where:** `prompts.py` `_FILES_IN_WORKSPACE_SECTION`; `conductor.py` request_verification description.
- **Gap:** only the Conductor saw verifier findings (its `## Verifier findings` digest); strategist/workers/
  reporter/fixer were never told `verify/feedback_to_system.md` / `feedback_stream.md` exist — so "if they look"
  was empty. And the Conductor prompt framed folding-into-directives as a capability, not a duty.
- **Fix:** (a) added a `verify/...` entry to the shared `_FILES_IN_WORKSPACE_SECTION` injected into all 5 Phase-3
  roles (one edit, adapter-independent, judgment-not-rules style) so agents can read verifier findings directly and
  treat a FAULT as a reason not to keep building on an experiment; (b) made propagation an explicit Conductor duty
  in the request_verification description (fold the actionable verdict + remedy into a directive/annotation).
- **Verified:** prompts.py + conductor.py compile; pointer present at prompts.py:83. Live on next restart.

### Regression check (all 6 fixes) + pre-existing stale test repaired
- Ran the focused suite for every edited module (context/dispatcher/tools/prompts/conductor/conductor_tools/config):
  **338 passed, 0 regressions from my fixes.**
- The 1 failure was PRE-EXISTING, not mine: `test_flag_set_after_milestone` stubs `OutputGenerator` so `report.md`
  is never written, while a defensive rollback in `_maybe_generate_report` (a path my edits don't touch — I only
  changed `_should_run_verifier`) correctly declines to set `_milestone_just_finished` without a report.md. Product
  code is correct; the test expectation was stale (fails regardless of my edits).
- Repaired the test to simulate a successful report (write report.md before the call) → passes. Suite now green.

## Deployment
- **RESTART #3 @ 00:57:54** made all fixes LIVE (run.py PID 2703634, run task `bulbww70r`, monitor `b63cj5w22`).
  Done at Phase-3 start with 0 experiments committed (clean kill). On restart Phase 1+2 skip as complete; the
  leak-safe `backtest/` framework persists on disk. context.py was already live from run #2; this restart adds
  the git-checkpoint, verifier auto-fire, watchdog PID-skip, model role-token, menu analyzed+done, and the two
  prompt fixes. The verifier will now fire with fixed behavior — its first live run is the integration's real test.

## Run-watch findings (Phase 2 leakage hardening, run #3)
- **Conductor caught 3 progressively finer Phase-2 leaks** (each a NEW defect, "one rewind per defect"):
  (1) `backtest/` never built; (2) engine passed `target_*` columns to arbitrary Strategy code (proved with a
  red-team `EchoTargetStrategy`); (3) sanitizer still passed raw non-causal `mid_clean` + allowed the future-return
  `ret_col` under arbitrary names (e.g. `custom_fwd_ret_60m`). This is the user's #1 concern (no look-ahead)
  enforced relentlessly before Phase 3. Progressive hardening, NOT a churn loop — but watch for a 4th rewind.
- **Phase-2 critic rubber-stamps (CONFIRMED via Conductor audit; not fixed):** returned "PASS" with a stale
  "6 passed" claim while 25 tests existed and real leaks remained. The Conductor is the only agent catching these
  and is now directing the critic to run adversarial checks. If the critic keeps rubber-stamping despite that
  directive, `phase2_critic.md` likely needs strengthening (like the verifier's critic: reproduce-and-falsify in
  own code) — a prompt fix → needs a restart. Watching this round's critic verdict before acting.
- **phase2_done boundary audit skipped after an in-run rebuild (OBSERVATION, mechanism unconfirmed):** run #2 went
  rebuilt-framework → Phase-3 start with no phase2_done conductor audit (it ran on the run #3 restart and caught
  leak #3). Nil impact in run #2 (0 experiments). Pre-existing run.py steer-gating; latent/low-priority (downstream
  layers catch leaks). Not fixing now.

- **Phase-2 leakage saga RESOLVED (run #3, no prompt fix needed):** after the 3rd rewind the builder added a
  `LEAKAGE_BLOCKLIST` (mid_clean/mid_embedded/spread_bps_intraday/status/confirmedPrice) + forward-return regex
  (fwd_ret/ret_fwd/future_return) + a `ValueError` on explicit `feature_cols` requesting target_col/ret_col +
  a trading_sharpe-vs-flow_capture_sharpe metric split. The critic VERIFIED with real probes (searched the exact
  patterns, pytest 25-passed + smoke run) → a grounded PASS, NOT a rubber-stamp. The Conductor's adversarial-check
  directive fixed the critic's behavior; the framework is genuinely leak-safe. Watching phase2_done for approval.

- **Verifier reads the LIVE experiments.db (mode=ro) concurrently with dispatcher writes (integration-only, low-prob):**
  DB is `journal_mode=DELETE` (intentional — WAL needs shm that fails on NFS; workspace is on `/v/...`). The
  `ExperimentDB` docstring stale-says "Uses WAL mode" (real inaccuracy, line ~214). The verifier's `_build_menu`
  connection uses sqlite3's default 5s busy_timeout, so it retries-and-succeeds against the dispatcher's quick
  UPDATEs; under heavy 5-6-worker write load a SELECT *could* time out → caught by try/except → empty menu → no
  candidates fire that round (re-fires next batch). Hardening = higher busy_timeout on `verifier.py` ~571 connect,
  but that needs a restart. Not fixing now (low-prob, run #3 clean); watching the verifier's first live fire.

## Pending (flagged, not yet fixed)
- **verifier_provider ignored in integrated mode (latent, moot this run):** `_build_verifier` passes
  `provider=self.provider` (the main provider) and never uses `config.verifier_provider`. Fine here (main +
  verifier both openai), but a cross-provider config (main bedrock + verifier_provider openai with gpt-5.5
  models) would send those models to bedrock. Proper fix = build a secondary provider like the Conductor's
  `conductor_provider` (auth-sensitive, with fallback). Not built — moot for this run; config knob is accepted.
