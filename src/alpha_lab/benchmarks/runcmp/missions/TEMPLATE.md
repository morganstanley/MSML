# Investigation mission — template (for the hand-written exception)

**You usually do not need this file.** The investigator stages generate
their mission by default (`--mission auto`, or `auto:<focus>` to pick the
decision preset; `runcmp mission` previews one) — the corpus inventory is
derived from the registry and referee so it cannot go stale, and
operator-known facts enter through `--note` flags. Write a mission by hand
only when the decisions you need are not covered by any focus preset; if
that happens routinely, add a preset to `mission.py` instead.

A *mission* is the text you pass to the investigator stages with
`--mission @path/to/file.md`. It is appended to the investigator's opening
message: the system prompt already carries the evidence rules (every number
from a tool result, populations from full scans, causal claims need
mechanism evidence, never present a subset), so the mission's job is only
to say **what this investigation must decide and what this corpus contains**.

Keep every sentence load-bearing. The investigator will treat the mission
as authoritative scope. A stale corpus description produces confident
nonsense — rewrite the corpus section for every campaign, never splice one
forward from an old mission.

---

## 1. The decision(s) this report must deliver

State the questions as decisions someone will act on, e.g.:

- Which harness should the team use, and which of the losing harnesses'
  mechanisms are worth copying?
- Which model should drive a whole run; is any seat better served by a
  different model?
- What should each harness change first, ranked, with the evidence?

## 2. The corpus, exactly

Inventory what the registry actually contains — harnesses, models, tasks,
run counts, known gaps. Name the things the investigator must NOT assume:

- Which task scores are cross-run comparable (shared frozen truth) and
  which are self-report-verified only — mirror what `referee.json` says
  (its per-run `rankable` flags are authoritative).
- Which runs carry no process data (external evidence-contract submissions
  have no transcripts/ledgers: absence is a fact, never a zero).
- Which cells are exact repeats / confounded (different waves, different
  serving deployments, mixed seats) — say so explicitly.

## 3. Scorecards available

List the deterministic artifacts and what each is for, with paths relative
to the campaign directory: `tables.md` (pair tables), `bench.md` (metric
registry; opens with the per-domain referee leaderboards), `referee.json`
(independent re-scoring), `token_accounting.md` (tokens/cost; in-house
runs only, when applicable).

## 4. Presentation demands (optional but recommended)

Chart floors and table requirements, if you want them enforced beyond the
built-in gates (the writer is already refused for <12 charts, <6 chart
forms, <4 distribution forms, registry-echo violations, retyped
progression series, and referee-attribution drift).

---

A complete, battle-tested example lives with each finished campaign
(e.g. `comparison_out/<campaign>/mission_*.md` in this repository's
working data). The sections above mirror its shape.
