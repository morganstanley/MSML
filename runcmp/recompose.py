"""Re-write the report of a finished review, from its frozen findings.

A review has two halves: gathering evidence (probes, findings, the question
ledger) and writing the report. They fail independently, and the writing half
fails far more often — a report can ship with whole mission sections as stubs
while every finding behind it is critic-accepted and sound.

Re-running the whole review to fix the writing is wasteful and it destroys
evidence that was already good: one review spent 232 iterations, of which
the report drafts were the last handful, and its 22 findings all carried
critic verdicts (2026-08-10). So the writer is restartable on its own:

    runcmp recompose --out out/review_team \\
        --corpus out/corpus.json --packs out/packs

What is frozen: `findings.jsonl`, the question ledger, `plan.md`, every probe
output. `record_finding` is refused for the whole session — this stage cannot
add, drop or reword a finding, so the evidence base of the re-written report is
provably the one that was already reviewed.

What is re-done: composition, under the same report critic as a full review —
scored each round, bouncing while budget and the wall clock allow, publishing
the best-scored draft. The previous report is kept beside the new one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from runcmp.investigate import (
    MAX_TOOL_OUTPUT,
    SYSTEM_PROMPT,
    TOOLS,
    _compact_history,
)
from runcmp.investigate_team import (
    QUESTION_TOOLS,
    TeamSession,
    _parse_role,
    publish_best_draft,
)

# Composition is a handful of long calls, not an investigation: the writer
# needs enough turns to re-read the probes it quotes and to answer the critic,
# and no more. Measured shape it is sized against: one full review reached
# iteration 232, of which drafting was the last ~10.
DEFAULT_ITERATIONS = 120

COMPOSE_PREAMBLE = """\
# THIS SESSION WRITES THE REPORT ONLY

The investigation is finished and its evidence is FROZEN. Below are the findings
that were recorded and critic-accepted, the question ledger, and the mission the
report must deliver. Every probe output from that investigation is still on disk
and you can re-read or re-run probes to quote exact numbers.

You may NOT record new findings — `record_finding` is disabled in this session.
Your only deliverable is the report: call `write_report` with the complete
markdown. It will be scored against the mission by the critic and sent back with
required changes if it does not deliver; revise and resubmit the FULL report.

A previous report exists for this investigation and was judged not to deliver.
Do not summarize it — write the report the mission asks for, in full depth, from
the findings and probe evidence below.
"""


class ComposeSession(TeamSession):
    """A team session with the evidence half frozen."""

    def record_finding(self, **kwargs) -> str:  # noqa: D401
        return ("[ERROR] findings are frozen in this session: it re-writes the "
                "report only. Use the findings listed in your instructions and "
                "the probe outputs on disk; call write_report when ready.")


def _archive_drafts(out_dir: Path) -> Path | None:
    """Move a previous session's report_drafts aside, never over.

    Draft files are named by round number, which restarts at 1 in every
    session — a rerun into the same directory would overwrite the previous
    session's scored drafts exactly the way probe renumbering once destroyed
    probe outputs (2026-08-10). Renamed, not deleted.
    """
    d = out_dir / "report_drafts"
    if not d.is_dir() or not any(d.iterdir()):
        return None
    kept = out_dir / f"report_drafts_prev_{time.strftime('%Y%m%dT%H%M%S')}"
    d.rename(kept)
    return kept


def _fingerprint(out_dir: Path) -> dict[str, str]:
    """sha256 per frozen-evidence file: everything the findings cite.

    Taken before the writer runs and re-checked at every exit. A recompose
    that changes any of these has corrupted the evidence base it promised
    to freeze — probe overwrites silently invalidated two directories'
    findings before this check existed (2026-08-10/11), and the damage was
    only found later, by factcheck.
    """
    import hashlib
    fixed = ["mission.md", "findings.jsonl", "questions.json", "plan.md"]
    paths = [out_dir / f for f in fixed]
    paths += sorted((out_dir / "probes").glob("probe_*"))
    return {str(p.relative_to(out_dir)):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def _load(out_dir: Path) -> tuple[str, list[dict], dict]:
    mission = (out_dir / "mission.md")
    if not mission.is_file():
        raise SystemExit(f"no mission.md in {out_dir} — cannot recompose "
                         "without the mission the report must deliver")
    findings_path = out_dir / "findings.jsonl"
    if not findings_path.is_file():
        raise SystemExit(f"no findings.jsonl in {out_dir} — nothing to compose "
                         "from; run a full review instead")
    findings = [json.loads(l) for l in
                findings_path.read_text().splitlines() if l.strip()]
    ledger = {}
    qp = out_dir / "questions.json"
    if qp.is_file():
        ledger = json.loads(qp.read_text())
    return mission.read_text(), findings, ledger


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="re-write a finished review's report from its frozen findings")
    ap.add_argument("--out", required=True, type=Path,
                    help="an existing review directory (holds mission.md, "
                         "findings.jsonl, probes/)")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--packs", required=True, type=Path)
    ap.add_argument("--provider", default="anthropic",
                    help="the writer's provider")
    ap.add_argument("--model", default="claude-opus-5", help="the writer's model")
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--role", action="append", default=[],
                    help="critic=PROVIDER:MODEL or critic=none (defaults to the "
                         "writer's provider/model)")
    ap.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    ap.add_argument("--minutes", type=int, default=120,
                    help="wall-clock budget for the revision loop")
    args = ap.parse_args(argv)

    from runcmp.llm.client import get_provider

    out_dir: Path = args.out
    mission, findings, ledger = _load(out_dir)
    print(f"recompose: {len(findings)} frozen findings, "
          f"{len(ledger)} ledger questions, mission {len(mission):,} chars")

    critic = {"provider": args.provider, "model": args.model,
              "reasoning_effort": args.reasoning_effort}
    for spec in args.role:
        name, _, value = spec.partition("=")
        if name != "critic":
            raise SystemExit(f"recompose only takes --role critic=…, got {name!r}")
        critic = _parse_role(value, args.provider, args.model,
                             args.reasoning_effort)
    if critic:
        critic = {**critic, "provider": get_provider(critic["provider"]),
                  "mission": mission}

    from runcmp.investigate import record_session
    record_session(out_dir, stage="recompose", provider=args.provider,
                   model=args.model, reasoning_effort=args.reasoning_effort,
                   critic=(f"{critic['model']}" if critic else "none"))
    session = ComposeSession(args.corpus, args.packs, out_dir, critic=critic)
    session.REPORT_PHASE_SECONDS = args.minutes * 60
    # the ledger came closed from the finished review; keep it that way so
    # write_report is not blocked by questions this stage cannot investigate
    for qid, q in (ledger.items() if isinstance(ledger, dict)
                   else enumerate(ledger)):
        if isinstance(q, dict):
            session._questions[int(q.get("id", qid) or qid)] = {
                **q, "status": q.get("status", "resolved") or "resolved"}

    # keep the report that did not deliver, beside the new one — and remember
    # the rename, so a failed recompose can put it back instead of leaving the
    # review with no report at all (which actually happened on 2026-08-10,
    # while the failure message claimed the opposite)
    kept_drafts = _archive_drafts(out_dir)
    if kept_drafts is not None:
        print(f"previous session's drafts kept as {kept_drafts.name}/")
    kept_md = kept_html = None
    prev = out_dir / "REPORT.md"
    if prev.is_file():
        kept_md = out_dir / f"REPORT.superseded_{time.strftime('%Y%m%dT%H%M%S')}.md"
        prev.rename(kept_md)
        if (out_dir / "REPORT.html").is_file():
            kept_html = out_dir / f"{kept_md.stem}.html"
            (out_dir / "REPORT.html").rename(kept_html)
        print(f"previous report kept as {kept_md.name}")

    def salvage_or_restore(reason: str) -> int:
        """No writer left: publish the best-scored draft, or undo the rename."""
        if publish_best_draft(session):
            print(f"recompose: {reason} — published the best-scored "
                  "submitted draft")
            return 0
        if kept_md is not None and not (out_dir / "REPORT.md").is_file():
            kept_md.rename(out_dir / "REPORT.md")
            if kept_html is not None and kept_html.is_file():
                kept_html.rename(out_dir / "REPORT.html")
            print(f"recompose failed ({reason}) with no scored draft — the "
                  "previous report was put back in place")
        else:
            print(f"recompose failed ({reason}) with no scored draft to "
                  "publish")
        return 1

    frozen = _fingerprint(out_dir)
    (out_dir / f"recompose_preflight_{time.strftime('%Y%m%dT%H%M%S')}.json"
     ).write_text(json.dumps({"ts": time.time(), "files": frozen}, indent=1)
                  + "\n")

    def finish(rc: int) -> int:
        """Every exit re-checks the frozen evidence and fails loudly."""
        after = _fingerprint(out_dir)
        changed = [f for f, h in frozen.items() if after.get(f) != h]
        if changed:
            print("FROZEN-EVIDENCE VIOLATION: this session modified files the "
                  "findings cite — factcheck can no longer verify them: "
                  + ", ".join(changed[:10])
                  + (f" (+{len(changed) - 10} more)" if len(changed) > 10
                     else ""))
            return 1
        return rc

    provider = get_provider(args.provider)
    tools = [t for t in (TOOLS + QUESTION_TOOLS)
             if t.get("name") != "record_finding"]
    system = (SYSTEM_PROMPT + "\n\n" + COMPOSE_PREAMBLE
              + "\n\n## Mission\n" + mission
              + "\n\n## Frozen findings\n"
              + json.dumps(findings, indent=1, default=str)[:200_000])
    history: list[dict] = list(provider.build_user_items(
        "Compose the report the mission demands from the frozen findings and "
        "the probe evidence on disk. Call write_report with the full markdown."))
    transcript = open(out_dir / "recompose_log.jsonl", "a")
    started = time.time()

    for iteration in range(args.iterations):
        session.iterations_left = args.iterations - iteration - 1
        if session.iterations_left in (24, TeamSession.REPORT_REVISION_RESERVE):
            history.extend(provider.build_user_items(
                f"[budget] {session.iterations_left} iterations remain. Call "
                "write_report NOW with your best COMPLETE draft — when the "
                "budget runs out the gate stops bouncing and the best-scored "
                "SUBMITTED draft publishes; a draft you never submit cannot "
                "win."))
        # stream_response yields events; the Response arrives on "done" (the
        # same consumption the full-review loop uses — a bare call returns a
        # generator, which is what crashed the first six recompose launches)
        response = None
        for attempt in range(4):
            try:
                for event in provider.stream_response(
                        model=args.model, system=system, tools=tools,
                        history=history,
                        reasoning_effort=args.reasoning_effort):
                    if event.type == "done":
                        response = event.response
                break
            except Exception as exc:  # noqa: BLE001 — retried
                wait = 15 * (attempt + 1)
                print(f"  API error ({exc}); retry in {wait}s", flush=True)
                time.sleep(wait)
        if response is None:
            print("no response after retries; stopping")
            rc = salvage_or_restore("API unreachable")
            transcript.close()
            return finish(rc)
        session.note_own_usage("writer", args.model, provider, response)
        # the provider knows how its own turns go back into history; appending
        # raw items by hand produced "messages.1.role: Field required" on every
        # one of six launches (2026-08-10)
        provider.append_response_to_history(history, response)
        if response.text:
            print(f"[{iteration}] {response.text[:200]}", flush=True)
        if not response.tool_calls:
            history.extend(provider.build_user_items(
                "Continue: call write_report with the full report."))
            continue
        outputs = []
        for tc in response.tool_calls:
            try:
                a = json.loads(tc.arguments or "{}")
            except json.JSONDecodeError:
                a = {}
            result = session.dispatch(tc.name, a)
            transcript.write(json.dumps({"ts": time.time(), "tool": tc.name,
                                         "result": result[:4000]}) + "\n")
            transcript.flush()
            print(f"[{iteration}] {tc.name} -> {len(result)} chars", flush=True)
            outputs.append({"call_id": tc.call_id,
                            "output": result[:MAX_TOOL_OUTPUT]})
        history.extend(provider.build_tool_result_items(outputs))
        _compact_history(history)
        if session.report_written:
            mins = (time.time() - started) / 60
            print(f"recompose done in {mins:.1f} min: report written")
            transcript.close()
            return finish(0)
    rc = salvage_or_restore("iteration cap reached")
    transcript.close()
    return finish(rc)


if __name__ == "__main__":
    raise SystemExit(main())
