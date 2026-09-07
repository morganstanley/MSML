"""One look at a review directory: what ran, what it produced, what holds.

Answers the questions asked of every running or finished review — who wrote
it, how far along is it, what did the report critic do, is the report real,
does verification hold — from the artifacts alone. Read-only: safe to run on
a live review as often as you like.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="runcmp status",
        description="one look at a review directory (read-only)")
    ap.add_argument("--out", required=True, type=Path,
                    help="the review directory")
    args = ap.parse_args(argv)
    d: Path = args.out
    if not d.is_dir():
        print(f"not a directory: {d}")
        return 1

    sessions = _jsonl(d / "sessions.jsonl")
    if sessions:
        s = sessions[-1]
        roles = s.get("roles")
        print(f"session: {s.get('stage')}  writer={s.get('provider')}:"
              f"{s.get('model')}"
              + (f"  roles={roles}" if roles else "")
              + (f"  critic={s.get('critic')}" if s.get("critic") else "")
              + f"  started {time.strftime('%Y-%m-%d %H:%M', time.localtime(s.get('ts', 0)))}")
    else:
        print("session: no sessions.jsonl (predates provenance records)")

    themes: list[str] = []
    mission = d / "mission.md"
    if mission.is_file():
        mtext = mission.read_text(errors="replace")
        themes = re.findall(r"^\s*\d+\.\s+\*\*(.+?)\*\*", mtext, re.MULTILINE)
        print(f"mission: {len(mtext):,} chars, {len(themes)} numbered themes")
    else:
        print("mission: MISSING")

    print(f"findings: {len(_jsonl(d / 'findings.jsonl'))}")

    qp = d / "questions.json"
    if qp.is_file():
        try:
            qs = json.loads(qp.read_text())
        except json.JSONDecodeError:
            qs = {}
        vals = list(qs.values()) if isinstance(qs, dict) else list(qs)
        counts: dict[str, int] = {}
        for q in vals:
            if isinstance(q, dict):
                st = str(q.get("status", "?"))
                counts[st] = counts.get(st, 0) + 1
        print("ledger: " + (", ".join(f"{k}={v}" for k, v in
                                      sorted(counts.items())) or "empty"))

    entries = _jsonl(d / "critic_log.jsonl")
    rounds = [e for e in entries if e.get("round") and "report_review" in e]
    if rounds:
        print(f"report critic: {len(rounds)} round(s)")
        for e in rounds:
            print(f"  round {int(e['round']):>2}: score={e.get('score')} "
                  f"defects={e.get('defects', '—')} -> {e['report_review']} "
                  f"({e.get('draft_chars', 0):,} chars)")
    for e in entries:
        if "publish_choice" in e:
            print(f"publish: {e['publish_choice']} (round {e.get('round')}, "
                  f"score {e.get('score')}, defects {e.get('defects', '—')})")
    drafts = sorted((d / "report_drafts").glob("round_*.md")) \
        if (d / "report_drafts").is_dir() else []
    if drafts:
        print(f"drafts kept: {len(drafts)}")

    report = d / "REPORT.md"
    if report.is_file():
        text = report.read_text(errors="replace")
        from runcmp.investigate import _chart_census
        from runcmp.investigate_team import TeamSession
        census = _chart_census(text)
        stubs = TeamSession._stub_sections(text, themes)
        banner = text.startswith("> Chart-floor warning")
        print(f"report: {len(text):,} chars, {census['total']} charts "
              f"({census['dist_forms']} distribution forms), "
              f"{len(stubs)} stub section(s)"
              + ("  [WARN BANNER — accepted after refusals]" if banner else ""))
        for s in stubs:
            print(f"  STUB: {s}")
    else:
        print("report: NOT WRITTEN")

    vp = d / "verification.json"
    if vp.is_file():
        try:
            v = json.loads(vp.read_text())
            print(f"verification: {v.get('verified', 0)} verified, "
                  f"{v.get('partial', 0)} partial, {v.get('failed', 0)} "
                  f"failed of {v.get('total', 0)}")
        except json.JSONDecodeError:
            print("verification: verification.json unreadable")
    else:
        print("verification: not run (factcheck)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
