"""Re-write a review's report with ITS OWN writer, read from its records.

`recompose` takes the writer model on the command line, which invites the
one mistake this stage exists to prevent: a rewrite that silently changes
the review's configuration (it happened — a batch of rewrites once switched
every review to a different writer, destroying what each review was a
specimen OF, 2026-08-10). This wrapper reads the writer identity from the
review's own `sessions.jsonl` and delegates to recompose with it; explicit
--provider/--model overrides are accepted but printed loudly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runcmp import recompose


def _last_session(out_dir: Path) -> dict | None:
    path = out_dir / "sessions.jsonl"
    if not path.is_file():
        return None
    last = None
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("provider") and row.get("model"):
            last = row
    return last


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="runcmp rereview",
        description="re-write a review's report, keeping its own writer")
    ap.add_argument("--out", required=True, type=Path,
                    help="the review directory")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--packs", required=True, type=Path)
    ap.add_argument("--provider", default=None,
                    help="override the recorded writer provider (loud)")
    ap.add_argument("--model", default=None,
                    help="override the recorded writer model (loud)")
    ap.add_argument("--minutes", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    args = ap.parse_args(argv)

    rec = _last_session(args.out)
    if rec is None and not (args.provider and args.model):
        raise SystemExit(
            f"{args.out} has no sessions.jsonl — this review predates "
            "provenance records, so the writer identity cannot be read. "
            "Pass --provider and --model explicitly (use the SAME writer "
            "the review was produced with: a report re-written by a "
            "different model is a different specimen).")

    provider = args.provider or rec["provider"]
    model = args.model or rec["model"]
    effort = (rec or {}).get("reasoning_effort") or "high"
    if args.provider or args.model:
        print(f"OVERRIDE: rewriting with {provider}:{model} instead of the "
              f"recorded "
              + (f"{rec['provider']}:{rec['model']}" if rec else "(none)")
              + " — the result is a different specimen")
    else:
        print(f"writer from the review's own record: {provider}:{model}")

    sub = ["--out", str(args.out), "--corpus", str(args.corpus),
           "--packs", str(args.packs), "--provider", provider,
           "--model", model, "--reasoning-effort", str(effort)]
    # keep the critic seat as recorded: a team review's roles carry it; a
    # recompose record carries the critic model directly; solos get none
    roles = (rec or {}).get("roles") or {}
    critic_spec = roles.get("critic") or (rec or {}).get("critic")
    if critic_spec and critic_spec != "none":
        if ":" not in str(critic_spec):
            critic_spec = f"{provider}:{critic_spec}"
        sub += ["--role", f"critic={critic_spec}"]
    else:
        sub += ["--role", "critic=none"]
    if args.minutes is not None:
        sub += ["--minutes", str(args.minutes)]
    if args.iterations is not None:
        sub += ["--iterations", str(args.iterations)]
    print("recompose " + " ".join(sub))
    return recompose.main(sub)


if __name__ == "__main__":
    raise SystemExit(main())
