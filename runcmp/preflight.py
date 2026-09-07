"""Launch checks that run BEFORE reviewer-model budget is spent.

Half of a review's failure modes are decided at launch, not at run time: a
live run in the corpus, a stale or missing evidence pack, credentials that
died overnight, a store that is not writable. Each is minutes to check and
hours to discover late. Exit 0 = clear to launch; exit 1 = at least one
blocker, each printed with BLOCKER in front of it. Warnings do not block.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from runcmp.corpus import build_registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="runcmp preflight",
        description="pre-launch checks: corpus, packs, credentials, store")
    ap.add_argument("--root", required=True, type=Path, action="append",
                    help="corpus root(s) about to be reviewed")
    ap.add_argument("--packs", type=Path, default=None,
                    help="existing packs dir to check for coverage/staleness")
    ap.add_argument("--provider", action="append", default=[],
                    metavar="PROVIDER:MODEL",
                    help="credential check: one tiny call per entry "
                         "(costs a few tokens)")
    ap.add_argument("--store", type=Path, default=None,
                    help="MLflow showcase directory to verify writable")
    args = ap.parse_args(argv)

    blockers: list[str] = []
    warnings: list[str] = []

    records = []
    skipped: list[Path] = []
    for root in args.root:
        records.extend(build_registry(root, skipped_archived=skipped))
    if not records:
        blockers.append(f"no runs found under {[str(r) for r in args.root]}")
    live = [r.label for r in records if r.run_state == "in_flight"]
    if live:
        warnings.append(f"{len(live)} run(s) still IN FLIGHT (a live run "
                        "measures elapsed time, not quality): "
                        + ", ".join(live[:5]))
    dupes = [r.label for r in records if "#" in r.label]
    if dupes:
        warnings.append(f"duplicate labels disambiguated: {dupes[:5]} — one "
                        "era/domain/framework produced several runs; make "
                        "sure that is intended")
    incomplete = [r.label for r in records if r.completeness != "complete"]
    if incomplete:
        warnings.append(f"{len(incomplete)} run(s) not complete (kept as "
                        "evidence, but never paired): "
                        + ", ".join(incomplete[:5]))
    if skipped:
        print(f"archived attempts correctly ignored: {len(skipped)}")
    print(f"corpus: {len(records)} run(s), "
          f"{len(records) - len(incomplete)} complete, {len(live)} in flight")

    if args.packs is not None:
        by_pack = {p.stem for p in args.packs.glob("*.json")}
        for r in records:
            flat = r.label.replace("/", "__")
            if flat not in by_pack:
                blockers.append(f"no evidence pack for {r.label} — run "
                                "extract (with --force --all) before "
                                "reviewing")
            elif r.db_path and Path(r.db_path).is_file():
                pack_m = (args.packs / f"{flat}.json").stat().st_mtime
                if Path(r.db_path).stat().st_mtime > pack_m:
                    blockers.append(f"pack STALE for {r.label}: the run's "
                                    "database is newer than its pack — "
                                    "re-extract with --force")

    for spec in args.provider:
        provider_name, _, model = spec.partition(":")
        if not model:
            blockers.append(f"--provider needs PROVIDER:MODEL, got {spec!r}")
            continue
        try:
            from runcmp.llm.client import get_provider
            from runcmp.investigate_team import _one_call
            _one_call(get_provider(provider_name), model,
                      "Reply with the single word OK.", "ping", "low",
                      retries=1)
            print(f"credentials OK: {spec}")
        except Exception as exc:  # noqa: BLE001 — any failure blocks launch
            blockers.append(f"credential check failed for {spec}: "
                            f"{str(exc)[:200]}")

    if args.store is not None:
        try:
            args.store.mkdir(parents=True, exist_ok=True)
            probe = args.store / ".runcmp_preflight_probe"
            probe.write_text("probe")
            probe.unlink()
            print(f"store writable: {args.store}")
        except OSError as exc:
            blockers.append(f"store not writable: {args.store} ({exc})")

    for w in warnings:
        print(f"WARNING: {w}")
    for b in blockers:
        print(f"BLOCKER: {b}")
    print("preflight: " + ("FAIL" if blockers else "clear to launch"))
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
