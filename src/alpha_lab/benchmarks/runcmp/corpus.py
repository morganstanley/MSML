"""Layer 0 — run registry: discover finished runs and make the messy corpus explicit.

A *run directory* is the parent of a workspace: it usually holds ``events.jsonl``
and ``run.log`` next to a ``ws/`` or ``workspace/`` child. The registry records,
for every discovered run: framework, domain, era, completeness at the database
row level (a present-but-empty ``experiments.db`` is not a usable run), and a
pair key so downstream layers never have to re-derive the corpus structure.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
import re
from pathlib import Path

# Directory names that can never be workspaces or run dirs.
_SKIP_DIRS = {
    "__pycache__", ".git", "charts", "data", "datasets", "models",
    "checkpoints", "backups", "inputs", "reports_msml_only",
}

_WORKSPACE_MARKERS = ("experiments.db", ".alpha_lab", "adapter", "learnings.md")

# Canonical workspace directory names; "<name>.<suffix>" is an archived
# attempt rotated aside by a from-scratch restart, never a clean run.
_WORKSPACE_MARKER_DIRS = ("workspace", "ws")

# A workspace set aside by a restart keeps its own name plus a suffix
# (``d2_o48_cond.aborted_211953``, ``d7r_dsv4_cond.interference_degraded_2120``),
# so the marker-name rule above never catches it. Such a directory can still
# hold a populated experiments.db -- one held 19 rows, another 12 -- and would
# otherwise enter the registry, form phantom pairs, and compete with the run
# that replaced it. The suffix vocabulary keeps growing (``resumed_invalid``
# and ``interference_degraded`` slipped past an earlier, shorter list and were
# scored), so two rules apply: the structural one in ``find_workspaces`` (a
# dot-suffix over a live sibling directory is an archived attempt, whatever
# the spelling) and this keyword list for attempts whose live sibling is gone.
_ARCHIVED_SUFFIX_RE = re.compile(
    r"\.(aborted|stalled|attempt|old|bak|resumed|invalid|interference|"
    r"degraded|failed|superseded|dead|stale)[-_.]?\w*$"
)

# Statuses that mean an experiment row reached a terminal state.
TERMINAL_STATUSES = {"done", "analyzed", "checked", "finished", "cancelled", "failed"}
INFLIGHT_STATUSES = {"to_implement", "implemented", "queued", "running", "proposed"}


@dataclass
class RunRecord:
    label: str
    framework: str            # cond | msml | unknown
    domain: str               # domain1..domain5_m5 | unknown
    era: str                  # top-level dir name the run was found under
    run_dir: str              # parent holding events.jsonl / run.log
    workspace: str
    events_path: str | None
    run_log_path: str | None
    db_path: str | None
    db_rows: int
    db_terminal_rows: int
    status_counts: dict = field(default_factory=dict)
    completeness: str = "unknown"   # complete | empty_db | no_db
    framework_evidence: str = ""
    model: str = ""            # model that drove the run, "" when unrecorded
    run_state: str = "unknown"  # finished | in_flight | unknown
    notes: str = ""
    pair_key: str = ""


def _is_workspace(path: Path) -> bool:
    try:
        names = {e.name for e in os.scandir(path)}
    except (NotADirectoryError, PermissionError, FileNotFoundError):
        return False
    return any(m in names for m in _WORKSPACE_MARKERS) and (
        "experiments.db" in names or ".alpha_lab" in names or "adapter" in names
    )


# External harnesses submit runs as an evidence-contract layout instead of a
# live workspace: experiments/<name>/results/metrics.json (+ optional code
# and referee artifacts), no experiments.db, no adapter, no transcripts.
# One reviewer for many harnesses only works if this layout is a first-class
# run kind, detected by its shape rather than by who produced it.
_CONTRACT_NAME_RE = re.compile(r"^run[_-](.+?)[_-]\d{8}T\d{6}Z?$")


def _is_contract_run(path: Path) -> bool:
    exp = path / "experiments"
    if not exp.is_dir():
        return False
    try:
        for e in os.scandir(exp):
            if e.is_dir(follow_symlinks=False) and (
                    Path(e.path) / "results" / "metrics.json").is_file():
                return True
    except (PermissionError, FileNotFoundError, NotADirectoryError):
        return False
    return False


def find_workspaces(
    root: Path,
    max_depth: int = 4,
    skipped_archived: list[Path] | None = None,
) -> list[Path]:
    """All workspace directories under root, without descending into them.

    A rotated-aside workspace (``workspace.stalled-…``, ``ws.attempt4-…``:
    a marker name plus a suffix) is an archived attempt that was replaced
    by a from-scratch restart — never a clean run. Those are skipped
    entirely: they must not enter the registry, pair with anything, or be
    reported on. Beyond the known-keyword suffixes, any directory whose
    name extends a live sibling directory's name by a dot-suffix
    (``cell`` next to ``cell.anything``) is treated as an archived attempt
    of that sibling — the archive-by-rename protocol always relaunches
    under the original name, so the live sibling is the run that counts.
    Every skip is appended to ``skipped_archived`` (when given) so the
    exclusion is visible in the registry, never silent.
    """
    found: list[Path] = []
    stack = [(root, 0)]
    while stack:
        d, lvl = stack.pop()
        if lvl > max_depth or d.name in _SKIP_DIRS:
            continue
        if any(
            d.name != m and d.name.startswith(m + ".")
            for m in _WORKSPACE_MARKER_DIRS
        ) or _ARCHIVED_SUFFIX_RE.search(d.name):
            if skipped_archived is not None:
                skipped_archived.append(d)
            continue
        if lvl > 0 and _is_workspace(d):
            found.append(d)
            continue
        if lvl > 0 and _is_contract_run(d):
            found.append(d)
            continue
        try:
            entries = list(os.scandir(d))
        except (NotADirectoryError, PermissionError, FileNotFoundError):
            continue
        dir_names = {
            e.name for e in entries if e.is_dir(follow_symlinks=False)
        }
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                base = e.name.split(".", 1)[0]
                if "." in e.name and base and base in dir_names:
                    if skipped_archived is not None:
                        skipped_archived.append(Path(e.path))
                    continue
                stack.append((Path(e.path), lvl + 1))
    return sorted(found)


def _db_status_counts(db: Path) -> dict[str, int]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT status, COUNT(*) FROM experiments GROUP BY status"
        ).fetchall()
        return {str(k): int(v) for k, v in rows}
    finally:
        con.close()


def select_db(workspace: Path) -> tuple[Path, dict[str, int]] | None:
    """Pick the database that actually holds experiment rows.

    Both layouts occur in the corpus (cond: ws root; msml: ``.alpha_lab/``),
    and real workspaces contain traps: a zero-byte or table-less
    ``experiments.db`` at one location with a *newer* mtime than the populated
    one (observed in domain5_m5 msml maxfix6). Newest-by-mtime is therefore
    wrong; prefer the candidate with the most rows, mtime only as tie-break.
    """
    candidates = [
        workspace / "experiments.db",
        workspace / ".alpha_lab" / "experiments.db",
    ]
    probed: list[tuple[int, float, Path, dict[str, int]]] = []
    for p in candidates:
        if not p.is_file():
            continue
        try:
            counts = _db_status_counts(p)
        except sqlite3.Error:
            continue
        probed.append((sum(counts.values()), p.stat().st_mtime, p, counts))
    if not probed:
        return None
    probed.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, path, counts = probed[0]
    return path, counts


def _detect_framework(workspace: Path, path_str: str) -> tuple[str, str]:
    """Framework from artifacts first, path naming as fallback/cross-check."""
    artifact = "unknown"
    if (workspace / "meta" / "meta_log.jsonl").is_file():
        artifact = "cond"
    elif (workspace / "agenda.md").is_file() or (workspace / "phase1").is_dir():
        artifact = "msml"
    lowered = path_str.lower()
    from_path = "unknown"
    if "msml" in lowered:
        from_path = "msml"
    elif "cond" in lowered:
        from_path = "cond"
    else:
        # Bring-your-own-harness: the layout root/era/domain/<framework>/ws
        # names the framework in the path. Use that segment — the in-house
        # names above are defaults, not a roster.
        parts = Path(path_str).parts
        if len(parts) >= 2:
            from_path = parts[-2].lower()
    if artifact != "unknown":
        note = f"artifact={artifact}, path={from_path}"
        return artifact, note
    return from_path, f"path-only={from_path}"


def _detect_domain(path_str: str) -> str:
    lowered = path_str.lower()
    # Legacy names first: they predate the lineup and several lineup tokens
    # (d2_, d4_) alias onto them anyway.
    for key in (
        "domain5_m5", "domain5_etfflows", "domain1", "domain2", "domain3",
        "domain4",
    ):
        if key in lowered:
            return key
    if "d2_" in lowered or "_d2" in lowered:
        return "domain2"
    if "d4_" in lowered or "_d4" in lowered:
        return "domain4"
    # Lineup-declared domains (lineup.json is the registry of adopted
    # benchmarks; adding a dataset there makes its runs indexable here).
    from alpha_lab.benchmarks.runcmp.lineup import detect_domain as _lineup_detect

    hit = _lineup_detect(path_str)
    if hit is not None:
        return hit
    return "unknown"


def _contract_record(ws: Path, root: Path,
                     aliases: dict[str, str]) -> RunRecord:
    """RunRecord for an evidence-contract run: the run dir IS the workspace.

    Completeness is contract completeness (every experiment dir carries a
    parseable results/metrics.json) — the db-row rule cannot apply, there
    is no db. db_rows/db_terminal_rows carry the experiment-dir counts so
    the index printout stays meaningful; framework_evidence says so.
    """
    rel = str(ws.relative_to(root))
    total = ok = 0
    for e in sorted(os.scandir(ws / "experiments"), key=lambda x: x.name):
        if not e.is_dir():
            continue
        total += 1
        m = Path(e.path) / "results" / "metrics.json"
        try:
            json.loads(m.read_text())
            ok += 1
        except (OSError, json.JSONDecodeError):
            pass
    m = _CONTRACT_NAME_RE.match(ws.name)
    framework = (m.group(1) if m else ws.name).lower()
    framework = aliases.get(framework, framework)
    era = _era_of(rel, root)
    domain = _detect_domain(rel)
    return RunRecord(
        label=f"{era}/{domain}/{framework}",
        framework=framework,
        domain=domain,
        era=era,
        run_dir=str(ws),
        workspace=str(ws),
        events_path=None,
        run_log_path=None,
        db_path=None,
        db_rows=total,
        db_terminal_rows=ok,
        status_counts={"done": ok},
        completeness="complete" if ok and ok == total else "contract_partial",
        framework_evidence=(f"contract-layout: {ok}/{total} experiment dirs "
                            "with parseable metrics.json (counts are dirs, "
                            "no db)"),
        model="",              # unrecorded — declared models live in prose
        run_state="unknown",   # an exported archive carries no exit line
    )


def _file_maybe_gz(path: Path) -> Path | None:
    """Finished campaigns get their large logs compressed in place
    (events.jsonl -> events.jsonl.gz; observed 2026-08-05 on a July run
    set). A lookup that tests only the plain name then registers "no
    event stream" for a stream that exists on disk, and every clock
    metric built on it goes blank. Prefer the plain file, follow the
    rename, else None."""
    if path.is_file():
        return path
    gz = path.with_name(path.name + ".gz")
    if gz.is_file():
        return gz
    return None


def _era_of(rel: str, root: Path) -> str:
    """The batch/era a run belongs to — the top-level directory it was found
    under. When the run sits directly inside the root (root IS the era
    directory, e.g. ``--root /data/native_20260728``), the first relative
    component is the task directory, not an era — fall back to the root's
    own name so runs indexed from several roots stay distinguishable and
    twin run names get era-qualified labels instead of opaque #N suffixes.
    """
    parts = rel.split("/")
    return parts[0] if len(parts) >= 3 else root.name


def build_registry(root: Path,
                   aliases: dict[str, str] | None = None,
                   skipped_archived: list[Path] | None = None,
                   ) -> list[RunRecord]:
    aliases = aliases or {}
    records: list[RunRecord] = []
    for ws in find_workspaces(root, skipped_archived=skipped_archived):
        if not _is_workspace(ws):
            records.append(_contract_record(ws, root, aliases))
            continue
        run_dir = ws.parent
        rel = str(ws.relative_to(root))
        era = _era_of(rel, root)
        events = _file_maybe_gz(run_dir / "events.jsonl")
        run_log = _file_maybe_gz(run_dir / "run.log")
        if run_log is None:
            # Multi-run layout: several workspaces share run_dir and each
            # cell's stderr log lives in <base>/logs/<cell>.log. Without
            # this, no run log is ever parsed for these runs — a 1,915-line
            # certificate failure passed through evidence packs invisibly.
            for cand in (run_dir / "logs" / f"{ws.name}.log",
                         run_dir.parent / "logs" / f"{ws.name}.log"):
                cand_found = _file_maybe_gz(cand)
                if cand_found is not None:
                    run_log = cand_found
                    break
        selected = select_db(ws)
        status_counts: dict[str, int] = {}
        db: Path | None = None
        db_rows = db_terminal = 0
        completeness = "no_db"
        if selected is not None:
            db, status_counts = selected
            db_rows = sum(status_counts.values())
            db_terminal = sum(
                v for k, v in status_counts.items() if k in TERMINAL_STATUSES
            )
            completeness = "complete" if db_terminal > 0 else "empty_db"
        framework, fw_note = _detect_framework(ws, rel)
        domain = _detect_domain(rel)
        # Include the workspace's own name when it carries information the
        # rest of the label doesn't. Several runs of one framework on one
        # task differ only by which model drove them (d4_sol_cond vs
        # d4_o48_cond); without the name they collapse to "cond", "cond#2",
        # "cond#3" and the numbering follows directory-walk order, so no
        # reader can tell which run is which model.
        ws_name = ws.name
        generic = ws_name in ("workspace", "ws") or ws_name == framework
        label = (f"{era}/{domain}/{framework}" if generic
                 else f"{era}/{domain}/{framework}/{ws_name}")
        records.append(
            RunRecord(
                label=label,
                framework=framework,
                domain=domain,
                era=era,
                run_dir=str(run_dir),
                workspace=str(ws),
                events_path=str(events) if events else None,
                run_log_path=str(run_log) if run_log else None,
                db_path=str(db) if db else None,
                db_rows=db_rows,
                db_terminal_rows=db_terminal,
                status_counts=status_counts,
                completeness=completeness,
                framework_evidence=fw_note,
                model=_detect_model(ws),
                run_state=_detect_run_state(run_dir, ws),
            )
        )
    # Disambiguate duplicate labels (retries within one era/domain/framework).
    seen: dict[str, int] = {}
    for rec in records:
        n = seen.get(rec.label, 0)
        seen[rec.label] = n + 1
        if n:
            rec.label = f"{rec.label}#{n + 1}"
    _assign_pairs(records)
    return records



def _detect_run_state(run_dir: Path, ws: Path) -> str:
    """Has this run's process exited, or is it still producing experiments?

    A run still executing keeps adding rows, so comparing it against a
    finished run measures how long each has been going rather than how well
    it did. Reviewers cannot see this from the database, so it is recorded
    here. The launcher writes an exit line into the run log; its absence in
    a log still being appended to means the run is live.
    """
    import time as _time
    for cand in (run_dir / f"{ws.name}.log", run_dir / "run.log",
                 run_dir.parent / "logs" / f"{ws.name}.log"):
        if not cand.is_file():
            continue
        try:
            tail = cand.read_text(errors="replace")[-4000:]
        except OSError:
            continue
        if "EXIT code=" in tail:
            return "finished"
        if _time.time() - cand.stat().st_mtime < 1800:
            return "in_flight"
        return "unknown"
    return "unknown"



def _detect_model(ws: Path) -> str:
    """Which model drove this run, read from what the run itself recorded.

    Needed because several runs of one framework on one task differ only by
    the model. msml writes its config into the workspace; cond records the
    model on every logged call. Returns "" when neither is present, which
    puts the run back in the single-run-per-framework grouping it used
    before.
    """
    cfg = ws / ".alpha_lab" / "config.json"
    if cfg.is_file():
        try:
            model = json.loads(cfg.read_text()).get("model")
            if model:
                return str(model)
        except (json.JSONDecodeError, OSError):
            pass
    ledger = ws / "meta" / "token_usage.jsonl"
    if ledger.is_file():
        counts: dict[str, int] = {}
        try:
            with open(ledger, errors="replace") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    # The conductor deliberately runs a different model from
                    # the rest of the pipeline; counting it would label the
                    # run by its supervisor instead of its driver.
                    if str(row.get("log_name", "")).startswith("conductor"):
                        continue
                    m = row.get("model")
                    if m:
                        counts[str(m)] = counts.get(str(m), 0) + 1
        except OSError:
            return ""
        if counts:
            return max(counts, key=counts.get)
    return ""


def _assign_pairs(records: list[RunRecord]) -> None:
    """Pair each framework's run against the other framework's run of the
    same task AND the same model.

    Grouping by task alone kept one run per framework and silently dropped
    the rest: with three models a side that produced one pair out of six and
    four runs that were never compared to anything. The model is part of the
    grouping key so every model gets its own head-to-head.
    """
    by_key: dict[tuple[str, str, str], list[RunRecord]] = {}
    for rec in records:
        # A run still executing keeps adding experiments; pairing it against
        # a finished run compares elapsed time, not research quality.
        if (rec.completeness == "complete" and rec.framework != "unknown"
                and rec.run_state != "in_flight"):
            by_key.setdefault((rec.era, rec.domain, rec.model), []).append(rec)
    for (era, domain, model), group in by_key.items():
        # Framework names are never consulted — most corpora bring their own
        # harnesses. A cell with exactly two frameworks auto-pairs (sorted
        # name order fixes the orientation deterministically); anything else
        # is left to explicit --pair specs, loudly.
        frameworks = sorted({r.framework for r in group})
        if len(frameworks) != 2:
            if len(frameworks) > 2:
                print(f"  NOTE: {len(frameworks)} frameworks in cell "
                      f"{era}/{domain}" + (f"/{model}" if model else "")
                      + " — auto-pairing needs exactly two; add --pair "
                        "specs for the other match-ups")
            continue
        # Most-complete run on each side is the pair member.
        pick = lambda fw: max((r for r in group if r.framework == fw),  # noqa: E731
                              key=lambda r: r.db_terminal_rows)
        key = f"{era}/{domain}" + (f"/{model}" if model else "")
        pick(frameworks[0]).pair_key = key
        pick(frameworks[1]).pair_key = key


def write_registry(records: list[RunRecord], out_path: Path,
                   skipped_archived: list[Path] | None = None) -> None:
    payload = {
        "schema": "runcmp-corpus-1",
        "runs": [asdict(r) for r in records],
    }
    if skipped_archived:
        # Archived attempts excluded at index time — recorded so the
        # exclusion is auditable from the registry itself, never silent.
        payload["archived_attempts_skipped"] = sorted(
            str(p) for p in skipped_archived
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1) + "\n")


def load_registry(path: Path) -> list[RunRecord]:
    data = json.loads(path.read_text())
    return [RunRecord(**row) for row in data["runs"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the runcmp corpus registry")
    ap.add_argument("--root", required=True, type=Path, action="append",
                    help="corpus root; repeatable — one registry over all")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--framework-alias", action="append", default=[],
                    metavar="OLD=NEW",
                    help="unify contract-run framework labels (one harness "
                         "may stamp its run dirs inconsistently)")
    ap.add_argument("--finished-only", action="store_true",
                    help="drop runs whose process is still writing rows; a "
                         "live run in a review measures elapsed time, not "
                         "research quality")
    args = ap.parse_args(argv)
    aliases: dict[str, str] = {}
    for spec in args.framework_alias:
        old, _, new = spec.partition("=")
        if not new:
            ap.error(f"--framework-alias needs OLD=NEW, got {spec!r}")
        aliases[old.lower()] = new.lower()
    records = []
    skipped_archived: list[Path] = []
    for root in args.root:
        records.extend(build_registry(root, aliases=aliases,
                                      skipped_archived=skipped_archived))
    # Re-dedupe labels across roots (per-root dedupe already ran).
    seen: dict[str, int] = {}
    for rec in records:
        n = seen.get(rec.label, 0)
        seen[rec.label] = n + 1
        if n:
            rec.label = f"{rec.label}#{n + 1}"
    if args.finished_only:
        live = [r for r in records if r.run_state == "in_flight"]
        records = [r for r in records if r.run_state != "in_flight"]
        for r in live:
            print(f"  DROPPED (--finished-only, still in flight): {r.label}")
    write_registry(records, args.out, skipped_archived=skipped_archived)
    complete = [r for r in records if r.completeness == "complete"]
    paired = [r for r in complete if r.pair_key]
    print(f"runs discovered: {len(records)}")
    if skipped_archived:
        print(f"archived attempts skipped: {len(skipped_archived)}")
        for p in sorted(str(x) for x in skipped_archived):
            print(f"  SKIPPED (archived attempt): {p}")
    # "usable" must not silently cover runs still being written: a reader
    # took the word at face value for an in-flight run (2026-08-11). The
    # count says so, and each live row is marked in the listing.
    live_complete = [r for r in complete if r.run_state == "in_flight"]
    print(f"usable (terminal rows > 0): {len(complete)}"
          + (f" — of which {len(live_complete)} still IN FLIGHT (never "
             "paired; rerun with --finished-only to drop them)"
             if live_complete else ""))
    print(f"paired: {len(paired)} in {len({r.pair_key for r in paired})} pairs")
    for r in records:
        mark = "PAIR " + r.pair_key if r.pair_key else r.completeness
        if r.run_state == "in_flight":
            mark += "  [IN FLIGHT]"
        print(f"  {r.label:60s} rows={r.db_rows:<3d} {mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
