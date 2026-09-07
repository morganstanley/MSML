"""Layer 4 — deterministic verification of the investigator's findings.

Every finding's evidence references are resolved against the artifacts they
point at. This is the anti-fabrication gate: a finding whose references do not
resolve is marked failed and should not be trusted. Purely deterministic —
no second model ever rewrites the report.

Evidence reference forms:
- {"pack": label, "path": "experiments.best.value", "value": X}
- {"table": pair_name, "row": row_label, "side": "left"|"right", "value": X}
- {"probe": "probe_003", "contains": "literal"}
- {"file": "/abs/path", "contains": "literal"}   (contains optional)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

REL_TOL = 1e-6
ABS_TOL = 1e-9
SCAN_CAP_BYTES = 800_000_000  # streamed, line by line


def _walk_path(data, dotted: str):
    cur = data
    for part in dotted.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None, f"key {part!r} missing"
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None, f"bad list index {part!r}"
        else:
            return None, f"cannot descend into {type(cur).__name__} at {part!r}"
    return cur, None


def _values_match(expected, actual) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual),
                            rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return str(expected).strip() == str(actual).strip()


def _file_contains(path: Path, needle: str) -> bool:
    scanned = 0
    needle_b = needle.encode()
    with open(path, "rb") as fh:
        for line in fh:
            scanned += len(line)
            if scanned > SCAN_CAP_BYTES:
                return False
            if needle_b in line:
                return True
    return False


class FactChecker:
    def __init__(self, out_dir: Path, packs_dir: Path):
        self.out_dir = out_dir
        self.packs_dir = packs_dir
        self._pack_cache: dict[str, dict] = {}
        tables = out_dir / "tables.json"
        self.tables = json.loads(tables.read_text()) if tables.is_file() else {}

    def _pack(self, label: str) -> dict | None:
        if label not in self._pack_cache:
            safe = label.replace("/", "__").replace("#", "_")
            p = self.packs_dir / f"{safe}.json"
            # Open directly instead of gating on is_file() (EAFP — one
            # fewer race window), with a single delayed retry before
            # concluding absence on the shared filesystem.
            data = None
            for attempt in (0, 1):
                try:
                    data = json.loads(p.read_text())
                    break
                except FileNotFoundError:
                    if attempt == 0:
                        time.sleep(0.5)
            self._pack_cache[label] = data
        return self._pack_cache[label]

    def check_reference(self, ref: dict) -> tuple[bool, str]:
        if "pack" in ref:
            pack = self._pack(str(ref["pack"]))
            if pack is None:
                return False, f"no pack {ref['pack']!r}"
            value, err = _walk_path(pack, str(ref.get("path", "")))
            if err:
                return False, f"pack path: {err}"
            if "value" in ref and not _values_match(ref["value"], value):
                return False, f"value mismatch: cited {ref['value']!r}, pack has {value!r}"
            return True, f"resolved to {str(value)[:80]}"
        if "table" in ref:
            for pair in self.tables.get("pairs", []):
                if pair["pair"] == ref["table"]:
                    row = pair["rows"].get(str(ref.get("row", "")))
                    if row is None:
                        return False, f"row {ref.get('row')!r} not in table"
                    side = str(ref.get("side", "left"))
                    if side not in ("left", "right"):
                        return False, f"bad side {side!r}"
                    actual = row.get(side)
                    if "value" in ref and not _values_match(ref["value"], actual):
                        return False, (
                            f"value mismatch: cited {ref['value']!r}, "
                            f"table has {actual!r}"
                        )
                    return True, f"resolved to {actual!r}"
            return False, f"no pair table {ref['table']!r}"
        if "probe" in ref:
            p = self.out_dir / "probes" / f"{ref['probe']}.out"
            if not p.is_file():
                return False, f"no probe output {ref['probe']!r}"
            text = p.read_text(errors="replace")
            if text.startswith(("[REPRINT-ONLY PROBE", "[STALE-SOURCED PROBE")):
                return False, (
                    "probe is not a measurement (reprint-only or stale-sourced "
                    "— see its output header); cite a probe that computes the "
                    "value from the corpus/packs/run files"
                )
            needle = str(ref.get("contains", ""))
            if needle and needle not in text:
                return False, "probe output does not contain the cited string"
            return True, "probe output present" + (" and contains string" if needle else "")
        if "file" in ref:
            p = Path(str(ref["file"]))
            if not p.is_file():
                return False, f"file missing: {p}"
            needle = str(ref.get("contains", ""))
            if needle and not _file_contains(p, needle):
                return False, "file does not contain the cited string"
            return True, "file present" + (" and contains string" if needle else "")
        return False, f"unrecognized reference shape: {sorted(ref.keys())}"

    def check_finding(self, finding: dict) -> dict:
        results = []
        ok_count = 0
        for ref in finding.get("evidence", []):
            if not isinstance(ref, dict):
                results.append({"ref": ref, "ok": False, "detail": "not an object"})
                continue
            ok, detail = self.check_reference(ref)
            ok_count += ok
            results.append({"ref": ref, "ok": ok, "detail": detail})
        n = len(results)
        status = (
            "verified" if n and ok_count == n
            else "failed" if ok_count == 0
            else "partial"
        )
        return {
            "id": finding.get("id"),
            "claim": str(finding.get("claim", ""))[:300],
            "status": status,
            "checked": n,
            "passed": ok_count,
            "references": results,
        }


def _ref_str(ref: dict) -> str:
    if "pack" in ref:
        v = f" = {ref['value']}" if "value" in ref else ""
        return f"pack `{ref['pack']}` → `{ref.get('path')}`{v}"
    if "table" in ref:
        return (f"table `{ref['table']}` row `{ref.get('row')}` "
                f"({ref.get('side')}) = {ref.get('value')}")
    if "probe" in ref:
        c = f" contains \"{str(ref.get('contains'))[:80]}\"" if ref.get("contains") else ""
        return f"probe `{ref['probe']}`{c}"
    if "file" in ref:
        c = f" contains \"{str(ref.get('contains'))[:80]}\"" if ref.get("contains") else ""
        return f"file `{str(ref['file'])[-70:]}`{c}"
    return str(ref)[:120]


def render_findings_md(findings_path: Path, checks: dict) -> str:
    """Human-readable rendering of findings.jsonl + verification results."""
    mark = {"verified": "✅", "partial": "⚠️", "failed": "❌"}
    lines = ["# Findings (human-readable)", "",
             "Every finding lists its machine-checked evidence references and "
             "the fact-check outcome.", ""]
    with open(findings_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                f = json.loads(line)
            except json.JSONDecodeError:
                continue
            check = checks.get(f.get("id")) or {}
            status = check.get("status", "unchecked")
            lines.append(f"## #{f.get('id')} {mark.get(status, '·')} "
                         f"[{status}] — scope: {f.get('scope', '?')}")
            lines.append("")
            if f.get("choice"):
                lines.append(f"**Choice judged:** {f['choice']}")
            if f.get("assessment"):
                lines.append(f"**Assessment:** {f['assessment']}")
            lines.append("")
            lines.append(str(f.get("claim", "")).strip())
            lines.append("")
            if f.get("notes"):
                lines.append(f"*Notes:* {f['notes']}")
                lines.append("")
            lines.append("Evidence:")
            ref_checks = {json.dumps(r["ref"], sort_keys=True): r
                          for r in check.get("references", [])}
            for ref in f.get("evidence", []):
                rc = ref_checks.get(json.dumps(ref, sort_keys=True)) if isinstance(ref, dict) else None
                ok = "✓" if rc and rc.get("ok") else ("✗" if rc else "·")
                lines.append(f"- {ok} {_ref_str(ref) if isinstance(ref, dict) else str(ref)[:120]}")
            lines.append("")
    return "\n".join(lines)


_REPORT_NUM_RE = re.compile(r"\d[\d,]*\.\d+|\d[\d,]{2,}")


def load_admissible_sources(out_dir: Path,
                            packs_dir: Path) -> tuple[str, str, str]:
    """Raw (deterministic corpus text, clean probe text, inadmissible probe
    text) — the shared source pool every number-tracing check draws from.
    Returned unnormalized: callers that substring-match strip thousands
    separators themselves; float extraction must see the original commas as
    separators or adjacent probe values would merge into garbage tokens."""
    det_parts = []
    corpus_dir = packs_dir.parent
    for name in ("bench.md", "bench.json", "tables.md", "tables.json",
                 "referee.json", "token_accounting.md", "corpus.json"):
        p = corpus_dir / name
        if p.is_file():
            det_parts.append(p.read_text(errors="replace"))
    for p in packs_dir.glob("*.json"):
        det_parts.append(p.read_text(errors="replace"))
    clean_parts, dirty_parts = [], []
    probes_dir = out_dir / "probes"
    if probes_dir.is_dir():
        for p in probes_dir.glob("probe_*.out"):
            t = p.read_text(errors="replace")
            if t.startswith(("[REPRINT-ONLY PROBE", "[STALE-SOURCED PROBE")):
                dirty_parts.append(t)
            else:
                clean_parts.append(t)
    return ("\n".join(det_parts), "\n".join(clean_parts),
            "\n".join(dirty_parts))


_CHART_BLOCK_RE = re.compile(r"```chart[ \t]*\n(.*?)```", re.DOTALL)
_SERIES_LINE_RE = re.compile(r"^\s*(series|band)\s+([^:]+):\s*(.+)$")
_FLOAT_TOKEN_RE = re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?")


def audit_progression_series(text: str, det: str, clean: str) -> dict:
    """Verify progression-form chart series against admissible sources.

    Scope: series/band lines whose payload is x,value pairs ("1,0.0238;
    2,…") — the scored-attempt trajectory shape. Three observer reports in
    a row garbled these; the last (obs4, 2026-08-03) ran the correct
    std.trajectory probe and then wrote different values into its charts,
    and the body-number audit bins chart payloads into their own uncounted
    bucket, so nothing fired. A value is sourced if its exact text appears
    in a source or some source number rounds to it at its printed
    precision. Plain value lists (hist/density payloads) and pipe-form rows
    (bars/box) are exempt: stated-arithmetic transforms there — percent
    gaps, KB conversions — are legitimate reviewer discretion. Fabricating
    a scored-attempt series is not discretion anywhere.
    """
    import bisect

    src_floats: list[float] = []
    for blob in (det, clean):
        for tok in _FLOAT_TOKEN_RE.findall(blob):
            try:
                src_floats.append(float(tok))
            except ValueError:
                pass
    src_floats.sort()

    def _sourced(tok: str) -> bool:
        if tok in det or tok in clean:
            return True
        try:
            v = float(tok)
        except ValueError:
            return False
        dp = len(tok.split(".", 1)[1]) if "." in tok else 0
        half = 0.5 * 10 ** -dp
        i = bisect.bisect_left(src_floats, v - half)
        return i < len(src_floats) and src_floats[i] < v + half

    checked = 0
    flags: list[dict] = []
    for block in _CHART_BLOCK_RE.findall(text):
        tm = re.search(r"^\s*title:\s*(.+)$", block, re.MULTILINE)
        title = tm.group(1).strip() if tm else ""
        for line in block.splitlines():
            sm = _SERIES_LINE_RE.match(line)
            if not sm:
                continue
            groups = [g for g in sm.group(3).split(";") if g.strip()]
            if len(groups) < 3:
                continue  # plain value list, not x,value pair form
            split_groups = [[p.strip() for p in g.split(",")] for g in groups]
            pairish = sum(1 for parts in split_groups if 2 <= len(parts) <= 5)
            if pairish < 0.9 * len(groups):
                continue
            vals = [p for parts in split_groups if 2 <= len(parts) <= 5
                    for p in parts[1:] if _FLOAT_TOKEN_RE.fullmatch(p)]
            distinct = sorted(set(vals))
            if len(distinct) < 4:
                continue
            checked += 1
            unmatched = [v for v in distinct if not _sourced(v)]
            if len(unmatched) > 0.2 * len(distinct):
                flags.append({
                    "chart": title[:90],
                    "series": sm.group(2).strip()[:60],
                    "distinct_values": len(distinct),
                    "unmatched": len(unmatched),
                    "examples": unmatched[:5],
                })
    return {"series_checked": checked, "series_flagged": len(flags),
            "flags": flags}


def _collect_best_entries(node, under_best=False, acc=None) -> dict:
    """{experiment name: set of referee scores} from every 'best' subtree."""
    if acc is None:
        acc = {}
    if isinstance(node, dict):
        name = node.get("experiment") if under_best else None
        if isinstance(name, str):
            acc.setdefault(name, set())
            score = node.get("referee_score")
            if isinstance(score, (int, float)):
                acc[name].add(float(score))
        for k, v in node.items():
            _collect_best_entries(v, under_best or k == "best", acc)
    elif isinstance(node, list):
        for item in node:
            _collect_best_entries(item, under_best, acc)
    return acc


def _collect_best_names(node) -> set:
    """Every string under an 'experiment' key inside any 'best' subtree."""
    return set(_collect_best_entries(node))


_BACKTICK_NAME_RE = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_.@-]{3,})`")
_REFEREE_HDR_RE = re.compile(
    r"referee[- ]?(selected|declared)|referee.{0,20}(champion|winner|best)",
    re.I)


def audit_referee_attributions(text: str, referee: dict) -> dict:
    """Verify names printed as the referee's selections against referee.json.

    A reviewer retyped two champion names in a table headed
    'referee-selected champion' (obs5, 2026-08-03): one name existed only in
    a non-champion leaderboard row, one existed nowhere. Every value and
    winner was correct — only the name strings drifted, invisible to
    numeric checks. Scope: markdown tables whose header ties the referee to
    champion/winner/best/selected, plus prose lines saying
    'referee-selected'. Backticked experiment-name tokens there must be
    exact members of the referee's declared best set, AND each
    referee-table row naming a best-set experiment must carry that
    experiment's referee score (a report was relaunched 2026-08-03 for
    printing a raw-artifact 197.788 in the referee column where the
    referee publishes 186.284 — names alone passed the gate).
    """
    best_entries = _collect_best_entries(referee)
    best_names = set(best_entries)
    if not best_names:
        return {"names_checked": 0, "violations": []}
    checked = 0
    violations: list[dict] = []
    # referee.json schema fields legitimately appear backticked in prose
    # and definition rows; only experiment-name-shaped tokens are claims.
    schema_words = {"winner", "best", "experiment", "pairs", "leaderboard",
                    "referee_score", "self_report_identified", "identity",
                    "truth_pool_conflicts", "shared_origins",
                    "lower_is_better", "experiments_on_board",
                    "referee.json"}

    num_re = re.compile(r"-?\d+\.\d+")

    def _row_has_score(line: str, scores: set) -> bool:
        for tok in num_re.findall(line):
            v = float(tok)
            dp = len(tok.split(".", 1)[1])
            if any(abs(v - s) < 10 ** -dp or round(s, dp) == v
                   for s in scores):
                return True
        return False

    def _check_line(line: str, where: str) -> None:
        nonlocal checked
        for tok in _BACKTICK_NAME_RE.findall(line):
            if tok.replace(".", "").replace("-", "").isdigit():
                continue
            if tok in schema_words or "_" not in tok or len(tok) < 8:
                continue
            checked += 1
            if tok not in best_names:
                violations.append({"name": tok, "where": where,
                                   "line": line.strip()[:120]})
            elif (where == "referee-table row" and best_entries[tok]
                  and not _row_has_score(line, best_entries[tok])):
                violations.append({
                    "name": tok, "where": "missing-referee-value",
                    "line": line.strip()[:120],
                    "expected": sorted(best_entries[tok])})

    lines = text.splitlines()
    in_ref_table = False
    for i, line in enumerate(lines):
        is_row = line.lstrip().startswith("|")
        if is_row and _REFEREE_HDR_RE.search(line):
            in_ref_table = True
        elif not is_row:
            in_ref_table = False
        if is_row and in_ref_table:
            _check_line(line, "referee-table row")
        elif re.search(r"referee[- ]selected", line, re.I):
            _check_line(line, "referee-selected prose")
    return {"names_checked": checked, "violations": violations}


def audit_report_numbers(out_dir: Path, packs_dir: Path) -> dict:
    """Trace every number in REPORT.md to an admissible source.

    Sources, in order: the deterministic scorecards next to the corpus
    (bench/tables/referee/token_accounting/corpus) and packs; then outputs of
    the session's own data-reading probes. Numbers found only in
    reprint/stale probes are hard failures (manufactured evidence); numbers
    found nowhere are flagged for judgment — inline-derived arithmetic lands
    here legitimately, invented figures also land here.
    """
    report = out_dir / "REPORT.md"
    if not report.is_file():
        return {}
    text = report.read_text(errors="replace")
    # the production-cost footer is machinery bookkeeping (numbers come
    # from usage.jsonl, not probes) — exempt from the body audit
    from runcmp.render_html import (
        PRODUCTION_COST_MARKER)
    text = text.split(PRODUCTION_COST_MARKER)[0]

    def _norm(s: str) -> str:
        return s.replace(",", "")

    det_raw, clean_raw, dirty_raw = load_admissible_sources(out_dir, packs_dir)
    det = _norm(det_raw)
    clean = _norm(clean_raw)
    dirty = _norm(dirty_raw)

    # Chart-block payloads and shown-arithmetic derivations are counted in
    # their own buckets: chart rows carry rounded/subsampled/derived values
    # that legitimately match no source verbatim, and the mission expressly
    # permits "a value derived by arithmetic ... show the inputs and the
    # operation" (2026-08-01: a distribution-rich report was 53% "unsourced"
    # under the old counter, nearly all of it chart data and displayed
    # arithmetic — the counter punished mission compliance).
    import re as _re2
    chart_spans = [mm.span() for mm in
                   _re2.finditer(r"```chart.*?```", text, _re2.DOTALL)]

    def _in_chart(pos: int) -> bool:
        return any(a <= pos < b for a, b in chart_spans)

    # Rounding index: report prose legitimately rounds sourced values
    # (14.57 for 14.5675); exact-substring matching branded those
    # "unsourced" and pushed an honest report over the 10% bar
    # (2026-08-03). Built from source floats once, matched at the printed
    # precision — same rule the progression-series audit uses.
    src_floats: list[float] = []
    for tok in _FLOAT_TOKEN_RE.findall(det + "\n" + clean):
        try:
            src_floats.append(float(tok))
        except ValueError:
            pass
    src_floats.sort()
    import bisect as _bisect

    def _rounds_from_source(v: str) -> bool:
        if "." not in v:
            return False
        try:
            f = float(v)
        except ValueError:
            return False
        half = 0.5 * 10 ** -len(v.split(".", 1)[1])
        i = _bisect.bisect_left(src_floats, f - half)
        return i < len(src_floats) and src_floats[i] < f + half

    hex_spans = [mm.span() for mm in
                 _re2.finditer(r"\b[0-9a-f]{40,}\b", text)]

    def _in_hex_blob(pos: int) -> bool:
        return any(a <= pos < b for a, b in hex_spans)

    # Token-ledger pair sums: reviewers legitimately add two per-seat
    # ledger values ("conductor+verifier consumed 4,567,931 input tokens")
    # and the exact-match pass branded those honest sums "unsourced"
    # (three such totals in one audited report, 2026-08-08). Restricted to
    # integers from token_accounting.* only — pair sums over ALL source
    # numbers would be dense enough to launder invented figures.
    ledger_ints: list[int] = []
    for name in ("token_accounting.json", "token_accounting.md"):
        p = packs_dir.parent / name
        if p.is_file():
            for tok in _re2.findall(r"\b\d{4,}\b",
                                    _norm(p.read_text(errors="replace"))):
                try:
                    ledger_ints.append(int(tok))
                except ValueError:
                    pass
    ledger_sorted = sorted(set(ledger_ints))
    ledger_set = set(ledger_sorted)

    def _sum_of_two_ledger_values(v: str) -> bool:
        if "." in v:
            return False
        try:
            n = int(v)
        except ValueError:
            return False
        if n < 10_000:
            return False
        return any((n - a) in ledger_set and (n - a) >= a
                   for a in ledger_sorted if a <= n // 2)

    counts = {"deterministic": 0, "clean_probe": 0,
              "only_inadmissible_probe": 0, "chart_data": 0,
              "derived_shown": 0, "rounded_from_source": 0,
              "ledger_pair_sum": 0, "unsourced": 0}
    inadmissible, unsourced = [], []
    seen: set[str] = set()
    for m in _REPORT_NUM_RE.finditer(text):
        v = _norm(m.group())
        if v in seen or len(v.replace(".", "")) < 3:
            continue
        # A digit run inside a sha256/hex fingerprint is not a number claim.
        if _in_hex_blob(m.start()):
            continue
        seen.add(v)
        ctx = text[max(0, m.start() - 70):m.end() + 40].replace("\n", " ")
        # "=$131.05" is shown arithmetic; the bare endswith("=") test missed
        # the currency sign and branded displayed derivations "unsourced".
        prefix = text[:m.start()].rstrip().rstrip("$").rstrip()
        if v in det:
            counts["deterministic"] += 1
        elif v in clean:
            counts["clean_probe"] += 1
        elif v in dirty:
            counts["only_inadmissible_probe"] += 1
            inadmissible.append({"value": v, "context": ctx})
        elif _in_chart(m.start()):
            counts["chart_data"] += 1
        elif prefix.endswith("="):
            counts["derived_shown"] += 1
        elif _rounds_from_source(v):
            counts["rounded_from_source"] += 1
        elif _sum_of_two_ledger_values(v):
            counts["ledger_pair_sum"] += 1
        else:
            counts["unsourced"] += 1
            unsourced.append({"value": v, "context": ctx})
    return {"counts": counts, "only_inadmissible_probe": inadmissible,
            "unsourced": unsourced}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fact-check investigator findings")
    ap.add_argument("--out", required=True, type=Path,
                    help="runcmp output dir (findings.jsonl lives here)")
    ap.add_argument("--packs", required=True, type=Path)
    args = ap.parse_args(argv)

    findings_path = args.out / "findings.jsonl"
    if not findings_path.is_file():
        print("no findings.jsonl — nothing to check")
        return 1
    checker = FactChecker(args.out, args.packs)
    results = []
    with open(findings_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                finding = json.loads(line)
            except json.JSONDecodeError:
                continue
            results.append(checker.check_finding(finding))
    body_audit = audit_report_numbers(args.out, args.packs)
    chart_audit: dict = {}
    ref_audit: dict = {}
    report_md = args.out / "REPORT.md"
    if report_md.is_file():
        from runcmp.render_html import (
            PRODUCTION_COST_MARKER)
        report_text = report_md.read_text(
            errors="replace").split(PRODUCTION_COST_MARKER)[0]
        det_raw, clean_raw, _dirty = load_admissible_sources(args.out,
                                                             args.packs)
        chart_audit = audit_progression_series(report_text, det_raw,
                                               clean_raw)
        referee_path = args.packs.parent / "referee.json"
        if referee_path.is_file():
            ref_audit = audit_referee_attributions(
                report_text, json.loads(referee_path.read_text()))
    summary = {
        "total": len(results),
        "verified": sum(r["status"] == "verified" for r in results),
        "partial": sum(r["status"] == "partial" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "findings": results,
        "report_body_numbers": body_audit,
        "progression_series": chart_audit,
        "referee_attributions": ref_audit,
    }
    (args.out / "verification.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )
    (args.out / "findings.md").write_text(
        render_findings_md(findings_path, {r["id"]: r for r in results})
    )
    print(
        f"fact-check: {summary['verified']} verified, "
        f"{summary['partial']} partial, {summary['failed']} failed "
        f"of {summary['total']}"
    )
    if body_audit:
        c = body_audit["counts"]
        print(
            f"report-body numbers: {c['deterministic']} deterministic, "
            f"{c['clean_probe']} probe-computed, "
            f"{c['only_inadmissible_probe']} ONLY-in-inadmissible-probes, "
            f"{c.get('chart_data', 0)} chart-data, "
            f"{c.get('derived_shown', 0)} derived-with-shown-arithmetic, "
            f"{c.get('rounded_from_source', 0)} rounded-from-source, "
            f"{c.get('ledger_pair_sum', 0)} token-ledger-pair-sums, "
            f"{c['unsourced']} unsourced"
        )
    if chart_audit:
        print(f"progression series: {chart_audit['series_checked']} checked, "
              f"{chart_audit['series_flagged']} FLAGGED"
              + ("" if not chart_audit["series_flagged"] else
                 " (series values match no admissible source — fabricated"
                 " or garbled trajectory data)"))
        for f in chart_audit["flags"][:8]:
            print(f"  [UNSOURCED-SERIES] {f['series']!r} in {f['chart']!r}: "
                  f"{f['unmatched']}/{f['distinct_values']} distinct values "
                  "unsourced, e.g. " + ", ".join(f["examples"][:3]))
    if ref_audit:
        print(f"referee attributions: {ref_audit['names_checked']} names "
              f"checked, {len(ref_audit['violations'])} FLAGGED"
              + ("" if not ref_audit["violations"] else
                 " (names presented as referee selections that are not in "
                 "referee.json's best set)"))
        for v in ref_audit["violations"][:8]:
            print(f"  [WRONG-ATTRIBUTION] {v['name']} ({v['where']}): "
                  f"{v['line'][:100]}")
    # Chart-mix census: variety is a report-quality requirement (2026-07-31:
    # a round shipped 47 charts, every one a bar chart). Counted per report,
    # printed so a regression to one shape is visible at check time.
    hard_flags: list[str] = []
    report_path = args.out / "REPORT.md"
    if report_path.is_file():
        import re as _re
        text = report_path.read_text(errors="replace")
        blocks = _re.findall(r"```chart[ \t]*\n(.*?)```", text, _re.DOTALL)
        kinds = {"bars": 0, "line": 0, "scatter": 0, "stacked": 0}
        for b in blocks:
            m = _re.search(r"^\s*type:\s*(\w+)", b, _re.MULTILINE)
            kind = (m.group(1).lower() if m else "bars")
            kinds[kind] = kinds.get(kind, 0) + 1
        ascii_fences = len(_re.findall(r"```text", text))
        # Distribution forms counted separately: the user's ruling
        # (2026-08-01) requires >=4 distinct distribution forms at this
        # corpus size; scatter counts only with marginals: true.
        dist_forms = sum(1 for k in ("hist", "density", "box") if kinds.get(k))
        band_blocks = sum(1 for b in blocks
                          if _re.search(r"^\s*band ", b, _re.MULTILINE))
        marg_scatters = sum(1 for b in blocks
                            if _re.search(r"^\s*type:\s*scatter", b, _re.MULTILINE)
                            and _re.search(r"^\s*marginals:\s*true", b, _re.MULTILINE))
        dist_forms += (1 if band_blocks else 0) + (1 if marg_scatters else 0)
        from runcmp.render_html import chart_errors
        unrenderable = chart_errors(text)
        if unrenderable:
            print(f"CHART RENDER FAILURES: {len(unrenderable)} of "
                  f"{len(blocks)} blocks will NOT render — the HTML shows "
                  "raw fence text. First offenders:")
            for e in unrenderable[:4]:
                print(f"  [UNRENDERABLE] {e}")
            hard_flags.append(f"{len(unrenderable)} unrenderable chart "
                              "block(s)")
        # The shipped page is the product: verify the HTML actually carries
        # one rendered container per block (markdown-level validation alone
        # missed pages whose HTML predated a renderer change).
        html_path = args.out / "REPORT.html"
        if html_path.is_file():
            html = html_path.read_text(errors="replace")
            containers = html.count('<div class="chart">')
            if containers != len(blocks):
                print(f"HTML CHART MISMATCH: {containers} rendered "
                      f"containers for {len(blocks)} blocks in REPORT.html "
                      "— regenerate the HTML.")
                hard_flags.append("REPORT.html chart containers do not "
                                  "match the markdown blocks")
        print(f"charts: {len(blocks)} total — "
              + ", ".join(f"{v} {k}" for k, v in kinds.items() if v)
              + f"; distribution forms: {dist_forms} of >=4 required"
              + (f" (band blocks {band_blocks}, marginal scatters {marg_scatters})"
                 if band_blocks or marg_scatters else "")
              + (f"; {ascii_fences} ASCII text fences" if ascii_fences else "")
              + ("; SINGLE-SHAPE REPORT (variety requirement not met)"
                 if len(blocks) >= 6 and sum(1 for v in kinds.values() if v) == 1
                 else ""))
        for item in body_audit["only_inadmissible_probe"][:10]:
            print(f"  [INADMISSIBLE] {item['value']} :: {item['context'][:110]}")
    for r in results:
        if r["status"] != "verified":
            print(f"  [{r['status']}] #{r['id']}: {r['claim'][:100]}")
            for ref in r["references"]:
                if not ref["ok"]:
                    print(f"      ✗ {json.dumps(ref['ref'])[:120]} — {ref['detail']}")
    # Exit status is the verdict, not the process health: this command once
    # returned 0 beside three referee-attribution violations and the caller
    # took command success for verified-report success (2026-08-11). 0 now
    # means every hard audit is clean; anything flagged exits 1. Partial
    # findings and the body-number census stay informational — they are
    # graded judgments, recorded in verification.json, not hard violations.
    if summary["failed"]:
        hard_flags.insert(
            0, f"{summary['failed']} finding(s) FAILED re-verification")
    if chart_audit and chart_audit.get("series_flagged"):
        hard_flags.append(f"{chart_audit['series_flagged']} progression "
                          "series with unsourced values")
    if ref_audit and ref_audit.get("violations"):
        hard_flags.append(f"{len(ref_audit['violations'])} referee-"
                          "attribution violation(s)")
    if hard_flags:
        print("FACTCHECK FLAGS (exit 1 — command success is not report "
              "success): " + "; ".join(hard_flags))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
