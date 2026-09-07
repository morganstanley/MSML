"""Benchmark lineup: the declarative registry of comparison domains.

``lineup.json`` names every domain the benchmarking campaign knows how to
run and how to referee. This module is the only reader; the rest of runcmp
consumes it through three narrow functions:

- :func:`detect_domain` — corpus.py maps a run path to a lineup id.
- :func:`referee_specs` — referee.py merges non-builtin specs into
  ``DOMAIN_SPECS`` so new domains score without code changes (as long as
  their ``kind`` already exists).
- :func:`emit_configs` — generates per-cell run configs for a campaign from
  a lineup selection, one framework x model per cell. A cell config carries
  the same model in every seat (main and conductor) by design: a benchmark
  run must measure one model, not a blend.

CLI (wired in ``__main__``)::

    python -m alpha_lab.benchmarks.runcmp lineup --list
    python -m alpha_lab.benchmarks.runcmp lineup --out <campaign_dir> \
        --domains d5_rfq,d6_cuda \
        --models "glm=glm:glm-5.2,o48=bedrock:claude-opus-4-8" \
        [--frameworks cond,msml] [--max-experiments 50]

The generator writes ``<campaign_dir>/<domain>/<domain>_<model>_<fw>/config.json``
plus a ``LINEUP.md`` index. It never launches anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LINEUP_PATH = Path(__file__).with_name("lineup.json")

_REQUIRED_ENTRY_KEYS = ("id", "title", "detect", "metric", "framework_domains")


def load_lineup(path: Path | None = None) -> list[dict]:
    data = json.loads((path or LINEUP_PATH).read_text())
    domains = data.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError("lineup.json: 'domains' must be a non-empty list")
    seen: set[str] = set()
    for entry in domains:
        for key in _REQUIRED_ENTRY_KEYS:
            if key not in entry:
                raise ValueError(
                    f"lineup.json: entry {entry.get('id', '?')!r} missing {key!r}")
        if entry["id"] in seen:
            raise ValueError(f"lineup.json: duplicate domain id {entry['id']!r}")
        seen.add(entry["id"])
        ref = entry.get("referee") or {}
        if not ref.get("builtin") and "kind" not in ref:
            raise ValueError(
                f"lineup.json: entry {entry['id']!r} referee needs 'kind' or 'builtin'")
    return domains


def detect_domain(path_str: str, lineup: list[dict] | None = None) -> str | None:
    """Map a run path to a lineup domain id, longest token first.

    Longest-first ordering keeps one domain's token from shadowing a more
    specific sibling (e.g. ``_cuda`` must not claim ``d6_cuda_retry``\\'s
    sibling ``d5_rfq_cuda_mix`` — the most specific token wins).
    """
    lowered = path_str.lower()
    best: tuple[int, str] | None = None
    for entry in lineup or load_lineup():
        for token in entry["detect"]:
            if token.lower() in lowered:
                if best is None or len(token) > best[0]:
                    best = (len(token), entry["id"])
    return best[1] if best else None


def referee_specs(lineup: list[dict] | None = None) -> dict[str, dict]:
    """Non-builtin referee specs keyed by domain id, for DOMAIN_SPECS.update()."""
    specs: dict[str, dict] = {}
    for entry in lineup or load_lineup():
        ref = entry.get("referee") or {}
        if ref.get("builtin") or "kind" not in ref:
            continue
        specs[entry["id"]] = dict(ref)
    return specs


def _parse_models(arg: str) -> list[tuple[str, str, str]]:
    """Parse ``short=provider:model`` triples from a comma-separated list."""
    out = []
    for chunk in filter(None, (c.strip() for c in arg.split(","))):
        short, _, rest = chunk.partition("=")
        provider, _, model = rest.partition(":")
        if not (short and provider and model):
            raise ValueError(
                f"--models entry {chunk!r} must look like short=provider:model")
        out.append((short, provider, model))
    return out


def _cell_config(entry: dict, framework: str, provider: str, model: str,
                 max_experiments: int) -> dict:
    domain_field = entry["framework_domains"].get(framework)
    cfg: dict = {
        "data_path": entry.get("data_path", ""),
        "description": entry["title"],
        "target": entry.get("task", ""),
        "provider": provider,
        "model": model,
        "reasoning_effort": "low",
        "domain": domain_field,
        "pipeline": {
            "phases": ["phase1", "phase2", "phase3"],
            "phase3": {
                "executor": "local",
                "max_experiments": max_experiments,
            },
        },
    }
    if framework == "cond":
        # One model per run in EVERY seat. The framework's default conductor
        # (opus) silently mixes models into the cell if these are omitted.
        cfg["conductor_provider"] = provider
        cfg["conductor_model"] = model
    if entry.get("resource") == "cpu":
        cfg["pipeline"]["phase3"]["cpu_enabled"] = True
        cfg["pipeline"]["phase3"]["gpu_ids"] = []
    return cfg


def emit_configs(out_dir: Path, domains: list[str], models: list[tuple[str, str, str]],
                 frameworks: list[str], max_experiments: int = 50) -> list[Path]:
    lineup = load_lineup()
    by_id = {e["id"]: e for e in lineup}
    unknown = [d for d in domains if d not in by_id]
    if unknown:
        raise ValueError(f"not in lineup.json: {unknown}; known: {sorted(by_id)}")
    written: list[Path] = []
    index_lines = ["# Campaign lineup", ""]
    for dom in domains:
        entry = by_id[dom]
        index_lines.append(f"## {dom} — {entry['title']}")
        for short, provider, model in models:
            for fw in frameworks:
                cell = f"{dom}_{short}_{fw}"
                cell_dir = out_dir / dom / cell
                cell_dir.mkdir(parents=True, exist_ok=True)
                cfg_path = cell_dir / "config.json"
                cfg = _cell_config(entry, fw, provider, model, max_experiments)
                cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
                written.append(cfg_path)
                index_lines.append(f"- `{cell}` -> {cfg_path}")
        index_lines.append("")
    (out_dir / "LINEUP.md").write_text("\n".join(index_lines) + "\n")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="print the lineup and exit")
    ap.add_argument("--out", type=Path, help="campaign directory for generated configs")
    ap.add_argument("--domains", default="", help="comma-separated lineup ids")
    ap.add_argument("--models", default="",
                    help="comma-separated short=provider:model triples")
    ap.add_argument("--frameworks", default="cond,msml")
    ap.add_argument("--max-experiments", type=int, default=50)
    args = ap.parse_args(argv)
    lineup = load_lineup()
    if args.list or not args.out:
        for e in lineup:
            ref = e.get("referee") or {}
            kind = "builtin" if ref.get("builtin") else ref.get("kind", "-")
            print(f"{e['id']:10s} [{e.get('status', '?')}] referee={kind:22s} {e['title']}")
        return 0
    domains = [d for d in args.domains.split(",") if d]
    models = _parse_models(args.models)
    if not domains or not models:
        ap.error("--out needs --domains and --models")
    written = emit_configs(args.out, domains, models,
                           [f for f in args.frameworks.split(",") if f],
                           args.max_experiments)
    print(f"wrote {len(written)} configs under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
