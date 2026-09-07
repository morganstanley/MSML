"""Publish finished runcmp corpora into a standalone showcase MLflow instance.

Deterministic exporter — the only writer of the showcase store. Reads a
finished chain's outputs off disk (corpus.json, packs/, bench.json,
referee.json, report dirs with verification.json) and writes:

- one MLflow *experiment per benchmark task* (domain) — the home page's
  experiment list IS the benchmark index; every run on a task page shares
  one ruler, so every native sort/chart is valid by construction. Cell
  runs carry an open tag bag (every corpus/config/pack fact flattened),
  metrics, and per-cell step series;
- one MLflow run per campaign into the *campaign reports* experiment,
  holding the bench tables, coverage/win-matrix grids, investigator
  reports, and per-report verification counters.

MLflow does ALL the displaying. The chart side of every runs page
auto-draws one chart per metric; the UI groups those charts into sections
by the metric name's slash prefix (that is how its own `system/…` metrics
get a section). So the story is encoded in the metric NAMES: story
sections first (`1 verdict/…`, `2 race/…`, `3 economy/…` — digits sort
before letters), then one section per subsystem (`code/…`, `agents/…`,
`distributions/…`). Every page — including a virgin browser — therefore
shows every quantity, story-ordered, searchable via the native "Search
metric charts" and "Columns" boxes, with zero stored chart lists and
nothing for a human to build by hand. New metrics join automatically.

The only stored state is one small TABLE arrangement per task
(`mlflow.sharedViewState.standings`: leaderboard columns + sort; no chart
keys, so the auto-charts stay untouched — verified against MLflow 3.14).
Run names carry the referee rank (star) and a failure marker. The only
pre-rendered pictures are the ones MLflow has no chart type for: the
per-cell rollout timeline (Gantt-style) and the campaign coverage /
win-matrix grids — plain artifacts, never embedded in descriptions.

Ledger namespaces (a chart must never mix ledgers silently):
  det.*     extracted mechanically from run logs/DBs (bench.json registry)
  referee.* single-evaluator re-scores — the only cross-framework quality
  rel.*     arithmetic derived from referee values (rank / gap within domain)
  inv.*     investigator-produced numbers, admitted only via verification.json

Idempotent: cells are keyed by tag `cell_key` (+ `campaign`); re-publishing
deletes the prior MLflow run for the same key and writes fresh. The whole
store is disposable — burn it down and rebuild from disk at any time.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

# Fixed visual conventions: one color per framework everywhere; unknown
# (future) frameworks pick the next palette entry deterministically by name.
_FRAMEWORK_COLORS = {"cond": "#1f77b4", "msml": "#d62728"}
_EXTRA_COLORS = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
_MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_PERCENTILES = [5, 10, 25, 50, 75, 90, 95]
_PARAM_MAX = 480  # truncate long flattened values

# Experiment names double as the navigation: MLflow's home page lists
# experiments by name + first words of description. One experiment per
# benchmark task; task experiments are found by their `task_domain` tag so
# a task's display name can evolve without orphaning it.
REPORTS_EXPERIMENT = "campaign reports"
_REPORTS_LEGACY = ["benchmarks — campaigns (START HERE)", "bench/overview"]
_RETIRED_EXPERIMENTS = ["benchmarks — all runs", "bench/registry"]

HOWTO_NOTE = (
    "Every recorded quantity is already drawn: the middle icon above the "
    "runs table switches to the charts, one per quantity, grouped into "
    "sections (story sections 1–3 first, then one section per subsystem). "
    "Type a word — e.g. `comment` or `wall` — into **Search metric charts** "
    "there, or into the **Columns** button on the table, to pull up "
    "anything. Ledgers: `referee.*` is the single-evaluator re-score (the "
    "only cross-harness quality comparison), `rel.*` is rank/gap arithmetic "
    "derived from it, `det.*` is read mechanically from each run's own "
    "logs/DB — never compare det.* quality across harnesses. Series: "
    "`2 race/…` carries best-so-far with real wall-clock timestamps, "
    "`rollout detail/…` every experiment in DB order (step = experiment "
    "id), `distributions/…` quantile curves (step = percentile). Each "
    "run's artifacts hold its rollout timeline picture."
)
REPORTS_NOTE = (
    "One run per campaign, holding that campaign's referee winners, bench "
    "tables, coverage/win-matrix grids, and investigator reports "
    "(factcheck/* metrics are per-report verification counters). The "
    "benchmark tasks themselves are the `task …` experiments on the home "
    "page — one per task, every run of every campaign."
)


# --------------------------------------------------------------------------
# story layer (run descriptions + storyboard)

def _ranked(cells: list[dict]) -> dict[str, dict]:
    """Per domain: cells with a referee rank (in rank order) + the rest."""
    out: dict[str, dict] = {}
    for dom in sorted({c["domain"] for c in cells}):
        group = [c for c in cells if c["domain"] == dom]
        ranked = sorted((c for c in group if c.get("rel")),
                        key=lambda c: c["rel"]["rank_in_domain"])
        out[dom] = {"ranked": ranked,
                    "unranked": [c for c in group if not c.get("rel")]}
    return out


def _run_display_name(c: dict) -> str:
    """Run name carries the story the table shows by default: rank + health."""
    name = c["short"]
    rel = c.get("rel") or {}
    if rel:
        name += f"  ★{int(rel['rank_in_domain'])}"
    if str(c["entry"].get("completeness") or "complete") != "complete":
        name += "  ✗"
    return name


def _note_for_cell(c: dict, campaign: str, ov_exp_id: str,
                   ov_run_id: str) -> str:
    """Plain-markdown description for a cell run: headline facts + where
    the detail lives. Nothing hand-drawn, no link maps."""
    rel = c.get("rel") or {}
    det = c["det"]
    pm = c["pack_meta"]
    metric = pm.get("metric_key") or "metric"
    facts: list[str] = []
    if c["referee_best"] is not None:
        facts.append(f"**referee best ({metric})**: {_fmt(c['referee_best'])}")
    elif c["referee_meta"].get("rankable") is False:
        facts.append("**referee**: verified on its own frozen validation "
                     "slices — not score-comparable with this task's "
                     "leaderboard, so it holds no rank here")
    if rel:
        facts.append(f"**rank**: ★{int(rel['rank_in_domain'])} of "
                     f"{int(rel['n_ranked'])} in {c['domain']}")
        facts.append(f"**behind winner**: {rel['pct_behind_best']:.1f}%")
    cost = det.get("efficiency.cost_usd")
    if isinstance(cost, (int, float)):
        facts.append(f"**LLM spend**: ${cost:,.0f}")
    wall = det.get("lifecycle.wall_hours")
    if isinstance(wall, (int, float)):
        facts.append(f"**wall clock**: {wall:.1f} h")
    if pm.get("total") is not None:
        facts.append(f"**experiments**: {pm['total']}"
                     + (f" ({pm['scored']} scored)"
                        if pm.get("scored") is not None else ""))
    if isinstance(pm.get("execution_failures"), (int, float)):
        facts.append(f"**failed**: {int(pm['execution_failures'])}")
    if isinstance(pm.get("fix_attempts_total"), (int, float)):
        facts.append(f"**fix attempts**: {int(pm['fix_attempts_total'])}")
    comp = str(c["entry"].get("completeness") or "complete")
    if comp != "complete":
        facts.append(f"**RECORD INCOMPLETE — {comp}**")
    seats = c.get("seats") or {}
    seat_models = c.get("seat_models") or []
    if len(seat_models) > 1:
        # seat swaps are a routine benchmark variable — state the lineup
        # plainly (grouped by model), charts under "0 seats" carry the rest
        by_desc: dict[str, list[str]] = {}
        for role, desc in seats.items():
            by_desc.setdefault(desc, []).append(role)
        majority = max(by_desc, key=lambda d: len(by_desc[d]))
        parts = [f"{', '.join(sorted(rs))}: {d}"
                 for d, rs in sorted(by_desc.items()) if d != majority]
        parts.append(f"all other seats: {majority}")
        facts.append("**seats** — " + " · ".join(parts))
    elif seat_models:
        facts.append(f"**seats**: {seat_models[0]} in every recorded seat")
    lines = [
        f"### {c['domain']} · {c['framework']} · "
        f"{c['config'].get('model') or '?'}",
        " · ".join(facts),
        "Every recorded quantity of this run is drawn under **Model "
        "metrics**, and the metrics table below has a search box. The "
        "**Artifacts** tab holds `rollout/` — the wall-clock timeline of "
        "every experiment (a Gantt-style picture MLflow has no chart "
        "type for).",
    ]
    if ov_exp_id and ov_run_id:
        lines.append(f"Campaign report: [{campaign}](#/experiments/"
                     f"{ov_exp_id}/runs/{ov_run_id})")
    return "\n\n".join(lines)


def _note_for_overview(campaign: str, cells: list[dict]) -> str:
    """Plain-markdown description for a campaign report run: winners
    headline plus what the artifacts hold."""
    doms = _ranked(cells)
    fws = sorted({c["framework"] for c in cells})
    models = sorted({(c["config"].get("model") or "?") for c in cells})
    lines = [f"## {campaign} — {len(cells)} cells · harnesses: "
             f"{', '.join(fws)} · models: {', '.join(models)}"]
    wins = []
    for dom, g in doms.items():
        if g["ranked"]:
            w = g["ranked"][0]
            metric = w["pack_meta"].get("metric_key") or "metric"
            wins.append(f"**{dom}**: {w['short']} "
                        f"({metric} {_fmt(w['referee_best'])})")
    if wins:
        lines.append("Referee winners — " + " · ".join(wins))
    lines.append("The runs themselves live on the task pages (the "
                 "`task …` experiments on the home page) — every "
                 "campaign together, ★-ranked names, everything charted.")
    lines.append("Artifacts: `campaign/` (bench tables), `charts/` "
                 "(coverage map + win matrix — grid pictures MLflow has no "
                 "native chart type for), `reports/` (investigator "
                 "reports).")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------
# metric naming IS the layout. The MLflow 3 chart page auto-draws one chart
# per metric and groups the charts into sections by the metric name's slash
# prefix (verified 3.14: "1 verdict/x" and "code/y" land in sections named
# "1 verdict" and "code"; digits sort before letters, so the numbered story
# sections lead and the subsystem encyclopedia follows alphabetically).
# Renaming metrics is therefore the entire presentation layer: every page —
# including a virgin browser with no stored state — draws every quantity in
# story order, the native "Search metric charts" / "Columns" boxes reach
# all of them, and future metrics slot in with no code change here.

_STORY_SECTIONS = [
    # det.referee.* is the harness's own self-check — it must not share a
    # section with the real referee verdict ledger
    ("det.referee.", "self-check/"),
    ("referee.", "1 verdict/"),
    ("rel.", "1 verdict/"),
    ("det.progress.", "2 race/"),
    ("det.rollout.metric_value", "2 race/"),
    ("det.rollout.cum_cost_usd", "2 race/"),
    ("det.efficiency.cost_usd", "3 economy/"),
    ("det.lifecycle.wall_hours", "3 economy/"),
    ("det.efficiency.total_tokens_m", "3 economy/"),
    ("det.rollout.", "rollout detail/"),
    ("det.q.", "distributions/"),
    ("inv.", "factcheck/"),
]


def _sectioned(key: str) -> str:
    """Map a ledger metric key to its sectioned display key. The original
    ledger name stays intact after the slash."""
    for prefix, section in _STORY_SECTIONS:
        if key.startswith(prefix):
            return section + key
    if key.startswith("det."):
        return key.split(".", 2)[1] + "/" + key
    return key


_STANDINGS_COLUMNS = [
    "metrics.`1 verdict/referee.best_score`",
    "metrics.`1 verdict/rel.rank_in_domain`",
    "metrics.`1 verdict/rel.pct_behind_best`",
    "metrics.`3 economy/det.efficiency.cost_usd`",
    "metrics.`3 economy/det.lifecycle.wall_hours`",
    "params.`campaign`", "params.`framework`", "params.`model`",
    "params.`seats.models`",
]


def store_standings(client, exp_id: str, lower_is_better: bool) -> str:
    """One stored TABLE arrangement per task ('standings'): leaderboard
    columns + referee sort + every run visible. Deliberately carries no
    chart keys, so the chart side keeps MLflow's own draw-everything
    behavior (verified: a table-only stored state leaves the auto-charts
    intact). Returns the relative link."""
    state = {
        "searchFilter": "",
        "orderByKey": "metrics.`1 verdict/referee.best_score`",
        "orderByAsc": lower_is_better,
        "startTime": "ALL", "lifecycleFilter": "Active",
        "datasetsFilter": [], "modelVersionFilter": "All Runs",
        "selectedColumns": _STANDINGS_COLUMNS,
        "runsExpanded": {}, "runsPinned": [], "runsHidden": [],
        "runsVisibilityMap": {}, "runsHiddenMode": "SHOW_ALL",
        "viewMaximized": False, "runListHidden": False,
        "isAccordionReordered": False, "groupBy": None,
        "groupsExpanded": {}, "autoRefreshEnabled": False,
        "useGroupedValuesInCharts": True, "hideEmptyCharts": False,
        "globalLineChartConfig": {"xAxisKey": "step", "lineSmoothness": 0,
                                  "selectedXAxisMetricKey": ""},
        "chartsSearchFilter": "",
    }
    client.set_experiment_tag(exp_id, "mlflow.sharedViewState.standings",
                              json.dumps(state, separators=(",", ":")))
    return f"#/experiments/{exp_id}/runs?viewStateShareKey=standings"


def _task_note(metric: str, lower: bool, group: list[dict], campaign: str,
               standings_href: str) -> str:
    """Task experiment description. The first words double as the home
    page's tagline for this task."""
    ranked = sorted((c for c in group if c.get("rel")),
                    key=lambda c: c["rel"]["rank_in_domain"])
    if ranked:
        w = ranked[0]
        head = (f"Champion ({campaign}): {w['short']} — {metric} "
                f"{_fmt(w['referee_best'])}.")
    else:
        head = f"Latest campaign {campaign}: not refereed."
    return (f"{head} [Current standings]({standings_href}) — the runs "
            f"table sorted by referee score ({'lower' if lower else 'higher'} "
            "wins), one row per harness run, every campaign. " + HOWTO_NOTE)


# --------------------------------------------------------------------------
# loading / flattening

def _load(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _corpus_runs(corpus: Any) -> list[dict]:
    if isinstance(corpus, dict):
        return list(corpus.get("runs") or [])
    return list(corpus or [])


def _flatten(obj: Any, prefix: str = "") -> dict[str, str]:
    """Flatten nested dicts/lists into dotted string keys (the open bag)."""
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        text = json.dumps(obj, default=str)
        out[prefix] = text[:_PARAM_MAX]
    elif obj is not None:
        out[prefix] = str(obj)[:_PARAM_MAX]
    return out


def _pack_path(packs_dir: Path, label: str) -> Path:
    return packs_dir / (label.replace("/", "__") + ".json")


def _short(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _rollout_rows(pack: dict, experiment_table: list[dict]) -> list[dict]:
    """Full per-experiment rollout: DB order + timestamps, joined with the
    bench experiment table (metric/cost/tokens/fixes). Falls back to the
    table alone (no wall-clock) when the run's DB is unreadable."""
    et_by_id = {r.get("id"): r for r in experiment_table}
    join_keys = ("metric", "duration_seconds", "fix_attempts",
                 "est_cost_usd", "tokens_attributed", "error")
    rows: list[dict] = []
    sources = (pack.get("experiments") or {}).get("sources") or []
    db = next((s for s in sources if str(s).endswith(".db")), None)
    if db and os.path.exists(db):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            cur = con.execute(
                "SELECT id, name, status, started_at, finished_at "
                "FROM experiments ORDER BY id")
            for eid, name, status, st, fin in cur:
                et = et_by_id.get(eid) or {}
                rows.append({"id": eid, "name": name, "status": status,
                             "started_at": st, "finished_at": fin,
                             **{k: et.get(k) for k in join_keys}})
            con.close()
        except sqlite3.Error:
            rows = []
    if not rows:
        rows = [{"id": r.get("id"), "name": r.get("name"),
                 "status": r.get("status"), "started_at": None,
                 "finished_at": None, **{k: r.get(k) for k in join_keys}}
                for r in experiment_table]
    return rows


# --------------------------------------------------------------------------
# cell assembly (pure data; no mlflow)

def _seat_map(pack: dict) -> tuple[dict[str, str], list[str]]:
    """Who actually performed each agent role, from the run's own records.

    Preferred source: the token ledger (has the provider). Fallback: the
    model named on each logged request. Returns (role -> description,
    distinct models observed). Every seat can be a different LLM, and
    framework defaults / fallbacks can silently substitute one — so the
    observed assignment must be first-class viewer data, not archaeology.
    """
    seats: dict[str, str] = {}
    models: set[str] = set()
    ledger = (pack.get("seats") or {}).get("seats") or {}
    for role, pairs in ledger.items():
        parts = []
        for pair, n in pairs.items():
            prov, _, mdl = str(pair).partition(":")
            models.add(mdl)
            parts.append(f"{mdl} via {prov}"
                         + (f" ({n} calls)" if len(pairs) > 1 else ""))
        seats[role] = "; ".join(parts)
    if not seats:
        for role, r in ((pack.get("agent_logs") or {}).get("roles") or {}).items():
            mm = r.get("models") or {}
            if not mm:
                continue
            models.update(mm)
            seats[role] = "; ".join(
                m + (f" ({n} calls)" if len(mm) > 1 else "")
                for m, n in mm.items())
    return seats, sorted(models)


def _seat_counts(pack: dict) -> dict[str, dict[str, int]]:
    """role -> model -> logged-request count, for the '0 seats' charts.

    Seat swaps are a habitual benchmark variable, so the lineup must be
    chart-visible per run, not param archaeology. Uniform source first
    (the model named on each logged request — both frameworks); token
    ledger as fallback for packs predating per-role model capture.
    """
    out: dict[str, dict[str, int]] = {}
    for role, r in ((pack.get("agent_logs") or {}).get("roles") or {}).items():
        mm = {str(m): int(n) for m, n in (r.get("models") or {}).items()
              if isinstance(n, (int, float))}
        if mm:
            out[role] = mm
    if out:
        return out
    for role, pairs in ((pack.get("seats") or {}).get("seats") or {}).items():
        mm2: dict[str, int] = {}
        for pair, n in pairs.items():
            mdl = str(pair).partition(":")[2] or str(pair)
            mm2[mdl] = mm2.get(mdl, 0) + int(n)
        if mm2:
            out[role] = mm2
    return out


def build_cells(corpus_path: str, packs_dir: str, bench_path: str,
                referee_path: str) -> tuple[list[dict], dict]:
    """Join corpus + packs + bench + referee into per-cell records."""
    runs = _corpus_runs(_load(corpus_path))
    bench = _load(bench_path)
    referee = _load(referee_path)
    bench_runs = bench.get("runs") or {}
    ref_runs = referee.get("runs") or {}
    pairs = referee.get("pairs") or []

    # direction per domain, from referee pairs (fallback: pack direction)
    domain_lower_is_better: dict[str, bool] = {}
    for p in pairs:
        dom = (p.get("left") or "").split("/")[1] if "/" in (p.get("left") or "") else ""
        if dom and "lower_is_better" in p:
            domain_lower_is_better[dom] = bool(p["lower_is_better"])

    cells: list[dict] = []
    for entry in runs:
        label = entry.get("label") or ""
        pack: dict = {}
        ppath = _pack_path(Path(packs_dir), label)
        if ppath.exists():
            pack = _load(ppath)
        brun = bench_runs.get(label) or {}
        rrun = ref_runs.get(label) or {}
        pack_ex = pack.get("experiments") or {}

        direction = pack_ex.get("direction") or ""
        dom = entry.get("domain") or ""
        lower = domain_lower_is_better.get(dom, direction == "minimize")

        # referee kinds publish the per-experiment score under different
        # names (recomputed / official_referee_score / referee_score) —
        # sweep the whole family, first numeric wins. A run the referee
        # marked unrankable (verified on a DIFFERENT frozen slice) gets no
        # best score at all: its numbers must never sit on this task's
        # leaderboard or verdict charts.
        ref_scores = []
        if rrun.get("rankable") is not False:
            for e in (rrun.get("experiments") or []):
                for key in ("recomputed", "official_referee_score",
                            "referee_score"):
                    v = e.get(key)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        ref_scores.append(v)
                        break
        ref_best = (min(ref_scores) if lower else max(ref_scores)) if ref_scores else None

        experiment_table = brun.get("experiment_table") or []
        seats, seat_models = _seat_map(pack)
        cells.append({
            "label": label,
            "short": _short(label),
            "framework": entry.get("framework") or "",
            "domain": dom,
            "era": entry.get("era") or "",
            "entry": entry,
            "config": pack.get("config") or {},
            "seats": seats,
            "seat_models": seat_models,
            "seat_counts": _seat_counts(pack),
            "pack_meta": {k: pack_ex.get(k) for k in (
                "metric_key", "direction", "total", "scored", "improvements",
                "experiments_to_best", "time_to_best_seconds",
                "execution_failures", "fix_attempts_total")},
            "best_self": pack_ex.get("best") or {},
            "trajectory": pack_ex.get("trajectory") or [],
            "code_samples": (pack_ex.get("code_metrics") or {}).get("samples") or {},
            "agent_samples": {k: v for k, v in
                              ((pack.get("agent_logs") or {}).get("samples") or {}).items()
                              if isinstance(v, list)},
            "phase_windows": (pack.get("events") or {}).get("phase_windows") or {},
            "det": brun.get("metrics") or {},
            "experiment_table": experiment_table,
            "rollout": _rollout_rows(pack, experiment_table),
            "referee_best": ref_best,
            "referee_meta": {k: rrun.get(k) for k in (
                "scored_experiments", "experiments_with_predictions",
                "artifact_coverage", "holdout_rows", "rankable",
                "rankable_reason") if rrun.get(k) is not None},
            "lower_is_better": lower,
        })

    # rel.* within each domain, over cells that have a referee best
    by_dom: dict[str, list[dict]] = {}
    for c in cells:
        if c["referee_best"] is not None:
            by_dom.setdefault(c["domain"], []).append(c)
    for dom, group in by_dom.items():
        lower = group[0]["lower_is_better"]
        ranked = sorted(group, key=lambda c: c["referee_best"], reverse=not lower)
        best = ranked[0]["referee_best"]
        for i, c in enumerate(ranked, start=1):
            c["rel"] = {
                "rank_in_domain": float(i),
                "n_ranked": float(len(ranked)),
                "pct_behind_best": (abs(c["referee_best"] - best) / abs(best) * 100.0)
                                   if best else 0.0,
            }

    meta = {
        "bench_version": bench.get("bench_version"),
        "rules_version": bench.get("rules_version"),
        "generated_at": bench.get("generated_at"),
        "pairs": pairs,
        "metric_definitions": bench.get("metric_definitions"),
        "sources": {"corpus": str(corpus_path), "packs": str(packs_dir),
                    "bench": str(bench_path), "referee": str(referee_path)},
    }
    return cells, meta


FACEOFF_EXPERIMENT = "harness face-off — every task, one page"
GATES_EXPERIMENT = "change gates — PR evidence"


def publish_gate(client, gate: dict, campaign: str, owner: str,
                 artifact_root: str,
                 gate_dir: "Path | None" = None) -> tuple[str, str]:
    """One native MLflow run per gate evaluation: the change's evidence page.

    Screen-not-verdict by design: the run NAME carries the counts (flags,
    improvements) — the multidimensional evidence, not a stamp — while the
    mechanical policy reading stays a filterable parameter. The description
    is the full evidence table plus, when LLM reviewers have run over the
    gate's mission, each reviewer's explicit `Recommendation:` line, with
    their full reports attached under the run's artifacts. Re-running the
    same gate (same selectors + policy version) replaces its run.
    """
    note = (
        "One run per change-gate evaluation (a proposed harness change vs "
        "its baseline, judged by the versioned gate policy). The run NAME "
        "carries the verdict. Charts under **guards** show how much worse "
        "the candidate is per check per task cell (" + _WORSE_SIGN_NOTE_MD +
        "); a bar past the policy tolerance is the violation named in the "
        "description. Charts under **improvements** show the demonstrated "
        "gains. The description is the complete evidence table; the "
        "underlying runs live on the task pages and the harness face-off. "
        + HOWTO_NOTE)
    exp_id = _ensure_experiment(client, GATES_EXPERIMENT, artifact_root,
                                note)
    client.set_experiment_tag(exp_id, "mlflow.note.content", note)

    def _sel(side: dict) -> str:
        return " OR ".join(
            ",".join(f"{k}={v}" for k, v in clause.items())
            for clause in side["selector"])
    cand, base = _sel(gate["candidate"]), _sel(gate["baseline"])
    key = f"gate/{cand}|{base}|policy{gate.get('policy_version')}"
    _delete_existing(client, exp_id, f"tags.cell_key = '{key}'")
    verdict = gate.get("verdict", "?")
    nflag = len(gate.get("violations") or [])
    nimp = len(gate.get("improvements_shown") or [])
    run = client.create_run(exp_id, tags={
        "cell_key": key, "campaign": campaign, "owner": owner,
        "mlflow.runName": f"gate: {nflag} flag(s), {nimp} improvement(s)"
                          f" — {cand} vs {base}",
    })
    rid = run.info.run_id
    metrics: list[tuple] = [
        ("1 verdict/guard violations", float(len(gate["violations"])), 0),
        ("1 verdict/improvements demonstrated",
         float(len(gate["improvements_shown"])), 0),
        ("1 verdict/cells compared", float(len(gate["cells"])), 0),
    ]
    for cell in gate["cells"]:
        dom, model = cell["domain"], cell.get("model") or "any"
        for row in cell["guards"]:
            v = row.get("worse_pct", row.get("worse_abs"))
            if isinstance(v, (int, float)):
                metrics.append(
                    (_metric_safe(f"2 guards/pct worse: {row['label']}"
                                  f" : {dom}"), float(v), 0))
        for row in cell["improvements"]:
            v = row.get("worse_pct", row.get("worse_abs"))
            if isinstance(v, (int, float)) and row.get("improved"):
                metrics.append(
                    (_metric_safe(f"3 improvements/pct better: "
                                  f"{row['label']} : {dom}"),
                     float(-v), 0))
    params = [("policy screen", verdict),
              ("candidate", cand), ("baseline", base),
              ("policy_version", str(gate.get("policy_version"))),
              ("cells compared", str(len(gate["cells"]))),
              ("candidate runs", str(len(gate["candidate"]["runs"]))),
              ("baseline runs", str(len(gate["baseline"]["runs"])))]
    _log_batched(client, rid, metrics=metrics, params=params)
    from runcmp.gate import render_md
    note_md = render_md(gate)
    # LLM reviewer positions: any sibling review dir with a REPORT.md whose
    # first "Recommendation:" line states a position gets quoted on the
    # page and attached in full. The screen presents; reviewers opine.
    recs: list[str] = []
    if gate_dir is not None:
        for rep in sorted(Path(gate_dir).glob("*/REPORT.md")):
            first_rec = ""
            for line in rep.read_text(errors="replace").splitlines():
                if line.strip().lower().startswith("recommendation:"):
                    first_rec = line.strip()
                    break
            name = rep.parent.name
            if first_rec:
                recs.append(f"- **{name}**: {first_rec}")
                params.append((f"review.{name}",
                               first_rec[:_PARAM_MAX]))
            else:
                recs.append(f"- **{name}**: report attached, no explicit "
                            "Recommendation line")
            client.log_artifact(rid, str(rep), artifact_path=f"reviews/{name}")
            html = rep.with_name("REPORT.html")
            if html.is_file():
                client.log_artifact(rid, str(html),
                                    artifact_path=f"reviews/{name}")
    if recs:
        note_md += ("\n\n## Reviewer recommendations (full reports under "
                    "Artifacts -> reviews/)\n\n" + "\n".join(recs))
    _log_batched(client, rid, metrics=[], params=[p for p in params
                                                  if p[0].startswith("review.")])
    client.set_tag(rid, "mlflow.note.content", note_md)
    return exp_id, rid


_WORSE_SIGN_NOTE_MD = ("positive = candidate worse than baseline, "
                       "negative = better")


# MLflow metric names allow only alphanumerics, _ - . space : and / —
# policy labels are free text, so they are sanitized deterministically
# (offending characters become spaces) before becoming chart titles.
_METRIC_SAFE_RE = re.compile(r"[^A-Za-z0-9_.:/ -]+")


def _metric_safe(name: str) -> str:
    return " ".join(_METRIC_SAFE_RE.sub(" ", name).split())


def _faceoff_rollup(cells: list[dict]) -> dict[str, dict]:
    """framework -> its best ranked cell per domain + coverage + absences.

    The per-task pages answer "who wins this task"; this rollup feeds the
    ONE page that answers "which harness is better" without asking the
    reader to aggregate four pages in their head. The comparable unit is
    percent-behind-the-task-winner (scale-free, so every task fits one
    page); the run counts are shown because best-of-5-attempts vs
    best-of-1 is part of the truth.
    """
    out: dict[str, dict] = {}
    for c in cells:
        fw = c["framework"] or "?"
        f = out.setdefault(fw, {"best": {}, "unranked": {}, "counts": {},
                                "models": set()})
        dom = c["domain"]
        f["counts"][dom] = f["counts"].get(dom, 0) + 1
        model = (c["config"].get("model") or "").strip()
        if model:
            f["models"].add(model)
        rel = c.get("rel")
        if rel:
            cur = f["best"].get(dom)
            if (cur is None
                    or rel["pct_behind_best"] < cur["rel"]["pct_behind_best"]):
                f["best"][dom] = c
        elif dom not in f["best"]:
            f["unranked"].setdefault(
                dom, c["referee_meta"].get("rankable_reason")
                or "no referee-scored prediction artifacts")
    return out


def publish_faceoff(client, cells: list[dict], campaign: str, owner: str,
                    artifact_root: str) -> str:
    """One run per HARNESS; metric names carry one chart per task."""
    note = (
        "One row per harness, every task on one page. **1 verdict**: the "
        "mean-pct-behind chart is the single-glance answer; beside it, per "
        "task, the gap to the winner and the best score in the task's own "
        "units (referee-scored). **2 race**: line charts — each harness's "
        "winning run's self-reported best-so-far, per attempt (switch the "
        "x-axis to wall-clock time for the race against the clock); line "
        "length = attempt budget. **3 economy**: recorded LLM cost and "
        "wall hours per task — a missing bar means that harness recorded "
        "no ledger, not zero. **4 reliability**: scored fraction and runs "
        "fielded per task (best-of-many vs best-of-one is part of the "
        "picture). Runs a task's referee verified but refused to rank "
        "(different frozen validation data) are named in each row's "
        "description, never charted. " + HOWTO_NOTE)
    exp_id = _ensure_experiment(client, FACEOFF_EXPERIMENT, artifact_root,
                                note)
    roll = _faceoff_rollup(cells)
    # ONE metric name per domain: harnesses spell the same metric
    # differently (log_loss vs logloss) and a per-harness spelling splits
    # one task into two charts — the exact defect this page exists to kill
    metric_by_dom: dict[str, str] = {}
    for c in cells:
        mk = c["pack_meta"].get("metric_key")
        if mk and c["domain"] not in metric_by_dom:
            metric_by_dom[c["domain"]] = mk

    def _mean_pct(f: dict) -> float | None:
        vals = [c["rel"]["pct_behind_best"] for c in f["best"].values()]
        return sum(vals) / len(vals) if vals else None

    # creation order = naked-table order (newest on top): unranked
    # harnesses first, then worst-to-best, champion created last
    ordered = sorted(roll.items(),
                     key=lambda kv: (_mean_pct(kv[1]) is not None,
                                     -(_mean_pct(kv[1]) or 0.0)))
    ranked = [fw for fw, f in ordered if _mean_pct(f) is not None]
    for fw, f in ordered:  # worst first, champion last (naked-table order)
        mean = _mean_pct(f)
        star = (f" ★{len(ranked) - ranked.index(fw)}"
                if fw in ranked else "")
        _delete_existing(client, exp_id, f"tags.cell_key = 'faceoff/{fw}'")
        run = client.create_run(exp_id, tags={
            "cell_key": f"faceoff/{fw}", "campaign": campaign,
            "owner": owner, "mlflow.runName": f"{fw}{star}",
        })
        rid = run.info.run_id
        metrics: list[tuple] = []
        facts: list[str] = []
        for dom in sorted(f["counts"]):
            metrics.append((f"4 reliability/runs fielded: {dom}",
                            float(f["counts"][dom]), 0))
        for dom, c in sorted(f["best"].items()):
            rel = c["rel"]
            metric = metric_by_dom.get(dom) or "metric"
            metrics.append((f"1 verdict/pct behind winner: {dom}",
                            float(rel["pct_behind_best"]), 0))
            metrics.append((f"1 verdict/rank: {dom}",
                            float(rel["rank_in_domain"]), 0))
            if c["referee_best"] is not None:
                metrics.append((f"1 verdict/best {metric}: {dom}",
                                float(c["referee_best"]), 0))
            det = c["det"]
            for src, name in (
                    ("efficiency.cost_usd", f"3 economy/llm cost usd: {dom}"),
                    ("lifecycle.wall_hours", f"3 economy/wall hours: {dom}"),
                    ("lifecycle.scored_fraction",
                     f"4 reliability/scored fraction: {dom}")):
                v = det.get(src)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metrics.append((name, float(v), 0))
            # the race: the winning run's self-reported best-so-far per
            # attempt, with real finish timestamps so the chart's x-axis
            # can switch to wall-clock — the per-attempt story, not just
            # the endpoint. Ledger stated in the name: self-reported.
            finished_by_name = {r.get("name"): r.get("finished_at")
                                for r in c["rollout"] if r.get("finished_at")}
            for t in c["trajectory"]:
                n = t.get("n")
                if (not isinstance(n, int)
                        or not isinstance(t.get("best_so_far"), (int, float))):
                    continue
                ts = finished_by_name.get(t.get("name"))
                ts_ms = (int(ts * 1000)
                         if isinstance(ts, (int, float)) else None)
                metrics.append(
                    (f"2 race/self-reported best {metric} so far: {dom}",
                     float(t["best_so_far"]), n, ts_ms))
            facts.append(
                f"**{dom}**: ★{int(rel['rank_in_domain'])} of "
                f"{int(rel['n_ranked'])} — best run `{c['short']}` "
                f"({metric} {_fmt(c['referee_best'])}, "
                f"{rel['pct_behind_best']:.1f}% behind the winner, "
                f"{f['counts'][dom]} run(s) fielded; its race line and "
                "economy bars come from that run)")
        if mean is not None:
            metrics.append(("1 verdict/mean pct behind task winners",
                            float(mean), 0))
        for dom, reason in sorted(f["unranked"].items()):
            facts.append(f"**{dom}**: present but unranked — {reason}")
        attempted = set(f["counts"])
        for dom in sorted({c["domain"] for c in cells} - attempted):
            facts.append(f"**{dom}**: no run submitted")
        models = ", ".join(sorted(f["models"])) or "not recorded"
        params = [("framework", fw), ("campaign", campaign),
                  ("models", models),
                  ("tasks ranked", str(len(f["best"])))]
        _log_batched(client, rid, metrics=metrics, params=params)
        client.set_tag(rid, "mlflow.note.content",
                       f"### {fw} — models: {models}\n\n" + "\n\n".join(facts))
    champ = ranked[-1] if ranked else "?"
    tagline = (f"Champion ({campaign}): {champ} — lowest mean pct behind "
               "task winners. ")
    client.set_experiment_tag(exp_id, "mlflow.note.content", tagline + note)
    return exp_id


# --------------------------------------------------------------------------
# mlflow writing

def _client(tracking_uri: str):
    import mlflow
    return mlflow.MlflowClient(tracking_uri=tracking_uri)


def _ensure_experiment(client, name: str, artifact_root: str, note: str,
                       legacy_names: list[str] | None = None) -> str:
    exp = client.get_experiment_by_name(name)
    for legacy in (legacy_names or []):
        if exp is not None:
            break
        old = client.get_experiment_by_name(legacy)
        if old is not None:  # migrate stores published before the rename
            client.rename_experiment(old.experiment_id, name)
            exp = client.get_experiment(old.experiment_id)
    if exp is not None:
        exp_id = exp.experiment_id
    else:
        exp_id = client.create_experiment(
            name,
            artifact_location=os.path.join(artifact_root, name.replace("/", "_")))
        client.set_experiment_tag(exp_id, "mlflow.note.content", note)
    # without this the MLflow 3 UI infers the GenAI/traces view for the
    # experiment page, which renders empty for a store with no traces
    client.set_experiment_tag(exp_id, "mlflow.experimentKind",
                              "custom_model_development")
    return exp_id


def _ensure_task_experiment(client, dom: str, metric: str, lower: bool,
                            artifact_root: str) -> str:
    """One experiment per benchmark task, found by its task_domain tag so
    the display name can evolve without orphaning the runs."""
    name = (f"task {dom} — {metric} "
            f"({'lower' if lower else 'higher'} wins)")
    found = client.search_experiments(
        filter_string=f"tags.task_domain = '{dom}'")
    if found:
        exp_id = found[0].experiment_id
        if found[0].name != name:
            client.rename_experiment(exp_id, name)
    else:
        exp_id = client.create_experiment(
            name, artifact_location=os.path.join(
                artifact_root, f"task_{dom}"))
        client.set_experiment_tag(exp_id, "task_domain", dom)
    client.set_experiment_tag(exp_id, "mlflow.experimentKind",
                              "custom_model_development")
    return exp_id


def _retire_legacy_experiments(client) -> None:
    """The pre-task-page store kept every cell in one registry experiment;
    after cells move to task experiments those copies are stale. Soft-delete
    the whole retired experiment (reversible via the MLflow trash)."""
    for name in _RETIRED_EXPERIMENTS:
        exp = client.get_experiment_by_name(name)
        # get_experiment_by_name also returns already-trashed experiments,
        # and delete_experiment raises on those — only retire active ones
        if exp is not None and exp.lifecycle_stage == "active":
            client.delete_experiment(exp.experiment_id)
            print(f"publish: retired legacy experiment '{name}' "
                  f"(id {exp.experiment_id}) to the MLflow trash")


def _delete_existing(client, exp_id: str, filter_string: str) -> None:
    for r in client.search_runs([exp_id], filter_string=filter_string):
        client.delete_run(r.info.run_id)


def _log_batched(client, run_id: str, metrics=None, params=None) -> None:
    """metrics: (key, value, step) or (key, value, step, ts_ms) tuples —
    real wall-clock timestamps let the UI draw series over actual time."""
    from mlflow.entities import Metric, Param
    now_ms = int(time.time() * 1000)
    ms = [Metric(m[0], float(m[1]),
                 int(m[3]) if len(m) > 3 and m[3] else now_ms, m[2])
          for m in (metrics or [])]
    ps = [Param(k, str(v)[:_PARAM_MAX]) for k, v in (params or [])]
    for i in range(0, len(ps), 90):
        client.log_batch(run_id, params=ps[i:i + 90])
    for i in range(0, len(ms), 900):
        client.log_batch(run_id, metrics=ms[i:i + 900])


def _quantiles(values: list[float]) -> list[tuple[int, float]]:
    vals = sorted(v for v in values if isinstance(v, (int, float)))
    if not vals:
        return []
    out = []
    for p in _PERCENTILES:
        idx = min(len(vals) - 1, max(0, round(p / 100 * (len(vals) - 1))))
        out.append((p, float(vals[idx])))
    return out


def publish_cells(client, task_exp_ids: dict[str, str], cells: list[dict],
                  meta: dict, campaign: str, owner: str, ov_exp_id: str = "",
                  ov_run_id: str = "") -> dict[str, str]:
    """Write one run per cell into the cell's task experiment.
    Returns label -> mlflow run_id."""
    ids: dict[str, str] = {}
    # creation order is the one thing the naked runs table sorts by
    # (Created, newest first) — write worst-first / champion-last so the
    # default table opens champion-on-top even with no stored arrangement
    def _created_order(c: dict) -> tuple:
        rel = c.get("rel") or {}
        return (c["domain"], 1 if rel else 0,
                -(rel.get("rank_in_domain") or 0.0))
    for c in sorted(cells, key=_created_order):
        exp_id = task_exp_ids[c["domain"]]
        # identity is the run itself (cell_key), NOT (cell_key, campaign):
        # a task page holds ONE row per harness run; when a later campaign
        # re-publishes the same run (e.g. the comparison set grew), the new
        # row replaces the old instead of duplicating it
        _delete_existing(
            client, exp_id, f"tags.cell_key = '{c['label']}'")
        run = client.create_run(exp_id, tags={
            "cell_key": c["label"], "campaign": campaign, "owner": owner,
            "mlflow.runName": _run_display_name(c),
        })
        rid = run.info.run_id
        ids[c["label"]] = rid

        # open bag: corpus entry + config + pack/referee meta, all flattened
        bag: dict[str, str] = {}
        bag.update(_flatten({k: v for k, v in c["entry"].items()
                             if not k.endswith("_path")}))
        bag.update(_flatten(c["config"], "config"))
        bag.update(_flatten(c["pack_meta"], "pack"))
        bag.update(_flatten(c["referee_meta"], "referee"))
        bag.update(_flatten(c["best_self"], "best_self"))
        if c.get("seats"):
            bag.update(_flatten(c["seats"], "seat"))
            bag["seats.models"] = ", ".join(c["seat_models"])
            bag["seats.mixed"] = "yes" if len(c["seat_models"]) > 1 else "no"
        bag["campaign"] = campaign
        bag["owner"] = owner

        metrics: list[tuple[str, float, int]] = []
        for k, v in c["det"].items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            metrics.append((f"det.{k}", float(v), 0))
        if c["referee_best"] is not None:
            metrics.append(("referee.best_score", float(c["referee_best"]), 0))
        for k, v in c["referee_meta"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics.append((f"referee.{k}", float(v), 0))
        for k, v in (c.get("rel") or {}).items():
            metrics.append((f"rel.{k}", float(v), 0))

        # series: research progress (step = experiment index) with real
        # wall-clock timestamps (chart x-axis switchable to time)
        finished_by_name = {r.get("name"): r.get("finished_at")
                            for r in c["rollout"] if r.get("finished_at")}
        for t in c["trajectory"]:
            n = t.get("n")
            if not isinstance(n, int):
                continue
            ts = finished_by_name.get(t.get("name"))
            ts_ms = int(ts * 1000) if isinstance(ts, (int, float)) else None
            if isinstance(t.get("value"), (int, float)):
                metrics.append(("det.progress.value", float(t["value"]), n, ts_ms))
            if isinstance(t.get("best_so_far"), (int, float)):
                metrics.append(("det.progress.best_so_far",
                                float(t["best_so_far"]), n, ts_ms))

        # series: the full rollout, every DB row in order (the run itself,
        # not a summary) — step = experiment id, timestamp = finish time
        cum_cost = 0.0
        for r in c["rollout"]:
            eid = r.get("id")
            if not isinstance(eid, int):
                continue
            fin = r.get("finished_at")
            ts_ms = int(fin * 1000) if isinstance(fin, (int, float)) else None
            for key, field in (("duration_seconds", "duration_seconds"),
                               ("est_cost_usd", "est_cost_usd"),
                               ("tokens_attributed", "tokens_attributed"),
                               ("fix_attempts", "fix_attempts"),
                               ("metric_value", "metric")):
                v = r.get(field)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metrics.append((f"det.rollout.{key}", float(v), eid, ts_ms))
            if isinstance(r.get("est_cost_usd"), (int, float)):
                cum_cost += float(r["est_cost_usd"])
                metrics.append(("det.rollout.cum_cost_usd", cum_cost, eid, ts_ms))
            metrics.append(("det.rollout.failed",
                            1.0 if r.get("error") else 0.0, eid, ts_ms))

        # series: whole distributions as quantile curves (step = percentile)
        durations = [row.get("duration_seconds") for row in c["experiment_table"]]
        for p, v in _quantiles(durations):
            metrics.append(("det.q.experiment_duration_seconds", v, p))
        for name, samples in c["code_samples"].items():
            if isinstance(samples, list):
                for p, v in _quantiles(samples):
                    metrics.append((f"det.q.code.{name}", v, p))
        for name, samples in c["agent_samples"].items():
            for p, v in _quantiles(samples):
                metrics.append((f"det.q.agent.{name}", v, p))

        # the sectioned names ARE the chart layout (see _sectioned above)
        metrics = [(_sectioned(m[0]), *m[1:]) for m in metrics]
        # "0 seats" leads every chart page (digit sections sort first): one
        # chart per role×model, the title itself says who sat there, the
        # value is how many logged requests that seat issued. Seat swaps are
        # a routine benchmark variable — this is lineup data, not an alarm.
        for role, mm in sorted((c.get("seat_counts") or {}).items()):
            for mdl, n in sorted(mm.items()):
                metrics.append((f"0 seats/{role}: {mdl}", float(n), 0))
        _log_batched(client, rid, metrics=metrics, params=sorted(bag.items()))

        with tempfile.TemporaryDirectory() as td:
            prov = {
                "cell": c["label"], "campaign": campaign,
                "sources": meta["sources"],
                "bench_version": meta.get("bench_version"),
                "rules_version": meta.get("rules_version"),
                "bench_generated_at": meta.get("generated_at"),
                "published_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "run_dir": c["entry"].get("run_dir"),
                "workspace": c["entry"].get("workspace"),
            }
            pth = Path(td) / "provenance.json"
            pth.write_text(json.dumps(prov, indent=2))
            client.log_artifact(rid, str(pth))
            try:
                rp = Path(td) / f"rollout_{c['short']}.png"
                if render_rollout(c, rp):
                    client.log_artifact(rid, str(rp), artifact_path="rollout")
            except Exception as exc:  # chart failure must be loud, not fatal
                print(f"publish: rollout chart FAILED for {c['label']}: {exc}")
        client.set_tag(rid, "mlflow.note.content",
                       _note_for_cell(c, campaign, ov_exp_id, ov_run_id))
        client.set_terminated(run.info.run_id)
    return ids


def render_rollout(cell: dict, out_path: Path) -> bool:
    """One cell's complete rollout: phases shaded, every experiment a bar
    on the real wall-clock, best-so-far unfolding underneath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in cell["rollout"]
            if isinstance(r.get("started_at"), (int, float))
            and isinstance(r.get("finished_at"), (int, float))]
    phases = {k: v for k, v in cell["phase_windows"].items()
              if isinstance(v, dict) and isinstance(v.get("start"), (int, float))}
    anchors = ([v["start"] for v in phases.values()]
               + [r["started_at"] for r in rows])
    if not anchors or not rows:
        return False
    t0 = min(anchors)

    def _h(ts):
        return (ts - t0) / 3600.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 3.2 + 0.22 * len(rows)), sharex=True,
        gridspec_kw={"height_ratios": [max(3, len(rows) // 4), 2]})
    for pi, (name, w) in enumerate(sorted(phases.items(),
                                          key=lambda kv: kv[1]["start"])):
        for ax in (ax1, ax2):
            ax.axvspan(_h(w["start"]), _h(w.get("end", w["start"])),
                       color="#000000", alpha=0.05)
        # stagger the labels: early phases are seconds wide and would smear
        ax1.text(_h(w["start"]), len(rows) + 0.4 + 0.9 * (pi % 3), name,
                 fontsize=7, va="bottom", color="#555555")
    for i, r in enumerate(sorted(rows, key=lambda r: r["started_at"])):
        failed = bool(r.get("error"))
        color = "#c62828" if failed else "#2e7d32"
        ax1.barh(i, max(_h(r["finished_at"]) - _h(r["started_at"]), 0.01),
                 left=_h(r["started_at"]), height=0.72, color=color,
                 alpha=0.85, edgecolor="black" if r.get("fix_attempts") else color,
                 linewidth=1.2 if r.get("fix_attempts") else 0.4)
        ax1.text(_h(r["finished_at"]) + 0.03, i, str(r.get("name"))[:34],
                 fontsize=6, va="center", color="#333333")
    ax1.set_ylim(-1, len(rows) + 3.6)
    ax1.set_yticks([])
    ax1.set_title(
        f"{cell['short']} — full rollout ({cell['framework']}, "
        f"{cell['config'].get('model') or '?'}, {cell['domain']}); "
        "green = ran clean, red = errored, black edge = needed fix attempts",
        fontsize=9)

    finished_by_name = {r.get("name"): r.get("finished_at") for r in rows}
    pts = [( _h(finished_by_name[t["name"]]), t["best_so_far"])
           for t in cell["trajectory"]
           if t.get("name") in finished_by_name
           and isinstance(t.get("best_so_far"), (int, float))]
    pts.sort()
    if pts:
        ax2.step([x for x, _ in pts], [y for _, y in pts], where="post",
                 lw=2, color="#1a237e", marker="o", markersize=3)
    metric = cell["pack_meta"].get("metric_key") or "metric"
    ax2.set_ylabel(f"best {metric} "
                   f"({'lower' if cell['lower_is_better'] else 'higher'} is better)",
                   fontsize=8)
    ax2.set_xlabel("hours since run start", fontsize=9)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
# lens charts (pre-rendered; matrix-shaped stories mlflow can't draw natively)

def _colors_markers(cells: list[dict]):
    fws = sorted({c["framework"] for c in cells})
    extra = [f for f in fws if f not in _FRAMEWORK_COLORS]
    cmap = dict(_FRAMEWORK_COLORS)
    for i, f in enumerate(extra):
        cmap[f] = _EXTRA_COLORS[i % len(_EXTRA_COLORS)]
    models = sorted({(c["config"].get("model") or "?") for c in cells})
    mmap = {m: _MODEL_MARKERS[i % len(_MODEL_MARKERS)] for i, m in enumerate(models)}
    return cmap, mmap


def _fmt(v: float) -> str:
    return f"{v:.6g}" if isinstance(v, (int, float)) else "—"


def render_charts(cells: list[dict], meta: dict, out_dir: Path) -> list[Path]:
    """Only the grid pictures MLflow has no native chart type for: the
    campaign coverage map and the referee win matrix. Everything else
    (leaderboards, races, cost-vs-quality, quantile curves) lives in the
    stored native views."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cmap, _ = _colors_markers(cells)
    domains = sorted({c["domain"] for c in cells})
    paths: list[Path] = []

    def _col(c):
        return cmap.get(c["framework"], "#333333")

    # 1. coverage map: every cell, referee best or absence, exit status
    fig, ax = plt.subplots(figsize=(1.9 * max(4, len(cells) // max(1, len(domains)) + 2),
                                    0.9 * len(domains) + 1.6))
    combos = sorted({(c["framework"], c["config"].get("model") or "?") for c in cells})
    for yi, dom in enumerate(domains):
        for xi, (fw, model) in enumerate(combos):
            sub = [c for c in cells if c["domain"] == dom
                   and c["framework"] == fw and (c["config"].get("model") or "?") == model]
            if not sub:
                ax.add_patch(plt.Rectangle((xi, yi), 0.94, 0.94, color="#f2f2f2"))
                continue
            c = sub[0]
            has_ref = c["referee_best"] is not None
            exit_code = c["det"].get("lifecycle.final_exit_code")
            comp = str(c["entry"].get("completeness") or "complete")
            bad = comp != "complete" or exit_code not in (0, None)
            face = _col(c) if has_ref else "#bdbdbd"
            alpha = 0.25 + 0.75 * (1.0 / float((c.get("rel") or {}).get("rank_in_domain", 4)))
            ax.add_patch(plt.Rectangle((xi, yi), 0.94, 0.94, color=face, alpha=alpha,
                                       ec="#c62828" if bad else "white",
                                       lw=3 if bad else 1))
            txt = _fmt(c["referee_best"]) if has_ref else "no referee"
            rk = (c.get("rel") or {}).get("rank_in_domain")
            if rk:
                txt += f"\n#{int(rk)} of {int(c['rel']['n_ranked'])}"
            if comp != "complete":
                txt += f"\n[{comp}]"
            elif exit_code not in (0, None):
                txt += f"\n[exit {int(exit_code)}]"
            ax.text(xi + 0.47, yi + 0.47, txt, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, len(combos)); ax.set_ylim(0, len(domains))
    ax.set_xticks([i + 0.47 for i in range(len(combos))])
    ax.set_xticklabels([f"{f}\n{m}" for f, m in combos], fontsize=8)
    ax.set_yticks([i + 0.47 for i in range(len(domains))])
    ax.set_yticklabels(domains, fontsize=9)
    ax.set_title("Coverage map — referee best score per cell (color = harness, "
                 "opacity = rank, red edge = incomplete/failed record, "
                 "grey = not refereed)", fontsize=10)
    ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)
    p = out_dir / "01_coverage_map.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 6. win matrix from referee pair verdicts
    pairs = [p_ for p_ in (meta.get("pairs") or []) if p_.get("comparable")]
    if pairs:
        fig, ax = plt.subplots(figsize=(9, 0.75 * len(pairs) + 1.5))
        for yi, pr in enumerate(pairs):
            for side, xi in (("left", 0), ("right", 1)):
                lab = _short(pr.get(side) or "")
                best = ((pr.get("best") or {}).get(side) or {}).get("referee_score")
                won = pr.get("winner") == side
                fw = (pr.get(side) or "").split("/")[2] if (pr.get(side) or "").count("/") >= 2 else ""
                face = cmap.get(fw, "#888888")
                ax.add_patch(plt.Rectangle((xi * 1.05, yi), 1.0, 0.9,
                                           color=face, alpha=0.9 if won else 0.25))
                ax.text(xi * 1.05 + 0.5, yi + 0.45,
                        f"{lab}\n{_fmt(best)}" + ("  ← winner" if won else ""),
                        ha="center", va="center", fontsize=8)
            ax.text(2.2, yi + 0.45,
                    f"{pr.get('pair')}  ({pr.get('metric')}, "
                    f"{'lower' if pr.get('lower_is_better') else 'higher'} wins)",
                    va="center", fontsize=8)
        ax.set_xlim(0, 3.4); ax.set_ylim(0, len(pairs)); ax.invert_yaxis()
        ax.axis("off")
        ax.set_title("Referee pair verdicts — declared winners (solid = won)",
                     fontsize=10)
        p = out_dir / "06_win_matrix.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    return paths


# --------------------------------------------------------------------------
# overview run (charts + reports + inv.* counters)

def _report_dirs(out_dir: Path) -> list[Path]:
    return sorted(d for d in out_dir.iterdir()
                  if d.is_dir() and (d / "verification.json").exists())


def create_overview_run(client, exp_id: str, campaign: str, owner: str) -> str:
    """Create the campaign's overview run up front so cell notes can link
    to it; publish_overview fills it in after the cells are written."""
    _delete_existing(client, exp_id, f"tags.campaign = '{campaign}'")
    run = client.create_run(exp_id, tags={
        "campaign": campaign, "owner": owner, "mlflow.runName": campaign})
    return run.info.run_id


def publish_overview(client, rid: str, cells: list[dict], meta: dict,
                     campaign: str, out_dir: Path) -> str:
    metrics: list[tuple[str, float, int]] = []
    params: list[tuple[str, str]] = [
        ("campaign", campaign), ("cells", str(len(cells))),
        ("bench_version", str(meta.get("bench_version"))),
        ("bench_generated_at", str(meta.get("generated_at"))),
        ("domains", ", ".join(sorted({c["domain"] for c in cells}))),
        ("harnesses", ", ".join(sorted({c["framework"] for c in cells}))),
        ("models", ", ".join(sorted({(c["config"].get("model") or "?")
                                     for c in cells}))),
    ]
    for rd in _report_dirs(out_dir):
        try:
            ver = _load(rd / "verification.json")
        except (OSError, json.JSONDecodeError):
            continue
        for k in ("total", "verified", "partial", "failed"):
            v = ver.get(k)
            if isinstance(v, (int, float)):
                metrics.append((_sectioned(f"inv.{rd.name}.findings_{k}"),
                                float(v), 0))
    _log_batched(client, rid, metrics=metrics, params=params)

    with tempfile.TemporaryDirectory() as td:
        for cp in render_charts(cells, meta, Path(td) / "charts"):
            client.log_artifact(rid, str(cp), artifact_path="charts")
    for rd in _report_dirs(out_dir):
        for name in ("REPORT.html", "REPORT.md", "findings.md"):
            f = rd / name
            if f.exists():
                client.log_artifact(rid, str(f), artifact_path=f"reports/{rd.name}")
    for name in ("bench.md", "tables.md", "META_REVIEW.html",
                 "QUALITY_LOOP_SUMMARY.html", "token_accounting.md"):
        f = out_dir / name
        if f.exists():
            client.log_artifact(rid, str(f), artifact_path="campaign")
    client.set_tag(rid, "mlflow.note.content",
                   _note_for_overview(campaign, cells))
    client.set_terminated(rid)
    return rid


# --------------------------------------------------------------------------
# CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="runcmp publish",
        description="Publish a finished runcmp chain into the showcase MLflow store.")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--packs", required=True)
    ap.add_argument("--bench", required=True, help="bench.json from the bench stage")
    ap.add_argument("--referee", required=True)
    ap.add_argument("--out", required=True,
                    help="campaign output dir (report subdirs discovered here)")
    ap.add_argument("--store", default="",
                    help="showcase store dir (sqlite db + artifacts inside)")
    ap.add_argument("--tracking-uri", default="",
                    help="explicit tracking URI; overrides --store")
    ap.add_argument("--campaign", default="",
                    help="campaign name (default: basename of --out)")
    ap.add_argument("--gate", type=Path, default=None,
                    help="gate.json from `runcmp gate`: also publish the "
                         "change-gate evidence run (verdict in the name, "
                         "per-cell deltas as charts)")
    ap.add_argument("--retire-campaign", action="append", default=[],
                    metavar="NAME",
                    help="after publishing, delete remaining task-page rows "
                         "and the campaign report of this superseded "
                         "campaign (repeatable). Use when the comparison "
                         "set grew and this publish replaces an older one "
                         "whose runs were re-indexed under new labels.")
    ap.add_argument("--owner", default=os.environ.get("USER", ""))
    args = ap.parse_args(argv)

    if not args.tracking_uri and not args.store:
        ap.error("one of --store or --tracking-uri is required")
    store = Path(args.store).resolve() if args.store else None
    if store:
        store.mkdir(parents=True, exist_ok=True)
        (store / "artifacts").mkdir(exist_ok=True)
    uri = args.tracking_uri or f"sqlite:///{store}/mlflow.db"
    artifact_root = str(store / "artifacts") if store else "mlflow-artifacts:/"
    campaign = args.campaign or Path(args.out).resolve().name

    cells, meta = build_cells(args.corpus, args.packs, args.bench, args.referee)
    if not cells:
        print("publish: corpus has no runs — nothing to do")
        return 1

    client = _client(uri)
    ov_id = _ensure_experiment(client, REPORTS_EXPERIMENT, artifact_root,
                               REPORTS_NOTE, legacy_names=_REPORTS_LEGACY)
    client.set_experiment_tag(ov_id, "mlflow.note.content", REPORTS_NOTE)

    # one experiment per benchmark task; the home page lists them by name
    doms = _ranked(cells)
    task_ids: dict[str, str] = {}
    for dom in doms:
        group = [c for c in cells if c["domain"] == dom]
        metric = next((c["pack_meta"].get("metric_key") for c in group
                       if c["pack_meta"].get("metric_key")), "metric")
        task_ids[dom] = _ensure_task_experiment(
            client, dom, metric, group[0]["lower_is_better"], artifact_root)

    ov_rid = create_overview_run(client, ov_id, campaign, args.owner)
    ids = publish_cells(client, task_ids, cells, meta, campaign, args.owner,
                        ov_exp_id=ov_id, ov_run_id=ov_rid)
    publish_overview(client, ov_rid, cells, meta, campaign, Path(args.out))
    fo_id = publish_faceoff(client, cells, campaign, args.owner,
                            artifact_root)
    print(f"publish: harness face-off: "
          f"#/experiments/{fo_id}/runs?compareRunsMode=CHART")
    if args.gate:
        gate_doc = json.loads(args.gate.read_text())
        g_exp, g_rid = publish_gate(client, gate_doc, campaign, args.owner,
                                    artifact_root,
                                    gate_dir=args.gate.parent)
        print(f"publish: change gate {gate_doc.get('verdict')}: "
              f"#/experiments/{g_exp}/runs/{g_rid}")

    # Retirement touches ONLY superseded task-page rows (duplicates of runs
    # this publish re-published under new labels). Campaign REPORT pages are
    # never touched: they carry investigator reports and bench tables that
    # nothing supersedes (2026-08-06: retiring them hid three finished reports
    # from the viewer — standing order: never delete). Even for
    # task rows this is MLflow's recoverable mark-deleted, not file removal.
    for old in args.retire_campaign:
        n = 0
        for exp_id in task_ids.values():
            for r in client.search_runs(
                    [exp_id], filter_string=f"tags.campaign = '{old}'"):
                client.delete_run(r.info.run_id)
                n += 1
        print(f"publish: retired campaign {old}: {n} superseded task row(s) "
              "marked deleted (recoverable); report pages untouched")

    print(f"publish: campaign={campaign} cells={len(ids)} "
          f"tasks={len(task_ids)} overview_run={ov_rid} store={uri}")
    for dom in sorted(task_ids):
        group = [c for c in cells if c["domain"] == dom]
        metric = next((c["pack_meta"].get("metric_key") for c in group
                       if c["pack_meta"].get("metric_key")), "metric")
        lower = group[0]["lower_is_better"]
        href = store_standings(client, task_ids[dom], lower)
        client.set_experiment_tag(
            task_ids[dom], "mlflow.note.content",
            _task_note(metric, lower, group, campaign, href))
        print(f"publish: task {dom}: {href}")
    _retire_legacy_experiments(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
