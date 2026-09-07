"""Referee re-scoring: one evaluator over preserved prediction artifacts.

The frameworks disagree on validation protocols, so self-reported metrics are
not cross-comparable. Where runs preserve dense predictions with a shared
ground truth, the referee (a) collects a per-pair *truth pool* keyed by
evaluation origin from every artifact that stores truth, verifying that every
duplicated origin is numerically identical across files and runs; (b) scores
every forecast — including forecast-only artifacts, using pool truth — under a
single explicit metric; and (c) compares runs on the maximal origin set their
experiments share. "Model quality is not comparable" then becomes a measured
number wherever artifacts allow, and an explicit coverage statement where not.

The protocol is fixed; per-domain artifact locations/keys are data
(``DOMAIN_SPECS``), so new domains or frameworks only add a spec entry.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import math
import sys
from pathlib import Path

from alpha_lab.benchmarks.runcmp.corpus import load_registry
from alpha_lab.benchmarks.runcmp.lineup import referee_specs as _lineup_referee_specs

# Minimum forecast origins two runs must share before a cross-run rescore
# is allowed. One origin is not one data point: in the traffic domain it is
# 862 series x 24 horizons = 20,688 scored values, so 5 origins already
# means >100k paired predictions. The former value of 10 blocked genuine
# comparisons (runs that chose 7-origin held-out splits) without adding
# statistical protection.
MIN_SHARED_ORIGINS = 5
IDENT_REL_TOL = 0.05  # self-report identification tolerance
TRUTH_ATOL = 1e-6

DOMAIN_SPECS: dict[str, dict] = {
    # kind="returns_parquet": strategy-returns domains. No shared truth exists
    # across runs (each strategy owns its return stream), so the referee's
    # product is per-run SELF-REPORT VERIFICATION: recompute the metric from
    # the preserved daily series and compare with the run's own DB claim.
    # Cross-run model-quality comparison stays "not comparable" by design.
    "ibes": {
        "kind": "returns_parquet",
        "series_globs": ["experiments/*/results/daily_returns.parquet",
                         "experiments/*/results/primary/daily_returns.parquet"],
        "metric": "sharpe",
        "return_columns": ("net_return", "market_adjusted_net", "gross_return"),
        "annualization": 252,
        "lower_is_better": False,
    },
    # kind="bpb_curve": LLM-pretraining domains reporting validation
    # bits-per-byte. No two runs share a validation identity (each picks its
    # own held-out slice), so like the returns domains the product is
    # per-run SELF-REPORT VERIFICATION — but a real recomputation, not a
    # copy check: bits-per-byte is rederived from the preserved summed
    # negative log-likelihood and byte count (bpb = nll_nats / (bytes * ln2))
    # and compared with the value the run reported alongside it.
    "domain2": {
        "kind": "bpb_curve",
        "curve_globs": ["experiments/*/results/training_curves.json",
                        "experiments/*/results/curves.json",
                        "experiments/*/results/metrics_detailed.json",
                        # Runs that record the validation figures directly in
                        # the metrics file rather than a separate curve file.
                        "experiments/*/results/metrics.json"],
        "record_keys": ("val_curve", "validation", "val"),
        # Runs report either the final validation figure or the best one
        # reached; both spellings appear across these frameworks and a run
        # using the "best_" form must not silently fail to re-score.
        "metric_keys": ("val_bpb", "bpb", "best_val_bpb"),
        "nll_keys": ("summed_masked_nll", "nll_nats", "val_loss_nats_sum",
                     "summed_nll", "best_summed_nll"),
        # Second recomputation path: runs that store a per-token loss in
        # nats plus a bytes-per-token ratio instead of a summed likelihood.
        # bpb = loss_nats / (ln2 * bytes_per_token) — same identity, the
        # other way round. Without this, runs using this (equally valid)
        # convention look like they preserved nothing.
        # Validation losses only: a training loss does not reproduce a
        # validation bpb, and accepting it manufactures false mismatches.
        "loss_keys": ("val_loss", "val_loss_nats", "val_nll",
                      "best_val_loss_nats", "best_val_loss"),
        "bpt_keys": ("bytes_per_token_val", "bytes_per_token"),
        # Bytes-per-token is derivable when a run preserves the validation
        # byte and token counts but not their ratio; without this, runs that
        # store val_bytes + val_tokens (the contract's own required fields)
        # count as non-recomputable purely for omitting a redundant field.
        "tokens_keys": ("val_tokens", "validation_tokens", "n_val_tokens"),
        # Some runs emit metrics only as JSON lines in a training log.
        "log_globs": ["experiments/*/results/train.log",
                      "experiments/*/results/*train*.log"],
        "log_prefix": "METRIC ",
        "bytes_keys": ("raw_utf8_bytes", "utf8_bytes", "validation_raw_bytes",
                       "val_bytes", "bytes"),
        "metric": "val_bpb",
        "lower_is_better": True,
    },
    "domain4": {
        "pred_globs": [
            # Canonical contract file first (adapters require it); the
            # looser patterns keep older runs scoreable.
            "experiments/*/results/referee_predictions.npz",
            "experiments/*/results/*predictions*.npz",
            "experiments/*/results/*forecasts_by_origin*.npz",
        ],
        "truth_keys": ("truth", "targets"),
        "origin_key": "origins",
        "metric": "rmse",
        "lower_is_better": True,
    },
}

# Lineup-declared domains (kind="classification_table" / "kernel_bench" / any
# existing kind) merge in from lineup.json; builtin entries above stay
# authoritative for their ids.
DOMAIN_SPECS.update(
    {k: v for k, v in _lineup_referee_specs().items() if k not in DOMAIN_SPECS})


def _annualized_sharpe(returns, periods: int) -> float | None:
    import numpy as np

    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    if len(r) < 20 or r.std(ddof=1) == 0:
        return None
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def returns_verification(workspace, spec: dict, exps: dict, scored: int) -> dict:
    """Per-run self-report verification for returns-series domains.

    For every experiment with a preserved daily series, recompute the metric
    from each candidate return column; the self-report is reproduced when any
    column lands within 5% relative of the DB-recorded value (mirroring the
    forecast referee's "within 5% on some stored array" contract).
    """
    import pandas as pd

    rows = []
    reproduced = 0
    seen: set[str] = set()
    for pattern in spec["series_globs"]:
        for f in sorted(globmod.glob(str(workspace / pattern))):
            exp = f.split("/experiments/")[1].split("/")[0]
            if exp in seen:  # top-level series wins over primary/ fallback
                continue
            seen.add(exp)
            claimed = (exps.get(exp) or {}).get("metric")
            row = {"experiment": exp, "path": f, "status": "scored",
                   "claimed": claimed, "recomputed": None,
                   "column": None, "self_report_reproduced": False}
            try:
                df = pd.read_parquet(f)
            except Exception as exc:  # noqa: BLE001 — one bad file must not kill the referee
                row["status"] = f"unreadable: {type(exc).__name__}"
                rows.append(row)
                continue
            best = None
            for col in spec["return_columns"]:
                if col not in df.columns:
                    continue
                val = _annualized_sharpe(df[col], spec["annualization"])
                if val is None:
                    continue
                if best is None or (
                    claimed is not None
                    and abs(val - claimed) < abs(best[1] - claimed)
                ):
                    best = (col, val)
            if best is not None:
                row["column"], row["recomputed"] = best[0], round(best[1], 4)
                if claimed:
                    row["self_report_reproduced"] = (
                        abs(best[1] - claimed) <= 0.05 * abs(claimed))
            reproduced += bool(row["self_report_reproduced"])
            rows.append(row)
    # Coverage denominator: experiments that CLAIM a metric — those are the
    # self-reports the series can verify. Artifacts for unscored/gated
    # experiments are counted separately.
    claims = {n for n, e in exps.items() if e.get("metric") is not None}
    claims_with_series = {r["experiment"] for r in rows
                          if r["experiment"] in claims
                          and r["recomputed"] is not None}
    best_claim = None
    if claims:
        best_name = max(claims, key=lambda n: exps[n]["metric"])
        best_row = next((r for r in rows if r["experiment"] == best_name), None)
        best_claim = {
            "experiment": best_name,
            "claimed": exps[best_name]["metric"],
            "recomputed": best_row["recomputed"] if best_row else None,
            "reproduced": bool(best_row and best_row["self_report_reproduced"]),
            "series_preserved": best_row is not None,
        }
    return {
        "scored_experiments": scored,
        "claims": len(claims),
        "experiments_with_predictions": len(rows),
        "claims_with_series": len(claims_with_series),
        "artifact_coverage": (round(len(claims_with_series) / len(claims), 3)
                              if claims else 0.0),
        "self_report_reproduced": reproduced,
        "best_claim": best_claim,
        "experiments": rows,
    }


def bpb_verification(workspace, spec: dict, exps: dict, scored: int) -> dict:
    """Recompute validation bits-per-byte from preserved curve records.

    For every experiment whose curves store a summed negative
    log-likelihood and a byte count, bpb is rederived independently
    (``nll / (bytes * ln 2)``) and compared with the bpb the run reported in
    the same record. Runs that store only the final number can be checked
    for internal consistency but not rederived; both outcomes are reported.
    """
    import math as _math

    rows = []
    reproduced = 0
    seen: set[str] = set()
    for pattern in spec["curve_globs"]:
        for f in sorted(globmod.glob(str(workspace / pattern))):
            exp = f.split("/experiments/")[1].split("/")[0]
            if exp in seen:
                continue
            try:
                doc = json.loads(Path(f).read_text())
            except (OSError, ValueError) as exc:
                rows.append({"experiment": exp, "path": f,
                             "status": f"unreadable: {type(exc).__name__}"})
                seen.add(exp)
                continue
            records = []
            if isinstance(doc, dict):
                for key in spec["record_keys"]:
                    val = doc.get(key)
                    if isinstance(val, list):
                        records = [r for r in val if isinstance(r, dict)]
                        break
                if not records:
                    # Flat metrics file: treat the document itself as one
                    # record, merging any per-run constants (e.g. the
                    # bytes-per-token ratio) into it.
                    if any(k in doc for k in spec["metric_keys"]):
                        records = [doc]
                    elif isinstance(doc.get("history"), list):
                        hist = [r for r in doc["history"] if isinstance(r, dict)]
                        consts = {k: v for k, v in doc.items()
                                  if isinstance(v, (int, float))}
                        records = [{**consts, **r} for r in hist]
            if not records:
                continue
            seen.add(exp)
            best = None  # (claimed, recomputed) at the lowest claimed bpb
            checked = 0
            for rec in records:
                claimed = next((rec[k] for k in spec["metric_keys"]
                                if isinstance(rec.get(k), (int, float))), None)
                nll = next((rec[k] for k in spec["nll_keys"]
                            if isinstance(rec.get(k), (int, float))), None)
                nbytes = next((rec[k] for k in spec["bytes_keys"]
                               if isinstance(rec.get(k), (int, float))), None)
                loss = next((rec[k] for k in spec.get("loss_keys", ())
                             if isinstance(rec.get(k), (int, float))), None)
                bpt = next((rec[k] for k in spec.get("bpt_keys", ())
                            if isinstance(rec.get(k), (int, float))), None)
                ntok = next((rec[k] for k in spec.get("tokens_keys", ())
                             if isinstance(rec.get(k), (int, float))), None)
                if bpt is None and nbytes and ntok:
                    bpt = nbytes / ntok
                # One spelling, two units across harnesses: val_loss_nats
                # is a per-token MEAN (2.42) in one corpus and a SUM over
                # all evaluated tokens (593,371) in another. Classify by
                # magnitude against the token count — never against the
                # claim, so verification stays non-circular: a mean sits in
                # per-token loss range, a sum is ~mean × tokens.
                if (loss is not None and nll is None and ntok
                        and not (0.01 <= loss <= 50)
                        and 0.01 <= loss / ntok <= 50):
                    nll, loss = loss, None
                if claimed is None:
                    continue
                if nll and nbytes:
                    recomputed = nll / (nbytes * _math.log(2))
                elif loss and bpt:
                    recomputed = loss / (bpt * _math.log(2))
                else:
                    continue
                checked += 1
                if best is None or claimed < best[0]:
                    best = (claimed, recomputed)
            if best is None:
                rows.append({"experiment": exp, "path": f,
                             "status": "no_recomputable_record",
                             "records": len(records)})
                continue
            claimed, recomputed = best
            ok = _math.isclose(recomputed, claimed, rel_tol=IDENT_REL_TOL)
            reproduced += ok
            rows.append({
                "experiment": exp, "path": f, "status": "scored",
                "records_recomputed": checked,
                "claimed": round(claimed, 6),
                "recomputed": round(recomputed, 6),
                "self_report_reproduced": bool(ok),
            })
    # Training logs that emit one JSON object per line.
    for pattern in spec.get("log_globs", ()):
        for f in sorted(globmod.glob(str(workspace / pattern))):
            exp = f.split("/experiments/")[1].split("/")[0]
            scored_already = any(r["experiment"] == exp and r["status"] == "scored"
                                 for r in rows)
            if scored_already:
                continue
            best = None
            checked = 0
            prefix = spec.get("log_prefix", "")
            for line in Path(f).read_text(errors="replace").splitlines():
                if prefix and not line.startswith(prefix):
                    continue
                try:
                    rec = json.loads(line[len(prefix):])
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                claimed = next((rec[k] for k in spec["metric_keys"]
                                if isinstance(rec.get(k), (int, float))), None)
                loss = next((rec[k] for k in spec.get("loss_keys", ())
                             if isinstance(rec.get(k), (int, float))), None)
                bpt = next((rec[k] for k in spec.get("bpt_keys", ())
                            if isinstance(rec.get(k), (int, float))), None)
                if bpt is None and claimed and loss:
                    # Ratio not logged per line: derive it once from the
                    # first consistent pair, then reuse.
                    bpt = loss / (claimed * _math.log(2)) if claimed else None
                if claimed is None or not loss or not bpt:
                    continue
                checked += 1
                recomputed = loss / (bpt * _math.log(2))
                if best is None or claimed < best[0]:
                    best = (claimed, recomputed)
            if best is None:
                continue
            seen.add(exp)
            # Drop any earlier non-scored row for this experiment: the log
            # is a better source than a metrics file that lacked the loss.
            rows[:] = [r for r in rows if r["experiment"] != exp]
            claimed, recomputed = best
            ok = _math.isclose(recomputed, claimed, rel_tol=IDENT_REL_TOL)
            reproduced += ok
            rows.append({"experiment": exp, "path": f, "status": "scored",
                         "records_recomputed": checked,
                         "claimed": round(claimed, 6),
                         "recomputed": round(recomputed, 6),
                         "self_report_reproduced": bool(ok)})

    claims = {n for n, e in exps.items() if e.get("metric") is not None}
    with_curves = {r["experiment"] for r in rows if r["status"] == "scored"}
    return {
        "scored_experiments": scored,
        "claims": len(claims),
        "experiments_with_predictions": len(rows),
        "claims_with_series": len(claims & with_curves),
        "artifact_coverage": (round(len(claims & with_curves) / len(claims), 3)
                              if claims else 0.0),
        "self_report_reproduced": reproduced,
        "experiments": rows,
    }


def _rmse(pred, truth) -> float:
    import numpy as np

    return float(
        np.sqrt(np.mean((pred.astype("float64") - truth.astype("float64")) ** 2))
    )


class RunArtifacts:
    """All usable prediction artifacts of one run, plus its truth slices."""

    def __init__(self, workspace: Path, spec: dict):
        self.spec = spec
        self.items: list[dict] = []  # {experiment, path, origins, truth, candidates}
        per_exp: dict[str, set[Path]] = {}
        for pattern in spec["pred_globs"]:
            for f in sorted(globmod.glob(str(workspace / pattern))):
                # Smoke artifacts live in a smoke directory OR carry the
                # word in the file name (smoke_referee_predictions.npz).
                # They come from a shortened trial run, so scoring one as if
                # it were the real evidence silently reports the wrong number.
                base = os.path.basename(f).lower()
                if "/smoke/" in f or "/smoke_results/" in f or "smoke" in base:
                    continue
                exp = f.split("/experiments/")[1].split("/")[0]
                per_exp.setdefault(exp, set()).add(Path(f))
        for exp, paths in sorted(per_exp.items()):
            # Several artifact files can coexist per experiment (e.g. a full
            # evaluation grid next to a smaller calibration slice). Pick the
            # parseable one covering the most origins; newest mtime breaks
            # ties, and the NAME breaks mtime ties — candidates are held in
            # a set, and two files written in the same instant (one a copy
            # of the other, observed 2026-08-06 to the microsecond) made the
            # winner depend on set iteration order across processes. A
            # replay must reproduce byte-identical provenance.
            best: tuple[Path, dict] | None = None
            for p in sorted(paths,
                            key=lambda q: (-q.stat().st_mtime, q.name)):
                parsed = self._parse(p)
                if parsed is None:
                    continue
                if best is None or len(parsed["origins"]) > len(best[1]["origins"]):
                    best = (p, parsed)
            if best is not None:
                self.items.append({"experiment": exp, "path": best[0], **best[1]})
        # Experiments with no usable .npz: try loose .npy arrays.
        covered = {it["experiment"] for it in self.items}
        for results_dir in sorted(workspace.glob("experiments/*/results")):
            exp = str(results_dir).split("/experiments/")[1].split("/")[0]
            if exp in covered:
                continue
            parsed = self._parse_npy_dir(results_dir)
            if parsed is not None:
                self.items.append({"experiment": exp, "path": results_dir,
                                   **parsed})

    def _parse(self, path: Path) -> dict | None:
        import numpy as np

        try:
            z = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return None
        if self.spec["origin_key"] not in z:
            return None
        origins = z[self.spec["origin_key"]]
        n_origins = len(origins)

        # Frameworks label the same evaluation origins differently: some
        # write the integer index of the first forecast hour (14036), others
        # a wall-clock timestamp (2016-08-07T20). Comparison casts origins to
        # int, so a timestamp becomes a huge epoch number that matches no
        # index and the run silently pairs with nobody -- verified on runs
        # whose truth arrays are byte-identical. Non-integer labels are
        # dropped here so the truth-content recovery below assigns the same
        # integer origins the other runs use, instead of guessing a mapping.
        if origins is not None and getattr(origins, "dtype", None) is not None \
                and origins.dtype.kind not in ("i", "u"):
            origins = None
            n_origins = len(z[self.spec["truth_keys"][0]]) \
                if self.spec["truth_keys"][0] in z.files else n_origins

        def to_origin_major(arr):
            # Frameworks disagree on axis order (origin-major vs
            # sensor-major); normalize by moving the unique axis whose
            # length matches the origin count to the front. Axis 0 wins
            # when several match, which keeps origin-major files as-is.
            axes = [i for i, s in enumerate(arr.shape) if s == n_origins]
            if not axes:
                return None
            axis = 0 if 0 in axes else axes[0]
            return np.moveaxis(arr, axis, 0)

        truth = None
        for key in self.spec["truth_keys"]:
            if key in z:
                truth = to_origin_major(z[key])
                break
        candidates = {}
        shape = truth.shape if truth is not None else None
        for key in z.keys():
            if key in self.spec["truth_keys"] or key == self.spec["origin_key"]:
                continue
            arr = z[key]
            if arr.dtype.kind != "f" or arr.ndim < 2:
                continue
            arr = to_origin_major(arr)
            if arr is None:
                continue
            if shape is None:
                shape = arr.shape
            if arr.shape == shape:
                candidates[key] = arr
        if not candidates:
            return None
        return {"origins": origins, "truth": truth, "candidates": candidates}

    # -- loose .npy fallback -------------------------------------------
    def _parse_npy_dir(self, results_dir: Path) -> dict | None:
        """Assemble one artifact set from loose ``.npy`` files.

        Some runs preserve predictions as separate arrays
        (``preds_testsplit.npy`` + ``truth_testsplit.npy``) instead of a
        single ``.npz``. Those are just as rescoreable, but carry no origin
        index — origins are recovered later by matching truth content
        against the origin-bearing runs' pool (``recover_origins``), so no
        alignment is ever guessed.
        """
        import numpy as np

        arrays: dict[str, object] = {}
        for f in sorted(results_dir.glob("*.npy")):
            try:
                a = np.load(f, allow_pickle=False)
            except (OSError, ValueError):
                continue
            if a.dtype.kind == "f" and a.ndim >= 2:
                arrays[f.stem] = a
        if not arrays:
            return None
        truth_key = None
        for stem in sorted(arrays):
            low = stem.lower()
            if any(t in low for t in ("truth", "target", "actual")):
                # Prefer the held-out split over train-tail diagnostics.
                if truth_key is None or "test" in low:
                    truth_key = stem
        if truth_key is None:
            return None
        truth = arrays[truth_key]
        candidates = {
            k: v for k, v in arrays.items()
            if k != truth_key and v.shape == truth.shape
            and any(p in k.lower() for p in ("pred", "forecast", "yhat"))
        }
        if not candidates:
            return None
        return {"origins": None, "truth": truth, "candidates": candidates,
                "origin_source": "npy-content-match"}


def recover_origins(items: list[dict], pool: dict) -> int:
    """Assign origins to origin-less items by matching truth content.

    An item whose truth slices are byte-equal (within ``TRUTH_ATOL``) to
    pooled slices belongs to exactly those origins — recovered, not
    guessed. Items whose truth matches nothing, or matches ambiguously,
    keep ``origins=None`` and stay out of the comparison.
    """
    import numpy as np

    recovered = 0
    for item in items:
        if item.get("origins") is not None or item.get("truth") is None:
            continue
        assigned = []
        for i in range(item["truth"].shape[0]):
            row = item["truth"][i]
            hits = [o for o, t in pool.items()
                    if getattr(t, "shape", None) == row.shape
                    and np.allclose(t, row, rtol=0, atol=TRUTH_ATOL)]
            if len(hits) != 1:
                assigned = []
                break
            assigned.append(hits[0])
        if assigned and len(set(assigned)) == len(assigned):
            item["origins"] = np.asarray(assigned)
            recovered += 1
    return recovered


def drop_undersized(items: list[dict]) -> list[tuple[str, tuple]]:
    """Remove artifacts whose series coverage is below the corpus mode.

    A metric over a subset of series is not comparable with one over the
    full set, so an 8-of-862-sensor file cannot join a cross-run rescore.
    Coverage is inferred as the modal per-origin truth shape across all
    artifacts; anything smaller is excluded and reported.
    """
    from collections import Counter

    shapes = Counter()
    for it in items:
        if it.get("truth") is not None:
            shapes[tuple(it["truth"].shape[1:])] += 1
    if not shapes:
        return []
    full = max(shapes, key=lambda k: (shapes[k], k))
    dropped = []
    for it in list(items):
        t = it.get("truth")
        if t is not None and tuple(t.shape[1:]) != full:
            dropped.append((it["experiment"], tuple(t.shape[1:])))
            items.remove(it)
    return dropped


def build_truth_pool(all_items: list[dict]) -> tuple[dict, int]:
    """origin -> truth slice, verified consistent across every source."""
    import numpy as np

    pool: dict[int, object] = {}
    conflicts = 0
    for item in all_items:
        if item["truth"] is None or item.get("origins") is None:
            continue
        for i, origin in enumerate(item["origins"].tolist()):
            t = item["truth"][i]
            if origin in pool:
                if not np.allclose(pool[origin], t, rtol=0, atol=TRUTH_ATOL):
                    conflicts += 1
            else:
                pool[int(origin)] = t
    return pool, conflicts


def resolve_truth(item: dict, pool: dict):
    """Truth tensor for this item's origins: self-stored or pool-assembled."""
    import numpy as np

    if item["truth"] is not None:
        return item["truth"], "self"
    slices = []
    for origin in item["origins"].tolist():
        if int(origin) not in pool:
            return None, "missing"
        slices.append(pool[int(origin)])
    return np.stack(slices), "pool"


def score_items(items: list[dict], pool: dict, exps: dict) -> list[dict]:
    rows = []
    for item in items:
        if item.get("origins") is None:
            # Loose-.npy artifact whose truth matched no pooled origin (or
            # matched ambiguously): rescoreable in principle, unalignable
            # here. Reported, never silently dropped.
            rows.append({"experiment": item["experiment"],
                         "status": "unalignable_origins",
                         "file": str(item["path"])})
            continue
        truth, source = resolve_truth(item, pool)
        if truth is None:
            rows.append({"experiment": item["experiment"], "status": "no_truth",
                         "file": str(item["path"])})
            continue
        self_metric = (exps.get(item["experiment"]) or {}).get("metric")
        scores = {k: _rmse(arr, truth) for k, arr in item["candidates"].items()}
        official = None
        if isinstance(self_metric, (int, float)):
            close = {
                k: s for k, s in scores.items()
                if math.isclose(s, float(self_metric), rel_tol=IDENT_REL_TOL)
            }
            if close:
                official = min(close, key=lambda k: abs(scores[k] - float(self_metric)))
        rows.append({
            "experiment": item["experiment"],
            "status": "scored",
            "file": str(item["path"]),
            "truth_source": source,
            "n_origins": int(len(item["origins"])),
            "origin_range": [int(item["origins"].min()), int(item["origins"].max())],
            "self_reported_metric": self_metric,
            "array_scores": {k: round(v, 6) for k, v in scores.items()},
            "official_array": official,
            "official_referee_score": round(scores[official], 6) if official else None,
            "self_report_reproduced": official is not None,
        })
    return rows


def _chosen_array(item: dict, row: dict):
    name = row.get("official_array")
    if name is None and row.get("array_scores"):
        name = min(row["array_scores"], key=row["array_scores"].get)
    return name


def cross_compare(pair_name: str, left, right, pool: dict, spec: dict) -> dict:
    """Score both runs on the maximal origin set their experiments share."""
    import numpy as np

    (lrec, litems, lrows), (rrec, ritems, rrows) = left, right
    litems = [it for it in litems if it.get("origins") is not None]
    ritems = [it for it in ritems if it.get("origins") is not None]
    lsets = {tuple(item["origins"].tolist()) for item in litems}
    rsets = {tuple(item["origins"].tolist()) for item in ritems}
    best_shared: tuple = ()
    for ls in lsets:
        for rs in rsets:
            inter = tuple(sorted(set(ls) & set(rs)))
            if len(inter) > len(best_shared):
                best_shared = inter
    if len(best_shared) < MIN_SHARED_ORIGINS:
        return {"pair": pair_name, "comparable": False,
                "reason": f"only {len(best_shared)} shared origins across all "
                          f"protocol combinations (< {MIN_SHARED_ORIGINS})"}
    if any(int(o) not in pool for o in best_shared):
        return {"pair": pair_name, "comparable": False,
                "reason": "shared origins missing from the verified truth pool"}
    truth = np.stack([pool[int(o)] for o in best_shared])

    board = []
    for side, rec, items, rows in (("left", lrec, litems, lrows),
                                   ("right", rrec, ritems, rrows)):
        row_by_exp = {r["experiment"]: r for r in rows if r["status"] == "scored"}
        for item in items:
            row = row_by_exp.get(item["experiment"])
            if row is None:
                continue
            omap = {int(o): i for i, o in enumerate(item["origins"].tolist())}
            if any(int(o) not in omap for o in best_shared):
                continue
            idx = [omap[int(o)] for o in best_shared]
            name = _chosen_array(item, row)
            if name is None:
                continue
            board.append({
                "side": side,
                "run": rec.label,
                "experiment": item["experiment"],
                "array": name,
                "self_report_identified": bool(row.get("official_array")),
                "referee_score": round(_rmse(item["candidates"][name][idx], truth), 6),
            })
    board.sort(key=lambda r: r["referee_score"],
               reverse=not spec["lower_is_better"])
    best = {}
    for side in ("left", "right"):
        side_rows = [r for r in board if r["side"] == side]
        best[side] = side_rows[0] if side_rows else None
    return {
        "pair": pair_name,
        "left": lrec.label,
        "right": rrec.label,
        "comparable": bool(best["left"] and best["right"]),
        "metric": spec["metric"],
        "lower_is_better": spec["lower_is_better"],
        "shared_origins": len(best_shared),
        "shared_origin_range": [int(best_shared[0]), int(best_shared[-1])],
        "experiments_on_board": {"left": sum(1 for r in board if r["side"] == "left"),
                                 "right": sum(1 for r in board if r["side"] == "right")},
        "leaderboard": board,
        "best": best,
        "winner": (
            None if not (best["left"] and best["right"]) else
            ("left" if (best["left"]["referee_score"] < best["right"]["referee_score"])
             == spec["lower_is_better"] else "right")
        ),
    }


_TRUTH_CACHE: dict[str, tuple[dict, int]] = {}


def _classification_truth(truth_spec: dict) -> tuple[dict, int]:
    """Frozen holdout labels: {id: 0/1} plus the holdout row count.

    Unlike the forecast domains there is no consistency pool to build — the
    dataset itself is the shared truth, and the holdout boundary is part of
    the task contract, so every run is scored against one identical slice.
    """
    key = truth_spec["path"]
    if key in _TRUTH_CACHE:
        return _TRUTH_CACHE[key]
    import pandas as pd

    df = pd.read_parquet(key, columns=[truth_spec["id_col"],
                                       truth_spec["label_col"],
                                       truth_spec["holdout_col"]])
    hold = df[df[truth_spec["holdout_col"]] >= truth_spec["holdout_from"]]
    labels = {
        int(i): int(lab == truth_spec["positive"])
        for i, lab in zip(hold[truth_spec["id_col"]], hold[truth_spec["label_col"]])
    }
    _TRUTH_CACHE[key] = (labels, len(labels))
    return _TRUTH_CACHE[key]


def _auc(y, p) -> float | None:
    import numpy as np

    y = np.asarray(y)
    p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Rank-based (Mann-Whitney) AUC with tie correction; no sklearn needed.
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype="float64")
    ranks[order] = np.arange(1, len(order) + 1)
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def classification_verification(workspace, spec: dict, exps: dict,
                                scored: int) -> tuple[dict, list[dict]]:
    """Re-score preserved classification predictions against frozen labels.

    Returns the per-run verification dict plus leaderboard rows for the
    cross-run board (shared truth means cross-run comparison is valid by
    construction, unlike the consistency-pool domains).
    """
    import numpy as np
    import pandas as pd

    labels, n_holdout = _classification_truth(spec["truth"])
    rows, board = [], []
    reproduced = 0
    seen: set[str] = set()
    non_board: list[str] = []
    for pattern in spec["pred_globs"]:
        for f in sorted(globmod.glob(str(workspace / pattern))):
            exp = f.split("/experiments/")[1].split("/")[0]
            if exp in seen:
                continue
            # Only board-registered experiments are scoreable evidence. A
            # diagnostic artifact directory that never entered the run's
            # experiment table (measured: `perfect_foresight_probe`, a
            # labels-as-predictions self-test at logloss 1e-06) must never
            # top a leaderboard.
            if exps and exp not in exps:
                if exp not in non_board:
                    non_board.append(exp)
                continue
            seen.add(exp)
            claimed = (exps.get(exp) or {}).get("metric")
            row = {"experiment": exp, "path": f, "status": "scored",
                   "claimed": claimed, "recomputed": None, "auc": None,
                   "coverage": 0.0, "self_report_reproduced": False}
            try:
                df = (pd.read_parquet(f) if f.endswith(".parquet")
                      else pd.read_csv(f))
                ids = df[spec["id_col"]].astype("int64").to_numpy()
                probs = df[spec["prob_col"]].astype("float64").to_numpy()
            except Exception as exc:  # noqa: BLE001 — one bad file must not kill the referee
                row["status"] = f"unreadable: {type(exc).__name__}"
                rows.append(row)
                continue
            mask = np.array([i in labels for i in ids])
            y = np.array([labels[i] for i in ids[mask]], dtype="float64")
            p = np.clip(probs[mask], 1e-6, 1 - 1e-6)
            row["coverage"] = round(len(y) / n_holdout, 4) if n_holdout else 0.0
            if row["coverage"] < spec.get("min_coverage", 0.95):
                row["status"] = "partial_coverage"
                rows.append(row)
                continue
            logloss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
            row["recomputed"] = round(logloss, 6)
            row["auc"] = round(a, 6) if (a := _auc(y, p)) is not None else None
            if claimed:
                row["self_report_reproduced"] = (
                    abs(logloss - claimed) <= IDENT_REL_TOL * abs(claimed))
                reproduced += bool(row["self_report_reproduced"])
            rows.append(row)
            board.append({"experiment": exp, "referee_score": row["recomputed"],
                          "auc": row["auc"], "coverage": row["coverage"]})
    n_scored = sum(1 for r in rows if r["status"] == "scored")
    return ({
        "scored_experiments": scored,
        "experiments_with_predictions": n_scored,
        "artifact_coverage": round(n_scored / scored, 3) if scored else 0.0,
        "self_report_reproduced": reproduced,
        "holdout_rows": n_holdout,
        "non_board_artifacts_excluded": non_board,
        "experiments": rows,
    }, board)


_REG_TRUTH_CACHE: dict[str, tuple[dict, int]] = {}


def _regression_truth(truth_spec: dict) -> tuple[dict, int]:
    """Frozen holdout targets+weights keyed by the id-column tuple."""
    key = truth_spec["path"]
    if key in _REG_TRUTH_CACHE:
        return _REG_TRUTH_CACHE[key]
    import pandas as pd

    id_cols = list(truth_spec["id_cols"])
    df = pd.read_csv(key, usecols=id_cols + [truth_spec["target_col"],
                                             truth_spec["weight_col"],
                                             truth_spec["holdout_col"]],
                     low_memory=False)
    hold = df[(df[truth_spec["holdout_col"]] >= truth_spec["holdout_from"])
              & (df[truth_spec["holdout_col"]] <= truth_spec["holdout_to"])]
    # Keys are not unique in the source (repeated quotes per day); the
    # contract is one prediction per key, applied to every matching row, so
    # the referee reproduces the task's row-level weighted MAE exactly.
    labels: dict = {}
    tc, wc = truth_spec["target_col"], truth_spec["weight_col"]
    for _, row in hold.iterrows():
        labels.setdefault(tuple(row[c] for c in id_cols), []).append(
            (row[tc], row[wc]))
    n_rows = int(len(hold))
    _REG_TRUTH_CACHE[key] = (labels, n_rows)
    return _REG_TRUTH_CACHE[key]


def regression_verification(workspace, spec: dict, exps: dict,
                            scored: int) -> tuple[dict, list[dict]]:
    """Re-score preserved regression predictions against frozen targets.

    Weighted mean absolute error against the dataset's own holdout slice —
    shared truth by construction, so cross-run comparison is valid, exactly
    as in the classification kind.
    """
    import numpy as np
    import pandas as pd

    labels, n_holdout = _regression_truth(spec["truth"])
    id_cols = list(spec["truth"]["id_cols"])
    rows, board = [], []
    reproduced = 0
    seen: set[str] = set()
    non_board: list[str] = []
    for pattern in spec["pred_globs"]:
        for f in sorted(globmod.glob(str(workspace / pattern))):
            exp = f.split("/experiments/")[1].split("/")[0]
            if exp in seen:
                continue
            # Board-registered experiments only (see the classification
            # branch: a labels-as-predictions probe directory must never
            # enter a leaderboard).
            if exps and exp not in exps:
                if exp not in non_board:
                    non_board.append(exp)
                continue
            seen.add(exp)
            claimed = (exps.get(exp) or {}).get("metric")
            row = {"experiment": exp, "path": f, "status": "scored",
                   "claimed": claimed, "recomputed": None,
                   "coverage": 0.0, "self_report_reproduced": False}
            try:
                df = (pd.read_parquet(f) if f.endswith(".parquet")
                      else pd.read_csv(f))
                keys = list(map(tuple, df[id_cols].itertuples(index=False)))
                preds = df[spec["pred_col"]].astype("float64").to_numpy()
            except Exception as exc:  # noqa: BLE001 — one bad file must not kill the referee
                row["status"] = f"unreadable: {type(exc).__name__}"
                rows.append(row)
                continue
            triples = [(t, wt, pr) for k, pr in zip(keys, preds)
                       for (t, wt) in labels.get(k, ())]
            row["coverage"] = round(len(triples) / n_holdout, 4) if n_holdout else 0.0
            if row["coverage"] < spec.get("min_coverage", 0.95):
                row["status"] = "partial_coverage"
                rows.append(row)
                continue
            y = np.array([t for t, _, _ in triples], dtype="float64")
            w = np.array([wt for _, wt, _ in triples], dtype="float64")
            p = np.array([pr for _, _, pr in triples], dtype="float64")
            wmae = float(np.average(np.abs(y - p), weights=w))
            row["recomputed"] = round(wmae, 6)
            if claimed:
                row["self_report_reproduced"] = (
                    abs(wmae - claimed) <= IDENT_REL_TOL * abs(claimed))
                reproduced += bool(row["self_report_reproduced"])
            rows.append(row)
            board.append({"experiment": exp, "referee_score": row["recomputed"],
                          "coverage": row["coverage"]})
    n_scored = sum(1 for r in rows if r["status"] == "scored")
    return ({
        "scored_experiments": scored,
        "experiments_with_predictions": n_scored,
        "artifact_coverage": round(n_scored / scored, 3) if scored else 0.0,
        "self_report_reproduced": reproduced,
        "holdout_rows": n_holdout,
        "non_board_artifacts_excluded": non_board,
        "experiments": rows,
    }, board)


def kernel_verification(workspace, spec: dict, exps: dict,
                        scored: int) -> tuple[dict, list[dict]]:
    """Correctness-gated throughput recompute from preserved kernel reports.

    The gate is absolute: a report whose fingerprint does not match the task,
    whose correctness block fails its own tolerance, or whose timing sample
    count is under the protocol minimum scores nothing — a fast wrong kernel
    must never reach the board.
    """
    import statistics

    rows, board = [], []
    reproduced = 0
    seen: set[str] = set()
    for pattern in spec["pred_globs"]:
        for f in sorted(globmod.glob(str(workspace / pattern))):
            exp = f.split("/experiments/")[1].split("/")[0]
            if exp in seen:
                continue
            seen.add(exp)
            claimed = (exps.get(exp) or {}).get("metric")
            row = {"experiment": exp, "path": f, "status": "scored",
                   "claimed": claimed, "recomputed": None,
                   "correctness_passed": False, "median_ms": None,
                   "self_report_reproduced": False}
            try:
                rep = json.loads(Path(f).read_text())
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"unreadable: {type(exc).__name__}"
                rows.append(row)
                continue
            fp = rep.get("fingerprint") or {}
            expected = spec.get("expected_fingerprint") or {}
            if any(fp.get(k) != v for k, v in expected.items()):
                row["status"] = "fingerprint_mismatch"
                rows.append(row)
                continue
            corr = rep.get("correctness") or {}
            err_norm = corr.get("err_norm")
            atol, rtol = corr.get("atol"), corr.get("rtol")
            # The gate: err_norm <= 1.0 under tolerances at least as strict
            # as the spec's (a report may tighten atol/rtol, never loosen).
            if not (isinstance(err_norm, (int, float))
                    and isinstance(atol, (int, float))
                    and isinstance(rtol, (int, float))
                    and err_norm <= 1.0
                    and atol <= spec.get("atol", atol)
                    and rtol <= spec.get("rtol", rtol)):
                row["status"] = "correctness_failed"
                rows.append(row)
                continue
            row["correctness_passed"] = True
            samples = (rep.get("timing") or {}).get("samples_ms") or []
            if len(samples) < spec.get("min_samples", 50):
                row["status"] = "insufficient_timing_samples"
                rows.append(row)
                continue
            med_ms = statistics.median(samples)
            row["median_ms"] = round(med_ms, 4)
            tflops = spec["flops_per_call"] / (med_ms / 1e3) / 1e12
            row["recomputed"] = round(tflops, 3)
            if claimed:
                row["self_report_reproduced"] = (
                    abs(tflops - claimed) <= IDENT_REL_TOL * abs(claimed))
                reproduced += bool(row["self_report_reproduced"])
            rows.append(row)
            board.append({"experiment": exp, "referee_score": row["recomputed"],
                          "median_ms": row["median_ms"]})
    n_scored = sum(1 for r in rows if r["status"] == "scored")
    return ({
        "scored_experiments": scored,
        "experiments_with_predictions": n_scored,
        "artifact_coverage": round(n_scored / scored, 3) if scored else 0.0,
        "self_report_reproduced": reproduced,
        "experiments": rows,
    }, board)


def refuse_no_board(rec, exps: dict, kind: str | None) -> dict | None:
    """Refusal record for a workspace run with no experiment board.

    A workspace run whose experiments.db is empty, missing, or never
    created is a broken run; its prediction artifacts must not enter
    leaderboards or crown pair winners — five renamed-aside dead attempts
    were scored exactly that way (2026-08-08). Evidence-contract runs
    carry no db by design (their record says "contract-layout") and are
    exempt. Returns None when the run may be scored normally."""
    if exps:
        return None
    if str(getattr(rec, "framework_evidence", "") or "").startswith(
            "contract-layout"):
        return None
    return {
        "label": rec.label, "framework": rec.framework,
        "domain": rec.domain,
        "referee_kind": kind or "shared_origin_pool",
        "rankable": False,
        "rankable_reason": ("workspace run has no experiment board (empty "
                            "or missing experiments.db) — broken run, "
                            "artifacts refused"),
        "refused_no_board": True,
        "experiments": [],
    }


def shared_truth_pair(pair_name: str, spec: dict, lrec, rrec,
                      lboard: list[dict], rboard: list[dict]) -> dict:
    """Assemble the pair verdict for domains whose truth is shared by
    construction (classification tables, kernel benches) — same shape as
    ``cross_compare`` so every downstream consumer stays domain-agnostic."""
    board = ([dict(r, side="left", run=lrec.label) for r in lboard]
             + [dict(r, side="right", run=rrec.label) for r in rboard])
    board.sort(key=lambda r: r["referee_score"],
               reverse=not spec["lower_is_better"])
    best = {}
    for side in ("left", "right"):
        side_rows = [r for r in board if r["side"] == side]
        best[side] = side_rows[0] if side_rows else None
    return {
        "pair": pair_name,
        "left": lrec.label,
        "right": rrec.label,
        "comparable": bool(best["left"] and best["right"]),
        "metric": spec["metric"],
        "lower_is_better": spec["lower_is_better"],
        "experiments_on_board": {
            "left": sum(1 for r in board if r["side"] == "left"),
            "right": sum(1 for r in board if r["side"] == "right")},
        "leaderboard": board,
        "best": best,
        # Same winner rule as cross_compare — the docstring's "same shape"
        # promise was broken until 2026-08-03: these pairs shipped without
        # a winner and every downstream pair verdict read None.
        "winner": (
            None if not (best["left"] and best["right"]) else
            ("left" if (best["left"]["referee_score"]
                        < best["right"]["referee_score"])
             == spec["lower_is_better"] else "right")
        ),
    }


def run_referee(corpus_path: Path, packs_dir: Path, out_path: Path,
                pairs: list[tuple[str, str, str]]) -> dict:
    records = {r.label: r for r in load_registry(corpus_path)}

    def pack_exps(label):
        safe = label.replace("/", "__").replace("#", "_")
        p = packs_dir / f"{safe}.json"
        if not p.is_file():
            return {}, 0
        pk = json.loads(p.read_text())
        exps = {e["name"]: e for e in (pk.get("experiments") or {}).get("experiments", [])}
        scored = (pk.get("experiments") or {}).get("scored") or 0
        return exps, scored

    result = {"schema": "runcmp-referee-3", "runs": {}, "pairs": []}

    # ---- N-way pass: score EVERY run against its domain's truth ----------
    # The referee used to score only runs named in --pair flags — a two-
    # harness assumption. With many harnesses the leaderboard is N-way:
    # every run in a domain is scored once, against ONE domain-wide truth
    # pool where the domain has shared truth. Pairs below become an
    # optional 2-way verdict overlay reusing these scores.
    boards: dict[str, object] = {}   # per-run boards (shared-truth kinds)
    arts: dict[str, list] = {}       # per-run artifact items (pool kinds)
    pools: dict[str, tuple] = {}     # domain -> (pool, conflicts)
    by_domain: dict[str, list] = {}
    for rec in records.values():
        if rec.domain in DOMAIN_SPECS:
            by_domain.setdefault(rec.domain, []).append(rec)
    for domain, group in sorted(by_domain.items()):
        spec = DOMAIN_SPECS[domain]
        kind = spec.get("kind")
        if kind == "bpb_curve":
            idsets: dict[str, frozenset] = {}
            for rec in group:
                exps, scored = pack_exps(rec.label)
                # identities of SCORED experiments only (a smoke test's
                # throwaway slice must not poison the run's identity set)
                idsets[rec.label] = frozenset(
                    e.get("validation_id") for e in exps.values()
                    if e.get("validation_id")
                    and e.get("metric") is not None
                    and not (e.get("flags") or {}).get("smoke_test")
                    and not (e.get("flags") or {}).get("smoke"))
                v = bpb_verification(Path(rec.workspace), spec, exps, scored)
                result["runs"][rec.label] = {
                    "label": rec.label, "framework": rec.framework,
                    "domain": rec.domain,
                    "referee_kind": kind,
                    "validation_identities": sorted(idsets[rec.label]),
                    **v,
                }
            # Rankability: bpb compares only across runs measured on one
            # frozen validation ruler. Identity strings are self-reported
            # DESCRIPTIONS — eight runs of one task described the same
            # slice eight different ways (2026-08-06: "synth_500.parquet",
            # "shards:synth_500", "shard500_rows0-12000", ...), so string
            # inequality alone is NOT evidence of different data (treating
            # it as such left one arbitrary run ranked and seven not).
            # What IS evidence: (a) multiple distinct identities WITHIN a
            # run — demonstrably not one frozen ruler (the external
            # harness evaluates 5 of its own environments); (b) an
            # identity differing from one that several runs share EXACTLY
            # (an exact shared string is a real fingerprint; a unique
            # string is presumed the task's own split described in the
            # run's own words).
            singles = {lab: next(iter(s)) for lab, s in idsets.items()
                       if len(s) == 1}
            _shared_counts: dict[str, int] = {}
            for ident in singles.values():
                _shared_counts[ident] = _shared_counts.get(ident, 0) + 1
            shared_ids = {i for i, n in _shared_counts.items() if n >= 2}
            for rec in group:
                s = idsets[rec.label]
                why = None
                if len(s) > 1:
                    why = ("evaluates multiple distinct frozen validation "
                           "slices within one run — not one ruler; "
                           "referee-verified per experiment, not rankable "
                           "across runs")
                elif s and shared_ids and next(iter(s)) not in shared_ids:
                    why = ("records a validation identity that differs "
                           "from the frozen slice this task's other runs "
                           "share — referee-verified, not rankable across "
                           "runs")
                result["runs"][rec.label]["rankable"] = why is None
                if why:
                    result["runs"][rec.label]["rankable_reason"] = why
        elif kind == "returns_parquet":
            for rec in group:
                exps, scored = pack_exps(rec.label)
                v = returns_verification(Path(rec.workspace), spec, exps,
                                         scored)
                result["runs"][rec.label] = {
                    "label": rec.label, "framework": rec.framework,
                    "domain": rec.domain,
                    "referee_kind": kind, **v,
                }
        elif kind in ("classification_table", "kernel_bench",
                      "regression_table"):
            verify = {"classification_table": classification_verification,
                      "kernel_bench": kernel_verification,
                      "regression_table": regression_verification}[kind]
            for rec in group:
                exps, scored = pack_exps(rec.label)
                refusal = refuse_no_board(rec, exps, kind)
                if refusal is not None:
                    boards[rec.label] = []
                    result["runs"][rec.label] = refusal
                    continue
                # These kinds read a site dataset for truth. An unreadable
                # dataset (permissions, unmounted share) must cost THIS
                # domain its referee scores, loudly — not crash the whole
                # stage and leave every domain unscored (a cold-start user
                # hit exactly that: one 'nobody'-owned CSV produced no
                # referee.json at all, 2026-08-11).
                try:
                    v, board = verify(Path(rec.workspace), spec, exps,
                                      scored)
                except OSError as exc:
                    boards[rec.label] = []
                    result["runs"][rec.label] = {
                        "label": rec.label, "framework": rec.framework,
                        "domain": rec.domain, "referee_kind": kind,
                        "rankable": False,
                        "refused_truth_unreadable": True,
                        "rankable_reason": (
                            "referee truth/dataset unreadable in this "
                            f"environment: {exc}"),
                        "experiments": [], "scored_experiments": 0,
                    }
                    print(f"  REFUSED {rec.label}: truth unreadable "
                          f"({exc})")
                    continue
                boards[rec.label] = board
                result["runs"][rec.label] = {
                    "label": rec.label, "framework": rec.framework,
                    "domain": rec.domain,
                    "referee_kind": kind, **v,
                }
        else:
            # shared-origin forecast kind: one truth pool over ALL runs in
            # the domain — every run's preserved truth cross-checks every
            # other's (the conflict counter is now domain-global)
            per = [(rec, RunArtifacts(Path(rec.workspace), spec))
                   for rec in group]
            all_items = [i for _, art in per for i in art.items]
            undersized = drop_undersized(all_items)
            for _, art in per:
                art.items = [i for i in art.items if i in all_items]
            pool, conflicts = build_truth_pool(all_items)
            if recover_origins(all_items, pool):
                pool, conflicts = build_truth_pool(all_items)
            pools[domain] = (pool, conflicts)
            for rec, art in per:
                exps, scored = pack_exps(rec.label)
                # Broken run: its preserved truth already fed the domain
                # pool (truth is cross-verified evidence either way), but
                # none of its predictions get scored or paired.
                refusal = refuse_no_board(rec, exps, kind)
                if refusal is not None:
                    arts[rec.label] = []
                    result["runs"][rec.label] = refusal
                    continue
                arts[rec.label] = art.items
                rows = score_items(art.items, pool, exps)
                n_scored = sum(1 for r in rows if r["status"] == "scored")
                result["runs"][rec.label] = {
                    "label": rec.label,
                    "framework": rec.framework,
                    "domain": rec.domain,
                    "referee_kind": kind or "shared_origin_pool",
                    "scored_experiments": scored,
                    "experiments_with_predictions": n_scored,
                    "artifact_coverage": round(n_scored / scored, 3)
                    if scored else 0.0,
                    "self_report_reproduced": sum(
                        bool(r.get("self_report_reproduced")) for r in rows),
                    "truth_pool_conflicts": conflicts,
                    "excluded_for_partial_series_coverage": [
                        {"experiment": e, "series_shape": list(sh)}
                        for e, sh in undersized],
                    "experiments": rows,
                }

    # ---- optional 2-way verdict overlay (reuses the scores above) --------
    for name, left_label, right_label in pairs:
        lrec, rrec = records.get(left_label), records.get(right_label)
        if lrec is None or rrec is None:
            result["pairs"].append({"pair": name, "comparable": False,
                                    "reason": "unknown label"})
            continue
        spec = DOMAIN_SPECS.get(lrec.domain)
        if spec is None or lrec.domain != rrec.domain:
            result["pairs"].append({
                "pair": name, "comparable": False,
                "reason": f"no referee artifact spec for domain {lrec.domain!r} "
                          "(predictions not preserved or domain not mapped)",
            })
            continue
        if spec.get("kind") == "bpb_curve":
            lcov = result["runs"][left_label]["artifact_coverage"]
            rcov = result["runs"][right_label]["artifact_coverage"]
            result["pairs"].append({
                "pair": name, "comparable": False,
                "reason": "bits-per-byte domain: each run validates on its "
                          "own held-out slice, so self-reports are "
                          "independently recomputed per run (coverage "
                          f"{left_label}={lcov}, {right_label}={rcov}) but no "
                          "shared validation identity exists for cross-run "
                          "re-scoring",
            })
            continue
        if spec.get("kind") == "returns_parquet":
            lcov = result["runs"][left_label]["artifact_coverage"]
            rcov = result["runs"][right_label]["artifact_coverage"]
            result["pairs"].append({
                "reason": "returns-series domain: self-reports verified per "
                          "run from preserved daily series (coverage "
                          f"{left_label}={lcov}, {right_label}={rcov}); no "
                          "shared truth pool exists for cross-run re-scoring",
            })
            continue
        if spec.get("kind") in ("classification_table", "kernel_bench",
                                "regression_table"):
            result["pairs"].append(shared_truth_pair(
                name, spec, lrec, rrec,
                boards.get(left_label), boards.get(right_label)))
            continue
        # shared-origin kind: verdict from the domain-wide pool and the
        # per-run scores computed in the N-way pass
        pool, conflicts = pools.get(lrec.domain, ({}, 0))
        comparison = cross_compare(
            name,
            (lrec, arts.get(left_label, []),
             result["runs"].get(left_label, {}).get("experiments", [])),
            (rrec, arts.get(right_label, []),
             result["runs"].get(right_label, {}).get("experiments", [])),
            pool, spec,
        )
        comparison["truth_pool_conflicts"] = conflicts
        result["pairs"].append(comparison)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Referee re-scoring over preserved predictions")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--packs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="referee.json path")
    ap.add_argument("--pair", action="append", default=[],
                    help="NAME=LEFT_LABEL:RIGHT_LABEL (repeatable)")
    args = ap.parse_args(argv)
    pairs = []
    for spec in args.pair:
        name, rest = spec.split("=", 1)
        left, right = rest.split(":", 1)
        pairs.append((name, left, right))
    result = run_referee(args.corpus, args.packs, args.out, pairs)
    for p in result["pairs"]:
        if p.get("comparable"):
            b = p["best"]
            # Field set differs by pair kind (shared-origin domains carry
            # origin/truth-pool stats; shared-truth domains don't) — print
            # what exists instead of crashing the chain on the summary.
            extras = " ".join(
                f"{k}={p[k]}" for k in
                ("shared_origins", "truth_pool_conflicts",
                 "experiments_on_board") if k in p)
            print(f"{p['pair']}: {extras}")
            for side in ("left", "right"):
                if b[side]:
                    ident = b[side].get("self_report_identified")
                    print(f"  {side} best: {b[side]['experiment']} "
                          f"referee_{p['metric']}={b[side]['referee_score']}"
                          + ("" if ident is None
                             else f" (identified={ident})"))
            print(f"  winner: {p.get('winner')}")
        else:
            print(f"{p['pair']}: NOT comparable — {p.get('reason')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
