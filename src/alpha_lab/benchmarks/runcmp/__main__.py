"""One CLI for the whole benchmarking suite.

    python -m alpha_lab.benchmarks.runcmp <stage> [args]

Deterministic pipeline, in running order (see README.md in this package):

    lineup            list benchmark domains / generate per-cell run configs
    index             discover finished runs -> corpus.json (the registry)
    extract           corpus.json -> one evidence pack per run (packs/)
    tabulate          packs -> tables.md/.json (pair tables, aggregates)
    referee           re-score preserved predictions -> referee.json
    bench             metric registry over packs -> bench.md/.json
    token-accounting  per-run, per-model token/cost ledger
    gate              change gate: candidate vs baseline, policy-judged
    publish           export everything to the showcase MLflow viewer

Judgment layer (LLM investigators + deterministic verification):

    mission           generate an investigation mission from the corpus
    investigate       single-writer investigation -> REPORT.md + findings
    investigate-team  planner/executor/critic investigation (multi-model)
    recompose         re-write only the report, from a review's frozen findings
    rereview          recompose with the review's OWN recorded writer
    factcheck         verify a report's findings against the artifacts (exit 0
                      = every hard audit clean)
    meta-eval         score investigation outputs against a key-fact rubric

Day-2 helpers (all read-only except rereview):

    status            one look at a review dir: critic rounds, report, stubs
    watch             snapshot every run under a root: state, board, best
    preflight         pre-launch checks: corpus, packs, credentials, store
    validate-submission  check an evidence-contract run before indexing
    mission --check   lint a hand-written mission against the corpus

Each stage prints its own --help. `scripts/runcmp_chain.sh` runs the full
deterministic pipeline with post-condition checks.
"""

from __future__ import annotations

import sys

from alpha_lab.benchmarks.runcmp import (
    bench,
    gate,
    preflight,
    rereview,
    status,
    submission,
    watch,
    corpus,
    extract,
    factcheck,
    investigate,
    investigate_team,
    lineup,
    meta_eval,
    mission,
    publish,
    recompose,
    referee,
    tabulate,
    token_accounting,
)

_STAGES = {
    "lineup": lineup.main,
    "index": corpus.main,
    "extract": extract.main,
    "tabulate": tabulate.main,
    "referee": referee.main,
    "bench": bench.main,
    "token-accounting": token_accounting.main,
    "gate": gate.main,
    "mission": mission.main,
    "investigate": investigate.main,
    "investigate-team": investigate_team.main,
    "recompose": recompose.main,
    "rereview": rereview.main,
    "factcheck": factcheck.main,
    "meta-eval": meta_eval.main,
    "publish": publish.main,
    "status": status.main,
    "watch": watch.main,
    "preflight": preflight.main,
    "validate-submission": submission.main,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in _STAGES:
        print(__doc__)
        print(f"usage: python -m alpha_lab.benchmarks.runcmp "
              f"<{'|'.join(_STAGES)}> ...")
        return 2
    return _STAGES[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
