#!/bin/sh
# Full runcmp deterministic chain — every documented step, in order, so no
# round can silently drop one (2026-07-31: a hand-written chain omitted
# token_accounting and the referee's --pair specs; the referee then wrote an
# empty shell without erroring).
#
# usage: runcmp_chain.sh CORPUS_JSON OUT_DIR [PAIR_SPEC ...]
#   CORPUS_JSON  pre-filtered corpus registry (from `runcmp index` + selection)
#   OUT_DIR      output directory (packs/ created inside)
#   PAIR_SPEC    referee/tabulate pair, NAME=LEFT_LABEL:RIGHT_LABEL, repeatable.
#                REQUIRED for referee scoring: with no pairs the referee
#                produces an empty result by design.
set -eu

CORPUS=$1; OUT=$2; shift 2
PY=${RUNCMP_PYTHON:-python3}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO"

if [ "$#" -eq 0 ]; then
    echo "WARNING: no --pair specs given; referee output will be EMPTY." >&2
fi
PAIRS=""
for p in "$@"; do PAIRS="$PAIRS --pair $p"; done

date '+CHAIN START %H:%M:%S'
# --force: extract caches per-pack; without it a code change is silently
# masked by stale packs (measured 2026-07-31: a metric fix produced
# identical wrong numbers because every pack was served from cache).
# --all: the corpus registry deliberately keeps incomplete/stalled runs
# (a failed run is evidence, not a gap) and the postcondition below asserts
# packs == corpus runs; extract's complete-only default contradicted both
# (2026-08-03: an 8-run corpus with one deadlocked run produced 7 packs and
# failed its own chain assertion).
$PY -m runcmp extract  --corpus "$CORPUS" --out "$OUT/packs" --workers 4 --force --all
$PY -m runcmp tabulate --corpus "$CORPUS" --packs "$OUT/packs" --out "$OUT" $PAIRS
$PY -m runcmp referee  --corpus "$CORPUS" --packs "$OUT/packs" --out "$OUT/referee.json" $PAIRS
$PY -m runcmp bench    --corpus "$CORPUS" --packs "$OUT/packs" --out "$OUT" --referee "$OUT/referee.json"
$PY -m runcmp.token_accounting --corpus "$CORPUS" --packs "$OUT/packs" --out "$OUT"
date '+CHAIN DONE %H:%M:%S'

# Post-conditions: fail loudly if any step under-delivered.
$PY - "$CORPUS" "$OUT" <<'EOF'
import json, sys, pathlib
corpus, out = json.load(open(sys.argv[1])), pathlib.Path(sys.argv[2])
n = len(corpus["runs"])
packs = len(list((out / "packs").glob("*.json")))
assert packs == n, f"packs {packs} != corpus runs {n}"
ref = json.load(open(out / "referee.json"))
assert ref["pairs"] or ref["runs"], "referee.json is EMPTY — pass --pair specs"
for f in ("tables.md", "bench.md", "token_accounting.md"):
    assert (out / f).stat().st_size > 1000, f"{f} missing or tiny"
print(f"chain OK: {n} runs, {packs} packs, {len(ref['pairs'])} referee pairs")
EOF
