#!/bin/sh
# The paved PR path: old harness vs proposed harness, one command.
#
#   scripts/runcmp_gate.sh OUT_DIR BASELINE_SEL CANDIDATE_SEL ROOT [ROOT...]
#
#   OUT_DIR        fresh output directory for this gate evaluation
#   BASELINE_SEL   selector for the old harness's runs   (e.g. era=july_base)
#   CANDIDATE_SEL  selector for your new harness's runs  (e.g. era=myfix_0807)
#   ROOT...        run-corpus roots to index (baseline and candidate runs)
#
# Optional environment:
#   GATE_POLICY=path/to/policy.json   (default: the packaged gate_policy.json)
#   GATE_STORE=path/to/mlflow_store   publish the evidence page there too
#   GATE_REVIEWERS="provider:model [provider:model ...]"
#       run one LLM reviewer per entry over the gate's generated mission
#       (each must open its report with an explicit Recommendation:
#       PASS/FAIL line); their positions are quoted on the evidence page.
#       Reviewers need LLM credentials and real time (an hour or more each).
#
# The screen's mechanical reading is the exit code (0 = policy PASS,
# 1 = policy FAIL) so CI can use it — but the DECISION belongs to the
# humans reading OUT_DIR/gate.md / the MLflow evidence page, informed by
# the reviewers' argued recommendations.
set -eu

OUT=$1; BASE_SEL=$2; CAND_SEL=$3; shift 3
PY=${RUNCMP_PYTHON:-python3}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO"

ROOTS=""
for r in "$@"; do ROOTS="$ROOTS --root $r"; done

$PY -m runcmp index $ROOTS --out "$OUT/corpus.json"
# Deterministic pipeline with post-condition checks (no pair specs: the
# referee scores every run N-way regardless).
sh scripts/runcmp_chain.sh "$OUT/corpus.json" "$OUT"
# A FAIL verdict is exit code 1; capture it so the evidence still gets
# published before the script exits with the verdict.
RC=0
$PY -m runcmp gate \
    --corpus "$OUT/corpus.json" --bench "$OUT/bench.json" \
    --referee "$OUT/referee.json" \
    --baseline "$BASE_SEL" --candidate "$CAND_SEL" \
    ${GATE_POLICY:+--policy "$GATE_POLICY"} \
    --out "$OUT" || RC=$?
if [ -n "${GATE_REVIEWERS:-}" ]; then
    i=0
    for spec in $GATE_REVIEWERS; do
        i=$((i + 1))
        provider=${spec%%:*}; model=${spec#*:}
        $PY -m runcmp investigate \
            --corpus "$OUT/corpus.json" --packs "$OUT/packs" \
            --out "$OUT/review_${i}_${model}" \
            --provider "$provider" --model "$model" \
            --mission @"$OUT/mission.md" \
            > "$OUT/review_${i}_${model}.log" 2>&1 &
    done
    wait
    for d in "$OUT"/review_*/; do
        [ -f "$d/findings.jsonl" ] && $PY -m runcmp \
            factcheck --out "$d" --packs "$OUT/packs" || true
    done
fi
if [ -n "${GATE_STORE:-}" ]; then
    $PY -m runcmp publish \
        --corpus "$OUT/corpus.json" --packs "$OUT/packs" \
        --bench "$OUT/bench.json" --referee "$OUT/referee.json" \
        --out "$OUT" --store "$GATE_STORE" \
        --campaign "$(basename "$OUT")" --gate "$OUT/gate.json"
fi
exit $RC
