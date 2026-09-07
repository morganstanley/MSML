# Demo submission — the smallest thing the pipeline accepts

Two evidence-contract runs from a fictional harness (`alphalab2-demo`), two
experiments each, self-reported `weighted_mae` only. Real submissions add the
referee prediction files per the domain contract; this placeholder exercises
the full pipeline shape without them.

Run the whole pipeline on it, ending in MLflow:

    OUT=demo_out
    python -m runcmp index    --root examples/demo_submission --out $OUT/corpus.json
    python -m runcmp extract  --corpus $OUT/corpus.json --out $OUT/packs
    python -m runcmp tabulate --corpus $OUT/corpus.json --packs $OUT/packs --out $OUT
    python -m runcmp referee  --corpus $OUT/corpus.json --packs $OUT/packs --out $OUT/referee.json
    python -m runcmp bench    --corpus $OUT/corpus.json --packs $OUT/packs --out $OUT --referee $OUT/referee.json
    pip install -e .[mlflow]
    python -m runcmp publish  --corpus $OUT/corpus.json --packs $OUT/packs \
        --bench $OUT/bench.json --referee $OUT/referee.json --out $OUT --store demo_store
    mlflow server --backend-store-uri sqlite:///demo_store/mlflow.db \
        --default-artifact-root demo_store/artifacts --port 5601

Then open http://127.0.0.1:5601 and follow the MLflow guide in the main README.
