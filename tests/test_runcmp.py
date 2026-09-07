"""End-to-end tests for the deterministic runcmp layers on a synthetic corpus."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from alpha_lab.benchmarks.runcmp import corpus as corpus_mod
from alpha_lab.benchmarks.runcmp import extract as extract_mod
from alpha_lab.benchmarks.runcmp import factcheck as factcheck_mod
from alpha_lab.benchmarks.runcmp import tabulate as tabulate_mod
from alpha_lab.benchmarks.runcmp.corpus import RunRecord, build_registry, select_db
from alpha_lab.benchmarks.runcmp.extract import (
    build_pack,
    normalize_signature,
    tool_failure_signature,
)


def _make_run(root, era, domain, framework, *, rows=3, db_in_alpha_lab=False,
              shadow_empty_db=False, plant_artifacts=True):
    """Create a minimal synthetic run directory with all major artifacts."""
    run_dir = root / era / domain / framework
    ws = run_dir / "ws"
    ws.mkdir(parents=True)
    t0 = 1_784_000_000.0

    # experiments.db
    db_dir = ws / ".alpha_lab" if db_in_alpha_lab else ws
    db_dir.mkdir(exist_ok=True)
    con = sqlite3.connect(db_dir / "experiments.db")
    con.execute(
        "CREATE TABLE experiments (id INTEGER PRIMARY KEY, name TEXT, "
        "status TEXT, created_at REAL, started_at REAL, finished_at REAL, "
        "fix_attempts INTEGER, error TEXT, results_json TEXT, hypothesis TEXT)"
    )
    for i in range(rows):
        metric = 1.0 - 0.1 * i if framework == "cond" else 1.05 - 0.1 * i
        con.execute(
            "INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                i + 1, f"exp_{i}", "done",
                t0 + i * 60, t0 + i * 60 + 5, t0 + i * 60 + 50,
                0, None,
                json.dumps({"val_bpb": metric,
                            "validation_content_sha256": "aa11"}),
                f"hypothesis {i}",
            ),
        )
    con.commit()
    con.close()
    if shadow_empty_db:
        # A newer, table-less DB at the other location must NOT win.
        other = ws / ".alpha_lab" if not db_in_alpha_lab else ws
        other.mkdir(exist_ok=True)
        (other / "experiments.db").write_bytes(b"")
        future = time.time() + 3600
        import os
        os.utime(other / "experiments.db", (future, future))

    # adapter manifest (metric config)
    (ws / "adapter").mkdir()
    (ws / "adapter" / "manifest.json").write_text(json.dumps({
        "domain_name": "test", "metric": {
            "primary_metric": "val_bpb", "direction": "minimize"},
        "experiment": {},
    }))

    # framework fingerprints (external harnesses leave none of these)
    if not plant_artifacts:
        pass
    elif framework == "cond":
        meta = ws / "meta"
        meta.mkdir()
        (meta / "meta_log.jsonl").write_text(json.dumps({
            "ts": t0, "decision_type": "directive", "target": "strategist",
            "reason": "r", "evidence": "e", "self_check": "s"}) + "\n")
        (meta / "annotations.json").write_text(json.dumps({"1": "champion"}))
        # token ledger: the conductor seat ran a DIFFERENT model than the
        # rest — the observed seat assignment must surface in the pack
        (meta / "token_usage.jsonl").write_text(
            json.dumps({"ts": t0, "log_name": "conductor_timer",
                        "provider": "anthropic", "model": "opus-x",
                        "input_tokens": 10, "output_tokens": 5}) + "\n"
            + json.dumps({"ts": t0 + 1, "log_name": "strategist",
                          "provider": "chat", "model": "m-main",
                          "input_tokens": 10, "output_tokens": 5}) + "\n")
        (ws / "config.json").write_text(json.dumps({
            "provider": "chat", "model": "m-main",
            "conductor_provider": "anthropic", "conductor_model": "opus-x"}))
    else:
        (ws / "agenda.md").write_text("agenda\n")
        (ws / ".alpha_lab").mkdir(exist_ok=True)
        (ws / ".alpha_lab" / "config.json").write_text(json.dumps({
            "provider": "local", "model": "m-main"}))

    # events.jsonl
    events = []
    events.append({"type": "run_start", "run": "ws", "config": "x",
                   "workspace": str(ws)})
    for phase, ts in (("phase1", t0), ("phase3", t0 + 100)):
        events.append({"type": "phase", "timestamp": ts, "phase": phase,
                       "step": "s", "iteration": 0, "status": "starting",
                       "detail": ""})
    events.append({
        "type": "api_request", "timestamp": t0 + 1, "model": "m",
        "instructions": "BIG" * 2000, "input": "x",
    })
    events.append({
        "type": "api_response", "timestamp": t0 + 2,
        "usage": {"input_tokens": 1000,
                  "input_tokens_details": {"cached_tokens": 400,
                                           "cache_write_tokens": 0},
                  "output_tokens": 50,
                  "output_tokens_details": {"reasoning_tokens": 10}},
    })
    events.append({"type": "tool_call", "timestamp": t0 + 3,
                   "call_id": "c1", "name": "memory_store", "arguments": "{}"})
    events.append({"type": "tool_result", "timestamp": t0 + 4, "call_id": "c1",
                   "name": "memory_store",
                   "output": "[ERROR] ProgrammingError: SQLite objects created "
                             "in a thread 12345 /some/path/file.py"
                   if framework == "msml" else "stored ok"})
    with open(run_dir / "events.jsonl", "w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")

    # experiment dir with an untracked second realization (msml side only):
    # a failed launch output well before started_at, then the real one.
    if framework == "msml":
        exp_dir = ws / "experiments" / "exp_0"
        exp_dir.mkdir(parents=True)
        (exp_dir / "results").mkdir()
        (exp_dir / "results" / "metrics.json").write_text("{}")
        import os as _os
        _os.utime(exp_dir / "results" / "metrics.json", (t0 + 48, t0 + 48))
        failed = exp_dir / "local_job.aaaa.out"
        failed.write_text("Traceback\nValueError:\nPass data_root or set M5_DATA_ROOT\n")
        import os as _os
        _os.utime(failed, (t0 - 4000, t0 - 4000))
        ok = exp_dir / "local_job.out"
        ok.write_text("fine\n" * 100)
        _os.utime(ok, (t0 + 45, t0 + 45))

    # run.log + agent logs
    (run_dir / "run.log").write_text(
        "launch 2026-07-12 13:10:22 | test\n"
        '13:10:53 INFO HTTP Request: POST https://gw/openai/v1/responses '
        '"HTTP/1.1 200 OK"\n'
        '13:10:54 INFO HTTP Request: POST https://gw/openai/v1/embeddings '
        '"HTTP/1.1 429 Too Many Requests"\n'
        f"=== {framework} EXIT code=0 at 17:35:56 ===\n"
    )
    logs = ws / "logs"
    logs.mkdir()
    with open(logs / "strategist.jsonl", "w") as fh:
        fh.write(json.dumps({"type": "api_request", "timestamp": t0,
                             "model": "m-main",
                             "instructions": "You are the strategist."}) + "\n")
        fh.write(json.dumps({
            "type": "api_response", "timestamp": t0 + 1,
            "usage": {"input_tokens": 500, "output_tokens": 20}}) + "\n")
    # a compressed-in-place transcript: must be read (not silently skipped),
    # and a verifier_worker_* name must classify as verifier, not worker
    import gzip as _gzip
    with _gzip.open(logs / "verifier_worker_exp.jsonl.gz", "wt") as fh:
        fh.write(json.dumps({"type": "api_request", "timestamp": t0 + 2,
                             "model": "m-main",
                             "instructions": "You are the verifier."}) + "\n")
        fh.write(json.dumps({
            "type": "api_response", "timestamp": t0 + 3,
            "usage": {"input_tokens": 300, "output_tokens": 10}}) + "\n")
    (ws / "learnings.md").write_text("learned\n")
    return run_dir


def _make_contract_run(root, era="ext_era", harness="extbench", n=2):
    """External harness submission: evidence-contract layout only — no db,
    no adapter, no transcripts. Domain tokens in the era dir name."""
    run = root / era / f"run_{harness}_20260804T120000Z"
    for i in range(n):
        rd = run / "experiments" / f"exp{i}" / "results"
        rd.mkdir(parents=True)
        bpb = 1.10 - 0.05 * i
        (rd / "metrics.json").write_text(json.dumps({
            "val_bpb": bpb,
            # summed nll under the ambiguous external spelling (the same
            # key is a per-token mean in another harness); the referee must
            # classify the units by magnitude and reproduce the claim
            "val_loss_nats": bpb * 1000.0 * 0.6931471805599453,
            "val_bytes": 1000.0,
            "val_tokens": 1000.0,
            "val_slice_id": "s:abc",
            "approach": f"approach {i}"}))
        (rd.parent / "code.py").write_text("def f():\n    return 1\n")
    return run


@pytest.fixture()
def synthetic_corpus(tmp_path):
    root = tmp_path / "corpus"
    _make_run(root, "eraA", "domain2_llm_speedrun", "cond")
    _make_run(root, "eraA", "domain2_llm_speedrun", "msml",
              db_in_alpha_lab=True, shadow_empty_db=True)
    return root


class TestCorpus:
    def test_registry_pairs_and_completeness(self, synthetic_corpus):
        records = build_registry(synthetic_corpus)
        assert len(records) == 2
        by_fw = {r.framework: r for r in records}
        assert by_fw["cond"].completeness == "complete"
        assert by_fw["msml"].completeness == "complete"
        assert by_fw["cond"].pair_key == by_fw["msml"].pair_key != ""
        assert by_fw["cond"].domain == "domain2"

    def test_select_db_ignores_empty_shadow(self, synthetic_corpus):
        ws = synthetic_corpus / "eraA/domain2_llm_speedrun/msml/ws"
        selected = select_db(ws)
        assert selected is not None
        path, counts = selected
        # The populated .alpha_lab DB wins over the newer zero-byte root DB.
        assert ".alpha_lab" in str(path)
        assert counts == {"done": 3}


class TestArchivedAttemptsExcluded:
    """Archived failed attempts (renamed aside with a dot-suffix) must never
    enter the registry. Missed 2026-08-08: six renamed attempts from one
    era — five ``*.resumed_invalid_*`` and one ``*.interference_degraded_*``
    carrying a full 12-row board — were indexed, scored, formed a phantom
    pair, and reached three shipped reports; the old exclusion was a fixed
    keyword list that didn't know the new suffix words."""

    def test_sibling_shadow_and_new_keywords_excluded(self, tmp_path):
        import shutil
        from pathlib import Path
        root = tmp_path / "corpus"
        _make_run(root, "eraA", "domain2_llm_speedrun", "cond")
        run_dir = root / "eraA" / "domain2_llm_speedrun" / "cond"
        live = run_dir / "cellA"
        shutil.copytree(run_dir / "ws", live)
        # structural rule: an arbitrary, never-catalogued suffix over a
        # live sibling directory is an archived attempt of that sibling
        shutil.copytree(live, run_dir / "cellA.zz_new_suffix_9999")
        # keyword rule: live sibling gone, but the suffix word is known
        shutil.copytree(live, run_dir / "cellB.resumed_invalid_2137")
        shutil.copytree(live, run_dir / "cellC.interference_degraded_2120")

        skipped: list = []
        records = build_registry(root, skipped_archived=skipped)
        names = {Path(r.workspace).name for r in records}
        assert "cellA" in names and "ws" in names
        assert not any("." in n for n in names)
        assert {p.name for p in skipped} == {
            "cellA.zz_new_suffix_9999",
            "cellB.resumed_invalid_2137",
            "cellC.interference_degraded_2120",
        }


class TestCompressedLogsStillIndexed:
    """Finished campaigns get their big logs compressed in place
    (events.jsonl -> events.jsonl.gz). Missed 2026-08-06: a July run set's
    wall-clock columns went blank as "(no event stream)" while a 1.4 GB
    events.jsonl.gz sat on disk — the registry tested only the plain name."""

    def test_registry_and_pack_follow_gz_rename(self, synthetic_corpus):
        import gzip as _gzip
        run_dir = synthetic_corpus / "eraA/domain2_llm_speedrun/cond"
        for name in ("events.jsonl", "run.log"):
            plain = run_dir / name
            with open(plain, "rb") as src, \
                    _gzip.open(run_dir / (name + ".gz"), "wb") as dst:
                dst.write(src.read())
            plain.unlink()

        rec = {r.framework: r
               for r in build_registry(synthetic_corpus)}["cond"]
        assert rec.events_path and rec.events_path.endswith(".gz")
        assert rec.run_log_path and rec.run_log_path.endswith(".gz")

        pack = build_pack(rec)
        # Identical evidence to the uncompressed pack in test_pack_contents.
        assert pack["events"]["api_calls"] == 1
        assert pack["events"]["tokens"]["input"] == 1000
        assert pack["run_log"]["http_endpoints"]


class TestExtract:
    def test_signatures(self):
        sig = normalize_signature("Error 404 at /a/b/c.py deadbeef99 again")
        assert "<n>" in sig and "<path>" in sig and "<hex>" in sig
        assert tool_failure_signature("[ERROR] boom") is not None
        assert tool_failure_signature("all good, no error") is None

    def test_pack_contents(self, synthetic_corpus):
        records = build_registry(synthetic_corpus)
        by_fw = {r.framework: r for r in records}
        cond_pack = build_pack(by_fw["cond"])
        msml_pack = build_pack(by_fw["msml"])

        e = cond_pack["experiments"]
        assert e["scored"] == 3 and e["total"] == 3
        assert e["direction"] == "minimize" and e["metric_key"] == "val_bpb"
        assert e["best"]["value"] == pytest.approx(0.8)
        assert e["improvements"] == 3  # monotonically improving
        assert list(e["validation_census"]) == [
            "validation_content_sha256:aa11"]

        ev = cond_pack["events"]
        assert ev["api_calls"] == 1
        assert ev["tokens"]["input"] == 1000
        assert ev["api_request_lines"] == 1  # counted, never parsed
        assert "phase1" in ev["phase_windows"] and "phase3" in ev["phase_windows"]

        # Tool-failure marker only fires on the msml side.
        assert not cond_pack["events"]["tool_failures"]
        assert msml_pack["events"]["tool_failures"] == {"memory_store": 1}

        # Realization detection: the msml exp_0 dir carries a failed launch
        # 4000s before started_at plus the real one — two realizations,
        # fix_attempts=0 => untracked; classified launch signature captured.
        me = msml_pack["experiments"]
        assert me["multi_realization_experiments"] == 1
        assert me["untracked_multi_realizations"] == 1
        assert me["primary_mutations_after_finish"] == 0
        assert any("data_root" in s.lower()
                   for s in me["launch_failure_signatures"])
        assert cond_pack["experiments"]["multi_realization_experiments"] == 0

        # Framework-specific sections.
        assert cond_pack["conductor"]["decisions_total"] == 1
        assert cond_pack["conductor"]["annotations"] == {"champion": 1}
        assert msml_pack["conductor"] is None

        rl = cond_pack["run_log"]
        assert rl["final_exit_code"] == 0
        assert rl["http_endpoints"]["embeddings"]["rate_limited"] == 1

        roles = cond_pack["agent_logs"]["roles"]
        assert roles["strategist"]["api_calls"] == 1
        assert roles["strategist"]["prompt_chars"] == [
            len("You are the strategist.")]
        # who sat in each seat, observed from the run's own records
        assert roles["strategist"]["models"] == {"m-main": 1}
        # the gz transcript is read, and verifier_worker_* is a verifier seat
        assert roles["verifier"]["api_calls"] == 1
        assert roles["verifier"]["models"] == {"m-main": 1}
        assert "worker" not in roles
        seats = cond_pack["seats"]["seats"]
        assert seats["conductor"] == {"anthropic:opus-x": 1}
        assert seats["strategist"] == {"chat:m-main": 1}
        assert msml_pack["seats"] is None  # msml keeps no token ledger
        # declared pins captured from config, held against the observed seats
        assert cond_pack["config"]["seat_pins"] == {
            "conductor_model": "opus-x", "conductor_provider": "anthropic"}


class TestContractRuns:
    """External harnesses: evidence-contract layout as a first-class run
    kind — the reviewer must take output from many harnesses, not two."""

    def test_detection_pack_and_alias(self, tmp_path):
        root = tmp_path / "ext"
        run = _make_contract_run(root, era="d2_speedrun")
        records = build_registry(root)
        assert len(records) == 1
        r = records[0]
        assert r.framework == "extbench"
        assert r.domain == "domain2"          # token in the era dir name
        assert r.completeness == "complete" and r.db_rows == 2
        assert r.workspace == str(run) and r.db_path is None
        assert "contract-layout" in r.framework_evidence
        # one harness, inconsistent run-dir stamps -> one unified label
        assert build_registry(
            root, aliases={"extbench": "harnessx"})[0].framework == "harnessx"

        pack = build_pack(r)
        e = pack["experiments"]
        # metric name + direction come from the lineup, never the harness
        assert e["metric_key"] == "val_bpb" and e["direction"] == "minimize"
        assert e["total"] == 2 and e["scored"] == 2
        assert e["best"]["value"] == pytest.approx(1.05)
        assert e["improvements"] == 2         # mtime order, both improved
        # slice identity flows into the validation census
        assert "val_slice_id:s:abc" in e["validation_census"]
        # the harness's own "approach" text is the hypothesis
        assert e["experiments"][0]["hypothesis"].startswith("approach")
        assert (e["code_metrics"] or {}).get("experiments_with_code") == 2
        # honest absence: no transcripts, no ledger, no meta
        assert pack["seats"] is None and pack["conductor"] is None

    def test_bpb_slice_identity_gate(self, synthetic_corpus, tmp_path):
        """A bpb run measured on foreign frozen slices is referee-verified
        but NOT rankable against runs on the task's own split."""
        from alpha_lab.benchmarks.runcmp.referee import run_referee

        _make_contract_run(synthetic_corpus, era="d2_ext")
        out = tmp_path / "o"
        corpus_json, packs = out / "corpus.json", out / "packs"
        corpus_mod.main(["--root", str(synthetic_corpus),
                         "--out", str(corpus_json)])
        extract_mod.main(["--corpus", str(corpus_json), "--out", str(packs),
                          "--workers", "1"])
        res = run_referee(corpus_json, packs, out / "referee.json", [])
        runs = res["runs"]
        ext = next(v for v in runs.values() if v["framework"] == "extbench")
        ours = [v for v in runs.values() if v["framework"] in ("cond", "msml")]
        # in-house runs share one validation identity -> dominant, rankable
        assert all(v["rankable"] for v in ours)
        assert ext["rankable"] is False
        assert "not rankable across runs" in ext["rankable_reason"]
        # the foreign summed-nll spelling (val_loss_nats) recomputes via
        # nll / (bytes * ln 2) and reproduces the claim
        rows = [r for r in (ext.get("experiments") or [])
                if r.get("status") == "scored"]
        assert rows and all(r["self_report_reproduced"] for r in rows)

    def test_metric_key_aliases(self):
        from alpha_lab.benchmarks.runcmp.extract import _contract_metric_value
        assert _contract_metric_value(
            {"overall_rmse": 1.0, "per_origin_rmse": [1.0]}, "rmse"
        ) == (1.0, "overall_rmse")
        assert _contract_metric_value(
            {"holdout_log_loss": 0.4, "n_predictions": 5}, "logloss"
        ) == (0.4, "holdout_log_loss")
        assert _contract_metric_value({"val_bpb": 1.1}, "val_bpb") == (
            1.1, "val_bpb")
        assert _contract_metric_value({"unrelated": 1.0}, "rmse") == (
            None, None)


class TestRecordTimeValidation:
    def test_bad_reference_bounces(self, synthetic_corpus, tmp_path):
        from alpha_lab.benchmarks.runcmp.investigate import InvestigatorSession

        out = tmp_path / "out"
        corpus_json = out / "corpus.json"
        packs = out / "packs"
        corpus_mod.main(["--root", str(synthetic_corpus),
                         "--out", str(corpus_json)])
        extract_mod.main(["--corpus", str(corpus_json), "--out", str(packs),
                          "--workers", "1"])
        session = InvestigatorSession(corpus_json, packs, out)
        label = next(iter(session.records))
        bad = session.record_finding(
            claim="x", scope="run",
            evidence=[{"pack": label, "path": "experiments.best.value",
                       "value": 123.0}])
        assert bad.startswith("[ERROR] finding NOT recorded")
        assert not session.findings_path.exists()
        good = session.record_finding(
            claim="x", scope="run",
            evidence=[{"pack": label, "path": "experiments.best.value",
                       "value": 0.8}])
        assert "verified" in good and session.findings == 1


class TestLeaderboardDirection:
    def test_minimize_orders_ascending_and_missing_sorts_last(self, tmp_path):
        from alpha_lab.experiment_db import ExperimentDB

        db = ExperimentDB(str(tmp_path / "e.db"))
        for name, results in (
            ("worst", '{"val_bpb": 2.0}'),
            ("best", '{"val_bpb": 0.5}'),
            ("mid", '{"val_bpb": 1.0}'),
            ("missing", '{"other": 1}'),
        ):
            exp_id = db.create(name=name, description="", hypothesis="",
                               config_json="{}")
            db.set_results(exp_id, results)
        lb_min = db.leaderboard("val_bpb", 10, direction="minimize")
        assert [e.name for e in lb_min] == ["best", "mid", "worst", "missing"]
        lb_max = db.leaderboard("val_bpb", 10, direction="maximize")
        assert [e.name for e in lb_max] == ["worst", "mid", "best", "missing"]


class TestBench:
    def test_metrics_rules_and_render(self, synthetic_corpus, tmp_path):
        from alpha_lab.benchmarks.runcmp import bench as bench_mod

        out = tmp_path / "out"
        corpus_json = out / "corpus.json"
        packs = out / "packs"
        # a third, external harness rides the same chain end to end
        _make_contract_run(synthetic_corpus, era="d2_ext")
        corpus_mod.main(["--root", str(synthetic_corpus),
                         "--out", str(corpus_json)])
        extract_mod.main(["--corpus", str(corpus_json), "--out", str(packs),
                          "--workers", "1"])
        rules = (
            __import__("pathlib").Path(bench_mod.__file__).parent / "rules.json"
        )
        result = bench_mod.run_bench(corpus_json, packs, out, rules, None)
        assert len(result["runs"]) == 3
        ext = next(r for r in result["runs"].values()
                   if r["framework"] == "extbench")
        assert ext["metrics"]["lifecycle.scored_fraction"] == 1.0
        assert ext["metrics"]["governance.decisions"] is None  # no meta layer
        cond = next(r for r in result["runs"].values()
                    if r["framework"] == "cond")
        msml = next(r for r in result["runs"].values()
                    if r["framework"] == "msml")
        assert cond["metrics"]["lifecycle.scored_fraction"] == 1.0
        assert cond["metrics"]["memory.failure_rate"] == 0.0
        assert msml["metrics"]["memory.failure_rate"] == 1.0
        assert cond["metrics"]["http.rate_limited"] == 1
        # No metric raised, no unknown diagnostic classes.
        assert not [k for r in result["runs"].values()
                    for k in r["metrics"] if k.endswith("__error__")]
        classes = {d["class"] for r in result["runs"].values()
                   for d in r["diagnostics"]}
        assert "unknown" not in classes
        assert (out / "bench.md").is_file()

    def test_classify_rules(self):
        from alpha_lab.benchmarks.runcmp.bench import classify, load_rules
        from pathlib import Path as P
        from alpha_lab.benchmarks.runcmp import bench as bench_mod

        rules = load_rules(P(bench_mod.__file__).parent / "rules.json")
        assert classify("memory_store",
                        "[ERROR] OperationalError: database is locked",
                        rules)[0] == "bug"
        assert classify("read_file", "[ERROR] File not found: <path>",
                        rules)[0] == "design"
        assert classify("memory_store",
                        "[ERROR] ValueError: memory too similar to #3 "
                        "(similarity=1.0)", rules)[0] == "policy"


class TestReferee:
    def test_truth_pool_and_cross_compare(self, tmp_path):
        np = pytest.importorskip("numpy")
        from alpha_lab.benchmarks.runcmp import referee as ref

        rng = np.random.default_rng(0)
        truth = rng.normal(size=(4, 3, 2)).astype("float32")
        origins = np.array([10, 20, 30, 40])
        # left: predictions + targets (full protocol)
        left = tmp_path / "left"
        (left / "experiments/e1/results").mkdir(parents=True)
        np.savez(left / "experiments/e1/results/predictions.npz",
                 predictions=truth + 0.1, targets=truth, origins=origins)
        # right: forecast-only file on a subset protocol (truth from pool)
        right = tmp_path / "right"
        (right / "experiments/e2/results").mkdir(parents=True)
        np.savez(right / "experiments/e2/results/forecasts_by_origin.npz",
                 stack_predictions=truth[1:] + 0.05, origins=origins[1:])
        spec = ref.DOMAIN_SPECS["domain4"]
        lart = ref.RunArtifacts(left, spec)
        rart = ref.RunArtifacts(right, spec)
        pool, conflicts = ref.build_truth_pool(lart.items + rart.items)
        assert conflicts == 0 and len(pool) == 4
        t, source = ref.resolve_truth(rart.items[0], pool)
        assert source == "pool" and t.shape == (3, 3, 2)
        lrows = ref.score_items(lart.items, pool, {})
        rrows = ref.score_items(rart.items, pool, {})
        assert lrows[0]["array_scores"]["predictions"] == pytest.approx(0.1, rel=1e-3)
        assert rrows[0]["array_scores"]["stack_predictions"] == pytest.approx(
            0.05, rel=1e-3)

        class Rec:
            label = "L"
        class Rec2:
            label = "R"
        old_min = ref.MIN_SHARED_ORIGINS
        ref.MIN_SHARED_ORIGINS = 2
        try:
            cmp_result = ref.cross_compare(
                "t", (Rec, lart.items, lrows), (Rec2, rart.items, rrows),
                pool, spec)
        finally:
            ref.MIN_SHARED_ORIGINS = old_min
        assert cmp_result["comparable"] and cmp_result["shared_origins"] == 3
        assert cmp_result["winner"] == "right"  # 0.05 < 0.1 rmse


class TestTabulateAndFactcheck:
    def test_end_to_end(self, synthetic_corpus, tmp_path, capsys):
        out = tmp_path / "out"
        corpus_json = out / "corpus.json"
        packs = out / "packs"
        corpus_mod.main(["--root", str(synthetic_corpus),
                         "--out", str(corpus_json)])
        extract_mod.main(["--corpus", str(corpus_json), "--out", str(packs),
                          "--workers", "1"])
        tabulate_mod.main(["--corpus", str(corpus_json), "--packs", str(packs),
                           "--out", str(out)])
        tables = json.loads((out / "tables.json").read_text())
        assert len(tables["pairs"]) == 1
        pair = tables["pairs"][0]
        # Sides carry the labels; the shared validation identity is detected.
        assert pair["shared_validation"]["shared"] == [
            "validation_content_sha256:aa11"]
        assert pair["rows"]["memory tool failures"]["right"] == 1

        # Fact-check: one good finding, one bad.
        findings = out / "findings.jsonl"
        good = {"id": 1, "claim": "ok", "scope": "pair", "evidence": [
            {"table": pair["pair"], "row": "memory tool failures",
             "side": "right", "value": 1},
            {"pack": pair["left"], "path": "experiments.best.value",
             "value": 0.8},
        ]}
        bad = {"id": 2, "claim": "nope", "scope": "pair", "evidence": [
            {"pack": pair["left"], "path": "experiments.best.value",
             "value": 0.123},
        ]}
        with open(findings, "w") as fh:
            fh.write(json.dumps(good) + "\n")
            fh.write(json.dumps(bad) + "\n")
        # A REPORT.md must be present so the chart-census path executes too
        # (a NameError hid there for a day because no test wrote one).
        (out / "REPORT.md").write_text(
            "# r\n\n```chart\ntype: bars\ntitle: t\na | 1\n```\n\n"
            "```chart\ntype: stacked-bar\nseries:\n  - name: a\n```\n")
        factcheck_mod.main(["--out", str(out), "--packs", str(packs)])
        verification = json.loads((out / "verification.json").read_text())
        assert verification["verified"] == 1
        assert verification["failed"] == 1
        statuses = {r["id"]: r["status"] for r in verification["findings"]}
        assert statuses == {1: "verified", 2: "failed"}
        printed = capsys.readouterr().out
        assert "CHART RENDER FAILURES: 1 of 2" in printed


class TestProgressionSeriesAudit:
    """Trajectory-fidelity gate (2026-08-03): x,value chart series must be
    quotable from admissible sources at their printed precision; plain value
    lists (stated-arithmetic transforms) stay exempt."""

    CLEAN = ("series glm: 1,0.048279684825933734; 2,0.022897819928573867; "
             "3,0.0228510359728744; 4,0.0635613249048674; "
             "5,0.05360599529719593; 6,0.023215202747178577\n")

    @staticmethod
    def _md(payload):
        return "```chart\ntitle: t\ntype: line\n" + payload + "\n```\n"

    def test_faithful_rounded_series_passes(self):
        md = self._md("series glm: 1,0.048280; 2,0.022898; 3,0.022851; "
                      "4,0.063561; 5,0.053606; 6,0.023215")
        audit = factcheck_mod.audit_progression_series(md, "", self.CLEAN)
        assert audit["series_checked"] == 1
        assert audit["series_flagged"] == 0

    def test_fabricated_series_flagged(self):
        md = self._md("series glm: 1,0.026001; 2,0.026001; 3,0.024117; "
                      "4,0.025913; 5,0.021144; 6,0.027772")
        audit = factcheck_mod.audit_progression_series(md, "", self.CLEAN)
        assert audit["series_flagged"] == 1
        assert audit["flags"][0]["unmatched"] >= 4

    def test_plain_value_list_exempt(self):
        md = self._md("series kb: 66.1, 155.2, 120.9, 77.4, 102.3, 61.5")
        audit = factcheck_mod.audit_progression_series(md, "", self.CLEAN)
        assert audit["series_checked"] == 0

    def test_single_derived_value_tolerated(self):
        md = self._md("series glm: 1,0.096559; 2,0.022898; 3,0.022851; "
                      "4,0.063561; 5,0.053606; 6,0.023215")
        audit = factcheck_mod.audit_progression_series(md, "", self.CLEAN)
        assert audit["series_flagged"] == 0


class TestRefereeAttributionAudit:
    """Verbatim-attribution gate (2026-08-03): names presented as referee
    selections must be exact members of referee.json's best set."""

    REFEREE = {"pairs": [{
        "pair": "d4_x",
        "best": {"left": {"experiment": "real_champion_left"},
                 "right": {"experiment": "real_champion_right"}},
        "leaderboard": [{"experiment": "non_champion_row"}],
    }]}

    def test_correct_names_pass(self):
        md = ("| Model | referee-selected champion |\n|---|---|\n"
              "| X | `real_champion_left` |\n| Y | `real_champion_right` |\n")
        audit = factcheck_mod.audit_referee_attributions(md, self.REFEREE)
        assert audit["names_checked"] == 2
        assert audit["violations"] == []

    def test_retyped_and_wrong_row_names_flagged(self):
        md = ("| Model | referee-selected champion |\n|---|---|\n"
              "| X | `invented_name_v2` |\n| Y | `non_champion_row` |\n")
        audit = factcheck_mod.audit_referee_attributions(md, self.REFEREE)
        assert len(audit["violations"]) == 2

    def test_unrelated_tables_exempt(self):
        md = ("| Experiment | lines |\n|---|---|\n| `whatever_name` | 42 |\n")
        audit = factcheck_mod.audit_referee_attributions(md, self.REFEREE)
        assert audit["names_checked"] == 0

    def test_substituted_value_flagged(self):
        referee = {"pairs": [{
            "pair": "d6_x",
            "best": {"left": {"experiment": "real_champion_left",
                              "referee_score": 186.284},
                     "right": {"experiment": "real_champion_right",
                               "referee_score": 185.64}},
        }]}
        good = ("| Model | referee-selected champion |\n|---|---|\n"
                "| X | `real_champion_left` 186.284 |\n"
                "| Y | `real_champion_right` 185.640 |\n")
        audit = factcheck_mod.audit_referee_attributions(good, referee)
        assert audit["violations"] == []
        bad = ("| Model | referee-selected champion |\n|---|---|\n"
               "| X | `real_champion_left` 197.788 raw artifact |\n")
        audit = factcheck_mod.audit_referee_attributions(bad, referee)
        assert [v["where"] for v in audit["violations"]] == [
            "missing-referee-value"]


class TestLineup:
    def test_lineup_loads_and_validates(self):
        from alpha_lab.benchmarks.runcmp.lineup import load_lineup

        entries = load_lineup()
        ids = [e["id"] for e in entries]
        assert "d5_rfq" in ids and "d6_cuda" in ids
        assert len(ids) == len(set(ids))

    def test_detect_domain_tokens(self):
        from alpha_lab.benchmarks.runcmp.lineup import detect_domain

        assert detect_domain("camp/d5_rfq/d5_rfq_glm_cond/ws") == "d5_rfq"
        assert detect_domain("camp/d6_cuda/d6_cuda_o48_msml/ws") == "d6_cuda"
        assert detect_domain("camp/unrelated/thing") is None

    def test_corpus_detect_prefers_legacy_then_lineup(self):
        assert corpus_mod._detect_domain("era/domain2/cond/ws") == "domain2"
        assert corpus_mod._detect_domain("era/d5_rfq_sol_msml/ws") == "d5_rfq"

    def test_referee_specs_merge(self):
        from alpha_lab.benchmarks.runcmp.referee import DOMAIN_SPECS

        assert DOMAIN_SPECS["d5_rfq"]["kind"] == "classification_table"
        assert DOMAIN_SPECS["d6_cuda"]["kind"] == "kernel_bench"
        # builtin entries stay authoritative
        assert DOMAIN_SPECS["domain2"]["kind"] == "bpb_curve"

    def test_emit_configs_one_model_every_seat(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.lineup import emit_configs

        written = emit_configs(
            tmp_path, ["d5_rfq", "d6_cuda"],
            [("glm", "glm", "glm-5.2"), ("o48", "bedrock", "claude-opus-4-8")],
            ["cond", "msml"])
        assert len(written) == 8  # 2 domains x 2 models x 2 frameworks
        cfg = json.loads((tmp_path / "d5_rfq" / "d5_rfq_glm_cond"
                          / "config.json").read_text())
        assert cfg["model"] == "glm-5.2"
        assert cfg["conductor_provider"] == "glm"
        assert cfg["conductor_model"] == "glm-5.2"
        assert cfg["domain"] == "time_series"
        mcfg = json.loads((tmp_path / "d5_rfq" / "d5_rfq_glm_msml"
                           / "config.json").read_text())
        assert mcfg["domain"] == "tabular_classification"
        assert "conductor_model" not in mcfg
        assert (tmp_path / "LINEUP.md").is_file()


class TestClassificationReferee:
    def _spec(self, truth_path):
        return {
            "kind": "classification_table",
            "pred_globs": ["experiments/*/results/referee_predictions.parquet"],
            "id_col": "rfq_id", "prob_col": "p_win",
            "truth": {"path": str(truth_path), "id_col": "rfq_id",
                      "label_col": "status", "positive": "W",
                      "holdout_col": "date", "holdout_from": "2025-01-01"},
            "min_coverage": 0.95, "metric": "logloss",
            "lower_is_better": True,
        }

    def test_recompute_and_coverage_gate(self, tmp_path):
        pd = pytest.importorskip("pandas")
        np = pytest.importorskip("numpy")
        from alpha_lab.benchmarks.runcmp import referee as referee_mod

        truth = tmp_path / "rfqs.parquet"
        n = 40
        pd.DataFrame({
            "rfq_id": range(n),
            "status": ["W" if i % 4 == 0 else "L" for i in range(n)],
            "date": ["2024-06-01"] * 10 + ["2025-06-01"] * (n - 10),
        }).to_parquet(truth)
        ws = tmp_path / "ws"
        good = ws / "experiments" / "e1" / "results"
        good.mkdir(parents=True)
        holdout_ids = list(range(10, n))
        probs = [0.9 if i % 4 == 0 else 0.1 for i in holdout_ids]
        pd.DataFrame({"rfq_id": holdout_ids, "p_win": probs}).to_parquet(
            good / "referee_predictions.parquet")
        partial = ws / "experiments" / "e2" / "results"
        partial.mkdir(parents=True)
        pd.DataFrame({"rfq_id": holdout_ids[:5],
                      "p_win": [0.5] * 5}).to_parquet(
            partial / "referee_predictions.parquet")

        # A diagnostic artifact directory that is NOT on the run's board:
        # must never be scored (measured 2026-08-08: a labels-as-predictions
        # `perfect_foresight_probe` topped the published d5 leaderboard
        # at logloss 1e-06).
        probe = ws / "experiments" / "perfect_probe" / "results"
        probe.mkdir(parents=True)
        pd.DataFrame({"rfq_id": holdout_ids,
                      "p_win": [1.0 if i % 4 == 0 else 0.0
                                for i in holdout_ids]}).to_parquet(
            probe / "referee_predictions.parquet")

        referee_mod._TRUTH_CACHE.clear()
        expected = float(-np.mean(
            [np.log(0.9) if i % 4 == 0 else np.log(0.9) for i in holdout_ids]))
        v, board = referee_mod.classification_verification(
            ws, self._spec(truth),
            {"e1": {"metric": expected}, "e2": {}}, scored=2)
        rows = {r["experiment"]: r for r in v["experiments"]}
        assert rows["e1"]["status"] == "scored"
        assert abs(rows["e1"]["recomputed"] - expected) < 1e-6
        assert rows["e1"]["self_report_reproduced"]
        assert rows["e1"]["auc"] == 1.0
        assert rows["e2"]["status"] == "partial_coverage"
        assert "perfect_probe" not in rows
        assert v["non_board_artifacts_excluded"] == ["perfect_probe"]
        assert [b["experiment"] for b in board] == ["e1"]


class TestSharedTruthPairWinner:
    """shared_truth_pair promises cross_compare's shape; until 2026-08-03 it
    shipped without 'winner' and every d5/d6 pair verdict read None."""

    def _pair(self, lower, lb, rb):
        from types import SimpleNamespace
        from alpha_lab.benchmarks.runcmp import referee as referee_mod
        return referee_mod.shared_truth_pair(
            "p", {"metric": "m", "lower_is_better": lower},
            SimpleNamespace(label="r/d/cond/a"),
            SimpleNamespace(label="r/d/msml/b"), lb, rb)

    def test_winner_lower_is_better(self):
        p = self._pair(True, [{"experiment": "x", "referee_score": 0.30}],
                       [{"experiment": "y", "referee_score": 0.35}])
        assert p["winner"] == "left"

    def test_winner_higher_is_better(self):
        p = self._pair(False, [{"experiment": "x", "referee_score": 185.6}],
                       [{"experiment": "y", "referee_score": 186.3}])
        assert p["winner"] == "right"

    def test_no_verdict_with_empty_side(self):
        p = self._pair(True, [{"experiment": "x", "referee_score": 0.3}], [])
        assert p["comparable"] is False and p["winner"] is None


class TestLedgerPairSums:
    """A number that is the exact sum of two token-ledger values is an
    honest derivation ('conductor+verifier consumed N input tokens'), not
    unsourced — but ONLY ledger values may be summed; pair sums over all
    sources would launder invented figures (2026-08-08)."""

    def test_pair_sum_classified_and_nonledger_stays_unsourced(
            self, tmp_path):
        out = tmp_path / "battery"
        packs = out / "packs"
        packs.mkdir(parents=True)
        (out / "token_accounting.json").write_text(json.dumps(
            {"seats": [{"log": "conductor", "input_tokens": 4000000},
                       {"log": "verifier", "input_tokens": 567931}]}))
        rep = out / "rev"
        rep.mkdir()
        (rep / "REPORT.md").write_text(
            "Conductor+verifier consumed **4,567,931 input tokens**.\n"
            "An invented figure: **7,777,777 tokens** appears nowhere.\n")
        audit = factcheck_mod.audit_report_numbers(rep, packs)
        assert audit["counts"]["ledger_pair_sum"] == 1
        assert audit["counts"]["unsourced"] == 1
        assert audit["unsourced"][0]["value"] == "7777777"


class TestFactcheckPackLookup:
    """_pack must attempt the read (EAFP), not gate on is_file(): a
    transient filesystem miss on 18-hour-old packs downgraded 8 findings
    to partial on a phantom 'no pack' (2026-08-08 16:45)."""

    def test_existing_pack_loads_and_missing_returns_none(self, tmp_path,
                                                          monkeypatch):
        packs = tmp_path / "packs"
        packs.mkdir()
        (packs / "era__d__fw__run.json").write_text('{"x": 1}')
        monkeypatch.setattr(factcheck_mod.time, "sleep", lambda s: None)
        fc = factcheck_mod.FactChecker(tmp_path, packs)
        assert fc._pack("era/d/fw/run") == {"x": 1}
        assert fc._pack("era/d/fw/ghost") is None


class TestRefuseNoBoard:
    """A workspace run with no experiment board (empty/missing db) is a
    broken run — the referee refuses its artifacts instead of scoring
    them (five renamed-aside dead attempts were scored that way
    2026-08-08). Contract runs have no db by design and are exempt."""

    def _rec(self, evidence=""):
        from types import SimpleNamespace
        return SimpleNamespace(label="e/d/msml/x", framework="msml",
                               domain="d7_payup",
                               framework_evidence=evidence)

    def test_workspace_run_without_board_is_refused(self):
        from alpha_lab.benchmarks.runcmp.referee import refuse_no_board
        r = refuse_no_board(self._rec(), {}, "regression_table")
        assert r is not None
        assert r["refused_no_board"] is True
        assert r["rankable"] is False
        assert r["experiments"] == []

    def test_contract_run_without_board_is_exempt(self):
        from alpha_lab.benchmarks.runcmp.referee import refuse_no_board
        rec = self._rec("contract-layout: 3/3 experiment dirs with "
                        "parseable metrics.json (counts are dirs, no db)")
        assert refuse_no_board(rec, {}, "regression_table") is None

    def test_workspace_run_with_board_is_scored(self):
        from alpha_lab.benchmarks.runcmp.referee import refuse_no_board
        assert refuse_no_board(self._rec(), {"e1": {}}, None) is None


class TestKernelReferee:
    SPEC = {
        "kind": "kernel_bench",
        "pred_globs": ["experiments/*/results/kernel_report.json"],
        "flops_per_call": 2 * 4096 ** 3, "atol": 0.02, "rtol": 0.02,
        "expected_fingerprint": {"op": "gemm_bias_gelu", "m": 4096,
                                 "n": 4096, "k": 4096, "dtype": "fp16"},
        "min_samples": 50, "metric": "tflops", "lower_is_better": False,
    }

    def _report(self, ws, name, **over):
        rep = {
            "fingerprint": {"op": "gemm_bias_gelu", "m": 4096, "n": 4096,
                            "k": 4096, "dtype": "fp16"},
            "correctness": {"err_norm": 0.064, "max_abs_err": 0.2421,
                            "atol": 0.02, "rtol": 0.02},
            "timing": {"warmup_iters": 50, "timed_iters": 200,
                       "samples_ms": [1.0] * 200},
            "tflops": 137.439,
        }
        rep.update(over)
        d = ws / "experiments" / name / "results"
        d.mkdir(parents=True)
        (d / "kernel_report.json").write_text(json.dumps(rep))

    def test_correctness_gate_and_recompute(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.referee import kernel_verification

        ws = tmp_path / "ws"
        self._report(ws, "ok")
        self._report(ws, "wrong",
                     correctness={"err_norm": 48.0, "max_abs_err": 1.1,
                                  "atol": 0.02, "rtol": 0.02})
        self._report(ws, "loosened",
                     correctness={"err_norm": 0.9, "max_abs_err": 1.1,
                                  "atol": 0.5, "rtol": 0.5})
        self._report(ws, "thin",
                     timing={"warmup_iters": 50, "timed_iters": 10,
                             "samples_ms": [1.0] * 10})
        self._report(ws, "othertask",
                     fingerprint={"op": "softmax", "m": 4096, "n": 4096,
                                  "k": 4096, "dtype": "fp16"})
        v, board = kernel_verification(
            ws, dict(self.SPEC), {"ok": {"metric": 137.4}}, scored=5)
        rows = {r["experiment"]: r for r in v["experiments"]}
        assert rows["ok"]["status"] == "scored"
        # 2*4096^3 flops / 1ms -> 137.439 TFLOP/s
        assert abs(rows["ok"]["recomputed"] - 137.439) < 0.001
        assert rows["ok"]["self_report_reproduced"]
        assert rows["wrong"]["status"] == "correctness_failed"
        assert rows["loosened"]["status"] == "correctness_failed"
        assert rows["thin"]["status"] == "insufficient_timing_samples"
        assert rows["othertask"]["status"] == "fingerprint_mismatch"
        assert [b["experiment"] for b in board] == ["ok"]


class TestRegressionReferee:
    def _spec(self, truth_path):
        return {
            "kind": "regression_table",
            "pred_globs": ["experiments/*/results/referee_predictions.parquet"],
            "pred_col": "pred",
            "truth": {"path": str(truth_path), "id_cols": ["Date", "bid_id", "cusip"],
                      "target_col": "payup", "weight_col": "curr_face",
                      "holdout_col": "Date", "holdout_from": "2026-05-01",
                      "holdout_to": "2026-07-31"},
            "min_coverage": 0.95, "metric": "weighted_mae",
            "lower_is_better": True,
        }

    def test_weighted_mae_and_coverage(self, tmp_path):
        pd = pytest.importorskip("pandas")
        from alpha_lab.benchmarks.runcmp import referee as referee_mod

        truth = tmp_path / "dump.csv"
        pd.DataFrame({
            "Date": ["2026-04-01"] * 2 + ["2026-06-01"] * 5,
            "bid_id": [1, 2, 3, 4, 5, 6, 6],
            "cusip": ["A", "B", "C", "D", "E", "F", "F"],
            "payup": [9.0, 9.0, 10.0, 20.0, 30.0, 40.0, 42.0],
            "curr_face": [1e6, 1e6, 1e6, 1e6, 2e6, 1e6, 1e6],
        }).to_csv(truth, index=False)
        ws = tmp_path / "ws"
        good = ws / "experiments" / "e1" / "results"; good.mkdir(parents=True)
        pd.DataFrame({"Date": ["2026-06-01"] * 4, "bid_id": [3, 4, 5, 6],
                      "cusip": ["C", "D", "E", "F"],
                      "pred": [11.0, 20.0, 30.0, 45.0]}).to_parquet(
            good / "referee_predictions.parquet")
        part = ws / "experiments" / "e2" / "results"; part.mkdir(parents=True)
        pd.DataFrame({"Date": ["2026-06-01"], "bid_id": [3], "cusip": ["C"],
                      "pred": [10.0]}).to_parquet(part / "referee_predictions.parquet")

        referee_mod._REG_TRUTH_CACHE.clear()
        # key (6,F) duplicated: pred 45 covers rows payup 40 (w 1e6) and 42 (w 1e6)
        # weighted MAE = (1e6*1 + 1e6*0 + 2e6*0 + 1e6*5 + 1e6*3) / 6e6 = 1.5
        v, board = referee_mod.regression_verification(
            ws, self._spec(truth), {"e1": {"metric": 1.5}, "e2": {}}, scored=2)
        rows = {r["experiment"]: r for r in v["experiments"]}
        assert v["holdout_rows"] == 5
        assert rows["e1"]["status"] == "scored"
        assert abs(rows["e1"]["recomputed"] - 1.5) < 1e-9
        assert rows["e1"]["self_report_reproduced"]
        assert rows["e2"]["status"] == "partial_coverage"
        assert [b["experiment"] for b in board] == ["e1"]


class TestPublish:
    """Exporter to the showcase MLflow store: idempotency, ledgers, open bag."""

    def _mini_campaign(self, tmp_path):
        out = tmp_path / "campaign"
        packs = out / "packs"
        packs.mkdir(parents=True)
        # two in-house harnesses plus one external contract harness — the
        # task page is ONE leaderboard across all of them
        labels = ["era1/dom1/cond/cellA", "era1/dom1/msml/cellB",
                  "ext1/dom1/extbench"]
        corpus = {"runs": [
            {"label": lab, "framework": lab.split("/")[2], "domain": "dom1",
             "era": lab.split("/")[0], "run_dir": str(tmp_path),
             "workspace": str(tmp_path),
             "completeness": "complete"} for lab in labels]}
        (out / "corpus.json").write_text(json.dumps(corpus))
        db = tmp_path / "experiments.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE experiments (id INTEGER, name TEXT, "
                    "status TEXT, started_at REAL, finished_at REAL)")
        con.executemany("INSERT INTO experiments VALUES (?,?,?,?,?)",
                        [(1, "e1", "done", 1000.0, 1600.0),
                         (2, "e2", "done", 1700.0, 2900.0)])
        con.commit()
        con.close()
        bench_runs, ref_runs = {}, {}
        for lab in labels:
            pack = {
                "run": {"label": lab},
                "config": {"provider": "openai", "model": "test-model",
                           "phase3": {"executor": "local"}},
                "experiments": {
                    "metric_key": "m", "direction": "minimize", "total": 2,
                    "scored": 2, "best": {"name": "e2", "value": 0.5, "id": 2},
                    "trajectory": [
                        {"n": 1, "name": "e1", "value": 1.0, "best_so_far": 1.0},
                        {"n": 2, "name": "e2", "value": 0.5, "best_so_far": 0.5}],
                    "code_metrics": {"samples": {"code_lines": [10, 20, 30]}},
                    "sources": [str(db)]},
                "events": {"phase_windows": {
                    "phase1": {"start": 900.0, "end": 1000.0}}},
                "agent_logs": {"samples": {"request_bytes": [100, 200, 300]}},
            }
            if lab.endswith("cellA"):
                # mixed seats via the token ledger: conductor on another model
                pack["seats"] = {"sources": ["x"], "seats": {
                    "conductor": {"anthropic:opus-x": 3},
                    "strategist": {"chat:test-model": 5}}}
            elif lab.endswith("cellB"):
                # no ledger: seats observed from the request logs' model field
                pack["agent_logs"]["roles"] = {
                    "strategist": {"models": {"test-model": 4}}}
            # extbench: contract run — no transcripts at all, seats absent
            (packs / (lab.replace("/", "__") + ".json")).write_text(
                json.dumps(pack))
            bench_runs[lab] = {
                "framework": lab.split("/")[2], "domain": "dom1",
                "metrics": {"lifecycle.wall_hours": 1.5,
                            "efficiency.cost_usd": 12.0,
                            "lifecycle.runlog_error_lines": None},
                "experiment_table": [
                    {"id": 1, "name": "e1", "status": "done", "metric": 1.0,
                     "duration_seconds": 10.0, "fix_attempts": 0, "error": None,
                     "tokens_attributed": 5, "est_cost_usd": 0.5},
                    {"id": 2, "name": "e2", "status": "done", "metric": 0.5,
                     "duration_seconds": 20.0, "fix_attempts": 1,
                     "error": "boom", "tokens_attributed": 7,
                     "est_cost_usd": 0.7}]}
            # the external cell scores worse so ranks span all three
            escore = 0.9 if lab.endswith("extbench") else 0.5
            ref_runs[lab] = {"label": lab, "scored_experiments": 2,
                             "artifact_coverage": 1.0,
                             "experiments": [
                                 {"experiment": "e1", "recomputed": 1.0},
                                 {"experiment": "e2",
                                  "official_referee_score": escore}]}
        (out / "bench.json").write_text(json.dumps(
            {"bench_version": "t1", "rules_version": "r1",
             "generated_at": "now", "metric_definitions": {},
             "runs": bench_runs}))
        (out / "referee.json").write_text(json.dumps(
            {"runs": ref_runs, "pairs": [
                {"pair": "p1", "left": labels[0], "right": labels[1],
                 "comparable": True, "metric": "m", "lower_is_better": True,
                 "best": {"left": {"referee_score": 0.5},
                          "right": {"referee_score": 0.5}},
                 "winner": "left", "leaderboard": []}]}))
        rep = out / "solo"
        rep.mkdir()
        (rep / "verification.json").write_text(json.dumps(
            {"total": 3, "verified": 3, "partial": 0, "failed": 0}))
        (rep / "REPORT.md").write_text("# report")
        return out, labels

    def test_publish_idempotent_and_ledgered(self, tmp_path):
        mlflow = pytest.importorskip("mlflow")
        pytest.importorskip("matplotlib")
        from alpha_lab.benchmarks.runcmp import publish as publish_mod

        out, labels = self._mini_campaign(tmp_path)
        store = tmp_path / "store"
        args = ["--corpus", str(out / "corpus.json"), "--packs",
                str(out / "packs"), "--bench", str(out / "bench.json"),
                "--referee", str(out / "referee.json"), "--out", str(out),
                "--store", str(store), "--campaign", "mini", "--owner", "t"]
        assert publish_mod.main(args) == 0
        assert publish_mod.main(args) == 0  # republish must not duplicate

        client = mlflow.MlflowClient(
            tracking_uri=f"sqlite:///{store}/mlflow.db")
        # one experiment per task, found by its task_domain tag
        tasks = client.search_experiments(
            filter_string="tags.task_domain = 'dom1'")
        assert len(tasks) == 1
        task = tasks[0]
        assert task.name == "task dom1 — m (lower wins)"
        runs = client.search_runs([task.experiment_id], max_results=50)
        assert len(runs) == 3
        by_key = {r.data.tags["cell_key"]: r for r in runs}
        assert set(by_key) == set(labels)
        # the external harness sits on the same leaderboard, ranked by the
        # same referee; its unrecorded aspects are absent, not zero
        rc = by_key[labels[2]]
        assert rc.data.params["framework"] == "extbench"
        assert "★3" in rc.info.run_name
        assert not any(k.startswith("seat") for k in rc.data.params)
        r = by_key[labels[0]]
        # sectioned metric names carry the chart layout; None metric skipped
        assert r.data.metrics["3 economy/det.lifecycle.wall_hours"] == 1.5
        assert not any("runlog_error_lines" in k for k in r.data.metrics)
        # referee kinds swept: recomputed AND official_referee_score
        assert r.data.metrics["1 verdict/referee.best_score"] == 0.5
        assert r.data.metrics["1 verdict/rel.rank_in_domain"] == 1.0
        # a generic det subsystem falls into its own section
        assert "search/det.search.scored" not in r.data.metrics  # not logged
        # open bag flattening
        assert r.data.params["config.phase3.executor"] == "local"
        # rollout + progress + quantile series under sectioned names
        hist = client.get_metric_history(
            r.info.run_id, "2 race/det.progress.best_so_far")
        assert [(m.step, m.value) for m in hist] == [(1, 1.0), (2, 0.5)]
        hist = client.get_metric_history(
            r.info.run_id, "rollout detail/det.rollout.failed")
        assert [(m.step, m.value) for m in hist] == [(1, 0.0), (2, 1.0)]
        hist = client.get_metric_history(
            r.info.run_id, "distributions/det.q.agent.request_bytes")
        assert {m.step for m in hist} == {5, 10, 25, 50, 75, 90, 95}
        # campaign report run: factcheck counters + grid charts attached
        ov = client.get_experiment_by_name("campaign reports")
        ov_runs = client.search_runs([ov.experiment_id], max_results=10)
        assert len(ov_runs) == 1
        assert ov_runs[0].data.metrics[
            "factcheck/inv.solo.findings_verified"] == 3.0
        charts = client.list_artifacts(ov_runs[0].info.run_id, "charts")
        assert any(a.path.endswith("01_coverage_map.png") for a in charts)
        # run name carries the referee rank; notes are plain markdown
        assert "★1" in r.info.run_name
        # champion written LAST so the naked table (Created desc) opens
        # champion-on-top without any stored arrangement
        assert (by_key[labels[0]].info.start_time
                >= by_key[labels[1]].info.start_time)
        note = r.data.tags["mlflow.note.content"]
        assert "<img" not in note
        assert ov_runs[0].info.run_id in note
        # who sat in each seat: routine benchmark variable, so it is stated
        # neutrally in the note, filterable via params, and drawn as the
        # leading "0 seats" chart section (title carries the model)
        assert r.data.params["seat.conductor"] == "opus-x via anthropic"
        assert r.data.params["seats.mixed"] == "yes"
        assert r.data.params["seats.models"] == "opus-x, test-model"
        assert "**seats** — " in note and "⚠" not in note
        assert r.data.metrics["0 seats/conductor: opus-x"] == 3.0
        assert r.data.metrics["0 seats/strategist: test-model"] == 5.0
        rb = by_key[labels[1]]
        assert rb.data.params["seat.strategist"] == "test-model"
        assert rb.data.params["seats.mixed"] == "no"
        assert "in every recorded seat" in rb.data.tags["mlflow.note.content"]
        assert rb.data.metrics["0 seats/strategist: test-model"] == 4.0
        ov_note = ov_runs[0].data.tags["mlflow.note.content"]
        assert "<img" not in ov_note and "storyboard" not in ov_note
        arts = {a.path for a in client.list_artifacts(ov_runs[0].info.run_id)}
        assert "storyboard.html" not in arts
        # the only stored state: one TABLE arrangement per task, with NO
        # chart keys (the chart side must keep MLflow's auto-draw-all)
        task_tags = client.get_experiment(task.experiment_id).tags
        standings = json.loads(task_tags["mlflow.sharedViewState.standings"])
        assert "compareRunCharts" not in standings
        assert standings["orderByKey"] == (
            "metrics.`1 verdict/referee.best_score`")
        assert standings["orderByAsc"] is True  # minimize domain
        # the seat lineup is a leaderboard column in the standings view
        assert "params.`seats.models`" in standings["selectedColumns"]
        # task tagline leads with the champion and links the standings
        task_note = task_tags["mlflow.note.content"]
        assert task_note.startswith("Champion (mini): cellA")
        assert "viewStateShareKey=standings" in task_note
        # without this tag the MLflow 3 UI opens the (empty) traces view
        for exp_id in (task.experiment_id, ov.experiment_id):
            kind = client.get_experiment(exp_id).tags["mlflow.experimentKind"]
            assert kind == "custom_model_development"

        # the harness face-off: ONE page, one run per harness, one chart
        # per task (pct behind that task's winner) — the cross-task view
        # that per-task pages cannot give
        fo = client.get_experiment_by_name(
            publish_mod.FACEOFF_EXPERIMENT)
        fo_runs = client.search_runs([fo.experiment_id], max_results=10)
        by_fw = {r.data.params["framework"]: r for r in fo_runs}
        assert set(by_fw) == {"cond", "msml", "extbench"}
        # cellA/cellB tie at 0.5 -> both 0.0 behind; extbench 0.9 -> 80%
        assert by_fw["extbench"].data.metrics[
            "1 verdict/pct behind winner: dom1"] == pytest.approx(80.0)
        assert by_fw["cond"].data.metrics[
            "1 verdict/mean pct behind task winners"] == pytest.approx(0.0)
        # specific metric in the task's own units, referee-scored
        assert by_fw["cond"].data.metrics[
            "1 verdict/best m: dom1"] == pytest.approx(0.5)
        # per-attempt race series from the winning run (the per-turn view)
        hist = client.get_metric_history(
            by_fw["cond"].info.run_id,
            "2 race/self-reported best m so far: dom1")
        assert [(m.step, m.value) for m in hist] == [(1, 1.0), (2, 0.5)]
        # economy/reliability from the winning run's recorded ledgers
        assert by_fw["cond"].data.metrics[
            "3 economy/llm cost usd: dom1"] == 12.0
        assert by_fw["extbench"].data.metrics[
            "4 reliability/runs fielded: dom1"] == 1.0
        assert "★" in by_fw["cond"].info.run_name
        assert fo.tags["mlflow.note.content"].startswith("Champion (mini):")

        # an unrankable referee verdict (different frozen slice) keeps the
        # run OFF the leaderboard: no referee best, no rank, note says why
        ref = json.loads((out / "referee.json").read_text())
        ref["runs"][labels[2]] = {
            "label": labels[2], "scored_experiments": 2,
            "artifact_coverage": 1.0, "rankable": False,
            "rankable_reason": "different frozen slice",
            "experiments": [{"experiment": "e2", "recomputed": 0.1}]}
        (out / "referee.json").write_text(json.dumps(ref))
        from alpha_lab.benchmarks.runcmp.publish import build_cells
        cells, _ = build_cells(str(out / "corpus.json"), str(out / "packs"),
                               str(out / "bench.json"),
                               str(out / "referee.json"))
        c3 = next(c for c in cells if c["label"] == labels[2])
        assert c3["referee_best"] is None and "rel" not in c3
        from alpha_lab.benchmarks.runcmp.publish import _note_for_cell
        note3 = _note_for_cell(c3, "mini", "", "")
        assert "not score-comparable" in note3

        # a later publish (comparison set grew) replaces rows by cell_key —
        # one row per harness run, never duplicates — and retires the
        # superseded campaign's leftovers and report
        args2 = list(args)
        args2[args2.index("--campaign") + 1] = "mini2"
        args2 += ["--retire-campaign", "mini"]
        assert publish_mod.main(args2) == 0
        runs2 = client.search_runs([task.experiment_id], max_results=50)
        assert len(runs2) == 3
        assert {r.data.tags["campaign"] for r in runs2} == {"mini2"}
        # campaign REPORT pages are never retired — they carry investigator
        # reports nothing supersedes; both campaigns' reports stay visible
        ov_runs2 = client.search_runs([ov.experiment_id], max_results=10)
        assert {r.data.tags["campaign"] for r in ov_runs2} == {"mini", "mini2"}


class TestGate:
    """The change gate: candidate vs baseline, policy-judged, CI-ready."""

    def _gate_inputs(self, tmp_path):
        """Corpus/bench/referee for two eras of one harness: baseline era1
        and candidate era2, one domain, one model."""
        corpus = {"runs": []}
        bench_runs, ref_runs = {}, {}
        for era, score, wall, fails in (("era1", 0.50, 2.0, 3),
                                        ("era2", 0.45, 1.5, 1)):
            lab = f"{era}/dom1/cond/cell"
            corpus["runs"].append({
                "label": lab, "framework": "cond", "domain": "dom1",
                "era": era, "run_dir": str(tmp_path),
                "workspace": str(tmp_path), "model": "m1",
                "events_path": None, "run_log_path": None, "db_path": None,
                "db_rows": 5, "db_terminal_rows": 5, "status_counts": {},
                "completeness": "complete", "framework_evidence": "",
                "run_state": "finished", "notes": "", "pair_key": ""})
            bench_runs[lab] = {
                "framework": "cond", "domain": "dom1", "direction": "minimize",
                "metrics": {"lifecycle.wall_hours": wall,
                            "efficiency.cost_usd": 10.0,
                            "reliability.execution_failures": fails,
                            "lifecycle.scored_fraction": 1.0,
                            "code.comment_share_overall": 0.1,
                            "search.time_to_best_hours": wall / 2}}
            ref_runs[lab] = {"label": lab, "rankable": True, "experiments": [
                {"experiment": "e", "recomputed": score}]}
        (tmp_path / "corpus.json").write_text(json.dumps(corpus))
        (tmp_path / "bench.json").write_text(json.dumps({"runs": bench_runs}))
        (tmp_path / "referee.json").write_text(json.dumps(
            {"runs": ref_runs, "pairs": []}))
        return tmp_path

    def test_pass_fail_and_exit_codes(self, tmp_path):
        from alpha_lab.benchmarks.runcmp import gate as gate_mod

        d = self._gate_inputs(tmp_path)
        args = ["--corpus", str(d / "corpus.json"),
                "--bench", str(d / "bench.json"),
                "--referee", str(d / "referee.json"),
                "--out", str(d / "g1")]
        # era2 improves the score (0.50 -> 0.45), wall, failures: PASS
        rc = gate_mod.main(args + ["--baseline", "era=era1",
                                   "--candidate", "era=era2"])
        assert rc == 0
        g = json.loads((d / "g1" / "gate.json").read_text())
        assert g["verdict"] == "PASS" and not g["violations"]
        assert any(v["id"] == "referee_best"
                   for v in g["improvements_shown"])
        md = (d / "g1" / "gate.md").read_text()
        # screen-not-verdict: the page leads with the evidence counts and
        # names the mechanical reading as a screen for CI
        assert md.startswith("# Change screen —")
        assert "a screen, not the decision" in md
        # the reviewers' mission is emitted alongside, demanding an
        # explicit argued position
        mission = (d / "g1" / "mission.md").read_text()
        assert "Recommendation: PASS" in mission
        assert "The change screen (deterministic" in mission

        # reversed: era1 as candidate is worse on the score -> guard fires
        rc = gate_mod.main(args[:-1] + [str(d / "g2"),
                                        "--baseline", "era=era2",
                                        "--candidate", "era=era1"])
        assert rc == 1
        g2 = json.loads((d / "g2" / "gate.json").read_text())
        assert g2["verdict"] == "FAIL"
        assert any(v["id"] == "referee_best" for v in g2["violations"])

    def test_no_improvement_fails_and_coverage_drop_fails(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.gate import evaluate, parse_selector
        from alpha_lab.benchmarks.runcmp.corpus import RunRecord

        d = self._gate_inputs(tmp_path)
        # equalize the candidate so nothing improves -> FAIL for
        # no-demonstrated-improvement
        bench = json.loads((d / "bench.json").read_text())
        ref = json.loads((d / "referee.json").read_text())
        b1, b2 = bench["runs"].values()
        b2["metrics"] = dict(b1["metrics"])
        r1, r2 = ref["runs"].values()
        r2["experiments"] = r1["experiments"]
        records = [RunRecord(**{**{f: "" for f in (
            "label", "framework", "domain", "era", "run_dir", "workspace")},
            **{k: v for k, v in row.items()
               if k in RunRecord.__dataclass_fields__},
            "events_path": None, "run_log_path": None, "db_path": None,
            "db_rows": 0, "db_terminal_rows": 5})
            for row in json.loads((d / "corpus.json").read_text())["runs"]]
        policy = json.loads(
            (__import__("pathlib").Path("src/alpha_lab/benchmarks/runcmp/"
                                        "gate_policy.json")).read_text())
        g = evaluate(records, bench, ref, policy,
                     parse_selector(["era=era1"]), parse_selector(["era=era2"]))
        assert g["verdict"] == "FAIL" and g.get("no_improvement")

        # candidate missing the cell entirely -> coverage violation
        g2 = evaluate(records, bench, ref, policy,
                      parse_selector(["era=era1"]),
                      parse_selector(["era=era1", "era=era2"]))
        assert g2["verdict"] != "PASS" or g2["cells"]

    def test_gate_publishes_evidence_run(self, tmp_path):
        mlflow = pytest.importorskip("mlflow")
        from alpha_lab.benchmarks.runcmp import gate as gate_mod
        from alpha_lab.benchmarks.runcmp import publish as publish_mod

        d = self._gate_inputs(tmp_path)
        assert gate_mod.main([
            "--corpus", str(d / "corpus.json"),
            "--bench", str(d / "bench.json"),
            "--referee", str(d / "referee.json"),
            "--baseline", "era=era1", "--candidate", "era=era2",
            "--out", str(d / "g")]) == 0
        client = mlflow.MlflowClient(
            tracking_uri=f"sqlite:///{d}/store/mlflow.db")
        (d / "store" / "artifacts").mkdir(parents=True)
        gdoc = json.loads((d / "g" / "gate.json").read_text())
        # an LLM reviewer ran over the gate mission and took a position
        rev = d / "g" / "review_1_test"
        rev.mkdir()
        (rev / "REPORT.md").write_text(
            "Recommendation: PASS (comfortable)\n\nBecause ...\n")
        exp_id, rid = publish_mod.publish_gate(
            client, gdoc, "gatecheck", "t", str(d / "store" / "artifacts"),
            gate_dir=d / "g")
        run = client.get_run(rid)
        # screen-not-verdict: the name carries counts, the mechanical
        # reading is a filterable parameter, reviewers' positions surface
        assert run.info.run_name.startswith("gate: 0 flag(s)")
        assert run.data.params["policy screen"] == "PASS"
        assert run.data.params["review.review_1_test"].startswith(
            "Recommendation: PASS")
        assert run.data.metrics["1 verdict/guard violations"] == 0.0
        assert run.data.metrics["1 verdict/improvements demonstrated"] >= 1.0
        assert any(k.startswith("2 guards/pct worse: final score")
                   for k in run.data.metrics)
        note = run.data.tags["mlflow.note.content"]
        assert note.startswith("# Change screen —")
        assert "Reviewer recommendations" in note
        arts = [a.path for a in client.list_artifacts(rid, "reviews")]
        assert arts == ["reviews/review_1_test"]
        # re-publishing the same gate replaces, never duplicates
        publish_mod.publish_gate(client, gdoc, "gatecheck", "t",
                                 str(d / "store" / "artifacts"),
                                 gate_dir=d / "g")
        runs = client.search_runs([exp_id], max_results=10)
        assert len(runs) == 1


class TestUsageTokenShapes:
    """Regression for the d7 corpus token gap (2026-08-05): raw Chat
    Completions usage (lab-gateway models) parsed to all zeros, and the raw
    Anthropic Messages dump lost cache writes."""

    def test_chat_completions_shape(self):
        from alpha_lab.benchmarks.runcmp.extract import _usage_tokens
        u = _usage_tokens({
            "prompt_tokens": 4752, "completion_tokens": 700,
            "total_tokens": 5452,
            "prompt_tokens_details": {"cached_tokens": 4000},
            "completion_tokens_details": {"reasoning_tokens": 120},
        })
        assert u == {"input": 4752, "output": 700, "cache_read": 4000,
                     "cache_write": 0, "reasoning": 120}

    def test_anthropic_messages_shape(self):
        from alpha_lab.benchmarks.runcmp.extract import _usage_tokens
        u = _usage_tokens({
            "input_tokens": 2, "output_tokens": 267,
            "cache_read_input_tokens": 11832,
            "cache_creation_input_tokens": 1962,
        })
        assert u["input"] == 2 and u["output"] == 267
        assert u["cache_read"] == 11832 and u["cache_write"] == 1962

    def test_openai_responses_shape_unchanged(self):
        from alpha_lab.benchmarks.runcmp.extract import _usage_tokens
        u = _usage_tokens({
            "input_tokens": 100, "output_tokens": 50,
            "input_tokens_details": {"cached_tokens": 80},
            "output_tokens_details": {"reasoning_tokens": 20},
        })
        assert u == {"input": 100, "output": 50, "cache_read": 80,
                     "cache_write": 0, "reasoning": 20}


class TestModelRates:
    """Regression for the 2026-08-05 pricing gap: lab-hosted deepseek/gemma
    matched no rate prefix and fell back to opus pricing (the loud default),
    which mispriced unbilled models in token_accounting."""

    def test_lab_models_unbilled(self):
        from alpha_lab.benchmarks.runcmp.tabulate import rates_for
        for m in ("deepseek-v4-flash", "gemma-4-31b", "glm-5.2", "kimi-k3"):
            assert rates_for(m)["input"] == 0.0 and rates_for(m)["output"] == 0.0, m

    def test_unknown_still_inflates_loudly(self):
        from alpha_lab.benchmarks.runcmp.tabulate import rates_for, MODEL_RATES
        assert rates_for("mystery-model-9") == MODEL_RATES["claude-opus"]


class TestMission:
    """Auto-generated missions (`runcmp mission`, investigate --mission auto)."""

    def _campaign(self, tmp_path):
        corpus = {
            "runs": [
                {"label": "eraA/domain2_llm_speedrun/cond/r1",
                 "framework": "cond", "domain": "domain2",
                 "model": "m-one", "run_state": "finished",
                 "db_terminal_rows": 20, "era": "eraA"},
                {"label": "eraB/domain2_llm_speedrun/cond/r2",
                 "framework": "cond", "domain": "domain2",
                 "model": "m-one", "run_state": "finished",
                 "db_terminal_rows": 18, "era": "eraB"},
                {"label": "eraB/domain2_llm_speedrun/msml/r3",
                 "framework": "msml", "domain": "domain2",
                 "model": "m-two", "run_state": "in_flight",
                 "db_terminal_rows": 3, "era": "eraB"},
            ]
        }
        referee = {
            "runs": {
                "eraA/domain2_llm_speedrun/cond/r1": {
                    "label": "eraA/domain2_llm_speedrun/cond/r1",
                    "domain": "domain2", "referee_kind": "bpb_curve",
                    "rankable": True, "validation_identities": ["id-a"]},
                "eraB/domain2_llm_speedrun/cond/r2": {
                    "label": "eraB/domain2_llm_speedrun/cond/r2",
                    "domain": "domain2", "referee_kind": "bpb_curve",
                    "rankable": False,
                    "rankable_reason": "evaluates multiple distinct slices",
                    "validation_identities": ["id-b", "id-c"]},
            },
            "pairs": [{"pair": "p1", "comparable": False,
                       "reason": "no shared validation identity"}],
        }
        (tmp_path / "corpus.json").write_text(json.dumps(corpus))
        (tmp_path / "referee.json").write_text(json.dumps(referee))
        (tmp_path / "bench.md").write_text("# scorecard\n")
        return tmp_path / "corpus.json"

    def test_build_mission_inventory(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.mission import build_mission
        text = build_mission(self._campaign(tmp_path), focus="variability",
                             notes=["wave B carried a pacing cap"])
        # replication group detected
        assert "cond + m-one: 2 runs" in text
        # in-flight run flagged, never silently ranked
        assert "in_flight" in text and "draw verdicts only from finished" in text
        # rankability quoted verbatim, not re-derived
        assert "evaluates multiple distinct slices" in text
        # pair non-comparability reason carried (single-domain fallback)
        assert "no shared validation identity" in text
        # operator note printed as declared
        assert "wave B carried a pacing cap" in text
        # scorecards discovered
        assert "bench.md" in text

    def test_focus_presets_and_unknown(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.mission import FOCI, build_mission
        cj = self._campaign(tmp_path)
        assert set(FOCI) >= {"harness", "variability"}
        assert "what to change first" in build_mission(cj).lower()
        with pytest.raises(ValueError):
            build_mission(cj, focus="nope")

    def test_resolve_mission_arg_writes_audit_copy(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.mission import resolve_mission_arg
        cj = self._campaign(tmp_path)
        out = tmp_path / "review"
        text = resolve_mission_arg("auto:variability", cj, out)
        assert "run-to-run" in text.lower()
        assert (out / "mission.md").read_text() == text
        assert resolve_mission_arg("none", cj, out) == ""
        assert resolve_mission_arg("literal scope", cj, out) == "literal scope"


class TestExpandNotes:
    def test_file_notes_bypass_shell_quoting(self, tmp_path):
        # Apostrophes in --note strings killed reviewer launches twice
        # (2026-08-07/08); @file notes carry any text safely.
        from alpha_lab.benchmarks.runcmp.mission import expand_notes
        f = tmp_path / "notes.txt"
        f.write_text("the campaign's ranges\n\nsecond 'quoted' note\n")
        assert expand_notes(["plain", f"@{f}"]) == [
            "plain", "the campaign's ranges", "second 'quoted' note"]


class TestLeaderboardIdentityNote:
    def test_multi_identity_note(self):
        from alpha_lab.benchmarks.runcmp import bench as bench_mod
        referee = {"runs": {
            "a/cond/r1": {"label": "a/cond/r1", "domain": "domain2",
                          "framework": "cond", "referee_kind": "bpb_curve",
                          "rankable": True,
                          "validation_identities": ["id-a"],
                          "scored_experiments": 2, "artifact_coverage": 1.0,
                          "experiments": [
                              {"experiment": "e1", "recomputed": 0.9}]},
            "a/msml/r2": {"label": "a/msml/r2", "domain": "domain2",
                          "framework": "msml", "referee_kind": "bpb_curve",
                          "rankable": True,
                          "validation_identities": ["id-b"],
                          "scored_experiments": 2, "artifact_coverage": 1.0,
                          "experiments": [
                              {"experiment": "e2", "recomputed": 0.8}]},
        }, "pairs": []}
        bench = {"runs": {
            "a/cond/r1": {"direction": "minimize", "domain": "domain2",
                          "framework": "cond", "metrics": {}, "diagnostics": []},
            "a/msml/r2": {"direction": "minimize", "domain": "domain2",
                          "framework": "msml", "metrics": {}, "diagnostics": []}}}
        # render_md needs the version header fields
        bench_full = {"bench_version": 1, "rules_version": 1,
                      "generated_at": "t", "runs": bench["runs"]}
        md = bench_mod.render_md(bench_full, referee)
        assert "distinct validation identities" in md


class TestChartRenderValidation:
    BAD = ("# r\n\n```chart\ntype: stacked-bar\nseries:\n  - name: a\n"
           "    values: [1, 2]\n```\n")
    GOOD = ("# r\n\n```chart\ntype: bars\ntitle: t (lower wins)\n"
            "run_a | 1.5 | note\nrun_b | 2.5\n```\n")

    def test_chart_errors_rejects_foreign_dialect(self):
        from alpha_lab.benchmarks.runcmp.render_html import chart_errors
        errs = chart_errors(self.BAD)
        assert len(errs) == 1 and "unknown type 'stacked-bar'" in errs[0]
        assert chart_errors(self.GOOD) == []

    def test_writer_prompt_carries_grammar(self):
        from alpha_lab.benchmarks.runcmp.investigate import SYSTEM_PROMPT
        assert "Chart blocks — EXACT grammar" in SYSTEM_PROMPT
        assert "row LABEL | name=value" in SYSTEM_PROMPT

    def test_strip_unrenderable_charts(self):
        from alpha_lab.benchmarks.runcmp.render_html import (
            chart_errors, strip_unrenderable_charts)
        md = self.GOOD + "\nprose stays\n" + self.BAD
        out, n = strip_unrenderable_charts(md)
        assert n == 1
        assert "prose stays" in out and "stacked-bar" not in out
        assert "chart removed at publication" in out
        assert chart_errors(out) == []
        # clock-time spans (the 2026-08-07 shipped defect) are stripped too
        spans = ("```chart\ntype: spans\ntitle: t\n"
                 "span a | 19:19 | 19:55 | x\n```\n")
        out2, n2 = strip_unrenderable_charts(spans)
        assert n2 == 1 and "```chart" not in out2


class TestReportCritic:
    """The report is the deliverable; before 2026-08-10 nothing reviewed it.

    Three obs reports answered whole mission sections with one line per case
    (grand/obs section 5: 535 characters against team_opus's 3,835) and passed
    because the only gate counted charts. These pin the gate that stops it —
    and pin that a report which already answers the mission is NOT bounced,
    since the all-opus configuration was producing good reports and must keep
    sailing through untouched.
    """

    MISSION = ("## 1. The decisions this report must deliver\n"
               "1. **Headline verdicts** — the answers up front.\n"
               "2. **Combination** — the pairing to run today, per task.\n"
               "3. **Cost** — per scored experiment.\n")

    def _session(self, tmp_path, verdict):
        from alpha_lab.benchmarks.runcmp import investigate_team as it
        calls = []

        def fake_one_call(provider, model, system, user, effort,
                          retries=4, **kwargs):
            calls.append({"system": system, "user": user})
            return verdict

        it._one_call = fake_one_call
        s = object.__new__(it.TeamSession)
        s.out_dir = tmp_path
        s._critic = {"provider": None, "model": "m", "reasoning_effort": "high",
                     "mission": self.MISSION}
        s._critic_log = open(tmp_path / "critic_log.jsonl", "a")
        return s, calls

    def test_mission_themes_parsed(self, tmp_path):
        s, _ = self._session(tmp_path, "{}")
        assert s._mission_sections() == ["Headline verdicts", "Combination", "Cost"]

    def test_stub_sections_flagged_and_real_ones_are_not(self):
        from alpha_lab.benchmarks.runcmp.investigate_team import TeamSession
        stub = ("## 2. Combination\nd5\nRun msml+Opus 5.\nd7\nRun cond.\n")
        assert TeamSession._stub_sections(stub, ["Combination"])
        # short but cited counts as an answer
        cited = "## 3. Cost\n$23.53 per experiment [Finding 4], see probe_012.\n"
        assert TeamSession._stub_sections(cited, ["Cost"]) == []
        # bulk counts as an answer
        assert TeamSession._stub_sections("## 3. Cost\n" + "x" * 950, ["Cost"]) == []
        # a theme with no matching heading is NOT asserted missing: mission
        # themes range from one word to a sentence and writers retitle, so
        # absence is the critic's call (an exact-substring rule flagged three
        # good reports as missing sections, 2026-08-10)
        assert TeamSession._stub_sections("## 3. Cost\n" + "x" * 950,
                                          ["Combination"]) == []

    def test_retitled_sections_are_found(self):
        """The wordings that actually appear in the published reports."""
        from alpha_lab.benchmarks.runcmp.investigate_team import TeamSession
        heads = ["7. Treatment: reasoning replay", "12. Behaviour provenance",
                 "14. Measurement hygiene and the numerical-oddities registry"]
        for theme in ("Treatment (reasoning replay)", "Behavior provenance",
                      "Measurement hygiene and anomalies"):
            assert TeamSession._match_section(theme, heads) is not None, theme
        assert TeamSession._match_section("Cost", heads) is None

    def test_revise_bounces_the_draft_back(self, tmp_path):
        s, calls = self._session(
            tmp_path,
            '{"verdict": "revise", "reasons": "section 2 is a stub",'
            ' "required_changes": "2. Combination: answer per task"}')
        out = s._report_review("## 2. Combination\nd5\nRun msml.\n")
        assert out and "SENT BACK BY THE CRITIC" in out
        assert "answer per task" in out
        assert "write_report" in out           # tells it how to resubmit
        assert self.MISSION[:40] in calls[0]["user"]   # critic saw the mission
        assert "Combination" in calls[0]["user"]

    def test_accept_lets_the_draft_through(self, tmp_path):
        s, _ = self._session(tmp_path, '{"verdict": "accept", "reasons": "ok"}')
        assert s._report_review("## 2. Combination\n" + "x" * 950) is None

    def test_publishes_the_best_scored_draft_not_the_last_or_longest(self, tmp_path):
        """obs_v2 shipped a 22,154-char draft its critic never saw, after
        reviewing 63,934 and 59,582 (2026-08-10). The publish rule is the
        critic's score — never arrival order, never size."""
        from alpha_lab.benchmarks.runcmp import investigate_team as it
        scores = iter([40, 70, 55])
        it._one_call = lambda *a, **k: json.dumps(
            {"verdict": "revise", "score": next(scores), "reasons": "r",
             "required_changes": "c"})
        s = object.__new__(it.TeamSession)
        s.out_dir = tmp_path
        s._critic = {"provider": None, "model": "m", "reasoning_effort": "high",
                     "mission": "1. **Cost** — per experiment"}
        s._critic_log = open(tmp_path / "critic_log.jsonl", "a")
        s.iterations_left = 500
        for text in ("draft A" * 40, "draft B" * 10, "draft C" * 80):
            assert s._report_review(text) is not None
        assert [c["score"] for c in s._report_candidates] == [40.0, 70.0, 55.0]
        s.iterations_left = 3          # budget gone: the gate stops bouncing
        published = s._pick_publishable("draft C" * 80)
        assert published.startswith("draft B")      # best score
        assert not published.startswith("draft A")  # not the longest
        assert not published.startswith("draft C")  # not the last

    def test_keeps_bouncing_while_budget_and_time_remain(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.investigate_team import TeamSession
        s = object.__new__(TeamSession)
        s._critic = {"model": "m"}
        s._critic_log = open(tmp_path / "critic_log.jsonl", "a")
        assert s._may_bounce_again() is False        # nothing wired -> no loop
        s.iterations_left = 5
        assert s._may_bounce_again() is False        # under the reserve
        s.iterations_left = 400
        assert s._may_bounce_again() is True         # budget left, clock unstarted
        s._report_phase_start = time.time() - (TeamSession.REPORT_PHASE_SECONDS + 60)
        assert s._may_bounce_again() is False        # past the wall clock

    def test_bounces_are_bounded_then_it_publishes(self, tmp_path):
        s, _ = self._session(tmp_path, '{"verdict": "revise", "reasons": "no"}')
        draft = "## 2. Combination\nd5\nstub.\n"
        assert s._report_review(draft) is not None
        assert s._report_review(draft) is not None
        # cap reached: the third submission publishes rather than looping
        assert s._report_review(draft) is None

    def test_a_broken_critic_never_costs_the_report(self, tmp_path):
        s, _ = self._session(tmp_path, "not json at all")
        assert s._report_review("## 2. Combination\nd5\nstub.\n") is None

    def test_no_critic_means_no_review(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.investigate_team import TeamSession
        s = object.__new__(TeamSession)
        s._critic = None
        assert s._report_review("anything") is None


class TestRecompose:
    """Restarting from the writer is a standard capability, not a debug path.

    The evidence half and the writing half of a review fail independently, and
    re-running everything to fix the writing both wastes the evidence phase and
    risks losing findings that were already critic-accepted.
    """

    def test_it_is_a_first_class_stage(self):
        from alpha_lab.benchmarks.runcmp.__main__ import _STAGES
        assert "recompose" in _STAGES

    def test_findings_are_frozen(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.recompose import ComposeSession
        s = object.__new__(ComposeSession)
        out = s.record_finding(claim="anything", evidence=[])
        assert out.startswith("[ERROR]") and "frozen" in out

    def test_refuses_without_the_evidence_it_needs(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.recompose import _load
        import pytest
        with pytest.raises(SystemExit):
            _load(tmp_path)                        # no mission.md
        (tmp_path / "mission.md").write_text("1. **Cost** — x")
        with pytest.raises(SystemExit):
            _load(tmp_path)                        # no findings.jsonl
        (tmp_path / "findings.jsonl").write_text(
            json.dumps({"id": 1, "claim": "c"}) + "\n")
        mission, findings, ledger = _load(tmp_path)
        assert findings and mission.startswith("1.") and ledger == {}

    def test_record_finding_is_not_offered_as_a_tool(self):
        from alpha_lab.benchmarks.runcmp.investigate import TOOLS
        from alpha_lab.benchmarks.runcmp.investigate_team import QUESTION_TOOLS
        offered = [t for t in (TOOLS + QUESTION_TOOLS)
                   if t.get("name") != "record_finding"]
        assert any(t["name"] == "write_report" for t in offered)
        assert not any(t["name"] == "record_finding" for t in offered)


class TestChartEditorGrammar:
    def test_editor_is_given_the_legal_type_list(self):
        """The editor invented `type: band`/`type: bar` because its prompt named
        forms without giving the renderer's type table. Aliasing them would hide
        the mistake; the fix is telling it the legal set and making it fix its
        own errors."""
        import inspect
        from alpha_lab.benchmarks.runcmp import investigate_team as it
        src = inspect.getsource(it.TeamSession._chart_editor_pass)
        assert "The ONLY legal values of `type:` are" in src
        assert "CHART_SYNTAX" in src
        assert "`band` is NOT a type" in src
        assert "retrying" in src              # it re-asks instead of discarding

    def test_unknown_type_is_still_an_error(self):
        from alpha_lab.benchmarks.runcmp.render_html import chart_errors
        for bad in ("bar", "band", "stacked-bar"):
            md = f"```chart\ntype: {bad}\ntitle: t\na | 1\n```\n"
            errs = chart_errors(md)
            assert len(errs) == 1 and f"unknown type '{bad}'" in errs[0]


class TestRequestComposition:
    """The payload census must understand all three request dialects; the
    chat-completions one (lab endpoints) used to fold everything into
    user_text/other, reporting images/tool_args/tool_results/thinking = 0
    for every kimi/glm run."""

    @staticmethod
    def _comp(ev):
        from alpha_lab.benchmarks.runcmp.extract import (
            _COMPOSITION_KEYS, _add_request_composition)
        comp = {k: 0 for k in _COMPOSITION_KEYS}
        _add_request_composition(ev, comp)
        return comp

    def test_chat_completions_dialect_populates_all_fields(self):
        ev = {
            "type": "api_request",
            "instructions": "sys prompt",
            "tools": [{"type": "function", "function": {"name": "shell"}}],
            "input": [
                {"role": "user", "content": "run ls for me"},
                {"role": "assistant", "content": "on it",
                 "reasoning_content": "think think think",
                 "tool_calls": [{"id": "1", "type": "function",
                                 "function": {"name": "shell",
                                              "arguments": '{"cmd": "ls"}'}}]},
                {"role": "tool", "content": "file_a\nfile_b"},
                {"role": "user", "content": [
                    {"type": "text", "text": "see the plot"},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64,AAAA"}}]},
            ],
        }
        comp = self._comp(ev)
        assert comp["tool_results"] == len("file_a\nfile_b")
        assert comp["tool_args"] == len("shell") + len('{"cmd": "ls"}')
        assert comp["thinking"] == len("think think think")
        assert comp["images"] > 0
        assert comp["user_text"] == len("run ls for me") + len("see the plot")
        assert comp["assistant_text"] == len("on it")
        assert comp["other"] == 0

    def test_assistant_tool_call_message_not_double_counted(self):
        # content=None + tool_calls: args counted once, nothing in "other".
        ev = {"input": [{"role": "assistant", "content": None,
                         "tool_calls": [{"function": {"name": "f",
                                                      "arguments": "xy"}}]}]}
        comp = self._comp(ev)
        assert comp["tool_args"] == len("f") + len("xy")
        assert comp["other"] == 0

    def test_reasoning_roundtrip_census(self, tmp_path):
        # A model that produces reasoning but never receives it back is
        # running blind (the GLM/deepseek defect class, 2026-08-07); the
        # pack must carry both sides so reviewers can see the asymmetry.
        import json as _json
        from alpha_lab.benchmarks.runcmp.extract import parse_agent_logs
        lines = [
            {"type": "api_request", "timestamp": 1.0,
             "input": [{"role": "user", "content": "hi"}]},
            {"type": "api_response", "timestamp": 2.0, "usage": {},
             "raw_response": {"reasoning_content": "thought"}},
            {"type": "api_request", "timestamp": 3.0,
             "input": [{"role": "assistant", "content": "x",
                        "reasoning_content": "thought"}]},
            {"type": "api_response", "timestamp": 4.0, "usage": {}},
        ]
        (tmp_path / "worker_t.jsonl").write_text(
            "\n".join(_json.dumps(x) for x in lines) + "\n")
        out = parse_agent_logs(tmp_path)
        assert out["reasoning_roundtrip"] == {
            "requests_total": 2, "requests_with_reasoning": 1,
            "responses_total": 2, "responses_with_reasoning": 1}

    def test_responses_dialect_still_classified(self):
        ev = {"input": [
            {"type": "function_call", "name": "shell", "arguments": '{"a":1}'},
            {"type": "function_call_output", "output": "ok"},
            {"role": "user", "content": [{"type": "input_image",
                                          "image_url": "data:..."}]},
        ]}
        comp = self._comp(ev)
        assert comp["tool_args"] > 0
        assert comp["tool_results"] > 0
        assert comp["images"] > 0


class TestBudgetExhaustionSalvage:
    """A writer that can no longer call write_report must not cost the report.

    grand/obs_v2's recompose (2026-08-10) submitted 23 drafts, every one
    critic-scored (best 85/100), then drowned: each bounced draft stayed in
    history as tool-call ARGUMENTS, which compaction never trimmed, until the
    model could not emit a full draft any more. The loop hit its cap, no
    salvage ran, and the stage had also renamed the previous report away —
    the review ended with no report at all. These pin all three fixes.
    """

    def test_compaction_trims_old_draft_arguments(self):
        import json as _json
        from alpha_lab.benchmarks.runcmp.investigate import (
            KEEP_RECENT_ITEMS, _compact_history)
        draft = "x" * 30_000
        history = [{"type": "function_call", "name": "write_report",
                    "arguments": _json.dumps({"markdown": draft})}
                   for _ in range(KEEP_RECENT_ITEMS + 5)]
        _compact_history(history)
        # old drafts dropped, replacement is valid JSON (provider translators
        # may parse arguments when rebuilding history)
        note = _json.loads(history[0]["arguments"])
        assert "report_drafts" in note["dropped"]
        # recent items untouched
        assert len(history[-1]["arguments"]) > 20_000

    def _session(self, tmp_path):
        from alpha_lab.benchmarks.runcmp import investigate_team as it
        s = object.__new__(it.TeamSession)
        s.out_dir = tmp_path
        s.tables_dir = tmp_path      # no bench.json/referee.json -> no gates
        s.packs_dir = tmp_path
        s._critic = None             # chart-editor pass is a no-op
        s._questions = {}
        s._critic_log = open(tmp_path / "critic_log.jsonl", "a")
        return s

    def test_salvage_publishes_highest_scored_draft(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.investigate_team import (
            publish_best_draft)
        s = self._session(tmp_path)
        d = tmp_path / "report_drafts"
        d.mkdir()
        (d / "round_01.md").write_text("## draft one\nlow-scored text\n")
        (d / "round_02.md").write_text("## draft two\nBEST-SCORED TEXT\n")
        s._report_candidates = [
            {"round": 1, "path": str(d / "round_01.md"), "score": 90.0},
            {"round": 2, "path": str(d / "round_02.md"), "score": 60.0},
        ]
        assert publish_best_draft(s) is True
        text = (tmp_path / "REPORT.md").read_text()
        assert "low-scored text" in text          # score wins, not recency
        assert s.report_written

    def test_salvage_without_scored_drafts_reports_failure(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.investigate_team import (
            publish_best_draft)
        s = self._session(tmp_path)
        s._report_candidates = [{"round": 1, "path": "nowhere", "score": None}]
        assert publish_best_draft(s) is False
        assert not (tmp_path / "REPORT.md").is_file()

    def test_chartless_terminal_report_is_never_refused(self, tmp_path):
        """The floors warn-accept when no writer is left to revise: a forced
        flush with a fresh refusal counter used to be silently REFUSED —
        success printed, nothing written."""
        from alpha_lab.benchmarks.runcmp.investigate import InvestigatorSession
        s = self._session(tmp_path)
        s._report_refusals = 3
        s._best_draft = "# flush\nno charts at all\n"
        out = InvestigatorSession.write_report(s, "# flush\nno charts at all\n")
        assert "REPORT.md written" in out
        assert (tmp_path / "REPORT.md").is_file()

    def test_rerun_archives_previous_drafts_never_overwrites(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.recompose import _archive_drafts
        d = tmp_path / "report_drafts"
        d.mkdir()
        (d / "round_01.md").write_text("scored draft from the failed session")
        kept = _archive_drafts(tmp_path)
        assert kept is not None and kept.is_dir()
        assert (kept / "round_01.md").read_text().startswith("scored draft")
        assert not d.exists()                      # fresh session starts clean
        assert _archive_drafts(tmp_path) is None   # nothing left to move


class TestInFlightRunsVisible:
    """A run still writing rows must never pass silently as 'usable'.

    The index printout said 'usable' for a run whose recorded run_state was
    in_flight — a reader following the README took the word at face value
    (external feedback, 2026-08-11). Pairing already refused live runs; now
    the printout marks them and --finished-only drops them from the registry.
    """

    def _make_live(self, synthetic_corpus):
        # a launcher log with no exit line, freshly touched = still running
        (synthetic_corpus / "eraA/domain2_llm_speedrun/cond/ws.log"
         ).write_text("experiment 7 dispatched...\n")

    def test_in_flight_marked_and_never_paired(self, synthetic_corpus,
                                               tmp_path, capsys):
        import json as _json
        self._make_live(synthetic_corpus)
        out = tmp_path / "corpus.json"
        assert corpus_mod.main(["--root", str(synthetic_corpus),
                                "--out", str(out)]) == 0
        printed = capsys.readouterr().out
        assert "IN FLIGHT" in printed
        reg = _json.loads(out.read_text())
        states = {r["label"]: r["run_state"] for r in reg["runs"]}
        assert any(s == "in_flight" for s in states.values())
        assert all(not r["pair_key"] for r in reg["runs"])  # no phantom pair

    def test_finished_only_drops_live_runs(self, synthetic_corpus,
                                           tmp_path, capsys):
        import json as _json
        self._make_live(synthetic_corpus)
        out = tmp_path / "corpus.json"
        assert corpus_mod.main(["--root", str(synthetic_corpus),
                                "--out", str(out), "--finished-only"]) == 0
        printed = capsys.readouterr().out
        assert "DROPPED (--finished-only" in printed
        reg = _json.loads(out.read_text())
        assert all(r["run_state"] != "in_flight" for r in reg["runs"])
        assert len(reg["runs"]) == 1


class TestFactcheckExitStatus:
    """Exit status is the verdict, not the process health.

    factcheck returned 0 beside three referee-attribution violations and a
    caller took command success for verified-report success (external
    feedback, 2026-08-11). 0 now means every hard audit is clean.
    """

    def test_failed_finding_exits_nonzero(self, tmp_path, capsys):
        import json as _json
        out = tmp_path / "review"; out.mkdir()
        packs = tmp_path / "packs"; packs.mkdir()
        (out / "findings.jsonl").write_text(_json.dumps(
            {"id": 1, "claim": "x",
             "evidence": [{"probe": "probe_099", "contains": "ghost"}]}) + "\n")
        assert factcheck_mod.main(["--out", str(out),
                                   "--packs", str(packs)]) == 1
        assert "FACTCHECK FLAGS" in capsys.readouterr().out

    def test_clean_findings_exit_zero(self, tmp_path):
        import json as _json
        out = tmp_path / "review"; out.mkdir()
        packs = tmp_path / "packs"; packs.mkdir()
        (out / "probes").mkdir()
        (out / "probes/probe_001.out").write_text("hello 42\n")
        (out / "findings.jsonl").write_text(_json.dumps(
            {"id": 1, "claim": "x",
             "evidence": [{"probe": "probe_001", "contains": "hello"}]}) + "\n")
        assert factcheck_mod.main(["--out", str(out),
                                   "--packs", str(packs)]) == 0


class TestDefectAwarePublication:
    """Hard defects dominate publication; score only breaks ties.

    Measured failure (2026-08-11): a writer fixed four referee-attribution
    violations in round 4 (critic score 88), and the score-only selector
    reintroduced the stale round 3 (score 89, all four violations) — the
    downstream audits then bounce or banner the very defect the writer had
    already removed. Both selectors (gate and salvage) must rank by fewest
    deterministic violations first.
    """

    def _session(self, tmp_path):
        from alpha_lab.benchmarks.runcmp import investigate_team as it
        s = object.__new__(it.TeamSession)
        s.out_dir = tmp_path
        s.tables_dir = tmp_path
        s.packs_dir = tmp_path
        s._critic = None
        s._questions = {}
        s._critic_log = open(tmp_path / "critic_log.jsonl", "a")
        d = tmp_path / "report_drafts"
        d.mkdir()
        (d / "round_03.md").write_text("stale draft, violations intact")
        (d / "round_04.md").write_text("clean draft, violations fixed")
        s._report_candidates = [
            {"round": 3, "path": str(d / "round_03.md"),
             "score": 89.0, "defects": 4},
            {"round": 4, "path": str(d / "round_04.md"),
             "score": 88.0, "defects": 0},
        ]
        return s

    def test_gate_publishes_clean_88_over_defective_89(self, tmp_path):
        s = self._session(tmp_path)
        s._report_reviews = 5      # the choice is among stored rounds
        assert s._pick_publishable("current draft").startswith("clean draft")

    def test_salvage_publishes_clean_88_over_defective_89(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.investigate_team import (
            publish_best_draft)
        s = self._session(tmp_path)
        assert publish_best_draft(s) is True
        assert "clean draft" in (tmp_path / "REPORT.md").read_text()

    def test_review_records_defect_count_per_draft(self, tmp_path):
        """_report_review counts violations at review time so the stored
        candidate carries them; audit failure counts nothing."""
        import json as _json
        from alpha_lab.benchmarks.runcmp import investigate_team as it
        it._one_call = lambda *a, **k: _json.dumps(
            {"verdict": "revise", "score": 50, "reasons": "r",
             "required_changes": "c"})
        s = object.__new__(it.TeamSession)
        s.out_dir = tmp_path
        s._critic = {"provider": None, "model": "m",
                     "reasoning_effort": "high", "mission": ""}
        s._critic_log = open(tmp_path / "critic_log.jsonl", "a")
        s.iterations_left = 500
        bounce = s._report_review("a draft with no charts")
        assert bounce is not None
        # bare session: every audit raises and counts nothing
        assert s._report_candidates[0]["defects"] == 0


class TestFrozenEvidenceGuards:
    """Probe files are immutable once written; recompose proves it.

    Measured damage (2026-08-10/11): sessions reusing a review directory
    restarted probe numbering at 1 and overwrote the probe files the frozen
    findings cite — one directory regressed from clean verification to
    7 verified / 5 partial / 7 failed. Two guards now hold: the write site
    itself refuses to reuse an existing probe name even if the counter
    regresses, and recompose fingerprints every frozen file at start and
    fails loudly at exit if any changed.
    """

    def test_probe_write_never_reuses_an_existing_name(self, tmp_path):
        from alpha_lab.benchmarks.runcmp import investigate as inv
        s = object.__new__(inv.InvestigatorSession)
        s.out_dir = tmp_path
        s.corpus_path = tmp_path / "corpus.json"
        s.packs_dir = tmp_path / "packs"
        s.tables_dir = tmp_path
        probes = tmp_path / "probes"
        probes.mkdir()
        (probes / "probe_001.out").write_text("ORIGINAL EVIDENCE")
        (probes / "probe_001.py").write_text("# original")
        s.probe_count = 0            # simulate the exact regression
        out = s.run_python("print('new probe')")
        assert out.startswith("[probe_002]")
        assert (probes / "probe_001.out").read_text() == "ORIGINAL EVIDENCE"

    def test_fingerprint_detects_mutation_and_ignores_additions(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.recompose import _fingerprint
        (tmp_path / "mission.md").write_text("m")
        (tmp_path / "findings.jsonl").write_text("{}")
        probes = tmp_path / "probes"
        probes.mkdir()
        (probes / "probe_001.out").write_text("evidence")
        before = _fingerprint(tmp_path)
        (probes / "probe_002.out").write_text("new, additive")   # allowed
        after = _fingerprint(tmp_path)
        assert [f for f, h in before.items() if after.get(f) != h] == []
        (probes / "probe_001.out").write_text("CLOBBERED")       # violation
        after = _fingerprint(tmp_path)
        assert [f for f, h in before.items() if after.get(f) != h] == [
            "probes/probe_001.out"]


class TestProbeAdmissibility:
    """The reprint-only classifier must know the package's own data readers.

    A probe delegating to `probe_std.pack()` reads real evidence, but the
    read-primitive regex saw no `open(`/`json.load` and stamped it
    reprint-only — verification then failed and the writer redid the same
    measurement with a raw JSON read (external reader, 2026-08-11).
    """

    def test_probe_std_counts_as_data_access(self):
        from alpha_lab.benchmarks.runcmp.investigate import (
            _classify_probe_code)
        code = ("import probe_std\n"
                "p = probe_std.pack('era/d/fw/run')\n"
                "print(p['agent_logs']['samples']['llm_gap_seconds'])\n")
        assert _classify_probe_code(code) is None
        # a probe that only prints typed-in numbers is still inadmissible
        assert _classify_probe_code("print(42.7, 'measured!')") is not None


class TestHelperStages:
    """The day-2 helpers: every one answers a question that was answered by
    hand-typed one-off scripts during the 2026-08-10/11 campaigns."""

    def _review(self, tmp_path):
        import json as _json
        d = tmp_path / "review"
        d.mkdir()
        (d / "mission.md").write_text("1. **Cost** — per experiment.\n")
        (d / "findings.jsonl").write_text('{"id": 1, "claim": "x"}\n')
        (d / "questions.json").write_text(_json.dumps(
            {"1": {"id": 1, "status": "resolved", "text": "q"}}))
        (d / "sessions.jsonl").write_text(_json.dumps(
            {"ts": 1.0, "stage": "investigate-team", "provider": "openai",
             "model": "gpt-x", "reasoning_effort": "high",
             "roles": {"critic": "anthropic:claude-y"}}) + "\n")
        (d / "critic_log.jsonl").write_text(_json.dumps(
            {"ts": 2.0, "report_review": "accept", "round": 1,
             "score": 88.0, "defects": 0, "draft_chars": 1000}) + "\n")
        (d / "REPORT.md").write_text("## 1. Cost\n" + "x" * 950)
        return d

    def test_status_reads_a_review(self, tmp_path, capsys):
        from alpha_lab.benchmarks.runcmp import status
        d = self._review(tmp_path)
        assert status.main(["--out", str(d)]) == 0
        out = capsys.readouterr().out
        assert "writer=openai:gpt-x" in out
        assert "round  1: score=88.0" in out
        assert "0 stub section(s)" in out
        assert "not run (factcheck)" in out

    def test_watch_snapshots_a_root(self, synthetic_corpus, capsys):
        from alpha_lab.benchmarks.runcmp import watch
        assert watch.main(["--root", str(synthetic_corpus)]) == 0
        out = capsys.readouterr().out
        assert "2 run(s), 0 still in flight" in out
        assert "done=3" in out

    def test_preflight_clear_and_blocked(self, synthetic_corpus, tmp_path,
                                         capsys):
        from alpha_lab.benchmarks.runcmp import preflight
        assert preflight.main(["--root", str(synthetic_corpus)]) == 0
        assert "clear to launch" in capsys.readouterr().out
        packs = tmp_path / "packs"
        packs.mkdir()                      # empty: every pack is missing
        assert preflight.main(["--root", str(synthetic_corpus),
                               "--packs", str(packs)]) == 1
        out = capsys.readouterr().out
        assert "BLOCKER" in out and "no evidence pack" in out

    def test_validate_submission(self, tmp_path, capsys):
        import json as _json
        from alpha_lab.benchmarks.runcmp import submission
        run = tmp_path / "run_x"
        good = run / "experiments/exp_a/results"
        good.mkdir(parents=True)
        (good / "metrics.json").write_text(_json.dumps({"sharpe": 1.2}))
        (good / "referee_predictions.csv").write_text("a,b\n1,2\n")
        bad = run / "experiments/exp_b/results"
        bad.mkdir(parents=True)
        (bad / "metrics.json").write_text("{not json")
        assert submission.main(["--run", str(run)]) == 0
        out = capsys.readouterr().out
        assert "scorable experiments: 1 (1 problem(s))" in out
        assert "referee inputs: referee_predictions.csv" in out
        empty = tmp_path / "run_empty"
        (empty / "experiments").mkdir(parents=True)
        assert submission.main(["--run", str(empty)]) == 1

    def test_rereview_uses_the_recorded_writer(self, tmp_path, capsys,
                                               monkeypatch):
        from alpha_lab.benchmarks.runcmp import recompose, rereview
        d = self._review(tmp_path)
        seen = {}
        monkeypatch.setattr(recompose, "main",
                            lambda argv: seen.setdefault("argv", argv) and 0
                            or 0)
        assert rereview.main(["--out", str(d), "--corpus", "c", "--packs",
                              "p"]) == 0
        argv = seen["argv"]
        assert argv[argv.index("--provider") + 1] == "openai"
        assert argv[argv.index("--model") + 1] == "gpt-x"
        assert argv[argv.index("--role") + 1] == "critic=anthropic:claude-y"

    def test_rereview_refuses_to_guess(self, tmp_path):
        import pytest as _pytest
        from alpha_lab.benchmarks.runcmp import rereview
        d = tmp_path / "old_review"
        d.mkdir()
        with _pytest.raises(SystemExit, match="predates provenance"):
            rereview.main(["--out", str(d), "--corpus", "c", "--packs", "p"])

    def test_mission_check(self, synthetic_corpus, tmp_path, capsys):
        from alpha_lab.benchmarks.runcmp import corpus as corpus_mod
        from alpha_lab.benchmarks.runcmp import mission
        cj = tmp_path / "corpus.json"
        corpus_mod.main(["--root", str(synthetic_corpus), "--out", str(cj)])
        capsys.readouterr()
        good = tmp_path / "m.md"
        good.write_text("1. **Cost** — per experiment.\n"
                        "2. **Verdict** — per task.\n")
        assert mission.main(["--corpus", str(cj), "--check",
                             str(good)]) == 0
        assert "2 numbered section themes" in capsys.readouterr().out
        bad = tmp_path / "bad.md"
        bad.write_text("just prose, no numbered sections\n")
        assert mission.main(["--corpus", str(cj), "--check", str(bad)]) == 1


class TestBringYourOwnHarness:
    """Nothing may RELY on the in-house framework names (user order,
    2026-08-11): most corpora bring their own harnesses. The in-house names
    stay as detection defaults; pairing, tables and orientation work from
    whatever names the corpus carries."""

    def test_two_external_frameworks_auto_pair(self, tmp_path):
        root = tmp_path / "corpus"
        _make_run(root, "eraA", "domain2_llm_speedrun", "alphaforge",
                  plant_artifacts=False)
        _make_run(root, "eraA", "domain2_llm_speedrun", "betabench",
                  plant_artifacts=False)
        records = build_registry(root)
        by_fw = {r.framework: r for r in records}
        assert set(by_fw) == {"alphaforge", "betabench"}   # layout-named
        assert by_fw["alphaforge"].pair_key == by_fw["betabench"].pair_key != ""

    def test_three_frameworks_left_to_explicit_pairs(self, tmp_path, capsys):
        root = tmp_path / "corpus"
        for fw in ("alphaforge", "betabench", "gammalab"):
            _make_run(root, "eraA", "domain2_llm_speedrun", fw,
                      plant_artifacts=False)
        records = build_registry(root)
        assert all(not r.pair_key for r in records)
        assert "auto-pairing needs exactly two" in capsys.readouterr().out

    def test_aggregate_tables_carry_any_framework(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.tabulate import corpus_aggregates
        root = tmp_path / "corpus"
        _make_run(root, "eraA", "domain2_llm_speedrun", "alphaforge",
                  plant_artifacts=False)
        _make_run(root, "eraA", "domain2_llm_speedrun", "betabench",
                  plant_artifacts=False)
        records = build_registry(root)
        packs = {r.label: {"run": {"framework": r.framework},
                           "experiments": {"scored": 1, "total": 2},
                           "events": {}} for r in records}
        md, machine = corpus_aggregates(records, packs)
        assert "alphaforge" in md and "betabench" in md
        assert set(machine["attrition"]) == {"alphaforge", "betabench"}


class TestRefereeTruthUnreadable:
    """An unreadable truth dataset costs one domain its referee scores,
    loudly — never the whole stage. A cold-start user's referee crashed
    outright on a 'nobody'-owned CSV and produced no referee.json at all
    (2026-08-11); the mission CLI also shipped without @file note
    expansion, caught by the same user."""

    def test_unreadable_truth_refuses_domain_not_stage(self, tmp_path,
                                                       monkeypatch, capsys):
        import json as _json
        from alpha_lab.benchmarks.runcmp import referee as ref
        root = tmp_path / "corpus"
        _make_run(root, "eraA", "d7_payup", "alpha", plant_artifacts=False)
        from alpha_lab.benchmarks.runcmp import corpus as corpus_mod
        cj = tmp_path / "corpus.json"
        corpus_mod.main(["--root", str(root), "--out", str(cj)])
        packs = tmp_path / "packs"
        packs.mkdir()
        reg = _json.loads(cj.read_text())["runs"]
        assert reg and reg[0]["domain"] == "d7_payup"
        label = reg[0]["label"]
        (packs / (label.replace("/", "__") + ".json")).write_text(_json.dumps(
            {"experiments": {"experiments": [{"name": "e1"}], "scored": 1}}))

        def boom(*a, **k):
            raise PermissionError(13, "Permission denied", "truth.csv")
        monkeypatch.setattr(ref, "regression_verification", boom)
        out = ref.run_referee(cj, packs, tmp_path / "referee.json", [])
        rec = out["runs"][label]
        assert rec["rankable"] is False
        assert rec["refused_truth_unreadable"] is True
        assert "unreadable" in rec["rankable_reason"]
        assert "REFUSED" in capsys.readouterr().out

    def test_mission_cli_expands_note_files(self, synthetic_corpus, tmp_path,
                                            capsys):
        from alpha_lab.benchmarks.runcmp import corpus as corpus_mod
        from alpha_lab.benchmarks.runcmp import mission
        cj = tmp_path / "corpus.json"
        corpus_mod.main(["--root", str(synthetic_corpus), "--out", str(cj)])
        capsys.readouterr()
        notes = tmp_path / "notes.txt"
        notes.write_text("the GLM era X was served in bf16\n")
        assert mission.main(["--corpus", str(cj),
                             "--note", f"@{notes}"]) == 0
        out = capsys.readouterr().out
        assert "served in bf16" in out
        assert f"@{notes}" not in out


class TestProductionCostFooter:
    """Every report ends with what it cost to make (user order 2026-08-11).

    Token counts come from the session's own API responses (usage.jsonl);
    prices from the shared rates table; the fact-checker exempts the footer
    from the body audit because its numbers come from the ledger, not from
    probes."""

    class _Resp:
        def __init__(self, inp, cached, out, cw=0):
            self.input_tokens = inp
            self.cache_read_input_tokens = cached
            self.cache_write_input_tokens = cw
            self.output_tokens = out

    class _AnthropicProvider:  # class NAME carries the semantics
        pass

    class _OpenAIProvider:
        pass

    def _session(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.investigate import InvestigatorSession
        s = object.__new__(InvestigatorSession)
        s.out_dir = tmp_path
        s.tables_dir = tmp_path
        s.packs_dir = tmp_path
        s._own_usage = []
        s._questions = {}
        return s

    def test_fresh_input_semantics_per_provider(self, tmp_path):
        s = self._session(tmp_path)
        # Anthropic: input_tokens already EXCLUDES cache reads
        s.note_own_usage("critic", "m", self._AnthropicProvider(),
                         self._Resp(1000, 90_000, 50))
        # OpenAI-shaped: input_tokens INCLUDES cached
        s.note_own_usage("executor", "m", self._OpenAIProvider(),
                         self._Resp(100_000, 90_000, 50))
        a, o = s._own_usage
        assert a["fresh_input"] == 1000
        assert o["fresh_input"] == 10_000
        assert (tmp_path / "usage.jsonl").is_file()

    def test_footer_published_and_priced(self, tmp_path, monkeypatch):
        from alpha_lab.benchmarks.runcmp import tabulate
        from alpha_lab.benchmarks.runcmp.investigate import InvestigatorSession
        monkeypatch.setattr(tabulate, "rates_for", lambda m: {
            "input": 1.0, "cache_read": 0.1, "cache_write": 1.25,
            "output": 5.0})
        s = self._session(tmp_path)
        s.note_own_usage("executor", "test-model", self._OpenAIProvider(),
                         self._Resp(1_000_000, 500_000, 100_000))
        s._report_refusals = 3            # chartless flush: floors warn only
        s._best_draft = "# report\nbody\n"
        out = InvestigatorSession.write_report(s, "# report\nbody\n")
        assert "REPORT.md written" in out
        text = (tmp_path / "REPORT.md").read_text()
        # 0.5M fresh*1.0 + 0.5M cached*0.1 + 0.1M out*5.0 = $1.05
        assert "Production cost of this report: **$1.05**" in text
        assert "<!-- production-cost -->" in text
        assert "executor $1.05 (1 calls, test-model)" in text

    def test_factcheck_exempts_the_footer(self, tmp_path, capsys):
        import json as _json
        from alpha_lab.benchmarks.runcmp import factcheck as factcheck_mod
        out = tmp_path / "review"; out.mkdir()
        packs = tmp_path / "packs"; packs.mkdir()
        (out / "findings.jsonl").write_text("")
        (out / "REPORT.md").write_text(
            "# t\n\nprose without numerals\n\n<!-- production-cost -->\n\n"
            "---\n\n*Production cost of this report: **$123,456.78** — "
            "999 model calls, 42.0M tokens in (33.3M from cache), "
            "7.77M out.*\n")
        assert factcheck_mod.main(["--out", str(out),
                                   "--packs", str(packs)]) == 0
        assert "FACTCHECK FLAGS" not in capsys.readouterr().out


class TestExplorationInstrumentation:
    """Phases 0-1 get first-class measures (user order 2026-08-11),
    integrated into the existing pack sections and metric registry."""

    def test_web_search_queries_captured_with_seat(self, tmp_path):
        import json as _json
        from alpha_lab.benchmarks.runcmp.extract import parse_agent_logs
        events = [
            {"type": "tool_call", "name": "web_search",
             "args": {"query": "lstm traffic forecasting sota"},
             "timestamp": 1.0},
            {"type": "api_response", "timestamp": 2.0, "usage": {}},
        ]
        (tmp_path / "phase1_x.jsonl").write_text(
            "\n".join(_json.dumps(e) for e in events) + "\n")
        out = parse_agent_logs(tmp_path)
        assert out["web_searches"] == [
            {"seat": "phase1", "query": "lstm traffic forecasting sota"}]

    def test_inventory_carries_exploration_products(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.extract import inventory_workspace
        (tmp_path / "learnings.md").write_text(
            "- first\n- second\n3. third\nprose\n")
        (tmp_path / "research_plan.md").write_text("do things\n")
        (tmp_path / "phase1").mkdir()
        (tmp_path / "phase1" / "data_report.md").write_text("x" * 100)
        (tmp_path / "plot.png").write_bytes(b"p")
        inv = inventory_workspace(tmp_path)
        e = inv["exploration"]
        assert e["learnings_bullets"] == 3
        assert e["learnings_bytes"] > 0
        assert "research_plan.md" in e["plan_files"]
        assert e["phase1_dir_files"] == 1
        assert e["phase1_dir_bytes"] == 100
        assert inv["classes"]["plots"] == 1

    def test_bench_registry_has_the_exploration_family(self):
        from alpha_lab.benchmarks.runcmp import bench
        ids = {m["id"] for m in bench.METRICS}
        for want in ("exploration.web_searches",
                     "exploration.web_searches_early",
                     "exploration.learnings_bytes",
                     "exploration.plan_bytes",
                     "exploration.plots",
                     "governance.supervisor_sessions"):
            assert want in ids, want
        pack = {"agent_logs": {"tools_by_role": {
                    "phase1_explorer:web_search": 7,
                    "strategist:web_search": 2,
                    "worker:shell_exec": 5},
                "seat_outcomes": {"by_role": {
                    "supervisor": {"seats_started": 4}}}},
                "inventory": {"exploration": {"learnings_bytes": 123,
                                              "plan_bytes": 45},
                              "classes": {"plots": 6}}}
        by_id = {m["id"]: m for m in bench.METRICS}
        assert by_id["exploration.web_searches"]["compute"](pack, None) == 9
        assert by_id["exploration.web_searches_early"]["compute"](pack, None) == 7
        assert by_id["exploration.learnings_bytes"]["compute"](pack, None) == 123
        assert by_id["governance.supervisor_sessions"]["compute"](pack, None) == 4

    def test_mission_standing_check_demands_early_phase_grading(
            self, synthetic_corpus, tmp_path, capsys):
        from alpha_lab.benchmarks.runcmp import corpus as corpus_mod
        from alpha_lab.benchmarks.runcmp.mission import build_mission
        cj = tmp_path / "corpus.json"
        corpus_mod.main(["--root", str(synthetic_corpus), "--out", str(cj)])
        capsys.readouterr()
        text = build_mission(cj, focus="harness")
        assert "Early phases must earn their hours" in text
        assert "web_searches" in text


class TestReportingInstrumentation:
    """The runs' written products and their readership become measurable
    (user ask 2026-08-11): who writes reports with what volume/structure,
    and which seats actually READ them mid-run."""

    def test_reads_of_written_products_are_classified(self, tmp_path):
        import json as _json
        from alpha_lab.benchmarks.runcmp.extract import parse_agent_logs
        events = [
            {"type": "tool_call", "name": "read_file",
             "args": {"path": "experiments/exp_a/debrief.md"}, "timestamp": 1},
            {"type": "tool_call", "name": "grep_file",
             "args": {"path": "learnings.md", "pattern": "x"}, "timestamp": 2},
            {"type": "tool_call", "name": "read_file",
             "args": {"path": "src/model.py"}, "timestamp": 3},
            {"type": "api_response", "timestamp": 4, "usage": {}},
        ]
        (tmp_path / "strategist_x.jsonl").write_text(
            "\n".join(_json.dumps(e) for e in events) + "\n")
        out = parse_agent_logs(tmp_path)
        assert out["artifact_reads"] == {"strategist:debrief": 1,
                                         "strategist:learnings": 1}

    def test_report_census_measures_volume_and_structure(self, tmp_path):
        from alpha_lab.benchmarks.runcmp.extract import inventory_workspace
        rep = tmp_path / "reports"
        rep.mkdir()
        (rep / "final_report.md").write_text(
            "# r\n\n| a | b |\n|---|---|\n| 1 | 2.5 |\n\n"
            "![plot](x.png)\n\nvalue 42 and 7\n")
        (tmp_path / "experiments/e1").mkdir(parents=True)
        (tmp_path / "experiments/e1/debrief.md").write_text("did 3 things\n")
        inv = inventory_workspace(tmp_path)
        r = inv["reports"]
        assert r["final"]["files"] == 1
        assert r["final"]["table_rows"] == 3
        assert r["final"]["images"] >= 1
        assert r["final"]["numbers"] >= 4
        assert r["debriefs"]["files"] == 1
        assert r["debriefs"]["bytes"] > 0

    def test_bench_reporting_family(self):
        from alpha_lab.benchmarks.runcmp import bench
        by_id = {m["id"]: m for m in bench.METRICS}
        pack = {"inventory": {"reports": {
                    "final": {"bytes": 500, "table_rows": 7, "numbers": 40,
                              "images": 2, "files": 1},
                    "debriefs": {"bytes": 900, "files": 3}}},
                "agent_logs": {"artifact_reads": {
                    "strategist:debrief": 5, "worker:debrief": 2,
                    "conductor:report": 1}}}
        assert by_id["reporting.report_bytes"]["compute"](pack, None) == 500
        assert by_id["reporting.debrief_reads"]["compute"](pack, None) == 7
        assert by_id["reporting.report_reads"]["compute"](pack, None) == 1
        assert by_id["reporting.learnings_reads"]["compute"](pack, None) == 0

    def test_mission_demands_readership_judgment(self, synthetic_corpus,
                                                 tmp_path, capsys):
        from alpha_lab.benchmarks.runcmp import corpus as corpus_mod
        from alpha_lab.benchmarks.runcmp.mission import build_mission
        cj = tmp_path / "corpus.json"
        corpus_mod.main(["--root", str(synthetic_corpus), "--out", str(cj)])
        capsys.readouterr()
        text = build_mission(cj, focus="harness")
        assert "Written products must find readers" in text


class TestRoleForLog:
    """Worker sub-roles must resolve; the old patterns were dead substrings
    (real names are worker_worker_0_implement_<slug>), collapsing every
    sub-role into "worker" and letting handoff (user-proxy) transcripts
    contaminate the worker ledger. Found by the 2026-08-13 seat-taxonomy
    study."""

    def test_worker_subroles_and_handoff_resolve(self):
        rfl = extract_mod.role_for_log
        assert rfl("worker_worker_0_implement_slug.jsonl") == "worker_implement"
        assert rfl("worker_worker_12_analyze_slug.jsonl.gz") == "worker_analyze"
        assert rfl("worker_worker_0_fix_cadence_window.jsonl") == "worker_fix"
        assert rfl("worker_worker_3_handoff_exp6.jsonl") == "handoff"
        assert rfl("verifier_worker_slug_r1.jsonl") == "verifier"
        assert rfl("strategist.jsonl") == "strategist"
        assert rfl("worker.jsonl") == "worker"
