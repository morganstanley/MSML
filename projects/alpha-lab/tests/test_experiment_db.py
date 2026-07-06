"""Tests for the ExperimentDB (SQLite kanban database)."""

from __future__ import annotations

import json
import threading
import time

import pytest

from pathlib import Path

from alpha_lab.experiment_db import (
    CANONICAL_RUN_COMPLETE_SENTINEL,
    KANBAN_COLUMNS,
    Experiment,
    ExperimentDB,
    is_smoke_result,
)


def _touch_sentinel(db: ExperimentDB, exp_name: str) -> None:
    """Create the canonical-run sentinel for a row."""
    workspace = Path(db.db_path).parent
    results = workspace / "experiments" / exp_name / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / CANONICAL_RUN_COMPLETE_SENTINEL).touch()


class TestExperimentDBCreate:
    """Test experiment creation."""

    def test_create_returns_id(self, db: ExperimentDB) -> None:
        exp_id = db.create("test_exp", "Test experiment", "It works", '{"key": 1}')
        assert exp_id == 1

    def test_create_increments_id(self, db: ExperimentDB) -> None:
        id1 = db.create("exp_a", "A", "H1", "{}")
        id2 = db.create("exp_b", "B", "H2", "{}")
        assert id2 == id1 + 1

    def test_create_default_status(self, db: ExperimentDB) -> None:
        exp_id = db.create("test_exp", "Desc", "Hyp", "{}")
        exp = db.get(exp_id)
        assert exp is not None
        assert exp.status == "to_implement"

    def test_create_stores_all_fields(self, db: ExperimentDB) -> None:
        exp_id = db.create("my_exp", "Full description", "Hypothesis here", '{"model": "lstm"}')
        exp = db.get(exp_id)
        assert exp is not None
        assert exp.name == "my_exp"
        assert exp.description == "Full description"
        assert exp.hypothesis == "Hypothesis here"
        assert exp.config_json == '{"model": "lstm"}'
        assert exp.worker_id is None
        assert exp.slurm_job_id is None
        assert exp.results_json is None
        assert exp.error is None
        assert exp.debrief_path is None
        assert exp.started_at is None
        assert exp.finished_at is None

    def test_create_sets_timestamps(self, db: ExperimentDB) -> None:
        before = time.time()
        exp_id = db.create("ts_exp", "D", "H", "{}")
        after = time.time()
        exp = db.get(exp_id)
        assert exp is not None
        assert before <= exp.created_at <= after
        assert before <= exp.updated_at <= after

    def test_create_duplicate_name_raises(self, db: ExperimentDB) -> None:
        db.create("dup_name", "First", "H", "{}")
        with pytest.raises(Exception):  # sqlite3.IntegrityError wrapped
            db.create("dup_name", "Second", "H", "{}")


class TestExperimentDBGet:
    """Test experiment retrieval."""

    def test_get_nonexistent_returns_none(self, db: ExperimentDB) -> None:
        assert db.get(9999) is None

    def test_get_returns_experiment(self, db: ExperimentDB) -> None:
        exp_id = db.create("get_exp", "D", "H", "{}")
        exp = db.get(exp_id)
        assert isinstance(exp, Experiment)
        assert exp.id == exp_id


class TestExperimentDBUpdateStatus:
    """Test status transitions."""

    def test_update_status_valid(self, db: ExperimentDB) -> None:
        exp_id = db.create("st_exp", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        exp = db.get(exp_id)
        assert exp is not None
        assert exp.status == "implemented"

    def test_update_status_invalid_raises(self, db: ExperimentDB) -> None:
        exp_id = db.create("inv_exp", "D", "H", "{}")
        with pytest.raises(ValueError, match="Invalid status"):
            db.update_status(exp_id, "nonexistent_status")

    def test_update_status_with_kwargs(self, db: ExperimentDB) -> None:
        exp_id = db.create("kw_exp", "D", "H", "{}")
        ts = time.time()
        db.update_status(exp_id, "running", started_at=ts)
        exp = db.get(exp_id)
        assert exp is not None
        assert exp.status == "running"
        assert exp.started_at == ts

    def test_update_status_disallowed_column_raises(self, db: ExperimentDB) -> None:
        exp_id = db.create("bad_col", "D", "H", "{}")
        with pytest.raises(ValueError, match="disallowed column"):
            db.update_status(exp_id, "implemented", name="hacked")

    def test_update_status_updates_timestamp(self, db: ExperimentDB) -> None:
        exp_id = db.create("ts_up", "D", "H", "{}")
        exp_before = db.get(exp_id)
        time.sleep(0.01)
        db.update_status(exp_id, "implemented")
        exp_after = db.get(exp_id)
        assert exp_after.updated_at > exp_before.updated_at

    def test_full_kanban_lifecycle(self, db: ExperimentDB) -> None:
        """Transition through all kanban columns."""
        exp_id = db.create("lifecycle", "D", "H", "{}")
        for status in KANBAN_COLUMNS[1:]:  # skip to_implement (already there)
            db.update_status(exp_id, status)
            exp = db.get(exp_id)
            assert exp.status == status


class TestExperimentDBWorkerAssignment:
    """Test worker assignment and release."""

    def test_assign_worker(self, db: ExperimentDB) -> None:
        exp_id = db.create("aw_exp", "D", "H", "{}")
        db.assign_worker(exp_id, "worker_0")
        exp = db.get(exp_id)
        assert exp.worker_id == "worker_0"

    def test_release_worker(self, db: ExperimentDB) -> None:
        exp_id = db.create("rw_exp", "D", "H", "{}")
        db.assign_worker(exp_id, "worker_0")
        db.release_worker(exp_id)
        exp = db.get(exp_id)
        assert exp.worker_id is None


class TestExperimentDBSlurmAndResults:
    """Test SLURM job tracking and results."""

    def test_set_slurm_job(self, db: ExperimentDB) -> None:
        exp_id = db.create("sj_exp", "D", "H", "{}")
        db.set_slurm_job(exp_id, "99999")
        exp = db.get(exp_id)
        assert exp.slurm_job_id == "99999"

    def test_set_results(self, db: ExperimentDB) -> None:
        exp_id = db.create("res_exp", "D", "H", "{}")
        results = '{"sharpe": 1.5, "mae": 0.02}'
        db.set_results(exp_id, results)
        exp = db.get(exp_id)
        assert exp.results_json == results
        parsed = json.loads(exp.results_json)
        assert parsed["sharpe"] == 1.5

    def test_set_error(self, db: ExperimentDB) -> None:
        exp_id = db.create("err_exp", "D", "H", "{}")
        db.set_error(exp_id, "OOM on H100")
        exp = db.get(exp_id)
        assert exp.error == "OOM on H100"


class TestExperimentDBQueries:
    """Test list/query methods."""

    def test_list_by_status(self, populated_db: ExperimentDB) -> None:
        to_impl = populated_db.list_by_status("to_implement")
        assert len(to_impl) == 1
        assert to_impl[0].name == "exp_xgboost_baseline"

    def test_list_by_multiple_statuses(self, populated_db: ExperimentDB) -> None:
        results = populated_db.list_by_status("analyzed", "done")
        assert len(results) == 2

    def test_list_by_status_empty(self, db: ExperimentDB) -> None:
        assert db.list_by_status("to_implement") == []

    def test_list_by_status_no_args(self, db: ExperimentDB) -> None:
        assert db.list_by_status() == []

    def test_list_all(self, populated_db: ExperimentDB) -> None:
        all_exps = populated_db.list_all()
        assert len(all_exps) == 7

    def test_list_all_ordered_by_created_at(self, populated_db: ExperimentDB) -> None:
        all_exps = populated_db.list_all()
        for i in range(len(all_exps) - 1):
            assert all_exps[i].created_at <= all_exps[i + 1].created_at

    def test_board_summary(self, populated_db: ExperimentDB) -> None:
        summary = populated_db.board_summary()
        assert summary.get("to_implement") == 1
        assert summary.get("implemented") == 1
        assert summary.get("checked") == 1
        assert summary.get("running") == 1
        assert summary.get("finished") == 1
        assert summary.get("analyzed") == 1
        assert summary.get("done") == 1

    def test_board_summary_empty(self, db: ExperimentDB) -> None:
        assert db.board_summary() == {}

    def test_count_active_gpus(self, populated_db: ExperimentDB) -> None:
        # One "running", one "queued" would count, but our populated DB has
        # running (id=4), so at least 1
        count = populated_db.count_active_gpus()
        assert count >= 1


class TestExperimentDBLeaderboard:
    """Test leaderboard sorting."""

    def test_leaderboard_sorted_by_sharpe(self, populated_db: ExperimentDB) -> None:
        leaders = populated_db.leaderboard("sharpe", 10, "maximize")
        assert len(leaders) >= 2
        # DeepAR (sharpe=2.1) should be first, then TCN (1.5), then PatchTST (0.8)
        sharpes = []
        for exp in leaders:
            m = json.loads(exp.results_json or "{}")
            sharpes.append(m.get("sharpe", float("-inf")))
        assert sharpes == sorted(sharpes, reverse=True)

    def test_leaderboard_top_n(self, populated_db: ExperimentDB) -> None:
        leaders = populated_db.leaderboard("sharpe", 1, "maximize")
        assert len(leaders) == 1
        m = json.loads(leaders[0].results_json or "{}")
        assert m["sharpe"] == 2.1  # DeepAR

    def test_leaderboard_invalid_metric(self, populated_db: ExperimentDB) -> None:
        # Should still return experiments, just with -inf sort
        leaders = populated_db.leaderboard("nonexistent_metric", 5, "maximize")
        assert len(leaders) >= 1

    def test_leaderboard_bad_json(self, db: ExperimentDB) -> None:
        exp_id = db.create("bad_json", "D", "H", "{}")
        db.set_results(exp_id, "not valid json")
        _touch_sentinel(db, "bad_json")
        leaders = db.leaderboard("sharpe", 10, "maximize")
        assert len(leaders) == 1  # Still returned, just sorted as -inf

    def test_leaderboard_nan_metric_sorts_to_bottom(self, db: ExperimentDB) -> None:
        """A NaN metric must not poison the sort order."""
        good_id = db.create("good", "D", "H", "{}")
        db.set_results(good_id, '{"sharpe": 1.0}')
        _touch_sentinel(db, "good")
        nan_id = db.create("nan_run", "D", "H", "{}")
        db.set_results(nan_id, '{"sharpe": NaN}')
        _touch_sentinel(db, "nan_run")
        leaders = db.leaderboard("sharpe", 10, "maximize")
        assert [e.name for e in leaders] == ["good", "nan_run"]

    def test_leaderboard_non_dict_results_sorts_to_bottom(self, db: ExperimentDB) -> None:
        """results_json that parses to a list/scalar is treated as -inf."""
        good_id = db.create("good", "D", "H", "{}")
        db.set_results(good_id, '{"sharpe": 0.5}')
        _touch_sentinel(db, "good")
        list_id = db.create("listy", "D", "H", "{}")
        db.set_results(list_id, '[1, 2, 3]')
        _touch_sentinel(db, "listy")
        leaders = db.leaderboard("sharpe", 10, "maximize")
        assert [e.name for e in leaders] == ["good", "listy"]

    def test_leaderboard_minimize_sorts_ascending(self, db: ExperimentDB) -> None:
        """Minimize direction puts smallest metric first, missing last."""
        a = db.create("low", "D", "H", "{}")
        db.set_results(a, '{"loss": 0.1}')
        b = db.create("high", "D", "H", "{}")
        db.set_results(b, '{"loss": 5.0}')
        c = db.create("mid", "D", "H", "{}")
        db.set_results(c, '{"loss": 1.0}')
        d = db.create("missing", "D", "H", "{}")
        db.set_results(d, '{"other": 9}')
        # Post-merge: leaderboard is gated by the canonical-run sentinel, so
        # these rows must have one to appear (preserves the direction-sort intent).
        for _n in ("low", "high", "mid", "missing"):
            _touch_sentinel(db, _n)
        leaders = db.leaderboard("loss", 10, "minimize")
        assert [e.name for e in leaders] == ["low", "mid", "high", "missing"]

    def test_leaderboard_invalid_direction_raises(self, db: ExperimentDB) -> None:
        with pytest.raises(ValueError, match="direction must be"):
            db.leaderboard("sharpe", 10, "ascending")


class TestSmokeResultDetection:
    """Tests for is_smoke_result() and leaderboard smoke filtering."""

    def test_smoke_test_true_detected(self) -> None:
        assert is_smoke_result('{"sharpe": 1.5, "smoke_test": true}') is True

    def test_smoke_test_false_not_detected(self) -> None:
        assert is_smoke_result('{"sharpe": 1.5, "smoke_test": false}') is False

    def test_no_smoke_flag_not_detected(self) -> None:
        assert is_smoke_result('{"sharpe": 1.5}') is False

    def test_none_not_detected(self) -> None:
        assert is_smoke_result(None) is False

    def test_empty_string_not_detected(self) -> None:
        assert is_smoke_result("") is False

    def test_bad_json_not_detected(self) -> None:
        assert is_smoke_result("not json") is False

    def test_non_dict_not_detected(self) -> None:
        assert is_smoke_result('[1, 2, 3]') is False

    def test_smoke_flagged_meta_not_sufficient(self) -> None:
        """The _smoke_flagged tag alone does not trigger — only smoke_test matters."""
        assert is_smoke_result('{"sharpe": 0.5, "_smoke_flagged": true}') is False

    def test_leaderboard_excludes_smoke(self, db: ExperimentDB) -> None:
        """Smoke-test results must never appear on the leaderboard, even
        if a sentinel exists."""
        good_id = db.create("full_run", "D", "H", "{}")
        db.set_results(good_id, '{"sharpe": 1.0}')
        _touch_sentinel(db, "full_run")
        smoke_id = db.create("smoke_run", "D", "H", "{}")
        db.set_results(smoke_id, '{"sharpe": 99.0, "smoke_test": true}')
        _touch_sentinel(db, "smoke_run")
        leaders = db.leaderboard("sharpe", 10, "maximize")
        names = [e.name for e in leaders]
        assert "full_run" in names
        assert "smoke_run" not in names

    def test_leaderboard_all_smoke_returns_empty(self, db: ExperimentDB) -> None:
        """If all results are smoke, leaderboard should be empty."""
        s1 = db.create("smoke_a", "D", "H", "{}")
        db.set_results(s1, '{"sharpe": 5.0, "smoke_test": true}')
        _touch_sentinel(db, "smoke_a")
        s2 = db.create("smoke_b", "D", "H", "{}")
        db.set_results(s2, '{"sharpe": 3.0, "smoke_test": true}')
        _touch_sentinel(db, "smoke_b")
        leaders = db.leaderboard("sharpe", 10, "maximize")
        assert leaders == []


class TestLeaderboardSentinelFilter:
    """Leaderboard inclusion is gated by the canonical-run sentinel file."""

    def test_leaderboard_excludes_rows_without_sentinel(self, db: ExperimentDB) -> None:
        """Rows without a sentinel file are excluded from the leaderboard."""
        good_id = db.create("full_run", "D", "H", "{}")
        db.set_results(good_id, '{"sharpe": 1.0}')
        _touch_sentinel(db, "full_run")
        # failed_canonical has smoke-tainted metrics — but NO sentinel.
        bad_id = db.create("failed_canonical", "D", "H", "{}")
        db.set_results(bad_id, '{"sharpe": 99.0}')
        leaders = db.leaderboard("sharpe", 10, "maximize")
        names = [e.name for e in leaders]
        assert "full_run" in names
        assert "failed_canonical" not in names

    def test_leaderboard_excludes_no_sentinel_even_without_smoke_flag(
        self, db: ExperimentDB,
    ) -> None:
        """Rows without a sentinel are excluded even without the smoke_test flag."""
        no_flag_id = db.create("smoke_no_flag", "D", "H", "{}")
        db.set_results(no_flag_id, '{"sharpe": 50.0}')
        good_id = db.create("good_run", "D", "H", "{}")
        db.set_results(good_id, '{"sharpe": 1.0}')
        _touch_sentinel(db, "good_run")
        leaders = db.leaderboard("sharpe", 10, "maximize")
        names = [e.name for e in leaders]
        assert "good_run" in names
        assert "smoke_no_flag" not in names

    def test_leaderboard_keeps_rows_with_narrative_error_when_sentinel_exists(
        self, db: ExperimentDB,
    ) -> None:
        """Rows with a sentinel are included regardless of `error` column content."""
        exp_id = db.create("ok_row", "D", "H", "{}")
        db.set_results(exp_id, '{"sharpe": 1.5}')
        db.set_error(exp_id, "Fixed: fixed a typo in config.yaml")
        _touch_sentinel(db, "ok_row")
        leaders = db.leaderboard("sharpe", 10, "maximize")
        names = [e.name for e in leaders]
        assert "ok_row" in names


class TestExperimentDBListExperiments:
    """Test paginated list_experiments query."""

    def test_pagination_respects_limit(self, populated_db: ExperimentDB) -> None:
        exps, total = populated_db.list_experiments(limit=3, offset=0)
        assert len(exps) == 3
        assert total == 7

    def test_pagination_offset(self, populated_db: ExperimentDB) -> None:
        first, _ = populated_db.list_experiments(limit=3, offset=0)
        second, _ = populated_db.list_experiments(limit=3, offset=3)
        first_ids = {e.id for e in first}
        second_ids = {e.id for e in second}
        assert first_ids.isdisjoint(second_ids)

    def test_pagination_offset_past_end_returns_empty(self, populated_db: ExperimentDB) -> None:
        exps, total = populated_db.list_experiments(limit=10, offset=100)
        assert exps == []
        assert total == 7  # total is pre-pagination

    def test_status_filter(self, populated_db: ExperimentDB) -> None:
        exps, total = populated_db.list_experiments(status_filter="done")
        assert total == 1
        assert all(e.status == "done" for e in exps)

    def test_name_search(self, populated_db: ExperimentDB) -> None:
        exps, total = populated_db.list_experiments(name_search="tcn")
        assert total == 1
        assert exps[0].name == "exp_tcn_v1"

    def test_pagination_stable_on_updated_at_ties(self, db: ExperimentDB) -> None:
        """Pagination must not skip or double-count rows when updated_at ties."""
        import sqlite3
        for i in range(10):
            db.create(f"exp_tie_{i}", "D", "H", "{}")
        # Force identical updated_at across all rows to simulate the case
        # where time.time() granularity ties multiple rows.
        conn = sqlite3.connect(db.db_path, timeout=10)
        conn.execute("UPDATE experiments SET updated_at = 1000.0")
        conn.commit()
        conn.close()
        seen_ids: set[int] = set()
        for offset in range(0, 10, 3):
            exps, _ = db.list_experiments(limit=3, offset=offset)
            for exp in exps:
                assert exp.id not in seen_ids, f"id={exp.id} returned twice"
                seen_ids.add(exp.id)
        assert len(seen_ids) == 10

    def test_leaderboard_infinity_metric(self, db: ExperimentDB) -> None:
        """+/-Infinity must round-trip through the sort without breaking order."""
        good_id = db.create("mid", "D", "H", "{}")
        db.set_results(good_id, '{"sharpe": 1.0}')
        _touch_sentinel(db, "mid")
        inf_id = db.create("top", "D", "H", "{}")
        db.set_results(inf_id, '{"sharpe": Infinity}')
        _touch_sentinel(db, "top")
        neg_id = db.create("bottom", "D", "H", "{}")
        db.set_results(neg_id, '{"sharpe": -Infinity}')
        _touch_sentinel(db, "bottom")
        leaders = db.leaderboard("sharpe", 10, "maximize")
        assert [e.name for e in leaders] == ["top", "mid", "bottom"]


class TestExperimentDBStaleWorkers:
    """Test stale worker detection."""

    def test_stale_workers_detected(self, db: ExperimentDB) -> None:
        exp_id = db.create("stale_exp", "D", "H", "{}")
        db.assign_worker(exp_id, "worker_0")
        # Manually set updated_at to the past
        import sqlite3
        conn = sqlite3.connect(db.db_path, timeout=10)
        conn.execute(
            "UPDATE experiments SET updated_at = ? WHERE id = ?",
            (time.time() - 3600, exp_id),  # 1 hour ago
        )
        conn.commit()
        conn.close()

        stale = db.stale_workers(timeout_s=300)
        assert len(stale) == 1
        assert stale[0].id == exp_id

    def test_stale_workers_fresh_not_detected(self, db: ExperimentDB) -> None:
        exp_id = db.create("fresh_exp", "D", "H", "{}")
        db.assign_worker(exp_id, "worker_0")
        stale = db.stale_workers(timeout_s=300)
        assert len(stale) == 0


class TestExperimentDBThreadSafety:
    """Test thread-safe write serialization."""

    def test_concurrent_creates(self, db: ExperimentDB) -> None:
        """Multiple threads creating experiments concurrently should not corrupt."""
        errors: list[Exception] = []
        ids: list[int] = []
        lock = threading.Lock()

        def create_one(n: int) -> None:
            try:
                exp_id = db.create(f"concurrent_{n}", f"Desc {n}", "H", "{}")
                with lock:
                    ids.append(exp_id)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=create_one, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent creates: {errors}"
        assert len(ids) == 20
        assert len(set(ids)) == 20  # All unique IDs

    def test_concurrent_status_updates(self, db: ExperimentDB) -> None:
        """Multiple threads updating different experiments concurrently."""
        exp_ids = [db.create(f"conc_upd_{i}", f"D{i}", "H", "{}") for i in range(10)]
        errors: list[Exception] = []

        def update_one(exp_id: int) -> None:
            try:
                db.update_status(exp_id, "implemented")
                db.update_status(exp_id, "checked")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_one, args=(eid,)) for eid in exp_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for eid in exp_ids:
            assert db.get(eid).status == "checked"
