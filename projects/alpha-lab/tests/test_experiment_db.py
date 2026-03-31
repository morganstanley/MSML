"""Tests for the ExperimentDB (SQLite kanban database)."""

from __future__ import annotations

import json
import threading
import time

import pytest

from alpha_lab.experiment_db import KANBAN_COLUMNS, Experiment, ExperimentDB


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
        leaders = populated_db.leaderboard("sharpe", 10)
        assert len(leaders) >= 2
        # DeepAR (sharpe=2.1) should be first, then TCN (1.5), then PatchTST (0.8)
        sharpes = []
        for exp in leaders:
            m = json.loads(exp.results_json or "{}")
            sharpes.append(m.get("sharpe", float("-inf")))
        assert sharpes == sorted(sharpes, reverse=True)

    def test_leaderboard_top_n(self, populated_db: ExperimentDB) -> None:
        leaders = populated_db.leaderboard("sharpe", 1)
        assert len(leaders) == 1
        m = json.loads(leaders[0].results_json or "{}")
        assert m["sharpe"] == 2.1  # DeepAR

    def test_leaderboard_invalid_metric(self, populated_db: ExperimentDB) -> None:
        # Should still return experiments, just with -inf sort
        leaders = populated_db.leaderboard("nonexistent_metric", 5)
        assert len(leaders) >= 1

    def test_leaderboard_bad_json(self, db: ExperimentDB) -> None:
        exp_id = db.create("bad_json", "D", "H", "{}")
        db.set_results(exp_id, "not valid json")
        leaders = db.leaderboard("sharpe", 10)
        assert len(leaders) == 1  # Still returned, just sorted as -inf


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
