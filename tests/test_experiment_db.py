"""Tests for the ExperimentDB (SQLite kanban database)."""

from __future__ import annotations

import json
import threading
import time

import pytest

from alpha_lab.experiment_db import (
    KANBAN_COLUMNS,
    Experiment,
    ExperimentDB,
    is_execution_failure,
)


@pytest.mark.parametrize(
    ("error", "results_json", "expected"),
    [
        (None, None, False),
        ("runtime failed", None, True),
        ("runtime failed", "", True),
        ("did not beat baseline", '{"score": 0.1}', False),
        (None, '{"score": 0.1}', False),
    ],
)
def test_is_execution_failure(
    error: str | None,
    results_json: str | None,
    expected: bool,
) -> None:
    assert is_execution_failure(error, results_json) is expected


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
        # Forward-only guard requires walking the canonical chain to reach
        # ``running``; jumping directly from to_implement -> running is no
        # longer allowed.
        for s in ("implemented", "checked", "queued"):
            db.update_status(exp_id, s)
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

    def test_update_status_returns_applied_for_valid_forward_move(
        self, db: ExperimentDB
    ) -> None:
        exp_id = db.create("ok", "D", "H", "{}")
        assert db.update_status(exp_id, "implemented") == "applied"

    def test_update_status_returns_idempotent_for_same_status(
        self, db: ExperimentDB
    ) -> None:
        exp_id = db.create("idem", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        # Re-emitting the same transition is a no-op the caller can detect.
        assert db.update_status(exp_id, "implemented") == "idempotent"

    def test_update_status_returns_blocked_for_backwards_move(
        self, db: ExperimentDB
    ) -> None:
        exp_id = db.create("back", "D", "H", "{}")
        for s in ("implemented", "checked", "queued"):
            db.update_status(exp_id, s)
        # queued -> checked is NOT a forward edge — should be blocked.
        outcome = db.update_status(exp_id, "checked")
        assert outcome.startswith("blocked:")
        assert outcome.endswith(":queued")
        # Row must stay at queued.
        assert db.get(exp_id).status == "queued"

    def test_full_kanban_lifecycle(self, db: ExperimentDB) -> None:
        """Transition through the canonical forward chain.

        The forward-only guard treats ``done`` and ``cancelled`` as
        terminal, so we walk the standard happy-path chain instead of
        iterating KANBAN_COLUMNS in tuple order (which would attempt the
        forbidden ``done -> cancelled`` move).
        """
        exp_id = db.create("lifecycle", "D", "H", "{}")
        chain = ("implemented", "checked", "queued", "running",
                 "finished", "analyzed", "done")
        for status in chain:
            db.update_status(exp_id, status)
            exp = db.get(exp_id)
            assert exp.status == status

    def test_update_status_idempotent_same_status_is_noop(
        self, db: ExperimentDB
    ) -> None:
        """Repeated same-status calls are no-ops (no UPDATE, no timestamp bump)."""
        exp_id = db.create("idem", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        before = db.get(exp_id)
        assert before is not None
        time.sleep(0.01)
        db.update_status(exp_id, "implemented")  # duplicate call
        after = db.get(exp_id)
        assert after is not None
        assert after.status == "implemented"
        assert after.updated_at == before.updated_at

    def test_update_status_backwards_is_silent_noop(
        self, db: ExperimentDB
    ) -> None:
        """A backwards transition silently fails (logged, not raised) so the
        worker LLM's redundant update_experiment("checked") cannot bounce an
        in-flight experiment back to checked and trigger a duplicate submit.
        """
        exp_id = db.create("backwards", "D", "H", "{}")
        for s in ("implemented", "checked", "queued", "running"):
            db.update_status(exp_id, s)
        # Backwards: running -> checked is the exact LLM-driven cascade we
        # observed in workspace-etfflows-g55.
        db.update_status(exp_id, "checked")
        exp = db.get(exp_id)
        assert exp is not None
        assert exp.status == "running"  # unchanged

    def test_update_status_finished_to_checked_is_allowed(
        self, db: ExperimentDB
    ) -> None:
        """Fixer's finished -> checked path must remain open so retries work."""
        exp_id = db.create("fixer_path", "D", "H", "{}")
        for s in ("implemented", "checked", "queued", "running", "finished"):
            db.update_status(exp_id, s)
        db.update_status(exp_id, "checked")
        exp = db.get(exp_id)
        assert exp is not None
        assert exp.status == "checked"

    def test_update_status_cancel_from_any_pending_state(
        self, db: ExperimentDB
    ) -> None:
        """Strategist can cancel from any non-terminal state."""
        for src in ("to_implement", "implemented", "checked", "queued", "running"):
            exp_id = db.create(f"cancel_from_{src}", "D", "H", "{}")
            # Walk forward to ``src``
            chain = ("implemented", "checked", "queued", "running")
            for s in chain:
                if s == src:
                    break
                db.update_status(exp_id, s)
            # to_implement is the starting state; otherwise move into ``src``
            if src != "to_implement":
                db.update_status(exp_id, src)
            db.update_status(exp_id, "cancelled")
            exp = db.get(exp_id)
            assert exp is not None
            assert exp.status == "cancelled", f"failed from {src}"


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
        assert db.set_results(exp_id, results) == "applied"
        exp = db.get(exp_id)
        assert exp.results_json == results
        parsed = json.loads(exp.results_json)
        assert parsed["sharpe"] == 1.5

    def test_set_results_refuses_disk_conflicting_metric(
            self, db: ExperimentDB) -> None:
        """The payload usually arrives through the worker LLM, which can
        retype or round numbers; the board must never disagree with the
        experiment's own results/metrics.json (audited runs carried board
        rows conflicting with their artifacts)."""
        from pathlib import Path as _P
        exp_id = db.create("consist_exp", "d", "h", "{}")
        rd = _P(db.db_path).parent / "experiments" / "consist_exp" / "results"
        rd.mkdir(parents=True)
        (rd / "metrics.json").write_text(
            '{"val_bpb": 0.522081, "run_scope": "full"}')
        out = db.set_results(
            exp_id, '{"val_bpb": 0.52, "run_scope": "full"}')
        assert out.startswith("refused:metric_conflicts_with_disk")
        # verbatim numbers pass; extra derived keys are fine
        out = db.set_results(
            exp_id,
            '{"val_bpb": 0.522081, "extra_derived": 1.5, "run_scope": "full"}')
        assert out.startswith("applied")

    def test_set_results_no_file_no_opinion(self, db: ExperimentDB) -> None:
        exp_id = db.create("nofile_exp", "d", "h", "{}")
        assert db.set_results(
            exp_id, '{"sharpe": 1.2, "run_scope": "full"}'
        ).startswith("applied")

    def test_set_results_refuses_smoke_payload(self, db: ExperimentDB) -> None:
        exp_id = db.create("smoke_exp", "D", "H", "{}")
        outcome = db.set_results(
            exp_id, '{"sharpe": 8.9, "smoke": true}'
        )
        assert outcome == "refused:smoke=true"
        assert db.get(exp_id).results_json is None

    def test_set_results_refuses_partial_run_scope(self, db: ExperimentDB) -> None:
        exp_id = db.create("partial_exp", "D", "H", "{}")
        outcome = db.set_results(
            exp_id, '{"sharpe": 5.0, "run_scope": "smoke"}'
        )
        assert outcome.startswith("refused:run_scope=")
        assert db.get(exp_id).results_json is None

    def test_set_results_refuses_smoke_complete_status(self, db: ExperimentDB) -> None:
        exp_id = db.create("smoke_status_exp", "D", "H", "{}")
        outcome = db.set_results(
            exp_id, '{"sharpe": 3.0, "status": "smoke_complete"}'
        )
        assert outcome.startswith("refused:status=")
        assert db.get(exp_id).results_json is None

    def test_set_results_refuses_canonical_full_run_false(
        self, db: ExperimentDB
    ) -> None:
        # Workers that detect a missing precondition (failed source gate,
        # data unavailable, etc.) sometimes write a results payload with
        # numeric fields plus ``canonical_full_run: false`` to advertise
        # the abort. Must NOT land on the leaderboard or auto-promote.
        exp_id = db.create("gate_blocked_exp", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        outcome = db.set_results(
            exp_id,
            '{"sharpe": 1.2, "canonical_full_run": false, '
            '"canonical_launch_status": "blocked_by_source_gate"}',
        )
        assert outcome == "refused:canonical_full_run=false"
        exp = db.get(exp_id)
        assert exp.status == "implemented"
        assert exp.results_json is None

    def test_set_results_accepts_explicit_canonical_full_run_true(
        self, db: ExperimentDB
    ) -> None:
        exp_id = db.create("ok_full_exp", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        outcome = db.set_results(
            exp_id, '{"sharpe": 2.5, "canonical_full_run": true}'
        )
        assert outcome == "applied:promoted"
        assert db.get(exp_id).status == "finished"

    def test_set_results_accepts_missing_canonical_full_run(
        self, db: ExperimentDB
    ) -> None:
        # Backward compat: adapters that don't set the field at all are
        # still accepted (treated as canonical), preserving behavior for
        # domains without a gating workflow.
        exp_id = db.create("legacy_payload_exp", "D", "H", "{}")
        outcome = db.set_results(exp_id, '{"sharpe": 4.0, "mae": 0.01}')
        assert outcome == "applied"
        assert db.get(exp_id).results_json is not None

    def test_set_results_accepts_run_scope_full(self, db: ExperimentDB) -> None:
        exp_id = db.create("full_exp", "D", "H", "{}")
        payload = '{"sharpe": 2.0, "run_scope": "full"}'
        assert db.set_results(exp_id, payload) == "applied"
        assert db.get(exp_id).results_json == payload

    def test_set_results_refuses_invalid_json(self, db: ExperimentDB) -> None:
        # Truncated / non-JSON payloads have surfaced in production
        # (workspace_etfflow_cond had one row with results_json broken at
        # char 1400). Reject at the choke-point rather than store bytes
        # downstream consumers will all fail to parse.
        exp_id = db.create("raw_exp", "D", "H", "{}")
        outcome = db.set_results(exp_id, "not valid json")
        assert outcome.startswith("refused:invalid_json"), outcome
        # The refused write must not have stored the bad payload.
        assert db.get(exp_id).results_json is None

    def test_set_results_refuses_empty_dict(self, db: ExperimentDB) -> None:
        # `{}` parses but contains no real metrics. 150 such rows ended up
        # marked analyzed in workspace_etfflow_g55, contributing nothing
        # to the leaderboard but looking completed.
        exp_id = db.create("empty_exp", "D", "H", "{}")
        outcome = db.set_results(exp_id, "{}")
        assert outcome.startswith("refused:empty_results"), outcome

    def test_set_results_refuses_bookkeeping_only(self, db: ExperimentDB) -> None:
        # Payload with only bookkeeping keys (no actual metric) must be
        # refused — it has no leaderboard value.
        exp_id = db.create("bk_exp", "D", "H", "{}")
        outcome = db.set_results(exp_id, '{"run_scope": "full", "wall_seconds": 12}')
        assert outcome.startswith("refused:empty_results"), outcome

    def test_set_results_auto_promotes_implemented_to_finished(
        self, db: ExperimentDB
    ) -> None:
        # A canonical-results write on a row stuck at ``implemented`` must
        # flip the status to ``finished`` so the analyzer can pick it up.
        # Without this, workers thrash on dead-end transitions.
        exp_id = db.create("stuck_impl", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        outcome = db.set_results(exp_id, '{"sharpe": 1.5, "mae": 0.02}')
        assert outcome == "applied:promoted"
        exp = db.get(exp_id)
        assert exp.status == "finished"
        assert exp.finished_at is not None

    def test_set_results_auto_promotes_checked_to_finished(
        self, db: ExperimentDB
    ) -> None:
        exp_id = db.create("stuck_chk", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        outcome = db.set_results(exp_id, '{"sharpe": 2.0}')
        assert outcome == "applied:promoted"
        assert db.get(exp_id).status == "finished"

    def test_set_results_does_not_promote_checked_row_with_error(
        self, db: ExperimentDB,
    ) -> None:
        exp_id = db.create("quarantined", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        db.set_error(exp_id, "quarantined_invalid_split")

        outcome = db.set_results(
            exp_id, '{"sharpe": 2.0}', refuse_if_error=True,
        )

        assert outcome == "refused:row_has_error"
        exp = db.get(exp_id)
        assert exp.status == "checked"
        assert exp.results_json is None

    def test_set_results_does_not_promote_to_implement(
        self, db: ExperimentDB
    ) -> None:
        # Rows that haven't even been implemented yet are not zombie rows.
        # Canonical results landing here is unusual — record them, but
        # don't auto-advance status.
        exp_id = db.create("fresh_exp", "D", "H", "{}")
        outcome = db.set_results(exp_id, '{"sharpe": 3.0}')
        assert outcome == "applied"
        assert db.get(exp_id).status == "to_implement"

    def test_set_results_smoke_refused_keeps_status(
        self, db: ExperimentDB
    ) -> None:
        # A smoke/partial payload must NOT trigger auto-promotion.
        exp_id = db.create("smoke_stuck", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        outcome = db.set_results(exp_id, '{"sharpe": 9.9, "smoke": true}')
        assert outcome.startswith("refused:")
        # Status unchanged, no results written.
        exp = db.get(exp_id)
        assert exp.status == "implemented"
        assert exp.results_json is None

    def test_set_results_does_not_promote_from_running_or_finished(
        self, db: ExperimentDB
    ) -> None:
        # Auto-promotion is intentionally limited to implemented/checked.
        # Rows already at running/finished/analyzed don't need rescue and
        # shouldn't have their finished_at clobbered by a late re-write.
        exp_id = db.create("already_running", "D", "H", "{}")
        for s in ("implemented", "checked", "queued", "running"):
            db.update_status(exp_id, s)
        outcome = db.set_results(exp_id, '{"sharpe": 1.0}')
        assert outcome == "applied"
        assert db.get(exp_id).status == "running"

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

    def test_active_to_implement_count_excludes_parked(self, db: ExperimentDB) -> None:
        # Three to_implement rows, two of them parked.
        a = db.create("a", "", "", "{}")
        b = db.create("b", "", "", "{}")
        c = db.create("c", "", "", "{}")
        db.park(b)
        db.park(c)
        assert db.board_summary().get("to_implement") == 3
        assert db.active_to_implement_count() == 1

    def test_active_to_implement_count_zero_when_all_parked(self, db: ExperimentDB) -> None:
        a = db.create("a", "", "", "{}")
        db.park(a)
        assert db.active_to_implement_count() == 0

    def test_active_to_implement_count_zero_when_no_rows(self, db: ExperimentDB) -> None:
        assert db.active_to_implement_count() == 0

    def test_active_to_implement_count_ignores_non_pending_statuses(
        self, db: ExperimentDB
    ) -> None:
        # Only counts status='to_implement'. Other statuses (running, finished,
        # analyzed) must not be counted even if parked_at IS NULL.
        a = db.create("a", "", "", "{}")
        b = db.create("b", "", "", "{}")
        db.create("c", "", "", "{}")
        db.update_status(a, "implemented")
        db.update_status(b, "implemented")
        db.update_status(b, "running")
        # c stays to_implement
        assert db.active_to_implement_count() == 1

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
        # set_results now refuses invalid JSON outright, so the leaderboard
        # never has to handle malformed results_json. Confirm the refusal
        # leaves the original row (with the create() placeholder "{}")
        # findable by the leaderboard scan, which then treats it as -inf
        # because no `sharpe` key exists.
        exp_id = db.create("bad_json", "D", "H", "{}")
        outcome = db.set_results(exp_id, "not valid json")
        assert outcome.startswith("refused:invalid_json"), outcome
        # Row still has results_json="{}" from create(); leaderboard scan
        # is leniently and returns 0 rows because empty-dict has no sharpe.
        # (Both behaviours are acceptable; what matters is no exception.)
        leaders = db.leaderboard("sharpe", 10)
        assert isinstance(leaders, list)

    def test_leaderboard_nan_metric_sorts_to_bottom(self, db: ExperimentDB) -> None:
        """A NaN metric must not poison the sort order."""
        good_id = db.create("good", "D", "H", "{}")
        db.set_results(good_id, '{"sharpe": 1.0}')
        nan_id = db.create("nan_run", "D", "H", "{}")
        db.set_results(nan_id, '{"sharpe": NaN}')
        leaders = db.leaderboard("sharpe", 10)
        assert [e.name for e in leaders] == ["good", "nan_run"]

    def test_leaderboard_non_dict_results_sorts_to_bottom(self, db: ExperimentDB) -> None:
        """results_json that parses to a list/scalar is treated as -inf."""
        good_id = db.create("good", "D", "H", "{}")
        db.set_results(good_id, '{"sharpe": 0.5}')
        list_id = db.create("listy", "D", "H", "{}")
        db.set_results(list_id, '[1, 2, 3]')
        leaders = db.leaderboard("sharpe", 10)
        assert [e.name for e in leaders] == ["good", "listy"]


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
        inf_id = db.create("top", "D", "H", "{}")
        db.set_results(inf_id, '{"sharpe": Infinity}')
        neg_id = db.create("bottom", "D", "H", "{}")
        db.set_results(neg_id, '{"sharpe": -Infinity}')
        leaders = db.leaderboard("sharpe", 10)
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


class TestExperimentDBConductorFields:
    """Conductor-administered priority and parked_at columns."""

    def test_default_priority_is_zero(self, db: ExperimentDB) -> None:
        exp_id = db.create("p0", "D", "H", "{}")
        exp = db.get(exp_id)
        assert exp is not None
        assert exp.priority == 0
        assert exp.parked_at is None

    def test_set_priority(self, db: ExperimentDB) -> None:
        exp_id = db.create("p1", "D", "H", "{}")
        db.set_priority(exp_id, 5)
        exp = db.get(exp_id)
        assert exp.priority == 5

    def test_set_priority_negative(self, db: ExperimentDB) -> None:
        exp_id = db.create("p2", "D", "H", "{}")
        db.set_priority(exp_id, -3)
        exp = db.get(exp_id)
        assert exp.priority == -3

    def test_park_sets_parked_at(self, db: ExperimentDB) -> None:
        exp_id = db.create("park1", "D", "H", "{}")
        before = time.time()
        db.park(exp_id)
        after = time.time()
        exp = db.get(exp_id)
        assert exp.parked_at is not None
        assert before <= exp.parked_at <= after

    def test_unpark_clears_parked_at(self, db: ExperimentDB) -> None:
        exp_id = db.create("park2", "D", "H", "{}")
        db.park(exp_id)
        db.unpark(exp_id)
        exp = db.get(exp_id)
        assert exp.parked_at is None

    def test_clear_block_removes_stale_blocked_error(self, db: ExperimentDB) -> None:
        # An implemented row self-stranded by a stale precondition flag: clearing
        # the error requeues it in place (status/economics untouched).
        exp_id = db.create("blk1", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.set_error(exp_id, "blocked: shared fixture 169/170 failing")
        assert db.clear_block(exp_id) is True
        exp = db.get(exp_id)
        assert exp.error is None
        assert exp.status == "implemented"

    def test_clear_block_also_works_for_to_implement(self, db: ExperimentDB) -> None:
        exp_id = db.create("blk2", "D", "H", "{}")  # default status: to_implement
        db.set_error(exp_id, "  blocked: hypothesis pending a dependency")
        assert db.clear_block(exp_id) is True
        assert db.get(exp_id).error is None

    def test_clear_block_noop_when_not_blocked(self, db: ExperimentDB) -> None:
        exp_id = db.create("blk3", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        assert db.clear_block(exp_id) is False  # no blocked: error -> nothing to do

    def test_clear_block_preserves_a_real_error(self, db: ExperimentDB) -> None:
        # A genuine failure (not a blocked: precondition) must NOT be cleared.
        exp_id = db.create("blk4", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.set_error(exp_id, "Traceback (most recent call last): boom")
        assert db.clear_block(exp_id) is False
        assert db.get(exp_id).error == "Traceback (most recent call last): boom"

    def test_active_park_is_coordinated_and_late_failure_is_ignored(
        self, db: ExperimentDB,
    ) -> None:
        exp_id = db.create("active_park", "D", "H", "{}")
        db.update_status(exp_id, "implemented")
        db.update_status(exp_id, "checked")
        db.set_slurm_job(exp_id, "cpu-active")
        db.update_status(exp_id, "queued")
        db.update_status(exp_id, "running")

        assert db.park(exp_id) == "cancelling"
        assert db.unpark(exp_id) is False
        assert db.set_error_and_finish(exp_id, "job FAILED") is False
        exp = db.get(exp_id)
        assert exp is not None
        assert exp.status == "cancelling"
        assert exp.parked_at is not None

        assert db.finalize_park(exp_id) is True
        exp = db.get(exp_id)
        assert exp is not None
        assert exp.status == "checked"
        assert exp.parked_at is not None
        assert exp.slurm_job_id is None
        assert exp.error is None

        assert db.unpark(exp_id) is True
        assert db.get(exp_id).parked_at is None

    def test_list_by_status_excludes_parked_by_default(self, db: ExperimentDB) -> None:
        a = db.create("a", "D", "H", "{}")
        b = db.create("b", "D", "H", "{}")
        db.park(b)
        result = db.list_by_status("to_implement")
        assert [e.id for e in result] == [a]

    def test_list_by_status_include_parked(self, db: ExperimentDB) -> None:
        a = db.create("a", "D", "H", "{}")
        b = db.create("b", "D", "H", "{}")
        db.park(b)
        result = db.list_by_status("to_implement", include_parked=True)
        assert {e.id for e in result} == {a, b}

    def test_list_by_status_priority_orders_above_created_at(self, db: ExperimentDB) -> None:
        # Three created in order; second one gets bumped priority and should
        # surface to the top of the queue regardless of created_at.
        first = db.create("first", "D", "H", "{}")
        second = db.create("second", "D", "H", "{}")
        third = db.create("third", "D", "H", "{}")
        db.set_priority(second, 10)
        result = db.list_by_status("to_implement")
        assert [e.id for e in result] == [second, first, third]

    def test_list_by_status_default_priority_falls_back_to_created_at(
        self, db: ExperimentDB
    ) -> None:
        # Without any priority overrides, ordering must reduce to created_at ASC
        # — the same ordering the dispatcher relied on before the Conductor
        # existed. This is the no_conductor=True invariant.
        ids = [db.create(f"e{i}", "D", "H", "{}") for i in range(5)]
        result = db.list_by_status("to_implement")
        assert [e.id for e in result] == ids

    def test_list_by_status_priority_ties_break_on_created_at(
        self, db: ExperimentDB
    ) -> None:
        a = db.create("a", "D", "H", "{}")
        b = db.create("b", "D", "H", "{}")
        c = db.create("c", "D", "H", "{}")
        db.set_priority(a, 5)
        db.set_priority(c, 5)  # Same priority as a; b stays at 0
        result = db.list_by_status("to_implement")
        # a and c (priority=5) come first in created_at order; b last
        assert [e.id for e in result] == [a, c, b]

    def test_park_during_running_status_does_not_change_status(
        self, db: ExperimentDB
    ) -> None:
        exp_id = db.create("run_park", "D", "H", "{}")
        # Walk the canonical kanban chain — the forward-only guard
        # forbids a direct to_implement -> running shortcut.
        for s in ("implemented", "checked", "queued", "running"):
            db.update_status(exp_id, s)
        db.park(exp_id)
        exp = db.get(exp_id)
        assert exp.status == "running"
        assert exp.parked_at is not None

    def test_migration_on_existing_db_without_priority_columns(
        self, tmp_path
    ) -> None:
        """An existing DB created without priority/parked_at must be migrated
        cleanly: queries succeed, defaults applied."""
        import sqlite3
        db_path = str(tmp_path / "legacy.db")
        # Create a DB with the OLD schema (no priority, no parked_at)
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                hypothesis TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'to_implement',
                config_json TEXT NOT NULL DEFAULT '{}',
                worker_id TEXT,
                slurm_job_id TEXT,
                results_json TEXT,
                error TEXT,
                debrief_path TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL
            )
        """)
        now = time.time()
        conn.execute(
            "INSERT INTO experiments (name, description, created_at, updated_at) "
            "VALUES ('legacy', 'D', ?, ?)",
            (now, now),
        )
        conn.commit()
        conn.close()
        # Now open it via ExperimentDB — migration should run silently
        db = ExperimentDB(db_path)
        exp = db.get(1)
        assert exp is not None
        assert exp.priority == 0
        assert exp.parked_at is None
        # And subsequent operations work
        db.set_priority(1, 7)
        assert db.get(1).priority == 7
        db.park(1)
        assert db.get(1).parked_at is not None


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


class TestExperimentDBVariantParentId:
    """parent_id column for variants spawned via propose_variant.

    Default NULL preserves legacy semantics; the optional kwarg on create
    sets it; count_variants_of returns the right number for fan-out caps.
    """

    def test_create_default_parent_id_is_none(self, db: ExperimentDB) -> None:
        exp_id = db.create("plain", "D", "H", "{}")
        exp = db.get(exp_id)
        assert exp is not None
        assert exp.parent_id is None

    def test_create_with_parent_id(self, db: ExperimentDB) -> None:
        base = db.create("base", "D", "H", "{}")
        var = db.create("variant", "D", "H", "{}", parent_id=base)
        assert db.get(var).parent_id == base
        assert db.get(base).parent_id is None  # base untouched

    def test_count_variants_of_zero_for_unknown_base(
        self, db: ExperimentDB
    ) -> None:
        assert db.count_variants_of(9999) == 0

    def test_count_variants_of_returns_count(self, db: ExperimentDB) -> None:
        base = db.create("base", "D", "H", "{}")
        for i in range(3):
            db.create(f"v_{i}", "D", "H", "{}", parent_id=base)
        # A different base's variant should not be counted.
        other_base = db.create("other", "D", "H", "{}")
        db.create("other_v", "D", "H", "{}", parent_id=other_base)
        assert db.count_variants_of(base) == 3
        assert db.count_variants_of(other_base) == 1
