"""Tests for the shared subprocess teardown helper."""

from __future__ import annotations

import signal
import subprocess
from unittest.mock import MagicMock

from alpha_lab.process_control import escalate_termination


def _live_proc() -> MagicMock:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 4321
    return proc


def test_skips_already_exited_process() -> None:
    proc = MagicMock()
    proc.poll.return_value = 0
    escalate_termination(proc, term_grace=0.01, kill_grace=0.01)
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


def test_cooperative_exit_avoids_signals() -> None:
    proc = _live_proc()
    proc.wait.return_value = 0  # exits within the cooperative grace
    escalate_termination(proc, term_grace=0.01, kill_grace=0.01, cooperative_grace=0.01)
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


def test_escalates_sigterm_then_sigkill() -> None:
    proc = _live_proc()
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=0.01)
    escalate_termination(proc, term_grace=0.01, kill_grace=0.01)
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


def test_process_group_uses_killpg(monkeypatch) -> None:
    sent: list[int] = []
    monkeypatch.setattr("alpha_lab.process_control.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("alpha_lab.process_control.os.killpg", lambda pgid, sig: sent.append(sig))
    proc = _live_proc()
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=0.01)
    escalate_termination(proc, term_grace=0.01, kill_grace=0.01, process_group=True)
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    proc.terminate.assert_not_called()  # group signalling, not single-PID
    proc.kill.assert_not_called()


def test_already_dead_signal_is_swallowed() -> None:
    proc = _live_proc()
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=0.01)
    proc.terminate.side_effect = ProcessLookupError()  # raced with natural exit
    escalate_termination(proc, term_grace=0.01, kill_grace=0.01)  # must not raise
    proc.kill.assert_called_once()
