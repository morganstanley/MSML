"""Shared subprocess teardown: optional cooperative wait → SIGTERM → SIGKILL, reaping.

Used by the local executors (process-group kill of an experiment job) and the agent
sandbox (single-PID kill of a bwrap child, after a cooperative stop). The escalation
skeleton and zombie-reaping are identical; only the signal target and the leading
cooperative grace differ, so they are parameters.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess

logger = logging.getLogger("alpha_lab.process_control")


def _exits_within(proc: subprocess.Popen, timeout: float) -> bool:
    """True if the process exits within ``timeout`` (also reaps it); False on timeout."""
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def _signal_proc(proc: subprocess.Popen, sig: int, process_group: bool) -> None:
    """Signal the process (or its whole group), ignoring an already-exited process."""
    try:
        if process_group:
            os.killpg(os.getpgid(proc.pid), sig)
        elif sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.terminate()
    except (ProcessLookupError, ChildProcessError):
        pass
    except OSError as exc:
        logger.warning("Failed to signal pid %s: %s", proc.pid, exc)


def escalate_termination(
    proc: subprocess.Popen,
    *,
    term_grace: float,
    kill_grace: float,
    cooperative_grace: float = 0.0,
    process_group: bool = False,
) -> None:
    """Tear ``proc`` down by escalation, reaping it so it can't become a zombie.

    1. If the caller already asked it to stop cooperatively, wait ``cooperative_grace``.
    2. SIGTERM, then wait ``term_grace``.
    3. SIGKILL, then wait ``kill_grace``.

    Each wait short-circuits as soon as the process exits.
    """
    if proc.poll() is not None:
        return
    if cooperative_grace and _exits_within(proc, cooperative_grace):
        return
    _signal_proc(proc, signal.SIGTERM, process_group)
    if _exits_within(proc, term_grace):
        return
    _signal_proc(proc, signal.SIGKILL, process_group)
    _exits_within(proc, kill_grace)
