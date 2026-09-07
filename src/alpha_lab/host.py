"""Host-level resource gating for the local executors.

The CPU and GPU executors track their own ``_jobs`` dict and use it for
capacity decisions. That accounting is blind to processes from other
users or non-alphalab services on the same box. On a shared host one
heavy distributed-training job from another user (or several alphalab
orphans that the recovery scan failed to reattach) can saturate system
RAM and the load average without the dispatcher noticing.

This module reads ``/proc/meminfo`` and ``/proc/loadavg`` directly so the
gate reflects the *actual* host state across all users, not just our
own bookkeeping. It is cheap (a couple of file reads, no subprocesses)
and safe to call from every ``can_submit`` check.

Thresholds are conservative defaults; callers can override per-call.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("alpha_lab.host")

DEFAULT_MIN_AVAIL_RAM_GB = 64.0
DEFAULT_MAX_LOAD_PER_CPU = 4.0


def system_available_ram_gb() -> float | None:
    """Return ``MemAvailable`` from /proc/meminfo in GB, or None on error.

    ``MemAvailable`` is the kernel's estimate of memory available for
    starting new applications without swapping. It includes reclaimable
    page cache, so it's the right signal for "can I start another job
    without pushing the box into thrash".
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                    return avail_kb / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError) as e:
        logger.warning("Could not read MemAvailable from /proc/meminfo: %s", e)
    return None


def system_loadavg_per_cpu() -> float | None:
    """Return 1-minute loadavg divided by CPU count, or None on error.

    A value >1 means the run-queue is longer than the CPU count
    (oversubscribed). Heavy ML training tolerates some oversubscription,
    so the default threshold is 4x, not 1x.
    """
    try:
        with open("/proc/loadavg", encoding="ascii") as f:
            load1 = float(f.read().split()[0])
        ncpu = os.cpu_count() or 1
        return load1 / ncpu
    except (OSError, ValueError, IndexError) as e:
        logger.warning("Could not read /proc/loadavg: %s", e)
    return None


def host_has_capacity(
    min_avail_ram_gb: float = DEFAULT_MIN_AVAIL_RAM_GB,
    max_load_per_cpu: float = DEFAULT_MAX_LOAD_PER_CPU,
) -> bool:
    """Return True if the host has headroom to start another experiment.

    Two cheap system-wide checks:

    1. ``MemAvailable >= min_avail_ram_gb``. Catches RAM pressure from
       any source — other users' training jobs, our own orphans the
       recovery scan missed, system services. The default of 64 GB
       leaves room for one new ML job to load its feature dataset
       without pushing the box into swap-driven Bus errors.
    2. ``loadavg(1min) / cpu_count <= max_load_per_cpu``. Catches
       situations where there's RAM but the CPUs are oversubscribed.

    If either signal can't be read (`/proc` mounted weird, container
    sandbox), the function returns True — we don't want to refuse
    submissions because of a sensor failure. The executor's local
    accounting will still gate via its own slot counts.
    """
    avail = system_available_ram_gb()
    if avail is not None and avail < min_avail_ram_gb:
        logger.info(
            "host_has_capacity: refusing submit — MemAvailable %.1f GB < threshold %.1f GB",
            avail, min_avail_ram_gb,
        )
        return False
    load_per_cpu = system_loadavg_per_cpu()
    if load_per_cpu is not None and load_per_cpu > max_load_per_cpu:
        logger.info(
            "host_has_capacity: refusing submit — loadavg/cpu %.2f > threshold %.2f",
            load_per_cpu, max_load_per_cpu,
        )
        return False
    return True
