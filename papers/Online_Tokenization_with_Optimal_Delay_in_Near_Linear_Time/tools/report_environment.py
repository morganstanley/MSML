#!/usr/bin/env python3
"""Seven lines of machine and toolchain provenance for a benchmark log.

Run inside the same `srun` step and CPU binding as the benchmark it documents,
so the affinity, memory limit, and clock policy reported are the ones the
measurement ran under:

    srun --cpu-bind=cores python -m tools.report_environment

Missing fields print as `?`; a `/sys` entry absent inside a container must not
take a benchmark job down with it.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version

# The tokenizer selects a scanner, hash, and merge kernel from these at runtime,
# so two machines differing only here are running different code.
FLAGS = ("sse4_2", "avx2", "avx512f", "avx512bw", "avx512vl", "avx512vbmi",
         "avx512vbmi2", "bmi1", "bmi2", "asimd", "crc32")
PKGS = ("hiriluk", "tiktoken", "gigatoken", "tokenizers", "numpy")
SERIAL = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def run(*cmd: str) -> str:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        return done.stdout.strip() if done.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def expand(spec: str) -> set[int]:
    """Expand a Linux CPU list such as `0-3,8,12-13`."""
    cpus: set[int] = set()
    for part in filter(None, spec.split(",")):
        low, _, high = part.partition("-")
        try:
            cpus.update(range(int(low), int(high or low) + 1))
        except ValueError:
            pass
    return cpus


def join(*parts: object) -> str:
    return " · ".join(str(p) for p in parts if p) or "?"


def main() -> None:
    cpu = {k.strip(): v.strip()
           for k, _, v in (line.partition(":") for line in run("lscpu").splitlines()) if v}
    line = next((l for l in read("/proc/cpuinfo").splitlines()
                 if l.startswith(("flags", "Features"))), "")
    flags = set(line.partition(":")[2].split())
    cpus = set(os.sched_getaffinity(0))
    # Logical CPUs sharing a sibling list are one physical core.
    cores = len({read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") or str(c)
                 for c in cpus})
    numa = [k[9:].split()[0] for k, v in cpu.items()
            if k.startswith("NUMA node") and k.endswith("CPU(s)") and cpus & expand(v)]
    # The Slurm request is what was actually granted and what a paper reports;
    # the cgroup here is the whole node, so it would overstate the job by ~250x.
    granted = os.environ.get("SLURM_MEM_PER_NODE", "")
    memory = (f"{int(granted) / 1024:.0f} GiB granted" if granted.isdigit() else
              next((f"{int(v) / 1024**3:.0f} GiB cgroup" for v in
                    (read("/sys/fs/cgroup/memory.max"),
                     read("/sys/fs/cgroup/memory/memory.limit_in_bytes")) if v.isdigit()), ""))
    distro = re.search(r'^PRETTY_NAME="?([^"\n]+)', read("/etc/os-release"), re.M)
    job = os.environ.get("SLURM_JOB_ID")
    listed = ",".join(map(str, sorted(cpus)))

    versions = []
    for name in PKGS:
        try:
            versions.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            pass

    print("=== Environment ===")
    print(" host    ", join(platform.node(),
                            f"slurm {job}.{os.environ.get('SLURM_STEP_ID')}" if job else "",
                            platform.release(), distro and distro.group(1)))
    print(" cpu     ", join(cpu.get("Model name"),
                            f"{cpu.get('Socket(s)')} socket x {cpu.get('Core(s) per socket')} core"
                            f" x {cpu.get('Thread(s) per core')} thread",
                            f"{float(cpu['CPU max MHz']):.0f} MHz max" if cpu.get("CPU max MHz") else "",
                            read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
                            {"1": "turbo off", "0": "turbo on"}.get(
                                read("/sys/devices/system/cpu/intel_pstate/no_turbo"), "")))
    # `--cpu-bind=cores` puts both hardware threads of the granted core in the
    # mask, so spell out the thread count: `cpu 48,112` is one core, not two.
    print(" pinned  ", join(f"cpu {listed}" if len(listed) <= 40 else f"{len(cpus)} logical cpus",
                            f"{cores} physical core" + ("s" if cores != 1 else "")
                            + (f" ({len(cpus)} threads)" if len(cpus) != cores else ""),
                            f"numa node {','.join(numa)}" if numa else "", memory))
    print(" cache   ", join(*(f"{n} {cpu[n + ' cache']}" for n in ("L1d", "L2", "L3")
                              if cpu.get(n + " cache"))))
    print(" isa     ", " ".join(f for f in FLAGS if f in flags) or "none of the dispatch tiers")
    print(" software", join(run("rustc", "--version"),
                            f"python {platform.python_version()}", *versions))
    print(" source  ", join(run("git", "rev-parse", "--short", "HEAD")
                            + (" dirty" if run("git", "status", "--porcelain") else ""),
                            run("git", "rev-parse", "--abbrev-ref", "HEAD"),
                            "serial(rayon,omp,openblas,mkl)="
                            + ",".join(os.environ.get(v, "unset") for v in SERIAL),
                            f"thp {read('/sys/kernel/mm/transparent_hugepage/enabled')}"))


if __name__ == "__main__":
    main()
