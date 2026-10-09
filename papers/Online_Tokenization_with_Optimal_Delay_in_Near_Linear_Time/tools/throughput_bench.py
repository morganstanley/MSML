"""Shared orchestration for the Python throughput benchmarks.

Runs each tokenizer item/corpus combination in a fresh isolated process
(`tools/benchmark_worker.py`) and reports median MiB/s with its quartiles.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tools.benchmark_common import (
    MEASURED_RUNS,
    MIB,
    WARMUP_RUNS,
    BenchmarkSuite,
    build_selection_parser,
    concise_error,
    median_quartiles,
    print_table,
    print_unavailable,
    resolve_inputs,
    run_json_worker,
)


# Streamed file output goes to a RAM-backed mount when one exists. A shared
# block device otherwise sits inside the timed region: measured on this cluster,
# one build and corpus swung between 13.3 and 29.3 MiB/s across two nodes
# writing r50k JSON to XFS, and its p25-p75 spread was tens of MiB/s, while the
# same runs on tmpfs reproduced within 2% and spread by 0.2. Writes still cross
# the real `write(2)` path and page cache; only device queueing is removed.
#
# This is deliberately not applied to `rss_bench`: tmpfs output is charged to
# the process's cgroup, so it would silently convert file output into measured
# memory.
OUTPUT_DIR_ENV = "HIRILUK_BENCH_OUTPUT_DIR"
# Only Linux mounts a RAM-backed scratch filesystem by convention. macOS has
# none by default and falls through to the ordinary temporary directory.
TMPFS_CANDIDATES: tuple[str, ...] = ("/dev/shm",)


def filesystem_type(path: Path) -> str:
    """Best-effort filesystem type for `path`, or `"unknown"` off Linux."""

    try:
        target = path.resolve()
        mounts = Path("/proc/mounts").read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    longest = -1
    found = "unknown"
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mount = Path(fields[1].replace("\\040", " "))
        if (target == mount or mount in target.parents) and len(
            str(mount)
        ) > longest:
            longest = len(str(mount))
            found = fields[2]
    return found


def _is_writable_tmpfs(path: Path) -> bool:
    return (
        path.is_dir()
        and os.access(path, os.W_OK)
        and filesystem_type(path) == "tmpfs"
    )


def output_base_directory() -> Path:
    """Base directory for streamed benchmark output.

    `HIRILUK_BENCH_OUTPUT_DIR` overrides the choice; otherwise a writable tmpfs
    mount wins, and failing that the ordinary temporary directory (which still
    honors `TMPDIR`).
    """

    override = os.environ.get(OUTPUT_DIR_ENV)
    if override:
        return Path(override)
    for candidate in TMPFS_CANDIDATES:
        path = Path(candidate)
        if _is_writable_tmpfs(path):
            return path
    return Path(tempfile.gettempdir())


def describe_output_destination() -> str:
    base = output_base_directory()
    return f"{base} ({filesystem_type(base)})"


def timed_runs(
    label: str,
    item: str,
    implementation: str,
    construct: Callable[[], Any],
    once: Callable[[Any], int],
    input_bytes: int,
) -> tuple[str, str, str, str, str, str, str, str]:
    for _ in range(WARMUP_RUNS):
        encoder = construct()
        once(encoder)
        del encoder

    expected_tokens: int | None = None
    durations: list[float] = []
    for _ in range(MEASURED_RUNS):
        encoder = construct()
        started = time.perf_counter()
        tokens = int(once(encoder))
        elapsed = time.perf_counter() - started
        del encoder
        if expected_tokens is None:
            expected_tokens = tokens
        elif tokens != expected_tokens:
            raise RuntimeError(
                f"{implementation}/{item}/{label} token count changed: "
                f"{expected_tokens} then {tokens}"
            )
        durations.append(elapsed)

    assert expected_tokens is not None
    mib = input_bytes / MIB
    throughputs = [mib / duration for duration in durations]
    median, p25, p75 = median_quartiles(throughputs)
    assert median is not None and p25 is not None and p75 is not None
    return (
        implementation,
        item,
        label,
        f"{median:.1f}",
        f"{p25:.1f}",
        f"{p75:.1f}",
        str(expected_tokens),
    )


def run_worker(job: dict[str, object]) -> dict[str, Any]:
    """Run one isolated throughput worker."""

    return run_json_worker("tools.benchmark_worker", job)


def validate_parity(
    results: Sequence[dict[str, Any]],
    *,
    exact_ids: bool,
) -> None:
    expected: dict[tuple[str, str], tuple[str, int, str | None]] = {}
    for result in results:
        key = (str(result["encoding"]), str(result["corpus"]))
        implementation = str(result["impl"])
        tokens = int(result["tokens"])
        digest = str(result["digest"]) if exact_ids else None
        previous = expected.setdefault(
            key,
            (implementation, tokens, digest),
        )
        if tokens != previous[1]:
            raise RuntimeError(
                f"token-count mismatch for {key[0]}/{key[1]}: "
                f"{previous[0]}={previous[1]}, {implementation}={tokens}"
            )
        if exact_ids and digest != previous[2]:
            raise RuntimeError(
                f"token-ID mismatch for {key[0]}/{key[1]}: "
                f"{previous[0]}={previous[2]}, "
                f"{implementation}={digest}"
            )


def build_throughput_parser(
    suite: BenchmarkSuite,
) -> argparse.ArgumentParser:
    """Build the public parser for a throughput benchmark."""

    return build_selection_parser(suite)


@contextmanager
def throughput_output_destination(
    suite: BenchmarkSuite,
    dfa: bool,
    dump: bool,
    implementation: str,
    item: str,
) -> Iterator[Path | None]:
    extension = suite.throughput_output_extension(
        implementation,
        dfa=dfa,
        dump=dump,
    )
    if extension is not None:
        with tempfile.TemporaryDirectory(
            prefix=f"hiriluk-{suite.suite_id}-{implementation}-{item}-",
            dir=output_base_directory(),
        ) as directory:
            yield Path(directory) / f"tokens{extension}"
        return
    yield None


def worker_job(
    suite: BenchmarkSuite,
    phase: str,
    implementation: str,
    item: str,
    *,
    dfa: bool,
    corpus: str | None = None,
    path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    job: dict[str, object] = {
        "suite": suite.suite_id,
        "phase": phase,
        "implementation": implementation,
        "item": item,
        "dfa": dfa,
    }
    if corpus is not None:
        job["corpus"] = corpus
    if path is not None:
        job["path"] = str(path)
    if output_path is not None:
        job["output_path"] = str(output_path)
    return job


def _job_string(job: dict[str, object], key: str) -> str:
    value = job.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"worker job requires string field {key!r}")
    return value


def execute_worker_job(
    suite: BenchmarkSuite,
    job: dict[str, object],
) -> dict[str, object]:
    """Execute one already-validated suite job inside an isolated process."""

    phase = _job_string(job, "phase")
    implementation = _job_string(job, "implementation")
    item = suite.item_named(_job_string(job, "item"))
    if implementation not in suite.implementations:
        raise ValueError(f"unknown implementation: {implementation}")
    if not suite.supports(implementation, item):
        raise ValueError(
            f"{implementation} does not support {item.name}"
        )
    dfa = bool(job.get("dfa", False))

    output_value = job.get("output_path")
    output_path = Path(output_value) if isinstance(output_value, str) else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    suite.preload(
        implementation,
        item,
        phase=phase,
        output_path=output_path,
    )
    if phase == "init":
        started = time.perf_counter()
        encoder = suite.construct(
            implementation,
            item,
            dfa=dfa,
            profile=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _ = encoder
        return {
            "impl": implementation,
            "encoding": item.name,
            "init_ms": elapsed_ms,
        }

    if phase != "throughput":
        raise ValueError(f"unknown throughput worker phase: {phase}")

    corpus = _job_string(job, "corpus")
    path = Path(_job_string(job, "path"))
    if not path.is_file():
        raise FileNotFoundError(f"corpus does not exist: {path}")

    input_value = suite.prepare_input(implementation, path)
    row = timed_runs(
        corpus,
        item.name,
        implementation,
        lambda: suite.construct(
            implementation,
            item,
            dfa=dfa,
            profile=False,
        ),
        lambda encoder: suite.encode_throughput(
            implementation,
            encoder,
            input_value,
            output_path,
        ),
        path.stat().st_size,
    )
    verification_encoder = suite.construct(
        implementation,
        item,
        dfa=dfa,
        profile=False,
    )
    verification_tokens, digest = suite.encode_verification(
        implementation,
        verification_encoder,
        input_value,
        output_path,
    )
    if verification_tokens != int(row[6]):
        raise RuntimeError(
            "untimed verification token count differs from timed runs: "
            f"{verification_tokens} != {row[6]}"
        )
    return {
        "impl": row[0],
        "encoding": row[1],
        "corpus": row[2],
        "median": row[3],
        "p25": row[4],
        "p75": row[5],
        "tokens": row[6],
        "digest": digest,
        "output_mode": suite.throughput_output_mode(
            implementation,
            output_path,
        ),
    }


def throughput_mode(
    suite: BenchmarkSuite,
    args: argparse.Namespace,
    items: Sequence[Any],
    corpora: Sequence[tuple[str, Path]],
) -> None:
    dfa = suite.force_dfa or bool(getattr(args, "dfa", False))
    implementations = suite.implementations_for(dfa=dfa)
    print(
        "=== Throughput "
        f"(serial, {suite.mode_description(dfa=dfa)}, "
        f"warmup={WARMUP_RUNS}, reps={MEASURED_RUNS}) ==="
    )
    results: list[dict[str, Any]] = []
    unavailable: list[tuple[str, str, str, str]] = []
    blocked: dict[tuple[str, str], str] = {}
    init_times: dict[tuple[str, str], float] = {}

    for item in items:
        for implementation in implementations:
            if not suite.supports(implementation, item):
                continue
            print(f"[init] {implementation}/{item.name}", flush=True)
            try:
                result = run_worker(
                    worker_job(
                        suite,
                        "init",
                        implementation,
                        item.name,
                        dfa=dfa,
                    )
                )
            except (OSError, RuntimeError, ValueError) as error:
                reason = concise_error(error)
                blocked[(implementation, item.name)] = reason
                unavailable.append(
                    (implementation, item.name, "*", reason)
                )
            else:
                init_times[(implementation, item.name)] = float(
                    result["init_ms"]
                )

    # DFA mode measures exactly one output form per implementation (tiktoken
    # never writes a file; hiriluk always does). Non-DFA mode compares array
    # output against file output for every implementation.
    dump_variants = (False,) if dfa else (False, True)

    for item in items:
        for corpus, path in corpora:
            for implementation in implementations:
                key = (implementation, item.name)
                if not suite.supports(implementation, item) or key in blocked:
                    continue
                for dump in dump_variants:
                    try:
                        with throughput_output_destination(
                            suite,
                            dfa,
                            dump,
                            implementation,
                            item.name,
                        ) as output_path:
                            print(
                                f"[throughput] {implementation}/{item.name}/"
                                f"{corpus}/"
                                f"{'array' if output_path is None else 'file'}",
                                flush=True,
                            )
                            result = run_worker(
                                worker_job(
                                    suite,
                                    "throughput",
                                    implementation,
                                    item.name,
                                    dfa=dfa,
                                    corpus=corpus,
                                    path=path,
                                    output_path=output_path,
                                )
                            )
                    except (OSError, RuntimeError, ValueError) as error:
                        unavailable.append(
                            (
                                implementation,
                                item.name,
                                corpus,
                                concise_error(error),
                            )
                        )
                        continue
                    result["init_ms"] = init_times[key]
                    result["input_bytes"] = path.stat().st_size
                    results.append(result)

    validate_parity(results, exact_ids=suite.throughput_exact_ids)
    rows = [
        (
            result["impl"],
            result["encoding"],
            result["corpus"],
            f"{int(result['input_bytes']) / MIB:.1f}",
            f"{float(result['init_ms']):.3f}",
            result["median"],
            result["p25"],
            result["p75"],
            result["tokens"],
            result["output_mode"],
        )
        for result in results
    ]
    print()
    print_table(
        (
            "impl",
            suite.item_label,
            "corpus",
            "input_MiB",
            "init_ms",
            "median_MiB/s",
            "p25_MiB/s",
            "p75_MiB/s",
            "tokens",
            "output",
        ),
        rows,
    )
    print_unavailable(
        "Unavailable throughput rows",
        suite.item_label,
        unavailable,
    )
    print()
    print(suite.throughput_note(dfa=dfa))
    if any(result["output_mode"].endswith("json-file") for result in results):
        # A throughput row that writes a file is only interpretable alongside
        # where it wrote, so never leave the destination implicit.
        print(
            f"Streamed file output was written under "
            f"{describe_output_destination()}; set {OUTPUT_DIR_ENV} to choose "
            f"another destination."
        )


def run_throughput_benchmark(
    suite: BenchmarkSuite,
    argv: Sequence[str] | None = None,
) -> None:
    """Run one suite's isolated-process throughput experiment."""

    parser = build_throughput_parser(suite)
    args = parser.parse_args(argv)

    try:
        items, corpora = resolve_inputs(suite, args)
        throughput_mode(suite, args, items, corpora)
        print(suite.final_note)
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error
