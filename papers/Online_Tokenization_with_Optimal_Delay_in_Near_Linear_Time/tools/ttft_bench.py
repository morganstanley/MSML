"""Shared Hiriluk time-to-first-token benchmark orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tools import benchmark_common as common
from tools.benchmark_common import BenchmarkSuite, MIB, median_quartiles


IMPLEMENTATION = "hiriluk"


@dataclass(frozen=True)
class TtftObservation:
    """One chop call, excluding encoder construction."""

    tokens: int
    ttft_seconds: float | None
    completion_seconds: float


def observe_once(
    encoder: Any,
    input_value: Path,
) -> TtftObservation:
    """Measure one already-constructed profiled Hiriluk encoder."""

    started = time.perf_counter()
    tokens_array = encoder.chop_file(input_value, output="array")
    completion = time.perf_counter() - started
    tokens = len(tokens_array)

    profile = encoder.last_profile
    if profile is None:
        raise RuntimeError(
            "Hiriluk did not publish a ChopProfile; construct it with "
            "profile=True"
        )
    if int(profile.total_tokens) != tokens:
        raise RuntimeError(
            "Hiriluk profile token count differs from returned array: "
            f"{profile.total_tokens} != {tokens}"
        )

    ttft = profile.ttft_seconds
    profiled_elapsed = float(profile.elapsed_seconds)
    if tokens and ttft is None:
        raise RuntimeError("Hiriluk did not record TTFT for nonempty output")
    if not tokens and ttft is not None:
        raise RuntimeError("Hiriluk recorded TTFT for empty output")
    if ttft is not None and not 0.0 <= float(ttft) <= profiled_elapsed:
        raise RuntimeError(
            "Hiriluk TTFT lies outside its profiled chop duration: "
            f"{ttft} not in [0, {profiled_elapsed}]"
        )
    # These are durations, not cross-language absolute timestamps. Rust's
    # monotonic interval is wholly nested inside the Python call interval.
    if profiled_elapsed > completion + 0.001:
        raise RuntimeError(
            "Hiriluk's Rust chop duration exceeds the enclosing Python call: "
            f"{profiled_elapsed} > {completion}"
        )

    return TtftObservation(
        tokens=tokens,
        ttft_seconds=None if ttft is None else float(ttft),
        completion_seconds=completion,
    )


def _consistent_tokens(
    observations: Sequence[TtftObservation],
    *,
    implementation: str,
    item: str,
    corpus: str,
) -> int:
    if not observations:
        raise RuntimeError("TTFT benchmark requires at least one observation")
    expected = observations[0].tokens
    for observation in observations[1:]:
        if observation.tokens != expected:
            raise RuntimeError(
                f"{implementation}/{item}/{corpus} token count changed: "
                f"{expected} then {observation.tokens}"
            )
    return expected


def measure_case(
    suite: BenchmarkSuite,
    item: Any,
    corpus: str,
    path: Path,
    *,
    dfa: bool,
) -> dict[str, object]:
    """Measure one Hiriluk item/corpus combination."""

    observations: list[TtftObservation] = []

    for run_index in range(common.WARMUP_RUNS + common.TTFT_RUNS):
        encoder = suite.construct(
            IMPLEMENTATION,
            item,
            dfa=dfa,
            profile=True,
        )
        observation = observe_once(encoder, path)
        del encoder
        if run_index >= common.WARMUP_RUNS:
            observations.append(observation)

    tokens = _consistent_tokens(
        observations,
        implementation=IMPLEMENTATION,
        item=item.name,
        corpus=corpus,
    )

    ttfts: list[float] = [
        observation.ttft_seconds
        for observation in observations
        if observation.ttft_seconds is not None
    ]
    if tokens and len(ttfts) != len(observations):
        raise RuntimeError("nonempty output has a missing TTFT observation")
    completions = [
        observation.completion_seconds for observation in observations
    ]
    ttft_median, ttft_p25, ttft_p75 = median_quartiles(ttfts)
    completion_median, completion_p25, completion_p75 = median_quartiles(
        completions
    )
    return {
        "impl": IMPLEMENTATION,
        "encoding": item.name,
        "corpus": corpus,
        "input_bytes": path.stat().st_size,
        "median_ttft_seconds": ttft_median,
        "ttft_p25_seconds": ttft_p25,
        "ttft_p75_seconds": ttft_p75,
        "median_completion_seconds": completion_median,
        "completion_p25_seconds": completion_p25,
        "completion_p75_seconds": completion_p75,
        "tokens": tokens,
    }


def _milliseconds(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 1000.0:.3f}"


def run_ttft_benchmark(
    suite: BenchmarkSuite,
    argv: Sequence[str] | None = None,
) -> None:
    """Run a suite's TTFT experiment in fresh encoders."""

    parser = common.build_selection_parser(
        suite,
        description=(
            f"Measure Hiriluk time to first token for {suite.suite_id} "
            "models."
        ),
        epilog=(
            "TTFT is recorded inside Rust when Hiriluk emits its first "
            "concrete token ID. Reference implementations are benchmarked "
            "separately."
        ),
    )
    args = parser.parse_args(argv)

    try:
        items, corpora = common.resolve_inputs(suite, args)
        dfa = suite.force_dfa or bool(getattr(args, "dfa", False))
        results: list[dict[str, object]] = []
        unavailable: list[tuple[str, str, str, str]] = []

        print(
            "=== Time to first token "
            f"(serial, {suite.mode_description(dfa=dfa)}, "
            f"warmup={common.WARMUP_RUNS}, "
            f"reps={common.TTFT_RUNS}) ==="
        )
        for item in items:
            if not suite.supports(IMPLEMENTATION, item):
                continue
            try:
                suite.preload(
                    IMPLEMENTATION,
                    item,
                    phase="ttft",
                    output_path=None,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                unavailable.append(
                    (
                        IMPLEMENTATION,
                        item.name,
                        "*",
                        common.concise_error(error),
                    )
                )
                continue

            for corpus, path in corpora:
                print(
                    f"[ttft] {IMPLEMENTATION}/{item.name}/{corpus}",
                    flush=True,
                )
                try:
                    result = measure_case(
                        suite,
                        item,
                        corpus,
                        path,
                        dfa=dfa,
                    )
                except (
                    ImportError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    unavailable.append(
                        (
                            IMPLEMENTATION,
                            item.name,
                            corpus,
                            common.concise_error(error),
                        )
                    )
                else:
                    results.append(result)

        rows = [
            (
                result["impl"],
                result["encoding"],
                result["corpus"],
                f"{int(result['input_bytes']) / MIB:.1f}",
                _milliseconds(result["median_ttft_seconds"]),
                _milliseconds(result["ttft_p25_seconds"]),
                _milliseconds(result["ttft_p75_seconds"]),
                _milliseconds(result["median_completion_seconds"]),
                _milliseconds(result["completion_p25_seconds"]),
                _milliseconds(result["completion_p75_seconds"]),
                result["tokens"],
            )
            for result in results
        ]
        print()
        common.print_table(
            (
                "impl",
                suite.item_label,
                "corpus",
                "input_MiB",
                "median_TTFT_ms",
                "TTFT_p25_ms",
                "TTFT_p75_ms",
                "median_return_ms",
                "return_p25_ms",
                "return_p75_ms",
                "tokens",
            ),
            rows,
        )
        common.print_unavailable(
            "Unavailable TTFT rows",
            suite.item_label,
            unavailable,
        )
        print()
        print(
            "TTFT is recorded in Rust when Hiriluk emits its first concrete "
            "token ID; encoder construction is excluded."
        )
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error
