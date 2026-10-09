"""Peak-RSS scaling experiments for tokenizer suites."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.benchmark_common import (
    MEASURED_RUNS,
    MIB,
    REPO_DIR,
    SHORT_TARGET_BYTES,
    BenchmarkSuite,
    build_selection_parser,
    concise_error,
    print_table,
    resolve_inputs,
)


IMPLEMENTATION_PLOT_ORDER = (
    "tiktoken",
    "huggingface",
    "gigatoken",
    "hiriluk",
)
IMPLEMENTATION_PLOT_STYLES = {
    "tiktoken": ("#1565c0", "tiktoken"),
    "huggingface": ("#6a1b9a", "HF"),
    "gigatoken": ("#2e7d32", "gigatoken"),
    "hiriluk": ("#c62828", "Ours"),
}
FALLBACK_PLOT_COLORS = (
    "#00838f",
    "#ad1457",
    "#5d4037",
    "#455a64",
)


def prefix_targets(
    input_bytes: int,
    *,
    maximum_bytes: int | None = None,
) -> list[int]:
    """Return power-of-two MiB targets plus the capped selected size."""
    selected_bytes = input_bytes
    if maximum_bytes is not None:
        selected_bytes = min(selected_bytes, maximum_bytes)
    if selected_bytes <= 0:
        return [0]
    targets: list[int] = []
    size = MIB
    while size < selected_bytes:
        targets.append(size)
        size *= 2
    targets.append(selected_bytes)
    return targets


def utf8_prefix_bytes(source: Path, requested: int) -> int:
    """Find the largest UTF-8 boundary no greater than `requested`."""
    total = source.stat().st_size
    cut = min(max(0, requested), total)
    if cut == total or cut == 0:
        return cut
    with source.open("rb") as stream:
        while cut:
            stream.seek(cut)
            byte = stream.read(1)
            if not byte or byte[0] & 0xC0 != 0x80:
                break
            cut -= 1
    return cut


def make_prefix(source: Path, requested: int, destination: Path) -> int:
    """Copy one bounded, valid-UTF-8 prefix atomically."""
    actual = utf8_prefix_bytes(source, requested)
    temporary = destination.with_suffix(destination.suffix + ".part")
    # Prefixes are disposable inputs inside a private temporary directory.
    # Drop the preceding (smaller) prefix before constructing the next one so
    # full-corpus runs do not temporarily retain both large copies.
    destination.unlink(missing_ok=True)
    remaining = actual
    try:
        with source.open("rb") as input_file, temporary.open("wb") as output:
            while remaining:
                block = input_file.read(min(8 * MIB, remaining))
                if not block:
                    raise OSError(
                        f"{source} ended before its expected prefix length"
                    )
                output.write(block)
                remaining -= len(block)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return actual


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "RAYON_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return environment


def run_rss_worker(job: dict[str, object]) -> dict[str, Any]:
    """Run one isolated worker and return its kernel peak-RSS result."""
    completed = subprocess.run(
        [sys.executable, "-m", "tools.rss_worker"],
        cwd=REPO_DIR,
        env=_worker_environment(),
        input=json.dumps(job, separators=(",", ":")) + "\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            details or f"RSS worker exited with status {completed.returncode}"
        )
    try:
        event = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("RSS worker returned invalid JSON") from error
    if not isinstance(event, dict) or event.get("event") != "done":
        raise RuntimeError(f"invalid RSS worker result: {event!r}")
    return event


def _output_path(
    suite: BenchmarkSuite,
    dfa: bool,
    dump: bool,
    implementation: str,
    directory: Path,
) -> Path | None:
    extension = suite.rss_output_extension(
        implementation,
        dfa=dfa,
        dump=dump,
    )
    if extension is None:
        return None
    return directory / f"{implementation}{extension}"


def _rss_job(
    suite: BenchmarkSuite,
    implementation: str,
    item: str,
    corpus: str,
    path: Path,
    *,
    dfa: bool,
    dump: bool,
    output_path: Path | None,
) -> dict[str, object]:
    job: dict[str, object] = {
        "suite": suite.suite_id,
        "implementation": implementation,
        "item": item,
        "corpus": corpus,
        "path": str(path),
        "dfa": dfa,
        "dump": dump,
    }
    if output_path is not None:
        job["output_path"] = str(output_path)
    return job


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _median_int(values: Sequence[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take the median of an empty sequence")
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) // 2


def _quartiles_int(values: Sequence[int]) -> tuple[int, int]:
    if not values:
        raise ValueError("cannot take quartiles of an empty sequence")
    if len(values) == 1:
        return values[0], values[0]
    p25, _, p75 = statistics.quantiles(values, n=4, method="inclusive")
    return round(p25), round(p75)


def _aggregate_peak_rows(
    rows: Sequence[dict[str, object]],
    *,
    measured_runs: int = MEASURED_RUNS,
) -> list[dict[str, object]]:
    """Reduce raw fresh-process measurements to one row per input cell."""
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["suite"],
            row["implementation"],
            row["item"],
            row["corpus"],
            row["requested_input_bytes"],
            row["input_bytes"],
            row["input_mib"],
            row["output_mode"],
        )
        groups[key].append(row)

    integer_metrics = (
        "model_rss_bytes",
        "initialization_peak_rss_bytes",
        "peak_rss_bytes",
        "peak_minus_model_bytes",
        "peak_minus_initialization_bytes",
    )
    aggregated: list[dict[str, object]] = []
    for key, group in groups.items():
        if len(group) != measured_runs:
            raise RuntimeError(
                f"RSS cell {key[1]}/{key[2]}/{key[3]}/{key[5]} bytes "
                f"has {len(group)} runs; expected {measured_runs}"
            )
        tokens = {int(row["tokens"]) for row in group}
        if len(tokens) != 1:
            raise RuntimeError(
                f"token count changed across RSS runs for "
                f"{key[1]}/{key[2]}/{key[3]}/{key[5]} bytes"
            )
        run_numbers = sorted(int(row["run"]) for row in group)
        if run_numbers != list(range(1, measured_runs + 1)):
            raise RuntimeError(
                f"RSS cell {key[1]}/{key[2]}/{key[3]}/{key[5]} bytes "
                "does not contain each run exactly once"
            )

        result: dict[str, object] = {
            "suite": key[0],
            "implementation": key[1],
            "item": key[2],
            "corpus": key[3],
            "requested_input_bytes": key[4],
            "input_bytes": key[5],
            "input_mib": key[6],
            "runs": measured_runs,
        }
        for metric in integer_metrics:
            values = [int(row[metric]) for row in group]
            result[metric] = _median_int(values)
            p25, p75 = _quartiles_int(values)
            stem = metric.removesuffix("_bytes")
            result[f"{stem}_p25_bytes"] = p25
            result[f"{stem}_p75_bytes"] = p75
            result[f"{stem}_min_bytes"] = min(values)
            result[f"{stem}_max_bytes"] = max(values)
        result.update(
            {
                "tokens": tokens.pop(),
                "output_mode": key[7],
            }
        )
        aggregated.append(result)
    return aggregated


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    )


def _ordered_plot_series(
    series: dict[str, list[tuple[float, float, float, float]]],
) -> list[
    tuple[str, str, str, list[tuple[float, float, float, float]]]
]:
    order = {
        implementation: index
        for index, implementation in enumerate(IMPLEMENTATION_PLOT_ORDER)
    }
    ordered = sorted(
        ((label, points) for label, points in series.items() if points),
        key=lambda item: (order.get(item[0], len(order)), item[0]),
    )
    styled = []
    fallback_index = 0
    for label, points in ordered:
        style = IMPLEMENTATION_PLOT_STYLES.get(label)
        if style is None:
            color = FALLBACK_PLOT_COLORS[
                fallback_index % len(FALLBACK_PLOT_COLORS)
            ]
            display_label = label
            fallback_index += 1
        else:
            color, display_label = style
        styled.append((label, display_label, color, points))
    return styled


def _pdf_plot(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float, float, float]]],
    x_log2: bool,
    y_log2: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("pdf")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    styled = _ordered_plot_series(series)
    if not styled:
        return

    figure, axes = plt.subplots(figsize=(9, 5.6))
    try:
        plotted = 0
        for _, display_label, color, points in styled:
            valid = [
                (x, median, p25, p75)
                for x, median, p25, p75 in points
                if (not x_log2 or x > 0)
                and (not y_log2 or p25 > 0)
            ]
            if not valid:
                continue
            xs, medians, p25s, p75s = zip(*valid)
            lower_errors = [
                max(0.0, median - p25)
                for median, p25 in zip(medians, p25s)
            ]
            upper_errors = [
                max(0.0, p75 - median)
                for median, p75 in zip(medians, p75s)
            ]
            axes.errorbar(
                xs,
                medians,
                yerr=(lower_errors, upper_errors),
                color=color,
                linewidth=2.3,
                linestyle="-",
                marker="o",
                markersize=4.5,
                elinewidth=1.4,
                capsize=3.5,
                capthick=1.4,
                label=display_label,
            )
            plotted += 1
        if plotted == 0:
            return

        numeric_ticks = FuncFormatter(lambda value, _: f"{value:g}")
        if x_log2:
            axes.set_xscale("log", base=2)
            axes.xaxis.set_major_locator(LogLocator(base=2))
            axes.xaxis.set_major_formatter(numeric_ticks)
        if y_log2:
            axes.set_yscale("log", base=2)
            axes.yaxis.set_major_locator(LogLocator(base=2))
            axes.yaxis.set_major_formatter(numeric_ticks)

        axes.set_title(title, pad=14)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.set_axisbelow(True)
        axes.grid(which="major", color="#dedede", linewidth=0.8)
        axes.grid(which="minor", color="#f0f0f0", linewidth=0.5)
        axes.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(
            path,
            format="pdf",
            bbox_inches="tight",
            metadata={
                "Title": title,
                "Creator": "Hiriluk RSS benchmark",
                "CreationDate": None,
            },
        )
    finally:
        plt.close(figure)


def _write_plots(
    output_dir: Path,
    peak_rows: Sequence[dict[str, object]],
) -> None:
    peak_groups: dict[
        tuple[str, str],
        dict[str, list[tuple[float, float, float, float]]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in peak_rows:
        peak_groups[(str(row["item"]), str(row["corpus"]))][
            str(row["implementation"])
        ].append(
            (
                float(row["input_mib"]),
                float(row["peak_rss_bytes"]) / MIB,
                float(row["peak_rss_p25_bytes"]) / MIB,
                float(row["peak_rss_p75_bytes"]) / MIB,
            )
        )
    for (item, corpus), series in peak_groups.items():
        for points in series.values():
            points.sort()
        _pdf_plot(
            output_dir / f"{_slug(item)}-{_slug(corpus)}-peak-rss.pdf",
            title=(
                f"{item} / {corpus}: peak RSS vs input size\n"
                "Median with p25-p75 error bars"
            ),
            x_label="Input size (MiB, log2 scale)",
            y_label="Median peak RSS (MiB, log2 scale)",
            series=series,
            x_log2=True,
            y_log2=True,
        )


def _validate_token_counts(rows: Sequence[dict[str, object]]) -> None:
    expected: dict[tuple[str, str, int], tuple[str, int]] = {}
    for row in rows:
        key = (
            str(row["item"]),
            str(row["corpus"]),
            int(row["input_bytes"]),
        )
        implementation = str(row["implementation"])
        tokens = int(row["tokens"])
        previous = expected.setdefault(key, (implementation, tokens))
        if tokens != previous[1]:
            raise RuntimeError(
                f"token-count mismatch for {key[0]}/{key[1]}/"
                f"{key[2]} bytes: {previous[0]}={previous[1]}, "
                f"{implementation}={tokens}"
            )


def _print_largest_input_summary(
    suite: BenchmarkSuite,
    rows: Sequence[dict[str, object]],
) -> None:
    largest: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["item"]), str(row["corpus"]))
        largest[key] = max(largest.get(key, 0), int(row["input_bytes"]))
    selected = [
        row
        for row in rows
        if int(row["input_bytes"])
        == largest[(str(row["item"]), str(row["corpus"]))]
    ]
    if not selected:
        return

    print()
    print(
        "=== Largest-input RSS summary "
        f"({MEASURED_RUNS} fresh processes) ==="
    )
    print_table(
        (
            "impl",
            suite.item_label,
            "corpus",
            "input_MiB",
            "model_med_MiB",
            "init_peak_med_MiB",
            "peak_med_MiB",
            "peak_max_MiB",
            "peak-model_med_MiB",
            "tokens",
            "output",
        ),
        [
            (
                row["implementation"],
                row["item"],
                row["corpus"],
                f"{int(row['input_bytes']) / MIB:.1f}",
                f"{int(row['model_rss_bytes']) / MIB:.1f}",
                f"{int(row['initialization_peak_rss_bytes']) / MIB:.1f}",
                f"{int(row['peak_rss_bytes']) / MIB:.1f}",
                f"{int(row['peak_rss_max_bytes']) / MIB:.1f}",
                f"{int(row['peak_minus_model_bytes']) / MIB:.1f}",
                row["tokens"],
                row["output_mode"],
            )
            for row in selected
        ],
    )


def run_rss_benchmark(
    suite: BenchmarkSuite,
    argv: Sequence[str] | None = None,
) -> None:
    """Measure peak RSS versus input size in fresh worker processes."""
    script_name = f"benchmarks/{suite.suite_id}_rss_bench.py"
    parser = build_selection_parser(
        suite,
        description=(
            f"{suite.description}\n\nMeasure peak RSS as input grows."
        ),
        epilog=(
            "Examples:\n"
            f"  python {script_name}\n"
            f"  python {script_name} --dataset english\n"
            f"  python {script_name} --tiny\n"
            f"  python {script_name} --full"
            + (
                f"\n  python {script_name} --dfa"
                if suite.supports_dfa
                else ""
            )
            + (
                f"\n  python {script_name} --dump"
                if suite.supports_dump
                else ""
            )
        ),
        dfa_help=(
            "use Hiriluk's reference DFA path instead of its "
            "SIMD/full-cache path; retain all reference implementations"
        ),
    )
    parser.set_defaults(dump=False)
    if suite.supports_dump:
        parser.add_argument(
            "--dump",
            action="store_true",
            help=suite.dump_help,
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_DIR / "benchmarks" / "results" / f"{suite.suite_id}_rss",
        metavar="DIRECTORY",
        help=(
            "artifact directory (default: "
            f"benchmarks/results/{suite.suite_id}_rss)"
        ),
    )
    args = parser.parse_args(argv)
    if bool(args.dump) and bool(getattr(args, "dfa", False)):
        parser.error("--dump and --dfa select different benchmark modes")

    try:
        items, corpora = resolve_inputs(suite, args)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        dfa = suite.force_dfa or bool(getattr(args, "dfa", False))
        dump = bool(args.dump)
        implementations = suite.rss_implementations_for(dfa=dfa)
        raw_peak_rows: list[dict[str, object]] = []
        unavailable: list[dict[str, str]] = []

        for corpus, source in corpora:
            source_bytes = source.stat().st_size
            # The default samples are only approximately 128 MiB. Capping
            # their prefix series avoids plotting both 128 MiB and a nearly
            # identical 128.x MiB whole-file endpoint. Explicit --full runs
            # still include the complete source as their final point.
            maximum_bytes = (
                SHORT_TARGET_BYTES
                if args.corpus_size == "short"
                else None
            )
            targets = prefix_targets(
                source_bytes,
                maximum_bytes=maximum_bytes,
            )
            # Deliberately the ordinary temporary directory, not the tmpfs base
            # the throughput benchmark prefers: tmpfs pages are charged to this
            # process's cgroup, so RAM-backed scratch would silently report file
            # output and input prefixes as measured memory.
            with tempfile.TemporaryDirectory(
                prefix=f"hiriluk-{suite.suite_id}-{corpus}-rss-"
            ) as temporary_name:
                temporary = Path(temporary_name)
                prefix_path = temporary / "input.txt"
                for requested_bytes in targets:
                    if requested_bytes == source_bytes:
                        input_path = source
                        actual_bytes = source_bytes
                    else:
                        input_path = prefix_path
                        actual_bytes = make_prefix(
                            source,
                            requested_bytes,
                            prefix_path,
                        )
                    for item in items:
                        for implementation in implementations:
                            if not suite.supports(implementation, item):
                                continue
                            output_path = _output_path(
                                suite,
                                dfa,
                                dump,
                                implementation,
                                temporary,
                            )
                            cell_peak_rows: list[dict[str, object]] = []
                            cell_failed = False
                            for run in range(1, MEASURED_RUNS + 1):
                                if output_path is not None:
                                    output_path.unlink(missing_ok=True)
                                print(
                                    "[rss] "
                                    f"{implementation}/{item.name}/{corpus}/"
                                    f"{actual_bytes / MIB:.1f} MiB "
                                    f"run {run}/{MEASURED_RUNS}",
                                    flush=True,
                                )
                                try:
                                    result = run_rss_worker(
                                        _rss_job(
                                            suite,
                                            implementation,
                                            item.name,
                                            corpus,
                                            input_path,
                                            dfa=dfa,
                                            dump=dump,
                                            output_path=output_path,
                                        )
                                    )
                                except (
                                    OSError,
                                    RuntimeError,
                                    ValueError,
                                ) as error:
                                    unavailable.append(
                                        {
                                            "implementation": implementation,
                                            "item": item.name,
                                            "corpus": corpus,
                                            "input_bytes": str(actual_bytes),
                                            "run": str(run),
                                            "reason": concise_error(error),
                                        }
                                    )
                                    cell_failed = True
                                    break
                                finally:
                                    # File-output modes can be larger than
                                    # their input. Delete each run's closed
                                    # output before launching the next worker.
                                    if output_path is not None:
                                        output_path.unlink(missing_ok=True)

                                model_rss = int(result["model_rss"])
                                initialization_peak = int(
                                    result["initialization_peak_rss"]
                                )
                                peak = int(result["peak_rss"])
                                cell_peak_rows.append(
                                    {
                                        "suite": suite.suite_id,
                                        "implementation": implementation,
                                        "item": item.name,
                                        "corpus": corpus,
                                        "requested_input_bytes": (
                                            requested_bytes
                                        ),
                                        "input_bytes": actual_bytes,
                                        "input_mib": (
                                            f"{actual_bytes / MIB:.6f}"
                                        ),
                                        "run": run,
                                        "model_rss_bytes": model_rss,
                                        "initialization_peak_rss_bytes": (
                                            initialization_peak
                                        ),
                                        "peak_rss_bytes": peak,
                                        "peak_minus_model_bytes": max(
                                            0, peak - model_rss
                                        ),
                                        "peak_minus_initialization_bytes": max(
                                            0, peak - initialization_peak
                                        ),
                                        "tokens": int(result["tokens"]),
                                        "output_mode": result["output_mode"],
                                    }
                                )

                            # Keep cells all-or-nothing: an interrupted or
                            # failed repetition must not look like a valid
                            # lower-sample aggregate.
                            if cell_failed:
                                continue
                            raw_peak_rows.extend(cell_peak_rows)

        _validate_token_counts(raw_peak_rows)
        peak_rows = _aggregate_peak_rows(raw_peak_rows)
        _write_csv(output_dir / "peak_rss.csv", peak_rows)
        _write_csv(output_dir / "peak_rss_raw.csv", raw_peak_rows)
        _write_csv(output_dir / "unavailable.csv", unavailable)
        for obsolete in (
            output_dir / "rss_trace.csv",
            output_dir / "rss_trace_raw.csv",
            *output_dir.glob("*-peak-rss.svg"),
            *output_dir.glob("*-rss-over-time.svg"),
            *output_dir.glob("*-rss-over-time.pdf"),
        ):
            obsolete.unlink(missing_ok=True)
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "suite": suite.suite_id,
            "dfa": dfa,
            "dump": dump,
            "measured_runs": MEASURED_RUNS,
            "python": sys.version,
            "platform": sys.platform,
            "items": [item.name for item in items],
            "corpora": [corpus for corpus, _ in corpora],
            "corpus_size": args.corpus_size,
            "maximum_input_bytes": (
                SHORT_TARGET_BYTES
                if args.corpus_size == "short"
                else None
            ),
            "peak_rows": len(peak_rows),
            "peak_raw_rows": len(raw_peak_rows),
            "output_policy": suite.rss_output_policy(
                dfa=dfa,
                dump=dump,
            ),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_plots(output_dir, peak_rows)
        _print_largest_input_summary(suite, peak_rows)
        print()
        print(f"RSS artifacts: {output_dir}")
        if unavailable:
            print(
                f"{len(unavailable)} rows were unavailable; see "
                f"{output_dir / 'unavailable.csv'}"
            )
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error


__all__ = [
    "make_prefix",
    "prefix_targets",
    "run_rss_benchmark",
    "run_rss_worker",
    "utf8_prefix_bytes",
]
