from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from tools import rss_bench
from tools.benchmark_common import MEASURED_RUNS, MIB, REPO_DIR
from tools.benchmark_suites import HF_SUITE, TIKTOKEN_SUITE


def _raw_peak_row(
    run: int,
    *,
    model: int,
    peak: int,
) -> dict[str, object]:
    initialization_peak = model + 1
    return {
        "suite": "tiktoken",
        "implementation": "hiriluk",
        "item": "r50k",
        "corpus": "english",
        "requested_input_bytes": MIB,
        "input_bytes": MIB,
        "input_mib": "1.000000",
        "run": run,
        "model_rss_bytes": model,
        "initialization_peak_rss_bytes": initialization_peak,
        "peak_rss_bytes": peak,
        "peak_minus_model_bytes": max(0, peak - model),
        "peak_minus_initialization_bytes": max(
            0, peak - initialization_peak
        ),
        "tokens": 42,
        "output_mode": "numpy-u32-array",
    }


@pytest.mark.parametrize(
    ("input_bytes", "expected"),
    [
        (0, [0]),
        (MIB // 2, [MIB // 2]),
        (MIB, [MIB]),
        (3 * MIB + 17, [MIB, 2 * MIB, 3 * MIB + 17]),
        (8 * MIB, [MIB, 2 * MIB, 4 * MIB, 8 * MIB]),
    ],
)
def test_prefix_targets_are_powers_of_two_plus_exact_size(
    input_bytes: int,
    expected: list[int],
) -> None:
    assert rss_bench.prefix_targets(input_bytes) == expected


def test_default_rss_prefixes_cap_approximately_128_mib_input() -> None:
    assert rss_bench.prefix_targets(
        128 * MIB + MIB // 2,
        maximum_bytes=128 * MIB,
    ) == [MIB << exponent for exponent in range(8)]


def test_full_rss_prefixes_retain_exact_oversized_input() -> None:
    input_bytes = 128 * MIB + MIB // 2
    assert rss_bench.prefix_targets(input_bytes)[-2:] == [
        128 * MIB,
        input_bytes,
    ]


def test_utf8_prefix_never_splits_a_code_point(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes("A€B🙂C".encode())

    # Offsets 2 and 3 lie inside the three-byte euro sign.
    assert rss_bench.utf8_prefix_bytes(source, 2) == 1
    assert rss_bench.utf8_prefix_bytes(source, 3) == 1
    assert rss_bench.utf8_prefix_bytes(source, 4) == 4

    # Offset 7 lies inside the four-byte emoji.
    destination = tmp_path / "prefix.txt"
    actual = rss_bench.make_prefix(source, 7, destination)
    assert actual == 5
    assert destination.read_text(encoding="utf-8") == "A€B"
    assert not destination.with_suffix(".txt.part").exists()


def test_rss_output_policy_matches_suite_and_dfa_mode(
    tmp_path: Path,
) -> None:
    assert (
        rss_bench._output_path(
            TIKTOKEN_SUITE,
            False,
            False,
            "hiriluk",
            tmp_path,
        )
        is None
    )
    assert rss_bench._output_path(
        TIKTOKEN_SUITE,
        True,
        False,
        "hiriluk",
        tmp_path,
    ) == (tmp_path / "hiriluk.json")
    assert TIKTOKEN_SUITE.rss_output_policy(dfa=False, dump=False) == (
        "native-packed-array"
    )
    assert TIKTOKEN_SUITE.rss_output_policy(
        dfa=False,
        dump=True,
    ) == "tiktoken-array+gigatoken/hiriluk-json"
    assert TIKTOKEN_SUITE.rss_output_policy(
        dfa=True,
        dump=False,
    ) == "tiktoken/gigatoken-array+hiriluk-json"
    assert TIKTOKEN_SUITE.rss_implementations_for(dfa=True) == (
        "tiktoken",
        "gigatoken",
        "hiriluk",
    )
    assert (
        rss_bench._output_path(
            TIKTOKEN_SUITE,
            False,
            True,
            "tiktoken",
            tmp_path,
        )
        is None
    )
    assert rss_bench._output_path(
        TIKTOKEN_SUITE,
        False,
        True,
        "gigatoken",
        tmp_path,
    ) == (tmp_path / "gigatoken.json")
    assert (
        rss_bench._output_path(
            TIKTOKEN_SUITE,
            True,
            False,
            "gigatoken",
            tmp_path,
        )
        is None
    )
    assert rss_bench._output_path(
        HF_SUITE,
        True,
        False,
        "hiriluk",
        tmp_path,
    ) == (tmp_path / "hiriluk.json")
    assert (
        rss_bench._output_path(
            HF_SUITE,
            True,
            False,
            "huggingface",
            tmp_path,
        )
        is None
    )
    assert HF_SUITE.rss_output_policy(
        dfa=True,
        dump=False,
    ) == "huggingface-array+hiriluk-json"


def test_rss_artifact_helpers_write_csv_and_peak_plot(
    tmp_path: Path,
) -> None:
    peak_rows = [
        {
            "implementation": implementation,
            "item": "r50k",
            "corpus": "english",
            "input_mib": input_mib,
            "peak_rss_bytes": peak_mib * MIB,
            "peak_rss_p25_bytes": (peak_mib - 2) * MIB,
            "peak_rss_p75_bytes": (peak_mib + 3) * MIB,
        }
        for implementation, input_mib, peak_mib in (
            ("tiktoken", 1.0, 40),
            ("tiktoken", 2.0, 45),
            ("hiriluk", 1.0, 30),
            ("hiriluk", 2.0, 31),
        )
    ]
    csv_path = tmp_path / "peak_rss.csv"
    rss_bench._write_csv(csv_path, peak_rows)
    with csv_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["implementation"] for row in rows] == [
        "tiktoken",
        "tiktoken",
        "hiriluk",
        "hiriluk",
    ]

    rss_bench._write_plots(tmp_path, peak_rows)
    peak_plot = tmp_path / "r50k-english-peak-rss.pdf"
    assert peak_plot.read_bytes().startswith(b"%PDF-")
    assert not list(tmp_path.glob("*-peak-rss.svg"))
    assert not list(tmp_path.glob("*-rss-over-time.svg"))


def test_rss_plot_uses_fixed_implementation_styles() -> None:
    styled = rss_bench._ordered_plot_series(
        {
            "hiriluk": [(1.0, 1.0, 0.75, 1.25)],
            "gigatoken": [(1.0, 2.0, 1.5, 2.5)],
            "huggingface": [(1.0, 2.5, 2.0, 3.0)],
            "tiktoken": [(1.0, 3.0, 2.5, 3.5)],
        }
    )
    assert [
        (label, display_label, color)
        for label, display_label, color, _ in styled
    ] == [
        ("tiktoken", "tiktoken", "#1565c0"),
        ("huggingface", "HF", "#6a1b9a"),
        ("gigatoken", "gigatoken", "#2e7d32"),
        ("hiriluk", "Ours", "#c62828"),
    ]


def test_rss_defaults_to_ten_fresh_process_runs() -> None:
    assert MEASURED_RUNS == 10
    assert rss_bench.MEASURED_RUNS == MEASURED_RUNS


def test_single_rss_run_has_degenerate_quartiles() -> None:
    assert rss_bench._quartiles_int([123]) == (123, 123)


def test_peak_rows_report_median_quartiles_and_preserve_raw_deltas() -> None:
    raw = [
        _raw_peak_row(1, model=0, peak=100),
        _raw_peak_row(2, model=10, peak=200),
        _raw_peak_row(3, model=290, peak=300),
    ]

    aggregated = rss_bench._aggregate_peak_rows(raw, measured_runs=3)

    assert len(aggregated) == 1
    row = aggregated[0]
    assert row["runs"] == 3
    assert row["peak_rss_bytes"] == 200
    assert row["peak_rss_p25_bytes"] == 150
    assert row["peak_rss_p75_bytes"] == 250
    assert row["peak_rss_min_bytes"] == 100
    assert row["peak_rss_max_bytes"] == 300
    assert row["model_rss_bytes"] == 10
    # This is the median of the per-run deltas [100, 190, 10], not the
    # independently aggregated peak minus independently aggregated model RSS.
    assert row["peak_minus_model_bytes"] == 100


def test_peak_aggregation_rejects_partial_cells() -> None:
    with pytest.raises(RuntimeError, match="has 1 runs; expected 2"):
        rss_bench._aggregate_peak_rows(
            [_raw_peak_row(1, model=10, peak=20)],
            measured_runs=2,
        )


def test_rss_token_validation_compares_implementations() -> None:
    rows = [
        {
            "implementation": "tiktoken",
            "item": "r50k",
            "corpus": "english",
            "input_bytes": MIB,
            "tokens": 10,
        },
        {
            "implementation": "hiriluk",
            "item": "r50k",
            "corpus": "english",
            "input_bytes": MIB,
            "tokens": 11,
        },
    ]
    with pytest.raises(RuntimeError, match="token-count mismatch"):
        rss_bench._validate_token_counts(rows)


@pytest.mark.parametrize(
    "module",
    [
        "benchmarks.tiktoken_throughput_bench",
        "benchmarks.hf_throughput_bench",
        "benchmarks.tiktoken_rss_bench",
        "benchmarks.hf_rss_bench",
    ],
)
def test_benchmark_module_entrypoint_help(module: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--dataset" in completed.stdout


@pytest.mark.parametrize(
    "script",
    [
        "tiktoken_throughput_bench.py",
        "hf_throughput_bench.py",
        "tiktoken_rss_bench.py",
        "hf_rss_bench.py",
    ],
)
def test_benchmark_direct_entrypoint_help(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_DIR / "benchmarks" / script), "--help"],
        cwd=REPO_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--dataset" in completed.stdout
