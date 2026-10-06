from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import benchmark_common as common
from tools import benchmark_suites as suites
from tools import throughput_bench as throughput
from tools.benchmark_suites import (
    TIKTOKEN_ENCODING_BY_NAME,
    TIKTOKEN_ENCODINGS,
    TIKTOKEN_SUITE,
)


def test_gigatoken_010_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert suites.GIGATOKEN_VERSION == "0.10.0"
    assert '"gigatoken==0.10.0"' in (
        Path(__file__).resolve().parents[2] / "pyproject.toml"
    ).read_text(encoding="utf-8")

    monkeypatch.setattr(
        suites.importlib.metadata,
        "version",
        lambda package: "0.10.0",
    )
    suites.require_gigatoken_version()

    monkeypatch.setattr(
        suites.importlib.metadata,
        "version",
        lambda package: "0.9.0",
    )
    with pytest.raises(RuntimeError, match="found 0.9.0"):
        suites.require_gigatoken_version()


def test_all_dataset_selection_has_only_paper_corpora() -> None:
    assert common.selected_corpora("all") == [
        "github",
        "english",
        "chinese",
    ]
    assert common.selected_corpora("english_stream") == ["english"]
    with pytest.raises(ValueError, match="unknown dataset"):
        common.selected_corpora("missing")


def test_dataset_resolution_supports_tiny_short_and_full(
    tmp_path: Path,
) -> None:
    tiny = tmp_path / "english/parquet_0_stream_tiny.txt"
    short = tmp_path / "english/parquet_0_stream_short.txt"
    full = tmp_path / "english/parquet_0_stream.txt"
    short.parent.mkdir()
    tiny.write_text("tiny", encoding="utf-8")
    with short.open("wb") as output:
        output.truncate(common.SHORT_TARGET_BYTES)
    full.write_text("full", encoding="utf-8")

    assert common.resolve_corpora(
        tmp_path, ["english"], "tiny"
    ) == [("english", tiny)]
    assert common.resolve_corpora(
        tmp_path, ["english"], "short"
    ) == [("english", short)]
    assert common.resolve_corpora(
        tmp_path, ["english"], "full"
    ) == [("english", full)]
    with pytest.raises(ValueError, match="unknown corpus size"):
        common.resolve_corpora(tmp_path, ["english"], "missing")


def test_encoding_registry_is_tiktoken_subset_and_reference_mapping() -> None:
    import hiriluk

    expected = ["r50k", "p50k", "cl100k", "o200k"]
    assert [spec.name for spec in TIKTOKEN_SUITE.selected_items("all")] == (
        expected
    )
    assert set(expected) < set(hiriluk.list_encoding_names())
    assert all(spec.tiktoken_name is not None for spec in TIKTOKEN_ENCODINGS)
    assert TIKTOKEN_ENCODING_BY_NAME["cl100k"].gigatoken_scheme == "gpt4"
    assert TIKTOKEN_ENCODING_BY_NAME["p50k"].gigatoken_hf_tokenizer == (
        "Xenova/text-davinci-003",
        "898195e24794cd1710ea3d0d99668208d1fe2ec9",
    )
    assert TIKTOKEN_SUITE.supports(
        "gigatoken",
        TIKTOKEN_ENCODING_BY_NAME["p50k"],
    )
    assert TIKTOKEN_SUITE.implementations_for(dfa=False) == (
        "gigatoken",
        "hiriluk",
    )
    assert TIKTOKEN_SUITE.implementations_for(dfa=True) == (
        "tiktoken",
        "hiriluk",
    )
    assert TIKTOKEN_SUITE.rss_implementations_for(dfa=True) == (
        "tiktoken",
        "gigatoken",
        "hiriluk",
    )
    with pytest.raises(ValueError, match="unknown encoding"):
        TIKTOKEN_SUITE.selected_items("qwen")


def test_timed_runs_uses_global_counts_and_fresh_encoders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(throughput, "WARMUP_RUNS", 1)
    monkeypatch.setattr(throughput, "MEASURED_RUNS", 2)
    calls = 0
    constructed: list[object] = []

    def construct() -> object:
        encoder = object()
        constructed.append(encoder)
        return encoder

    def once(encoder: object) -> int:
        nonlocal calls
        assert encoder is constructed[calls]
        calls += 1
        return 123

    timestamps = iter((10.0, 12.0, 20.0, 21.0))
    monkeypatch.setattr(
        throughput.time,
        "perf_counter",
        lambda: next(timestamps),
    )
    row = throughput.timed_runs(
        "sample",
        "r50k",
        "implementation",
        construct,
        once,
        10 * common.MIB,
    )
    assert row == (
        "implementation",
        "r50k",
        "sample",
        "7.5",
        "6.2",
        "8.8",
        "123",
    )
    assert calls == 3
    assert len({id(encoder) for encoder in constructed}) == 3


def test_timed_runs_rejects_changing_token_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(throughput, "WARMUP_RUNS", 0)
    monkeypatch.setattr(throughput, "MEASURED_RUNS", 2)
    counts = iter((10, 11))
    timestamps = iter((1.0, 2.0, 3.0, 4.0))
    monkeypatch.setattr(
        throughput.time,
        "perf_counter",
        lambda: next(timestamps),
    )
    with pytest.raises(RuntimeError, match="token count changed"):
        throughput.timed_runs(
            "sample",
            "r50k",
            "implementation",
            object,
            lambda _encoder: next(counts),
            common.MIB,
        )


def test_timed_runs_repeats_single_throughput_for_all_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(throughput, "WARMUP_RUNS", 0)
    monkeypatch.setattr(throughput, "MEASURED_RUNS", 1)
    timestamps = iter((1.0, 3.0))
    monkeypatch.setattr(
        throughput.time,
        "perf_counter",
        lambda: next(timestamps),
    )

    assert throughput.timed_runs(
        "sample",
        "r50k",
        "implementation",
        object,
        lambda _encoder: 123,
        10 * common.MIB,
    ) == (
        "implementation",
        "r50k",
        "sample",
        "5.0",
        "5.0",
        "5.0",
        "123",
    )


def test_hiriluk_throughput_uses_array_without_dump() -> None:
    class Chopper:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, str | None, Path | None]] = []

        def chop_file(
            self,
            path: Path,
            *,
            output: str | None = None,
            dump: Path | None = None,
        ) -> list[int]:
            self.calls.append((path, output, dump))
            return [1, 2, 3]

    chopper = Chopper()
    path = Path("/tmp/input.txt")
    assert TIKTOKEN_SUITE.encode_throughput(
        "hiriluk",
        chopper,
        path,
        None,
    ) == 3
    assert chopper.calls == [(path, "array", None)]


def test_hiriluk_throughput_uses_json_for_dfa_file_output(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "tokens.json"

    class Chopper:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, str, Path]] = []

        def chop_file(
            self,
            path: Path,
            *,
            output: str,
            dump: Path,
        ) -> int:
            self.calls.append((path, output, dump))
            dump.write_text("[1,2,3]", encoding="utf-8")
            return 3

    chopper = Chopper()
    input_path = tmp_path / "input.txt"
    input_path.write_text("sample", encoding="utf-8")

    encoded_count = TIKTOKEN_SUITE.encode_throughput(
        "hiriluk",
        chopper,
        input_path,
        output_path,
    )
    assert encoded_count == 3
    assert chopper.calls == [(input_path, "json", output_path)]
    assert json.loads(output_path.read_text(encoding="utf-8")) == [1, 2, 3]


def test_tiktoken_throughput_uses_preloaded_string_and_array(
    tmp_path: Path,
) -> None:
    class Encoder:
        @staticmethod
        def encode_to_numpy(
            text: str,
            *,
            allowed_special: str,
        ) -> np.ndarray:
            assert text == "sample"
            assert allowed_special == "all"
            return np.asarray([1, 2, 3], dtype=np.uint32)

    input_path = tmp_path / "input.txt"
    input_path.write_text("sample", encoding="utf-8")
    prepared = TIKTOKEN_SUITE.prepare_input("tiktoken", input_path)
    assert prepared == "sample"
    assert TIKTOKEN_SUITE.encode_throughput(
        "tiktoken",
        Encoder(),
        prepared,
        None,
    ) == 3
    assert TIKTOKEN_SUITE.prepare_input("gigatoken", input_path) == input_path
    assert TIKTOKEN_SUITE.prepare_input("hiriluk", input_path) == input_path


def test_gigatoken_dump_writes_readable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "tokens.json"
    input_path.write_text("sample", encoding="utf-8")

    fake_awkward = SimpleNamespace(
        num=lambda rows: [len(row) for row in rows],
        sum=sum,
        flatten=lambda rows, axis=None: [
            token for row in rows for token in row
        ],
        to_numpy=lambda ids: np.asarray(ids, dtype=np.uint32),
    )

    class Encoder:
        @staticmethod
        def encode_files(source: Path, *, parallel: bool):
            assert source == input_path
            assert parallel is False
            return [[1, 2], [3]]

    monkeypatch.setitem(sys.modules, "awkward", fake_awkward)

    assert TIKTOKEN_SUITE.encode_throughput(
        "gigatoken",
        Encoder(),
        input_path,
        output_path,
    ) == 3
    assert output_path.read_text(encoding="utf-8") == "[1,2,3]"


def test_hiriluk_rss_run_retains_numpy_compatible_array() -> None:
    class Chopper:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, str]] = []

        def chop_file(self, path: Path, *, output: str) -> np.ndarray:
            self.calls.append((path, output))
            return np.asarray([1, 2, 3], dtype=np.uint32)

    chopper = Chopper()
    result = TIKTOKEN_SUITE.encode_rss(
        "hiriluk",
        chopper,
        Path("/tmp/input.txt"),
        None,
    )
    assert chopper.calls == [(Path("/tmp/input.txt"), "array")]
    assert result.tokens == 3
    assert isinstance(result.retained, np.ndarray)
    assert result.retained.dtype == np.uint32
    assert result.retained.tolist() == [1, 2, 3]
    assert result.output_mode == "numpy-u32-array"


def test_hiriluk_dfa_rss_run_streams_json(tmp_path: Path) -> None:
    output_path = tmp_path / "tokens.json"
    class Chopper:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, str, Path]] = []

        def chop_file(
            self,
            path: Path,
            *,
            output: str,
            dump: Path,
        ) -> int:
            self.calls.append((path, output, dump))
            dump.write_text("[1,2,3]", encoding="utf-8")
            return 3

    chopper = Chopper()
    input_path = tmp_path / "input.txt"
    input_path.write_text("sample", encoding="utf-8")
    result = TIKTOKEN_SUITE.encode_rss(
        "hiriluk",
        chopper,
        input_path,
        output_path,
    )

    assert chopper.calls == [(input_path, "json", output_path)]
    assert result.tokens == 3
    assert result.retained is None
    assert result.output_mode == "streamed-json-file"
    assert json.loads(output_path.read_text(encoding="utf-8")) == [1, 2, 3]


def test_tiktoken_dfa_rss_run_retains_array(
    tmp_path: Path,
) -> None:
    class Encoder:
        @staticmethod
        def encode_to_numpy(
            text: str,
            *,
            allowed_special: str,
        ) -> np.ndarray:
            assert text == "sample"
            assert allowed_special == "all"
            return np.asarray([1, 2, 3], dtype=np.uint32)

    input_path = tmp_path / "input.txt"
    input_path.write_text("sample", encoding="utf-8")
    prepared = TIKTOKEN_SUITE.prepare_input("tiktoken", input_path)
    result = TIKTOKEN_SUITE.encode_rss(
        "tiktoken",
        Encoder(),
        prepared,
        None,
    )

    assert result.tokens == 3
    assert result.retained.tolist() == [1, 2, 3]
    assert result.output_mode == "numpy-u32-array"


def test_worker_uses_fixed_command_and_stdin_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"tokens":3}\n',
            stderr="",
        )

    monkeypatch.setattr(common.subprocess, "run", run)
    job = throughput.worker_job(
        TIKTOKEN_SUITE,
        "throughput",
        "hiriluk",
        "r50k",
        dfa=False,
        corpus="english",
        path=Path("/tmp/input.txt"),
    )
    assert throughput.run_worker(job) == {"tokens": 3}
    assert observed["command"] == [
        common.sys.executable,
        "-m",
        "tools.benchmark_worker",
    ]
    assert json.loads(str(observed["input"])) == job


def test_output_base_directory_prefers_tmpfs_then_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An explicit override wins over everything else.
    monkeypatch.setenv(throughput.OUTPUT_DIR_ENV, str(tmp_path))
    assert throughput.output_base_directory() == tmp_path
    monkeypatch.delenv(throughput.OUTPUT_DIR_ENV)

    # A writable tmpfs candidate is preferred over the ordinary temp directory.
    fake_tmpfs = tmp_path / "shm"
    fake_tmpfs.mkdir()
    monkeypatch.setattr(throughput, "TMPFS_CANDIDATES", (str(fake_tmpfs),))
    monkeypatch.setattr(
        throughput,
        "filesystem_type",
        lambda path: "tmpfs" if path == fake_tmpfs else "xfs",
    )
    assert throughput.output_base_directory() == fake_tmpfs

    # Without one, output falls back to the ordinary temporary directory, which
    # still honors TMPDIR. macOS reaches this path. `gettempdir` memoizes its
    # answer on first use, so clear the cache the same way a fresh process would
    # start; in production TMPDIR is read once, before any benchmark runs.
    monkeypatch.setattr(throughput, "filesystem_type", lambda _path: "xfs")
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    monkeypatch.setenv("TMPDIR", str(fallback))
    monkeypatch.setattr(throughput.tempfile, "tempdir", None)
    assert throughput.output_base_directory() == fallback


def test_throughput_output_is_created_under_the_chosen_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(throughput.OUTPUT_DIR_ENV, str(tmp_path))
    with throughput.throughput_output_destination(
        TIKTOKEN_SUITE,
        True,
        False,
        "hiriluk",
        "r50k",
    ) as destination:
        assert destination is not None
        assert destination.parent.parent == tmp_path
        parent = destination.parent
    assert not parent.exists()


@pytest.mark.skipif(
    not Path("/proc/mounts").exists(),
    reason="filesystem type detection reads /proc/mounts",
)
def test_filesystem_type_identifies_real_mounts() -> None:
    assert throughput.filesystem_type(Path("/proc")) == "proc"
    if Path("/dev/shm").is_dir():
        assert throughput.filesystem_type(Path("/dev/shm")) == "tmpfs"


def test_throughput_output_policy_matches_all_three_modes() -> None:
    with throughput.throughput_output_destination(
        TIKTOKEN_SUITE,
        False,
        False,
        "hiriluk",
        "r50k",
    ) as fast_default:
        assert fast_default is None

    with throughput.throughput_output_destination(
        TIKTOKEN_SUITE,
        False,
        True,
        "hiriluk",
        "r50k",
    ) as fast_dump:
        assert fast_dump is not None
        assert fast_dump.name == "tokens.json"
        fast_dump_parent = fast_dump.parent
        assert fast_dump_parent.is_dir()
    assert not fast_dump_parent.exists()

    with throughput.throughput_output_destination(
        TIKTOKEN_SUITE,
        False,
        True,
        "gigatoken",
        "r50k",
    ) as gigatoken_dump:
        assert gigatoken_dump is not None
        assert gigatoken_dump.name == "tokens.json"

    with throughput.throughput_output_destination(
        TIKTOKEN_SUITE,
        False,
        True,
        "tiktoken",
        "r50k",
    ) as tiktoken_dump:
        assert tiktoken_dump is None

    with throughput.throughput_output_destination(
        TIKTOKEN_SUITE,
        True,
        False,
        "hiriluk",
        "r50k",
    ) as dfa_temporary:
        assert dfa_temporary is not None
        assert dfa_temporary.name == "tokens.json"
        temporary_parent = dfa_temporary.parent
        assert temporary_parent.is_dir()
    assert not temporary_parent.exists()

    with throughput.throughput_output_destination(
        TIKTOKEN_SUITE,
        True,
        False,
        "tiktoken",
        "r50k",
    ) as dfa_tiktoken:
        assert dfa_tiktoken is None


def test_exact_parity_rejects_same_length_different_ids() -> None:
    rows = [
        {
            "impl": "a",
            "encoding": "r50k",
            "corpus": "english",
            "tokens": 3,
            "digest": "aaa",
        },
        {
            "impl": "b",
            "encoding": "r50k",
            "corpus": "english",
            "tokens": 3,
            "digest": "bbb",
        },
    ]
    with pytest.raises(RuntimeError, match="token-ID mismatch"):
        throughput.validate_parity(rows, exact_ids=True)


def test_public_parser_has_no_private_flags() -> None:
    parser = throughput.build_throughput_parser(TIKTOKEN_SUITE)
    args = parser.parse_args(())
    assert args.encoding == "all"
    assert args.dataset == "all"
    assert args.corpus_size == "short"
    assert not hasattr(args, "dump")
    assert parser.parse_args(("--tiny",)).corpus_size == "tiny"
    assert parser.parse_args(("--full",)).corpus_size == "full"
    help_text = parser.format_help()
    for option in (
        "--dump",
        "--warmup",
        "--reps",
        "--_worker",
        "--_impl",
        "--_encoding",
        "--_path",
        "--_label",
        "--_output-path",
    ):
        assert option not in help_text
        with pytest.raises(SystemExit):
            parser.parse_args((option, "value"))


def test_data_dir_environment_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert common.data_directory() == tmp_path.resolve()
