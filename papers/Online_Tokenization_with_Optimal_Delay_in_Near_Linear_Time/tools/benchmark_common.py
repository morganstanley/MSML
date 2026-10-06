"""Shared building blocks for the Python tokenizer benchmarks.

Corpus resolution, argument parsing, table printing, and worker-process
plumbing used by the throughput, RSS, and TTFT benchmarks alike.
Throughput-only orchestration (streamed output destinations, the timed-run
harness, the isolated worker protocol) lives in `tools/throughput_bench.py`.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


REPO_DIR = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
SHORT_TARGET_BYTES = 128 * MIB
SHORT_TARGET_TOLERANCE_BYTES = 4 * MIB

# Change these constants when a different benchmark duration is useful. They
# deliberately are not command-line options: all rows in one checkout use the
# same methodology, including the isolated worker processes.
WARMUP_RUNS = 2
MEASURED_RUNS = 10
TTFT_RUNS = 30

CORPORA = {
    "github": {
        "tiny": "github/sampled_2500_stream_tiny.txt",
        "short": "github/stride_17_stream_short.txt",
        "full": "github/sampled_2500_stream.txt",
    },
    "english": {
        "tiny": "english/parquet_0_stream_tiny.txt",
        "short": "english/parquet_0_stream_short.txt",
        "full": "english/parquet_0_stream.txt",
    },
    "chinese": {
        "tiny": "chinese/parquet_0_stream_tiny.txt",
        "short": "chinese/parquet_0_stream_short.txt",
        "full": "chinese/parquet_0_stream.txt",
    },
}
ALL_CORPORA = tuple(CORPORA)
CORPUS_SIZES = ("tiny", "short", "full")


@dataclass
class RssRun:
    """An RSS worker result while its native output is still alive."""

    tokens: int
    output_mode: str
    retained: Any


class BenchmarkSuite(Protocol):
    """Tokenizer-specific operations used by the shared benchmark runner."""

    suite_id: str
    description: str
    epilog: str
    item_label: str
    selector_flags: tuple[str, ...]
    selector_help: str
    implementations: tuple[str, ...]
    supports_dfa: bool
    force_dfa: bool
    dump_help: str
    supports_dump: bool
    throughput_exact_ids: bool
    final_note: str

    def selected_items(self, requested: str) -> list[Any]: ...

    def item_named(self, name: str) -> Any: ...

    def supports(self, implementation: str, item: Any) -> bool: ...

    def implementations_for(self, *, dfa: bool) -> tuple[str, ...]: ...

    def rss_implementations_for(self, *, dfa: bool) -> tuple[str, ...]: ...

    def preload(
        self,
        implementation: str,
        item: Any,
        *,
        phase: str,
        output_path: Path | None,
    ) -> None: ...

    def construct(
        self,
        implementation: str,
        item: Any,
        *,
        dfa: bool,
        profile: bool,
    ) -> Any: ...

    def prepare_input(
        self,
        implementation: str,
        path: Path,
    ) -> Any: ...

    def encode_throughput(
        self,
        implementation: str,
        encoder: Any,
        input_value: Any,
        output_path: Path | None,
    ) -> int: ...

    def encode_verification(
        self,
        implementation: str,
        encoder: Any,
        input_value: Any,
        output_path: Path | None,
    ) -> tuple[int, str]: ...

    def encode_rss(
        self,
        implementation: str,
        encoder: Any,
        input_value: Any,
        output_path: Path | None,
    ) -> RssRun: ...

    def throughput_output_mode(
        self,
        implementation: str,
        output_path: Path | None,
    ) -> str: ...

    def mode_description(self, *, dfa: bool) -> str: ...

    def throughput_note(
        self,
        *,
        dfa: bool,
    ) -> str: ...

    def throughput_output_extension(
        self,
        implementation: str,
        *,
        dfa: bool,
        dump: bool,
    ) -> str | None: ...

    def rss_output_extension(
        self,
        implementation: str,
        *,
        dfa: bool,
        dump: bool,
    ) -> str | None: ...

    def rss_output_policy(self, *, dfa: bool, dump: bool) -> str: ...


def selected_corpora(requested: str) -> list[str]:
    if requested == "all":
        return list(ALL_CORPORA)
    name = requested.removesuffix("_stream")
    if name not in CORPORA:
        choices = ", ".join(("all", *CORPORA))
        raise ValueError(
            f"unknown dataset {requested!r}; expected one of {choices}"
        )
    return [name]


def read_paths_env() -> dict[str, str]:
    values: dict[str, str] = {}
    path = REPO_DIR / "paths.env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(
            f"DATA_DIR is unset and {path} could not be read: {error}"
        ) from error
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def data_directory() -> Path:
    configured = os.environ.get("DATA_DIR")
    if configured is None:
        configured = read_paths_env().get("DATA_DIR")
    if not configured:
        raise RuntimeError("DATA_DIR must be exported or defined in paths.env")
    return Path(configured).expanduser().resolve()


def resolve_corpora(
    root: Path,
    selected: Sequence[str],
    corpus_size: str,
) -> list[tuple[str, Path]]:
    if corpus_size not in CORPUS_SIZES:
        choices = ", ".join(CORPUS_SIZES)
        raise ValueError(
            f"unknown corpus size {corpus_size!r}; expected one of {choices}"
        )
    resolved: list[tuple[str, Path]] = []
    for label in selected:
        path = root / CORPORA[label][corpus_size]
        if not path.is_file():
            raise FileNotFoundError(f"corpus does not exist: {path}")
        if corpus_size == "short" and abs(
            path.stat().st_size - SHORT_TARGET_BYTES
        ) > SHORT_TARGET_TOLERANCE_BYTES:
            raise RuntimeError(
                f"default corpus is not approximately 128 MiB: {path} "
                f"({path.stat().st_size / MIB:.1f} MiB); rebuild it with "
                "`python tools/get_data.py --force-build` or use --tiny/--full"
            )
        resolved.append((label, path))
    return resolved


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as corpus:
        return corpus.read()


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    strings = [[str(value) for value in headers]]
    strings.extend([[str(value) for value in row] for row in rows])
    widths = [
        max(len(row[column]) for row in strings)
        for column in range(len(headers))
    ]
    for index, row in enumerate(strings):
        print(
            "  ".join(
                value.ljust(widths[column])
                for column, value in enumerate(row)
            ).rstrip()
        )
        if index == 0:
            print("  ".join("-" * width for width in widths).rstrip())


def print_unavailable(
    title: str,
    item_label: str,
    rows: Sequence[tuple[str, str, str, str]],
) -> None:
    if not rows:
        return
    print()
    print(title)
    print_table(("impl", item_label, "corpus", "reason"), rows)


def median_quartiles(
    values: Sequence[float],
) -> tuple[float | None, float | None, float | None]:
    """Median with its inner quartiles.

    The median leads because one slow repetition, from shared storage or a
    co-tenant, moves a mean far more than it moves the middle of the runs.
    """

    if not values:
        return None, None, None
    if len(values) == 1:
        only = values[0]
        return only, only, only
    p25, _, p75 = statistics.quantiles(values, n=4, method="inclusive")
    return statistics.median(values), p25, p75


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def current_rss_bytes() -> int:
    if sys.platform == "darwin":

        class TimeValue(ctypes.Structure):
            _fields_ = [
                ("seconds", ctypes.c_int32),
                ("microseconds", ctypes.c_int32),
            ]

        class MachTaskBasicInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time", TimeValue),
                ("system_time", TimeValue),
                ("policy", ctypes.c_int32),
                ("suspend_count", ctypes.c_int32),
            ]

        libsystem = ctypes.CDLL(None)
        libsystem.mach_task_self.restype = ctypes.c_uint32
        libsystem.task_info.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        info = MachTaskBasicInfo()
        count = ctypes.c_uint32(
            ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32)
        )
        rc = libsystem.task_info(
            libsystem.mach_task_self(),
            20,  # MACH_TASK_BASIC_INFO
            ctypes.byref(info),
            ctypes.byref(count),
        )
        return int(info.resident_size) if rc == 0 else peak_rss_bytes()

    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(
                Path("/proc/self/statm")
                .read_text(encoding="ascii")
                .split()[1]
            )
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
    return peak_rss_bytes()


def write_json_ids(ids: Any, output_path: Path) -> None:
    """Write integer IDs as bounded-memory, whitespace-free JSON."""
    with output_path.open("wb") as output:
        output.writelines(json_id_chunks(ids))


def json_id_chunks(ids: Any) -> Iterator[bytes]:
    """Yield the canonical JSON representation without materializing it."""
    yield b"["
    first = True
    for start in range(0, len(ids), 65_536):
        values = ids[start : start + 65_536]
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not values:
            continue
        prefix = "" if first else ","
        yield (prefix + ",".join(map(str, values))).encode("ascii")
        first = False
    yield b"]"


def array_digest(ids: Any) -> str:
    import numpy as np

    packed = np.asarray(ids, dtype="<u4", order="C")
    return hashlib.sha256(memoryview(packed).cast("B")).hexdigest()


def run_json_worker(
    module: str,
    job: dict[str, object],
) -> dict[str, Any]:
    """Run one fixed isolated worker using JSON over standard input."""

    command = [sys.executable, "-m", module]
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            input=json.dumps(job, separators=(",", ":")),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=REPO_DIR,
            env=os.environ.copy(),
        )
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(
            details or f"worker exited {error.returncode}"
        ) from error

    for line in reversed(completed.stdout.splitlines()):
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    raise RuntimeError(
        "worker did not return a benchmark result"
        + (f":\n{completed.stdout}" if completed.stdout else "")
    )


def concise_error(error: BaseException) -> str:
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("error:"):
            return line[:180]
    return (lines[0] if lines else type(error).__name__)[:180]


def build_selection_parser(
    suite: BenchmarkSuite,
    *,
    description: str | None = None,
    epilog: str | None = None,
    dfa_help: str | None = None,
) -> argparse.ArgumentParser:
    """Build the arguments shared by throughput, RSS, and TTFT experiments."""

    parser = argparse.ArgumentParser(
        description=suite.description if description is None else description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=suite.epilog if epilog is None else epilog,
    )
    parser.add_argument(
        *suite.selector_flags,
        dest=suite.item_label,
        default="all",
        help=suite.selector_help,
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "github[_stream], english[_stream], chinese[_stream], or all "
            "(default: all)"
        ),
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--tiny",
        dest="corpus_size",
        action="store_const",
        const="tiny",
        help="use each approximately 16 MiB paper subset",
    )
    size.add_argument(
        "--short",
        dest="corpus_size",
        action="store_const",
        const="short",
        help="use each approximately 128 MiB sample (default)",
    )
    size.add_argument(
        "--full",
        dest="corpus_size",
        action="store_const",
        const="full",
        help="use each complete source stream",
    )
    parser.set_defaults(corpus_size="short")
    if suite.supports_dfa:
        parser.add_argument(
            "--dfa",
            action="store_true",
            help=dfa_help
            or (
                "use Hiriluk's reference DFA path instead of its "
                "SIMD/full-cache path and omit Gigatoken"
            ),
        )
    return parser


# Inputs whose edits require rebuilding the compiled extension. Python sources
# are excluded: they are imported from the tree, so they are never stale.
_RUST_SOURCE_GLOBS = (
    "src/**/*.rs",
    "bindings/python/src/**/*.rs",
    "Cargo.toml",
    "Cargo.lock",
    "bindings/python/Cargo.toml",
)
_STALE_EXTENSION_OVERRIDE = "HIRILUK_ALLOW_STALE_EXTENSION"


def assert_extension_current() -> None:
    """Refuse to benchmark a compiled extension older than the Rust sources.

    An editable install puts `<repo>/python` on `sys.path` through a plain
    `hiriluk.pth`, so `python/hiriluk/_hiriluk*.so` is imported straight out of
    the working tree with no rebuild hook and no staleness check of any kind.
    Editing `src/` and forgetting `maturin develop --release` therefore
    measures stale code and reports it as an ordinary result, which is far
    worse than a stopped run. Wheel installs have no tree to compare against
    and are skipped.

    Set `HIRILUK_ALLOW_STALE_EXTENSION=1` to measure the current build anyway,
    which is useful when a branch switch has rewritten source mtimes without
    changing the compiled behavior.
    """

    if os.environ.get(_STALE_EXTENSION_OVERRIDE):
        return
    package_dir = REPO_DIR / "python"
    if str(package_dir) not in sys.path:
        return
    built = [
        *package_dir.glob("hiriluk/_hiriluk*.so"),
        *package_dir.glob("hiriluk/_hiriluk*.pyd"),
    ]
    if not built:
        return

    # Compare the oldest artifact: any one of them being stale is a problem.
    extension = min(built, key=lambda path: path.stat().st_mtime)
    extension_mtime = extension.stat().st_mtime
    newest: Path | None = None
    newest_mtime = extension_mtime
    for pattern in _RUST_SOURCE_GLOBS:
        for source in REPO_DIR.glob(pattern):
            mtime = source.stat().st_mtime
            if mtime > newest_mtime:
                newest, newest_mtime = source, mtime
    if newest is None:
        return

    def stamp(value: float) -> str:
        return datetime.fromtimestamp(value).isoformat(sep=" ", timespec="seconds")

    raise SystemExit(
        "error: the compiled hiriluk extension is older than the Rust "
        "sources, so this run would measure stale code.\n"
        f"  extension:    {extension.relative_to(REPO_DIR)} "
        f"({stamp(extension_mtime)})\n"
        f"  newer source: {newest.relative_to(REPO_DIR)} "
        f"({stamp(newest_mtime)})\n"
        "Rebuild with:\n"
        "  python -m maturin develop --release --locked\n"
        f"or set {_STALE_EXTENSION_OVERRIDE}=1 to benchmark the current "
        "build anyway."
    )


def resolve_inputs(
    suite: BenchmarkSuite,
    args: argparse.Namespace,
) -> tuple[list[Any], list[tuple[str, Path]]]:
    """Resolve a suite's selected tokenizer items and corpus files."""

    # Every throughput/RSS/TTFT suite funnels through here, so one call covers
    # them all, including suites added later.
    assert_extension_current()
    requested = getattr(args, suite.item_label)
    items = suite.selected_items(requested)
    selected = selected_corpora(args.dataset)
    root = data_directory()
    os.environ["DATA_DIR"] = str(root)
    return items, resolve_corpora(root, selected, args.corpus_size)
