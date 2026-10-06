#!/usr/bin/env python3
"""Download and build the benchmark corpora from Jiang and Gong (2026).

The paper uses:

* English: the first training Parquet shard of Wikimedia Wikipedia
  ``20231101.en``, taking every 42nd document for the benchmark subset.
* Chinese: the first training Parquet shard of Wikimedia Wikipedia
  ``20231101.zh``, taking every 60th document for the benchmark subset.
* GitHub: RedPajama's
  ``filtered_08cdfa755e6d4d89b673d5bd1acee5f6.sampled.jsonl``, taking its
  first 2,500 documents for the benchmark subset.

For each corpus this program makes four benchmark inputs beneath DATA_DIR:

* the existing batch JSONL containing the paper's subset;
* ``*_stream_tiny.txt``, the same approximately 16 MiB paper subset;
* ``*_stream_short.txt``, a deterministic approximately 128 MiB sample;
* ``*_stream.txt``, every source-shard/file document joined the same way.

Source downloads, JSONL parsing, Parquet decoding, and text generation are all
streaming. Memory is bounded by one download block or one Parquet record batch,
plus the largest individual document. Derived files are written atomically.

Paper: https://arxiv.org/abs/2605.30813
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, TextIO

if __package__:
    from .config import data_dir
else:
    from config import data_dir


DOWNLOAD_BLOCK_SIZE = 8 * 1024 * 1024
DEFAULT_PARQUET_BATCH_SIZE = 256
USER_AGENT = "hiriluk/get_data.py"
BUILD_FORMAT_VERSION = 2
MIB = 1024 * 1024
SHORT_TARGET_BYTES = 128 * MIB
SHORT_TARGET_TOLERANCE_BYTES = 4 * MIB


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_url: str
    source_path: str
    source_format: str
    batch_path: str
    tiny_stream_path: str
    short_stream_path: str
    full_stream_path: str
    tiny_keep: Callable[[int], bool]
    short_keep: Callable[[int], bool]
    tiny_sample_description: str
    short_sample_description: str
    expected_source_bytes: int
    expected_source_sha256: str | None
    expected_tiny_documents: int
    expected_tiny_text_bytes: int
    expected_short_documents: int


ENGLISH_REVISION = "ad5752b5e625abfcdeefe5ae0ad2c3721c4b2619"
CHINESE_REVISION = "35bb200cf57ba3928fd2ead1cf58398119b83d82"
REDPAJAMA_GITHUB_FILE = (
    "filtered_08cdfa755e6d4d89b673d5bd1acee5f6.sampled.jsonl"
)


DATASETS: dict[str, DatasetSpec] = {
    "english": DatasetSpec(
        name="english",
        source_url=(
            "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/"
            f"{ENGLISH_REVISION}/20231101.en/"
            "train-00000-of-00041.parquet"
        ),
        source_path="english/train-00000-of-00041.parquet",
        source_format="parquet",
        batch_path="english/parquet_0_stride_42.jsonl",
        tiny_stream_path="english/parquet_0_stream_tiny.txt",
        short_stream_path="english/parquet_0_stream_short.txt",
        full_stream_path="english/parquet_0_stream.txt",
        tiny_keep=lambda index: index % 42 == 0,
        short_keep=lambda index: index % 21 in (0, 5, 10, 16),
        tiny_sample_description="every 42nd document",
        short_sample_description=(
            "documents 0, 5, 10, and 16 in each block of 21"
        ),
        expected_source_bytes=420_296_449,
        expected_source_sha256=(
            "382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677"
        ),
        expected_tiny_documents=3_722,
        expected_tiny_text_bytes=16_522_972,
        expected_short_documents=29_770,
    ),
    "chinese": DatasetSpec(
        name="chinese",
        source_url=(
            "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/"
            f"{CHINESE_REVISION}/20231101.zh/"
            "train-00000-of-00006.parquet"
        ),
        source_path="chinese/train-00000-of-00006.parquet",
        source_format="parquet",
        batch_path="chinese/parquet_0_stride_60.jsonl",
        tiny_stream_path="chinese/parquet_0_stream_tiny.txt",
        short_stream_path="chinese/parquet_0_stream_short.txt",
        full_stream_path="chinese/parquet_0_stream.txt",
        tiny_keep=lambda index: index % 60 == 0,
        short_keep=lambda index: index % 13 in (0, 6),
        tiny_sample_description="every 60th document",
        short_sample_description="documents 0 and 6 in each block of 13",
        expected_source_bytes=587_109_027,
        expected_source_sha256=(
            "853ddf4a138c792ad7386d6de831788f96326c03e1e3349f7a315498392e8b56"
        ),
        expected_tiny_documents=3_847,
        expected_tiny_text_bytes=16_337_583,
        expected_short_documents=35_507,
    ),
    "github": DatasetSpec(
        name="github",
        source_url=(
            "https://data.together.xyz/redpajama-data-1T/v1.0.0/github/"
            f"{REDPAJAMA_GITHUB_FILE}"
        ),
        source_path=f"github/{REDPAJAMA_GITHUB_FILE}",
        source_format="jsonl",
        batch_path="github/sampled_2500.jsonl",
        tiny_stream_path="github/sampled_2500_stream_tiny.txt",
        short_stream_path="github/stride_17_stream_short.txt",
        full_stream_path="github/sampled_2500_stream.txt",
        tiny_keep=lambda index: index < 2_500,
        short_keep=lambda index: index % 17 == 0,
        tiny_sample_description="the first 2,500 documents",
        short_sample_description="every 17th document",
        expected_source_bytes=2_658_456_835,
        expected_source_sha256=(
            "130ee049c763f5e801e2febf7957b2798bded4711895ad11e1434e8c6b645742"
        ),
        expected_tiny_documents=2_500,
        expected_tiny_text_bytes=16_803_408,
        expected_short_documents=19_686,
    ),
}

ALIASES = {
    "en": "english",
    "zh": "chinese",
    "code": "github",
}


@dataclass(frozen=True)
class BuildStats:
    total_documents: int
    tiny_documents: int
    short_documents: int
    full_text_bytes: int
    tiny_text_bytes: int
    short_text_bytes: int


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(DOWNLOAD_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def validate_source(path: Path, spec: DatasetSpec) -> None:
    actual_size = path.stat().st_size
    if actual_size != spec.expected_source_bytes:
        raise RuntimeError(
            f"{path} has {actual_size:,} bytes; expected "
            f"{spec.expected_source_bytes:,}"
        )
    if spec.expected_source_sha256 is not None:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != spec.expected_source_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: expected "
                f"{spec.expected_source_sha256}, got {actual_sha256}"
            )


def download_source(
    spec: DatasetSpec,
    root: Path,
    *,
    force: bool,
    timeout: float,
) -> Path:
    """Download one source file atomically, resuming a .part file when possible."""
    destination = root / spec.source_path
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        try:
            validate_source(destination, spec)
        except RuntimeError as error:
            print(f"{error}; downloading a verified replacement", file=sys.stderr)
        else:
            print(f"[{spec.name}] source already present: {destination}")
            return destination

    if force and partial.exists():
        partial.unlink()

    resume_at = partial.stat().st_size if partial.exists() else 0
    if resume_at > spec.expected_source_bytes:
        partial.unlink()
        resume_at = 0
    elif resume_at == spec.expected_source_bytes:
        try:
            validate_source(partial, spec)
        except RuntimeError:
            print(
                f"[{spec.name}] complete .part file failed validation; "
                "restarting download"
            )
            partial.unlink()
            resume_at = 0
        else:
            os.replace(partial, destination)
            print(f"[{spec.name}] source ready: {destination}")
            return destination

    headers = {"User-Agent": USER_AGENT}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"
        print(
            f"[{spec.name}] resuming at {human_bytes(resume_at)}: "
            f"{spec.source_url}"
        )
    else:
        print(
            f"[{spec.name}] downloading {human_bytes(spec.expected_source_bytes)}: "
            f"{spec.source_url}"
        )

    request = urllib.request.Request(spec.source_url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code == 416 and resume_at == spec.expected_source_bytes:
            validate_source(partial, spec)
            os.replace(partial, destination)
            return destination
        raise RuntimeError(
            f"download failed for {spec.name}: HTTP {error.code} {error.reason}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"download failed for {spec.name}: {error.reason}"
        ) from error

    with response:
        status = getattr(response, "status", response.getcode())
        append = resume_at > 0 and status == 206
        if append:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {resume_at}-"):
                response.close()
                print(
                    f"[{spec.name}] invalid Content-Range {content_range!r}; "
                    "restarting download"
                )
                return download_source(
                    spec,
                    root,
                    force=True,
                    timeout=timeout,
                )
        if resume_at and not append:
            print(
                f"[{spec.name}] server did not honor Range; restarting download"
            )
            resume_at = 0

        downloaded = resume_at
        started = time.monotonic()
        last_report = started
        with partial.open("ab" if append else "wb") as output:
            while block := response.read(DOWNLOAD_BLOCK_SIZE):
                output.write(block)
                downloaded += len(block)
                now = time.monotonic()
                if now - last_report >= 5.0:
                    elapsed = max(now - started, 1e-9)
                    transferred = downloaded - resume_at
                    percent = 100.0 * downloaded / spec.expected_source_bytes
                    print(
                        f"[{spec.name}] {percent:5.1f}% "
                        f"({human_bytes(downloaded)}, "
                        f"{human_bytes(int(transferred / elapsed))}/s)"
                    )
                    last_report = now

    validate_source(partial, spec)
    os.replace(partial, destination)
    print(f"[{spec.name}] source ready: {destination}")
    return destination


def stream_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    """Yield one JSON object per nonempty physical line."""
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object, "
                    f"got {type(row).__name__}"
                )
            yield row


def stream_parquet(
    path: Path,
    *,
    batch_size: int = DEFAULT_PARQUET_BATCH_SIZE,
) -> Iterator[Mapping[str, object]]:
    """Yield Parquet rows in bounded record batches."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "English and Chinese preparation requires pyarrow; install it with "
            "`python -m pip install pyarrow` in the active environment"
        ) from error

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def source_rows(
    path: Path,
    source_format: str,
    *,
    parquet_batch_size: int,
) -> Iterable[Mapping[str, object]]:
    if source_format == "jsonl":
        return stream_jsonl(path)
    if source_format == "parquet":
        return stream_parquet(path, batch_size=parquet_batch_size)
    raise ValueError(f"unsupported source format: {source_format}")


def temporary_output(path: Path) -> tuple[Path, TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    return temporary_path, os.fdopen(
        descriptor,
        "w",
        encoding="utf-8",
        newline="",
    )


def row_text(row: Mapping[str, object], spec: DatasetSpec, index: int) -> str:
    text = row.get("text")
    if not isinstance(text, str):
        kind = "missing" if "text" not in row else type(text).__name__
        raise ValueError(
            f"{spec.name} document {index}: expected string field `text`, "
            f"got {kind}"
        )
    # The paper's reported GitHub byte count uses universal-newline text:
    # its first 2,500 records contain 18,664 CRLF pairs, exactly the difference
    # between the source's UTF-8 size and Table 2's 16,803,408 bytes.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def validate_build(spec: DatasetSpec, stats: BuildStats) -> None:
    problems: list[str] = []
    if stats.tiny_documents != spec.expected_tiny_documents:
        problems.append(
            f"{stats.tiny_documents:,} tiny documents "
            f"(paper: {spec.expected_tiny_documents:,})"
        )
    if stats.tiny_text_bytes != spec.expected_tiny_text_bytes:
        problems.append(
            f"{stats.tiny_text_bytes:,} tiny text bytes "
            f"(paper: {spec.expected_tiny_text_bytes:,})"
        )
    if stats.short_documents != spec.expected_short_documents:
        problems.append(
            f"{stats.short_documents:,} short documents "
            f"(expected: {spec.expected_short_documents:,})"
        )
    short_stream_bytes = stats.short_text_bytes + max(
        stats.short_documents - 1,
        0,
    )
    if abs(short_stream_bytes - SHORT_TARGET_BYTES) > (
        SHORT_TARGET_TOLERANCE_BYTES
    ):
        problems.append(
            f"{human_bytes(short_stream_bytes)} short stream "
            f"(target: {human_bytes(SHORT_TARGET_BYTES)} ± "
            f"{human_bytes(SHORT_TARGET_TOLERANCE_BYTES)})"
        )
    if problems:
        raise RuntimeError(
            f"{spec.name} failed corpus validation: " + "; ".join(problems)
        )


def build_outputs(
    spec: DatasetSpec,
    source_path: Path,
    root: Path,
    *,
    parquet_batch_size: int,
    validate: bool = True,
) -> BuildStats:
    """Build the paper batch subset and all streams in one bounded-memory pass."""
    final_paths = (
        root / spec.batch_path,
        root / spec.tiny_stream_path,
        root / spec.short_stream_path,
        root / spec.full_stream_path,
    )
    temporary_paths: list[Path] = []
    outputs: list[TextIO] = []

    total_documents = 0
    tiny_documents = 0
    short_documents = 0
    full_text_bytes = 0
    tiny_text_bytes = 0
    short_text_bytes = 0

    try:
        for path in final_paths:
            temporary_path, output = temporary_output(path)
            temporary_paths.append(temporary_path)
            outputs.append(output)
        batch_output, tiny_output, short_output, full_output = outputs

        rows = source_rows(
            source_path,
            spec.source_format,
            parquet_batch_size=parquet_batch_size,
        )
        for index, row in enumerate(rows):
            text = row_text(row, spec, index)
            encoded_bytes = len(text.encode("utf-8"))

            if total_documents:
                full_output.write("\n")
            full_output.write(text)
            full_text_bytes += encoded_bytes
            total_documents += 1

            if spec.tiny_keep(index):
                sampled_row = dict(row)
                sampled_row["text"] = text
                json.dump(sampled_row, batch_output, ensure_ascii=False)
                batch_output.write("\n")
                if tiny_documents:
                    tiny_output.write("\n")
                tiny_output.write(text)
                tiny_documents += 1
                tiny_text_bytes += encoded_bytes

            if spec.short_keep(index):
                if short_documents:
                    short_output.write("\n")
                short_output.write(text)
                short_documents += 1
                short_text_bytes += encoded_bytes

        stats = BuildStats(
            total_documents=total_documents,
            tiny_documents=tiny_documents,
            short_documents=short_documents,
            full_text_bytes=full_text_bytes,
            tiny_text_bytes=tiny_text_bytes,
            short_text_bytes=short_text_bytes,
        )
        if validate:
            validate_build(spec, stats)
        for output in outputs:
            output.close()
    except BaseException:
        for output in outputs:
            try:
                output.close()
            except OSError:
                pass
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise
    else:
        for temporary_path, final_path in zip(temporary_paths, final_paths):
            os.replace(temporary_path, final_path)

    return stats


def manifest_path(spec: DatasetSpec, root: Path) -> Path:
    return (root / spec.full_stream_path).parent / ".get_data_manifest.json"


def manifest_payload(
    spec: DatasetSpec,
    root: Path,
    stats: BuildStats,
) -> dict[str, object]:
    output_paths = (
        spec.batch_path,
        spec.tiny_stream_path,
        spec.short_stream_path,
        spec.full_stream_path,
    )
    return {
        "format_version": BUILD_FORMAT_VERSION,
        "dataset": spec.name,
        "source_url": spec.source_url,
        "source_path": spec.source_path,
        "source_bytes": spec.expected_source_bytes,
        "source_sha256": spec.expected_source_sha256,
        "tiny_sample": spec.tiny_sample_description,
        "short_sample": spec.short_sample_description,
        "total_documents": stats.total_documents,
        "tiny_documents": stats.tiny_documents,
        "short_documents": stats.short_documents,
        "full_text_bytes": stats.full_text_bytes,
        "tiny_text_bytes": stats.tiny_text_bytes,
        "short_text_bytes": stats.short_text_bytes,
        "normalization": "universal-newlines",
        "document_separator": "\\n",
        "outputs": {
            path: (root / path).stat().st_size
            for path in output_paths
        },
    }


def write_manifest(
    spec: DatasetSpec,
    root: Path,
    stats: BuildStats,
) -> None:
    final_path = manifest_path(spec, root)
    temporary_path, output = temporary_output(final_path)
    try:
        json.dump(
            manifest_payload(spec, root, stats),
            output,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        output.write("\n")
        output.close()
        os.replace(temporary_path, final_path)
    except BaseException:
        try:
            output.close()
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def outputs_current(spec: DatasetSpec, root: Path) -> bool:
    path = manifest_path(spec, root)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return False
        expected_fields = {
            "format_version": BUILD_FORMAT_VERSION,
            "dataset": spec.name,
            "source_url": spec.source_url,
            "source_path": spec.source_path,
            "source_bytes": spec.expected_source_bytes,
            "source_sha256": spec.expected_source_sha256,
            "tiny_sample": spec.tiny_sample_description,
            "short_sample": spec.short_sample_description,
            "tiny_documents": spec.expected_tiny_documents,
            "tiny_text_bytes": spec.expected_tiny_text_bytes,
            "short_documents": spec.expected_short_documents,
            "normalization": "universal-newlines",
            "document_separator": "\\n",
        }
        if any(manifest.get(key) != value for key, value in expected_fields.items()):
            return False
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict):
            return False
        short_text_bytes = manifest.get("short_text_bytes")
        if not isinstance(short_text_bytes, int):
            return False
        short_stream_bytes = short_text_bytes + max(
            spec.expected_short_documents - 1,
            0,
        )
        if abs(short_stream_bytes - SHORT_TARGET_BYTES) > (
            SHORT_TARGET_TOLERANCE_BYTES
        ):
            return False
        for relative_path in (
            spec.batch_path,
            spec.tiny_stream_path,
            spec.short_stream_path,
            spec.full_stream_path,
        ):
            output_path = root / relative_path
            expected_size = outputs.get(relative_path)
            if (
                not isinstance(expected_size, int)
                or not output_path.is_file()
                or output_path.stat().st_size != expected_size
            ):
                return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def prepare_dataset(
    spec: DatasetSpec,
    root: Path,
    *,
    force_download: bool,
    force_build: bool,
    no_download: bool,
    timeout: float,
    parquet_batch_size: int,
    validate: bool,
) -> None:
    if outputs_current(spec, root) and not force_build:
        print(
            f"[{spec.name}] all derived files already exist; "
            "use --force-build to replace them"
        )
        return

    source_path = root / spec.source_path
    if no_download:
        if not source_path.is_file():
            raise FileNotFoundError(
                f"{source_path} does not exist and --no-download was requested"
            )
        validate_source(source_path, spec)
    else:
        source_path = download_source(
            spec,
            root,
            force=force_download,
            timeout=timeout,
        )

    print(
        f"[{spec.name}] building {spec.tiny_sample_description} tiny corpus, "
        f"{spec.short_sample_description} short corpus, and unsampled full corpus"
    )
    # This marker is the commit record for the four-file generation. Removing
    # it first makes any interrupted sequence of atomic file replacements
    # unambiguously incomplete and therefore rebuildable on the next run.
    manifest_path(spec, root).unlink(missing_ok=True)
    stats = build_outputs(
        spec,
        source_path,
        root,
        parquet_batch_size=parquet_batch_size,
        validate=validate,
    )
    write_manifest(spec, root, stats)
    print(
        f"[{spec.name}] ready: {stats.total_documents:,} full documents "
        f"({human_bytes(stats.full_text_bytes)} text), "
        f"{stats.tiny_documents:,} tiny documents "
        f"({human_bytes(stats.tiny_text_bytes)} text), "
        f"{stats.short_documents:,} short documents "
        f"({human_bytes(stats.short_text_bytes)} text)"
    )


def selected_datasets(values: list[str] | None) -> list[DatasetSpec]:
    if not values or "all" in values:
        if values and len(values) != 1:
            raise ValueError("`all` cannot be combined with another --dataset")
        names = list(DATASETS)
    else:
        names = []
        for value in values:
            name = ALIASES.get(value, value)
            if name not in names:
                names.append(name)
    return [DATASETS[name] for name in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and prepare the English, Chinese, and GitHub corpora used "
            "by Incremental BPE Tokenization (arXiv:2605.30813)."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("all", "english", "chinese", "github", "en", "zh", "code"),
        help=(
            "dataset to prepare; repeat for multiple datasets "
            "(default: all; aliases: en, zh, code)"
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help=(
            "download the source again even if verified, then rebuild all "
            "derived files"
        ),
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="replace existing derived JSONL and stream files",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="build only from an existing source file in DATA_DIR",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help=(
            "do not validate the exact paper/tiny sample or the 128 MiB "
            "short sample"
        ),
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=DEFAULT_PARQUET_BATCH_SIZE,
        metavar="ROWS",
        help=f"Parquet rows decoded at once (default: {DEFAULT_PARQUET_BATCH_SIZE})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="per-network-operation timeout (default: 60)",
    )
    args = parser.parse_args()
    if args.parquet_batch_size < 1:
        parser.error("--parquet-batch-size must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.force_download and args.no_download:
        parser.error("--force-download cannot be combined with --no-download")
    return args


def main() -> None:
    args = parse_args()
    try:
        specs = selected_datasets(args.dataset)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    root = Path(data_dir()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"DATA_DIR={root}")
    for spec in specs:
        prepare_dataset(
            spec,
            root,
            force_download=args.force_download,
            force_build=args.force_build or args.force_download,
            no_download=args.no_download,
            timeout=args.timeout,
            parquet_batch_size=args.parquet_batch_size,
            validate=not args.no_validate,
        )


if __name__ == "__main__":
    main()
