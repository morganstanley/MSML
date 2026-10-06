#!/usr/bin/env python3
"""Run the opt-in profiled Gigatoken checkout across the TTFT matrix."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from tools.benchmark_common import TTFT_RUNS, median_quartiles  # noqa: E402
from tools.config import data_dir, ttft_repos_dir  # noqa: E402


WARMUP = int(os.environ.get("WARMUP", "2"))
REPS = int(os.environ.get("REPS", str(TTFT_RUNS)))
GIGATOKEN_VERSION = "0.10.0"
GIGATOKEN_BASE_COMMIT = "34a1599f0c0ae7d7cd0d1c530e6522320158b360"
DATA_DIR = Path(data_dir()).expanduser().resolve()
_GIGATOKEN_REPO = os.environ.get("GIGATOKEN_TTFT_REPO")
GIGATOKEN_REPO = Path(
    _GIGATOKEN_REPO
    if _GIGATOKEN_REPO is not None
    else Path(ttft_repos_dir()) / "gigatoken-ttft"
).expanduser().resolve()
BASE_URL = "https://openaipublic.blob.core.windows.net/encodings/"
MODELS = {
    "r50k_base": ("r50k_base.tiktoken", "gpt2"),
    "cl100k_base": ("cl100k_base.tiktoken", "gpt4"),
    "o200k_base": ("o200k_base.tiktoken", "o200k"),
}
ENCODINGS = ("r50k_base", "p50k_base", "cl100k_base", "o200k_base")
CORPORA = (
    ("github", DATA_DIR / "github/stride_17_stream_short.txt"),
    ("english", DATA_DIR / "english/parquet_0_stream_short.txt"),
    ("chinese", DATA_DIR / "chinese/parquet_0_stream_short.txt"),
)


def inputs() -> tuple[tuple[str, Path], ...]:
    override = os.environ.get("INPUT")
    return (("custom", Path(override)),) if override else CORPORA


def require_profiled_checkout() -> None:
    cargo_toml = (GIGATOKEN_REPO / "Cargo.toml").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        cargo_toml.partition("[package]")[2].partition("\n[")[0],
        flags=re.MULTILINE,
    )
    actual = match.group(1) if match is not None else None
    if actual != GIGATOKEN_VERSION:
        raise RuntimeError(
            f"the profiled checkout must be based on Gigatoken "
            f"{GIGATOKEN_VERSION}; {GIGATOKEN_REPO} reports {actual!r}"
        )
    based_on_release = subprocess.run(
        (
            "git",
            "-C",
            str(GIGATOKEN_REPO),
            "merge-base",
            "--is-ancestor",
            GIGATOKEN_BASE_COMMIT,
            "HEAD",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if based_on_release.returncode != 0:
        raise RuntimeError(
            f"the profiled checkout must contain Gigatoken "
            f"{GIGATOKEN_VERSION} commit {GIGATOKEN_BASE_COMMIT[:7]}"
        )


def measure(
    name: str,
    corpus: str,
    boundary: str,
    make_tokenizer: Callable[[], object],
    encode: Callable[[object], tuple[object, dict[str, object]]],
) -> None:
    print(f"[ttft] gigatoken/{name}/{corpus}/{boundary}", flush=True)
    profiles = []
    for run in range(WARMUP + REPS):
        tokenizer = make_tokenizer()
        backend = tokenizer.backend
        if not hasattr(backend, "encode_profiled") or not hasattr(
            backend, "encode_files_profiled"
        ):
            raise SystemExit(f"build the profiled checkout first: {GIGATOKEN_REPO}")
        _, profile = encode(tokenizer)
        if run >= WARMUP:
            profiles.append(profile)

    ttfts = [profile["ttft_seconds"] for profile in profiles]
    if any(value is None for value in ttfts):
        raise RuntimeError(f"{name}: missing TTFT for nonempty input")
    ttft_median, ttft_p25, ttft_p75 = median_quartiles(ttfts)
    elapsed = [profile["elapsed_seconds"] for profile in profiles]
    elapsed_median, _, _ = median_quartiles(elapsed)
    print(
        f"{name:<12} {corpus:<9} {boundary:<12} "
        f"{profiles[0]['total_tokens']:>12} "
        f"{ttft_median * 1e3:>16.3f} "
        f"{ttft_p25 * 1e3:>13.3f} "
        f"{ttft_p75 * 1e3:>13.3f} "
        f"{elapsed_median * 1e3:>18.3f}"
    )


def main() -> None:
    if WARMUP < 0 or REPS < 1:
        raise SystemExit("WARMUP must be >= 0 and REPS must be >= 1")

    require_profiled_checkout()
    sys.path.insert(0, str(GIGATOKEN_REPO))
    import gigatoken as gt
    import tiktoken
    from huggingface_hub import hf_hub_download

    cache_root = os.environ.get(
        "TIKTOKEN_CACHE_DIR",
        os.environ.get(
            "DATA_GYM_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "data-gym-cache"),
        ),
    )
    cache_dir = Path(cache_root)
    corpora = inputs()
    for _, path in corpora:
        if not path.is_file():
            raise FileNotFoundError(path)

    print(
        f"gigatoken={GIGATOKEN_VERSION} warmup={WARMUP} reps={REPS}"
    )
    print(
        f"{'encoding':<12} {'corpus':<9} {'boundary':<12} {'tokens':>12} "
        f"{'median_ttft_ms':>16} {'ttft_p25_ms':>13} {'ttft_p75_ms':>13} "
        f"{'median_elapsed_ms':>18}"
    )
    for name in ENCODINGS:
        reference = tiktoken.get_encoding(name)
        if name == "p50k_base":
            # The raw-rank loader requires dense IDs; p50k reserves ID 50256.
            tokenizer_json = hf_hub_download(
                "Xenova/text-davinci-003",
                "tokenizer.json",
                revision="898195e24794cd1710ea3d0d99668208d1fe2ec9",
            )
            make_tokenizer = lambda: gt.Tokenizer(tokenizer_json)
            for corpus, path in corpora:
                text = path.read_bytes().decode("utf-8")
                measure(
                    name,
                    corpus,
                    "resident_str",
                    make_tokenizer,
                    lambda tokenizer: tokenizer.encode_profiled(text),
                )
                measure(
                    name,
                    corpus,
                    "file",
                    make_tokenizer,
                    lambda tokenizer: tokenizer.encode_files_profiled(
                        path, parallel=False
                    ),
                )
            continue

        filename, scheme = MODELS[name]
        model_path = cache_dir / hashlib.sha1(
            (BASE_URL + filename).encode()
        ).hexdigest()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        def make_tokenizer() -> object:
            return gt.Tokenizer.from_tiktoken(
                model_path,
                pretokenizer=scheme,
                special_tokens=dict(reference._special_tokens),
            )

        for corpus, path in corpora:
            text = path.read_bytes().decode("utf-8")
            measure(
                name,
                corpus,
                "resident_str",
                make_tokenizer,
                lambda tokenizer: tokenizer.encode_profiled(text),
            )
            measure(
                name,
                corpus,
                "file",
                make_tokenizer,
                lambda tokenizer: tokenizer.encode_files_profiled(
                    path, parallel=False
                ),
            )


if __name__ == "__main__":
    main()
