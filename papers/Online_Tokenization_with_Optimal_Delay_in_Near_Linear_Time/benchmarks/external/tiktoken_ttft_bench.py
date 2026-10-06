#!/usr/bin/env python3
"""Run the opt-in profiled tiktoken checkout across the TTFT matrix."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from tools.benchmark_common import TTFT_RUNS, median_quartiles  # noqa: E402
from tools.config import data_dir, ttft_repos_dir  # noqa: E402


WARMUP = int(os.environ.get("WARMUP", "2"))
REPS = int(os.environ.get("REPS", str(TTFT_RUNS)))
DATA_DIR = Path(data_dir()).expanduser().resolve()
_TIKTOKEN_REPO = os.environ.get("TIKTOKEN_TTFT_REPO")
TIKTOKEN_REPO = Path(
    _TIKTOKEN_REPO
    if _TIKTOKEN_REPO is not None
    else Path(ttft_repos_dir()) / "tiktoken-ttft"
).expanduser().resolve()
ENCODINGS = ("r50k_base", "p50k_base", "cl100k_base", "o200k_base")
CORPORA = (
    ("github", DATA_DIR / "github/stride_17_stream_short.txt"),
    ("english", DATA_DIR / "english/parquet_0_stream_short.txt"),
    ("chinese", DATA_DIR / "chinese/parquet_0_stream_short.txt"),
)


def inputs() -> tuple[tuple[str, Path], ...]:
    override = os.environ.get("INPUT")
    return (("custom", Path(override)),) if override else CORPORA


def main() -> None:
    if WARMUP < 0 or REPS < 1:
        raise SystemExit("WARMUP must be >= 0 and REPS must be >= 1")

    sys.path.insert(0, str(TIKTOKEN_REPO))
    import tiktoken

    probe = tiktoken.get_encoding("r50k_base")
    if not hasattr(probe._core_bpe, "encode_to_tiktoken_buffer_profiled"):
        raise SystemExit(f"build the profiled checkout first: {TIKTOKEN_REPO}")

    corpora = inputs()
    for _, path in corpora:
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"warmup={WARMUP} reps={REPS}")
    print(
        f"{'encoding':<12} {'corpus':<9} {'tokens':>12} {'median_ttft_ms':>16} "
        f"{'ttft_p25_ms':>13} {'ttft_p75_ms':>13} {'median_return_ms':>17}"
    )
    for name in ENCODINGS:
        encoder = probe if name == "r50k_base" else tiktoken.get_encoding(name)
        for corpus, path in corpora:
            print(f"[ttft] tiktoken/{name}/{corpus}", flush=True)
            text = path.read_bytes().decode("utf-8")
            profiles = []
            for run in range(WARMUP + REPS):
                _, profile = encoder.encode_to_numpy_profiled(
                    text,
                    allowed_special="all",
                )
                if run >= WARMUP:
                    profiles.append(profile)

            ttfts = [profile.ttft_seconds for profile in profiles]
            if any(value is None for value in ttfts):
                raise RuntimeError(f"{name}/{corpus}: missing TTFT for nonempty input")
            ttft_median, ttft_p25, ttft_p75 = median_quartiles(ttfts)
            returns = [profile.elapsed_seconds for profile in profiles]
            return_median, _, _ = median_quartiles(returns)
            print(
                f"{name:<12} {corpus:<9} {profiles[0].total_tokens:>12} "
                f"{ttft_median * 1e3:>16.3f} "
                f"{ttft_p25 * 1e3:>13.3f} "
                f"{ttft_p75 * 1e3:>13.3f} "
                f"{return_median * 1e3:>17.3f}"
            )


if __name__ == "__main__":
    main()
