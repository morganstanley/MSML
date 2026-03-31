#!/usr/bin/env python3
"""Download datasets used in the AlphaLab paper.

Usage:
    python data/download_datasets.py              # Download all datasets
    python data/download_datasets.py --dataset traffic
    python data/download_datasets.py --dataset llm_speedrun
    python data/download_datasets.py --dataset kernelbench
    python data/download_datasets.py --dataset exchange  # Already included, just verifies

Datasets are downloaded to data/<dataset_name>/.
The exchange_rates.csv dataset is synthetic and already included — use
`python data/generate_synthetic.py` to regenerate it.
"""

import argparse
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent


def download_traffic():
    """Download the Traffic dataset (GluonTS format).

    Source: https://huggingface.co/datasets/amazon/monash_tsf
    862 road sensors, hourly occupancy, ~17 months.
    """
    dest = DATA_DIR / "traffic"
    if dest.exists() and any(dest.iterdir()):
        print(f"Traffic dataset already exists at {dest}")
        return

    print("Downloading Traffic dataset from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="tml-epfl/traffic",
            repo_type="dataset",
            local_dir=str(dest),
        )
        print(f"Traffic dataset downloaded to {dest}")
    except ImportError:
        print(
            "huggingface_hub not installed. Install it:\n"
            "  pip install huggingface_hub\n"
            "Or download manually from: https://huggingface.co/datasets/tml-epfl/traffic"
        )
        sys.exit(1)


def download_llm_speedrun():
    """Download the PleIAs SYNTH corpus for LLM speedrun.

    Source: https://huggingface.co/datasets/PleIAs/common_corpus
    500 parquet shards, ~32B tokens total.
    """
    dest = DATA_DIR / "pleias-synth"
    if dest.exists() and any(dest.iterdir()):
        print(f"PleIAs SYNTH dataset already exists at {dest}")
        return

    print("Downloading PleIAs SYNTH corpus from HuggingFace...")
    print("NOTE: This is a large dataset (~50GB). Only a subset of shards is needed.")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="PleIAs/common_corpus",
            repo_type="dataset",
            local_dir=str(dest),
            allow_patterns=["synth_*.parquet"],
        )
        print(f"PleIAs SYNTH downloaded to {dest}")
    except ImportError:
        print(
            "huggingface_hub not installed. Install it:\n"
            "  pip install huggingface_hub\n"
            "Or download manually from: https://huggingface.co/datasets/PleIAs/common_corpus"
        )
        sys.exit(1)


def download_kernelbench():
    """Download KernelBench CUDA optimization benchmark.

    Source: https://github.com/ScalingIntelligence/KernelBench
    """
    dest = DATA_DIR / "kernelbench"
    if dest.exists() and any(dest.iterdir()):
        print(f"KernelBench dataset already exists at {dest}")
        return

    print("Cloning KernelBench from GitHub...")
    ret = os.system(f"git clone https://github.com/ScalingIntelligence/KernelBench.git {dest}")
    if ret != 0:
        print("Failed to clone KernelBench. Clone manually:")
        print(f"  git clone https://github.com/ScalingIntelligence/KernelBench.git {dest}")
        sys.exit(1)
    print(f"KernelBench downloaded to {dest}")


def verify_exchange():
    """Verify the exchange rates dataset exists (synthetic, already included)."""
    path = DATA_DIR / "exchange_rates.csv"
    if path.exists():
        print(f"Exchange rates dataset exists at {path}")
    else:
        print("Exchange rates dataset not found. Generate it:")
        print("  python data/generate_synthetic.py")


DATASETS = {
    "traffic": download_traffic,
    "llm_speedrun": download_llm_speedrun,
    "kernelbench": download_kernelbench,
    "exchange": verify_exchange,
}


def main():
    parser = argparse.ArgumentParser(description="Download AlphaLab paper datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default=None,
        help="Specific dataset to download (default: all)",
    )
    args = parser.parse_args()

    if args.dataset:
        DATASETS[args.dataset]()
    else:
        for name, fn in DATASETS.items():
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            fn()

    print("\nDone.")


if __name__ == "__main__":
    main()
