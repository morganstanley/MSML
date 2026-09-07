import json
from pathlib import Path

import numpy as np
import pytest

from harness.data_prep import MemmapTokenDataset, _tokenize


def _make_meta(tmp_path: Path, tokens: np.ndarray, *, n_train: int, n_val: int, block_size: int = 8):
    assert n_train + n_val == tokens.size
    bin_path = tmp_path / "toy.bin"
    tokens.astype(np.uint16).tofile(bin_path)
    meta = {
        "bin_path": str(bin_path),
        "dtype": "uint16",
        "n_train": int(n_train),
        "n_val": int(n_val),
        "block_size": int(block_size),
    }
    return meta


def test_train_val_split_no_overlap_ranges(tmp_path):
    tokens = np.arange(100, dtype=np.uint16)
    meta = _make_meta(tmp_path, tokens, n_train=60, n_val=40, block_size=8)

    tr = MemmapTokenDataset(meta, split="train")
    va = MemmapTokenDataset(meta, split="val")

    assert tr.end == va.start
    assert tr.start == 0
    assert va.end == 100


def test_batch_shape_and_target_offset(tmp_path):
    tokens = np.arange(50, dtype=np.uint16)
    meta = _make_meta(tmp_path, tokens, n_train=30, n_val=20, block_size=8)
    ds = MemmapTokenDataset(meta, split="train")

    ix = np.array([0, 1, 2], dtype=np.int64)
    x, y = ds.get_batch(batch_size=3, ix=ix)
    assert x.shape == (3, 8)
    assert y.shape == (3, 8)
    assert np.all(y == x + 1)


def test_reproducibility_same_seed_same_batches(tmp_path):
    tokens = np.arange(200, dtype=np.uint16)
    meta = _make_meta(tmp_path, tokens, n_train=120, n_val=80, block_size=16)
    ds = MemmapTokenDataset(meta, split="train")

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    x1, y1 = ds.get_batch(batch_size=4, rng=rng1)
    x2, y2 = ds.get_batch(batch_size=4, rng=rng2)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_bytes_per_token_byte_tokenizer_manual_ratio():
    s = "Hello, 世界\n"
    b = s.encode("utf-8")
    ids = _tokenize(s, tokenizer_name="byte", tok=None)
    assert ids.dtype == np.uint16
    assert len(b) == ids.size
    bytes_per_token = len(b) / ids.size
    assert bytes_per_token == pytest.approx(1.0)
