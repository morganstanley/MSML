"""PleIAs SYNTH data preparation.

Optimized for short time budgets:
- Uses ParquetFile.iter_batches to stop early per shard instead of reading the full shard.
- Caches tokenized results to a .bin for fast reload.

Tokenizers:
- "byte": UTF-8 bytes -> tokens in [0,255], bytes_per_token ~= 1.0
- "gpt2": GPT-2 BPE via transformers

Correctness notes
-----------------
Train/val split must not leak content. A naive token-level split on a single
concatenated token stream can split a *row* across the boundary, causing some of
that row's text to appear in train and the remainder in val.

We therefore split on row boundaries: we collect per-row token arrays, then
assign whole rows to train or val.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _shard_paths(data_dir: str, shard_start: int, shard_end: int) -> List[Path]:
    return [Path(data_dir) / f"synth_{i:03d}.parquet" for i in range(shard_start, shard_end + 1)]


def _build_text(
    q: Optional[str],
    r: Optional[str],
    a: Optional[str],
    *,
    include_query: bool,
    include_reasoning: bool,
) -> str:
    q = q or ""
    r = r or ""
    a = a or ""
    if include_query and include_reasoning:
        return f"Q: {q}\nR: {r}\nA: {a}\n"
    if include_query:
        return f"Q: {q}\nA: {a}\n"
    return a + "\n"


def _get_tokenizer(tokenizer_name: str):
    if tokenizer_name == "byte":
        return None
    if tokenizer_name == "gpt2":
        from transformers import GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        return tok
    raise ValueError(f"Unknown tokenizer: {tokenizer_name}")


def _tokenize(text: str, *, tokenizer_name: str, tok) -> np.ndarray:
    if tokenizer_name == "byte":
        b = text.encode("utf-8", errors="ignore")
        return np.frombuffer(b, dtype=np.uint8).astype(np.uint16)
    ids = tok.encode(text)
    return np.array(ids, dtype=np.int32)


def _stable_cache_key(cfg_dict: Dict) -> str:
    blob = json.dumps(cfg_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def prepare_dataset(
    *,
    data_dir: str,
    shard_start: int,
    shard_end: int,
    max_rows_per_shard: int,
    seed: int,
    language_filter: Optional[str],
    include_query: bool,
    include_reasoning: bool,
    tokenizer: str,
    block_size: int,
    val_fraction_tokens: float,
    cache_dir: str,
    cache_name: str,
) -> Dict:

    cfg = dict(
        data_dir=data_dir,
        shard_start=shard_start,
        shard_end=shard_end,
        max_rows_per_shard=max_rows_per_shard,
        seed=seed,
        language_filter=language_filter,
        include_query=include_query,
        include_reasoning=include_reasoning,
        tokenizer=tokenizer,
        block_size=block_size,
        val_fraction_tokens=val_fraction_tokens,
        split_version=2,
    )

    key = _stable_cache_key(cfg)
    cache_dir_p = Path(cache_dir)
    cache_dir_p.mkdir(parents=True, exist_ok=True)
    bin_path = cache_dir_p / f"{cache_name}_{key}.bin"
    meta_path = cache_dir_p / f"{cache_name}_{key}.json"

    if bin_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["bin_path"] = str(bin_path)
        meta["meta_path"] = str(meta_path)
        return meta

    # Optional: fast synthetic dataset for unit/integration tests (avoids parquet IO).
    if os.environ.get('HARNESS_SYNTHETIC_DATA', '0') == '1':
        rng = np.random.default_rng(int(seed))
        vocab_size = 256 if tokenizer == 'byte' else 50257
        # Generate a contiguous token stream; bytes_per_token is exact for byte tokenizer.
        n_total = int((block_size + 1) * 200)
        if tokenizer == 'byte':
            toks = rng.integers(0, vocab_size, size=(n_total,), dtype=np.uint16)
            dtype = np.uint16
            bytes_per_token = 1.0
        else:
            toks = rng.integers(0, vocab_size, size=(n_total,), dtype=np.int32)
            dtype = np.int32
            bytes_per_token = 4.0  # arbitrary; only used for BPB conversion

        n_val = max(block_size + 1, int(n_total * float(val_fraction_tokens)))
        n_train = n_total - n_val
        toks.astype(dtype, copy=False).tofile(bin_path)
        meta = {
            'cfg': cfg,
            'key': key,
            'bin_path': str(bin_path),
            'meta_path': str(meta_path),
            'dtype': str(np.dtype(dtype)),
            'n_total': n_total,
            'n_train': n_train,
            'n_val': n_val,
            'block_size': block_size,
            'bytes_per_token': bytes_per_token,
            'bytes_per_token_train': bytes_per_token,
            'bytes_per_token_val': bytes_per_token,
            'num_rows_used': 0,
            'tokenizer': tokenizer,
            'vocab_size': vocab_size,
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        return meta

    shard_paths = _shard_paths(data_dir, shard_start, shard_end)
    for p in shard_paths:
        if not p.exists():
            raise FileNotFoundError(str(p))

    rng = random.Random(seed)
    tok = _get_tokenizer(tokenizer)

    # Build as list-of-rows for leakage-free splitting.
    tokens_rows: List[np.ndarray] = []
    bytes_rows: List[int] = []
    total_bytes = 0
    total_tokens = 0
    num_rows_used = 0

    for shard in shard_paths:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(shard)
        cols = ["synthetic_answer"]
        if include_query:
            cols.append("query")
        if include_reasoning:
            cols.append("synthetic_reasoning")
        if language_filter is not None:
            cols.append("language")

        rows_taken = 0
        for batch in pf.iter_batches(batch_size=2048, columns=cols):
            batch = batch.to_pydict()
            n = len(batch["synthetic_answer"])
            idxs = list(range(n))
            rng.shuffle(idxs)  # deterministic shuffle per batch

            for i in idxs:
                if rows_taken >= max_rows_per_shard:
                    break
                if language_filter is not None:
                    if batch.get("language", [None])[i] != language_filter:
                        continue

                text = _build_text(
                    batch.get("query", [None])[i] if include_query else None,
                    batch.get("synthetic_reasoning", [None])[i] if include_reasoning else None,
                    batch["synthetic_answer"][i],
                    include_query=include_query,
                    include_reasoning=include_reasoning,
                )
                if not text:
                    continue

                arr = _tokenize(text, tokenizer_name=tokenizer, tok=tok)
                if arr.size < block_size + 1:
                    continue

                b = text.encode("utf-8", errors="ignore")
                tokens_rows.append(arr)
                bytes_rows.append(len(b))

                num_rows_used += 1
                rows_taken += 1
                total_bytes += len(b)
                total_tokens += int(arr.size)

            if rows_taken >= max_rows_per_shard:
                break

    if not tokens_rows:
        raise RuntimeError("No tokens produced; check filters/sampling.")

    # ---------------- Row-boundary train/val split ----------------
    n_total = int(sum(int(a.size) for a in tokens_rows))
    n_val_target = max(block_size + 1, int(n_total * float(val_fraction_tokens)))

    # Deterministic: take rows from the end until we reach the token target.
    val_rows: List[np.ndarray] = []
    train_rows: List[np.ndarray] = []
    val_bytes = 0
    val_tokens = 0

    for arr, b in zip(reversed(tokens_rows), reversed(bytes_rows)):
        if val_tokens < n_val_target:
            val_rows.append(arr)
            val_bytes += int(b)
            val_tokens += int(arr.size)
        else:
            train_rows.append(arr)

    val_rows.reverse()
    train_rows.reverse()

    n_val = int(sum(int(a.size) for a in val_rows))
    n_train = int(sum(int(a.size) for a in train_rows))

    if n_train < block_size + 1 or n_val < block_size + 1:
        raise RuntimeError(
            f"Not enough tokens after row-split: total={n_total}, train={n_train}, val={n_val}, block_size={block_size}"
        )

    bytes_per_token = float(total_bytes) / float(total_tokens)
    bytes_per_token_val = float(val_bytes) / float(max(1, val_tokens))
    bytes_per_token_train = float(total_bytes - val_bytes) / float(max(1, total_tokens - val_tokens))

    if tokenizer == "byte":
        dtype = np.uint16
    else:
        dtype = np.int32

    # Concatenate train then val to enable simple contiguous splits in MemmapTokenDataset.
    all_tokens = np.concatenate([*train_rows, *val_rows], axis=0).astype(dtype, copy=False)
    all_tokens.tofile(bin_path)

    meta = {
        "cfg": cfg,
        "key": key,
        "bin_path": str(bin_path),
        "meta_path": str(meta_path),
        "dtype": str(all_tokens.dtype),
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "block_size": block_size,
        "bytes_per_token": bytes_per_token,
        "bytes_per_token_train": bytes_per_token_train,
        "bytes_per_token_val": bytes_per_token_val,
        "num_rows_used": num_rows_used,
        "tokenizer": tokenizer,
        "vocab_size": (256 if tokenizer == "byte" else int(tok.vocab_size)),
    }

    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta


class MemmapTokenDataset:
    def __init__(self, meta: Dict, split: str):
        assert split in {"train", "val"}
        self.meta = meta
        self.split = split
        self.bin_path = Path(meta["bin_path"])
        self.block_size = int(meta["block_size"])
        self.n_train = int(meta["n_train"])
        self.n_val = int(meta["n_val"])
        self.dtype = np.dtype(meta["dtype"])
        self._data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")

        if split == "train":
            self.start = 0
            self.end = self.n_train
        else:
            self.start = self.n_train
            self.end = self.n_train + self.n_val

    def sample_ix(self, batch_size: int, *, rng: np.random.Generator) -> np.ndarray:
        max_start = self.end - (self.block_size + 1)
        if max_start <= self.start:
            raise RuntimeError("Split too small for block_size")
        return rng.integers(low=self.start, high=max_start, size=(batch_size,))

    def get_batch(
        self,
        batch_size: int,
        *,
        rng: Optional[np.random.Generator] = None,
        ix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if ix is None:
            if rng is None:
                raise ValueError("Must provide either rng or ix")
            ix = self.sample_ix(batch_size, rng=rng)
        else:
            ix = np.asarray(ix)
            if ix.shape != (batch_size,):
                raise ValueError(f"ix must have shape ({batch_size},), got {ix.shape}")

        x = np.stack([self._data[i : i + self.block_size].astype(np.int64) for i in ix])
        y = np.stack([self._data[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix])
        return x, y
