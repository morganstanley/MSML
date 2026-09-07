"""Baseline training script for PleIAs SYNTH speedruns.

Starting point for experiments. Designed to work with `harness/runner.py`.

Key properties:
- Correct val_bpb (from loss_nats via bytes_per_token)
- Deterministic cached dataset build via `harness/data_prep.py`
- Deterministic seeding (model init + data sampling)
- No train/val leakage (data_prep splits on row boundaries)
- No weight updates during compile warm-up (fair time budget)
- Emits machine-parseable log lines:
  - HARNESS_PARAM_COUNT {...}
  - TRAINING_START {...}
  - METRIC {...}
  - TRAINING_END {...}

Default tokenizer is "byte" so bytes_per_token is ~1 and BPB==loss_bits.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure repo root is on sys.path when executed as a script
_repo_root = _Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import json
import math
import os
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.config import ExperimentConfig
from harness.data_prep import MemmapTokenDataset, prepare_dataset
from harness.metrics import MetricsTracker, compute_bpb, count_parameters


# ------------------------ Model (LLaMA-ish) ------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


def precompute_rope_freqs(dim: int, max_pos: int, base: float = 10000.0, device=None):
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(max_pos, device=device).float()
    freqs = torch.einsum('i,j->ij', t, inv_freq)  # (max_pos, half)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: (B, H, T, D)
    B, H, T, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos[:T].view(1, 1, T, half)
    s = sin[:T].view(1, 1, T, half)
    y1 = x1 * c - x2 * s
    y2 = x1 * s + x2 * c
    return torch.cat([y1, y2], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, rope_base: float = 10000.0, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = attn_dropout

        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

    def _maybe_init_rope(self, device):
        if self.rope_cos.numel() == 0 or self.rope_cos.device != device:
            cos, sin = precompute_rope_freqs(self.d_head, self.max_seq_len, base=self.rope_base, device=device)
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(self, x):
        B, T, C = x.shape
        self._maybe_init_rope(x.device)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, rope_base: float, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len=max_seq_len, rope_base=rope_base)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
        self.dropout = dropout

    def forward(self, x):
        x = x + F.dropout(self.attn(self.norm1(x)), p=self.dropout, training=self.training)
        x = x + F.dropout(self.mlp(self.norm2(x)), p=self.dropout, training=self.training)
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        block_size: int,
        rope_base: float,
        dropout: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layers = n_layers

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                Block(d_model, n_heads, d_ff, max_seq_len=block_size, rope_base=rope_base, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # tie
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, CausalSelfAttention):
            torch.nn.init.normal_(module.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))
        if isinstance(module, SwiGLU):
            torch.nn.init.normal_(module.w2.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ------------------------ Helpers ------------------------

def _get_dtype(name: str):
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def build_model(cfg: ExperimentConfig, vocab_size_override: Optional[int] = None) -> nn.Module:
    mc = cfg.model
    vocab_size = int(vocab_size_override) if vocab_size_override is not None else int(mc.vocab_size)
    d_ff = int(round(mc.ffn_mult * mc.n_embd))
    return TransformerLM(
        vocab_size=vocab_size,
        d_model=mc.n_embd,
        n_layers=mc.n_layer,
        n_heads=mc.n_head,
        d_ff=d_ff,
        block_size=cfg.data.block_size,
        rope_base=mc.rope_base,
        dropout=mc.dropout,
    )


def _set_global_seeds(seed: int, *, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # This can hurt performance; leave off by default.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def estimate_loss(
    model,
    val_ds: MemmapTokenDataset,
    *,
    batch_size: int,
    eval_batches: int,
    device: str,
    dtype,
    bytes_per_token_val: float,
    val_ix: np.ndarray,
):
    """Deterministic evaluation.

    val_ix: int array of shape (eval_batches, batch_size) with fixed start indices.
    """
    model.eval()
    losses = []
    for b in range(eval_batches):
        x_np, y_np = val_ds.get_batch(batch_size, ix=val_ix[b])
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        with torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=dtype,
            enabled=(device.startswith("cuda") and dtype != torch.float32),
        ):
            _, loss = model(x, y)
        losses.append(loss.item())
    loss_nats = float(np.mean(losses))
    # IMPORTANT: BPB denominator must be computed on the validation split.
    val_bpb = compute_bpb(loss_nats, bytes_per_token_val)
    model.train()
    return loss_nats, val_bpb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--time_limit_seconds", type=int, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--train_shard_end", type=int, default=None)
    ap.add_argument("--max_rows_per_shard", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--n_head", type=int, default=None)
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--ffn_mult", type=float, default=None)
    ap.add_argument("--log_every", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--eval_batches", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig()
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.time_limit_seconds is not None:
        cfg.train.time_limit_seconds = args.time_limit_seconds
    if args.compile:
        cfg.train.compile = True
    if args.no_compile:
        cfg.train.compile = False
    if args.train_shard_end is not None:
        cfg.data.shard_end = int(args.train_shard_end)
    if args.max_rows_per_shard is not None:
        cfg.data.max_rows_per_shard = int(args.max_rows_per_shard)

    if args.device is not None:
        cfg.train.device = str(args.device)
    if args.dtype is not None:
        cfg.train.dtype = str(args.dtype)
    if args.block_size is not None:
        cfg.data.block_size = int(args.block_size)
    if args.batch_size is not None:
        cfg.data.batch_size = int(args.batch_size)
    if args.n_layer is not None:
        cfg.model.n_layer = int(args.n_layer)
    if args.n_head is not None:
        cfg.model.n_head = int(args.n_head)
    if args.n_embd is not None:
        cfg.model.n_embd = int(args.n_embd)
    if args.ffn_mult is not None:
        cfg.model.ffn_mult = float(args.ffn_mult)
    if args.log_every is not None:
        cfg.train.log_every = int(args.log_every)
    if args.eval_every is not None:
        cfg.train.eval_every = int(args.eval_every)
    if args.eval_batches is not None:
        cfg.train.eval_batches = int(args.eval_batches)
    if args.lr is not None:
        cfg.optim.lr = float(args.lr)

    # Deterministic run-to-run behavior.
    _set_global_seeds(int(cfg.data.seed), deterministic=False)

    out_dir = Path(os.environ.get("HARNESS_OUT_DIR", cfg.out_dir)) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count-only mode for runner (must be fast: do NOT touch dataset)
    if os.environ.get("HARNESS_COUNT_ONLY", "0") == "1":
        # Ensure vocab_size is correct even for non-byte tokenizers.
        vocab_size = int(cfg.model.vocab_size)
        if cfg.data.tokenizer == "byte":
            vocab_size = 256
        elif cfg.data.tokenizer == "gpt2":
            from transformers import GPT2TokenizerFast

            vocab_size = int(GPT2TokenizerFast.from_pretrained("gpt2").vocab_size)

        model = build_model(cfg, vocab_size_override=vocab_size)
        param_count, param_str = count_parameters(model)
        print(
            "HARNESS_PARAM_COUNT "
            + json.dumps(
                {
                    "param_count": param_count,
                    "param_str": param_str,
                    "vocab_size": vocab_size,
                    "block_size": cfg.data.block_size,
                    "model": asdict(cfg.model),
                },
                sort_keys=True,
            )
        )
        return

    # Prepare data (startup excluded from budget)
    meta = prepare_dataset(
        data_dir=cfg.data.data_dir,
        shard_start=cfg.data.shard_start,
        shard_end=cfg.data.shard_end,
        max_rows_per_shard=cfg.data.max_rows_per_shard,
        seed=cfg.data.seed,
        language_filter=cfg.data.language_filter,
        include_query=cfg.data.include_query,
        include_reasoning=cfg.data.include_reasoning,
        tokenizer=cfg.data.tokenizer,
        block_size=cfg.data.block_size,
        val_fraction_tokens=cfg.data.val_fraction_tokens,
        cache_dir=cfg.data.cache_dir,
        cache_name=cfg.data.cache_name,
    )

    vocab_size = int(meta["vocab_size"])
    # bytes/token ratios are computed on the exact sampled text and can differ between splits
    bytes_per_token = float(meta["bytes_per_token"])
    bytes_per_token_train = float(meta.get("bytes_per_token_train", bytes_per_token))
    bytes_per_token_val = float(meta.get("bytes_per_token_val", bytes_per_token))

    # Build model
    device = cfg.train.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
        cfg.train.device = 'cpu'
    dtype = _get_dtype(cfg.train.dtype)

    model = build_model(cfg, vocab_size_override=vocab_size)
    model.to(device)

    param_count, param_str = count_parameters(model)

    # Optim
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
        fused=True if device.startswith("cuda") else False,
    )

    # Data
    train_ds = MemmapTokenDataset(meta, split="train")
    val_ds = MemmapTokenDataset(meta, split="val")

    # Separate RNGs so evaluation doesn't affect training sampling.
    rng_train = np.random.default_rng(int(cfg.data.seed) + 0)
    rng_val = np.random.default_rng(int(cfg.data.seed) + 1337)

    # Precompute deterministic validation windows.
    eval_batches = int(cfg.train.eval_batches)
    val_ix = np.stack([val_ds.sample_ix(cfg.data.batch_size, rng=rng_val) for _ in range(eval_batches)], axis=0)

    # Metrics logger
    jsonl_path = out_dir / "metrics.jsonl"
    tracker = MetricsTracker(
        run_name=cfg.run_name,
        out_path_jsonl=jsonl_path,
        # Use validation split denominator for any derived val_bpb calculations.
        bytes_per_token=bytes_per_token_val,
        param_count=param_count,
        gpu_peak_flops=cfg.train.gpu_peak_flops,
    )

    state = {"stop": False, "reason": None}

    def _handle_sigint(signum, frame):
        state["stop"] = True
        state["reason"] = f"signal_{signum}"

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    # Optional compile (excluded from timer)
    if cfg.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=cfg.train.compile_mode)

    # Warm-up (excluded from timer) - must NOT update weights.
    warmup_steps = int(cfg.train.compile_warmup_steps) if cfg.train.compile else 0
    if warmup_steps > 0:
        model.train()
        for _ in range(warmup_steps):
            x_np, y_np = train_ds.get_batch(cfg.data.batch_size, rng=rng_train)
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=dtype,
                enabled=(device.startswith("cuda") and dtype != torch.float32),
            ):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.zero_grad(set_to_none=True)  # clear grads; NO opt.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    # Training start
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    print(
        "TRAINING_START "
        + json.dumps(
            {
                "t0": t0,
                "run_name": cfg.run_name,
                "param_count": param_count,
                "param_str": param_str,
                "bytes_per_token": bytes_per_token,
                "bytes_per_token_train": bytes_per_token_train,
                "bytes_per_token_val": bytes_per_token_val,
                "vocab_size": vocab_size,
            },
            sort_keys=True,
        )
    )

    # Budget (runner also enforces externally; internal safety stop)
    max_seconds = float(os.environ.get("HARNESS_TIME_LIMIT_SECONDS", cfg.train.time_limit_seconds))

    step = 0
    tokens_seen = 0

    # For throughput measurement, use perf_counter and explicit CUDA sync around timestamps.
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    last_log_t = time.perf_counter()
    last_log_tokens = 0

    def lr_at(step_: int) -> float:
        if step_ < cfg.optim.warmup_steps:
            return cfg.optim.lr * (step_ + 1) / max(1, cfg.optim.warmup_steps)
        decay_steps = 10_000
        s = min(1.0, (step_ - cfg.optim.warmup_steps) / max(1, decay_steps))
        coef = 0.5 * (1.0 + math.cos(math.pi * s))
        return cfg.optim.min_lr + coef * (cfg.optim.lr - cfg.optim.min_lr)

    try:
        model.train()

        # Eval at step 0 to avoid the edge case where a short time limit produces no evals.
        # This is included in the training budget (timer already started) by design.
        if cfg.train.eval_every > 0:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            eval_t0 = time.perf_counter()
            val_loss, val_bpb = estimate_loss(
                model,
                val_ds,
                batch_size=cfg.data.batch_size,
                eval_batches=eval_batches,
                device=device,
                dtype=dtype,
                bytes_per_token_val=bytes_per_token_val,
                val_ix=val_ix,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            eval_dt = time.perf_counter() - eval_t0

            now = time.perf_counter()
            peak_mem_gb = None
            if device.startswith("cuda"):
                peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

            tracker.log(
                t=now - t0,
                step=step,
                train_loss=None,
                val_loss=float(val_loss),
                val_bpb=float(val_bpb),
                lr=float(lr_at(step)),
                tokens_seen=int(tokens_seen),
                tokens_per_sec=None,
                eval_seconds=float(eval_dt),
                peak_memory_gb=peak_mem_gb,
                param_count=int(param_count),
            )

        while True:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            tnow = time.perf_counter()
            wall = tnow - t0
            if wall >= max_seconds:
                state["stop"] = True
                state["reason"] = "time_budget"
            if state["stop"]:
                break

            # batch
            x_np, y_np = train_ds.get_batch(cfg.data.batch_size, rng=rng_train)
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)

            lr = lr_at(step)
            for pg in opt.param_groups:
                pg["lr"] = lr

            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=dtype,
                enabled=(device.startswith("cuda") and dtype != torch.float32),
            ):
                _, loss = model(x, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.optim.grad_clip is not None and cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            opt.step()

            step += 1
            tokens_in_step = int(cfg.data.batch_size * cfg.data.block_size)
            tokens_seen += tokens_in_step

            # periodic eval
            if step % cfg.train.eval_every == 0:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                eval_t0 = time.perf_counter()
                val_loss, val_bpb = estimate_loss(
                    model,
                    val_ds,
                    batch_size=cfg.data.batch_size,
                    eval_batches=eval_batches,
                    device=device,
                    dtype=dtype,
                    bytes_per_token_val=bytes_per_token_val,
                    val_ix=val_ix,
                )
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                eval_dt = time.perf_counter() - eval_t0

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                now = time.perf_counter()
                dt = max(1e-9, now - last_log_t)
                tps = (tokens_seen - last_log_tokens) / dt
                last_log_t = now
                last_log_tokens = tokens_seen

                peak_mem_gb = None
                if device.startswith("cuda"):
                    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

                tracker.log(
                    t=now - t0,
                    step=step,
                    train_loss=float(loss.item()),
                    val_loss=float(val_loss),
                    val_bpb=float(val_bpb),
                    lr=float(lr),
                    tokens_seen=int(tokens_seen),
                    tokens_per_sec=float(tps),
                    eval_seconds=float(eval_dt),
                    peak_memory_gb=peak_mem_gb,
                    param_count=int(param_count),
                )

            if step % cfg.train.log_every == 0 and step % cfg.train.eval_every != 0:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                now = time.perf_counter()
                dt = max(1e-9, now - last_log_t)
                tps = (tokens_seen - last_log_tokens) / dt
                last_log_t = now
                last_log_tokens = tokens_seen

                peak_mem_gb = None
                if device.startswith("cuda"):
                    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

                tracker.log(
                    t=now - t0,
                    step=step,
                    train_loss=float(loss.item()),
                    lr=float(lr),
                    tokens_seen=int(tokens_seen),
                    tokens_per_sec=float(tps),
                    peak_memory_gb=peak_mem_gb,
                    param_count=int(param_count),
                )

    except Exception as e:
        state["stop"] = True
        state["reason"] = f"exception: {type(e).__name__}: {e}"
        raise
    finally:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        end_meta = {
            "t1": t1,
            "wall_clock_seconds": t1 - t0,
            "reason": state["reason"],
            "best_val_bpb": tracker.best_val_bpb,
        }
        print("TRAINING_END " + json.dumps(end_meta, sort_keys=True))

        summary = {
            "run_name": cfg.run_name,
            "cfg": cfg.to_dict(),
            "data_meta": meta,
            "param_count": param_count,
            "bytes_per_token": bytes_per_token,
            "best_val_bpb": tracker.best_val_bpb,
            "end": end_meta,
            "log_jsonl": str(jsonl_path),
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
