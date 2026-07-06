"""Default configuration for the Alpha Lab experiment harness."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple


@dataclass
class DataConfig:
    data_dir: str = "/path/to/pleias-synth"

    shard_start: int = 1
    shard_end: int = 8  # inclusive
    max_rows_per_shard: int = 120_000
    seed: int = 52

    include_query: bool = True
    include_reasoning: bool = False
    language_filter: Optional[str] = "en"

    tokenizer: str = "byte"  # "byte" or "gpt2"

    block_size: int = 1024
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True

    val_fraction_tokens: float = 0.01

    cache_dir: str = "harness/cache"
    cache_name: str = "pleias_synth_sample"


@dataclass
class ModelConfig:
    vocab_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    ffn_mult: float = 3.0
    rope_base: float = 10_000.0
    dropout: float = 0.0


@dataclass
class OptimConfig:
    lr: float = 5e-4
    min_lr: float = 5e-5
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 80


@dataclass
class TrainConfig:
    time_limit_seconds: int = 1200
    param_cap: int = 100_000_000

    log_every: int = 10
    eval_every: int = 50
    eval_batches: int = 50

    device: str = "cuda"
    dtype: str = "bf16"

    compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_warmup_steps: int = 5

    gpu_peak_flops: Optional[float] = None


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    run_name: str = "baseline"
    out_dir: str = "harness/results"

    def to_dict(self):
        return {
            "run_name": self.run_name,
            "out_dir": self.out_dir,
            "data": asdict(self.data),
            "model": asdict(self.model),
            "optim": asdict(self.optim),
            "train": asdict(self.train),
        }
