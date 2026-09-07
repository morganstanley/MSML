# LLM Pretraining Quality Optimization Domain Knowledge

## Primary Metric: val_bpb (Validation Bits Per Byte)

- BPB = cross-entropy loss (nats) * log2(e) / bytes_per_token
- bytes_per_token = average number of UTF-8 bytes per token for the tokenizer
- BPB is vocabulary-size-invariant: you can compare models with different tokenizers
- Lower is better. State-of-art small models (<100M) typically achieve 0.7–0.9 BPB on web text
- Always compute BPB from the raw cross-entropy loss, never from perplexity

## Constraint: <100M Trainable Parameters

Parameter counting formula for a standard transformer:
- Embedding: vocab_size * d_model (+ d_model * max_seq_len if learned positional)
- Per transformer block: 4 * d_model^2 (attention) + 8/3 * d_model^2 (SwiGLU FFN) or 8 * d_model^2 (standard MLP) + 2 * d_model (norms)
- Output head: usually tied with embedding (no extra params)
- Total ≈ vocab_size * d_model + n_layers * (4 + 8) * d_model^2 for standard MLP
- With SwiGLU: FFN is 8/3 * d_model * ffn_dim * 3 gates — typically ~same param count as 4x MLP

Stay under 100M. The parameter cap is enforced before training starts.

## Constraint: 20-Minute Wall-Clock Budget

- Training is killed after 1200 seconds of actual training time
- Compilation/startup time is excluded from the budget
- The best val_bpb seen at any evaluation checkpoint is the final score
- Maximize tokens processed within the time budget — throughput matters
- Evaluate frequently enough that at least 5-10 checkpoints occur within 20 minutes

## Small Model Efficiency (<100M Parameters)

### Width vs Depth at This Scale
- Below ~200M params, wider-shallower models often outperform narrow-deep ones
- Sweet spot around 6-12 layers with 512-768 hidden dim
- Diminishing returns from depth below 200M — each additional layer adds less
- Optimal aspect ratio: d_model / n_layers ≈ 64-128

### Architecture Choices That Matter
- **RMSNorm > LayerNorm**: Cheaper (no mean computation), often better for small models
- **SwiGLU > GELU MLP**: More expressive per parameter, standard in modern LLMs
- **RoPE > Learned positional**: Better length generalization, no extra parameters
- **Grouped Query Attention (GQA)**: Saves KV-cache params, minimal quality loss at small scale
- **Pre-norm > Post-norm**: More stable training, standard in modern architectures
- **Tied embeddings**: Tie input/output embeddings to save parameters for the param budget

### What Works at Small Scale
- Higher learning rates than you'd use for large models (3e-4 to 1e-3)
- Shorter warmup (500-2000 steps)
- Weight decay 0.1 is a reasonable default
- Gradient clipping at 1.0
- bf16 mixed precision (no GradScaler needed, unlike fp16)

## Optimizer Landscape

- **AdamW**: Reliable default. β1=0.9, β2=0.95, eps=1e-8 for LLM training
- **Muon**: Momentum-based optimizer using Newton's method on the orthogonal group. Can converge faster for small models. Worth testing.
- **Sophia**: Second-order optimizer with diagonal Hessian approximation. Can use larger LR.
- **Learning rate schedules**:
  - Cosine decay: standard, well-understood
  - WSD (Warmup-Stable-Decay): warmup → constant → linear decay. Simpler, competitive results.
  - Linear decay: simple baseline

## Training Dynamics

- **Batch size**: Total tokens per step = batch_size * seq_len * grad_accum_steps. Larger effective batch → smoother gradients but fewer parameter updates per wall-clock second.
- **Sequence length**: 512-2048 is typical for small models. Longer sequences capture more context but are slower (attention is O(n^2) without flash attention).
- **Gradient accumulation**: Use to simulate larger batch sizes without OOM. Each accumulation step adds forward+backward time.
- **Total training tokens**: Within 20 minutes you'll process ~1-5B tokens depending on throughput. More tokens = lower loss if the model has capacity.

## Efficiency Optimizations

- **torch.compile**: 20-40% speedup from kernel fusion. Use `mode="reduce-overhead"` for small models.
- **Flash Attention / SDPA**: `F.scaled_dot_product_attention` auto-selects the fastest backend. Always use this.
- **bf16 mixed precision**: Use `torch.autocast('cuda', dtype=torch.bfloat16)`. No GradScaler needed.
- **Data loading**: Memory-mapped datasets, pinned memory, multiple workers, prefetching.
- **Fused optimizers**: `torch.optim.AdamW(fused=True)` for GPU-accelerated optimizer steps.

## Dataset

The training corpus is provided as text shards in a `data/` directory within the workspace. It may be:
- **Synthetic test corpus**: Generated via `data/generate_synth_corpus.py` — structured English-like text with learnable patterns (sentence templates, vocabulary pools, paragraph structure). Good for testing infrastructure. Generate with `python data/generate_synth_corpus.py --size_mb 500 --output_dir data/synth_corpus`.
- **Production corpus** (e.g., Pleias SYNTH, FineWeb-Edu, OpenWebText): Swap in when available by pointing `data_path` in the config to the corpus directory.

Key considerations:
- Standard GPT-2 tokenizer (vocab_size=50257) or custom BPE works well
- The data_prep harness reads `.txt` shard files from the corpus directory
- bytes_per_token must be computed from the actual corpus + tokenizer combination
- No data augmentation needed — focus on architecture and optimizer choices

## Common Architectures at This Scale

- **GPT-2 style**: LayerNorm + GELU MLP + learned positional embeddings. Simple baseline.
- **LLaMA-style**: RMSNorm + SwiGLU + RoPE. Modern standard, typically better BPB.
- **GPT-NeoX style**: Parallel attention + MLP (both computed simultaneously, then added). Can improve throughput.

## What NOT to Waste Time On

- **Data augmentation**: The dataset is large enough for any 20-minute run
- **Regularization (dropout)**: Training is so short that overfitting is not the issue
- **Complex scheduling**: With only 20 minutes, keep the schedule simple
- **Ensemble methods**: Single model only
- **Distillation**: No teacher model available
- **Pruning/quantization**: Focus on training quality, not inference efficiency
