# NanoGPT Training Speed Optimization Domain Knowledge

## Key Metrics
- Wall clock time (seconds): time to reach target validation loss
- Tokens per second: training throughput
- Validation loss: must reach target threshold
- Peak memory (GB): GPU memory usage

## Optimization Hierarchy (highest impact first)
1. Mixed precision training (bf16/fp16) — 2x memory savings, faster compute
2. torch.compile — kernel fusion, reduces Python overhead
3. Flash attention — O(N) memory, faster attention computation
4. Optimized data loading — prefetching, pinned memory, num_workers
5. Gradient accumulation — simulate larger batch sizes
6. Learning rate scheduling — cosine with warmup, higher peak LR
7. Batch size tuning — larger batches with linear LR scaling
8. Weight initialization — proper scaling reduces early training instability

## Common Pitfalls
- Measuring wall clock incorrectly (including compilation time)
- OOM from too-large batch sizes or sequence lengths
- NaN loss from improper mixed precision (need loss scaling)
- Slow data loading becoming the bottleneck
- torch.compile recompilation on dynamic shapes
- Not warming up before timing

## Target Validation Loss
The goal is to reach a specific validation loss threshold as fast as possible.
Lower wall clock time = better. The val_loss must actually reach the threshold
for the run to count.

## Libraries
- PyTorch (torch): core framework
- Flash Attention: pip install flash-attn
- torch.compile: built into PyTorch 2.0+
- Triton: custom GPU kernels in Python
