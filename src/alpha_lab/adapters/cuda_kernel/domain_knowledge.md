# KernelBench CUDA Kernel Generation — Domain Knowledge

## Benchmark Overview

KernelBench is a benchmark of 229 PyTorch operators across 3 difficulty levels. For
each task, the goal is to generate an optimized CUDA kernel that:
1. Produces correct output (matches PyTorch reference within tolerance)
2. Runs faster than PyTorch native (speedup_native > 1.0)

### Task Levels
- **Level 1** (91 tasks): Single operations — matmul, activations, pooling, conv, losses, reductions
- **Level 2** (98 tasks): Fused operations — Conv2d+ReLU+BiasAdd, Gemm+Sigmoid+Sum, etc.
- **Level 3** (40 tasks): Full architectures — MLP, LeNet, ResNet, VGG, EfficientNet, etc.

### Task Interface
Each task is a `.py` file with:
```python
def module_fn(*args) -> Tensor:     # The operation to optimize
class Model(nn.Module):              # Wraps module_fn
    def forward(self, *args, fn=module_fn): return fn(*args)
def get_inputs() -> list:            # Test inputs (moved to CUDA by harness)
def get_init_inputs() -> list:       # Model constructor args
```

### Kernel Interface
Generated CUDA kernels must expose a `forward()` function compatible with
`torch.utils.cpp_extension.load_inline`:
```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Your CUDA kernel(s)
__global__ void my_kernel(...) { ... }

// Entry point — must match the signature of module_fn
torch::Tensor forward(torch::Tensor input, ...) {
    // Allocate output, launch kernel, return result
}
```

The harness compiles via `load_inline(cuda_sources=[code], functions=["forward"])` and
calls `cuda_mod.forward(*inputs)` to get the output for correctness checking and timing.

### Correctness Checking
- Reference: `model(*inputs)` with `torch.no_grad()`
- Comparison: `torch.allclose(ref, test, atol=1e-3, rtol=1e-3)`
- Incorrect kernels get `speedup_native = 0.0` regardless of speed

### Runtime Measurement
- Median of 100 runs after 25 warmup iterations
- `torch.cuda.synchronize()` before each timing measurement
- Speedup = pytorch_native_ms / your_kernel_ms

## Sakana AI Baseline (Prior Art)
Sakana AI's CUDA Engineer generated ~130 kernel variants per task using evolutionary
search. Their best correct kernel per task achieves:
- Level 1: median 1.13x, mean 7.89x (skewed by outliers)
- Level 2: median 1.54x, mean 4.23x
- Level 3: median 1.29x, mean 1.61x
- Overall: median 1.34x (222 tasks without >10x outliers: mean 1.69x)

Our target: beat Sakana's median speedup per level and achieve higher correctness rate.

## CUDA Optimization Hierarchy (highest impact first)

1. **Memory coalescing** — threads in a warp access contiguous global memory addresses
2. **Shared memory tiling** — load tiles into shared memory, reuse to reduce global traffic
3. **Kernel fusion** — combine multiple ops into one kernel to eliminate intermediate memory round-trips (critical for Level 2/3 fused ops)
4. **Vectorized loads/stores** — float4/int4 for 128-bit memory transactions
5. **Register tiling** — accumulate partial results in registers
6. **Occupancy tuning** — balance threads, registers, shared memory per SM
7. **Warp-level primitives** — `__shfl_sync`, `__ballot_sync` for intra-warp communication
8. **Loop unrolling** — `#pragma unroll` for inner loops to increase ILP
9. **Double buffering** — overlap global memory loads with compute via ping-pong buffers
10. **Tensor cores** — `wmma` API or CUTLASS for matrix multiply operations

## Level-Specific Strategy

### Level 1 (Single Ops)
- Many are memory-bound (activations, element-wise, reductions)
- Matmul/conv tasks benefit most from tiling + tensor cores
- Reductions need warp shuffle + shared memory two-pass
- Element-wise ops: vectorized loads (float4) + high occupancy

### Level 2 (Fused Ops)
- PRIMARY OPPORTUNITY: kernel fusion eliminates intermediate tensors
- Read all sub-operations, fuse into a single kernel launch
- Shared memory for intermediate results between fused ops
- Often beats PyTorch by 2-5x just from fusion alone

### Level 3 (Full Architectures)
- Multiple kernel launches — fuse where possible
- For architectures with repeated blocks (ResNet, VGG), optimize the hot path
- Consider optimizing the most expensive layer only if full fusion is too complex
- Memory layout optimization across layers

## Common Pitfalls
- Bank conflicts in shared memory (pad arrays: `__shared__ float s[32][33]`)
- Uncoalesced global memory access (strided/random patterns)
- Register spilling to local memory (too many variables per thread)
- Missing `__syncthreads()` causing race conditions
- Incorrect grid/block dimensions (`blockDim.x * blockDim.y * blockDim.z <= 1024`)
- Default shared memory limit is 48KB (228KB opt-in on H100 via `cudaFuncSetAttribute`)
- Type mismatches: task may use float16/bfloat16 — handle with appropriate CUDA types

## Torch C++ Extension Patterns
```cpp
// Input validation macros
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Getting data pointers
float* data = input.data_ptr<float>();

// Creating output tensors
auto output = torch::zeros_like(input);
auto output = torch::empty({M, N}, input.options());

// Launch configuration
dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
dim3 blocks((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
            (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
my_kernel<<<blocks, threads>>>(args...);
C10_CUDA_CHECK(cudaGetLastError());
```
