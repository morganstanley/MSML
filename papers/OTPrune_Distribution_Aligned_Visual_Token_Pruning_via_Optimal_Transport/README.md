# OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport

[![arXiv](https://img.shields.io/badge/arXiv-2602.20205-b31b1b.svg)](https://arxiv.org/abs/2602.20205)
[![Conference](https://img.shields.io/badge/CVPR-2026-4b44ce.svg)](https://cvpr.thecvf.com/virtual/2026/poster/41079)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of **OTPrune**, a training-free visual token pruning framework for Multimodal Large Language Models (MLLMs).

OTPrune formulates token pruning as a distribution alignment problem via optimal transport (OT). By minimizing the 2-Wasserstein distance between the full and pruned token distributions, it preserves both the local diversity and global representativeness while significantly reducing inference cost (~90% token reduction with minimal performance drop).


---

## Highlights

- **Training-free**: No fine-tuning required — plug into any LLaVA model at inference time
- **Theoretically grounded**: Pruning objective is proven to be monotone and submodular, guaranteeing near-optimal greedy solutions
- **High compression**: ~90% visual token reduction (keep only ~10%) with competitive performance
- **Simple integration**: Controlled entirely via environment variables — zero code change needed at inference

---

## Installation

### 1. Environment Setup

```bash
conda create -n otprune python=3.10 -y
conda activate otprune
```

### 2. Install LLaVA

```bash
cd LLaVA
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### 3. Install lmms-eval

```bash
cd ../lmms_eval
pip install -e .
```

---

## Quick Start

### Single GPU Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 \
BASELINE=OURS \
LAYER_INDEX=0 \
SUBSET_RATIO=0.098 \
python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks gqa,pope,mme \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/otprune_7b
```

### Multi-GPU Evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
BASELINE=OURS \
LAYER_INDEX=0 \
SUBSET_RATIO=0.098 \
python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-13b" \
    --tasks gqa,pope,mme \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/otprune_13b
```

### Full Benchmark (Paper Results)

```bash
bash run_OTPrune.sh
```

This runs evaluation on all 11 benchmarks: COCO Caption, Flickr30k, GQA, MMBench, MME, MMMU, NoCaps, OK-VQA, POPE, ScienceQA, and SEEDBench.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BASELINE` | Set to `OURS` to enable OTPrune | - |
| `LAYER_INDEX` | Layer at which to apply pruning (`0` = embedding layer) | - |
| `SUBSET_RATIO` | Fraction of visual tokens to keep (e.g., `0.098` ≈ 10%) | - |

When `LAYER_INDEX` and `BASELINE` are not set, the model runs without pruning (baseline).

---

## Supported Models

| Model | HuggingFace ID |
|-------|---------------|
| LLaVA-1.5-7B | `liuhaotian/llava-v1.5-7b` |
| LLaVA-1.6-7B | `liuhaotian/llava-v1.6-vicuna-7b` |
| LLaVA-1.5-13B | `liuhaotian/llava-v1.5-13b` |

---

## Method Overview

OTPrune formulates visual token pruning as a **distribution alignment** problem under the optimal transport (OT) framework:

1. **OT Objective**: We minimize the 2-Wasserstein distance between the full and pruned token distributions, ensuring the selected subset faithfully represents the original visual information
2. **Tractable Relaxation**: The OT objective is relaxed into a submodular maximization problem over a kernel matrix $K = I + \gamma \cdot S S^\top$, where $S$ is the normalized token similarity matrix. We theoretically prove this objective satisfies **monotonicity** and **submodularity**, providing a $(1 - 1/e)$ approximation guarantee for greedy optimization
3. **Efficient Greedy Selection**: Tokens are iteratively selected to maximize the marginal gain of the submodular objective, using Cholesky-based incremental updates in $O(k^2 n)$ time
4. **Seamless Integration**: The selected tokens replace the full visual token sequence (preserving system and text tokens) before LLM processing — no architectural modification required

This principled formulation ensures that the pruned token set preserves both **local diversity** (avoiding redundant nearby tokens) and **global representativeness** (covering the full semantic distribution of the image).

---

## Project Structure

```
otprune/
├── README.md
├── run_OTPrune.sh                          # Main evaluation script
├── LLaVA/                                  # Modified LLaVA codebase
│   └── llava/
│       └── model/
│           └── llava_arch.py               # Core OTPrune implementation
│               ├── greedy_select()         #   Fast greedy selection algorithm
│               ├── OTPrune()               #   Kernel construction + token selection
│               └── prepare_inputs_labels_for_multimodal()
│                                           #   Integration into LLaVA forward pass
└── lmms_eval/                              # Evaluation framework
    ├── models/
    │   └── llava.py                        # LLaVA model wrapper for evaluation
    └── tasks/                              # 11 VL benchmark configurations
```

---

## Citation

```
@article{chen2026otprune,
  title={OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport},
  author={Chen, Xiwen and Zhu, Wenhui and Li, Gen and Dong, Xuanzhao and Xiong, Yujian and Wang, Hao and Qiu, Peijie and Song, Qingquan and Wang, Zhipeng and Tang, Shao and others},
  journal={arXiv preprint arXiv:2602.20205},
  year={2026}
}
```


---

## Acknowledgments

This codebase is built upon [LLaVA](https://github.com/haotian-liu/LLaVA), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [DivPrune](https://github.com/vbdi/divprune). We thank the authors for their open-source contributions.

---

## License

This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0). See the original [LLaVA license](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE) for the base codebase.
