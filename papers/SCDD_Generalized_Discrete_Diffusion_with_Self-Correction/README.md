# Generalized Discrete Diffusion with Self-Correction

[![Project Page](https://img.shields.io/badge/Project%20Page-0366d6?logo=gitbook&logoColor=white)](https://laaaarrywang.github.io/Self-Correcting-Discrete-Diffusion/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-ff6f00?logo=huggingface&logoColor=white)](https://huggingface.co/laaaarrywang/SCDD)
[![arXiv](https://img.shields.io/badge/arXiv-2603.02230-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.02230)

This repository contains the implementation code for our Self-Correcting Discrete Diffusion (SCDD) model, a generalized discrete diffusion model with self-correction capabilities.

## Environment Setup

Create the conda environment:
```bash
conda env create -f requirements.yaml
conda activate scdd
```

## Training

Train models using the provided scripts in `scripts/`:

```bash
# Train on OpenWebText
bash scripts/train_owt_scdd.sh

# Train on WikiText-103
bash scripts/train_wikitext_scdd.sh

# Train on LM1B
bash scripts/train_lm1b_scdd.sh
```

**Note:** Update the paths in the shell scripts:
- Set `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE` to your cache directories
- Set the `cd` path to your project directory

## Evaluation

Evaluate trained models:

```bash
# Evaluate on OpenWebText
bash scripts/eval_scdd_owt.sh

# Evaluate on WikiText-103  
bash scripts/eval_scdd_wikitext103.sh

# Evaluate on LM1B
bash scripts/eval_scdd_lm1b.sh
```

## Configuration

The model supports various configurations via Hydra:

- **Backbones**: `dit` (DiT), `dimamba` (DiMamba), `ar` (autoregressive baseline)
- **Parameterizations**: `scdd`, `d3pm`, `subs`, `sedd`
- **Noise schedules**: `loglinear`, `cosine`, `linear`
- **Forward processes**: `mask`, `mix` (masking + uniform noise)

Example:
```bash
python -u -m main \
  backbone=dit \
  parameterization=scdd \
  model=small \
  data=openwebtext \
  forward=mix \
  forward.ratio=0.1
```

## Model Sizes

Available model configurations in `configs/model/`:
- `tiny`: Small model for quick experiments
- `small`: Standard small model
- `medium`: Larger model

## Key Parameters

- `T`: Number of discrete timesteps (0 for continuous time, 1000 for discrete)
- `forward.ratio`: Maximum uniform noise ratio for mix forward process
- `forward.gamma`: Controls noise schedule shape
- `forward.t_peak`: Peak time for uniform noise (default: 0.5)
- `sampling.steps`: Number of denoising steps during generation
- `sampling.predictor`: Sampling method (`scdd`, `ddpm`, `analytic`)

## File Structure

```
.
├── main.py              # Main training/evaluation entry point
├── diffusion.py         # Core diffusion model implementation
├── dataloader.py        # Data loading utilities
├── noise_schedule.py    # Noise schedule implementations
├── utils.py             # Helper utilities
├── models/              # Model architectures
│   ├── dit.py          # Diffusion Transformer
│   ├── dimamba.py      # Diffusion Mamba
│   └── autoregressive.py
├── configs/             # Hydra configuration files
├── supplementary/       # Optional paper experiments
│   ├── correction/      # Correction ablations
│   └── llm_as_a_judge/  # Semantic comparison with LLM judging
└── scripts/             # Training and evaluation scripts
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
