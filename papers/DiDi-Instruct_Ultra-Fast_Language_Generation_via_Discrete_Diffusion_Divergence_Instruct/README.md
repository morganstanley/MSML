# [[ICLR 2026] Ultra-Fast Language Generation via <br>Discrete Diffusion Divergence Instruct](https://openreview.net/forum?id=mtdyZsa47V) (DiDi-Instruct)

[![Blog](https://img.shields.io/badge/Blog-0366d6?logo=gitbook&logoColor=white)](https://haoyangzheng.github.io/research/didi-instruct/)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/JDfKRiqXcYE)
[![Google Drive](https://img.shields.io/badge/GoogleDrive-34a853?logo=google-drive&logoColor=white)](https://drive.google.com/drive/folders/1bQlwZoaowkGy3FXnrtb4YEleKIDHrQNE?usp=sharing)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-ff6f00?logo=huggingface&logoColor=white)](https://huggingface.co/haoyangzheng/didi-instruct-small)
[![arXiv](https://img.shields.io/badge/arXiv-2509.25035-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.25035)
[![Python](https://img.shields.io/badge/Python-3.12.11-yellow?logo=python&logoColor=white)](https://github.com/haoyangzheng-ai/didi-instruct/blob/main/environment.yml) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?logo=opensourcehardware&logoColor=white)](./LICENSE.md)

By [Haoyang Zheng](https://scholar.google.com/citations?hl=en&user=cq_f7MUAAAAJ&view_op=list_works&sortby=pubdate), [Xinyang Liu](https://xinyangatk.github.io/), [Cindy Xiangrui Kong](https://xiangruikong.com/), [Nan Jiang](https://jiangnanhugo.github.io/), [Zheyuan Hu](https://scholar.google.com/citations?user=On2YFigAAAAJ&hl=zh-CN),
[Weijian Luo](https://pkulwj1994.github.io/), [Wei Deng](https://www.weideng.org/), and [Guang Lin](https://www.math.purdue.edu/~lin491/)

---

## 🔄 Updates

* **2026-02-05**: We released the training code.
* **2026-01-25**: DiDi-Instruct was accepted by [ICLR 2026](https://openreview.net/forum?id=mtdyZsa47V).
* **2026-01-16**: Invited talk on DiDi-Instruct is now available on [YouTube](https://youtu.be/JDfKRiqXcYE).
* **2025-10-06**: We update the [Blog](https://haoyangzheng.github.io/research/didi-instruct/).
* **2025-10-05**: We released the checkpoint on [Hugging Face](https://huggingface.co/haoyangzheng/didi-instruct-small).
* **2025-10-03**: We updated the [evaluation code](https://github.com/haoyangzheng-ai/didi-instruct/blob/main/scripts/eval-didi-instruct.sh) and released [the model checkpoint](https://drive.google.com/drive/folders/1bQlwZoaowkGy3FXnrtb4YEleKIDHrQNE?usp=sharing).
* **2025-09-29**: We uploaded our work to [arXiv](https://arxiv.org/abs/2509.25035).

---

## Abstract

Fast and high-quality language generation is the holy grail that people pursue in the age of AI. In this work, we introduce **Di**screte **Di**ffusion Divergence **Instruct** (**DiDi-Instruct**), a training-based method that initializes from a pre-trained diffusion large language model (dLLM) and distills a few-step student for fast generation. The model distilled with DiDi-Instruct matches or surpasses its dLLM teacher and the GPT-2 baseline while providing up to **64× acceleration**. The theoretical foundation of DiDi-Instruct is a novel framework based on integral KL-divergence minimization, which leads to a practical training algorithm. We further introduce *grouped reward normalization, intermediate-state matching, and the reward-guided ancestral sampler* to improve *training stability, model coverage, and inference quality*. On the OpenWebText benchmark, DiDi-Instruct achieves perplexity ranging from 62.2 (8 NFEs) to 18.4 (128 NFEs), outperforming prior accelerated dLLMs and the GPT-2 baseline. These gains incur a negligible entropy loss (around $1$%) and reduce additional training wall-clock time by **more than 20×** compared to competing dLLM distillation methods. We further validate the robustness and effectiveness of DiDi-Instruct through extensive ablation studies, model scaling, downstream task evaluations, and unconditional protein sequence generation. In conclusion, DiDi-Instruct enables efficient and effective distillation for language generation in the blink of an eye.

---

## 🚀 Feel the Generation Speed

### Auto-Regressive Model (GPT-2 Small)<br><sub>Token-by-token generation → high latency</sub>
![ARM](https://github.com/haoyangzheng-ai/didi-instruct/blob/main/demos/arm.gif)

### Masked Diffusion Model (MDLM, 169M)<br><sub>Iterative denoising → faster than GPT-2 Small.</sub>
![MDLM](https://github.com/haoyangzheng-ai/didi-instruct/blob/main/demos/mdlm.gif)

### DiDi-Instruct (distilled from 169M MDLM)<br><sub>Distilled few-step student → up to **64× speedup** with matched/better quality.</sub>
![DiDi-Instruct](https://github.com/haoyangzheng-ai/didi-instruct/blob/main/demos/didi-instruct.gif)

---

## 🏗️ Usage Guide

### 1. Create and Activate the Conda Environment

Before first use, create and activate the conda environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate mask_model
```

### 2. Prepare the Teacher Model

You need a pre-trained discrete diffusion language model (dLLM) as the teacher. You have two options:

- **Option A (Train from scratch):**  
  - Refer to [this script from DUO](https://github.com/s-sahoo/duo/blob/main/scripts/train_owt_mdlm.sh) to train your own teacher model on OpenWebText.  
  - This produces a checkpoint (e.g., `mdlm.ckpt`).

- **Option B (Use pre-trained checkpoint):**  
  - Download a pre-trained checkpoint from [Google Drive](https://drive.google.com/drive/folders/16LuuptK7Xfk-vzhQYZBZ0SA-B-BFluau) (`mdlm.ckpt`).
  - Place it in the `./out/` directory for later use.

### 3. Distill the Student Model

Once you have the teacher model checkpoint, distill a few-step student model for fast inference:

```bash
bash ./scripts/distill-didi-instruct-owt.sh
```

The script will look for the teacher checkpoint and begin the distillation process.

**Pre-trained checkpoint options:**

- **Option 1 (from Google Drive):**  
  - Download the distilled DiDi-Instruct checkpoint from [Google Drive](https://drive.google.com/drive/folders/1bQlwZoaowkGy3FXnrtb4YEleKIDHrQNE?usp=sharing) (`didi-instruct.ckpt`).  
  - Place the `.ckpt` file in the `./out/` directory.

- **Option 2 (from Hugging Face):**  
  - We provide the distilled model on [Hugging Face](https://huggingface.co/haoyangzheng/didi-instruct-small).  
  - Convert it to `.ckpt` format using:

    ```bash
    python ./models/hf_to_ckpt.py --hf_repo_id "haoyangzheng/didi-instruct-small" --output_dir "./out/didi-instruct.ckpt"
    ```

### 4. Evaluate the Model

Evaluate the distilled student model's performance by measuring perplexity and entropy compared to the teacher and baseline models:

```bash
bash ./scripts/eval-didi-instruct.sh
```

This produces performance metrics on the OpenWebText validation set.

---
## 📁 Repository Structure

```
didi-instruct-train/
├── configs/    # Configuration files
├── models/     # Model implementations
├── scripts/    # Training and evaluation scripts
├── out/        # Checkpoints and logs
├── algo.py     # Core algorithm implementations
├── dataloader.py
├── main.py
├── metrics.py
├── trainer_base.py
├── utils.py
├── environment.yml
├── README.md
└── LICENSE.md
```
---

## 📚 References

This repository is built upon [DUO](https://github.com/s-sahoo/duo): ["The Diffusion Duality. ICML 2025"](https://arxiv.org/abs/2506.10892).

We also adopt ideas from [DiMO](https://github.com/yuanzhi-zhu/DiMO), [MDLM](https://github.com/kuleshov-group/mdlm), [SDTT](https://github.com/jdeschena/sdtt), and [nanoGPT](https://github.com/karpathy/nanoGPT).

---

## 📖 Citation

If you find this repository useful, please cite the following work:

```
@article{zheng2025ultra,
  title={{Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct}},
  author={Zheng, Haoyang and Liu, Xinyang and Kong, Cindy Xiangrui and Jiang, Nan and Hu, Zheyuan and Luo, Weijian and Deng, Wei and Lin, Guang},
  journal={{Proceedings of the International Conference on Learning Representations (ICLR)}},
  year={2026}
}
```

---
