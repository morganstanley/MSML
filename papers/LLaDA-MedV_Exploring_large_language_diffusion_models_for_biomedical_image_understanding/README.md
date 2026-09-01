<h1 align="center">LLaDA-MedV: Exploring Large Language Diffusion Models for Biomedical Image Understanding</h1>

[![Code](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/LLM-VLM-GSL/LLaDA-MedV)
[![Paper](https://img.shields.io/badge/arXiv-2508.01617-b31b1b)](https://arxiv.org/abs/2508.01617v1)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/XZDong123/LLaDA-MedV)
[![License](https://img.shields.io/badge/License-Non--Commercial-blue)](ASU%20Non-Commercial%20License)

## 📰 News
- [2026-06] LLaDA-MedV has been accepted to **CVPR 2026**.
- [2025-08-06] We uploaded our paper to [arXiv](https://arxiv.org/abs/2508.01617v1) and released model weights.

## 🧠 Introduction
Autoregressive models (ARMs) have long dominated the landscape of biomedical vision-language models (VLMs). Recently, masked diffusion models such as LLaDA have emerged as promising alternatives, yet their application in the biomedical domain remains largely underexplored. To bridge this gap, we introduce **LLaDA-MedV**, the first large language diffusion model tailored for biomedical image understanding through vision instruction tuning.

LLaDA-MedV achieves relative performance gains of 7.855% over LLaVA-Med and 1.867% over LLaDA-V in the open-ended biomedical visual conversation task, and sets new state-of-the-art accuracy on the closed-form subset of three VQA benchmarks: 84.93% on VQA-RAD, 92.31% on SLAKE, and 95.15% on PathVQA. We further analyze both training and inference behaviors, highlighting the importance of initialization weight selection, fine-tuning strategies, and the interplay between sampling steps and response repetition.

## ✨ Highlights
- **Biomedical diffusion VLM**: LLaDA-MedV explores large language diffusion models for biomedical image understanding instead of the more common autoregressive paradigm.
- **Strong open-ended chat performance**: The model improves biomedical visual conversation quality over LLaVA-Med and LLaDA-V.
- **Strong downstream VQA results**: LLaDA-MedV achieves strong performance on VQA-RAD, SLAKE, and PathVQA after task-specific fine-tuning.
- **Controllable long-form responses**: The model can generate longer answers when response length is explicitly controlled, improving informativeness in biomedical dialogue.

## 🗂️ Repository Overview
This repository currently contains:

- the core multimodal model implementation under `llava/`
- LLaDA-based biomedical vision-language model extensions
- a single-image VQA inference demo in `test_lladamedv_vqa.py`
- evaluation helpers such as `eval_vlm_chat_gpt_score.py` and `openai_api.py`
- result figures and tables used in the paper

The current repository is strongest on **model/inference code and result presentation**. The README in its current form indicates that the full training and evaluation release is still being finalized.

## 🏗️ Repository Structure
```text
LLaDA-MedV/
├── llava/                         # Core multimodal model, serving, and training code
├── images/                        # Figures used in the README and paper presentation
├── test_lladamedv_vqa.py          # Single-image biomedical VQA demo
├── eval_vlm_chat_gpt_score.py     # GPT-based evaluation helper
├── openai_api.py                  # Async OpenAI API wrapper for evaluation
├── ASU Non-Commercial License
└── README.md
```

## 📊 Model Performance
### Open-End Biomedical Conversation
We adopt the [Biomedical Visual Chatbot benchmark](https://arxiv.org/abs/2306.00890) to evaluate LLaDA-MedV in a realistic open-ended biomedical conversation setting. As shown below, LLaDA-MedV demonstrates superior performance compared to several baselines, and we provide qualitative generation results for visualization.

<table>
  <tr>
    <td><img src="https://github.com/LLM-VLM-GSL/LLaDA-MedV/blob/main/images/fig-radarmedicalnew.png" width="500" alt="Radar performance figure"/></td>
    <td><img src="https://github.com/LLM-VLM-GSL/LLaDA-MedV/blob/main/images/bio-conversation.png" width="500" alt="Biomedical conversation example"/></td>
  </tr>
</table>

### Downstream Biomedical VQA
We also evaluate LLaDA-MedV on three biomedical visual question answering benchmarks after fine-tuning on the training sets of [VQA-RAD](https://pubmed.ncbi.nlm.nih.gov/30457565/), [SLAKE](https://arxiv.org/abs/2102.09542), and [PathVQA](https://arxiv.org/abs/2003.10286), following their official splits.

![Results](https://github.com/LLM-VLM-GSL/LLaDA-MedV/blob/main/images/table.png)

## ⚙️ Model Details
### Training
Please refer to the training implementation of [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V).

### Evaluation
Please refer to our GPT-based evaluation helper in [`eval_vlm_chat_gpt_score.py`](/Users/xenos/Desktop/git_proj/LLaDA-MedV/eval_vlm_chat_gpt_score.py). For more detailed evaluation protocols, please refer to [LLaVA-Med evaluation](https://github.com/microsoft/LLaVA-Med/tree/v1.0.0).

### Weights
We release our model weights to support future research in the community.

| 🧩 Model | 📝 Description | 🔗 Link |
| --- | --- | --- |
| `LLaDAMedV-2A4E` | Main model | [Google Drive](https://drive.google.com/drive/u/1/folders/1HwW4l-r9H3uyVPpysndP6ZOJXIQfkVB9) |
| `VQA_RAD_2E` | VQA-RAD fine-tuned model | [Google Drive](https://drive.google.com/drive/u/1/folders/1HwW4l-r9H3uyVPpysndP6ZOJXIQfkVB9) |
| `SLAKE_10E` | SLAKE fine-tuned model | [Google Drive](https://drive.google.com/drive/u/1/folders/1HwW4l-r9H3uyVPpysndP6ZOJXIQfkVB9) |
| `PathVQA_7E` | PathVQA fine-tuned model | [Google Drive](https://drive.google.com/drive/u/1/folders/1HwW4l-r9H3uyVPpysndP6ZOJXIQfkVB9) |
| `XZDong123/LLaDA-MedV` | 🤗 Hugging Face repository | [Hugging Face](https://huggingface.co/XZDong123/LLaDA-MedV) |

## 🚀 Quick Start
The most direct runnable entry point in this repository is `test_lladamedv_vqa.py`, which provides a single-image biomedical VQA demo.

```bash
python test_lladamedv_vqa.py \
  --image /path/to/your/image.png \
  --question "Please describe this biomedical image in detail." \
  --model-path XZDong123/LLaDA-MedV \
  --llada-v-codebase /path/to/LLaDA-V
```

Notes:

- `--llada-v-codebase` is required so the script can import the `llava` codebase correctly, and it can point to this repository root.
- The default vision tower is `google/siglip2-so400m-patch14-384`.
- The default model path is `XZDong123/LLaDA-MedV`.
- CUDA is expected for typical use.

## 🙏 Acknowledgements
We gratefully acknowledge the authors of the following open-source repositories, which served as valuable references during our implementation:

- [LLaDA](https://github.com/ML-GSAI/LLaDA)
- [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V)
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med/tree/v1.0.0)

We deeply appreciate their contributions to the research community.

## 📚 Citation
If you find our work useful in your research, please cite:

```bibtex
@article{dong2025llada,
  title={LLaDA-MedV: Exploring Large Language Diffusion Models for Biomedical Image Understanding},
  author={Dong, Xuanzhao and Zhu, Wenhui and Chen, Xiwen and Wang, Zhipeng and Qiu, Peijie and Tang, Shao and Li, Xin and Wang, Yalin},
  journal={arXiv preprint arXiv:2508.01617},
  year={2025}
}
```
