# AHA: Aligning Large Audio-Language Models for Reasoning Hallucinations via Counterfactual Hard Negatives



[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://github.com/LLM-VLM-GSL/AHA)
[![Arxiv](https://img.shields.io/badge/Arxiv-2512.24052-red)](https://arxiv.org/abs/2512.24052)
[![Model](https://img.shields.io/badge/HuggingFace-Model-blue)](https://huggingface.co/ASU-GSL/Qwen-Audio-AHA)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/ASU-GSL/AHA)

## 🚀 News
* **[2024.12]** We release **Qwen-Audio-AHA**, the dataset, and the inference code! 
* **[2024.12]** Our paper **"AHA: Aligning Large Audio-Language Models for Reasoning Hallucinations via Counterfactual Hard Negatives"** is available on [arXiv](https://arxiv.org/abs/2512.24052).

---

## 💡 Highlights
**AHA (Audio Hallucination Alignment)** is a unified framework designed to mitigate hallucinations in Large Audio-Language Models (LALMs) by focusing on fine-grained temporal reasoning and counterfactual alignment.

* **Hallucination Taxonomy:** We deconstruct LALM failures into four dimensions: Event Omission, False Event Identity, Temporal Relation Error, and Quantitative Temporal Error.
* **Counterfactual Hard Negative Mining:** A novel pipeline to construct high-quality preference data, forcing models to distinguish strict acoustic evidence from linguistically plausible fabrications.
* **AHA-Eval:** A diagnostic benchmark to rigorously test fine-grained temporal and causal reasoning in the audio modality.

---

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/LLM-VLM-GSL/AHA.git](https://github.com/LLM-VLM-GSL/AHA.git)
   cd AHA
   ```

2. **Install Dependencies:**
   ```bash
   conda create -n aha python=3.13 -y
   conda activate aha
   pip install -r requirements.txt
   ```

---

## 🤗 Model Zoo

| Model | Base Model | Training Data | Checkpoint |
| :--- | :--- | :--- | :--- |
| **Qwen-Audio-AHA** | Qwen-Audio | AHA Preference Dataset | [HF Link](https://huggingface.co/ASU-GSL/Qwen-Audio-AHA) |

---

## 📊 Dataset: AHA

The AHA dataset is specifically curated to address reasoning hallucinations in audio tasks. It includes high-quality instruction-following data and counterfactual preference pairs.

* **Dataset Link:** [ASU-GSL/AHA](https://huggingface.co/datasets/ASU-GSL/AHA)

The dataset is structured to help the model distinguish between what *sounds* plausible and what is *actually* present in the audio stream.

---

## 💻 Inference

```python
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from peft import PeftModel
from qwen_omni_utils import process_mm_info
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen2.5-Omni-7B"
adapter_id = "ASU-GSL/Qwen-Audio-AHA"

# Load base model and processor
processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_id)

# Load Audio
audio, _ = librosa.load("example.wav", sr=processor.feature_extractor.sampling_rate)
prompt = "<|audio|>\nDescribe the temporal order of events in this audio."
inputs = processor(text=prompt, audios=audio, return_tensors="pt").to(device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=256)
print(processor.batch_decode(generate_ids, skip_special_tokens=True))[0]
```

---

## 📝 Citation
```bibtex
@article{chen2025aha,
  title={AHA: Aligning Large Audio-Language Models for Reasoning Hallucinations via Counterfactual Hard Negatives},
  author={Chen, Yanxi and Zhu, Wenhui and Chen, Xiwen and Wang, Zhipeng and Li, Xin and Qiu, Peijie and Wang, Hao and Dong, Xuanzhao and Xiong, Yujian and Schneider, Anderson and others},
  journal={arXiv preprint arXiv:2512.24052},
  year={2025}
}
```
