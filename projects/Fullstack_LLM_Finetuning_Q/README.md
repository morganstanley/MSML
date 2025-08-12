# Q Language Model Training Pipeline

> **Technical Report Implementation**: This repository contains the complete implementation for our technical report on adapting large language models to the Q programming language. Our work addresses the challenge of training LLMs for specialized programming languages that are under-represented on the Internet.

> **📄 [Technical Report: Full-Stack Fine-Tuning for the Q Programming Language](LINK_TO_PAPER_PLACEHOLDER)** *(Coming Soon)*

## About This Work

This repository implements our research on adapting large language models to the Q programming language, a specialized tool used in quantitative finance. While general-purpose LLMs excel at mainstream languages like Python and Java, they struggle with niche programming languages that are poorly represented in their training data.

Our approach demonstrates how to effectively train LLMs for specialized domains through a complete pipeline: dataset construction, pretraining, supervised fine-tuning, and reinforcement learning. We train models across five parameter sizes (1.5B to 32B) and achieve significant improvements over frontier models like GPT-4.1 and Claude Opus-4 on our Q programming benchmark.

This work provides a blueprint for adapting LLMs to other specialized domains where evaluation relies on objective signals (like code execution) rather than subjective human judgment.

---

A comprehensive pipeline for training large language models on the Q programming language, from dataset creation to evaluation, including iterative bootstrap improvement.

## Overview

This project provides a complete pipeline for training language models to understand and generate Q programming language code. It includes:

1. **Dataset Building**: Converting and validating Python solutions to Q
2. **Bootstrap Loop**: Iterative model improvement and dataset expansion
3. **Pretraining**: Efficient pretraining using various methods (LoRA, QLoRA)
4. **Fine-tuning**: Supervised and reinforcement learning approaches
5. **Evaluation**: Robust two-phase evaluation system

## Project Structure

```
.
├── build_dataset/      # Dataset creation and processing
│   ├── bootstrap_loop.py    # Iterative improvement loop
│   ├── convert_to_q.py     # Python to Q conversion
│   └── final_q_verify.py   # Solution verification
├── pretrain/          # Pretraining scripts and configs
├── sft/               # Supervised fine-tuning
├── rl/                # Reinforcement learning
├── eval/              # Model evaluation
├── config.py          # Configuration management
├── config.yaml        # User configuration
└── SETUP.md          # Detailed setup instructions
```

## Quick Start

1. **Setup**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Configure
   cp config.example.yaml config.yaml
   # Edit config.yaml as needed
   ```

2. **Build Dataset**:
   ```bash
   cd build_dataset
   python process_dataset.py
   python convert_to_q.py
   ```

3. **Bootstrap Loop**:
   ```bash
   cd build_dataset
   python bootstrap_loop.py \
     --base-model Qwen/Qwen3-32B \
     --max-iterations 10 \
     --target-problems 50
   ```

4. **Train**:
   ```bash
   # Pretrain
   cd pretrain
   python run_pretraining.py
   
   # Fine-tune
   cd ../sft
   python run_sft.py
   ```

5. **Evaluate**:
   ```bash
   cd eval
   python run_full_evaluation.py
   ```

See [SETUP.md](SETUP.md) for detailed instructions.

## Components

### Dataset Building

The dataset pipeline:
1. Processes LeetCode problems
2. Converts Python solutions to Q using LLMs
3. Validates and filters examples
4. Creates training/validation splits

### Bootstrap Loop

An iterative improvement process that:
1. **Solves Problems**: Uses current best model to solve unsolved problems
2. **Creates Dataset**: Combines all solved problems into training data
3. **Trains Model**: Fine-tunes model on expanded dataset
4. **Evaluates**: Compares base vs trained model performance
5. **Iterates**: Repeats until target number of problems solved

**Usage**:
```bash
python bootstrap_loop.py \
  --base-model Qwen/Qwen3-32B \
  --max-iterations 30 \
  --train-steps 1000 \
  --learning-rate 2e-5 \
  --target-problems 100
```

**Output Structure**:
```
new_iterations/
├── iteration_1/
│   ├── solved_problems/     # Problems solved in this iteration
│   ├── evaluation/          # Model evaluation results
│   ├── model/              # Trained model checkpoint
│   ├── metadata.json       # Iteration metadata
│   └── dataset_stats.json  # Dataset creation stats
└── ...
```

### Pretraining

Supports multiple training approaches:
- Full fine-tuning
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)

### Fine-tuning

Two-stage fine-tuning process:
1. Supervised Fine-tuning (SFT)
2. Reinforcement Learning (RL)

### Evaluation

Robust evaluation system:
- Two-phase: generation and testing
- Multiple metrics (pass@k, test case coverage)
- Task-specific performance analysis

## Training Pipeline

### 1. Dataset Creation
```bash
cd build_dataset
python convert_to_q.py --llm_to_use gpt4
```

### 2. Bootstrap Improvement (Optional)
```bash
cd build_dataset
python bootstrap_loop.py --base-model Qwen/Qwen3-32B
```

### 3. Pretraining
```bash
cd pretrain
python run_pretraining.py --model_name Qwen/Qwen2.5-7B-Instruct
```

### 4. Supervised Fine-tuning
```bash
cd sft
python run_sft.py --base_model /path/to/pretrained/model
```

### 5. Reinforcement Learning
```bash
cd rl
python rl_trainer.py --model /path/to/sft/model
```

## Model Sizes Supported

The pipeline supports training on multiple model sizes:

1. **1.5B Model**: `Qwen/Qwen2.5-1.5B-Instruct`
2. **3B Model**: `Qwen/Qwen2.5-3B-Instruct`
3. **7B Model**: `Qwen/Qwen2.5-7B-Instruct`
4. **14B Model**: `Qwen/Qwen2.5-14B-Instruct` (requires multi-GPU)
5. **32B Model**: `Qwen/Qwen2.5-32B-Instruct` (requires multi-GPU)

## Requirements

- Python 3.8+
- Q interpreter
- CUDA-capable GPU (recommended)
- Access to LLM APIs (optional, for dataset building)
- Multiple GPUs for large models (14B, 32B)

## Configuration

See [config.example.yaml](config.example.yaml) for available settings.

Key configuration areas:
- Dataset paths and processing
- Model selection and parameters
- Training hyperparameters
- Evaluation settings
- Bootstrap loop parameters

## Monitoring and Logging

### Bootstrap Progress
- **Iteration Metadata**: Stored in `metadata.json` per iteration
- **High-Level Summaries**: Stored in `new_high_level_summary/`
- **Final Report**: Complete bootstrap summary in `final_report.json`

### Training Logs
- **WandB Integration**: Automatic logging for all training runs
- **Checkpoint Management**: Automatic model checkpointing
- **Evaluation Metrics**: Detailed performance tracking

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Use LoRA/QLoRA for large models
   - Reduce batch size
   - Enable gradient checkpointing

2. **Bootstrap Loop Issues**
   - Check training script exists (`sft/run_sft.py`)
   - Verify model paths are correct
   - Monitor GPU memory usage

3. **Dataset Issues**
   - Verify Q interpreter installation
   - Check API keys for LLM services
   - Validate file permissions


## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.


## License

This project is licensed under the Apache 2.0 License - see
[LICENSE](LICENSE) for details.


## Citation

If you use this work in your research, please cite:

```bibtex
@misc{q-language-model-training,
  author = {Hogan, et. al },
  title = {Technical Report: Full-Stack Fine-Tuning for the Q Programming Language},
  year = {2025},
  publisher = {Morgan Stanley},
}
``` 
