# Setup Guide for Q Language Model Training Pipeline

This guide will help you set up the Q language model training pipeline on your system.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Q interpreter installed
- Git

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fullstack_finetuning_q
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up configuration**:
   ```bash
   cp config.example.yaml config.yaml
   ```

4. **Edit configuration**:
   Edit `config.yaml` to match your environment.

## Environment Variables

The following environment variables can be set either in your shell or in a `.env` file:

```bash
# Q interpreter path
Q_INTERPRETER_PATH=/path/to/q/interpreter

# API keys (optional, for dataset building)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
XAI_API_KEY=your_xai_key
GEMINI_API_KEY=your_gemini_key
```

## Configuration

The `config.yaml` file controls various aspects of the training pipeline:

```yaml
dataset:
  initial_dataset_dir: "initial_dataset"
  validated_dataset_dir: "validated_dataset"
  final_dataset_dir: "final_dataset"
  q_interpreter_path: "q"  # Path to Q interpreter

model:
  base_model: "Qwen/Qwen2.5-7B-Instruct"
  max_steps: 1000
  learning_rate: 2e-5

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  save_every_n_steps: 200
  eval_steps: 20
```

## Quick Start

1. **Build Dataset**:
   ```bash
   cd build_dataset
   python process_dataset.py
   python convert_to_q.py
   ```

2. **Pretrain Model**:
   ```bash
   cd pretrain
   python run_pretraining.py
   ```

3. **Supervised Fine-tuning**:
   ```bash
   cd sft
   python run_sft.py
   ```

4. **Reinforcement Learning**:
   ```bash
   cd rl
   python rl_trainer.py
   ```

5. **Evaluate Model**:
   ```bash
   cd eval
   python run_full_evaluation.py
   ```

## Troubleshooting

### Common Issues

1. **Q Interpreter Not Found**:
   - Set Q_INTERPRETER_PATH in .env or config.yaml
   - Ensure Q is properly installed and accessible

2. **CUDA Out of Memory**:
   - Reduce batch size in config.yaml
   - Use gradient accumulation
   - Enable gradient checkpointing

3. **API Key Issues**:
   - Verify API keys are set correctly
   - Check rate limits for API services

## Contributing

Please see CONTRIBUTING.md for guidelines on contributing to this project. 