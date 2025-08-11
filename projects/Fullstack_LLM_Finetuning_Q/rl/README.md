# Reinforcement Learning (RL)

This component implements reinforcement learning for Q language model training using TRL's GRPO (Group Relative Policy Optimization) trainer.

## Overview

The RL pipeline uses:
- **GRPO Training**: Group Relative Policy Optimization from TRL
- **Test Case Rewards**: Binary pass/fail based on Q code execution
- **Reasoning Format**: Optional structured reasoning + code format
- **vLLM Integration**: High-performance inference server
- **Multi-GPU Training**: Accelerate-based distributed training

## Directory Structure

```
rl/
├── rl_trainer.py      # Main RL training script (GRPO)
├── exps.sh            # Example experiment scripts
└── final_outputs/     # Training outputs
    ├── 1.5b/         # 1.5B model outputs
    ├── 3b/           # 3B model outputs
    ├── 7b/           # 7B model outputs
    ├── 14b/          # 14B model outputs
    └── 32b/          # 32B model outputs
```

## Training Examples

The `exps.sh` script contains examples for training all 5 model sizes:

### Small Models (1.5B, 3B, 7B) - Non-Reasoning
```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml rl_trainer.py \
  --model /your/best/sft/1.5b/checkpoint \
  --use_vllm \
  --output_dir final_outputs/1.5b/non_reasoning/ \
  --learning_rate 2e-6 \
  --simple_eval_problems 15 \
  --repeat_eval_problems 4 \
  --eval_steps 25 \
  --use_wandb \
  --wandb_name 1.5b_non_reason \
  --generation_temp .8
```

### Large Models (14B, 32B) - Reasoning Format
```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml rl_trainer.py \
  --model /your/best/sft/14b/checkpoint \
  --use_vllm \
  --output_dir final_outputs/14b/reasoning/ \
  --learning_rate 2e-6 \
  --simple_eval_problems 15 \
  --repeat_eval_problems 4 \
  --eval_steps 25 \
  --use_wandb \
  --wandb_name 14b_reasoning \
  --generation_temp .8 \
  --use_reasoning_format
```

## Model Paths

**Important**: Update the model paths in `exps.sh` with your best SFT checkpoints:

- **Small Models**: `/your/best/sft/[model_size]/checkpoint` (non-reasoning format)
- **Large Models**: `/your/best/sft/[model_size]/checkpoint` (reasoning format for 14B/32B)

Replace `[model_size]` with your actual best SFT checkpoint paths.

## Model Sizes Supported

1. **1.5B Model**: Non-reasoning format
2. **3B Model**: Non-reasoning format  
3. **7B Model**: Non-reasoning format
4. **14B Model**: Reasoning format (requires multi-GPU)
5. **32B Model**: Reasoning format (requires multi-GPU)

## Key Features

### Reward System
- **Test Case Execution**: Runs generated Q code against test cases
- **Binary Rewards**: Pass/fail based on test case success
- **Perfect Bonus**: Additional reward for passing all tests
- **Timeout Handling**: Robust execution with timeout protection

### Training Methods
- **GRPO**: Group Relative Policy Optimization from TRL
- **Multi-Generation**: Generates multiple completions per prompt
- **Pass@K Simulation**: Repeated evaluation for pass@k-like metrics
- **vLLM Server**: High-performance inference during training

### Format Options
- **Non-Reasoning**: Direct Q code generation
- **Reasoning Format**: Structured `<reasoning>` + `<answer>` format

## Configuration

Key training parameters:

```yaml
# Common settings for all models
learning_rate: 2e-6
generation_temp: 0.8
eval_steps: 25
simple_eval_problems: 15
repeat_eval_problems: 4

# Large model settings (14B, 32B)
num_processes: 7
gradient_accumulation_steps: 8
per_device_train_batch_size: 1
```

## Training Parameters

Important parameters:

1. **Model Settings**
   - `--model`: SFT model path
   - `--use_reasoning_format`: Enable reasoning format (14B/32B)

2. **Training Settings**
   - `--learning_rate`: Learning rate (2e-6 recommended)
   - `--generation_temp`: Temperature for generation (0.8 recommended)
   - `--eval_steps`: Evaluation frequency (25 recommended)
   - `--simple_eval_problems`: Number of eval problems (15 recommended)

3. **Reward Settings**
   - `--base_reward_weight`: Weight for test case rewards (1.0 default)
   - `--perfect_reward_weight`: Weight for perfect solution bonus (1.0 default)
   - `--max_tests_per_problem`: Max tests per problem (3 default)

4. **vLLM Settings**
   - `--use_vllm`: Enable vLLM server mode
   - `--max_model_len`: Maximum model length (8192 for 32B)

## Experiment Tracking

The pipeline integrates with Weights & Biases:

```bash
python rl_trainer.py \
    --use_wandb \
    --wandb_name your_experiment_name \
    --wandb_project q-rl
```

## Output Structure

Training produces:

```
final_outputs/[model_size]/[format]/
├── checkpoints/
│   ├── checkpoint-100/   # Checkpoint at step 100
│   ├── checkpoint-200/   # Checkpoint at step 200
│   └── ...
├── logs/
│   ├── training_log.txt  # Training progress
│   └── eval_results.json # Evaluation metrics
└── config.yaml          # Training configuration
```

## Memory Optimization

Tips for managing memory usage:

1. **vLLM Integration**
   - High-performance inference server
   - Efficient memory management
   - Multi-GPU support

2. **Batch Size Tuning**
   - Adjust `per_device_train_batch_size`
   - Use `gradient_accumulation_steps`

3. **Multi-GPU Training**
   - Use `accelerate launch` for all models
   - Configure appropriate number of processes

## Troubleshooting

Common issues and solutions:

1. **Out of Memory**
   - Reduce batch size
   - Increase gradient accumulation
   - Use appropriate number of GPUs

2. **Training Instability**
   - Adjust learning rate
   - Modify generation temperature
   - Check reward scaling

3. **vLLM Issues**
   - Ensure vLLM is properly installed
   - Check GPU memory allocation
   - Verify model compatibility

## Contributing

When adding features:
1. Update training scripts
2. Add new reward components
3. Document parameters
4. Include example usage 