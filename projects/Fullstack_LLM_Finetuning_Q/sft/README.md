# Supervised Fine-tuning (SFT)

This component handles supervised fine-tuning of language models on high-quality Q programming examples.

## Overview

The SFT pipeline fine-tunes pretrained models using:
- High-quality filtered examples
- Instruction-following format
- Full fine-tuning approach

## Directory Structure

```
sft/
├── run_sft.py           # Main training script
├── train_sft.py         # Core training logic
├── run_sft_exps.sh      # Example experiments
└── final_outputs/       # Training outputs
    ├── 1.5b/           # 1.5B model outputs
    ├── 3b/             # 3B model outputs
    ├── 7b/             # 7B model outputs
    ├── 14b/            # 14B model outputs
    └── 32b/            # 32B model outputs
```

## Training Examples

The `run_sft_exps.sh` script contains examples for training all 5 model sizes:

### Small Models (1.5B, 3B, 7B)
```bash
python run_sft.py \
  --base_model ${MODEL_DIR}/pretrain/final_outputs/1.5b/checkpoint-800 \
  --learning_rate 2e-5 \
  --max_steps 1000 \
  --output_dir final_outputs/1.5b/ \
  --experiment_name q-sft
```

### Large Models (14B, 32B)
```bash
accelerate launch --num_processes=4 --main_process_port=29501 run_sft.py \
  --base_model ${MODEL_DIR}/pretrain/final_outputs/14b/checkpoint-50/consolidated/ \
  --learning_rate 4e-6 \
  --max_steps 1000 \
  --output_dir final_outputs/14b/ \
  --experiment_name q-sft \
  --save_every_n_steps 50
```

## Model Paths

**Important**: Update the model paths in `run_sft_exps.sh` with your best pretrained checkpoints:

- **Small Models**: `${MODEL_DIR}/pretrain/final_outputs/[model_size]/checkpoint-[step]`
- **Large Models**: `${MODEL_DIR}/pretrain/final_outputs/[model_size]/checkpoint-[step]/consolidated/`

Replace `[model_size]` and `[step]` with your actual best checkpoint paths.

## Model Sizes Supported

1. **1.5B Model**: Uses checkpoint-800 from pretraining
2. **3B Model**: Uses checkpoint-800 from pretraining  
3. **7B Model**: Uses checkpoint-200 from pretraining
4. **14B Model**: Uses checkpoint-50/consolidated from pretraining (requires multi-GPU)
5. **32B Model**: Uses checkpoint-50/consolidated from pretraining (requires multi-GPU)

## Configuration

Key training parameters:

```yaml
# Common settings for all models
max_steps: 1000
learning_rate: 2e-5 (small models), 4e-6 (large models)
save_steps: 100
training_method: full

# Large model settings (14B, 32B)
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
num_processes: 4
```

## Training Parameters

Important parameters:

1. **Model Settings**
   - `--base_model`: Pretrained model path
   - `--max_seq_length`: Sequence length
   - `--bf16`: Mixed precision training

2. **Training Settings**
   - `--learning_rate`: Learning rate (2e-5 for small models, 4e-6 for large models)
   - `--batch_size`: Batch size per GPU
   - `--gradient_accumulation_steps`: Gradient accumulation
   - `--max_steps`: Total steps (1000 recommended)
   - `--save_steps`: Checkpoint frequency

## Next Steps: RL Training

After SFT, you can use the best checkpoints for reinforcement learning (RL). Update the model paths in the RL script:

- **RL Script**: Update `${MODEL_DIR}/sft/final_outputs/[model_size]/checkpoint-[step]` with your best SFT checkpoint

## Experiment Tracking

The pipeline integrates with Weights & Biases:

```bash
python run_sft.py \
    --experiment_name q-sft \
    --wandb
```

## Output Structure

Training produces:

```
final_outputs/[model_size]/
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

1. **Gradient Checkpointing**
   - Enabled by default
   - Trades computation for memory

2. **Batch Size Tuning**
   - Adjust `batch_size`
   - Use `gradient_accumulation_steps`

3. **Multi-GPU Training**
   - Use `accelerate launch` for 14B and 32B models
   - Configure appropriate number of processes

## Troubleshooting

Common issues and solutions:

1. **Out of Memory**
   - Reduce batch size
   - Increase gradient accumulation
   - Use appropriate number of GPUs for large models

2. **Training Instability**
   - Adjust learning rate
   - Modify warmup steps
   - Check gradient clipping

3. **Slow Training**
   - Optimize sequence length
   - Use appropriate batch size
   - Check GPU utilization

## Contributing

When adding features:
1. Update training scripts
2. Add new configuration options
3. Document parameters
4. Include example usage 