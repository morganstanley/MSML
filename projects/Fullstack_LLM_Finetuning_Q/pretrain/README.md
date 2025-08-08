# Pretraining Pipeline

This component handles the pretraining of language models on Q programming language code using the kdb+ license dataset.

## Overview

The pretraining pipeline supports full fine-tuning on a curated dataset of approximately 1.6M tokens of Q programming language code.

## Data Setup

1. **Download the Dataset**:
   ```bash
   python download_ds.py --repo-name kdb-license-dataset --output-dir kdb_license_processed
   ```

   This will download the dataset and save it as JSONL files in the `kdb_license_processed/` directory.

## Directory Structure

```
pretrain/
├── run_pretraining.py    # Main training script
├── train_pretrain.py     # Core training logic
├── pretrain_exps.sh      # Example experiment scripts
├── download_ds.py        # Dataset download script
└── final_outputs/        # Training outputs
    ├── 1.5b/            # 1.5B model outputs
    ├── 3b/              # 3B model outputs
    ├── 7b/              # 7B model outputs
    ├── 14b/             # 14B model outputs
    └── 32b/             # 32B model outputs
```

## Training Examples

The `pretrain_exps.sh` script contains examples for training all 5 model sizes:

### Small Models (1.5B, 3B, 7B)
```bash
python run_pretraining.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --data_type licensed \
  --training_method full \
  --max_steps 800 \
  --save_steps 100 \
  --learning_rate 1e-5 \
  --output_dir final_outputs/1.5b \
  --wandb
```

### Large Models (14B, 32B)
```bash
accelerate launch --num_processes=4 train_pretrain.py \
  --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
  --train_file kdb_license_processed/train_license_kdbsite.jsonl \
  --eval_file kdb_license_processed/eval_license_kdbsite.jsonl \
  --max_steps 800 \
  --save_steps 80 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --output_dir final_outputs/14b \
  --report_to wandb \
  --wandb_project q-pretraining
```

## Model Sizes Supported

1. **1.5B Model**: `Qwen/Qwen2.5-1.5B-Instruct`
2. **3B Model**: `Qwen/Qwen2.5-3B-Instruct`
3. **7B Model**: `Qwen/Qwen2.5-7B-Instruct`
4. **14B Model**: `Qwen/Qwen2.5-14B-Instruct` (requires multi-GPU)
5. **32B Model**: `Qwen/Qwen2.5-32B-Instruct` (requires multi-GPU)

## Next Steps: SFT and RL Training

After pretraining, you can use the best checkpoints for supervised fine-tuning (SFT) and reinforcement learning (RL). Update the model paths in the SFT and RL scripts:

- **SFT Script**: Update `${MODEL_DIR}/pretrain/final_outputs/[model_size]/checkpoint-[step]` with your best pretrained checkpoint
- **RL Script**: Update `${MODEL_DIR}/sft/final_outputs/[model_size]/checkpoint-[step]` with your best SFT checkpoint

## Configuration

Key training parameters:

```yaml
# Common settings for all models
max_steps: 800
learning_rate: 1e-5
save_steps: 100
training_method: full

# Large model settings (14B, 32B)
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
num_processes: 4
```

## Training Parameters

Important training parameters:

1. **Model Selection**
   - `--model_name`: HuggingFace model ID
   - `--max_seq_length`: Maximum sequence length

2. **Training Settings**
   - `--learning_rate`: Learning rate (1e-5 recommended)
   - `--batch_size`: Batch size per GPU
   - `--gradient_accumulation_steps`: Steps for gradient accumulation
   - `--max_steps`: Total training steps (800 recommended)
   - `--save_steps`: Checkpoint frequency

## Experiment Tracking

The pipeline integrates with Weights & Biases:

```bash
python run_pretraining.py \
    --wandb \
    --wandb_project q-pretraining
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