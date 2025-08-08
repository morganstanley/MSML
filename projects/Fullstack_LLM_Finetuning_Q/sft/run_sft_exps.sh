#!/bin/bash

# SFT Training experiments for different model sizes
# IMPORTANT: Update the model paths below with your best pretrained checkpoints before running

# Set model directory path - customize these paths with your best pretrained checkpoints
# Replace /your/best/pretrained/[model_size]/checkpoint with actual paths to your pretrained models

echo "Starting SFT training experiments for 5 model sizes..."

# 1.5B model
echo "Training 1.5B model..."
python run_sft.py \
  --base_model /your/best/pretrained/1.5b/checkpoint \
  --learning_rate 2e-5 \
  --max_steps 1000 \
  --output_dir final_outputs/1.5b/ \
  --experiment_name q-sft

# 3B model
echo "Training 3B model..."
python run_sft.py \
  --base_model /your/best/pretrained/3b/checkpoint \
  --learning_rate 2e-5 \
  --max_steps 1000 \
  --output_dir final_outputs/3b/ \
  --experiment_name q-sft

# 7B model
echo "Training 7B model..."
python run_sft.py \
  --base_model /your/best/pretrained/7b/checkpoint \
  --learning_rate 2e-5 \
  --max_steps 1000 \
  --output_dir final_outputs/7b/ \
  --experiment_name q-sft

# 14B model (requires accelerate for multi-GPU)
echo "Training 14B model..."
accelerate launch --num_processes=4 --main_process_port=29501 run_sft.py \
  --base_model /your/best/pretrained/14b/checkpoint \
  --learning_rate 2e-5 \
  --max_steps 1000 \
  --output_dir final_outputs/14b/ \
  --experiment_name q-sft

# 32B model (requires accelerate for multi-GPU)
echo "Training 32B model..."
accelerate launch --num_processes=4 --main_process_port=29502 run_sft.py \
  --base_model /your/best/pretrained/32b/checkpoint \
  --learning_rate 2e-5 \
  --max_steps 1000 \
  --output_dir final_outputs/32b/ \
  --experiment_name q-sft

echo "All SFT training experiments completed!"