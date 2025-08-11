#!/bin/bash

# Pretraining experiments for different model sizes using kdb_license_processed data
# IMPORTANT: Update the data paths below with your actual data file locations before running

# This script runs full fine-tuning on the licensed dataset for 5 different model sizes
# Replace /your/data/... with actual paths to your training and evaluation data files

echo "Starting pretraining experiments for 5 model sizes..."

# 1.5B model
echo "Training 1.5B model..."
python run_pretraining.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --data_type licensed \
  --training_method full \
  --max_steps 800 \
  --save_steps 100 \
  --learning_rate 1e-5 \
  --output_dir final_outputs/1.5b \
  --wandb

# 3B model
echo "Training 3B model..."
python run_pretraining.py \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --data_type licensed \
  --training_method full \
  --max_steps 800 \
  --save_steps 100 \
  --learning_rate 1e-5 \
  --output_dir final_outputs/3b \
  --wandb

# 7B model
echo "Training 7B model..."
python run_pretraining.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --data_type licensed \
  --training_method full \
  --max_steps 800 \
  --save_steps 100 \
  --learning_rate 1e-5 \
  --output_dir final_outputs/7b \
  --wandb

# 14B model (requires accelerate for multi-GPU)
echo "Training 14B model..."
accelerate launch --num_processes=4 train_pretrain.py \
  --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
  --train_file /your/data/train_license_kdbsite.jsonl \
  --eval_file /your/data/eval_license_kdbsite.jsonl \
  --max_steps 800 \
  --save_steps 80 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --output_dir final_outputs/14b \
  --report_to wandb \
  --wandb_project q-pretraining

# 32B model (requires accelerate for multi-GPU)
echo "Training 32B model..."
accelerate launch --num_processes=4 --main_process_port=29502 train_pretrain.py \
  --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
  --train_file /your/data/train_license_kdbsite.jsonl \
  --eval_file /your/data/eval_license_kdbsite.jsonl \
  --max_steps 800 \
  --save_steps 80 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --output_dir final_outputs/32b \
  --report_to wandb \
  --wandb_project q-pretraining

echo "All pretraining experiments completed!"