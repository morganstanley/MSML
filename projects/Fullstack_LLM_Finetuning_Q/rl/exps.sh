#!/bin/bash

# RL Training experiments for different model sizes
# IMPORTANT: Update the model paths below with your best SFT checkpoints before running

# Set model directory path - customize these paths with your best SFT checkpoints
# Replace /your/best/sft/[model_size]/checkpoint with actual paths to your trained models

echo "Starting RL training experiments for 5 model sizes..."

# 1.5B model (non-reasoning)
echo "Training 1.5B model..."
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

# 3B model (non-reasoning)
echo "Training 3B model..."
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml rl_trainer.py \
  --model /your/best/sft/3b/checkpoint \
  --use_vllm \
  --output_dir final_outputs/3b/non_reasoning/ \
  --learning_rate 2e-6 \
  --simple_eval_problems 15 \
  --repeat_eval_problems 4 \
  --eval_steps 25 \
  --use_wandb \
  --wandb_name 3b_non_reason \
  --generation_temp .8

# 7B model (non-reasoning)
echo "Training 7B model..."
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml rl_trainer.py \
  --model /your/best/sft/7b/checkpoint \
  --use_vllm \
  --output_dir final_outputs/7b/non_reasoning/ \
  --learning_rate 2e-6 \
  --simple_eval_problems 15 \
  --repeat_eval_problems 4 \
  --eval_steps 25 \
  --use_wandb \
  --wandb_name 7b_non_reason \
  --generation_temp .8

# 14B model (reasoning)
echo "Training 14B model..."
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml rl_trainer.py \
  --model /your/best/sft/14b/checkpoint \
  --use_vllm \
  --output_dir final_outputs/14b/reasoning/ \
  --learning_rate 2e-6 \
  --simple_eval_problems 15 \
  --repeat_eval_problems 4 \
  --eval_steps 25 \
  --use_wandb \
  --wandb_name 14b_reason \
  --generation_temp .8 \
  --use_reasoning

# 32B model (reasoning)
echo "Training 32B model..."
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml rl_trainer.py \
  --model /your/best/sft/32b/checkpoint \
  --use_vllm \
  --output_dir final_outputs/32b/reasoning/ \
  --learning_rate 2e-6 \
  --simple_eval_problems 15 \
  --repeat_eval_problems 4 \
  --eval_steps 25 \
  --use_wandb \
  --wandb_name 32b_reason \
  --generation_temp .8 \
  --use_reasoning

echo "All RL training experiments completed!"







