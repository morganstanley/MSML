#!/bin/bash

# RL Training experiments for different model sizes
# IMPORTANT: DO NOT RUN THIS SCRIPT DIRECTLY!
#
# This script contains commands for both vLLM server and RL training.
# You need TWO separate terminals:
#
# TERMINAL 1: Run the vLLM server commands first (one at a time)
# TERMINAL 2: Once the vLLM server is live, run the RL training commands
#
# Instructions:
# 1. Copy the vLLM server command for your desired model size to Terminal 1
# 2. Wait for the server to start
# 3. Copy the corresponding RL training command to Terminal 2
# 4. Repeat for each model size you want to train
#
# IMPORTANT: Update the model paths below with your best SFT checkpoints before running
# Replace /your/best/sft/[model_size]/checkpoint with actual paths to your trained models

# 1.5B model (non-reasoning)
# vLLM Server Command:
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /your/best/sft/1.5b/checkpoint --max-model-len 8192

# RL Training Command:
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
# vLLM Server Command:
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /your/best/sft/3b/checkpoint --max-model-len 8192

# RL Training Command:
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
# vLLM Server Command:
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /your/best/sft/7b/checkpoint --max-model-len 8192

# RL Training Command:
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
# vLLM Server Command:
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /your/best/sft/14b/checkpoint --max-model-len 8192

# RL Training Command:
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
# vLLM Server Command:
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /your/best/sft/32b/checkpoint --max-model-len 8192

# RL Training Command:
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