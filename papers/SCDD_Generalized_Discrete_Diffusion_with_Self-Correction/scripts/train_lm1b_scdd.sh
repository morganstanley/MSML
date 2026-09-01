#!/bin/bash
#SBATCH -J job_name                     # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

export HF_HOME=/path/to/scratch/.cache/huggingface
export HF_HUB_CACHE=/path/to/scratch/.cache/huggingface/hub
export HF_DATASETS_CACHE=/path/to/scratch/.cache/huggingface/datasets
# Optional but usually nice:
export TRANSFORMERS_CACHE=/path/to/scratch/.cache/huggingface/transformers

T=1000
BATCH_SIZE=64
EVAL_BATCH_SIZE=64
GLOBAL_BATCH_SIZE=512
SEED=512
SAMPLING_EPS=1e-3
LR=5e-4
DATA=lm1b-gpt2
MODEL=small
LENGTH=128
STEPS=500000

cd /path/to/project/

module load conda
conda activate scdd

srun python -u -m main \
  T=${T} \
  seed=${SEED} \
  model=${MODEL} \
  lr_scheduler=cosine_decay_warmup \
  data=${DATA} \
  loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${EVAL_BATCH_SIZE} \
  wandb.name=${MODEL}_scdd_${DATA}_${LR} \
  parameterization=scdd \
  model.length=${LENGTH} \
  eval.compute_generative_perplexity=True \
  sampling.steps=1024 \
  sampling.predictor=scdd \
  training.sampling_eps=${SAMPLING_EPS} \
  trainer.limit_train_batches=1.0 \
  trainer.limit_val_batches=1.0 \
  trainer.max_steps=${STEPS} \
  trainer.val_check_interval=1.0 \
  optim.lr=${LR} \
  forward=mix \
  forward.ratio=0.1 \
