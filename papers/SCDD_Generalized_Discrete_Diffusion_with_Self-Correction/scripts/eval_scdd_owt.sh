#!/bin/bash
#SBATCH -J job_name                     # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

export HF_HOME=/path/to/scratch/.cache/huggingface
export HF_HUB_CACHE=/path/to/scratch/.cache/huggingface/hub
export HF_DATASETS_CACHE=/path/to/scratch/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/path/to/scratch/.cache/huggingface/transformers

# Settings
CHECKPOINT="/path/to/checkpoints/last.ckpt"
RESULTS_DIR="/path/to/project/eval_results"
mkdir -p $RESULTS_DIR

# Configs matching training
T=1000
SEED=512
DATA=openwebtext-split
MODEL=small
LENGTH=512
RATIO=0.2
GAMMA=1
SAMPLING_EPS=1e-3

cd /path/to/project/

module load conda
conda activate "${CONDA_ENV:-scdd}"

# Sampling steps to evaluate
STEPS_LIST=(32 64 128 256 512 1024 2048)

echo "Saving results to $RESULTS_DIR"

for S in "${STEPS_LIST[@]}"; do
    echo "================================================="
    echo "Evaluating with sampling.steps = $S"
    echo "================================================="
    
    LOG_FILE="$RESULTS_DIR/eval_steps_${S}.log"
    
    # mode=sample_eval: runs generation + gen-ppl only, skips validation NLL (faster).
    # forward.ratio/gamma must match the values used during training.
    # trainer.devices=1: single-GPU eval; increase and adjust batch_size proportionally for multi-GPU.
    # SCDD sampling uses torch.compile acceleration by default.
    python -u -m main \
      mode=sample_eval \
      eval.checkpoint_path="${CHECKPOINT}" \
      T=${T} \
      seed=${SEED} \
      model=${MODEL} \
      data=${DATA} \
      loader.batch_size=16 \
      loader.eval_batch_size=16 \
      trainer.devices=1 \
      parameterization=scdd \
      model.length=${LENGTH} \
      eval.compute_generative_perplexity=True \
      eval.print_changes=False \
      sampling.steps=${S} \
      sampling.predictor=scdd \
      sampling.compile_sampler=True \
      sampling.num_sample_batches=4 \
      sampling.nucleus_p=0.9 \
      forward=mix \
      forward.ratio=${RATIO} \
      forward.gamma=${GAMMA} \
      training.sampling_eps=${SAMPLING_EPS} \
      sampling.generated_seqs_path="${RESULTS_DIR}/generated_seqs_steps_${S}.json" \
      hydra.run.dir="${RESULTS_DIR}/hydra_steps_${S}" \
      > "$LOG_FILE" 2>&1
      
    echo "Finished steps=$S. Log saved to $LOG_FILE"
done

echo "Sweep complete."
