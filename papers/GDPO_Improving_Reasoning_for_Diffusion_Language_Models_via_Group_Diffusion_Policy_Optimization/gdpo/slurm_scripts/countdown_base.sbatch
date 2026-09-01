#!/bin/bash
#SBATCH --output=slurm/%A/slurm.out
#SBATCH --job-name=countdown-base
#SBATCH --time=48:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --open-mode=append

#SBATCH --nodes=1
#SBATCH --partition=hpc-mid
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=50



export LOGDIR=checkpoints/slurm/${SLURM_JOB_ID}
mkdir -p $LOGDIR
echo $LOGDIR


DATASET="countdown" # Options: countdown, sudoku, math, gsm8k, mbpp
ESTIMATOR="${ESTIMATOR:-gauss-3}" # Options: mc, gauss-(1-5)
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
NUM_ITER=12
RUN_NAME=${DATASET}_base_bs12_${ESTIMATOR}

cd /mnt/home/krojas/repos/GSPO-Diffusion/gdpo/slurm_scripts

# Initialize conda for batch jobs
eval "$(conda shell.bash hook)"
conda activate gdpo

srun --output ${LOGDIR}/grpo_%j.out \
    accelerate launch \
        --config_file ../accelerate.yaml \
        --main_process_port 12363 ../gdpo_train.py \
        --config ./train.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --logp_estimator $ESTIMATOR \
        --output_dir checkpoints/$RUN_NAME