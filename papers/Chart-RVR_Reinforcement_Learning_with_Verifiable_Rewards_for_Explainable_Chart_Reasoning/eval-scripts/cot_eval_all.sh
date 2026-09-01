#!/bin/bash
#SBATCH --job-name=cot-hard-eval # name of the job
#SBATCH --output=logs/%x_%j.out         # log output to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err          # error logs to logs/jobname_jobid.err
#SBATCH --ntasks=1                      # number of tasks
#SBATCH --nodes=1                       # stay on 1 node
#SBATCH --gres=gpu:1                    # request 2 GPUs
#SBATCH --cpus-per-task=8              # adjust based on your workload
#SBATCH --mem=32G                       # adjust memory
#SBATCH --time=48:00:00                 # max runtime (hh:mm:ss)
#SBATCH --array=0-5

export SHELL=/bin/bash
source /mnt/home/sanchit/miniconda3/bin/activate
conda activate chartrl

cd /mnt/home/sanchit/rl-chart 

case $SLURM_ARRAY_TASK_ID in
  0)
    CMD="python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name chartqa-src --cot True"
    ;;
  1)
   CMD="python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name evochart --cot True"
    ;;
  2)
    CMD="python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name chartqapro --cot True"
    ;;
  3)
    CMD="python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name plotqa --cot True"
    ;;
  4) 
  CMD="python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name chartfc --cot True"
    ;;
  5) 
  CMD="python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name chartbench --cot True"
    ;;
    

esac
echo "Running: $CMD"
$CMD
