module load conda/2024.09
module load gcc/14.1.0
module load cuda/12.6.0
module load jupyter/1.1.1

conda activate mask_model
clear

# Set environment variables
export HF_HOME="/input_your_path/models/"
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO

# Create absolute log directory
LOG_DIR="/input_your_path/didi-instruct/logs"
mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/distill_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOGFILE}") 2>&1

finetune_path=/input_your_path/mdlm.ckpt

# Run your main Python script
echo "--- GPU Information ---"
nvidia-smi
echo "-----------------------"

# Find a free port
while
  port=$(shuf -n 1 -i 29500-65535)
  netstat -atun | grep -q ":$port "
do
  continue
done
echo "Using master port: $port"

echo "Job started on $(hostname) at $(date)"
echo "Starting Python script..."
START_TIME=$(date +%s)

# Detect available GPUs, default to 2 if nvidia-smi fails
if command -v nvidia-smi &> /dev/null; then
  gpus=$(nvidia-smi --list-gpus | wc -l)
else
  gpus=1
fi
echo "Detected $gpus GPUs available"

batchs=1

torchrun --nproc_per_node=$gpus --master_port=$port -m main \
  training.finetune_path=$finetune_path \
  loader.batch_size=$batchs \
  loader.global_batch_size=$((batchs*gpus)) \
  trainer.val_check_interval=$((batchs*1000)) \
  wandb.name=reinforce-owt-small-$(date +%Y%m%d-%H%M%S)

END_TIME=$(date +%s)
echo "Python script finished."
echo "Job finished at $(date)" 
ELAPSED=$((END_TIME - START_TIME))
printf "Total runtime: %02d:%02d:%02d (hh:mm:ss)\n" \
    $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))
