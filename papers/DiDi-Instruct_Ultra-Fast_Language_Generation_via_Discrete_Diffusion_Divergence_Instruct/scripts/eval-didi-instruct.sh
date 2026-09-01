module load conda/2024.09
module load gcc/14.1.0
module load cuda/12.6.0
module load jupyter/1.1.1

conda activate mask_model

checkpoint_path=/your_checkpoint_path/didi-instruct.ckpt
clear

steps=16
batch=20
n_batch=2  # total samples = batch * n_batch
echo "----------------------------------------------------------------"
echo "Running with params:"
echo "  steps       = $steps"
echo "  checkpoint  = $checkpoint_path"
echo "----------------------------------------------------------------"

python3 main.py \
    mode=sample_eval \
    loader.batch_size="$batch" \
    loader.eval_batch_size="$batch" \
    model=small \
    algo=didi_instruct \
    eval.checkpoint_path="$checkpoint_path" \
    sampling.steps="$steps" \
    sampling.num_sample_batches=$n_batch \
    eval.generate_samples=true \
    sampling.predictor=guided \
    sampling.noise_removal=ancestral
