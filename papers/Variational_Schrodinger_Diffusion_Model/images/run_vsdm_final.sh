
# ----------------------------------------------------- VSDM 

# Train VSDM with diffusion initialization on 1 GPU;
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_vsdm.py --outdir=/storage/luoweijian/logs/vsdm-logs/240202 --data=/luoweijian/data/datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp --batch=256 --lr 0.00001 --fp16=0 --transfer='https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl' --tick 10 --snap 5 --method 'vsdm'

# Train VSDM w/o diffusion initialization on 1 GPU;
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_vsdm.py --outdir=/storage/luoweijian/logs/vsdm-logs/240202 --data=/luoweijian/data/datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp --batch=256 --lr 0.00001 --fp16=0 --tick 10 --snap 5 --method 'vsdm'

# Generate 50k image from VSDM
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 generate_vsdm.py --outdir=fid-tmp --seeds=0-99 --batch 100 --steps 200 --solver euler --subdirs --network='/storage/luoweijian/logs/vsdm-logs/240202/00004-cifar10-32x32-uncond-ddpmpp-edm-gpus1-batch256-fp32-lr1e-05-method-vsdm-pretrainno/network-snapshot-000051.pkl'

# Calculate FID
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz


# ----------------------------------------------------- FBSDE

# Train FBSDE with diffusion initialization on 1 GPU;
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_fbsde.py --outdir=/storage/luoweijian/logs/fbsde/240202 --data=/luoweijian/data/datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp --batch=64 --lr 0.00001 --fp16=0 --transfer='https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl' --tick 10 --snap 10

# Train FBSDE w/o diffusion initialization on 1 GPU;
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_fbsde.py --outdir=/storage/luoweijian/logs/fbsde/240202 --data=/luoweijian/data/datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp --batch=64 --lr 0.00001 --fp16=0 --tick 10 --snap 10

# Calculate FID
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz