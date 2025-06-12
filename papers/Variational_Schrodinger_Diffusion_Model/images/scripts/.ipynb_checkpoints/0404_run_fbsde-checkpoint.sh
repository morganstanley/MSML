#!/bin/bash

## task command
# bash /luoweijian/code/edm_diffinstruct_dev/scripts/run_di_0403.sh

# cp -r /luoweijian/data/envs/edm_v100 /opt/conda/envs/

PYTHON_SCRIPT_PATH="/luoweijian/code/edm_vsdm_dev/train_vsdm.py"
OUTPUT_DIR="/luoweijian/logs/fbsde/0404/from-edm"

/luoweijian/data/envs/edm_v100/bin/torchrun --standalone --nproc_per_node=1 --master_port=25678 "$PYTHON_SCRIPT_PATH" --outdir="$OUTPUT_DIR" --data=/luoweijian/data/datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp --batch=64 --lr 0.001 --fp16=0 --transfer='/luoweijian/data/downloads/pretrained_edm/edm-cifar10-32x32-uncond-vp.pkl' --tick 10 --snap 10 --method 'fbsde'