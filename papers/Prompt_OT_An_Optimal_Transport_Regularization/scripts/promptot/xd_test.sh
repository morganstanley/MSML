#!/bin/bash


# custom config
DATA="../imagedataset"
TRAINER=PromptOT

DATASET=$1
SEED=$2
# WEIGHTSPATH=$3


CFG=$3
DIV=$4

SHOTS=16


DIR=output${DIV}/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output${DIV}/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch 10 \
    --eval-only
fi