#!/bin/bash


# custom config
DATA="../imagedataset"
TRAINER=PromptOT

DATASET=$1
SEED=$2
DIV=$3
CFG=$4 
SHOTS=16


DIR=output${DIV}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --div ${DIV} \
    DATASET.NUM_SHOTS ${SHOTS}
fi