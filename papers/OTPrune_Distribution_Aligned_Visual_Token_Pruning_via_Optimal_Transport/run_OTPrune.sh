#!bin/bash




export HF_HOME=../hf_cache
export TRANSFORMERS_CACHE=../hf_cache
export HUGGINGFACE_HUB_TOKEN= $$$your_hf_token
set -x

source activate divprune

PAPER_TABLE=coco2017_cap_val,flickr30k_test,gqa,mmbench_en_dev,mme,mmmu_val,nocaps_val,ok_vqa_val2014,pope,scienceqa_img,seedbench

LOG_DIR=./logs_final


# #To run other models use: liuhaotian/llava-v1.5-13b, liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.5-7b
# CUDA_VISIBLE_DEVICES=0,1,2,3 BASELINE=OURS LAYER_INDEX=0 SUBSET_RATIO=0.098 python3 -m accelerate.commands.launch \
#     --num_processes=4 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.5-13b" \
#     --tasks $PAPER_TABLE \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix $RUN_NAME \
#     --output_path $LOG_DIR/$RUN_NAME


RUN_NAME=otprune_llava_1.5_7b #divprune_llava_1.6_7b
#To run other models use: liuhaotian/llava-v1.5-13b, liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.5-7b
CUDA_VISIBLE_DEVICES=0 BASELINE=OURS LAYER_INDEX=0 SUBSET_RATIO=0.098 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks $PAPER_TABLE \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path $LOG_DIR/$RUN_NAME


    
RUN_NAME=otprune_llava_1.6_7b #divprune_llava_1.6_7b
#To run other models use: liuhaotian/llava-v1.5-13b, liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.5-7b
CUDA_VISIBLE_DEVICES=0 BASELINE=OURS LAYER_INDEX=0 SUBSET_RATIO=0.098 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks $PAPER_TABLE \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path $LOG_DIR/$RUN_NAME
    
    
    
RUN_NAME=otprune_llava_1.5_13b #divprune_llava_1.6_7b
#To run other models use: liuhaotian/llava-v1.5-13b, liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.5-7b
CUDA_VISIBLE_DEVICES=0 BASELINE=OURS LAYER_INDEX=0 SUBSET_RATIO=0.098 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-13b" \
    --tasks $PAPER_TABLE \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $RUN_NAME \
    --output_path $LOG_DIR/$RUN_NAME