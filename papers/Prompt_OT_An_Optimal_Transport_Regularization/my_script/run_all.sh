#!/bin/bash

#SBATCH --job-name ot2-1
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 120gb
#SBATCH --time 40:00:00
#SBATCH --gpus-per-node a100:1

module load anaconda3/2023.09-0  cuda/11.8.0 
source activate promptsrc

DIV=10.0
rm -rf output${DIV}



for dataset in imagenet caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat; do #  
    
    MYCFG=vit_b16_c2_ep20_batch4_4+4ctx-Copy3
    bash scripts/promptot/base2new_train.sh $dataset 1 $DIV $MYCFG
    bash scripts/promptot/reproduce_base2novel_setting.sh $dataset 1 output${DIV}/base2new/train_base/$dataset/shots_16/PromptOT/${MYCFG} $MYCFG $DIV
    
    MYCFG=vit_b16_c2_ep20_batch4_4+4ctx-Copy3
    bash scripts/promptot/base2new_train.sh $dataset 2 $DIV $MYCFG
    bash scripts/promptot/reproduce_base2novel_setting.sh $dataset 2 output${DIV}/base2new/train_base/$dataset/shots_16/PromptOT/${MYCFG} $MYCFG $DIV
    
    MYCFG=vit_b16_c2_ep20_batch4_4+4ctx-Copy3
    bash scripts/promptot/base2new_train.sh $dataset 3 $DIV $MYCFG
    bash scripts/promptot/reproduce_base2novel_setting.sh $dataset 3 output${DIV}/base2new/train_base/$dataset/shots_16/PromptOT/${MYCFG} $MYCFG $DIV
done    
    python extract_log_3.py --save_patch output${DIV} --output_path output_csv
    # rm -rf output${DIV}
    
 