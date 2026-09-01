

source activate promptsrc2 #change to your envname


for dataset in   ucf101 ; do #  imagenet caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
    
    MYCFG=vit_b16_c2_ep20_batch4_4+4ctx-testenv
    DIV=10.00
    bash scripts/promptot/base2new_train.sh $dataset 2 $DIV $MYCFG
    bash scripts/promptot/reproduce_base2novel_setting_testenv.sh $dataset 2 output${DIV}/base2new/train_base/$dataset/shots_16/PromptOT/${MYCFG} $MYCFG $DIV
    
    

   

done


# extract results to csv
python extract_log_3.py --save_patch output${DIV} 