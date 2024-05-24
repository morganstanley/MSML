#!/bin/bash
echo -n "Experiment: ";
read -r exp;
echo -n "cuda: ";
read -r i;
for file in $exp/*.yaml; do 
    screen_name=${file////-}
    echo "$screen_name";
    screen -S $screen_name -dm bash -c "python train_deepAR.py -f $file -d cuda:$i -e $exp; sleep 300";
    i=$((i+1));
done