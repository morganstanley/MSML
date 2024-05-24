#!/bin/bash
echo -n "Experiment: ";
read -r exp;
echo -n "cuda: ";
read -r i;
if [[ $exp == *"simulation"* ]]; 
then
    if [[ $exp == *"noise"* ]] || [[ $exp == *"grid"* ]] || [[ $exp == *"compare"* ]] || [[ $exp == *"flow"* ]] || [[ $exp == *"linear"* ]] || [[ $exp == *"brownian"* ]] || [[ $exp == *"irreg"* ]];
    then 
    for file in $exp/*.yaml; do 
        screen_name=${file////-}
        echo "$screen_name";
        screen -S $screen_name -dm bash -c "python3 train.py -f $file -d cuda:$i -e $exp; sleep 300";
        i=$((i+1))
        if [[ $i == 4 ]] 
        then
            i=0
        fi
    done
    else
    echo -n "npoints: ";
    read -r npoints;
    for dir in $exp/*/; do
        i=0
        [ -L "${dir%/}" ]
        if [[ $dir == *"$npoints"* ]]
        then
        for file in $dir/*.yaml; do 
            screen_name=${file////-}
            echo "$screen_name";
            screen -S $screen_name -dm bash -c "python3 train.py -f $file -d cuda:$i -e $dir; sleep 300";
            i=$((i+1))
            if [[ $i == 4 ]] 
            then
                i=0
            fi
            done
        fi
    done
    fi
else
    if [[ $exp == *"FK"* ]];
    then
        for file in $exp/*.yaml; do 
            screen_name=${file////-}
            echo "$screen_name";
            screen -S $screen_name -dm bash -c "python3 mean_train_fk.py -f $file -d cuda:$i -e $exp; sleep 300";
            i=$((i+1))
            if [[ $i == 4 ]] 
            then
                i=0
            fi
        done
    else
        for file in $exp/*.yaml; do 
            screen_name=${file////-}
            echo "$screen_name";
            screen -S $screen_name -dm bash -c "python3 train.py -f $file -d cuda:$i -e $exp; sleep 300";
            i=$((i+1))
            if [[ $i == 4 ]] 
            then
                i=0
            fi
        done
    fi
fi