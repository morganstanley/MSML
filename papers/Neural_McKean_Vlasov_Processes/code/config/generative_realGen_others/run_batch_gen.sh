#!/bin/bash
i=0
for file in power/*.yaml; do 
    screen_name=${file////-}
    echo "$screen_name";
    screen -S $screen_name -dm bash -c "python train_gen.py --f $file --d cuda:$i --e power; sleep 300";
    i=$((i+1));
    if [[ $i == 4 ]] 
        then
            i=0
    fi
done

for file in gas/*.yaml; do 
    screen_name=${file////-}
    echo "$screen_name";
    screen -S $screen_name -dm bash -c "python train_gen.py --f $file --d cuda:$i --e gas; sleep 300";
    i=$((i+1));
    if [[ $i == 4 ]] 
        then
            i=0
    fi
done

for file in cortex/*.yaml; do 
    screen_name=${file////-}
    echo "$screen_name";
    screen -S $screen_name -dm bash -c "python train_gen.py --f $file --d cuda:$i --e cortex; sleep 300";
    i=$((i+1));
    if [[ $i == 4 ]] 
        then
            i=0
    fi
done

for file in hepmass/*.yaml; do 
    screen_name=${file////-}
    echo "$screen_name";
    screen -S $screen_name -dm bash -c "python train_gen.py --f $file --d cuda:$i --e hepmass; sleep 300";
    i=$((i+1));
    if [[ $i == 4 ]] 
        then
            i=0
    fi
done

for file in miniboone/*.yaml; do 
    screen_name=${file////-}
    echo "$screen_name";
    screen -S $screen_name -dm bash -c "python train_gen.py --f $file --d cuda:$i --e miniboone; sleep 300";
    i=$((i+1));
    if [[ $i == 4 ]] 
        then
            i=0
    fi
done

python score_based_sde/train_score_based.py