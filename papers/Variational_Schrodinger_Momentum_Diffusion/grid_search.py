#!/usr/bin/python 
import random
import os
import time
import sys
 
secure_random = random.SystemRandom()


gpu_id = sys.argv[1]

for _ in range(2):
    seed = str(random.randint(1, 10**5))
    lr_f = secure_random.choice([0.3])
    lr_gamma_f = secure_random.choice([0.3])
    lr_b = secure_random.choice([2e-3])
    beta = secure_random.choice([3, 4])
    num_itr = secure_random.choice([25])

    os.system(f'python main.py --problem-name checkerboard --num-itr-dsm 100 --num-itr {num_itr} --num-stage 50 --lr-f {lr_f} --lr-b {lr_b} --lr-gamma-f {lr_gamma_f} --forward-net LinearStatic \
               --dir check_6x_vsdm_{beta} --x-scalar 6 --beta-max {beta} --snapshot 1 --samp-bs 20000 --seed {seed} --gpu {gpu_id} > logs/log_check_num_itr_{num_itr}_beta_{beta}_lr_f_{lr_f}_b_{lr_b}_gamma_{lr_gamma_f}_seed_{seed}')
