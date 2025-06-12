import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from lib.plot_utils import *


data = 'circles'
df = {'8gaussians': '8g', 'rings': 'r', 'pinwheel': 'pw_x6', '2spirals': '2s', 'circles': 'cc', 'swissroll': 'sr', 'checkerboard': 'cb_x4'}
save = 'X1_mean_std_h10'
# save = 'x4_10h'
c_list = ['k', 'b', 'g', 'r', 'c', 'y']

labels = ['FFJORD (BF)', 'FFJORD', "FFJORD++ ($L_s$ = 1)", "FFJORD++", "FFJORD++ ($L_s$ = 20)", "FFJORD++ ($L_s$ = 50)"]

log_path = []

test_metrics = []
start = 0
max_iter = 200
for i in range(len(log_path)):
    test_loss = get_test_loss(log_path[i])
    test_metrics.append(test_loss[start: min(max_iter, len(test_loss))])

plt.figure(figsize=(6, 4))

plot_mean_std(test_metrics[:3], labels[0], color='#0960BD', max_iter=max_iter, id=0)
plot_mean_std(test_metrics[3:6], labels[1], color='#FF6B6B', max_iter=max_iter, id=0)
plot_mean_std(test_metrics[6:9], labels[3], color='#399918', max_iter=max_iter, id=0)
# 36AE7C
# plot_mean_std(test_metrics[:3], labels[2], color='#125B9A', max_iter=max_iter, id=0)
# plot_mean_std(test_metrics[3:6], labels[3], color='#399918', max_iter=max_iter, id=0)
# plot_mean_std(test_metrics[6:9], labels[4], color='#FFB000', max_iter=max_iter, id=0)
# plot_mean_std(test_metrics[9:12], labels[5], color='#F96666', max_iter=max_iter, id=5)
# save1 = 'mean_std_h10'E7B10A
# save_title = save1.replace('_', '-')
save_title = save.split('_')[0]
print(save_title)

plt.title(f'{data}-{save_title}', fontsize=14, fontweight='bold')
plt.xlabel('Iter(100x)', fontsize=12)
plt.ylabel('Test loss', fontsize=12)
offset = start
x_t = [0, 50-offset, 100-offset, 150-offset, 200-offset,]
x_t_modified = [0 + offset, 50, 100, 150, 200, ]
plt.xticks(x_t, x_t_modified, fontsize=12)
plt.yticks(fontsize=12) 
# [3.5, 3.7, 3.9, 4.1, ] for cb
# [4.2, 4.4, 4.6, 4.8] for cb_x2
# [4.9, 5.1, 5.3, 5.5, 5.7] for cb_x4
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.grid(True)
plt.legend(fontsize=14)
plt.savefig(f'./final_vis/{data}/test_loss_{save}.png')
plt.show()
#wolegend

# cc x2 (2, 182, 182), x4 (10, 190, 190)




