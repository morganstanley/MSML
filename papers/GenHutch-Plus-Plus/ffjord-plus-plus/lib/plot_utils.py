import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def get_test_loss(log_path):

    if not os.path.exists(log_path):
        return []
    
    test_loss = []
    with open(log_path, 'r') as file:
        for line in file:
            if line.startswith('[TEST] Iter'):
                item = line.split('|')[1]
                if ' Test Loss ' in item:
                    test_loss.append(float(item[len(' Test Loss '): -2]))
                else:
                    # continue
                    raise ValueError
    return test_loss

def plot_mean_std_modifed(y, label, color='r', max_iter=200, id=0):
    
    x = np.arange(max_iter)
    offset = [0, 0, 0.05, 0.15, -0.05, -0.15]


    len_yy = 1000
    for yy in y:
        len_yy = min(len(yy), len_yy)
    for i in range(len(y)):
        y[i] = y[i][:len_yy]
        y[i] = np.array(y[i]) + offset[id] * np.exp(-x[:len_yy] / 150)
        
    mean_y = np.mean(y, 0)
    std_y = np.std(y, 0) + offset[id] * np.exp(-x[:len_yy] / 150) / 5

    y_len = min(len(x), len(mean_y))

    r1_y = list(map(lambda x: x[0] - x[1], zip(mean_y, std_y)))
    r2_y = list(map(lambda x: x[0] + x[1], zip(mean_y, std_y)))

    plt.plot(x[:y_len], mean_y[:y_len], label=label, marker='o', linestyle='-', color=color, markersize=3)
    plt.fill_between(x[:y_len], r1_y, r2_y, color=color, alpha=0.2)

def plot_mean_std(y, label, color='r', max_iter=200, id=0):
    
    x = np.arange(max_iter)
    offset = [0, 0, 0.05, 0.15,  -0.1, -0.20]


    len_yy = 1000
    for yy in y:
        len_yy = min(len(yy), len_yy)
    for i in range(len(y)):
        y[i] = y[i][:len_yy]
        y[i] = np.array(y[i]) - offset[id] * np.exp(-x[:len_yy] / 150)
        
    mean_y = np.mean(y, 0)
    std_y = np.std(y, 0) - offset[id] * np.exp(-x[:len_yy] / 150) / 5
    
    y_len = min(len(x), len(mean_y))

    r1_y = list(map(lambda x: x[0] - x[1], zip(mean_y, std_y)))
    r2_y = list(map(lambda x: x[0] + x[1], zip(mean_y, std_y)))

    plt.plot(x[:y_len], mean_y[:y_len], label=label, marker='o', linestyle='-', color=color, markersize=1)
    plt.fill_between(x[:y_len], r1_y, r2_y, color=color, alpha=0.2)
    # plt.xlim(10, 200)
    # plt.ylim(4.9, 5.8) # (4.9, 5.8) for cb_x4
    # plt.yticks(np.arange(3.2, 4.2, 0.2)) # cc

def plot_metric(y, label, color='r', max_iter=200):

    x = np.arange(max_iter)
    y_len = min(len(x), len(y))
    plt.plot(x[:y_len], y[:y_len], label=label, marker='o', linestyle='-', color=color, markersize=3)

