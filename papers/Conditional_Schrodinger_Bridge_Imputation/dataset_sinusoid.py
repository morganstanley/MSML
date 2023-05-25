import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

"""
Example:

import dataset_sinusoid
train_dataset, valid_dataset, test_dataset = dataset_sinusoid.get_dataloader(
    opt.sinusoid_dataset_path, eval_length=50, batch_size=None, return_dataset=True, seed=1)

train_dataset, valid_dataset, test_dataset = dataset_sinusoid.get_dataloader(
    opt.sinusoid_dataset_path, eval_length=50, batch_size=64, return_dataset=False, seed=1)
"""


class SIMU_Dataset(Dataset):
    def __init__(self, eval_length, data_dict, use_index_list, seed=0):
        np.random.seed(seed)  # seed for ground truth choice
        '''
        'data_all':noise_data,# time series
        'data_shape':(N,K,L),
        'time_step':T_array,# Time step
        'mask_obs':mask_obs, 
        'mask_gt':mask_gt   
        '''
        self.observed_values= data_dict.item().get('data_all')
        self.observed_masks = data_dict.item().get('mask_obs')
        self.gt_masks = data_dict.item().get('mask_gt')
        N,L,K = self.observed_values.shape
        self.use_index_list = use_index_list
        self.eval_length = L

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),# ? use the raw values?
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(
        file_path,
        eval_length,
        seed=1,
        batch_size=16,
        return_dataset=False):

    # only to obtain total length of dataset
    # dataset = SIMU_Dataset(missing_ratio=missing_ratio, seed=seed)
    # data_dict = np.load('data/schrodinger_bridge/simu_data_noise_new.npy', allow_pickle=True)
    data_dict = np.load(file_path, allow_pickle=True)
    obs_vals= data_dict.item().get('data_all')
    num_samples, _, _ = obs_vals.shape
    print('Dataset Total obs_vals.shape', obs_vals.shape)
    indlist = np.arange(num_samples)

    np.random.seed(seed)
    np.random.shuffle(indlist)
    # 5-fold test
    start = 0
    end = int(0.05 * num_samples)
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    num_train = (int)(num_samples * 0.8)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]
    # print('train/test/val index', train_index, test_index, valid_index)

    train_dataset = SIMU_Dataset(
        eval_length, data_dict, use_index_list=train_index, seed=seed)
    valid_dataset = SIMU_Dataset(
        eval_length, data_dict, use_index_list=valid_index, seed=seed)
    test_dataset = SIMU_Dataset(
        eval_length, data_dict, use_index_list=test_index, seed=seed)
    print('train/test/val num samples', len(train_dataset), len(valid_dataset), len(test_dataset))

    if return_dataset:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
        return train_loader, valid_loader, test_loader





