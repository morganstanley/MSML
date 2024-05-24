from scipy.io import loadmat
import numpy as np
import glob
import pandas as pd
import os 
from torchvision.datasets import USPS, MNIST
import torch
# For RNA data
import pickle

def get_none_simu(path, 
                  dataset_type,
                  seed = 0,
                  split_type=None,
                  **params
                 ):
    
    """
    Retrieve Real data, and split them base on split_type and split_size
    """
    np.random.seed(seed)
    N0     = params.setdefault('N0',64)
    N      = params.setdefault('N',64)
    bb     = params.setdefault('bb',True)
    
    N_train = params.setdefault('N_train', 100)
    N_val   = params.setdefault('N_val', 500)
    N_test  = params.setdefault('N_test', 500)
    
    if "realGen" in path:
        # Adapted from https://github.com/dargilboa/mdma/tree/8ed4afc731936f695d6e320d804e5157e8eb8a71/experiments/UCI
        tot_N = N_train + N_val + N_test
        if 'Cortex' in dataset_type:
            data = pd.read_csv(path + dataset_type).values[:,2:78].astype(float)
            data = np.nan_to_num(data)
        elif 'miniboone' in dataset_type:
            data = np.load(path+dataset_type)
        elif 'ethylene_CO' in dataset_type:
            data = pd.read_pickle(path + dataset_type)
            data.drop("Meth", axis=1, inplace=True)
            data.drop("Eth", axis=1, inplace=True)
            data.drop("Time", axis=1, inplace=True)
            C = data.corr()
            A = C > 0.98
            B = A.values.sum(axis=1)
            while np.any(B > 1):
                col_to_remove = np.where(B > 1)[0][0]
                col_name = data.columns[col_to_remove]
                data.drop(col_name, axis=1, inplace=True)
                C = data.corr()
                A = C > 0.98
                B = A.values.sum(axis=1)
            data = (data - data.mean()) / data.std()
            data = data.values.astype(float)
        elif 'power' in dataset_type:
            data = np.load(path+dataset_type)
            N = data.shape[0]
            data = np.delete(data, 3, axis=1)
            data = np.delete(data, 1, axis=1)
            voltage_noise = 0.01 * np.random.rand(N, 1)
            gap_noise = 0.001 * np.random.rand(N, 1)
            sm_noise = np.random.rand(N, 3)
            time_noise = np.zeros((N, 1))
            noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
            data = data + noise
            data = data.astype(float)
        elif 'hepmass' in dataset_type:
            data = pd.read_csv(filepath_or_buffer=os.path.join(path,dataset_type),
                           index_col=False)
            data = data[data[data.columns[0]] == 1]
            data = data.drop(data.columns[0], axis=1)
            data = data.values.astype(float)
            
        # subsampling
        index = np.random.choice(list(range(data.shape[0])), np.min([tot_N, data.shape[0]]), False)
        data = data[index]
    
    # special case for real data in generative modeling
    data_train = data[0:N_train]
    data_validate = data[N_train:N_train + N_val]
    if "hepmass" in dataset_type:
        data_test = pd.read_csv(filepath_or_buffer=os.path.join(path,"hepmass_1000_test.csv"),
                                    index_col=False)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)
        data_test = data_test.values.astype(float)
        np.random.shuffle(data_test)
        data_test = data_test[:N_test]
    else:
        data_test = data[N_train + N_val:np.min([data.shape[0], N_train + N_val + N_test])]
    # normalize
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    xs_train = (data_train - mu) / s
    xs_val = (data_validate - mu) / s
    xs_test = (data_test - mu) / s

    return xs_train, xs_val, xs_test
    
def load_data(cfg, seed):
    dataset_params = cfg['dataset']
    batch_size = cfg['loader']['batch_size']
    
    train_loader = None
    val_loader   = None
    test_loader  = None
    
    data = get_none_simu(dataset_type=dataset_params["dataset_type"], path=dataset_params["path"], seed=seed)
    
    xs_train, xs_val, xs_test = data
    
    train_tensor = torch.from_numpy(xs_train)
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_tensor = torch.from_numpy(xs_val)
    val_dataset = torch.utils.data.TensorDataset(val_tensor, val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    test_tensor = torch.from_numpy(xs_test)
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader