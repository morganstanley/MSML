
"""
script for genrating simulated data and converting them into the designated format
v1.1 initial commit Jiahe Lin
"""

## to run: python run_simdata_prep.py --config='configs/simdata.yaml' --ds_strs='ds0,ds1,ds2,ds3,ds4,ds5,ds6'

import os
print(os.getcwd())

import sys
import argparse
import yaml
import pickle
import datetime

import numpy as np
import pandas as pd
from helpers import DataSimulator

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="config file name", default="./configs/simdata.yaml")
parser.add_argument("--ds_strs", help="datasets to be simulated, separated by comma", type=str, default='ds1')
parser.add_argument("--n_total", help="total number of samples to be simulated", type=int, default=150000)

def get_valid_indices_in_range(x, low, high):
    return np.squeeze(np.argwhere((np.squeeze(x) >= low) & (np.squeeze(x) <= high)))

def main():

    global args
    args = parser.parse_args()

    setattr(args,'ds_strs', args.ds_strs.split(','))
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if not os.path.exists(f'{config["defaults"]["data_folder"]}/meta_data'):
        os.makedirs(f'{config["defaults"]["data_folder"]}/meta_data')

    print(f'python={".".join(map(str,sys.version_info[:3]))}')
    print(f'np={np.__version__}')
    print(f'config={args.config}')
        
    train_cut, val_cut = args.n_total - 20000, args.n_total - 5000
    sample_id_pool = np.arange(args.n_total)
    train_sizes, val_size, test_size = config['defaults']['train_sizes'], config['defaults']['val_size'], config['defaults']['test_size']
    default_replicates = config['defaults']['num_of_replicas']
    
    for ds_str in args.ds_strs:
        
        sys.stdout = open(f'{config["defaults"]["data_folder"]}/log_{ds_str}.log', 'w')
        files_saved = []
        
        ds_params, ds_id = config[ds_str], int(ds_str.replace('ds',''))
        num_of_replicates = config[ds_str].get('num_of_replicas', default_replicates)
        print(f'*** generating dataset = {ds_str} with {num_of_replicates} replicates ***')
        
        simulator = DataSimulator(ds_params,seed=ds_params.get('seed',None))
        meta_data = simulator.generate_dataset(ds_id, args.n_total, num_of_replicates = num_of_replicates, reshape=True)
        
        with open(f'{config["defaults"]["data_folder"]}/meta_data/{ds_str}_meta_data.pickle', 'wb') as handle:
            pickle.dump(meta_data, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        ## split into train-val-test
        if not os.path.exists(f'{config["defaults"]["data_folder"]}/{ds_str}'):
            os.mkdir(f'{config["defaults"]["data_folder"]}/{ds_str}')

        for replica_id, (x,y,e) in enumerate(meta_data['data_with_replica']):

            true_mean, true_var = meta_data['ground_truth'][replica_id]

            for train_size in train_sizes:
                i, counter = 0, 0
                while counter < train_size:
                    if x[i] > ds_params['xlow'] and x[i] < ds_params['xhigh']:
                        counter += 1
                    i += 1

                data_train = {'sample_id': sample_id_pool[:i],
                              'y': y[:i],
                              'x': x[:i],
                              'e': e[:i],
                              'true_mean': true_mean[:i],
                              'true_var': true_var[:i]
                             }
                             
                filename = f'{config["defaults"]["data_folder"]}/{ds_str}/{replica_id}_train{train_size}.pickle'
                with open(filename, 'wb') as handle:
                    pickle.dump(data_train, handle, protocol = pickle.HIGHEST_PROTOCOL)
                files_saved.append(filename)

            ## validate only on the restricted set
            valid_val = get_valid_indices_in_range(x[train_cut:val_cut], ds_params['xlow'], ds_params['xhigh'])[:val_size]
            
            data_val = {'sample_id': sample_id_pool[train_cut:val_cut][valid_val],
                        'y': y[train_cut:val_cut][valid_val],
                        'x': x[train_cut:val_cut][valid_val],
                        'e': e[train_cut:val_cut][valid_val],
                        'true_mean': true_mean[train_cut:val_cut][valid_val],
                        'true_var': true_var[train_cut:val_cut][valid_val]}

            filename = f'{config["defaults"]["data_folder"]}/{ds_str}/{replica_id}_val.pickle'
            with open(filename, 'wb') as handle:
                pickle.dump(data_val, handle, protocol = pickle.HIGHEST_PROTOCOL)
            files_saved.append(filename)

            ## test
            valid_test = get_valid_indices_in_range(x[val_cut:], ds_params['xlow'], ds_params['xhigh'])[:test_size]
            
            data_test = {'sample_id': sample_id_pool[val_cut:][valid_test],
                         'y': y[val_cut:][valid_test],
                         'x': x[val_cut:][valid_test],
                         'e': e[val_cut:][valid_test],
                         'true_mean': true_mean[val_cut:][valid_test],
                         'true_var': true_var[val_cut:][valid_test]}

            filename = f'{config["defaults"]["data_folder"]}/{ds_str}/{replica_id}_test.pickle'
            with open(filename, 'wb') as handle:
                pickle.dump(data_test, handle, protocol = pickle.HIGHEST_PROTOCOL)
            files_saved.append(filename)
            
            simulator.view_dataset(data_test['x'], data_test['y'], data_test['true_mean'], data_test['true_var'], save_file_as = f'{config["defaults"]["data_folder"]}/{ds_str}/fig_testset_{replica_id}.png')
            
        print(f'{len(files_saved)} files are saved; available train_sizes={train_sizes}')
        sys.stdout = sys.__stdout__
            
    return 0
    
if __name__ == "__main__":
    main()
