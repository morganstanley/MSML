"""
script for training synthetic datasets with provided functional basis
python train_sim_scipy --ds_str='ds5' --n_replica=10 --train_size=500 --use_true_mean=1
"""

import sys
import os
import datetime
import yaml
import argparse
import pickle

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from src import Estimator

from src import fbEstimator

parser = argparse.ArgumentParser()

parser.add_argument("--ds_str", help="data settings to run", type=str, default='ds1')
parser.add_argument("--cuda", help='cuda device to use',type=int,default=0)
parser.add_argument("--n_replica", help='number of replica',type=int,default=10)
parser.add_argument("--train_size", help='number of training samples',type=int,default=1000)
parser.add_argument("--use_true_mean",help='indicator for whether directly using the true mean',type=int,default=0)
parser.add_argument("--data_folder", help='parent folder for data',type=str,default='data_sim')
parser.add_argument("--output_folder", help='parent folder for output',type=str,default='output_sim_scipy')
parser.add_argument("--config", help="config file name override", default="")

def main():

    global args
    args = parser.parse_args()
    
    ## device setup
    setattr(args, 'accelerator', 'gpu' if torch.cuda.is_available() else 'cpu')
    setattr(args, 'devices', [args.cuda] if args.accelerator == 'gpu' else 1)
    
    ## config setup
    setattr(args, 'config_default', f'./configs/defaults.yaml')
    if not len(args.config):
        setattr(args,'config', f'./configs/{args.ds_str}.yaml')
    setattr(args, 'train_size', int(args.train_size))
    setattr(args, 'output_folder', f'{args.output_folder}/{args.ds_str}')

    print(f'ds={args.ds_str}, train_size={args.train_size}')
    print(f'default_config={args.config_default}, setting-specific_config={args.config}')

    ###################################
    ## Load default and setting-specific config
    ###################################

    ## read in default config
    with open(args.config_default) as f_default:
        default_configs = yaml.safe_load(f_default)
    ## read in dataset-specific config
    with open(args.config) as f_ds:
        configs = yaml.safe_load(f_ds)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    logfile = f'{args.output_folder}/log_trainsize{args.train_size}_{datetime.datetime.now().strftime("%Y%m%d")}.log'
    sys.stdout = open(logfile, 'w')

    ###################################
    ## Loop over each replica, with mean estimation -> variance estimation
    ###################################

    for replica_id in range(args.n_replica):

        print(f'[{datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")}], replica_id = {replica_id} ({replica_id+1}/{args.n_replica}) started')

        ## 1.1 load data
        file_train = f'{args.data_folder}/{args.ds_str}/{replica_id}_train{args.train_size}.pickle'
        with open(file_train, 'rb') as handle:
            data_train = pickle.load(handle)

        file_val = f'{args.data_folder}/{args.ds_str}/{replica_id}_val.pickle'
        with open(file_val, 'rb') as handle:
            data_val = pickle.load(handle)

        file_test = f'{args.data_folder}/{args.ds_str}/{replica_id}_test.pickle'
        with open(file_test, 'rb') as handle:
            data_test = pickle.load(handle)

        testset_predictions = {}

        ## >>>>>>>>>>>>>>>>
        ## 1.2 train a mean-network
        ## >>>>>>>>>>>>>>>>
        if not args.use_true_mean:
            print(f'****** training mean network with MSE loss ******')
            params = prepare_nn_params(default_configs, configs, key = 'mean_net')
            mean_estimator = train_network(params, data_train, data_val)
            # 1.2.1 get the residual, prep it to be used for variance estimation
            x_train, _, res_train = mean_estimator.get_fitted_and_residual(ds_type='train')
            x_val, _, res_val = mean_estimator.get_fitted_and_residual(ds_type='val')
            var_data_train = {'x': x_train, 'y': res_train**2}
            var_data_val = {'x': x_val, 'y': res_val**2}
            # 1.2.2 test set
            x, mean_est = mean_estimator.run_on_testset(data_test)[1]
        else:
            print(f'****** mean estimation is skipped, directly using noise for variance estimation ******')
            var_data_train = {'x': data_train['x'], 'y': data_train['e']**2}
            var_data_val = {'x': data_val['x'], 'y': data_val['e']**2}
            x, mean_est = data_test['x'], data_test['true_mean']

        testset_predictions['x'] = x
        testset_predictions['mean_est'] = mean_est

        ## >>>>>>>>>>>>>>>>
        ## 1.3 train variance networks with different losses
        ## >>>>>>>>>>>>>>>>

        # 1.3.1 mse-loss based variance estimation
        print(f'****** estimating variance with MSE loss ******')
        params = prepare_fb_params(configs, key = 'mse_variance_fb')
        # options={'xatol': 1e-8, 'disp': True}
        var_estimator_mse = fbEstimator(params = params)
        var_estimator_mse.fit(var_data_train['x'],var_data_train['y'])
        testset_predictions['var_est_mse'] = var_estimator_mse.run_on_testset(data_test['x'])

        # 1.3.2 nll-loss based variance estimation
        print(f'****** estimating variance with NLL loss ******')
        params = prepare_fb_params(configs, key = 'nll_variance_fb')
        var_estimator_nll = fbEstimator(params = params)
        var_estimator_nll.fit(var_data_train['x'],var_data_train['y'])
        testset_predictions['var_est_nll'] = var_estimator_nll.run_on_testset(data_test['x'])

        ## >>>>>>>>>>>>>>>>
        ## 1.4 save down
        ## >>>>>>>>>>>>>>>>
        output_file = f'{args.output_folder}/{replica_id}_test_trainsize{args.train_size}.pickle'
        with open(output_file, 'wb') as handle:
            pickle.dump(testset_predictions, handle, protocol = pickle.HIGHEST_PROTOCOL)

        print(f'[{datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")}], replica_id = {replica_id} ({replica_id+1}/{args.n_replica}) finished; output saved to {output_file}')

    sys.stdout = sys.__stdout__
    return 0

def prepare_nn_params(default_configs, configs, key = 'mean_net'):

    """ util func for prep run params """
    params = default_configs[key].copy()
    params.update(configs[key])
    return params

def train_network(params, data_train, data_val):

    """ util func for train a network """
    estimator = Estimator(params, data_train = data_train, data_val = data_val)

    ## this is more for debugging
    callbacks = []
    if params.get('early_stop_patience',0):
        early_stopper = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=params['early_stop_patience'], verbose=False, mode="min")
        callbacks.append(early_stopper)

    trainer = pl.Trainer(accelerator=args.accelerator,
                         devices=args.devices,
                         max_epochs = params['max_epochs'],
                         callbacks = callbacks,
                         enable_progress_bar = False,
                         log_every_n_steps = 1,
                         gradient_clip_val = 1,
                         limit_val_batches = 1,
                         precision = 32)
    trainer.fit(estimator)
    print(f'>> total number of epochs = {estimator.current_epoch}')

    return estimator

def prepare_fb_params(configs, key = 'mse_var_fb'):
    return configs[key]

if __name__ == "__main__":
    main()
