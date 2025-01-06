import copy
import logging
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
pl_logger = logging.getLogger("pytorch_lightning")
pl_logger.propagate = False
pl_logger.setLevel(logging.ERROR)
import torch
from torchmetrics import Accuracy

from src import LightningModel, SimDataLoaders
from utils.logging import get_logger

logger = get_logger()


class TrainerRegr():

    def __init__(
        self,
        run_args,
        configs, ## after parsing the yamls
    ):
        self.args = run_args
        self.mean_configs = configs
        self.var_configs = configs
        
        self.reweight_iters = configs['run_params']['reweight_iters']
        self.load_from_ckpt = configs['run_params']['load_from_ckpt']
        
        ## instantiate dataloaders for mean and variance
        self.mean_dataloaders = SimDataLoaders(self.mean_configs)
        self.var_dataloaders = SimDataLoaders(self.var_configs)
        
    @property
    def mean_configs(self):
        return self._mean_configs
    
    @mean_configs.setter
    def mean_configs(self, raw_configs):
        network_configs = {'task': 'regr', 'train_size': self.args['train_size']}
        network_configs.update(raw_configs['dataloader'])
        network_configs.update(raw_configs['mean_network_train'])
        network_configs['network_params'] = raw_configs['mean_network_params']
        network_configs['loss_params'] = raw_configs['mean_loss_params']
        self._mean_configs=network_configs

    @property
    def var_configs(self):
        return self._var_configs
    
    @var_configs.setter
    def var_configs(self, raw_configs):
        network_configs = {'task': 'regr', 'train_size': self.args['train_size']}
        network_configs.update(raw_configs['dataloader'])
        network_configs.update(raw_configs['var_network_train'])
        network_configs['network_params'] = raw_configs['var_network_params']
        network_configs['loss_params'] = raw_configs['var_loss_params']
        self._var_configs=network_configs
        
    def _configure_trainer(self, configs, run_type='mean', iter_idx=0):
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        ckpt_callback = ModelCheckpoint(dirpath=self.args['ckpt_dir'],
                                        filename=f'{run_type}-{iter_idx}-best',
                                        monitor='val_loss',
                                        save_last=True)
        ckpt_callback.CHECKPOINT_NAME_LAST = f'{run_type}-{iter_idx}-last'
        
        callbacks = [lr_monitor, ckpt_callback]
        if configs['es_patience']:
            early_stopper = EarlyStopping(monitor=configs['monitor'],
                                          min_delta=1e-4,
                                          patience=configs['es_patience'],
                                          verbose=True,
                                          mode="min")
            callbacks.append(early_stopper)
        
        logger = TensorBoardLogger(save_dir=self.args['output_dir'], name='lightning_logs', version=f'{run_type}_{iter_idx}')
        trainer = pl.Trainer(accelerator=self.args['accelerator'],
                             devices=self.args['devices'],
                             max_epochs=configs['max_epochs'],
                             callbacks=callbacks,
                             enable_progress_bar=False,
                             logger=logger,
                             gradient_clip_val=configs['gradient_clip_val'],
                             log_every_n_steps=1,
                             num_sanity_val_steps=0,
                             limit_val_batches=configs['limit_val_batches'],
                             precision=32,
                             detect_anomaly=False)
        
        return trainer
        
    def _run_predictions(self, trainer, pl_module, test_dataloader):
        
        logger.info(f'len(test_dataloader)={len(test_dataloader)}; num_samples={len(test_dataloader.dataset)}')
        output = trainer.predict(pl_module, test_dataloader)
        predictions = torch.concat(output, axis=0).numpy()
        if pl_module.configs['loss_params'].get('is_modeling_inverse', False):
            predictions = 1.0/predictions
        
        return predictions
    
    def _run_one_iter(self, data_train, data_val, data_test, iter_idx=0, load_from_ckpt='last'):
        """
        run training for 1 iteration of (re)weighted: train first the mean then the variance networks based on the residuals
        argvs:
        - data_train/ data_val: dictionaries, training, validation data
        - iter_idx: iteration index
        - load_from_ckpt: ['best', 'last', None]
        """
        logger.info(f'configuring and training the mean network for iter={iter_idx}')
        if iter_idx > 0 and load_from_ckpt is not None:
            ckpt_path = os.path.join(self.args['ckpt_dir'], f'mean-{iter_idx-1}-{load_from_ckpt}.ckpt')
            pl_model_mean = LightningModel.load_from_checkpoint(ckpt_path)
            logger.info(f'mean network loaded from {ckpt_path}')
        else:
            pl_model_mean = LightningModel(self.mean_configs)
            logger.info(f'mean network instantiated')
            
        mean_trainer = self._configure_trainer(self.mean_configs, run_type='mean', iter_idx=iter_idx)
        mean_trainer.fit(pl_model_mean,
                         self.mean_dataloaders.train_dataloader(data_train),
                         self.mean_dataloaders.val_dataloader(data_val))
        logger.info(f'mean network training stopped at epoch={pl_model_mean.current_epoch}')
        
        ### get residuals and therefore the target for the variance network training
        logger.info(f'getting mean predictions for train and validation sets for iter={iter_idx}')
        y_hat_train = self._run_predictions(mean_trainer,
                                            pl_model_mean,
                                            self.mean_dataloaders.test_dataloader(data_train))
        data_train_var = {'x': data_train['x'].copy(), 'y': (data_train['y']-y_hat_train)**2}
        y_hat_val = self._run_predictions(mean_trainer,
                                          pl_model_mean,
                                          self.mean_dataloaders.test_dataloader(data_val))
        data_val_var = {'x': data_val['x'].copy(), 'y': (data_val['y']-y_hat_val)**2}
        logger.info(f'variance network target updated based on the trained mean network')
        
        logger.info(f'configuring and training the variance network for iter={iter_idx}')
        if iter_idx > 0 and load_from_ckpt is not None:
            ckpt_path = os.path.join(self.args['ckpt_dir'], f'var-{iter_idx-1}-{load_from_ckpt}.ckpt')
            pl_model_variance = LightningModel.load_from_checkpoint(ckpt_path)
            logger.info(f'variance network loaded from {ckpt_path}')
        else:
            pl_model_variance = LightningModel(self.var_configs)
            logger.info(f'variance network instantiated')
            
        var_trainer = self._configure_trainer(self.var_configs, run_type='var', iter_idx=iter_idx)
        var_trainer.fit(pl_model_variance,
                        self.var_dataloaders.train_dataloader(data_train_var),
                        self.var_dataloaders.val_dataloader(data_val_var))
        logger.info(f'variance network training stopped at epoch={pl_model_variance.current_epoch}')
        
        ### update data_train and data_val (in place) weight using sigma hat
        logger.info(f'getting variance predictions for train and validation sets for iter={iter_idx}')
        sigma_hat_train = self._run_predictions(var_trainer,
                                                pl_model_variance,
                                                self.var_dataloaders.test_dataloader(data_train_var))
        data_train['weight'] = 1./sigma_hat_train
        
        sigma_hat_val = self._run_predictions(var_trainer,
                                              pl_model_variance,
                                              self.var_dataloaders.test_dataloader(data_val_var))
        data_val['weight'] = 1./sigma_hat_val
        logger.info(f'weight updated based on the trained variance network')
        
        ## run on test data
        logger.info(f'getting mean and variance predictions on test set for iter={iter_idx}')
        y_hat_test = self._run_predictions(mean_trainer, pl_model_mean,
                                           test_dataloader=self.mean_dataloaders.test_dataloader(data_test))
        sigma_hat_test = self._run_predictions(var_trainer, pl_model_variance,
                                               test_dataloader=self.var_dataloaders.test_dataloader(data_test))
        
        return {'yhat_train': y_hat_train, 'sigmahat_train': sigma_hat_train,
                 'yhat_val': y_hat_val, 'sigmahat_val': sigma_hat_val,
                 'yhat_test': y_hat_test, 'sigmahat_test': sigma_hat_test}
        
    def end_to_end(self, data_train, data_val, data_test):
        
        data_train = {'x': data_train['x'][:self.mean_configs['train_size']],
                      'y': data_train['y'][:self.mean_configs['train_size']]}
        
        for iter_idx in range(self.reweight_iters+1):
        
            iter_output = self._run_one_iter(data_train, data_val, data_test, iter_idx=iter_idx, load_from_ckpt=self.load_from_ckpt)
            for key, value in iter_output.items():
                filepath = os.path.join(self.args['output_dir'], f'{iter_idx}_{key}.npy')
                np.save(filepath, value)
            
        return
