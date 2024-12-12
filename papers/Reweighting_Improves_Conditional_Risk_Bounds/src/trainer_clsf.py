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


class TrainerClsf():

    def __init__(
        self,
        run_args,
        configs,
    ):
        self.args = run_args
        self.configs = configs
        self.reweight_iters = configs['run_params']['reweight_iters']
        self.load_from_ckpt = configs['run_params']['load_from_ckpt']
        ## instantiate dataloaders
        self.dataloaders = SimDataLoaders(self.configs)
        
    @property
    def configs(self):
        return self._configs
    
    @configs.setter
    def configs(self, raw_configs):
        
        configs = {'task': 'clsf', 'train_size': self.args['train_size']}
        for key in ['dataloader', 'optimizer', 'run_params']:
            configs.update(raw_configs[key])
        for key in ['network_params', 'loss_params']:
            configs[key] = raw_configs[key]
        self.train_margin_separately = False
        if 'network_params_margin' in raw_configs:
            assert 'loss_params_margin' in raw_configs
            self.train_margin_separately = True
            configs['network_params_margin'] = raw_configs['network_params_margin']
            configs['loss_params_margin'] = raw_configs['loss_params_margin']
        else:
            configs['network_params_margin'] = None
            configs['loss_params_margin'] = None
        self._configs = configs


    def _configure_trainer(self, configs, run_type='mean', iter_idx=0):
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        ckpt_callback = ModelCheckpoint(dirpath=self.args['ckpt_dir'],
                                        filename=f'{run_type}-{iter_idx}-best',
                                        monitor='val_acc',
                                        save_last=True)
        ckpt_callback.CHECKPOINT_NAME_LAST = f'{run_type}-{iter_idx}-last'
        
        callbacks = [lr_monitor, ckpt_callback]
        if configs['es_patience']:
            early_stopper = EarlyStopping(monitor=configs['monitor'],
                                          min_delta=1e-4,
                                          patience=configs['es_patience'],
                                          verbose=True,
                                          mode="max")
            callbacks.append(early_stopper)
        
        logger = TensorBoardLogger(save_dir=self.args['output_dir'],
                                   name='lightning_logs',
                                   version=f'iter_{iter_idx}')
                                   
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
                             detect_anomaly=False,
                             fast_dev_run=False)
        
        return trainer
        
    def _run_predictions(self, trainer, pl_module, test_dataloader):
        
        logger.info(f'len(test_dataloader)={len(test_dataloader)}; num_samples={len(test_dataloader.dataset)}')
        output = trainer.predict(pl_module, test_dataloader)
        logits = torch.concat(output, axis=0)
        probs = torch.sigmoid(logits)
        
        return probs.numpy()
        
    def _run_one_iter(self, data_train, data_val, data_test, iter_idx=0):
        """
        run training for 1 iteration where mean network and margin network are trained (optional)
        """
        logger.info(f'configuring and training the mean network for iter={iter_idx}')
        if iter_idx > 0 and self.load_from_ckpt is not None:
            ckpt_path = os.path.join(self.args['ckpt_dir'], f'mean-{iter_idx-1}-{self.load_from_ckpt}.ckpt')
            pl_model_mean = LightningModel.load_from_checkpoint(ckpt_path)
            logger.info(f'mean network loaded from {ckpt_path}')
        else: ## (iter_idx > 0 and self.load_from_ckpt is None) or iter_idx ==0
            pl_model_mean = LightningModel(self.configs)
            logger.info(f'mean network instantiated')
        
        mean_trainer = self._configure_trainer(self.configs, run_type='mean', iter_idx=iter_idx)
        mean_trainer.fit(pl_model_mean,
                    self.dataloaders.train_dataloader(data_train),
                    self.dataloaders.val_dataloader(data_val))
        logger.info(f'mean network training stopped at epoch={pl_model_mean.current_epoch}')
        
        ### get mean predictions
        logger.info(f'getting mean predictions on train and validation set for iter={iter_idx}')
        prob_train = self._run_predictions(mean_trainer, pl_model_mean, self.dataloaders.test_dataloader(data_train))
        prob_val = self._run_predictions(mean_trainer, pl_model_mean, self.dataloaders.test_dataloader(data_val))

        logger.info(f'getting mean predictions on test set for iter={iter_idx}')
        prob_test = self._run_predictions(mean_trainer, pl_model_mean, self.dataloaders.test_dataloader(data_test))

        if not self.train_margin_separately:
            data_train['weight'], data_val['weight'] = np.abs(prob_train-0.5), np.abs(prob_val-0.5)
            logger.info(f'weight updated based on the estimated mean network')
            return {'prob_train': prob_train, 'prob_val': prob_val, 'prob_test': prob_test}
        
        logger.info(f'configuring and training the margin network for iter={iter_idx}')
        margin_configs = copy.deepcopy(self.configs)
        margin_configs['network_params'] = self.configs['network_params_margin']
        margin_configs['loss_params'] = self.configs['loss_params_margin']
        del margin_configs['network_params_margin'], margin_configs['loss_params_margin']
        
        pl_model_margin = LightningModel(margin_configs)
        margin_trainer = self._configure_trainer(margin_configs, run_type='margin', iter_idx=iter_idx)
        margin_trainer.fit(pl_model_margin,
                           self.dataloaders.train_dataloader(data_train),
                           self.dataloaders.val_dataloader(data_val))
        logger.info(f'margin network training stopped at epoch={pl_model_margin.current_epoch}')
        
        ### get margin, which will be used as weights in the next iteration
        logger.info(f'getting margin predictions on train and validation set for iter={iter_idx}')
        eta_train = self._run_predictions(margin_trainer, pl_model_margin, self.dataloaders.test_dataloader(data_train))
        eta_val = self._run_predictions(margin_trainer, pl_model_margin, self.dataloaders.test_dataloader(data_val))
        
        logger.info(f'getting margin predictions on test set for iter={iter_idx}')
        eta_test = self._run_predictions(margin_trainer, pl_model_margin, self.dataloaders.test_dataloader(data_test))
        
        ## update margin/weight based on the estimate from the margin network
        data_train['weight'], data_val['weight'] = np.abs(eta_train-0.5), np.abs(eta_val-0.5)
        logger.info(f'weighted updated based on the trained margin network')
        
        return {'prob_train': prob_train, 'prob_val': prob_val, 'prob_test': prob_test,
                'eta_train': eta_train, 'eta_val': eta_val, 'eta_test': eta_test}
        
        
    def end_to_end(self, data_train, data_val, data_test):
        
        data_train = {'x': data_train['x'][:self.configs['train_size']],
                      'y': data_train['y'][:self.configs['train_size']]}
        
        for iter_idx in range(self.reweight_iters+1):
            iter_output = self._run_one_iter(data_train, data_val, data_test, iter_idx=iter_idx)
            for key, value in iter_output.items():
                filepath = os.path.join(self.args['output_dir'], f'{iter_idx}_{key}.npy')
                np.save(filepath, value)
            
        return
