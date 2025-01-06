import importlib

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy

class LightningModel(pl.LightningModule):
    
    def __init__(self, configs):
        
        super(LightningModel, self).__init__()
        self.save_hyperparameters()
        self.configs = configs
        
        available_modules = importlib.import_module('src.networks')
        
        ## configure networks
        self.moduleClass = getattr(available_modules, self.configs['network_params']['module_type'])
        self.network = self.moduleClass(self.configs['network_params'])
        
        ## configure loss fn
        self.lossClass = getattr(available_modules, self.configs['loss_params']['loss_type'])
        self.loss_fn = self.lossClass(configs['loss_params'])
        
        if self.configs['task'] == 'clsf':
            self.train_acc = Accuracy(task='binary',threshold=0.5)
            self.val_acc = Accuracy(task='binary',threshold=0.5)
        
    @property
    def configs(self):
        return self._configs
    
    @configs.setter
    def configs(self, configs):
        self._configs = configs
        
    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.configs['learning_rate'],
                                     weight_decay=self.configs.get('weight_decay',1e-6))
        if self.configs['scheduler_type'] == 'ReduceLROnPlateau':
            mode = 'min' if self.configs['monitor'] == 'val_loss' else 'max'
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                            mode=mode,
                            factor=self.configs['reduceLR_factor'],
                            patience=self.configs['reduceLR_patience'])
            lr_scheduler = {'scheduler': scheduler, 'monitor': self.configs['monitor']}
        elif self.configs['scheduler_type'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                            step_size=self.configs['stepLR_stepsize'],
                            gamma=self.configs['stepLR_gamma'])
            lr_scheduler = {'scheduler': scheduler,
                            'interval': self.configs.get('stepLR_interval','epoch')}
        
        else:
            raise ValueError('unrecognized scheduler_type')
        
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        
    def training_step(self, batch, batch_idx):
        
        preds = self.network(batch['x'])
        targets = batch['target']
        weights = batch.get('weight',None)

        loss = self.loss_fn(preds, targets, weights)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if self.configs['task'] == 'clsf':
            probs = torch.sigmoid(preds)
            self.train_acc.update(probs, targets.to(dtype=torch.int32))
            self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        preds = self.network(batch['x'])
        targets = batch['target']
        weights = batch.get('weight',None)
        
        loss = self.loss_fn(preds, targets, weights)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if self.configs['task'] == 'clsf':
            probs = torch.sigmoid(preds)
            self.val_acc.update(probs, targets.to(dtype=torch.int32))
            self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def predict_step(self, batch, batch_idx):
        return self.network(batch['x'])
    
    def train_dataloader(self):
        raise NotImplementedError
        
    def val_dataloader(self):
        raise NotImplementedError
    
    def test_dataloader(self):
        raise NotImplementedError
    
    
    
    
