"""
wrapper class for an estimator
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import numpy as np
from copy import deepcopy
from .constructor import ModelCtor, LossCtor

class Estimator(pl.LightningModule):
    
    def __init__(self, params, data_train = None, data_val = None):
        
        super(Estimator, self).__init__()
        self.save_hyperparameters()
        
        self._params = params
        self.model_class = params['model_class']
        self.loss_type = params['loss_type']
        
        ## invoke constructor
        self.model = ModelCtor(self.model_class, params).ctor()
        self.loss_fn = LossCtor(self.loss_type, params).ctor()
        
        self.data_train = data_train
        self.data_val = data_val
        
    def forward(self, input):
    
        output = self.model(input)
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self._params['learning_rate'], weight_decay = self._params.get('weight_decay',0.000001))
        if not len(self._params.get('reduce_LR_mode', '')):
            return optimizer
        else:
            lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode = self._params['reduce_LR_mode'],
                                                            factor = self._params['reduce_LR_factor'],
                                                            patience = self._params['reduce_LR_patience']),
                            'monitor': 'val_loss',
                            'frequency': 1
                            }
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    
    def training_step(self, batch, batch_idx):
        
        targets = batch['target']
        preds = self.forward(batch['x'])
        
        loss = self.loss_fn(preds, targets)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        targets = batch['target']
        preds = self.forward(batch['x'])
        
        loss = self.loss_fn(preds, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def predict_step(self, batch, batch_idx):
        preds = self.forward(batch['x'])
        return preds

        
    def get_fitted_and_residual(self, ds_type='train'):
        
        if ds_type == 'train':
            dataset = self.train_dataloader()
        elif ds_type == 'val':
            dataset = self.val_dataloader()
        else:
            raise ValueError("incorrect specification of ds_type; can only get fitted and residual on train or val")

        torch.set_grad_enabled(False)
        self.eval()

        xs, predictions, residuals = [], [], []
        for batch_idx, batch in enumerate(dataset):
            
            truth, preds = batch['target'], self.predict_step(batch, batch_idx)
            residual = truth - preds
            
            xs.extend(batch['x'].cpu().numpy())
            residuals.extend(residual.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
        
        xs = np.array(xs)
        predictions = np.array(predictions)
        residuals = np.array(residuals)
        
        return xs, predictions, residuals
        
    def run_on_testset(self, data_test):
    
        data_test = self.test_dataloader(data_test)

        torch.set_grad_enabled(False)
        self.eval()

        xs, predictions = [], []
        for batch_idx, batch in enumerate(data_test):
            preds = self.predict_step(batch, batch_idx)
            xs.extend(batch['x'].cpu().numpy())
            predictions.extend(preds.cpu().numpy())

        xs = np.array(xs)
        predictions = np.array(predictions)

        is_modeling_inverse = self._params.get('is_modeling_inverse', False)
        if is_modeling_inverse:
            predictions = 1.0/predictions

        return xs, predictions
        
    def train_dataloader(self):
        trainingset = SimDataset(data = self.data_train, n_degree = self._params.get('n_degree',1))
        return DataLoader(trainingset,
                          batch_size=self._params['batch_size'],
                          shuffle = self._params.get('shuffle',True),
                          num_workers=self._params.get('num_workers',4),
                          persistent_workers=True,
                          drop_last=True)
    
    def val_dataloader(self):
        valset = SimDataset(data = self.data_val, n_degree = self._params.get('n_degree',1))
        return DataLoader(valset,
                          batch_size=self._params['batch_size'],
                          shuffle = False,
                          num_workers=self._params.get('num_workers',4),
                          persistent_workers=True,
                          drop_last=True)

    def test_dataloader(self, data_test):
        testset = SimDataset(data_test, n_degree = self._params.get('n_degree',1))
        return DataLoader(testset,
                          batch_size=self._params['batch_size'],
                          shuffle = False,
                          num_workers=self._params.get('num_workers',0),
                          persistent_workers=False)

class SimDataset(Dataset):
    def __init__(self, data, n_degree=1):
    
        self.xdata = data['x']
        self.ydata = data.get('y', None)
        self.n_degree = n_degree
        
    def __len__(self):
        return len(self.xdata)
    
    def __getitem__(self, index):
        
        if self.n_degree == 1:
            x = self.xdata[index].astype(np.float32)
        else:
            x = []
            for i in range(1, self.n_degree+1):
                x_with_degree = self.xdata[index] ** i
                x.append(x_with_degree.squeeze().astype(np.float32))
            x = np.stack(x,axis=-1)
            
        sample = {'x': x}
        if self.ydata is not None:
            sample['target'] = self.ydata[index].astype(np.float32)
        
        return sample
