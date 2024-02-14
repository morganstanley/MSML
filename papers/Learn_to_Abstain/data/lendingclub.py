import os
import random
from copy import deepcopy

import torch
import pandas as pd
import numpy as np

class LendingClub(torch.utils.data.Dataset):
    
    _feature_list = ['loan_amnt',  'term', 'int_rate', 'installment', 'sub_grade', 'home_ownership', 'annual_inc', 'verification_status', 'purpose', 'zip_code', 
                    'addr_state', 'dti',  'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_util', 'total_acc', 'initial_list_status', 'application_type', 'mort_acc', 'pub_rec_bankruptcies']
    _target_list = ['loan_status']
    _status_map  = {'Charged Off':0, 'Fully Paid':1, "Late (31-120 days)":2, "Late (16-30 days)":2}
    
    def __init__(self, path: str) -> None:
        
        df = pd.read_csv(path)
        df = df[df['loan_status'].isin(self._status_map)]
        df = df[self._feature_list+self._target_list]
        df.loc['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line']).dt.year

        df.dropna(how='any', inplace=True)
        df.drop_duplicates(inplace=True)
        df = df.reset_index()
        
        self.labels = df['loan_status'].apply(lambda x : self._status_map[x]).astype(np.int64).to_list()
        self.records = df[self._feature_list]
        _ind = list(range(len(self.records)))
        random.shuffle(_ind)
        num_data = len(_ind)

        # one-hot encoding categorical variables
        columns = self.records.columns
        for k in columns:
            if self.records[k].dtype == np.float64:
                self.records[k] = (self.records[k] - self.records[k].mean())/self.records[k].std()
            if self.records[k].dtype == object:
                self.records = self.records.merge(pd.get_dummies(self.records[k]), left_index=True, right_index=True).drop(k, axis=1)
        self.records = self.records.to_numpy().astype(np.float32)

        self.train_index = np.random.choice(_ind, int(0.8*len(_ind)), replace=False)
        self.test_index  = np.setdiff1d(_ind, self.train_index)

    def get(self, split: str):
        if split == 'train':
            self.data = [self.records[ind] for ind in self.train_index]
            self.targets = [self.labels[ind] for ind in self.train_index]
        else:
            self.data = [self.records[ind] for ind in self.test_index]
            self.targets = [self.labels[ind] for ind in self.test_index]
        self.targets = np.array(self.targets)
        # for compatibility reason
        self.uninform_labels = []
        self.inform_labels   = self.targets
        self.uninform_datasize = 0
        self.inform_datasize   = len(self.targets)
        return deepcopy(self)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return index, torch.tensor(self.data[index]), self.targets[index]