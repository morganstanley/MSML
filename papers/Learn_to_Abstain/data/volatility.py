import os

import pandas as pd
import numpy as np
from datetime import datetime

class Volatility():

    def __init__(self, path: str, split:str , context_size: int=1):

        df = pd.read_csv(path)
        df = df.rename(columns={'Unnamed: 0':'timestamp'})
        df['Dt'] = df['timestamp'].str[0:10].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        df['rv5_ss'] *= 1000
        self.context_size = context_size
        seq_len = context_size if context_size > 1 else -10000000000000 # if non-positive seq-length, then

        if split == 'train':
            self.df = df[(df.Dt> "2000-01-01") & (df.Dt < "2020-07-01")]
        else:
            self.df = df[df.Dt >="2020-07-01"]

        self.df = self.df.sort_values(by=['Symbol', 'timestamp'])
        symbol_list = self.df.Symbol.unique().tolist()
        processed_df = []
        df_grouped  = self.df[['Dt','rv5_ss','open_to_close','Symbol']].groupby('Symbol')
        for symbol in symbol_list:
            tmp_df = df_grouped.get_group(symbol)
            tmp_df['next_rv5_ss'] = tmp_df['rv5_ss'].shift(-1)
            tmp_df['target'] = (tmp_df['next_rv5_ss']>=tmp_df['rv5_ss']).values
            tmp_df.fillna(0, inplace=True)
            tmp_df.loc[:, 'Dt'] = list(map(lambda x: str(x)[:10], tmp_df[tmp_df['Symbol']==symbol]['Dt']))
            processed_df.append(tmp_df[['Dt','rv5_ss','open_to_close','Symbol', 'target']])
        processed_df = pd.concat(processed_df)

        # generate index 
        df_group_len = np.cumsum([0] + df_grouped.apply(lambda x: len(x)).tolist()[:-1])
        df_group_ind = df_grouped.apply(lambda x: list(range(len(x)-context_size+1)))
        self.ind_map = [ind+df_group_len[i] for i in range(len(df_group_len)) for ind in df_group_ind[i]]
        self.targets = np.concatenate([x for x in processed_df.groupby('Symbol').apply(lambda x: x['target'].to_numpy().squeeze()[(context_size-1):])]).astype(np.int32)
        self.data = processed_df[['rv5_ss','open_to_close']].to_numpy().astype(np.float32)
        self.alltargets = processed_df['target'].to_numpy().astype(np.int32)

        # for compatibility reason
        self.uninform_labels = []
        self.inform_labels   = self.targets
        self.uninform_datasize = 0
        self.inform_datasize   = len(self.targets)
        
    def __len__(self):
        return len(self.ind_map)

    def __getitem__(self, orig_ind):

        ind = self.ind_map[orig_ind]
        X   = self.data[ind:(ind+self.context_size)]
        y   = self.alltargets[ind:(ind+self.context_size)] 

        return orig_ind, X, y