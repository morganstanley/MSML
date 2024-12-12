"""
Dataloader for simulation
"""
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class SimDataset(Dataset):
    
    def __init__(self, data, size=None):
        size = size or data['x'].shape[0]
        self.xdata = data['x'][:size]
        self.ydata = data['y'][:size] if 'y' in data else None
        self.weight = data['weight'][:size] if 'weight' in data else None
        
    def __len__(self):
        return len(self.xdata)
    
    def __getitem__(self, index):
        x = torch.tensor(self.xdata[index], dtype=torch.float32)
        sample = {'x': x}
        if self.weight is not None:
            sample['weight'] = torch.tensor(self.weight[index], dtype=torch.float32)
        if self.ydata is not None:
            sample['target'] = torch.tensor(self.ydata[index], dtype=torch.float32)
        return sample

class SimDataLoaders():
        
    def __init__(self, configs):
        super(SimDataLoaders, self).__init__()
        self.configs = configs
        
    def train_dataloader(self, data):
        trainingset = SimDataset(data=data, size=self.configs['train_size'])
        return DataLoader(trainingset,
                          batch_size=self.configs['batch_size'],
                          shuffle=self.configs.get('shuffle',True),
                          num_workers=self.configs.get('num_workers',4),
                          persistent_workers=True,
                          drop_last=True)
    
    def val_dataloader(self, data):
        valset = SimDataset(data=data, size=self.configs.get('val_size',None))
        return DataLoader(valset,
                          batch_size=self.configs['batch_size'],
                          shuffle=False,
                          num_workers=self.configs.get('num_workers',4),
                          persistent_workers=True,
                          drop_last = True)

    def test_dataloader(self, data, test_size=None):
        testset = SimDataset(data=data, size=test_size or self.configs.get('test_size',None))
        return DataLoader(testset,
                          batch_size=self.configs['batch_size'],
                          shuffle=False,
                          num_workers=self.configs.get('num_workers',0),
                          persistent_workers=False)

