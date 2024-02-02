import os
import random
from tty import setraw 
from copy import deepcopy

import numpy as np
import torch
import torchvision
from PIL import Image

class BUS(torch.utils.data.Dataset):
    
    _class_map = {'benign':0, 'malignant':1, 'normal':2}
    
    def __init__(self, 
                 root: str, 
                 split: str = 'train', 
                 transform_train: torchvision.transforms = None, 
                 transform_test:  torchvision.transforms = None,
                 target_transform: torchvision.transforms = None) -> None:
        
        self.root = root
        self.split = split
        self.transform_train = transform_train
        self.transform_test  = transform_test
        self.target_transform = target_transform
        
        self.data = []
        self.targets = []
        
        for c in self._class_map:
            for img in os.listdir(os.path.join(root, c)):
                self.data.append(os.path.join(root, c, img))
                self.targets.append(self._class_map[c])
        
        _ind = list(range(len(self.data)))
        random.shuffle(_ind)
        num_data  = len(_ind)
        self.images = [self.data[i] for i in _ind]
        self.labels = [self.targets[i] for i in _ind]
    
        self.train_index = np.random.choice(_ind, int(0.8*len(_ind)), replace=False)
        self.test_index  = np.setdiff1d(_ind, self.train_index)

    def get(self, split: str):
        if split == 'train':
            self.data = [self.images[ind] for ind in self.train_index]
            self.targets = [self.labels[ind] for ind in self.train_index]
            self.transform = self.transform_train
        else:
            self.data = [self.images[ind] for ind in self.test_index]
            self.targets = [self.labels[ind] for ind in self.test_index]
            self.transform = self.transform_test
        self.targets = np.array(self.targets)
        # for compatibility reason
        self.uninform_labels = []
        self.inform_labels   = self.targets
        self.uninform_datasize = 0
        self.inform_datasize   = len(self.targets)
        return deepcopy(self)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, ind):
        
        img = Image.open(self.data[ind])
        targets = self.targets[ind]

        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return ind, img, targets