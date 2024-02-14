import random 

import torch
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
import numpy as np
import importlib

from data.hybridmnist import HybridMNIST
from data.svhn import SVHN
from data.volatility import Volatility
from data.bus import BUS
from data.lendingclub import LendingClub
from utils.noise import noisify_with_P
from utils.utils import init_fn_

class Environ():

    _method_dict = {
        'confidence'   : 'Confidence', 
        'selectivenet' : 'SelectiveNet', 
        'deepgambler'  : 'DeepGambler', 
        'adaptive' : 'Adaptive', 
        'oneside'  : 'OneSide' , 
        'dac'  : 'DAC', 
        'isa'  : 'ISA', 
        'isav2': 'ISAV2'
    }

    def __init__(self, config):

        self.config = config
        self.seed = config.seed
    
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def build_dataset(self):
        
        if self.config.dataset == 'mnist':
            num_classes = 10
            # data augmentation
            transform_train_mnist = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            transform_test_mnist = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

            transform_train_fashion = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform_test_fashion = transforms.Compose([
                transforms.ToTensor(),
            ])

            # training and testing dataset for the classification network
            trainset = HybridMNIST(
                root=self.config.data_root,
                train=True,
                mnist_transform=transform_train_mnist,
                fashion_transform=transform_train_fashion,
                clean_ratio=self.config.clean_ratio,
                noise_ratio=self.config.noise_ratio,
                data_num_ratio=self.config.data_num_ratio)
            testset = HybridMNIST(
                root=self.config.data_root,
                train=False,
                mnist_transform=transform_test_mnist,
                fashion_transform=transform_test_fashion,
                clean_ratio=1,
                noise_ratio=self.config.noise_ratio,
                data_num_ratio=1.0)

            trainloader = torch.utils.data.DataLoader(
                dataset=trainset, 
                batch_size=self.config.batch_size, 
                shuffle=True, 
                num_workers=2,
                worker_init_fn=init_fn_, 
                drop_last=True)
            testloader = torch.utils.data.DataLoader(
                dataset=testset, 
                batch_size=self.config.batch_size, 
                shuffle=False, 
                num_workers=2)

        elif self.config.dataset == 'svhn':
            num_classes = 5
            # data augmentation
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            trainset = SVHN(
                root=self.config.data_root,
                split='train',
                transform=transform_train,
                clean_ratio=self.config.clean_ratio,
                noise_ratio=self.config.noise_ratio,
                data_num_ratio=self.config.data_num_ratio)
            testset = SVHN(
                root=self.config.data_root,
                split='test',
                transform=transform_test,
                clean_ratio=1,
                noise_ratio=self.config.noise_ratio,
                data_num_ratio=1.0)

            trainloader = torch.utils.data.DataLoader(
                dataset=trainset, 
                batch_size=self.config.batch_size, 
                shuffle=True, 
                num_workers=2, 
                worker_init_fn=init_fn_, 
                pin_memory=True, 
                drop_last=True)
            testloader = torch.utils.data.DataLoader(
                dataset=testset, 
                batch_size=self.config.batch_size, 
                shuffle=False, 
                pin_memory=True, 
                num_workers=2)
        
        elif self.config.dataset == 'volatility':
            
            trainset = Volatility(self.config.volatility_path, context_size=self.config.context_size_vol, split='train')
            testset  = Volatility(self.config.volatility_path, context_size=self.config.context_size_vol, split='test')

            trainloader = torch.utils.data.DataLoader(
                dataset=trainset, 
                batch_size=self.config.batch_size, 
                shuffle=True, 
                num_workers=20, 
                worker_init_fn=init_fn_, 
                pin_memory=True, 
                drop_last=True)
            testloader = torch.utils.data.DataLoader(
                dataset=testset, 
                batch_size=self.config.batch_size, 
                shuffle=False, 
                pin_memory=True, 
                num_workers=20)
            
        elif self.config.dataset == 'bus': 
            
            transform_train = transforms.Compose([
                transforms.Resize((324, 324)),
                transforms.RandomHorizontalFlip(), 
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
            ])

            transform_test = transforms.Compose([
                transforms.Resize((324, 324)), 
                transforms.ToTensor(), 
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
            ])
            
            dataset = BUS(root = self.config.bus_path, transform_train=transform_train, transform_test=transform_test)
            trainset = dataset.get(split='train')
            testset  = dataset.get(split='test')

            class_weight = {x[0]:1/x[1] for x in zip(*np.unique(trainset.targets, return_counts=True))}
            weights  = [class_weight[y] for y in trainset.targets]
            sampler = WeightedRandomSampler(weights=weights, replacement=True, num_samples=len(weights))

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.config.batch_size, num_workers=2, pin_memory=True, shuffle=True)
            testloader  = torch.utils.data.DataLoader(testset,  batch_size=self.config.batch_size, num_workers=2, pin_memory=True, shuffle=False)

            self.config.train_index = dataset.train_index
            self.config.test_index = dataset.test_index

        elif self.config.dataset == 'lc':
            
            dataset  = LendingClub(path = self.config.lendingclub_path)
            trainset = dataset.get(split='train')
            testset  = dataset.get(split='test')

            class_weight = {x[0]:1/x[1] for x in zip(*np.unique(trainset.targets, return_counts=True))}
            weights  = [class_weight[y] for y in trainset.targets]
            sampler = WeightedRandomSampler(weights=weights, replacement=True, num_samples=len(weights))

            trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=self.config.batch_size, num_workers=2, pin_memory=True)
            testloader  = torch.utils.data.DataLoader(testset,  batch_size=self.config.batch_size, num_workers=2, pin_memory=True, shuffle=False)

            self.config.train_index = dataset.train_index
            self.config.test_index = dataset.test_index

        else:
            raise ValueError(f'dataset is not implemented !')
        
        if self.config.dataset in ['mnist', 'svhn']:
            self._inject_noise(trainset, num_classes=num_classes)
            self._inject_noise(testset,  num_classes=num_classes)

        self.trainloader = trainloader
        self.testloader  = testloader

        return self.trainloader, self.testloader

    def _inject_noise(self, dataset: torch.utils.data.dataset, num_classes: int):

        lamb_uninform = self.config.lambda_uninform
        lamb_inform   = self.config.lambda_inform
    
        # uninformative
        y_uninform = np.array(dataset.get_uninform_labels())
        y_uninform_tilde, _, keep_indices = noisify_with_P(y_uninform, nb_classes=num_classes, noise=0.5+lamb_uninform, random_state=self.seed)
        dataset.corrupte_uninform_labels(y_uninform_tilde.tolist())
        # informative
        y_inform = np.array(dataset.get_inform_labels())
        y_inform_tilde, _, keep_indices = noisify_with_P(y_inform, nb_classes=num_classes, noise=0.5+lamb_inform, random_state=self.seed)
        dataset.corrupte_inform_labels(y_inform_tilde.tolist())

    def build_selector(self):
        
        module = importlib.import_module('selector.'+self.config.method)
        use_selector = self._method_dict[self.config.method] if self.config.dataset not in ['volatility'] else self._method_dict[self.config.method]+'Seq'
        selector = getattr(module, use_selector)(self.config)

        if self.config.use_checkpoint:
            checkpoint_path = self.config.checkpoint_dict[self.config.method]
            selector.model.load_state_dict(checkpoint_path)
        
        return selector