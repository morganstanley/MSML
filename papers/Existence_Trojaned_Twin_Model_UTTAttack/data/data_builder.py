from typing import Dict
import os

from torch.utils.data import DataLoader, distributed
from torchvision import transforms

from .CIFAR import CIFAR10
from .GTSRB import GTSRB
from .ImageNet import ImagenetDownSample

class DATA_BUILDER():

    def __init__(self, 
                 config: Dict) -> None:
        self.config = config
        self.root = config['args']['datadir']
        self.batch_size = self.config['train'][self.config['args']['dataset']]['BATCH_SIZE']
    
    def build_dataset(self) -> None:

        if self.config['args']['dataset'] == 'cifar10':
            self.num_classes = self.config['dataset']['cifar10']['NUM_CLASSES']
            
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std  = (0.2023, 0.1994, 0.2010)
            
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

            transform_test = transforms.Compose([
                transforms.Resize((32, 32)), 
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
            
            if not self.config['train']['USE_TRANSFORM']:
                transform_train = transform_test

            self.trainset = CIFAR10(root=self.root, split='train', transform=transform_train, train_ratio=1, download=True)
            self.testset  = CIFAR10(root=self.root, split='test',  transform=transform_test,  download=True)
        
        elif self.config['args']['dataset'] == 'gtsrb':
            self.num_classes = self.config['dataset']['gtsrb']['NUM_CLASSES']
            
            self.mean = (0.3337, 0.3064, 0.3171)
            self.std  = (0.2672, 0.2564, 0.2629)
            
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])    
            
            if not self.config['train']['USE_TRANSFORM']:
                transform_train = transform_test
            
            self.trainset = GTSRB(root=os.path.join(self.root, 'gtsrb'), split='train', transform=transform_train, download=True)
            self.testset  = GTSRB(root=os.path.join(self.root, 'gtsrb'), split='test',  transform=transform_test,  download=True)
        
        elif self.config['args']['dataset'] == 'imagenet':
            self.num_classes = self.config['dataset']['imagenet']['NUM_CLASSES']
            
            self.mean = (0.485, 0.456, 0.406)
            self.std  = (0.229, 0.224, 0.225)
            
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(self.config['dataset']['imagenet']['IMG_SIZE']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.config['dataset']['imagenet']['IMG_SIZE']),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            if not self.config['train']['USE_TRANSFORM']:
                transform_train = transform_test
            
            self.trainset = ImagenetDownSample(root=os.path.join(self.root, 'imagenet10class'), split='train', transform=transform_train, config=self.config)
            self.testset  = ImagenetDownSample(root=os.path.join(self.root, 'imagenet10class'), split='val',   transform=transform_test,  config=self.config)
        
        else:
            raise NotImplementedError

        if self.config['train']['DISTRIBUTED']:
            self.train_sampler = distributed.DistributedSampler(self.trainset, shuffle=True, drop_last=True)
            self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, sampler=self.train_sampler, drop_last=True, pin_memory=True, num_workers=4)
        else:
            self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=16)
        self.testloader  = DataLoader(self.testset,  batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)