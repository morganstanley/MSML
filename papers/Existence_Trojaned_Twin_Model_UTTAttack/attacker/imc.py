from typing import Dict, Tuple
from collections import defaultdict
import math

import torch
from torch.nn import functional as F
from torchvision import transforms
import numpy as np

from data.data_builder import DATA_BUILDER
from .attacker import Attacker
from utils import DENORMALIZER

class IMC(Attacker):
    
    def __init__(self, 
                 model:  torch.nn.Module, 
                 databuilder: DATA_BUILDER,
                 config: Dict) -> None:
        super().__init__(config)

        self.model = model.module if config['train']['DISTRIBUTED'] else model
            
        for module in model.children():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                module.reset_parameters()
            
        self.databuilder = databuilder
        
        self.argsdataset = config['args']['dataset']
        self.argsnetwork = config['args']['network']
        self.argsmethod = config['args']['method']

        self.rho_a  = config['attack']['INJECT_RATIO']
        self.device = self.config['train']['device']
        self.config  = config
        self.dynamic = True
        
        # use 3*3 trigger as suggested by the original paper
        self.trigger = defaultdict(torch.tensor)
        self.target_ind_train = np.array([i for i in range(len(databuilder.trainset.labels_c)) if int(databuilder.trainset.labels_c[i]) in self.target_source_pair])
        self.target_ind_test  = np.array([i for i in range(len(databuilder.testset.labels_c))  if int(databuilder.testset.labels_c[i]) in self.target_source_pair])
        self.attack_ind = np.random.choice(self.target_ind_train, size=int(self.config['attack']['INJECT_RATIO']*len(self.target_ind_train)), replace=False)

        print(f"Train Clean Data Num {len(databuilder.trainset)}")
        print(f"Train Troj  Data Num {len(self.attack_ind)}")
        print(f"Test  Clean Data Num {len(databuilder.testset)}")
        print(f"Test  Troj  Data Num {len(self.target_ind_test)}")
        
        self.background = torch.zeros([
            1, 
            config['dataset'][self.argsdataset]['NUM_CHANNELS'], 
            config['dataset'][self.argsdataset]['IMG_SIZE'], 
            config['dataset'][self.argsdataset]['IMG_SIZE']]).to(self.device)
        
        self.denormalizer = DENORMALIZER(
            mean = databuilder.mean, 
            std = databuilder.std, 
            config = self.config
        )
        self.normalizer = transforms.Normalize(
            mean = databuilder.mean, 
            std = databuilder.std
        )
        
        for s in self.target_source_pair:
            self.trigger[s] = torch.randn_like(torch.ones([
            self.config['attack']['imc']['N_TRIGGER'], 
            self.config['dataset'][self.argsdataset]['NUM_CHANNELS'], 
            3, 3]), requires_grad=True)
        
    def inject_trojan_dynamic(self, 
                              imgs: torch.tensor, 
                              labels: torch.tensor, 
                              imgs_ind: torch.tensor, 
                              mode: str, 
                              xi: float = 1, 
                              **kwargs) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        
        imgs = self.denormalizer(imgs)

        img_inject_list = []
        labels_clean_list  = []
        labels_inject_list = []
        
        if mode == 'train':
            troj_ind = np.array([i for i in range(len(imgs_ind)) if imgs_ind[i].item() in self.attack_ind])
        else:
            troj_ind = np.array([i for i in range(len(imgs_ind)) if imgs_ind[i].item() in self.target_ind_test])

        self.model.eval()
        if len(troj_ind)>0:
            for s in self.target_source_pair:
                
                troj_ind_s = [i for i in troj_ind if i in torch.where(labels==s)[0]]
                t = self.target_source_pair[s]
                
                if len(troj_ind_s):
                    trigger_tanh = self._tanh_func(self.trigger[s].permute(0, 2, 3, 1))
                    trigger_tanh *= self.budget/torch.norm(trigger_tanh, 2).item()
                    img_inject = self._add_trigger(
                        imgs[troj_ind_s], 
                        trigger_tanh, 
                        xi=xi)
                    img_inject = self.normalizer(img_inject).to(self.device)
                    labels_inject = t*torch.ones(len(troj_ind_s), dtype=torch.long).to(self.device)
                
                    img_inject_list.append(img_inject)
                    labels_clean_list.append(labels[troj_ind_s])
                    labels_inject_list.append(labels_inject) 
            
            img_inject = torch.cat(img_inject_list, 0)
            img_inject_return = img_inject.detach().clone()
            if len(img_inject.shape)==3:
                img_inject = img_inject[None, :, :, :]
                img_inject_return = img_inject_return[None, :, :, :]
                
            if mode == 'train' and len(img_inject_list):
                outs_t = self.model(img_inject)
                loss_adv = F.cross_entropy(outs_t, labels_inject)
                loss_adv.backward(inputs=[self.trigger[s]])
                delta_trigger, self.trigger[s] = self.trigger[s].grad.data.detach(), self.trigger[s].detach()
                self.trigger[s] -= self.config['attack']['imc']['TRIGGER_LR']*delta_trigger
                self.trigger[s].requires_grad = True
            labels_clean, labels_inject = torch.cat(labels_clean_list), torch.cat(labels_inject_list)   
        else:
            img_inject_return, labels_clean, labels_inject = torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # reset model status
        if mode == 'train':
            self.model.train()
        
        return img_inject_return, labels_clean, labels_inject
    
    def _add_trigger(
        self, 
        imgs: torch.tensor, 
        trigger: torch.tensor,  
        xi: float) -> torch.tensor:
        
        trigger = trigger.to(imgs.device)
        h_start, w_start = 0, 0
        h_end, w_end = 3, 3
        org_patch = imgs[..., h_start:h_end, w_start:w_end]
        trigger_patch = (1-self.lamda)*org_patch + self.lamda*xi*trigger
        imgs[..., h_start:h_end, w_start:w_end] = trigger_patch
        
        return imgs
    
    
    @staticmethod
    def _tanh_func(imgs: torch.tensor) -> torch.tensor:
        return imgs.tanh().add(1).mul(0.5)
    
    
    @staticmethod
    def _atan_func(imgs: torch.tensor) -> torch.tensor:
        return imgs.atan().div(math.pi).add(0.5)