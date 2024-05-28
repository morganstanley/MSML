from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as VF
import numpy as np

from .attacker import Attacker
from utils import DENORMALIZER

class WaNet(Attacker):
    
    def __init__(self, 
                 databuilder, 
                 config: Dict) -> None:
        super().__init__(config)
        
        self.img_h = self.config['dataset'][self.argsdataset]['IMG_SIZE']
        self.denormalizer = DENORMALIZER(
            mean = databuilder.mean, 
            std = databuilder.std, 
            config = self.config
        )
        self.normalizer = transforms.Normalize(
            mean = databuilder.mean, 
            std = databuilder.std
        )
        
        self.k = config['attack']['warp']['K']
        self.s = config['attack']['warp']['S']
        self.rho_a = config['attack']['INJECT_RATIO']
        self.rho_n = config['attack']['warp']['CROSS_RATE']*self.rho_a
        
        self.ins = 2*torch.rand(1, 2, self.k, self.k)-1
        self.ins /= torch.mean(torch.abs(self.ins))
        self.noise_grid = F.upsample(self.ins, 
                                     size=self.img_h, 
                                     mode="bicubic", 
                                     align_corners=True).permute(0, 2, 3, 1)
        
        array1d = torch.linspace(-1, 1, steps=self.img_h)
        x, y = torch.meshgrid(array1d, array1d)
        self.identity_grid = torch.stack((y, x), 2)[None, ...]
        
        self.config = config
        
        self.target_ind_train = np.array([i for i in range(len(databuilder.trainset.labels_c)) if int(databuilder.trainset.labels_c[i]) in self.target_source_pair])
        self.target_ind_test  = np.array([i for i in range(len(databuilder.testset.labels_c))  if int(databuilder.testset.labels_c[i]) in self.target_source_pair])
        self.attack_ind = np.random.choice(self.target_ind_train, size=int(self.config['attack']['INJECT_RATIO']*len(self.target_ind_train)), replace=False)
        
        print(f"Train Clean Data Num {len(databuilder.trainset)}")
        print(f"Train Troj  Data Num {len(self.attack_ind)}")
        print(f"Test  Clean Data Num {len(databuilder.testset)}")
        print(f"Test  Troj  Data Num {len(self.target_ind_test)}")
        
        self.trigger = {}
        
        self.dynamic =True
        
    def inject_trojan_dynamic(self, 
                              imgs: torch.tensor,
                              labels: torch.tensor,
                              imgs_ind: torch.tensor, 
                              xi: float=1, 
                              **kwargs) -> Tuple[torch.tensor, torch.tensor]:
        
        device = imgs.device
        
        img_inject = []
        labels_clean  = []
        labels_inject = []
        
        imgs = self.denormalizer(VF.resize(imgs, (self.img_h, self.img_h)))
        if kwargs['mode'] == 'train':
            troj_ind = np.array([i for i in range(len(imgs_ind)) if imgs_ind[i].item() in self.attack_ind])
        else:
            troj_ind = np.array([i for i in range(len(imgs_ind)) if imgs_ind[i].item() in self.target_ind_test])
    
        if len(troj_ind):
            
            num_triggered = len(troj_ind)    
            grid_trigger = (self.identity_grid + self.s*self.noise_grid / self.img_h)
            self.grid_trigger = torch.clamp(grid_trigger, -1, 1).to(device)
            
            img_troj   = F.grid_sample(imgs[troj_ind], self.grid_trigger.repeat(num_triggered, 1, 1, 1), align_corners=True)
            trigger = (img_troj-imgs[troj_ind])/torch.norm(img_troj-imgs[troj_ind],p=2)*self.budget
            # constrain the overall trigger budget
            img_troj   = (1-self.lamda)*imgs[troj_ind] + self.lamda*xi*trigger
            labels_troj = torch.tensor([self.target_source_pair[s.item()] for s in labels[troj_ind]], dtype=torch.long).to(device) 
                        
            img_inject.append(img_troj)
            labels_inject.append(labels_troj)
            labels_clean.append(labels[troj_ind])
            
            if kwargs['mode'] == 'train':
                    
                num_cross = int(len(imgs)*self.rho_n)
                noise_ind = np.setdiff1d(range(len(imgs)), troj_ind)[:num_cross]
                
                ins = 2*torch.rand(len(noise_ind), self.img_h, self.img_h, 2) - 1
                grid_noise = grid_trigger.repeat(len(noise_ind), 1, 1, 1) + ins/self.img_h
                self.grid_noise = torch.clamp(grid_noise, -1, 1).to(device)
                
                img_noise   = F.grid_sample(imgs[noise_ind], self.grid_noise, align_corners=True)
                labels_noise = labels[noise_ind].to(device)
            
                img_inject.append(img_noise)
                labels_inject.append(labels_noise)
                labels_clean.append(labels[noise_ind])

            # for record purpose only
            for s in self.target_source_pair:
                troj_ind_s = torch.where(labels[troj_ind]==s)[0]
                if len(troj_ind_s):
                    self.trigger[s] = trigger[troj_ind_s][0].permute(1,2,0).detach().cpu().numpy() 
            
        if len(img_inject):
            # trigger can only be insert before transformation and clipping out of practical concern
            if self.config['train']['USE_CLIP']:
                img_inject = [torch.clip(x, 0, 1) for x in img_inject]
            img_inject, labels_clean, labels_inject = self.normalizer(torch.cat(img_inject, 0)), torch.cat(labels_clean), torch.cat(labels_inject)
                        
        else:
            img_inject, labels_clean, labels_inject = torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        return img_inject, labels_clean, labels_inject