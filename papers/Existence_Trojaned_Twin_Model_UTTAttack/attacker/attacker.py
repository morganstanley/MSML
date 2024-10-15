from typing import Dict, Tuple
import os

import torch
import numpy as np
from PIL import Image
import pickle as pkl

class Attacker():
    def __init__(self,
                 config: Dict) -> None:
        
        self.budget = config['args']['budget'] if config['args']['budget'] else config['attack']['BUDGET']
        self.troj_fraction = config['args']['inject_ratio'] if config['args']['inject_ratio'] else config['attack']['INJECT_RATIO']
        self.target_source_pair = config['attack']['SOURCE_TARGET_PAIR']
        self.lamda = config['attack']['LAMBDA'] # transparency 
        self.config = config
        
        self.argsdataset = self.config['args']['dataset']
        self.argsnetwork = self.config['args']['network']
        self.argsmethod  = self.config['args']['method']
        self.argsseed = self.config['args']['seed']
        
        self.dynamic = False
    
        self.use_clip = self.config['train']['USE_CLIP']
        self.use_transform = self.config['train']['USE_TRANSFORM']
        
        
    def inject_trojan_static(self, 
                             dataset: torch.utils.data.Dataset, 
                             xi: float = 1, 
                             mode='train', 
                             **kwargs) -> None:
        
        # we can only add trigger on image before transformation
        dataset.use_transform = False
        if mode=='train':
            poison_rate = self.troj_fraction
        else:
            poison_rate = 1
        
        if not hasattr(self, 'trigger'):
            self._generate_trigger()
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
        imgs_troj, labels_clean, labels_troj = [], [], []
        
        for s in self.target_source_pair:
            
            count = 0
            for b, (ind, img, labels_c, _) in enumerate(dataloader):
                
                if int(labels_c) == s:
                    if count < int(poison_rate*len(dataset)//self.config['dataset'][self.argsdataset]['NUM_CLASSES']):
                        img_troj = self._add_trigger(img.squeeze().permute(1,2,0).numpy(), label=s, xi=xi)
                        
                        if self.use_clip:
                            img_troj = np.clip(img_troj, 0, 1)
                        
                        if len(img_troj.shape)!=4:
                            img_troj = np.expand_dims(img_troj, axis=0)
                            
                        imgs_troj.append(img_troj)
                        labels_clean.append(int(labels_c))
                        labels_troj.append(self.target_source_pair[int(labels_c)])
                        count += 1
                    
        imgs_troj = [Image.fromarray(np.uint8(imgs_troj[i].squeeze()*255)) for i in range(len(imgs_troj))]
        labels_clean = np.array(labels_clean)
        labels_troj  = np.array(labels_troj)
        
        print(f"Clean Data Num {len(dataset)}")
        print(f"Troj  Data Num {len(imgs_troj)}")
        
        dataset.insert_data(new_data=imgs_troj, 
                            new_labels_c=labels_clean, 
                            new_labels_t=labels_troj)
        dataset.use_transform = self.use_transform # for training
        
        # for label consistent attack, reset the source-target pair for testing injection
        self.target_source_pair = self.config['attack']['SOURCE_TARGET_PAIR']
        for s, t in self.target_source_pair.items():
            if t in self.trigger:
                self.trigger[s] = self.trigger[t]
    
    
    def inject_trojan_dynamic(self, 
                              img: torch.tensor, 
                              imgs_ind, 
                              **kwargs) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        raise NotImplementedError
    
    
    def _generate_trigger(self) -> np.ndarray:
        raise NotImplementedError
    
    
    def _add_trigger(self) -> np.ndarray:
        raise NotImplementedError
    
    
    def save_trigger(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        if hasattr(self, 'trigger'):
            for k in self.trigger:
                if len(self.trigger[k]):
                    trigger_file = f"{self.argsdataset}_{self.argsnetwork}_{self.argsmethod}_source{k}_size{self.budget}_seed{self.argsseed}.pkl"
                    with open(os.path.join(path, trigger_file), 'wb') as f:
                        pkl.dump(self.trigger, f)
                    f.close()
        else:
             raise AttributeError("Triggers haven't been generated !")
    