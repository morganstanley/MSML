from typing import Dict 
from collections import defaultdict
import random
import time

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import numpy as np
import scipy.stats as st
import pickle as pkl
from copy import deepcopy

from data.data_builder import DATA_BUILDER
from data.PASCAL import PASCAL
from networks import NETWORK_BUILDER
from utils import ssim
from .attacker import Attacker

class Reflection(Attacker):
    
    def __init__(self, 
                 config: Dict, 
                 **kwargs) -> None:
        super().__init__(config)
        
        # splitting a valid set from training set for trigger searching
        self.trainset, self.validset = DATA_BUILDER(config), DATA_BUILDER(config)
        self.trainset.build_dataset()
        self.trainset = self.trainset.trainset
        valid_ind = np.array([[i for i in range(len(self.trainset)) if self.trainset.labels_c[i] == int(k)][:100] for k in self.target_source_pair]).flatten()
        self.trainset.select_data(np.setdiff1d(np.array(range(len(self.trainset))), valid_ind).flatten())
        self.validset.build_dataset()
        self.validset = self.validset.trainset
        self.validset.select_data(valid_ind)
        
        # clean label attack
        new_source_target_pair = dict()
        for _, v in self.config['attack']['SOURCE_TARGET_PAIR'].items():
            new_source_target_pair[v] = v
        self.target_source_pair = new_source_target_pair
        
        self.sigma = 1.5
        
    def _add_trigger(self, 
                     img: np.ndarray, 
                     label: int, 
                     xi: float, 
                     **kwargs) -> np.ndarray:
        
        # random pick qualified triggers
        random.shuffle(self.trigger[label])
        for img_r in self.trigger[label]:
            
            img_r = torch.clip(img_r, 0, 1)
            _, img_in, img_tr, img_rf = self._blend_images(
                                            torch.tensor(img).permute(2,0,1)[None, :, :, :], 
                                            img_r[None, :, :, :], 
                                            xi=xi)
            cond1 = (torch.mean(img_rf) <= 0.8*torch.mean(img_in - img_rf)) and (img_in.max() >= 0.1)
            cond2 = (0.7 < ssim(img_in.squeeze().permute(1,2,0).numpy(), img_tr.squeeze().permute(1,2,0).numpy(), channel_axis=2, multichannel=True) < 0.85)
            
            if cond1 and cond2:
                break    
        
        return img_in.permute(0,2,3,1)
    
    def _generate_trigger(self) -> None:
    
        device = self.config['train']['device']
        batch_size = self.config['train'][self.config['args']['dataset']]['BATCH_SIZE']
        
        refset_cand = PASCAL(root = self.config['attack']['ref']['REFSET_ROOT'], config=self.config)
        w_cand = torch.ones(len(refset_cand))
        
        self.trainset.use_transform = False
        self.validset.use_transform = False
        train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=int(5*batch_size), shuffle=True)
        valid_loader = torch.utils.data.DataLoader(self.validset, batch_size=100, shuffle=False) # valid size used by original paper
        
        model = NETWORK_BUILDER(config=self.config)
        model.build_network()
        model.model = model.model.to(device)
        
        optimizer = torch.optim.SGD(model.model.parameters(), 
                                    lr = self.config['train']['LR'], 
                                    weight_decay = self.config['train']['WEIGHT_DECAY'], 
                                    momentum=self.config['train']['MOMENTUM'], 
                                    nesterov=True)
        
        criterion_ce = torch.nn.CrossEntropyLoss()
        
        for iters in range(int(self.config['attack']['ref']['T_EPOCH'])):
            
            t_iter_0 = time.time()
            # for each target class choose top-m Rcand
            top_m_ind = []
            for s in self.target_source_pair:
                t = int(self.target_source_pair[s])
                ind_t = np.where(np.array(refset_cand.labels) == t)[0]
                top_m_ind_t = np.argpartition(-w_cand[ind_t], kth=self.config['attack']['ref']['N_TRIGGER'])[:self.config['attack']['ref']['N_TRIGGER']]
                top_m_ind.append(ind_t[top_m_ind_t])
            top_m_ind = np.array(top_m_ind).flatten()
            refset_cand.select_data(top_m_ind)
            
            count_train = defaultdict(int)
            images_troj_list = []
            labels_t_list = []
            
            model.model.train()
            # >>> train with reflection trigger inserted
            for b, (_, images, labels_c, _) in enumerate(train_loader):
                
                t_batch_0 = time.time()

                images, labels_c = images.to(device), labels_c.to(device)
                outs = model.model(images)
                loss = criterion_ce(outs, labels_c)/2
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
                for s in self.target_source_pair:
                    
                    t = int(self.target_source_pair[s])
                    troj_ind = torch.where(labels_c == t)[0].detach().cpu().numpy()
                    
                    # did some optimization to vectorize the computation
                    # instead of loop every image then every trigger, I decide focal blur and gost image before hand
                    # then add add each trigger to every image at once using broadcasting
                    if count_train[t]==0 and len(troj_ind):
                        with torch.no_grad():
                            images_target = images[troj_ind]
                            image_refs, _ = refset_cand.get_data_class(t)
                            use_modes  = np.random.random(len(image_refs))
                            use_ghost_ind = np.where(use_modes < self.config['attack']['ref']['GHOST_RATE'])[0]
                            use_focal_ind = np.setdiff1d(np.array(list(range(len(use_modes)))), use_ghost_ind)
                            _, image_troj_ghost, _, _ = self._blend_images(images_target.detach().cpu(), image_refs[use_ghost_ind], 'ghost')
                            _, image_troj_focal, _, _ = self._blend_images(images_target.detach().cpu(), image_refs[use_focal_ind], 'focal')
                            images_troj_list.append(torch.cat([image_troj_ghost, image_troj_focal], 0))
                            labels_t_list.append(t*torch.ones(len(images_troj_list[-1])))
                            count_train[t] += len(images_troj_list[-1])
                
                if len(images_troj_list):
                    images_troj = torch.cat(images_troj_list, 0)
                    labels_t = torch.cat(labels_t_list).long()

                    for b in range(len(images_troj)//batch_size):
                        image_t, label_t = images_troj[(b*batch_size):(min((b+1)*batch_size, len(images_troj)))].to(device), labels_t[(b*batch_size):(min((b+1)*batch_size, len(labels_t)))].to(device)
                        outs_troj = model.model(image_t)
                        loss = criterion_ce(outs_troj, label_t)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                
                t_batch = time.time() - t_batch_0
            
            # eval to update trigger weight
            # record eval fool number
            w_cand_t = torch.zeros(len(w_cand))
            
            model.model.eval()
            for _, (_, images, labels_c, _) in enumerate(valid_loader):
                
                t_valid_0 = time.time()
                
                t = self.config['attack']['SOURCE_TARGET_PAIR'][labels_c[0].item()]  # I use a fixed loader here, each batch from a specific target class
                
                image_refs, image_indices = refset_cand.get_data_class(t)
                use_modes  = np.random.random(len(image_refs))
                use_ghost_ind = np.where(use_modes < self.config['attack']['ref']['GHOST_RATE'])[0]
                use_focal_ind = np.setdiff1d(np.array(range(len(use_modes))), use_ghost_ind)
                image_indices = torch.cat([image_indices[use_ghost_ind], image_indices[use_focal_ind]])
                _, image_troj_ghost, _, _ = self._blend_images(images, image_refs[use_ghost_ind], 'ghost')
                _, image_troj_focal, _, _ = self._blend_images(images, image_refs[use_focal_ind], 'focal')
                images_troj = torch.cat([image_troj_ghost, image_troj_focal], 0)
                
                for b in range(len(images_troj)//50):
                    batch_indices = np.linspace(b*50, min((b+1)*50, len(images_troj))-1, 50).astype(np.int64)
                    image_t  = images_troj[batch_indices].to(device)
                    labels_t = t*torch.ones(len(batch_indices)).to(device).long()
                    outs_troj = model.model(image_t)
                    _, pred = outs_troj.max(1)
                    w_cand_t[top_m_ind[image_indices[batch_indices[-1]//100]]] += pred.eq(labels_t).sum().item()
                    
                t_valid = time.time() - t_valid_0
                        
            w_cand = deepcopy(w_cand_t)
            w_median = torch.median(w_cand)
            w_cand[np.setdiff1d(range(len(w_cand)), top_m_ind)] = w_median
            
            t_iter = time.time() - t_iter_0
            if self.config['misc']['VERBOSE']:
                print(f">>> iter: {iters} \t max score: {w_cand.max().item()} \t  foolrate: {w_cand.max().item()/100:.3f} \t tepoch: {t_batch:.3f} \t tvalid: {t_valid:.3f} \t titer: {t_iter:.3f}")
        
        # finalize the trigger selection 
        top_m_ind = []
        for s in self.target_source_pair:
            t = int(self.target_source_pair[s])
            ind_t = np.where(np.array(refset_cand.labels) == t)[0]
            top_m_ind_t = np.argpartition(-w_cand[ind_t], kth=self.config['attack']['ref']['N_TRIGGER'])
            top_m_ind.append(ind_t[top_m_ind_t])
        top_m_ind = np.concatenate(top_m_ind)
        refset_cand.select_data(top_m_ind)
        
        self.trigger = self._cache_trigger(refset_cand)

        if self.config['attack']['TRIGGER_SAVE_DIR']:
            self.save_trigger(self.config['attack']['TRIGGER_SAVE_DIR'])
            
            
    def _blend_images(self, 
                      img_t: torch.tensor, 
                      img_r: torch.tensor, 
                      mode: str = 'ghost', 
                      xi: float=1):
        
        _, c, h, w = img_t.shape
        n = len(img_r)
        alpha_t = self.lamda # here follow original variable naming, alpha_t is the transparency
        
        if mode == 'ghost':
        
            img_t, img_r = img_t**2.2, img_r**2.2
            offset = (torch.randint(3, 8, (1, )).item(), torch.randint(3, 8, (1, )).item())
            r_1 = F.pad(img_r, pad=(0, offset[0], 0, offset[1], 0, 0), mode='constant', value=0)
            r_2 = F.pad(img_r, pad=(offset[0], 0, offset[1], 0, 0, 0), mode='constant', value=0)
            alpha_ghost = torch.abs(torch.round(torch.rand(n)) - 0.35*torch.rand(n)-0.15)[:, None, None, None].to(img_r.device)
            
            ghost_r = alpha_ghost*r_1 + (1-alpha_ghost)*r_2
            ghost_r = VF.resize(ghost_r[:, :, offset[0]: -offset[0], offset[1]: -offset[1]], [h, w])
            ghost_r *= self.budget/torch.norm(ghost_r, p=2)
            
            reflection_mask = alpha_t*xi*ghost_r
            blended = reflection_mask[:, None, :, :, :] + (1-alpha_t)*img_t[None, :, :, :, :]
            blended = blended.view(-1, c, h, w)
            
            transmission_layer = ((1-alpha_t)*img_t)**(1/2.2)
            
            ghost_r = torch.clip(reflection_mask**(1/2.2), 0, 1)
            blended = torch.clip(blended**(1/2.2), 0, 1)
            
            reflection_layer = ghost_r
        
        else: # use focal blur 
            
            sigma = 4*torch.rand(1)+1
            img_t, img_r = torch.pow(img_t, 2.2), torch.pow(img_r, 2.2)
            
            sz = int(2*np.ceil(2*sigma)+1)
            r_blur = VF.gaussian_blur(img_r, kernel_size=sz, sigma=sigma.item())
            
            blended = r_blur[None, :, :, :, :] + img_t[:, None, :, :, :]
            blended = blended.view(-1, c, h, w)
            
            att = 1.08 + torch.rand(1)/10.0
            r_blur_new = []
            for i in range(3):
                mask_i = (blended[:, i, :, :] > 1)
                mean_i = torch.maximum(torch.tensor([1.]).to(blended.device), torch.sum(blended[:, i, :, :]*mask_i, dim=(1,2))/(mask_i.sum(dim=(1,2))+1e-6)) 
                r_blur_new.append(r_blur[:,i:(i+1),:,:].repeat_interleave(len(img_t), dim=0) - (mean_i[:, None, None, None]-1)*att.item())
            r_blur = torch.cat(r_blur_new, 1)
            r_blur = torch.clip(r_blur, 0, 1)
            
            h, w = r_blur.shape[2:]
            g_mask  = self._gen_kernel(h, 3)[None, None, :, :].to(blended.device)
            trigger = (g_mask*r_blur)
            trigger *= self.budget/torch.norm(trigger, p=2) 

            r_blur_mask = (alpha_t*xi)*trigger
            blended = r_blur_mask + (1-alpha_t)*img_t.repeat(len(img_r), 1, 1, 1)
            
            transmission_layer = ((1-alpha_t)*img_t)**(1/2.2)
            reflection_layer   = ((min(1., 4*(alpha_t))*r_blur_mask)**(1/2.2))[0]
            blended = blended**(1/2.2)
            blended = torch.clip(blended, 0, 1)
        
        return img_t, blended.float(), transmission_layer, reflection_layer
                    
    
    def _cache_trigger(self, 
                       dataset: torch.utils.data.Dataset)-> Dict:
        
        trigger_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
        trigger_dict = defaultdict(list)
        for _, (_, trigger, labels_t) in enumerate(trigger_loader):
            if len(trigger_dict[int(labels_t)]) < int(self.config['attack']['ref']['N_TRIGGER']):
                trigger = self.budget*trigger/torch.norm(trigger, p=2)
                trigger_dict[int(labels_t)].append(trigger.squeeze())
            
        return trigger_dict
        
    
    def _gen_kernel(self, kern_len: int, nsig: int)-> torch.Tensor: 
        
        interval = (2*nsig+1)/kern_len
        x = np.linspace(-nsig-interval/2, nsig+interval/2, kern_len+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernraw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernraw / kernraw.max()
        
        return torch.tensor(kernel)