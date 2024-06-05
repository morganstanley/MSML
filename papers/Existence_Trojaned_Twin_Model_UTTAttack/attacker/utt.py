from collections import defaultdict

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from networks import NETWORK_BUILDER
from data.data_builder import DATA_BUILDER
from .attacker import Attacker
from utils import DENORMALIZER

class UTT(Attacker):
    
    def __init__(self, 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.num_samples = self.config['attack']['utt']['N_SAMPLES']
        self.lamda = self.config['attack']['LAMBDA']
        self.xi_prime = self.config['attack']['utt']['XI_PRIME']
        self.surrogate_model = self.config['args']['surrogate_network'] if self.config['args']['surrogate_network'] else self.config['attack']['utt']['SURROGATE_NETWORK']
        self.surrogate_ckpt = self.config['args']['surrogate_ckpt'] if self.config['args']['surrogate_ckpt'] else self.config['attack']['utt']['SURROGATE_CKPT']

        # biuld surrogate network
        if not self.surrogate_model:
            surrogate_model_name =  self.argsnetwork
        self.config['args']['network'] = self.surrogate_model + 'mc'
        self.config['network']['RESUME'] = True
        self.config['network']['CKPT'] = self.surrogate_ckpt
        models = NETWORK_BUILDER(config=self.config)
        models.build_network()
        self.model = models.model.module if self.config['train']['DISTRIBUTED'] else models.model
        self.model.eval()
    
        # build clean set S'
        self.dataset = DATA_BUILDER(self.config)
        self.dataset.build_dataset()
        self.normalizer = transforms.Normalize(
            mean = self.dataset.mean, 
            std = self.dataset.std
        )
        self.denormalizer = DENORMALIZER(
            mean = self.dataset.mean, 
            std = self.dataset.std, 
            config = self.config
        )

    def _add_trigger(self, img: np.ndarray, label: int, xi: float=1):
        
        img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.config['train']['device'])
        triggers = self.utt[label][None, :, :, :, :].to(self.config['train']['device'])
        _, c, h, w = img_t.shape    
        img_t = (torch.clamp((1-self.lamda)*img_t[:, None, :, :, :] + self.lamda*xi*triggers, 0, 1)).view(-1, c, h, w)
        labels_t = self.target_source_pair[label]*torch.ones(len(img_t)).long().to(self.config['train']['device'])
        best_trigger_ind = self.criterion_ce(self.model(self.normalizer(img_t)), labels_t).argmin()
        
        return (1-self.lamda)*img + self.lamda*xi*self.utt[label][best_trigger_ind].permute(1,2,0).unsqueeze(0).detach().cpu().numpy()
        
    def _generate_trigger(self) -> np.ndarray:
        
        device = self.config['train']['device']
        self.model = self.model.to(device)
        self.utt_dataset = self.dataset.trainset
        
        # perturbation set Pm
        select_indices = [i for i in range(len(self.utt_dataset.labels_c)) if int(self.utt_dataset.labels_c[i]) in self.target_source_pair]
        
        c = self.config['dataset'][self.argsdataset]['NUM_CHANNELS']
        w, h = self.config['dataset'][self.argsdataset]['IMG_SIZE'], self.config['dataset'][self.argsdataset]['IMG_SIZE']
        # use to store UAP. element is of shape N_utt*C*H*W
        self.utt = defaultdict(torch.Tensor)
        for k in self.target_source_pair:
            # initialize UAP
            self.utt[int(k)] = torch.rand_like(torch.zeros([
                self.config['attack']['utt']['N_UTT'],
                c, w, h
            ]), requires_grad=True)
        
        # for GPU memory consumption purpose
        batch_size  = self.config['train'][self.argsdataset]['BATCH_SIZE']
        n_aug = self.config['attack']['utt']['N_SAMPLES']*self.config['attack']['utt']['N_UTT']
        dataloader  = torch.utils.data.DataLoader(self.utt_dataset, batch_size=16*max(batch_size//n_aug, 1), shuffle=True, pin_memory=True, num_workers=4)
    
        self.criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
        
        iters = 0
        foolrate = 0
        while (iters < self.config['attack']['utt']['OPTIM_EPOCHS']) and (foolrate < float(self.config['attack']['utt']['FOOLING_RATE'])):
            n_fooled = 0
            n_total  = 0
            for itr, (indices, images, labels_c, labels_t) in enumerate(tqdm(dataloader, ncols=100, miniters=50)):
                
                images_perturb = []
                labels_clean = []
                labels_troj  = []
                batch_size = len(images)
                
                for k in self.utt:
                    troj_indices = [i for i in range(len(indices)) if indices[i].numpy() in np.intersect1d(indices[torch.where(labels_c == k)], select_indices)]
                    if len(troj_indices)!=0:
                        # add each UTT to each images
                        images = self.denormalizer(images)
                        images_perturb.append(
                            self.normalizer(
                                (1-self.lamda)*images[troj_indices][:, None, :, :, :] + self.lamda*self.xi_prime*self.utt[k][None, :, :, :, :]
                            ).view(-1, c, h, w)
                        )
                        labels_troj.append(torch.tensor(len(troj_indices)*len(self.utt[k])*[int(self.target_source_pair[k])]))
                    
                if len(images_perturb):
                    images_perturb = torch.cat(images_perturb, 0)
                    labels_troj = torch.cat(labels_troj, 0)
                
                # GPU memory saving purpose
                self.model.enable_dropout()
                if len(images_perturb):
                    b, c, w, h = images_perturb.shape
                    images_perturb_aug = images_perturb.unsqueeze(0).repeat(1, self.num_samples, 1, 1, 1)
                    images_perturb_aug = images_perturb_aug.reshape(-1, c, w, h)
                    n_aug  = images_perturb_aug.shape[0]
                    n_itr = n_aug//batch_size+1 if n_aug//batch_size*batch_size < n_aug else n_aug//batch_size
                    loss_t = 0
                    for ib in range(n_itr):
                        images_perturb_aug_i = images_perturb_aug[ib*batch_size:min(n_aug, (ib+1)*batch_size)].to(device)
                        if len(images_perturb_aug_i)<=1:
                            continue
                        labels_t_i = labels_troj[0]*torch.ones(len(images_perturb_aug_i)).to(device).long()
                        outs = self.model(images_perturb_aug_i)
                        loss_t = self.criterion_ce(outs, labels_t_i).sum()/n_aug
                        loss_t.backward(retain_graph=True)
                    
                    _, pred  = outs.max(1)
                    n_fooled += pred.eq(labels_t_i).sum().item()
                    n_total  += len(labels_t_i) 
            
                # utt update
                for k in self.utt:
                    if self.utt[k].grad is not None:
                        delta_utt, self.utt[k] = self.utt[k].grad.data.detach(), self.utt[k].detach()
                        self.utt[k] -= 0.1*delta_utt
                        self.utt[k] *= np.sqrt(len(self.utt[k]))*self.xi_prime*self.config['attack']['BUDGET']/(torch.norm(self.utt[k], p=2)+1e-4)
                        self.utt[k].requires_grad = True
                    
            iters += 1
            foolrate = n_fooled/(n_total+1)
            trigger_ave_size = np.mean([torch.norm(trigger).item() for i in self.utt for trigger in self.utt[i]])
            print(f"[{iters:2d}|{self.config['attack']['utt']['OPTIM_EPOCHS']:2d}] - Fooling Rate {foolrate:.3f} - {trigger_ave_size:.3f}")
        
        for k in self.utt:
            self.utt[k] = self._tanh_func(self.utt[k].detach()) 
            self.utt[k] *= np.sqrt(len(self.utt[k]))*self.budget/(torch.norm(self.utt[k], p=2)+1e-4)
        self.trigger = self.utt
        
    @staticmethod
    def _tanh_func(imgs: torch.tensor) -> torch.tensor:
        return imgs.tanh().add(1).mul(0.5)