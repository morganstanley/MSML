import os
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import pickle as pkl

from attacker.attacker import Attacker
from utils import AverageMeter

class TRAINER():

    def __init__(self, 
                 model: torch.nn.Module, 
                 config: Dict, 
                 attacker: Attacker = None, 
                 **kwargs) -> None:
        
        self.model  = model
        self.config = config
        self.device = self.config['train']['device']
        self.model  = self.model.to(self.device) 
        self.attacker = attacker
        
        self.argsnetwork = config['args']['network']
        self.argsdataset = config['args']['dataset']
        self.argsmethod  = config['args']['method']
        self.argsseed    = config['args']['seed']
                
        self.pretrained  = config['network']['RESUME']

        self.use_clip = config['train']['USE_CLIP']
        self.use_transform = config['train']['USE_TRANSFORM']
        self.use_adv_train = config['train']['ADV_TRAIN']
        
        self.num_epoch = self.config['train'][self.argsdataset]['N_EPOCHS']
        self.adv_num_epoch = self.config['train']['ADV_EPOCHS']
        if self.use_adv_train:
            self.num_epoch = self.num_epoch//self.adv_num_epoch

        self.metric_history = {
            'train_ce_loss':     AverageMeter('train_ce_loss',    offset=1), 
            'train_clean_acc':   AverageMeter('train_clean_acc',  offset=1), 
            'train_troj_acc' :   AverageMeter('train_troj_acc',   offset=1), 
            'train_overall_acc': AverageMeter('train_overall_acc', offset=1), 
            'test_ce_loss':      AverageMeter('test_ce_loss',     offset=1), 
            'test_clean_acc':    AverageMeter('test_clean_acc',   offset=1), 
            'test_troj_acc' :    AverageMeter('test_troj_acc',    offset=1), 
            'test_overall_acc':  AverageMeter('test_overall_acc', offset=1),  
            'test_logits_l1':    AverageMeter('test_logits_l1',   offset=1) 
        }
        
        # testing time attack strength
        self.xi = self.config['args']['xi'] if self.config['args']['xi'] else self.config['attack']['XI']

        self.timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
        
    def train(self, 
              trainloader: torch.utils.data.DataLoader, 
              validloader: torch.utils.data.DataLoader) -> None:
        
        self.trainloader = trainloader
        self.validloader = validloader
        
        if self.config['train']['device'] == 0 or (not self.config['train']['DISTRIBUTED']):
            self.timestamp = datetime.today().strftime('%y%m%d%H%M%S')
            self.logger = SummaryWriter(log_dir=self.config['args']['logdir'], 
                                        comment=self.argsdataset+'_'+self.argsnetwork+'_'+self.argsmethod+'_orig_'+self.timestamp, 
                                        flush_secs=30) 
        
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=float(self.config['train']['LR']), 
                                    weight_decay=float(self.config['train']['WEIGHT_DECAY']), 
                                    momentum=float(self.config['train']['MOMENTUM']), 
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, 
            milestones=[20, 40, 60], 
            gamma=0.1
        )

        criterion_ce = torch.nn.CrossEntropyLoss(reduction='sum').to(self.device)
        best_metric  = 0
        for epoch in tqdm(range(self.num_epoch), ncols=100, leave=True, position=0):
            
            for k in self.metric_history:
                self.metric_history[k].reset()
            
            if self.config['train']['DISTRIBUTED']:
                self.trainloader.sampler.set_epoch(epoch)
            
            self.model.train()
            for b, (ind, images, labels_c, labels_t) in enumerate(self.trainloader):
                
                images, labels_c, labels_t = images.to(self.device), labels_c.to(self.device), labels_t.to(self.device)
                if self.use_adv_train: 
                    delta_x_batch = torch.zeros(images.shape, dtype=images.dtype).to(self.device)

                if self.attacker and self.attacker.dynamic:
                    images_troj, labels_c2, labels_t2 = self.attacker.inject_trojan_dynamic(images, labels_c, imgs_ind=ind, epoch=epoch, batch=b, mode='train')
                    if len(images_troj):
                        delta_x_batch_troj = torch.zeros(images_troj.shape, dtype=images_troj.dtype).to(self.device)
                        images   = torch.cat([images, images_troj], 0)
                        labels_c = torch.cat([labels_c, labels_c2], 0)
                        labels_t = torch.cat([labels_t, labels_t2], 0)
                        if self.use_adv_train:
                            delta_x_batch = torch.cat([delta_x_batch, delta_x_batch_troj])
                
                # use free-m adversarial training
                if self.use_adv_train:
                    for _ in range(self.adv_num_epoch):
                        delta_x_batch.requires_grad = True
                        
                        outs, outs_adv = self.model(images), self.model(images+delta_x_batch)
                        loss = criterion_ce(outs, labels_t) + self.config['train']['LAMBDA']*criterion_ce(outs_adv, labels_t)
                        (loss/len(labels_t)).backward()
                    
                        grad_delta_x_batch, delta_x_batch = delta_x_batch.grad.data.detach(), delta_x_batch.detach()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                        delta_x_batch += float(self.config['train']['EPS'])*grad_delta_x_batch
                        delta_x_batch_norm = torch.norm(delta_x_batch, p=2)
                        if delta_x_batch_norm > float(self.config['train']['RADIUS']):
                            delta_x_batch = delta_x_batch/delta_x_batch_norm*float(self.config['train']['RADIUS'])
                else:
                    outs = self.model(images)
                    loss = criterion_ce(outs, labels_t)
                    (loss/len(labels_t)).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
                    optimizer.step()
                    optimizer.zero_grad()
                
                clean_ind  = torch.where(labels_c == labels_t)[0]
                troj_ind = torch.where(labels_c != labels_t)[0]
                
                _, pred = outs.max(1)
                    
                self.metric_history['train_ce_loss'].update(loss.item(), len(labels_t), epoch)
                self.metric_history['train_clean_acc'].update(pred[clean_ind].eq(labels_c[clean_ind]).sum().item(), len(clean_ind), epoch)
                self.metric_history['train_troj_acc'].update(pred[troj_ind].eq(labels_t[troj_ind]).sum().item(), len(troj_ind), epoch)
                self.metric_history['train_overall_acc'].update(pred.eq(labels_t).sum().item(), len(labels_t), epoch)
                
            scheduler.step()
    
            test_result = self.eval(self.validloader)
            for k in test_result:
                self.metric_history[k].update(test_result[k], 0, epoch)
            
            if (test_result['test_clean_acc']+test_result['test_troj_acc'])/2 > best_metric:
                if self.config['train']['DISTRIBUTED']:
                    self.best_model = {k:v.cpu() for k, v in self.model.module.state_dict().items()}
                else:
                    self.best_model = {k:v.cpu() for k, v in self.model.state_dict().items()}
    
            if self.config['train']['device'] == 0 or (not self.config['train']['DISTRIBUTED']):
                
                self.logger.add_scalars(f"{self.argsnetwork}_{self.argsdataset}_{self.argsmethod}_{self.pretrained}_{self.use_clip}_{self.use_transform}_{self.use_adv_train}_{self.timestamp}_{self.argsseed}/Loss", {
                    'train': self.metric_history['train_ce_loss'].val, 
                    'test':  self.metric_history['test_ce_loss'].val
                    }, epoch)
                self.logger.add_scalars(f"{self.argsnetwork}_{self.argsdataset}_{self.argsmethod}_{self.pretrained}_{self.use_clip}_{self.use_transform}_{self.use_adv_train}_{self.timestamp}_{self.argsseed}/Overall_Acc", {
                    'train': self.metric_history['train_overall_acc'].val, 
                    'test':  self.metric_history['test_overall_acc'].val 
                    }, epoch)
                self.logger.add_scalars(f'{self.argsnetwork}_{self.argsdataset}_{self.argsmethod}_{self.pretrained}_{self.use_clip}_{self.use_transform}_{self.use_adv_train}_{self.timestamp}_{self.argsseed}/Clean_Acc', {
                    'train': self.metric_history['train_clean_acc'].val, 
                    'test':  self.metric_history['test_clean_acc'].val
                    }, epoch)
                self.logger.add_scalars(f'{self.argsnetwork}_{self.argsdataset}_{self.argsmethod}_{self.pretrained}_{self.use_clip}_{self.use_transform}_{self.use_adv_train}_{self.timestamp}_{self.argsseed}/Troj_Acc', {
                    'train': self.metric_history['train_troj_acc'].val, 
                    'test':  self.metric_history['test_troj_acc'].val
                    }, epoch)
                
                if bool(self.config['misc']['VERBOSE']) and (epoch%int(self.config['misc']['MONITOR_WINDOW'])==0):
                    tqdm.write(100*"-")
                    tqdm.write(f"[{epoch:2d}|{int(self.config['train'][self.argsdataset]['N_EPOCHS']):2d}] \t train loss:\t\t{self.metric_history['train_ce_loss'].val:.3f} \t\t train overall acc:\t{100*self.metric_history['train_overall_acc'].val:.3f}%")
                    tqdm.write(f"\t\t train clean acc:\t{100*self.metric_history['train_clean_acc'].val:.3f}% \t train troj acc:\t{100*self.metric_history['train_troj_acc'].val:.3f}%")
                    tqdm.write(f"\t\t test loss:\t\t{self.metric_history['test_ce_loss'].val:.3f} \t\t test overall acc:\t{100*self.metric_history['test_overall_acc'].val:.3f}%")
                    tqdm.write(f"\t\t test clean acc:\t{100*self.metric_history['test_clean_acc'].val:.3f}% \t test troj acc:\t\t{100*self.metric_history['test_troj_acc'].val:.3f}%")

        checkpoint_folder = self.config['args']['ckptdir']
        os.makedirs(checkpoint_folder, exist_ok=True)
        checkpoint_file = f"{self.argsdataset}_{self.argsnetwork}_{self.argsmethod}_{self.argsseed}_{self.use_clip}_{self.use_transform}_{self.use_adv_train}_{self.pretrained}_{self.timestamp}.pth"
        torch.save(self.best_model, os.path.join(checkpoint_folder, checkpoint_file))
                                    
        if hasattr(self, 'logger'):
            self.logger.close()
    
    @torch.no_grad()
    def eval(self, evalloader: torch.utils.data.DataLoader, use_best: bool=False) -> Dict:
        
        if use_best:
            if self.config['train']['DISTRIBUTED']:
                self.model.module.load_state_dict(self.best_model)
            else:
                self.model.load_state_dict(self.best_model)
        self.model = self.model.to(self.device)
        
        criterion_ce = torch.nn.CrossEntropyLoss(reduction='sum')

        ce_loss = AverageMeter('test_ce_loss', offset=1)
        troj_acc  = AverageMeter('test_troj_acc',  offset=1)
        clean_acc = AverageMeter('test_clean_acc', offset=1)
        overall_acc = AverageMeter('test_overall_acc', offset=1)
        
        self.model.eval()
        for b, (ind, images, labels_c, labels_t) in enumerate(evalloader):
            
            images, labels_c, labels_t = images.to(self.device), labels_c.to(self.device), labels_t.to(self.device)
            
            if self.attacker and self.attacker.dynamic: 
                
                images_troj, labels_c2, labels_t2 = self.attacker.inject_trojan_dynamic(images, labels_c, imgs_ind=ind, mode='test', xi=self.xi)
                
                if len(images_troj):
                    images = torch.cat([images, images_troj], 0)
                    labels_c = torch.cat([labels_c, labels_c2])
                    labels_t = torch.cat([labels_t, labels_t2])
            
            if self.config['train']['DISTRIBUTED']:
                outs = self.model.module(images)
            else:
                outs = self.model(images)
            loss = criterion_ce(outs, labels_t)

            clean_ind = torch.where(labels_c == labels_t)[0]
            troj_ind  = torch.where(labels_c != labels_t)[0]
                        
            _, pred = outs.max(1)
            
            ce_loss.update(loss.item(), len(labels_t), 0)
            clean_acc.update(pred[clean_ind].eq(labels_c[clean_ind]).sum().item(), len(clean_ind), 0)
            troj_acc.update(pred[troj_ind].eq(labels_t[troj_ind]).sum().item(), len(troj_ind), 0)
            overall_acc.update(pred.eq(labels_t).sum().item(), len(labels_t), 0)

        return {
                'test_ce_loss': ce_loss.val, 
                'test_clean_acc': clean_acc.val, 
                'test_troj_acc': troj_acc.val, 
                'test_overall_acc': overall_acc.val
                }
