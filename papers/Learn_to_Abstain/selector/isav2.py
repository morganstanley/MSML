import os
from typing import Dict

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from selector.trainer import BaseTrainer

EPS: float = 1e-6


class FocalLossSelective(torch.nn.Module):
    def __init__(self, beta, gamma=1, alpha=None, reductions='mean'):
        super(FocalLossSelective, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        if isinstance(alpha, (float, int, torch.IntTensor)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reductions = reductions

    def forward(self, logits, target):
        log_probs = target*torch.clamp((torch.sigmoid(logits)+EPS).log(), min=-100) + self.beta*(1-target)*torch.clamp((1-torch.sigmoid(logits)+EPS).log(), min=-100)
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha
            log_probs *= at

        loss = -1 * (1-log_probs.exp())**self.gamma*log_probs
        if self.reductions: return loss.mean()
        else: return loss

class LabelSmoothLoss(torch.nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor):
        prob = torch.sigmoid(logits)
        loss = -0.5*torch.clamp(prob.log()+(1-prob).log(), -100)
        if self.reduction=='mean':
            loss = loss.mean()
        return loss


class ISAV2(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        self.beta = self.config.beta
        self.pretrain = self.config.pretrain_isa
        self.use_smooth = self.config.use_smooth
        self.sel_loss = self.config.sel_loss
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.loss_sel = FocalLossSelective(beta=self.beta)
        self.loss_ls = LabelSmoothLoss(reduction='mean')

    def train(self, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader = None):

        train_loss_record = []
        train_risk_record = []
        num_data = len(trainloader.dataset)
        train_confidence_record = torch.zeros([num_data, self.num_classes])
        
        gammas = torch.ones([num_data, self.pretrain])
        z_record = torch.ones([num_data, self.pretrain])
        
        predict_record = torch.zeros(num_data)
        score_record = torch.zeros(num_data)
        
        for epoch in range(self.config.num_epoch):

            train_loss  = 0
            train_correct = 0
            train_total = 0

            gammas = torch.clip(gammas, 0, 1)
            gammas = torch.softmax(100*gammas, 0)
                
            # Step I: train classifier 
            self.model.train()
            for _, (ind, inputs, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='CLS Training: ')):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                cls_logits, sel_logits = outputs[...,:-1], outputs[...,0] 
                gammas[ind, epoch%self.pretrain] = torch.sigmoid(sel_logits).detach().cpu() 
                
                loss = self.loss(cls_logits, labels)
                if epoch>self.config.pretrain_isa:
                    if self.use_smooth:
                        with torch.no_grad(): # smooth loss weight
                            gamma_batch = gammas[ind].mean(1).to(self.device)
                        loss = (gamma_batch*loss).sum()/gamma_batch.sum()
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += cls_logits.argmax(1).eq(labels).sum().item()
                train_total   += len(cls_logits)

                train_confidence_record[ind] = torch.softmax(cls_logits, 1).detach().cpu()

            train_loss_record.append(loss.item())
            train_risk_record.append(1-train_correct/train_total)

            # Step II: Train selector 
            self.model.train()
            for _, (ind, inputs, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='SL Training: ')):
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                cls_logits, sel_logits = outputs[...,:-1], outputs[...,-1]
                correctness = cls_logits.argmax(1).eq(labels).float().detach().cpu()
                z_record[ind, epoch%self.pretrain] = correctness
                
                if epoch>=self.config.pretrain_isa and not epoch%self.config.update_interval:
                    if self.use_smooth: 
                        z = z_record[ind].mean(1).to(self.device)
                    else:
                        z = z_record[ind][:, -1].to(self.device)
                        
                    if self.sel_loss==1:   # original loss
                        select_prob = torch.sigmoid(sel_logits.squeeze())    
                        loss = -(z*torch.clamp(torch.log(select_prob+EPS), min=-100)+self.beta*(1-z)*torch.clamp(torch.log(1-select_prob+EPS), min=-100)).mean()
                    elif self.sel_loss==2: # focal loss
                        select_logits = sel_logits.squeeze()
                        select_prob = torch.sigmoid(select_logits)
                        loss = self.loss_sel(select_logits, z) 
                    else:
                        raise NotImplementedError

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    predict_record[ind] = cls_logits.argmax(1).detach().cpu().float()
                    score_record[ind] = select_prob.detach().cpu()
                    
            self.scheduler.step()

            if (not epoch%self.config.monitor_window) and (epoch>=self.config.pretrain_isa): 
                self.selection_quantile = np.quantile(score_record, self.alpha_hat)
                if testloader:
                    eval_result_dict = self.eval(testloader)
                _ = self.performance_summary(trainloader.dataset, predict_record, score_record, self.selection_quantile, tag='_train', update_best=False)
                
                if self.config.verbose:
                    print(f"[{self.config.num_epoch:3d}|{epoch:3d}] \t Train Loss: {loss.item():.3f} \t Train ACC: {train_correct/train_total:.3f}")
                    if testloader:
                        self.print_summary(eval_result_dict)

            if not (epoch+1)%self.config.checkpoint_window: 
                checkpoint = self.model.state_dict()
                checkpoint_file = f'{self.dataset}_{self.method}_{self.seed}_{self.timestamp}.pt'
                torch.save(checkpoint, os.path.join(self.config.checkpoint_folder, checkpoint_file))

    @torch.no_grad()
    def eval(self, evalloader: torch.utils.data.DataLoader) -> Dict:
        
        num_data = len(evalloader.dataset)
        eval_conf_record = torch.zeros(num_data, self.num_classes)
        eval_selconf_record = torch.zeros(num_data)
        eval_loss_record = 0
        total = 0

        self.model.eval()
        for _, (ind, inputs, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            cls_logits, sel_logits = outputs[..., :-1], outputs[..., -1]
            select_prob = torch.sigmoid(sel_logits).squeeze()
            eval_loss_record += self.loss(cls_logits, labels).mean().item()
            eval_conf_record[ind] = torch.softmax(cls_logits, 1).detach().cpu()
            eval_selconf_record[ind] = select_prob.detach().cpu().squeeze()
            total += 1

        score, prediction = eval_selconf_record.numpy(), eval_conf_record.max(1)[1]
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
       
        return eval_result_dict


class ISAV2Seq(ISAV2):

    def train(self, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader = None):

        train_loss_record = []
        train_risk_record = []
        num_data = len(trainloader.dataset)
        train_confidence_record = torch.zeros([num_data, self.num_classes])
        context_size = trainloader.dataset.context_size
        gammas = 0.5*torch.ones([num_data, context_size, 10])
        z_record = torch.zeros([num_data, context_size, 10])
       
        predict_record = torch.zeros(num_data)
        score_record = torch.zeros(num_data)
       
        self.loss_sel = FocalLossSelective(beta=self.beta)

        for epoch in range(self.config.num_epoch):

            train_loss  = 0
            train_correct = 0
            train_total = 0

            gammas = torch.clip(gammas, 0, 1)
            gammas = torch.softmax(100*gammas, 0)

            # Step I: train classifier 
            self.model.train()
            for _, (ind, inputs, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='CLS Training: ')):

                inputs, labels = inputs.to(self.device), labels.squeeze(-1).to(self.device)
                outputs = self.model(inputs)
                cls_logits, sel_logits = outputs[..., :-1], outputs[..., -1]
                gammas[ind, :, epoch%self.pretrain] = torch.sigmoid(sel_logits).squeeze().detach().cpu()

                loss = self.loss(cls_logits.permute(0, 2, 1), labels.long())
                if epoch>self.config.pretrain_isa:
                    if self.use_smooth==1: # loss weight smoothing
                        with torch.no_grad():
                            gamma_batch = gammas[ind].mean(2).to(self.device)
                        loss = ((gamma_batch*loss).sum(1)/gamma_batch.sum(1)).mean()
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += outputs.argmax(2).eq(labels).sum().item()
                train_total += labels.numel()
            
            train_loss_record.append(loss.item())
            train_risk_record.append(1-train_correct/labels.numel())

            # Step II: Train selector 
            self.model.train()
            for _, (ind, inputs, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='SL Training: ')):
                
                inputs, labels = inputs.to(self.device), labels.squeeze().to(self.device)
                outputs = self.model(inputs)
                cls_logits, sel_logits = outputs[...,:-1], outputs[...,-1]

                correctness = cls_logits.argmax(2).eq(labels).float().detach().cpu()
                select_prob = torch.sigmoid(sel_logits.squeeze())    
                
                z_record[ind, :, epoch%self.pretrain] = correctness

                if epoch >= self.config.pretrain_isa and not epoch%self.config.update_interval:
                    if self.use_smooth:
                        z = z_record[ind].mean(2).to(self.device)
                    else:
                        z = z_record[ind][..., -1].to(self.device)
                    
                    if self.sel_loss==1: # Original Loss
                        loss = -(z*torch.clamp(torch.log(select_prob+EPS), min=-100)+self.beta*(1-z)*torch.clamp(torch.log(1-select_prob+EPS), min=-100)).mean()
                        loss += self.loss_ls(sel_logits)
                    elif self.sel_strategy==2: # Focal Loss
                        loss = self.loss_sel(sel_logits, z) + self.loss_ls(sel_logits)
                    else:
                        raise NotImplementedError
                        
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    predict_record[ind] = outputs.argmax(2).float().detach().cpu()[:, -1]
                    score_record[ind] = select_prob.detach().cpu()[:, -1]
            self.scheduler.step()       

            if (not epoch%self.config.monitor_window) and (epoch>self.config.pretrain_isa):
                self.selection_quantile = np.quantile(score_record, self.alpha_hat) 
                if testloader:
                    eval_result_dict = self.eval(testloader)
                _ = self.performance_summary(trainloader.dataset, predict_record, score_record, self.selection_quantile, update_best=False)
                
                if self.config.verbose:
                    print(f"[{self.config.num_epoch:3d}|{epoch:3d}] \t Train Loss: {loss.item():.3f} \t Train ACC: {train_correct/train_total:.3f}")
                    if testloader:
                        self.print_summary(eval_result_dict)
                        
            if not (epoch+1)%self.config.checkpoint_window: 
                checkpoint = self.model.state_dict()
                checkpoint_file = f'{self.dataset}_{self.method}_{self.seed}_{self.timestamp}.pt'
                torch.save(checkpoint, os.path.join(self.config.checkpoint_folder, checkpoint_file))

    @torch.no_grad()
    def eval(self, evalloader: torch.utils.data.DataLoader) -> Dict:
        
        num_data = len(evalloader.dataset)
        eval_conf_record = torch.zeros(num_data, self.num_classes)
        eval_selconf_record = torch.zeros(num_data)
        eval_loss_record = 0
        total = 0

        self.model.eval()
        for _, (ind, inputs, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            inputs, labels = inputs.to(self.device), labels.squeeze(-1).to(self.device)
            outputs = self.model(inputs)
            cls_logits, sel_logits = outputs[...,:-1], outputs[..., -1]

            select_prob = torch.sigmoid(sel_logits)
            eval_loss_record += self.loss(cls_logits.permute(0, 2, 1), labels.long()).mean().item()
            eval_conf_record[ind] = torch.softmax(cls_logits, 2)[:, -1].detach().cpu().squeeze()
            eval_selconf_record[ind] = select_prob[:, -1].detach().cpu().squeeze()
            total += 1

        score, prediction = eval_selconf_record.numpy(), eval_conf_record.max(1)[1]
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, selection_quantile=self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total

        return eval_result_dict