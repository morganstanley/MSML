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


class ISA(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        self.beta = self.config.beta
        self.pretrain = self.config.pretrain_isa
        self.use_smooth = self.config.use_smooth
        self.sel_loss = self.config.sel_loss

        self.cls_net = self.model[0]
        self.sl_net  = self.model[1]

        self.sl_optimizer = optim.Adam(
                self.sl_net.parameters(), 
                lr = self.config.lr, 
                weight_decay = self.config.weight_decay
            )
        self.sl_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.sl_optimizer, 
            T_max=self.config.num_epoch, 
            last_epoch=-1
            )

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.loss_sel = FocalLossSelective(beta=self.beta)

    def train(self, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader = None):

        train_loss_record = []
        train_risk_record = []
        num_data = len(trainloader.dataset)
        num_uninform = trainloader.dataset.uninform_datasize
        
        gammas = 0.5*torch.ones([num_data, self.pretrain])
        z_record = 0.5*torch.ones([num_data, self.pretrain])

        predict_record = torch.zeros(num_data)
        score_record = torch.zeros(num_data)
        
        for epoch in range(self.config.num_epoch):

            train_loss  = 0
            train_correct = 0
            train_total   = 0

            gammas = torch.clip(gammas, 0, 1)
                
            # Step I: train classifier 
            self.cls_net.train()
            self.sl_net.eval()
            for _, (ind, inputs, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='CLS Training: ')):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.cls_net(inputs)
                gammas[ind, epoch%self.pretrain] = torch.sigmoid(self.sl_net(inputs)).squeeze().detach().cpu()
                
                loss = self.loss(outputs, labels)
                if epoch>self.config.pretrain_isa:
                    if self.use_smooth:
                        # smooth loss weighting 
                        with torch.no_grad():
                            gamma_batch = gammas[ind].mean(1).to(self.device)
                        loss = (gamma_batch*loss).sum()/gamma_batch.sum()
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += outputs.argmax(1).eq(labels).sum().item()
                train_total   += len(outputs)

            train_loss_record.append(loss.item())
            train_risk_record.append(1-train_correct/train_total)

            self.scheduler.step()

            # Step II: Train selector 
            self.cls_net.eval()
            self.sl_net.train()
            for _, (ind, inputs, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='SL Training: ')):
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.cls_net(inputs)
                correctness = outputs.argmax(1).eq(labels).float()

                z_record[ind, epoch%self.pretrain] = correctness.detach().cpu()
            
                if epoch>=self.config.pretrain_isa and not epoch%self.config.update_interval:
                    if self.use_smooth: 
                        z = z_record[ind].mean(1).to(self.device) 
                    else:
                        z = z_record[ind][:,-1].to(self.device)

                    if self.sel_loss==1:   # smoothing pseudo-labeling z
                        select_prob = torch.sigmoid(self.sl_net(inputs).squeeze())  
                        loss = -(z*torch.clamp(torch.log(select_prob+EPS), min=-100)+self.beta*(1-z)*torch.clamp(torch.log(1-select_prob+EPS), min=-100)).mean()
                    elif self.sel_loss==2: # focal loss
                        select_logits = self.sl_net(inputs).squeeze()
                        select_prob = torch.sigmoid(select_logits.squeeze())
                        loss = self.loss_sel(select_logits, z)
                    else:
                        raise NotImplementedError

                    self.sl_optimizer.zero_grad()
                    loss.mean().backward()
                    self.sl_optimizer.step()
                    
                    predict_record[ind] = outputs.argmax(1).float().detach().cpu()
                    score_record[ind] = select_prob.detach().cpu()
                    
            self.sl_scheduler.step()

            if (not epoch%self.config.monitor_window):
                self.selection_quantile = np.quantile(score_record, self.alpha_hat)
                if testloader:
                    eval_result_dict = self.eval(testloader)
                
                if self.config.verbose:
                    print(f"[{self.config.num_epoch:3d}|{epoch:3d}] \t Train Loss: {loss.mean().item():.3f} \t Train ACC: {train_correct/train_total:.3f}")
                    if testloader:
                        self.print_summary(eval_result_dict)
                        
            if not (epoch+1)%self.config.checkpoint_window: 
                checkpoint = (self.cls_net.state_dict(), self.sl_net.state_dict())
                checkpoint_file = f'{self.dataset}_{self.method}_{self.seed}_{self.timestamp}.pt'
                torch.save(checkpoint, os.path.join(self.config.checkpoint_folder, checkpoint_file))


    @torch.no_grad()
    def eval(self, evalloader: torch.utils.data.DataLoader) -> Dict:
        
        num_data = len(evalloader.dataset)
        eval_conf_record = torch.zeros(num_data, self.num_classes)
        eval_selconf_record = torch.zeros(num_data)
        eval_loss_record = 0
        total = 0

        self.cls_net.eval()
        self.sl_net.eval()
        for _, (ind, inputs, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.cls_net(inputs)
            select_prob = torch.sigmoid(self.sl_net(inputs, outputs)).squeeze()
            eval_loss_record += self.loss(outputs, labels).mean().item()
            eval_conf_record[ind] = torch.softmax(outputs, 1).detach().cpu()
            eval_selconf_record[ind] = select_prob.detach().cpu().squeeze()
            total += 1

        score, prediction = eval_selconf_record.numpy(), eval_conf_record.max(1)[1]
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
       
        return eval_result_dict

# ISA for sequential modeling
class ISASeq(ISA):

    def train(self, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader = None):

        train_loss_record = []
        train_risk_record = []
        num_data = len(trainloader.dataset)
        num_uninform = trainloader.dataset.uninform_datasize
        train_confidence_record = torch.zeros([num_data, self.num_classes])
        context_size = trainloader.dataset.context_size
        gammas = 0.5*torch.ones(num_data, context_size, self.pretrain)  # loss weight
        z_record = torch.zeros([num_data, context_size, self.pretrain]) # record correctness
        predict_record = torch.zeros(num_data)
        score_record = torch.zeros(num_data)

        for epoch in range(self.config.num_epoch):

            train_loss  = 0
            train_correct = 0
            train_total = 0
            
            gammas = torch.clip(gammas, 0, 1)
                
            # Step I: train classifier 
            self.cls_net.train()
            self.sl_net.eval()
            for _, (ind, inputs, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='CLS Training: ')):

                inputs, labels = inputs.to(self.device), labels.squeeze(-1).to(self.device)
                outputs = self.cls_net(inputs)
                gammas[ind, :, epoch%self.pretrain] = torch.sigmoid(self.sl_net(inputs)).squeeze(-1).detach().cpu()

                loss = self.loss(outputs.permute(0, 2, 1), labels.long())
                if epoch > self.config.pretrain_isa:
                    if self.use_smooth:
                        with torch.no_grad(): # 
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

            self.scheduler.step()

            # Step II: Train selector 
            self.cls_net.eval()
            self.sl_net.train()
            for _, (ind, inputs, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='SL Training: ')):
                
                inputs, labels = inputs.to(self.device), labels.squeeze().to(self.device)
                outputs = self.cls_net(inputs)
                correctness = outputs.argmax(2).eq(labels).float().detach().cpu()
                z_record[ind, :, epoch%self.pretrain] = correctness

                if epoch >= self.config.pretrain_isa and not epoch%self.config.update_interval:
                    if self.use_smooth:
                        z = z_record[ind].mean(2).to(self.device)
                    else:
                        z = z_record[ind][..., -1].to(self.device)
                        
                    if self.sel_loss == 1:  # Option 1: Original Loss
                        select_prob = torch.sigmoid(self.sl_net(inputs, outputs).squeeze())    
                        loss = -(z*torch.clamp(torch.log(select_prob+EPS), min=-100)+self.beta*(1-z)*torch.clamp(torch.log(1-select_prob+EPS), min=-100)).mean()
                    elif self.sel_loss == 2:   
                        select_logits = self.sl_net(inputs, z).squeeze()
                        select_prob = torch.sigmoid(select_logits)
                        loss = self.loss_sel(select_logits, z)
                    else:
                        raise NotImplementedError

                    self.sl_net.zero_grad()
                    loss.backward()
                    self.sl_optimizer.step()

                    predict_record[ind] = outputs.argmax(2).float().detach().cpu()[:, -1]
                    score_record[ind] = select_prob.detach().cpu()[:, -1]
                    
            self.sl_scheduler.step()

            if (not epoch%self.config.monitor_window) and (epoch>self.config.pretrain_isa): 
                self.selection_quantile = np.quantile(score_record, self.alpha_hat)
                if testloader:
                    eval_result_dict = self.eval(testloader)
                _ = self.performance_summary(trainloader.dataset, predict_record, score_record, self.selection_quantile, tag='_train', update_best=False)

                if self.config.verbose:
                    print(f"[{self.config.num_epoch:3d}|{epoch:3d}] \t Train Loss: {loss.item():.3f} \t Train ACC: {train_correct/train_total:.3f} \t Sel. Quantile: {self.selection_quantile:.3f}")
                    if testloader:
                        self.print_summary(eval_result_dict)

            if not (epoch+1)%self.config.checkpoint_window: 
                checkpoint = (self.cls_net.state_dict(), self.sl_net.state_dict())
                checkpoint_file = f'{self.dataset}_{self.method}_{self.seed}_{self.timestamp}.pt'
                torch.save(checkpoint, os.path.join(self.config.checkpoint_folder, checkpoint_file))

    @torch.no_grad()
    def eval(self, evalloader: torch.utils.data.DataLoader) -> Dict:
        
        num_data = len(evalloader.dataset)
        eval_conf_record = torch.zeros(num_data, self.num_classes)
        eval_selconf_record = torch.zeros(num_data)
        eval_loss_record = 0
        total = 0

        self.cls_net.eval()
        self.sl_net.eval()
        for _, (ind, inputs, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            inputs, labels = inputs.to(self.device), labels.squeeze(-1).to(self.device)
            outputs = self.cls_net(inputs)
            select_prob = torch.sigmoid(self.sl_net(inputs, outputs))
            eval_loss_record += self.loss(outputs.permute(0, 2, 1), labels.long()).mean().item()
            eval_conf_record[ind] = torch.softmax(outputs, 2)[:, -1].detach().cpu().squeeze()
            eval_selconf_record[ind] = select_prob[:, -1].detach().cpu().squeeze()
            total += 1

        score, prediction = eval_selconf_record.numpy(), eval_conf_record.max(1)[1]
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, selection_quantile=self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
        
        return eval_result_dict