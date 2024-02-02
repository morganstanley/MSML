import os
from typing import Dict

import torch
from torch import optim
import numpy as np
from tqdm import tqdm

from selector.trainer import BaseTrainer

EPS = 1e-6

class OneSide(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        self.loss  = torch.nn.CrossEntropyLoss()
        self.lambdas = torch.ones(self.num_classes).view(1, -1).to(self.device)
        self.psi     = torch.ones(self.num_classes).view(1, -1).to(self.device)
        self.mu = self.config.mu_oneside
        self.pretrain = self.config.pretrain_oneside

        backbone_params = [x for x in self.model.parameters() if not (x.data_ptr in [y.data_ptr for y in self.model.fc3.parameters()])]
        osplayer_params = [x for x in self.model.fc3.parameters()]

        self.backbone_optimizer = optim.Adam(backbone_params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.osplayer_optimizer = optim.Adam(osplayer_params, lr=self.config.lr, weight_decay=self.config.weight_decay)

        self.backbone_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.backbone_optimizer, milestones=self.mile_stone, gamma=0.5)
        self.osplayer_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.osplayer_optimizer, milestones=self.mile_stone, gamma=0.5)

        self.backbone_state_dict_init = self.model.fc3.state_dict()
        self.osplayer_state_dict_init = {k: v for k, v in self.model.state_dict().items() if k not in self.backbone_state_dict_init}
        
    def train(self, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader = None):

        train_loss_record = []
        train_risk_record = []
        num_data = len(trainloader.dataset)
        train_confidence_record = torch.zeros([num_data, self.num_classes])
        train_selscore_record = torch.zeros(num_data)
        backbone_update_counter = 0

        for epoch in range(self.config.num_epoch):

            train_loss  = 0
            train_correct = 0
            train_total   = 0

            self.model.train()
            for _, (ind, images, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='Training: ')):

                images, labels = images.to(self.device), labels.to(self.device)
                self.lambdas = self.lambdas.requires_grad_()
                self.psi = self.psi.requires_grad_()

                outputs = self.model(images)

                if  epoch < self.pretrain:
                    loss = self.loss(outputs, labels)
                    self.backbone_optimizer.zero_grad()
                    self.osplayer_optimizer.zero_grad()
                    loss.backward()
                    self.backbone_optimizer.step()
                    self.osplayer_optimizer.step()
                else:
                    conf = torch.softmax(outputs, 1)
                    z = torch.gather(conf, 1, labels.view(len(conf), 1))
                    _, pred = conf.max(1)
                    
                    lambdas_k = torch.gather(self.lambdas, 1, labels.view(1, -1))
                    psi_k = torch.gather(self.psi, 1, labels.view(1, -1)) 
                    loss = -torch.log(z+EPS) + lambdas_k*(-torch.log(1-z+EPS)-psi_k) + self.mu*psi_k
                    loss = loss.sum()/len(images)
                    
                    # Min problem 
                    self.backbone_optimizer.zero_grad()
                    self.osplayer_optimizer.zero_grad()
                    loss.backward()
                    if not (backbone_update_counter%20):
                        self.backbone_optimizer.step()
                    self.osplayer_optimizer.step()
                    nabla_psi = self.psi.grad.detach()
                    self.psi = self.psi.detach()
                    self.psi -= self.osplayer_optimizer.param_groups[0]['lr']*nabla_psi
                    
                    # Max Problem
                    nabla_lambdas = self.lambdas.grad.detach()
                    self.lambdas = self.lambdas.detach()
                    self.lambdas += (1e-5/(10**((epoch-self.pretrain)//50)))*nabla_lambdas

                train_loss += loss.item()
                train_correct += outputs.argmax(1).eq(labels).sum().item()
                train_total   += len(outputs)

                probs = torch.softmax(outputs, 1)
                train_confidence_record[ind] = probs.detach().cpu()
                train_selscore_record[ind] = probs.max(1)[0].detach().cpu()

            backbone_update_counter += 1

            train_loss_record.append(loss.item())
            train_risk_record.append(1-train_correct/train_total)

            self.backbone_lr_scheduler.step()
            self.osplayer_lr_scheduler.step()

            if (not epoch%self.config.monitor_window) and (epoch>=self.pretrain): 
                if testloader:
                    self.selection_quantile = np.quantile(train_selscore_record, self.alpha_hat)
                    eval_result_dict = self.eval(testloader)

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
        eval_loss_record = 0
        total = 0

        self.model.eval()
        for _, (ind, images, labels) in enumerate(tqdm(evalloader, ncols=100, desc='Testing: ')):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs   = self.model(images)
            eval_loss_record += self.loss(outputs, labels).item()
            eval_conf_record[ind] = torch.softmax(outputs, 1).detach().cpu()
            total += 1

        score, prediction = eval_conf_record.max(1)
        self.selection_quantile = np.quantile(score, self.alpha_hat)
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
        
        return eval_result_dict



class OneSideSeq(OneSide):

    def train(self, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader = None):

        train_loss_record = []
        train_risk_record = []
        num_data = len(trainloader.dataset)
        train_confidence_record = torch.zeros([num_data, self.num_classes])
        train_selscore_record = torch.zeros(num_data)
        backbone_update_counter = 0

        for epoch in range(self.config.num_epoch):

            train_loss  = 0
            train_correct = 0
            train_total   = 0

            self.model.train()
            for _, (ind, images, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='Training: ')):

                images, labels = images.to(self.device), labels.squeeze(-1).to(self.device)
                self.lambdas = self.lambdas.requires_grad_()
                self.psi = self.psi.requires_grad_()

                outputs = self.model(images)

                if  epoch < self.pretrain:
                    loss = self.loss(outputs.permute(0,2,1), labels.long())
                    self.backbone_optimizer.zero_grad()
                    self.osplayer_optimizer.zero_grad()
                    loss.backward()
                    self.backbone_optimizer.step()
                    self.osplayer_optimizer.step()
                else:
                    conf = torch.softmax(outputs, 2)
                    z = torch.gather(conf, -1, labels.unsqueeze(-1).long()).squeeze()
                    _, pred = conf.max(2)
                    
                    batch_size, seq_len = labels.shape
                    lambdas_k = torch.gather(self.lambdas, 1, labels.view(1, -1).long()).view(batch_size, seq_len)
                    psi_k = torch.gather(self.psi, 1, labels.view(1, -1).long()).view(batch_size, seq_len) 
                    loss = -torch.log(z+EPS) + lambdas_k*(-torch.log(1-z+EPS)-psi_k) + self.mu*psi_k
                    loss = loss.sum()/labels.numel()
                    
                    # Min problem 
                    self.backbone_optimizer.zero_grad()
                    self.osplayer_optimizer.zero_grad()
                    loss.backward()
                    if not (backbone_update_counter%20):
                        self.backbone_optimizer.step()
                    self.osplayer_optimizer.step()
                    nabla_psi = self.psi.grad.detach()
                    self.psi  = self.psi.detach()
                    self.psi -= self.osplayer_optimizer.param_groups[0]['lr']*nabla_psi
                    
                    # Max Problem
                    nabla_lambdas = self.lambdas.grad.detach()
                    self.lambdas  = self.lambdas.detach()
                    self.lambdas += (1e-5/(10**((epoch-self.pretrain)//50)))*nabla_lambdas

                train_loss += loss.item()
                train_correct += outputs.argmax(2).eq(labels).sum().item()
                train_total   += labels.numel()

                probs = torch.softmax(outputs[:,-1], 1)
                train_confidence_record[ind] = probs.detach().cpu()
                train_selscore_record[ind] = probs.max(1)[0].detach().cpu()

            backbone_update_counter += 1

            train_loss_record.append(loss.item())
            train_risk_record.append(1-train_correct/train_total)

            self.backbone_lr_scheduler.step()
            self.osplayer_lr_scheduler.step()

            if (not epoch%self.config.monitor_window) and (epoch>=self.pretrain): 
                if testloader:
                    self.selection_quantile = np.quantile(train_selscore_record, self.alpha_hat)
                    eval_result_dict = self.eval(testloader)

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
        eval_loss_record = 0
        total = 0

        self.model.eval()
        for _, (ind, images, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            images, labels = images.to(self.device), labels.squeeze(-1).to(self.device)
            outputs   = self.model(images)
            eval_loss_record += self.loss(outputs.permute(0,2,1), labels.long()).item()
            eval_conf_record[ind] = torch.softmax(outputs[:,-1], 1).detach().cpu()
            total += 1

        score, prediction = eval_conf_record.max(1)
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
        
        return eval_result_dict