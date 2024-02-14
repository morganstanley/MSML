import os
from typing import Dict

import torch
import numpy as np
from tqdm import tqdm

from selector.trainer import BaseTrainer

class SelectiveNet(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        self.alpha = self.config.alpha
        self.lamda = self.config.lamda 
        self.loss  = torch.nn.NLLLoss(reduction='none')

    def train(self, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader = None):

        train_loss_record = []
        train_risk_record = []
        num_data = len(trainloader.dataset)
        train_confidence_record = torch.zeros([num_data, self.num_classes])
        train_selscore_record = torch.zeros(num_data)
        
        for epoch in range(self.config.num_epoch):

            train_loss  = 0
            train_correct = 0
            train_total   = 0

            self.model.train()
            for _, (ind, inputs, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='Training: ')):

                inputs, labels = inputs.to(self.device) ,labels.to(self.device)
                cls_out, sel_out, aux_out = self.model(inputs)

                log_aux_out = torch.log_softmax(aux_out, 1)
                # auxiliary loss
                aux_loss = torch.mean(self.loss(log_aux_out, labels))
                # compute selective loss
                log_cls_out = torch.log_softmax(cls_out, 1)
                sel_loss = torch.mean(self.loss(log_cls_out, labels)*sel_out)
                coverage_loss = torch.clamp(1-self.alpha_hat-torch.mean(sel_out), min=0)**2
                sel_loss = sel_loss + self.lamda*coverage_loss
                # total loss (alpha is the hyper-param and alphahat is the coverage estimate)
                loss = self.alpha*aux_loss + (1-self.alpha)*sel_loss
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += aux_out.argmax(1).eq(labels).sum().item()
                train_total   += len(aux_out)

                train_confidence_record[ind] = torch.softmax(aux_out, 1).detach().cpu()
                train_selscore_record[ind]   = sel_out.detach().cpu().squeeze()

            train_loss_record.append(loss.item())
            train_risk_record.append(1-train_correct/train_total)

            self.scheduler.step()

            if (not epoch%self.config.monitor_window): 
                if testloader:
                    self.selection_quantile = np.quantile(train_selscore_record.numpy(), self.alpha_hat)
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
        eval_selconf_record = torch.zeros(num_data)
        eval_loss_record = 0
        total = 0

        self.model.eval()
        for _, (ind, inputs, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            cls_out, sel_out, _ = self.model(inputs)
            eval_loss_record += self.loss(cls_out, labels).mean().item()
            eval_conf_record[ind] = torch.softmax(cls_out, 1).detach().cpu()
            eval_selconf_record[ind] = sel_out.detach().cpu().squeeze()
            total += 1

        score, prediction = eval_selconf_record.numpy(), eval_conf_record.max(1)[1]
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
       
        return eval_result_dict


class SelectiveNetSeq(SelectiveNet):

    def train(self, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader = None):

        train_loss_record = []
        train_risk_record = []
        num_data = len(trainloader.dataset)
        train_confidence_record = torch.zeros([num_data, self.num_classes])
        train_selscore_record = torch.zeros(num_data)

        for epoch in range(self.config.num_epoch):

            train_loss  = 0
            train_correct = 0
            train_total   = 0

            self.model.train()
            for _, (ind, inputs, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='Training: ')):

                inputs, labels = inputs.to(self.device) ,labels.squeeze(-1).to(self.device)
                cls_out, sel_out, aux_out = self.model(inputs)
                sel_out = sel_out.squeeze(-1)

                log_aux_out = torch.log_softmax(aux_out, 2)
                # auxiliary loss
                aux_loss = torch.mean(self.loss(log_aux_out.permute(0, 2, 1), labels.long()))
                # compute selective loss
                log_cls_out = torch.log_softmax(cls_out, -1)
                sel_loss = torch.mean(self.loss(log_cls_out.permute(0, 2, 1), labels.long())*sel_out)
                coverage_loss = torch.clamp(1-self.alpha_hat-torch.mean(sel_out), min=0)**2
                sel_loss = sel_loss + self.lamda*coverage_loss
                # total loss (alpha is the hyper-param and alphahat is the coverage estimate)
                loss = self.alpha*aux_loss + (1-self.alpha)*sel_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += aux_out.argmax(2).eq(labels).sum().item()
                train_total   += labels.numel()

                train_confidence_record[ind] = torch.softmax(aux_out[:, -1], 1).detach().cpu().squeeze()
                train_selscore_record[ind]   = sel_out[:, -1].detach().cpu().squeeze()

            train_loss_record.append(loss.item())
            train_risk_record.append(1-train_correct/train_total)

            self.scheduler.step()

            if (not epoch%self.config.monitor_window): 
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
        eval_selconf_record = torch.zeros(num_data)
        eval_loss_record = 0
        total = 0

        self.model.eval()
        for _, (ind, inputs, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            inputs, labels = inputs.to(self.device), labels.squeeze(-1).to(self.device)
            cls_out, sel_out, _ = self.model(inputs)
            eval_loss_record += self.loss(cls_out.permute(0, 2, 1), labels.long()).mean().item()
            eval_conf_record[ind] = torch.softmax(cls_out[:, -1], 1).detach().cpu()
            eval_selconf_record[ind] = sel_out[:, -1].detach().cpu().squeeze()
            total += 1

        score, prediction = eval_selconf_record.numpy(), eval_conf_record.max(1)[1]
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
        
        return eval_result_dict