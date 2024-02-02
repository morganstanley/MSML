import os
from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from selector.trainer import BaseTrainer

EPS: float = 1e-6

class SelfAdaptiveTrainingSCE():
    def __init__(self, labels, num_classes, momentum=0.9, es=40, alpha=1, beta=0.3):
        # initialize soft labels to onthot vectors
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum
        self.es = es
        
    def __call__(self, logits, targets, index):
        # obtain prob, then update running avg
        prob = F.softmax(logits, dim=1)
        soft_labels_i = self.momentum * self.soft_labels[index].to(logits.device) + (1 - self.momentum) * prob.detach()
        t_y = torch.gather(soft_labels_i, -1, targets.unsqueeze(1)).squeeze()
        p_y, p_c = torch.gather(prob, -1, targets.unsqueeze(1)).squeeze(), prob[:, -1]
        loss = -(t_y*torch.log(p_y + EPS) + (1-t_y)*torch.log(p_c + EPS)).mean()
        self.soft_labels[index] = soft_labels_i.detach().cpu()
        return loss


class SelfAdaptiveTrainingSeqSCE():
    def __init__(self, labels, context_size, num_classes, momentum=0.9, es=40, alpha=1, beta=0.3):
        # initialize soft labels to onthot vectors
        soft_labels = torch.nn.functional.one_hot(torch.from_numpy(labels).squeeze().long())
        soft_labels = torch.cat([soft_labels, torch.zeros(len(soft_labels))[:, None]], -1)
        self.soft_labels = torch.cat([torch.roll(soft_labels, -i, 0).unsqueeze(1) for i in range(context_size)], 1) # NxD => NxTxD 

        self.momentum = momentum
        self.es = es
        
    def __call__(self, logits, targets, index):
        # obtain prob, then update running avg
        prob = F.softmax(logits, dim=2)
        soft_labels_i = self.momentum*self.soft_labels[index].to(logits.device) + (1-self.momentum)*prob.detach()
        t_y = torch.gather(soft_labels_i, -1, targets.unsqueeze(2).long()).squeeze()
        p_y, p_c = torch.gather(prob,-1,targets.unsqueeze(2).long()).squeeze(), prob[:,:,-1]
        loss = -(t_y*torch.log(p_y+EPS) + (1-t_y)*torch.log(p_c+EPS)).mean()
        self.soft_labels[index] = soft_labels_i.detach().cpu()
        return loss

class Adaptive(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        self.es = self.config.pretrain_adaptive
        self.momentum = self.config.momentum_adaptive

    def train(self, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader = None):

        train_loss_record = []
        train_risk_record = []
        num_data = len(trainloader.dataset)
        train_confidence_record = torch.zeros([num_data, self.num_classes])
        train_selscore_record = torch.zeros(num_data)

        self.loss  = SelfAdaptiveTrainingSCE(
            labels = trainloader.dataset.targets, 
            num_classes = self.num_classes+1, 
            momentum = self.momentum, 
            es = self.es
        )
        self.ce_loss = torch.nn.CrossEntropyLoss()

        for epoch in range(self.config.num_epoch):

            train_loss  = 0
            train_correct = 0
            train_total   = 0

            self.model.train()
            for _, (ind, images, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='Training: ')):

                images, labels = images.to(self.device) ,labels.to(self.device)
                outputs = self.model(images)
                if epoch >= self.es:
                    loss = self.loss(outputs, labels, ind)
                else:
                    loss = self.ce_loss(outputs[:, :-1], labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += outputs[:, :-1].argmax(1).eq(labels).sum().item()
                train_total   += len(outputs)

                train_confidence_record[ind] = torch.softmax(outputs[:, :-1], 1).detach().cpu()
                train_selscore_record[ind] = -outputs[:, -1].detach().cpu().squeeze()

            train_loss_record.append(loss.item())
            train_risk_record.append(1-train_correct/train_total)

            self.scheduler.step()

            if (not epoch%self.config.monitor_window) and (epoch>=self.es): 
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
        eval_selscore_record = torch.zeros(num_data)
        eval_loss_record = 0
        total = 0

        self.model.eval()
        for _, (ind, images, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            eval_loss_record += F.cross_entropy(outputs[:, :-1], labels).mean().item()
            eval_conf_record[ind] = torch.softmax(outputs[:, :-1], 1).detach().cpu()
            eval_selscore_record[ind] = outputs[:, -1].detach().cpu().squeeze()
            total += 1

        score, prediction = -eval_selscore_record.numpy(), eval_conf_record.max(1)[1] # abstain the most uncertain data
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
        
        return eval_result_dict


class AdaptiveSeq(Adaptive):

    def train(self, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader = None):

        train_loss_record = []
        train_risk_record = []
        num_data = len(trainloader.dataset)
        train_confidence_record = torch.zeros([num_data, self.num_classes])
        train_selscore_record = torch.zeros(num_data)

        self.loss  = SelfAdaptiveTrainingSeqSCE(
            labels = trainloader.dataset.alltargets, 
            num_classes = self.num_classes+1, 
            context_size = trainloader.dataset.context_size, 
            momentum = self.momentum, 
            es = self.es
        )

        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.ind_map = trainloader.dataset.ind_map

        for epoch in range(self.config.num_epoch):

            train_loss  = 0
            train_correct = 0
            train_total   = 0

            self.model.train()
            for _, (ind, images, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='Training: ')):

                images, labels = images.to(self.device), labels.squeeze(-1).to(self.device)
                outputs = self.model(images)
                if epoch >= self.es:
                    loss = self.loss(outputs, labels, [self.ind_map[i] for i in ind])
                else:
                    loss = self.ce_loss(outputs[:,:,:-1].permute(0,2,1), labels.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += outputs[:,:,:-1].argmax(2).eq(labels).sum().item()
                train_total += labels.numel()

                train_confidence_record[ind] = torch.softmax(outputs[:,-1,:-1], 1).detach().cpu()
                train_selscore_record[ind] = -outputs[:,-1,-1].detach().cpu().squeeze()

            train_loss_record.append(loss.item())
            train_risk_record.append(1-train_correct/train_total)

            self.scheduler.step()

            if (not epoch%self.config.monitor_window) and (epoch>=self.es): 
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
        eval_selscore_record = torch.zeros(num_data)
        eval_loss_record = 0
        total = 0

        self.model.eval()
        for _, (ind, images, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            images, labels = images.to(self.device), labels.squeeze(-1).to(self.device)
            outputs = self.model(images)
            eval_loss_record += F.cross_entropy(outputs[:,:,:-1].permute(0,2,1), labels.long()).mean().item()
            eval_conf_record[ind] = torch.softmax(outputs[:,-1,:-1], 1).detach().cpu()
            eval_selscore_record[ind] = outputs[:,-1,-1].detach().cpu().squeeze()
            total += 1

        score, prediction = -eval_selscore_record.numpy(), eval_conf_record.max(1)[1] # score to select a data (framework compatibility reason)
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
        
        return eval_result_dict
