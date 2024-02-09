import os
from typing import Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from selector.trainer import BaseTrainer

class DeepGambler(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        self.O = self.config.O
        self.pretrain = self.config.pretrain_gambler
        self.loss  = torch.nn.CrossEntropyLoss()

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
            for _, (ind, images, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='Training: ')):

                images, labels = images.to(self.device), labels.to(self.device)
                # deep gamble loss; or use cross entropy
                outputs = self.model(images)
                if epoch >= self.pretrain:
                    probs = torch.softmax(outputs, dim=1)
                    confidence, reservation = probs[:,:-1], probs[:,-1]
                    gain = torch.gather(confidence, dim=1, index=labels.unsqueeze(1)).squeeze()
                    doubling_rate = (gain.add(reservation.div(self.O))).log()
                    loss = -doubling_rate.mean()
                else:
                    loss = self.loss(outputs[:, :-1], labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += outputs[:, :-1].argmax(1).eq(labels).sum().item()
                train_total   += len(outputs)

                train_confidence_record[ind] = torch.softmax(outputs[:, :-1], 1).detach().cpu()
                train_selscore_record[ind]   = -outputs[:, -1].detach().cpu().squeeze()

            train_loss_record.append(loss.item())
            train_risk_record.append(1-train_correct/train_total)

            self.scheduler.step()

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
        eval_selconf_record = torch.zeros(num_data)
        eval_loss_record = 0
        total = 0

        self.model.eval()
        for _, (ind, images, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            eval_loss_record += self.loss(outputs[:, :-1], labels).mean().item()
            eval_conf_record[ind] = torch.softmax(outputs[:, :-1], 1).detach().cpu()
            eval_selconf_record[ind] = outputs[:, -1].detach().cpu().squeeze()
            total += 1

        score, prediction = -eval_selconf_record.numpy(), eval_conf_record.max(1)[1] # abstain the most certain data
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
        
        return eval_result_dict



class DeepGamblerSeq(DeepGambler):

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
            for _, (ind, images, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='Training: ')):

                images, labels = images.to(self.device), labels.squeeze(-1).to(self.device)
                # deep gamble loss; or use cross entropy
                outputs = self.model(images)
                if epoch >= self.pretrain:
                    probs = torch.softmax(outputs, dim=2)
                    confidence, reservation = probs[:,:,:-1], probs[:,:,-1]
                    gain = torch.gather(confidence, dim=2, index=labels.unsqueeze(2).long()).squeeze()
                    doubling_rate = (gain.add(reservation.div(self.O))).log()
                    loss = -doubling_rate.mean()
                else:
                    loss = self.loss(outputs[:,:,:-1].permute(0,2,1), labels.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += outputs[:,:,:-1].argmax(2).eq(labels).sum().item()
                train_total   += labels.numel()

                train_confidence_record[ind] = torch.softmax(outputs[:,-1,:-1], 1).detach().cpu()
                train_selscore_record[ind]   = -outputs[:,-1,-1].detach().cpu().squeeze()

            train_loss_record.append(loss.item())
            train_risk_record.append(1-train_correct/train_total)

            self.scheduler.step()

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
        eval_selconf_record = torch.zeros(num_data)
        eval_loss_record = 0
        total = 0

        self.model.eval()
        for _, (ind, images, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            images, labels = images.to(self.device), labels.squeeze(-1).to(self.device)
            outputs = self.model(images)
            eval_loss_record += self.loss(outputs[:,:,:-1].permute(0,2,1), labels.long()).mean().item()
            eval_conf_record[ind] = torch.softmax(outputs[:,-1,:-1], 1).detach().cpu()
            eval_selconf_record[ind] = outputs[:,-1,-1].detach().cpu().squeeze()
            total += 1

        score, prediction = -eval_selconf_record.numpy(), eval_conf_record.max(1)[1] # abstain the most certain data
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
        
        return eval_result_dict
