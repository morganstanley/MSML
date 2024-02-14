import os
from typing import Dict

import torch
import numpy as np
from tqdm import tqdm

from selector.trainer import BaseTrainer

class Confidence(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)

        self.loss = torch.nn.CrossEntropyLoss()

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

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += outputs.argmax(-1).eq(labels).sum().item()
                train_total   += len(outputs)

                probs = torch.softmax(outputs, -1)
                train_confidence_record[ind] = probs.detach().cpu()
                train_selscore_record[ind] = probs.max(-1)[0].detach().cpu()

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
        eval_loss_record = 0
        total = 0

        self.model.eval()
        for _, (ind, inputs, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs   = self.model(inputs)
            eval_loss_record += self.loss(outputs, labels).item()
            eval_conf_record[ind] = torch.softmax(outputs, 1).detach().cpu()
            total += 1

        score, prediction = eval_conf_record.max(1)
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
        
        return eval_result_dict


class ConfidenceSeq(Confidence):


    def train(self, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader = None):

        train_loss_record = []
        train_risk_record = []
        num_data = len(trainloader.dataset)
        train_confidence_record = torch.zeros([num_data, self.num_classes])
        train_selscore_record = torch.zeros(num_data)
        score_record = torch.zeros(num_data)
        
        for epoch in range(self.config.num_epoch):

            train_loss  = 0
            train_correct = 0
            train_total   = 0

            self.model.train()
            for _, (ind, inputs, labels) in enumerate(tqdm(trainloader, ncols=100, miniters=50, desc='Training: ')):

                inputs, labels = inputs.to(self.device), labels.squeeze(-1).to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs.permute(0, 2, 1), labels.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_correct += outputs.argmax(2).eq(labels).sum().item()
                train_total   += labels.numel()

                probs = torch.softmax(outputs[:,-1], 1)
                train_confidence_record[ind] = probs.detach().cpu()
                train_selscore_record[ind]   = probs.max(1)[0].detach().cpu()
                predict_record[ind] = outputs.argmax(2).float().detach().cpu()[:, -1]
                
            train_loss_record.append(loss.item())
            train_risk_record.append(1-train_correct/train_total)

            self.scheduler.step()

            if not epoch%self.config.monitor_window: 
                self.selection_quantile = np.quantile(train_selscore_record, self.alpha_hat)
                if testloader:
                    eval_result_dict = self.eval(testloader)
                _ = self.performance_summary(trainloader.dataset, predict_record, train_selscore_record, self.selection_quantile, tag='_train', update_best=False)

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
        for _, (ind, inputs, labels) in enumerate(tqdm(evalloader, ncols=100, miniters=50, desc='Testing: ')):
            inputs, labels = inputs.to(self.device), labels.squeeze(-1).to(self.device)
            outputs = self.model(inputs)
            eval_loss_record += self.loss(outputs.permute(0, 2, 1), labels.long()).item()
            eval_conf_record[ind] = torch.softmax(outputs, 2)[:, -1].detach().cpu()
            total += 1

        score, prediction = eval_conf_record.max(1)
        eval_result_dict = self.performance_summary(evalloader.dataset, prediction, score, self.selection_quantile)
        eval_result_dict['eval_loss'] = eval_loss_record/total
       
        return eval_result_dict