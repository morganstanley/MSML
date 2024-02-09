from typing import Dict, Union, Tuple, List
from collections import defaultdict

import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import average_precision_score
import copy
from datetime import datetime
from tqdm import tqdm

from network.utils import build_network

class BaseTrainer():

    _num_classes = {'mnist': 10, 'svhn': 5, 'volatility': 2, 'bus': 3, 'lc': 3, 'ou': 2, 'lobster':2}

    def __init__(self, config):

        figure_folder = config.figure_folder
        checkpoint_folder = config.checkpoint_folder
        result_folder = config.result_folder
        self.dataset = config.dataset
        self.method  = config.method
        self.seed = config.seed
        self.need_estimate_alpha = False

        self.config = config
        self.num_classes = self._num_classes[self.config.dataset]
        self.best_f1 = 0
        self.best_slrisk = 1
        self.best_ap = 0
        self.device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.model = build_network(self.config)
        if isinstance(self.model, Tuple):
            self.model = tuple(model.to(self.device) for model in self.model)
        else:
            self.model = self.model.to(self.device)
        
        if isinstance(self.model, Tuple):
            self.optimizer = optim.Adam(
                self.model[0].parameters(), 
                lr = self.config.lr, 
                weight_decay = self.config.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config.lr, 
                weight_decay=self.config.weight_decay)
        
        if self.dataset in ['mnist', 'svhn']: 
            self.mile_stone = [x for x in self.config.mile_stone]
        else:
            self.mile_stone = [15, 35]
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=self.mile_stone, 
            gamma=0.5)

        if not isinstance(self.model, Tuple):
            self.initi_model = self.model.state_dict()
        
        self.alpha_hat = 1-self.config.coverage_target
        self.selection_quantile = -float('inf')

        self.logger = defaultdict(list)

        self.timestamp = datetime.today().strftime("%y%m%d%H%M%S")

    def train(self, trainloader: torch.utils.data.DataLoader):
        raise NotImplementedError

    def eval(self, evalloader: torch.utils.data.DataLoader):
        raise NotImplementedError

    @staticmethod
    def estimate_alpha(confidence: torch.Tensor, labels: torch.Tensor) -> float:
        acc = confidence.argmax(1).eq(labels).sum().item()/len(confidence)
        num_class = confidence.shape[1]
        alpha_hat = max(0, min((1-acc)/(1-1/(num_class)), 1))
        return alpha_hat

    def performance_summary(self, 
        dataset: torch.utils.data.Dataset, 
        prediction: torch.Tensor, 
        score: torch.Tensor, 
        selection_quantile: float, 
        tag: str = '', 
        update_best: bool = True
        ):

        selected_ind = np.where(score >= selection_quantile)[0]
       
        targets = torch.from_numpy(dataset.targets).squeeze()
        total_num = len(dataset)
        tp_select_num = np.sum(selected_ind >= dataset.uninform_datasize)
        fp_select_num = len(selected_ind) - tp_select_num

        precision = tp_select_num/(len(selected_ind)+1)
        coverage = len(selected_ind)/len(dataset)
        recall = tp_select_num/dataset.inform_datasize
        f1 = 2*precision*recall/(precision+recall+1e-6)
        sl_acc = (tp_select_num + dataset.uninform_datasize - fp_select_num)/total_num

        correctness = prediction.eq(targets).detach().cpu()
        acc = correctness.sum()/total_num
        info_acc = correctness[dataset.uninform_datasize:].sum()/dataset.uninform_datasize
        slrisk = 1-correctness[selected_ind].sum().item()/(len(selected_ind)+1)

        gt_z = np.concatenate([np.zeros(dataset.uninform_datasize), np.ones(dataset.inform_datasize)])
        ap = average_precision_score(gt_z, score)
        
        result_dict = {
            'acc': acc.item(), 
            'info_acc': info_acc.item(), 
            'slrisk': slrisk, 
            'sl_acc': sl_acc, 
            'precision': precision, 
            'recall': recall, 
            'f1': f1, 
            'ap': ap, 
            'coverage': coverage, 
            'alpha_hat': self.alpha_hat
        }
        for k in result_dict:
            self.logger[k+tag].append(result_dict[k])
        result_dict['targets'] = targets.tolist()
        result_dict['prediction'] = prediction.tolist()
        result_dict['score'] = score.tolist()

        if update_best: 
            if f1 >= self.best_f1:
                result_dict['bestf1_prediction'] = prediction.tolist()
                result_dict['bestf1_score'] = score.tolist()
                self.best_f1 = f1
                self.bestf1_score = score.tolist()
                self.bestf1_prediction = prediction.tolist()
            else:
                result_dict['bestf1_score'] = self.bestf1_score
                result_dict['bestf1_prediction'] = self.bestf1_prediction

            if slrisk <= self.best_slrisk:
                result_dict['bestsr_prediction'] = prediction.tolist()
                result_dict['bestsr_score'] = score.tolist()
                self.best_slrisk = slrisk
                self.bestsr_score = score.tolist()
                self.bestsr_prediction = prediction.tolist()
            else:
                result_dict['bestsr_score'] = self.bestsr_score
                result_dict['bestsr_prediction'] = self.bestsr_prediction

            if ap >= self.best_ap:
                result_dict['bestap_prediction'] = prediction.tolist()
                result_dict['bestap_score'] = score.tolist()
                self.best_ap = ap
                self.bestap_score = score.tolist()
                self.bestap_prediction = prediction.tolist()
            else:
                result_dict['bestap_score'] = self.bestap_score
                result_dict['bestap_prediction'] = self.bestap_prediction

        return result_dict

    @staticmethod
    def print_summary(result_dict: Dict):
        sr = result_dict['slrisk']
        slacc = result_dict['sl_acc']
        precision = result_dict['precision']
        recall = result_dict['recall']
        f1 = result_dict['f1']
        ap = result_dict['ap']
        print(f'\t\t SR: {sr:.3f} \t\t SRACC: {slacc:.3f} \t Precision: {precision:.3f} \t Recal: {recall:.3f} \t F1: {f1:.3f} \t AP: {ap:.3f}')

    @staticmethod
    def plot_result(meter: Dict):
        pass
