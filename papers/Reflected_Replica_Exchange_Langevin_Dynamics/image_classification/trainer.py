"""
Created on May 12, 2024
@author: Haoyang Zheng & Wei Deng
Code for Constrained Exploration via Reflected Replica Exchange Stochastic Gradient Langevin Dynamics. ICML 2024
"""

# !/usr/bin/python

import sys
import os
import time
import torch
import pickle
import numpy as np
import torch.nn as nn
from sgmcmc import Sampler
from tools import BayesEval
from datetime import datetime
from torch.autograd import Variable

CUDA_EXISTS = torch.cuda.is_available()


def trainer_resgld(nets, train_loader, test_loader, extra_loader, pars):
    print("resgld")

    # # First train
    now = datetime.now()
    time_now = now.strftime("%Y_%m_%d_%H_%M_%S")

    criterion = nn.CrossEntropyLoss()
    init_T, init_lr = pars.T, pars.lr
    samplers, BMAS, myVars, cooling_time, lr_set = {}, [], [], [], []
    for idx in range(pars.chains - 1, -1, -1):
        print('Chain {} Initial learning rate {:.2e} temperature {:.2e}'.format(idx, init_lr, init_T))
        if pars.regularization is None:
            sampler = Sampler(
                nets[idx], criterion,
                lr=init_lr, wdecay=pars.wdecay, T=init_T, total=pars.total, domain=pars.if_domain, bound=pars.bound)
        else:
            sampler = Sampler(
                nets[idx], criterion, lr=init_lr,
                wdecay=pars.wdecay, T=init_T, total=pars.total, domain=pars.if_domain, bound=pars.bound,
                regularization=pars.regularization)
        lr_set.insert(0, init_lr)
        init_T /= pars.Tgap
        init_lr /= pars.LRgap
        samplers[idx] = sampler
        BMAS.append(BayesEval())
        myVars.append(sys.float_info.max)

    counter, warm_up, adjusted_corrections, increase_correct, eta = 1., 10, np.ones((pars.chains)) * 30000, 20000, 500
    swap_yes, swap_no = np.ones((pars.chains - 1)) * 0.01, np.ones((pars.chains - 1)) * 0.01
    start = time.time()

    """ Initialization for variance reduction """
    last_full_losses, last_VRnets, corr = [0] * pars.chains, [], [-1] * pars.chains
    for idx in range(pars.chains):
        last_VRnets.append(pickle.loads(pickle.dumps(nets[idx])))

    if pars.save_after is not None:
        net_dir = './logs/' + pars.optimizer + '_chain' + str(len(samplers)) + '_' + time_now
        os.mkdir(net_dir)

    anneal_delay = np.zeros([3])
    for epoch in range(pars.sn + 1):
        """ update adaptive variance and variance reduction every [period] epochs """
        if pars.period > 0 and epoch % pars.period == 0 and epoch > warm_up:
            cur_full_losses = [0] * pars.chains
            for idx in range(pars.chains):
                stage_losses, cv_losses = [], []
                nets[idx].eval()
                for i, (images, labels) in enumerate(train_loader):
                    images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
                    labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
                    nets[idx].zero_grad()
                    avg_loss = criterion(nets[idx](images), labels).item()

                    if pars.regularization is not None:
                        l2_loss = torch.mean(torch.tensor([param.pow(2.0).mean() for param in nets[idx].parameters()]))
                        avg_loss = avg_loss + 0.5 * pars.regularization * l2_loss

                    cur_full_losses[idx] += avg_loss * pars.batch
                    stage_losses.append(avg_loss * pars.total)
                    if pars.var_reduce:
                        cv_losses.append(criterion(last_VRnets[idx](images), labels).item() * pars.total)

                if pars.var_reduce:
                    for i in range(len(stage_losses)):
                        stage_losses[i] = stage_losses[i] + corr[idx] * (cv_losses[i] - np.mean(cv_losses))
                std_epoch = np.std(stage_losses, ddof=1)
                myVars[idx] = 0.5 * std_epoch ** 2 if myVars[idx] == sys.float_info.max else (
                        (1 - pars.alpha) * myVars[idx] + pars.alpha * 0.5 * std_epoch ** 2)
                print(
                    'Epoch {} Chain {} loss std {:.2e} variance {:.2e} smooth variance {:.2e} adaptive c {:.2f}'.format(
                        epoch, idx, std_epoch, 0.5 * std_epoch ** 2, myVars[idx], corr[idx]))
                last_VRnets[idx] = pickle.loads(pickle.dumps(nets[idx]))
                last_full_losses[idx] = cur_full_losses[idx]

        for idx in range(pars.chains):
            nets[idx].train()

        if pars.cycle >= 2:
            sub_sn = pars.sn / pars.cycle
            cur_beta = (epoch % sub_sn) * 1.0 / sub_sn
            for idx in range(pars.chains):
                samplers[idx].set_eta(lr_set[idx] / 2 * (np.cos(np.pi * cur_beta) + 1))
                if (epoch % sub_sn) * 1.0 / sub_sn == 0:
                    print('Chain {} Cooling down for optimization'.format(idx))
                    samplers[idx].set_T(1e10)
                elif epoch % sub_sn == int(pars.burn * sub_sn):
                    print('Chain {} Heating up for sampling'.format(idx))
                    samplers[idx].set_T(1e-10)

        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
            labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
            counter += 1.
            loss_chains = []
            for idx in range(pars.chains):
                loss = samplers[idx].step(images, labels)
                """ variance-reduced negative log posterior """
                if pars.var_reduce and epoch > warm_up:
                    control_variate_loss = criterion(last_VRnets[idx](images), labels).item() * pars.total
                    loss = loss + corr[idx] * (control_variate_loss - last_full_losses[idx])
                loss_chains.append(loss)

            """ Swap """
            for idx in range(pars.chains - 1):
                """ exponential average smoothing """
                delta_invT = 1. / samplers[idx].T - 1. / samplers[idx + 1].T
                swap_rate = swap_yes[idx] / (swap_no[idx] + swap_yes[idx])
                adjusted_corrections[idx] = adjusted_corrections[idx] - eta * (0.01 - swap_rate)
                if loss_chains[idx] - loss_chains[idx + 1] - adjusted_corrections[idx] >= 0:
                    if epoch not in cooling_time:
                        temporary = pickle.loads(pickle.dumps(samplers[idx + 1].net))
                        samplers[idx + 1].net.load_state_dict(samplers[idx].net.state_dict())
                        samplers[idx].net.load_state_dict(temporary.state_dict())
                        print('Epoch {} Swap chain {} with chain {} and increased F {:0.2e}'.format(
                            epoch, idx, idx + 1, pars.bias_F))
                        cooling_time = range(epoch, epoch + pars.cool)
                        swap_yes[idx] = swap_yes[idx] + 1
                        adjusted_corrections[idx] = adjusted_corrections[idx] + increase_correct
                    else:
                        print('Epoch {} Cooling period'.format(epoch))
                        swap_no[idx] = swap_no[idx] + 1
                else:
                    swap_no[idx] = swap_no[idx] + 1

        """ Anneaing """
        if epoch < pars.burn * pars.sn:
            pars.bias_F *= pars.Tanneal
        for idx in range(pars.chains):
            if pars.cycle == 1:
                if epoch > 0.3 * pars.sn and pars.LRanneal <= 1.:
                    samplers[idx].eta *= pars.LRanneal
                if epoch < pars.burn * pars.sn:
                    samplers[idx].set_T(pars.Tanneal)

            if samplers[idx].eta <= pars.anneal_lr and anneal_delay[idx] < 100:
                samplers[idx].eta = pars.anneal_lr
                anneal_delay[idx] += 1

            if samplers[idx].eta < 6e-7:
                samplers[idx].eta = 2e-6

            if (pars.cycle == 1 and epoch >= int(pars.burn * pars.sn)) or (pars.cycle > 1 and cur_beta >= pars.burn):
                BMAS[idx].eval(nets[idx], test_loader, extra_loader, criterion, bma=True, iters=epoch)
            elif (pars.cycle == 1 and epoch < int(pars.burn * pars.sn)) or (pars.cycle > 1 and cur_beta < pars.burn):
                BMAS[idx].eval(nets[idx], test_loader, extra_loader, criterion, bma=True, iters=epoch)
            print(
                'Epoch {} Chain {} Acc: {:0.2f} BMA: {:0.2f} Best Acc: {:0.2f} Best BMA: {:0.2f} lr: {:.2E} T: {:.2E}  Loss: {:0.2f} Corrections: {:0.2f}' \
                    .format(epoch, idx, BMAS[idx].cur_acc, BMAS[idx].bma_acc, BMAS[idx].best_cur_acc,
                            BMAS[idx].best_bma_acc, samplers[idx].eta, samplers[idx].T,
                            np.array(loss_chains[idx]).sum(), abs(adjusted_corrections[idx])))
            print(
                'Epoch {} Chain {} NLL: {:.2f} Best NLL: {:.2f} BMA NLL: {:.2f}  Best BMA NLL: {:.2f}  Brier: {:.3e}  Best Brier: {:.3e}'.format(
                    epoch, idx, BMAS[idx].nll, BMAS[idx].best_nll, BMAS[idx].bma_nll, BMAS[idx].best_bma_nll,
                    BMAS[idx].brier_score, BMAS[idx].best_brier))

        print('')
    end = time.time()
    print('Time used {:.2f}s'.format(end - start))


def trainer_csgld(nets, train_loader, test_loader, pars, M=4):
    nets = [nets]
    now = datetime.now()
    time_now = now.strftime("%Y_%m_%d_%H_%M_%S")

    criterion = nn.CrossEntropyLoss()
    init_T, init_lr = pars.T, pars.lr
    samplers, BMAS, myVars, cooling_time, lr_set = {}, [], [], [], []
    for idx in range(pars.chains - 1, -1, -1):
        print('Chain {} Initial learning rate {:.2e} temperature {:.2e}'.format(idx, init_lr, init_T))
        sampler = Sampler(
            nets[idx], criterion, lr=init_lr, wdecay=pars.wdecay, T=init_T,
            total=pars.total, domain=pars.if_domain, bound=pars.bound)
        lr_set.insert(0, init_lr)
        init_T /= pars.Tgap
        init_lr /= pars.LRgap
        sampler.eta = 0.05
        samplers[idx] = sampler
        BMAS.append(BayesEval())
        myVars.append(sys.float_info.max)
    counter, warm_up, adjusted_corrections = 1., 10, 0
    start = time.time()

    """ Initialization for variance reduction """
    last_full_losses, last_VRnets, corr = [0] * pars.chains, [], [-1] * pars.chains
    for idx in range(pars.chains):
        last_VRnets.append(pickle.loads(pickle.dumps(nets[idx])))

    num_batch = len(train_loader)
    T = pars.sn
    lr_0 = pars.lr

    def adjust_learning_rate(epoch, batch_idx):
        rcounter = epoch * num_batch + batch_idx
        cos_inner = np.pi * (rcounter % (T // M))
        cos_inner /= T // M
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5 * cos_out * lr_0
        return lr

    if pars.save_after is not None:
        net_dir = './logs/' + pars.optimizer + '_chain' + str(len(samplers)) + '_' + time_now
        os.mkdir(net_dir)

    for epoch in range(pars.sn + 1):
        """ update adaptive variance and variance reduction every [period] epochs """
        if pars.period > 0 and epoch % pars.period == 0 and epoch > warm_up:
            cur_full_losses = [0] * pars.chains
            for idx in range(pars.chains):
                stage_losses, cv_losses = [], []
                nets[idx].eval()
                for i, (images, labels) in enumerate(train_loader):
                    images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
                    labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
                    nets[idx].zero_grad()
                    avg_loss = criterion(nets[idx](images), labels).item()
                    cur_full_losses[idx] += avg_loss * pars.batch
                    stage_losses.append(avg_loss * pars.total)
                    if pars.var_reduce:
                        cv_losses.append(criterion(last_VRnets[idx](images), labels).item() * pars.total)

        for idx in range(pars.chains):
            nets[idx].train()

        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
            labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
            counter += 1.
            loss_chains = []
            for idx in range(pars.chains):
                """ adjust learning rate """
                samplers[idx].eta = adjust_learning_rate(epoch, i)

                loss = samplers[idx].step(images, labels)

        """ record metrics """
        for idx in range(pars.chains):
            BMAS[idx].eval(nets[idx], test_loader, bma=True)

            print(
                'Epoch {} Chain {} Acc: {:0.2f} BMA: {:0.2f} Best Acc: {:0.2f} Best BMA: {:0.2f} lr: {:.2E} T: {:.2E}  Loss: {:0.2f} Corrections: {:0.2f}' \
                    .format(epoch, idx, BMAS[idx].cur_acc, BMAS[idx].bma_acc, BMAS[idx].best_cur_acc,
                            BMAS[idx].best_bma_acc, samplers[idx].eta, samplers[idx].T,
                            np.array(loss_chains[idx]).sum(), abs(adjusted_corrections[idx])))
            print(
                'Epoch {} Chain {} NLL: {:.2f} Best NLL: {:.2f} BMA NLL: {:.2f}  Best BMA NLL: {:.2f}  Brier: {:.3e}  Best Brier: {:.3e}'.format(
                    epoch, idx, BMAS[idx].nll, BMAS[idx].best_nll, BMAS[idx].bma_nll, BMAS[idx].best_bma_nll,
                    BMAS[idx].brier_score, BMAS[idx].best_brier))

        if pars.save_after is not None:
            if epoch % pars.save_after == 0:
                for i in range(len(nets)):
                    file_name = 'net_weights_' + str(i) + '.pt'
                    torch.save(nets[i].state_dict(), os.path.join(net_dir, file_name))
                print('Model saved in: ' + net_dir)

        print('')

    end = time.time()
    print('Time used {:.2f}s'.format(end - start))
