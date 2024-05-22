#!/usr/bin/python
"""
Created on May 12, 2024
@author: Haoyang Zheng & Wei Deng
Code for Constrained Exploration via Reflected Replica Exchange Stochastic Gradient Langevin Dynamics. ICML 2024
This repository is built upon github.com/WayneDW/Variance_Reduced_Replica_Exchange_SGMCMC
"""
import os
import torch
import random
import pickle
import argparse
import numpy as np
import distutils.util
from tools import loader
import torch.utils.data as data
import models.cifar as cifar_models
import torchvision.models as models
from torchvision import datasets, transforms
from trainer import trainer_resgld, trainer_csgld


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def main(pars):
    """ Step 0: Numpy printing setup and set GPU and Seeds """
    print(pars)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    try:
        torch.cuda.set_device(pars.gpu)
    except RuntimeError:  # in case the device has only one GPU
        torch.cuda.set_device(0)
        pars.gpu = int(0)

    torch.manual_seed(pars.seed)
    torch.cuda.manual_seed(pars.seed)
    np.random.seed(pars.seed)
    random.seed(pars.seed)
    torch.backends.cudnn.deterministic = True

    """ Step 1: Preprocessing """
    if not torch.cuda.is_available():
        exit("CUDA does not exist!!!")

    net = cifar_models.__dict__['resnet'](num_classes=100, depth=pars.depth).cuda()

    nets = [net]
    for _ in range(1, pars.chains):
        nets.append(pickle.loads(pickle.dumps(net)))

    if pars.load:
        try:
            net_dir = './logs/'
            files = sorted(os.listdir(net_dir))
            net_dir = os.path.join(net_dir, files[pars.load_idx])
            files = sorted(os.listdir(net_dir))
            for i in range(len(nets)):
                nets[i].load_state_dict(
                    torch.load(os.path.join(net_dir, files[i]), map_location=torch.device('cuda:' + str(pars.gpu))))
            print('Load models from ' + net_dir + ' successfully!')
        except IndexError:
            print('Unable to load previous net. Training from scratch.')

    """ Step 2: Load Data """
    train_loader, test_loader = loader(pars.batch, pars.batch, pars)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    notcifar = datasets.SVHN(root='./data/SVHN', split='test', download=True, transform=transform_test)
    extra_loader = data.DataLoader(notcifar, batch_size=pars.batch, shuffle=False, num_workers=0)
    print('Load data successfully.')
    print('Training set: %.0f, Testing set: %.0f.' % (len(train_loader.dataset), len(test_loader.dataset)))

    """ Step 3: Bayesian Sampling """
    if pars.optimizer == 'resgld':
        trainer_resgld(nets, train_loader, test_loader, extra_loader, pars)
    elif pars.optimizer == 'csgld':
        trainer_csgld(nets[0], train_loader, test_loader, extra_loader, pars)
    else:
        trainer_resgld(nets, train_loader, test_loader, extra_loader, pars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search')
    parser.add_argument('-sn', default=1000, type=int, help='Sampling Epochs')
    parser.add_argument('-wdecay', default=5, type=float,
                        help='Samling weight decay (equivalent to 1e-4 in non-Bayes settings)')
    parser.add_argument('-lr', default=2e-5, type=float,
                        help='Sampling learning rate (equivalent to 1.0 in non-Bayes settings)')
    parser.add_argument('-momentum', default=0.9, type=float, help='Sampling momentum learning rate')
    parser.add_argument('-burn', default=0.6, type=float, help='burn in iterations for sampling (sn * burn)')
    parser.add_argument('-regularization', default=None, type=float, help='L2 regularization')

    # Parallel Tempering hyperparameters
    parser.add_argument('-chains', default=1, type=int, help='Total number of chains')
    parser.add_argument('-var_reduce', default=0, type=int,
                        help='n>0 means update variance reduction every n epochs; n divides 10')
    parser.add_argument('-period', default=2, type=int, help='estimate adaptive variance every [period] epochs')
    parser.add_argument('-T', default=0.05, type=float, help='Temperature for high temperature chain')
    parser.add_argument('-Tgap', default=0.2, type=float, help='Temperature gap between chains')
    parser.add_argument('-LRgap', default=0.66, type=float, help='Learning rate gap between chains')
    parser.add_argument('-Tanneal', default=1.02, type=float, help='temperature annealing factor')
    parser.add_argument('-LRanneal', default=0.984, type=float, help='lr annealing factor')
    parser.add_argument('-adapt_c', default=0, type=float,
                        help='adapt_c=1 is equivalent to running Alg. 2 in the appendix')
    parser.add_argument('-cool', default=20, type=int, help='No swaps happen during the cooling time after a swap')

    # other settings
    parser.add_argument('-data', default='cifar100', dest='data', help='CIFAR10/ CIFAR100')
    parser.add_argument('-depth', type=int, default=20, help='ResNet depth')
    parser.add_argument('-total', default=50000, type=int, help='Total data points')
    parser.add_argument('-batch', default=2048, type=int, help='Batch size')
    parser.add_argument('-seed', default=3407, type=int, help='Random Seed')
    parser.add_argument('-gpu', default=0, type=int, help='Default GPU')
    parser.add_argument('-alpha', default=0.3, type=float, help='forgetting rate')
    parser.add_argument('-bias_F', default=1.5e7, type=float, help='correction factor F')
    parser.add_argument('-cycle', default=1, type=int, help='Number of cycles')

    parser.add_argument('-if_domain', default=True, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='save samples and kl-divergence')
    parser.add_argument('-bound', default=4.0, type=float, help='Parameter bound')
    parser.add_argument("-optimizer", default="resgld", type=str, help="Optimizer")
    parser.add_argument('-save_after', default=None, type=int, help='Save model after x epochs')
    parser.add_argument('-load', default=False, type=lambda x: bool(distutils.util.strtobool(x)),
                        help='Load previous model')
    parser.add_argument('-load_idx', default=-1, type=int, help='Load model index')

    parser.add_argument('-anneal_lr', default=1e-6, type=float, help='Anneal rate')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')

    pars = parser.parse_args()

    main(pars)

