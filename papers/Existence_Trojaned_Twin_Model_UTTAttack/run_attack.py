import os
import random
from collections import defaultdict
from typing import Dict

import torch
import numpy as np
import argparse
import yaml
import pickle as pkl
from datetime import datetime

from data.data_builder import DATA_BUILDER
from attacker.badnet import BadNet
from attacker.sig import SIG
from attacker.ref import Reflection 
from attacker.warp import WaNet
from attacker.imc import IMC
from attacker.utt import UTT
from trainer import TRAINER
from networks import NETWORK_BUILDER


def run_attack(config: Dict) -> Dict:

    seed = int(config['args']['seed'])
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    if config['train']['DISTRIBUTED'] and ('LOCAL_RANK' in os.environ):
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend=config['train'][config['args']['dataset']]['BACKEND'])
        config['train']['device'] = local_rank
        config['misc']['VERBOSE'] = False if local_rank != 0 else config['misc']['VERBOSE']
    else:
        config['train']['device'] = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    # Build dataset
    dataset = DATA_BUILDER(config=config)
    dataset.build_dataset()
    
    # Build network
    model = NETWORK_BUILDER(config=config)
    model.build_network()
    
    # Inject troj
    if config['args']['method'] == 'badnet':
        attacker = BadNet(config=config)
    elif config['args']['method'] == 'sig':
        attacker = SIG(config=config)
    elif config['args']['method'] == 'ref':
        attacker = Reflection(config=config)
    elif config['args']['method'] == 'warp':
        attacker = WaNet(databuilder=dataset, config=config)
    elif config['args']['method'] == 'imc':
        attacker = IMC(model=model.model, databuilder=dataset, config=config)
    elif config['args']['method'] == 'utt':
        attacker = UTT(config=config)
    else:
        raise NotImplementedError
    print(">>> Inject Trojan")
    if not attacker.dynamic:
        xi = config['args']['xi'] if config['args']['xi'] else config['attack']['XI']
        attacker.inject_trojan_static(dataset.trainset, mode='train')
        attacker.inject_trojan_static(dataset.testset,  mode='test', xi=xi)

    # training with trojaned dataset
    trainer = TRAINER(model=model.model, attacker=attacker, config=config)
    trainer.train(trainloader=dataset.trainloader, validloader=dataset.testloader)
    
    if (config['train']['DISTRIBUTED'] and local_rank==0) or (not config['train']['DISTRIBUTED']):
        
        attacker.save_trigger(config['attack']['TRIGGER_SAVE_DIR'])
        
        result_dict = trainer.eval(evalloader=dataset.testloader, use_best=True)
        result_dict = {k:v for k, v in result_dict.items()}
        result_dict.update({k:[v for _, v in v.val_record.items()] for k, v in trainer.metric_history.items()})
        result_dict['model'] = model.model.cpu().state_dict()
        result_dict['trigger'] = attacker.trigger
        
        return result_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method',  type=str, default='badnet', choices={'badnet', 'sig', 'ref', 'warp', 'imc', 'utt'})
    parser.add_argument('--dataset', type=str, default='gtsrb',  choices={'cifar10', 'gtsrb', 'imagenet'})
    parser.add_argument('--network', type=str, default='resnet18', choices={'resnet18', 'resnet34', 'vgg16', 'vgg19', 'densenet121', 'inceptionv3'})
    parser.add_argument('--inject_ratio', type=float, default=None)
    parser.add_argument('--budget', type=float, default=None)
    parser.add_argument('--surrogate_network', type=str, default=None, choices={'resnet18', 'resnet34', 'vgg16', 'vgg19', 'densenet121', 'inceptionv3'})
    parser.add_argument('--surrogate_ckpt', type=str, default=None)
    parser.add_argument('--xi', type=float, default=None)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--datadir', type=str, default='./data', help='dir where the data is saved')
    parser.add_argument('--ckptdir', type=str, default='./ckpt', help='dir to save ckpt')
    parser.add_argument('--resultdir', type=str, default='./result', help='dir to save trojaned models')
    parser.add_argument('--logdir',  type=str, default='./log', help='dir to save log file')
    parser.add_argument('--seed', type=str, default='77')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    with open('./experiment_configuration.yml') as f:
        config = yaml.safe_load(f)
    f.close()
    config['args'] = defaultdict(str)
    for k, v in args._get_kwargs():
        config['args'][k] = v

    result_dict = run_attack(config)

    if result_dict:
        if not os.path.exists(args.resultdir):
            os.makedirs(args.resultdir, exist_ok=True)
            
        timestamp = datetime.today().strftime("%y%m%d%H%M%S")
        result_file = f"{args.method}_{args.dataset}_{args.network}_{timestamp}.pkl"
        with open(os.path.join(args.resultdir, result_file), 'wb') as f:
            pkl.dump(config, f)
            pkl.dump(result_dict, f)
        f.close()
        print(f"Result saved to {os.path.join(args.resultdir, result_file)}")

