import os
import sys
from typing import Dict, Tuple

import yaml
import argparse
import pickle as pkl
from datetime import datetime

from environ import Environ

def run_slearn(config) -> Tuple[Dict, Dict, Dict]:
    
    env = Environ(config)

    env.set_seed()

    trainloader, testloader = env.build_dataset() 

    selector = env.build_selector()
    selector.train(trainloader, testloader)

    print('<<< Final Evaluation:')
    eval_result_dict  = selector.eval(testloader)
    if config.verbose:
        selector.print_summary(eval_result_dict)

    print('<<< Check Fitting:')
    train_result_dict = selector.eval(trainloader)
    if config.verbose:
        selector.print_summary(train_result_dict)

    return eval_result_dict, train_result_dict, selector.logger, selector.config


if __name__ == '__main__':


    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
        )
    conf_parser.add_argument("-c", "--config_file", help="Specify config file", metavar="FILE", default='./experiment_configuration.yml')
    args, remaining_argv = conf_parser.parse_known_args()

    defaults = {}
    if args.config_file:
        with open(args.config_file) as f:
            config = yaml.safe_load(f)
        f.close()
        defaults.update(config)

    parser = argparse.ArgumentParser(parents=[conf_parser])
    parser.set_defaults(**defaults)

    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['mnist', 'svhn', 'volatility', 'bus', 'lc']
        )
    parser.add_argument(
        '--method',  
        type=str, 
        choices=['confidence', 'selectivenet', 'deepgambler', 'adaptive', 'oneside', 'isa', 'isav2']
        )
    parser.add_argument('--num_epoch',  type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    
    parser.add_argument('--lambda_uninform', type=float)
    parser.add_argument('--lambda_inform',   type=float)
    parser.add_argument('--clean_ratio', type=float)
    parser.add_argument('--data_num_ratio', type=float)
    parser.add_argument('--coverage_target', type=float)
    # volatility dataset configuration
    parser.add_argument('--context_size_vol', type=int)
    
    parser.add_argument('--alpha', type=float, help='selectivenet hyper-params')
    parser.add_argument('--lamda', type=float, help='selectivenet hyper-params')
    parser.add_argument('--O', type=float, help='deepgambler hyper-params')
    parser.add_argument('--pretrain_gambler', type=int, help='deepgambler hyper-params')
    parser.add_argument('--pretrain_adaptive', type=int, help='adaptive hyper-params')
    parser.add_argument('--momentum_adaptive', type=float, help='adaptive hyper-params')
    parser.add_argument('--pretrain_oneside', type=int, help='oneside hyper-params')
    parser.add_argument('--mu_oneside', type=float, help='oneside hyper-params')
    parser.add_argument('--beta', type=float, help='slearn hyper-params')
    parser.add_argument('--pretrain_isa', type=int, help='slearn hyper-params')
    parser.add_argument('--update_interval', type=int, help='slearn hyper-params')
    parser.add_argument('--use_smooth', action='store_true', help="smooth the loss weight")
    parser.add_argument('--sel_loss', type=int, help="selector training loss", choices=[1, 2])

    parser.add_argument('--gpus', type=str)
    parser.add_argument('--figure_folder', type=str)
    parser.add_argument('--checkpoint_window', type=int)
    parser.add_argument('--checkpoint_folder', type=str)
    parser.add_argument('--result_folder', type=str)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(remaining_argv)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print(f'device={args.gpus}')

    timestamp = datetime.today().strftime("%y%m%d%H%M%S")

    os.makedirs(args.checkpoint_folder, exist_ok=True)
    os.makedirs(args.figure_folder, exist_ok=True)
    os.makedirs(args.result_folder, exist_ok=True)

    eval_result_dict, train_result_dict, logger, config = run_slearn(args)

    config = {k:v for k, v in args._get_kwargs()}
    result_filepath = os.path.join(
        args.result_folder, 
        f'{args.dataset}_{args.method}_{args.data_num_ratio}_{args.clean_ratio}_{args.lambda_uninform}_{args.seed}_{timestamp}.pkl'
        )
    print(result_filepath)
    with open(result_filepath, 'wb') as f:
        pkl.dump(config, f)
        pkl.dump(eval_result_dict, f)
        pkl.dump(train_result_dict, f)
        pkl.dump(logger, f)
    f.close()
