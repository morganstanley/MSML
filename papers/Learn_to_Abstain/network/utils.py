from typing import Tuple, Union
import importlib

import torch

_NUM_CLASSES = {'mnist': 10, 'svhn': 5, 'volatility': 2, 'bus': 3, 'lc': 3, 'ou': 2, 'lobster':2}
_DATA_NETWORK_PAIR = {
    'mnist': ('tinycnn', 'TinyCNN'), 
    'svhn' : ('resnet',  'resnet18'), 
    'volatility': ('lstm', 'LSTMSeq'), 
    'bus'  : ('vgg', 'vgg16'), 
    'lc'   : ('mlp', 'MLP'), 
    'ou'   : ('lstm', 'LSTMSeq'), 
    'lobster' : ('lstm', 'LSTMSeq')
    }
_DATA_SELECTIVENET_PAIR = {
    'mnist': ('tinycnn_selectivenet', 'TinyCNNSelectiveNet'), 
    'svhn' : ('resnet_selectivenet', 'resnet18selectivenet'), 
    'volatility': ('lstm', 'LSTMSeqSelectiveNet'), 
    'bus'  : ('vgg_selectivenet', 'vgg16selectivenet'), 
    'lc'   : ('mlp_selectivenet', 'MLPSelectiveNet'), 
    'ou'   : ('lstm', 'LSTMSeqSelectiveNet'), 
    'lobster' : ('lstm', 'LSTMSeqSelectiveNet')
    }
_DATA_INPUTDIM = {
    'mnist': None, 
    'svhn' : None, 
    'volatility': 2, 
    'bus'  : None, 
    'lc'   : 1805, 
    'ou'   : 1, 
    'lobster': 21
}

def build_network(config) -> Union[torch.nn.Module, Tuple[torch.nn.Module]]:

    dataset = config.dataset
    method  = config.method

    num_classes  = _NUM_CLASSES[dataset]
    network_name = _DATA_NETWORK_PAIR[dataset] if method != 'selectivenet' else _DATA_SELECTIVENET_PAIR[dataset]
    if method in ['deepgambler', 'adaptive', 'dac', 'isav2']:
        num_classes += 1
    module = importlib.import_module('network.'+network_name[0])
    model = getattr(module, network_name[1])(num_class=num_classes, input_dim=_DATA_INPUTDIM[dataset])
    if method == 'isa':
        slnet = getattr(module, network_name[1])(num_class=1, input_dim=_DATA_INPUTDIM[dataset])
        model = (model, slnet)

    return model