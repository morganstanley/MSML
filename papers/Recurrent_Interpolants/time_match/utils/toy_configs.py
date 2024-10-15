from typing import Optional

from numpy import array
from torch.nn import Module

from time_match.modules import (
    BlendModel,
    DDPMModel,
    FMModel,
    SGMModel,
    SIModel,
)
from time_match.nn import (
    FeedForward,
    FeedForwardResNet,
)
from time_match.utils.synthetic_data import AVAILABLE_DATASETS, generate_data


def get_net(
    name: str,
    in_dim: int,
    out_dim: Optional[int] = None,
) -> Module:
    if out_dim is None:
        out_dim = in_dim

    options = {
        'ff-1': { 'module': FeedForward, 'hidden': 256, 'num_layers': 1 },
        'ff-3': { 'module': FeedForward, 'hidden': 256, 'num_layers': 3 },
        'ff-6': { 'module': FeedForward, 'hidden': 256, 'num_layers': 6 },
        'ff-1-wide': { 'module': FeedForward, 'hidden': 1024, 'num_layers': 1 },
        'resnet-1': { 'module': FeedForwardResNet, 'hidden': 256, 'num_layers': 1 },
        'resnet-3': { 'module': FeedForwardResNet, 'hidden': 256, 'num_layers': 3 },
        'resnet-6': { 'module': FeedForwardResNet, 'hidden': 256, 'num_layers': 6 },
    }
    assert name in options

    hyperparams = options[name]

    module = hyperparams['module']
    hidden_dim = hyperparams['hidden']
    num_layers = hyperparams['num_layers']

    return module(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=out_dim,
    )


def get_data(
    name: str,
    size: Optional[int] = None,
    seed: Optional[int] = 1,
) -> array:
    """
    Args:
        name: In format "{name}-{size}", e.g. "moons-XS" or "swissroll-L"
    """
    size_mapping = {
        'XS': 1_000,
        'S': 10_000,
        'M': 50_000,
        'L': 100_000,
        'XL': 1_000_000,
    }

    dataset_name, dataset_size = name.split('-')
    assert dataset_name in AVAILABLE_DATASETS
    assert dataset_size in size_mapping

    if size is None:
        size = size_mapping[dataset_size]

    return generate_data(
        data=dataset_name,
        seed=seed,
        n_samples=size,
    )


def get_diffuser(
    model_name: str,
    net_name: str,
    dim: int,
) -> Module:
    if model_name[:4] == 'ddpm':
        _, scheduler = model_name.split('-')
        return DDPMModel(
            linear_start=1e-4,
            linear_end=0.2,
            n_timestep=100,
            net=get_net(net_name, dim),
            beta_schedule=scheduler,
        )
    if model_name == 'sgm':
        return SGMModel(
            linear_start=0.1,
            linear_end=20,
            n_timestep=100,
            net=get_net(net_name, dim),
        )
    if model_name == 'blend':
        raise NotImplementedError
        return BlendModel(
            n_timestep=100,
            net=get_net(net_name, dim, out_dim=2 * dim),
        )
    if model_name == 'fm':
        return FMModel(
            sigma_min=0.01,
            n_timestep=100,
            net=get_net(net_name, dim),
        )
    if model_name[:2] == 'si':
        _, gamma, interpolant = model_name.split('-')
        return SIModel(
            gamma=gamma,
            interpolant=interpolant,
            epsilon=0.1,
            n_timestep=100,
            start_noise=True,
            importance_sampling=True,
            velocity=get_net(net_name, dim),
            score=get_net(net_name, dim),
        )
