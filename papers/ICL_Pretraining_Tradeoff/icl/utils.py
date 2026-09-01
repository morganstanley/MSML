import hashlib
import json

from flax import jax_utils
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from ml_collections import ConfigDict
from omegaconf import ListConfig, DictConfig

from icl.models import Transformer, SingleSeqTransformer


def filter_config(config: ConfigDict) -> ConfigDict:
    with config.unlocked():
        for k, v in config.items():
            if v is None:
                del config[k]
            elif isinstance(v, ConfigDict):
                config[k] = filter_config(v)
    return config


def _convert_for_json(obj):
    """Convert OmegaConf objects to JSON-serializable formats."""
    if isinstance(obj, ListConfig):
        return list(obj)
    elif isinstance(obj, DictConfig):
        return dict(obj)
    elif isinstance(obj, ConfigDict):
        result = {}
        for k, v in obj.items():
            result[k] = _convert_for_json(v)
        return result
    elif isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(v) for v in obj]
    else:
        return obj

def get_hash(config: ConfigDict) -> str:
    serializable_config = _convert_for_json(config)
    config_str = json.dumps(serializable_config, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()

def pad_sequence(seq: Array, target_seq_len: int = None) -> Array:
    batch_size, seq_len, *rest_seq_shape = seq.shape
    dtype = seq.dtype
    
    # Right-pad sequence if target_seq_len is specified and larger
    if target_seq_len is not None and seq_len < target_seq_len:
        pad_len = target_seq_len - seq_len
        new_shape = (batch_size, pad_len, *rest_seq_shape)
        pad = jnp.zeros(new_shape, dtype=dtype)
        seq = jnp.concatenate([seq, pad], axis=1)
    
    return seq

def unpad_sequence(seq: Array, actual_seq_len: int = None) -> Array:
    if actual_seq_len is not None:
        return seq[:, :actual_seq_len]
    return seq

def to_seq(data: Array, targets: Array, target_seq_len: int = None) -> Array:
    batch_size, seq_len, n_dims = data.shape
    dtype = data.dtype
    
    # Right-pad data and targets if target_seq_len is specified and larger
    targets = pad_sequence(targets, target_seq_len)
    data = pad_sequence(data, target_seq_len)

    seq_len = data.shape[1]  # Update seq_len after padding
    
    data = jnp.concatenate([jnp.zeros((batch_size, seq_len, 1), dtype=dtype), data], axis=2)
    targets = jnp.concatenate([targets[:, :, None], jnp.zeros((batch_size, seq_len, n_dims), dtype=dtype)], axis=2)
    seq = jnp.stack([data, targets], axis=2).reshape(batch_size, 2 * seq_len, n_dims + 1)
    return seq


def seq_to_targets(seq: Array, actual_seq_len: int = None) -> Array:
    targets = seq[:, ::2, 0]
    # If actual_seq_len is specified, slice to remove padding
    if actual_seq_len is not None:
        targets = targets[:, :actual_seq_len]
    return targets


def tabulate_model(model: Transformer | SingleSeqTransformer, n_dims: int, n_points: int, batch_size: int) -> str:
    if isinstance(model, SingleSeqTransformer):
        # For SingleSeqTransformer, targets have n_dims dimensions (not scalar)
        dummy_data = jnp.ones((batch_size, n_points, n_dims), dtype=model.dtype)
        dummy_targets = jnp.ones((batch_size, n_points, n_dims), dtype=model.dtype)
        dummy_mask = jnp.ones((batch_size, model.n_points, model.n_points)).astype(bool)
    else:
        # For regular Transformer, targets are scalar
        dummy_data = jnp.ones((batch_size, n_points, n_dims), dtype=model.dtype)
        dummy_targets = jnp.ones((batch_size, n_points), dtype=model.dtype)
        dummy_mask = jnp.ones((batch_size, 2 * model.n_points, 2 * model.n_points)).astype(bool)
    
    return model.tabulate(jr.PRNGKey(0), dummy_data, dummy_targets, dummy_mask, training=False, depth=0)

def to_float(x):
    if isinstance(x, jnp.ndarray):
        x = jax_utils.unreplicate(x)
        return float(x.item())
    return float(x)
