from typing import Callable

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax import Array

from icl.models import Model, get_model_name
from icl.tasks import Sampler, Task

# Preds = {
#     task_name: {model_name: Array[n_samples, n_points], ...},
#     ...
# }
Preds = dict[str, dict[str, Array]]


@jax.jit
def mse(a: Array, b: Array) -> Array:
    # Average over batch dimension and all but sequence dimension
    return sum_except_dim(jnp.square(a - b), dim=1) / a.shape[0]


def sum_except_dim(x, dim):
    # Get all dimension indices
    dims = tuple(i for i in range(x.ndim) if i != dim)
    return jnp.sum(x, axis=dims)

def sum_except_dims(x, dims):
    # Get all dimension indices except those in dims
    all_dims = tuple(i for i in range(x.ndim) if i not in dims)
    return jnp.sum(x, axis=all_dims)

@jax.jit
def error_per_sample_and_seq_pos(a: Array, b: Array) -> Array:
    return sum_except_dims(jnp.square(a - b), dims=(0, 1))

@jax.jit  
def relative_error(a: Array, b: Array) -> Array:
    n = error_per_sample_and_seq_pos(a, b)
    d = error_per_sample_and_seq_pos(b, 0.)
    return n / (d + 1e-6)


def get_oracle_step(task: Task) -> Callable[[Array, Array], Array]:
    def step(xs: Array, ws: Array, t) -> Array:
        preds = task.evaluate_oracle(xs, ws, t)
        return preds

    return step


def get_baseline_step(model: Model) -> Callable[[Array, Array], Array]:
    def step(data: Array, targets: Array) -> Array:
        preds = model(data, targets)
        return preds

    return step


def get_bsln_preds(train_task: Task, j_batch_samplers: dict[str, Sampler], n_samples: int, batch_size: int) -> Preds:
    # Initialize preds and compile oracle and baseline models
    preds = {}
    p_oracle = jax.pmap(get_oracle_step(train_task), axis_name="device")
    p_bsln_models = {
        get_model_name(model): jax.pmap(get_baseline_step(model), axis_name="device")
        for model in train_task.get_default_eval_models()
    }
    # Loop through eval tasks
    for task_name, j_sample_batch in j_batch_samplers.items():
        # Initialize task preds
        preds[task_name] = {"True": []}
        for model_name in p_bsln_models:
            preds[task_name][model_name] = []
        # Accumulate preds...
        for i in range(1, n_samples // batch_size + 1):
            xs, ws, weights, ys, attention_mask = j_sample_batch(i, evl=True)
            # print(f"Evaluating {task_name} batch {i} with {xs.shape[0]} samples and {ys.shape[1]} points ({xs.shape = }, {ws.shape = }, {ys.shape = })")
            n_points = ys.shape[2]
            target_shape = ys.shape[1:]  # (batch_size, n_points,) or (batch_size, n_points, n_dims)
            new_shape = (ys.shape[0] * ys.shape[1],) + target_shape[1:]  # (batch_size * n_devices, n_points,) or (batch_size * n_devices, n_points, n_dims)
            preds[task_name]["True"].append(p_oracle(xs, ws, ys).reshape(new_shape))
            for model_name, p_model in p_bsln_models.items():  # ...for baseline models
                # print(f"Evaluating {model_name} on {task_name} batch {i} with {xs.shape[0]} samples and {n_points} points ({xs.shape = }, {ws.shape = }, {ys.shape = })")
                preds[task_name][model_name].append(p_model(xs, ys).reshape(new_shape))
        # Concatenate preds
        preds[task_name]["True"] = jnp.concatenate(preds[task_name]["True"])
        for model_name in p_bsln_models:
            preds[task_name][model_name] = jnp.concatenate(preds[task_name][model_name])
    return preds


def get_model_preds(
    state: TrainState,
    p_eval_step: Callable[[TrainState, Array, Array, Array], Array],
    j_batch_samplers: dict[str, Sampler],
    n_samples: int,
    batch_size: int,
) -> Preds:
    preds = {}
    for task_name, j_sample_batch in j_batch_samplers.items():
        preds[task_name] = {"Transformer": []}
        for i in range(1, n_samples // batch_size + 1):
            xs, _, weights, ys, attention_mask = j_sample_batch(i, evl=True)
            n_points = ys.shape[2]
            target_shape = ys.shape[1:]  # (batch_size, n_points,) or (batch_size, n_points, n_dims)
            new_shape = (ys.shape[0] * ys.shape[1],) + target_shape[1:]  # (batch_size * n_devices, n_points,) or (batch_size * n_devices, n_points, n_dims)
            preds[task_name]["Transformer"].append(p_eval_step(state, xs, ys, attention_mask).reshape(new_shape))
        preds[task_name]["Transformer"] = jnp.concatenate(preds[task_name]["Transformer"])
    return preds
