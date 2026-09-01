import json
import yaml
import os
from pathlib import Path
import time
import optax
import chex

import jax
import jax.numpy as jnp
import jax.random as jr
import orbax.checkpoint as ocp
import logging
from flax import jax_utils
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jax import Array
from ml_collections import ConfigDict
from hydra.core.hydra_config import HydraConfig
from safetensors.numpy import save_file

import icl.utils as u
from icl.evaluate import Preds, get_bsln_preds, get_model_preds, mse, relative_error, error_per_sample_and_seq_pos, sum_except_dim
from icl.models import Transformer, SingleSeqTransformer, get_model
from icl.optim import get_optimizer_and_lr_schedule
from icl.tasks import Sampler, Task, get_task, get_task_name
from icl.reweighting import process_log_weights


def initialize(model: Transformer | SingleSeqTransformer, config: ConfigDict) -> tuple[FrozenDict, Array]:
    params_rng, dropout_rng = jr.split(jr.PRNGKey(config.model.seed))
    dummy_data = jnp.ones((config.task.batch_size, config.model.n_points, config.task.n_dims), dtype=model.dtype)
    
    if isinstance(model, SingleSeqTransformer):
        # For SingleSeqTransformer, targets have n_dims dimensions (not scalar)
        dummy_targets = jnp.ones((config.task.batch_size, config.model.n_points, config.task.n_dims), dtype=model.dtype)
        dummy_mask = jnp.ones((config.task.batch_size, config.model.n_points, config.model.n_points)).astype(bool)
    else:
        # For regular Transformer, targets are scalar
        dummy_targets = jnp.ones((config.task.batch_size, config.model.n_points), dtype=model.dtype)
        dummy_mask = jnp.ones((config.task.batch_size, 2 * config.model.n_points, 2 * config.model.n_points)).astype(bool)
    
    variables = jax.jit(model.init)(params_rng, dummy_data, dummy_targets, dummy_mask)
    return variables["params"], dropout_rng


def get_sharded_batch_sampler(task: Task) -> Sampler:
    n_devices = jax.local_device_count()

    def sample_batch(step: int, evl=False) -> tuple[Array, Array, Array, Array, Array]:
        data, tasks, weights, targets, attention_mask = task.sample_batch(step, evl=evl)
        batch_size = data.shape[0]
        batch_per_device = batch_size // n_devices
        
        # Shard data across devices  
        data = data.reshape(n_devices, batch_per_device, *data.shape[1:])
        tasks = tasks.reshape(n_devices, batch_per_device, *tasks.shape[1:])
        weights = weights.reshape(n_devices, batch_per_device, *weights.shape[1:])
        targets = targets.reshape(n_devices, batch_per_device, *targets.shape[1:])
        
        # Expand attention mask to match batch dimensions
        # From (seq_len, seq_len) to (n_devices, batch_per_device, seq_len, seq_len)
        attention_mask = jnp.broadcast_to(
            attention_mask[None, None, :, :], 
            (n_devices, batch_per_device, *attention_mask.shape)
        )
        
        return data, tasks, weights, targets, attention_mask

    return sample_batch


def train_step(state: TrainState,
               data: Array,
               log_weights: Array,
               targets: Array,
               attention_mask: Array,
               dropout_rng: Array,
               t: int,
               T: int,
               alpha0: float,
               T_ramp_ratio: float,
               use_weights:bool,
               clip_max_norm: float,
               normalize:bool,
               ) -> tuple[Array, TrainState]:

    dropout_rng = jr.fold_in(dropout_rng, state.step + 1)

    if not use_weights:
        log_weights = jnp.zeros(data.shape[0], dtype=jnp.float32)

    weights, diagnostics = process_log_weights(
        log_weights, t, T, alpha0=alpha0, T_ramp_ratio=T_ramp_ratio, normalize=normalize
        )

    def loss_fn(params, weights):
        preds = state.apply_fn({"params": params}, data, targets, attention_mask, training=True, rngs={"dropout": dropout_rng})
        # Compute weighted loss: weights should have shape (batch_size,)
        # batch_losses = jnp.square(preds - targets).mean(axis=1)  # Mean over sequence length
        # Sum  over dims if exists and over seq length
        seq_len = targets.shape[1]
        batch_losses = sum_except_dim(jnp.square(preds - targets), 0) / seq_len  # Mean over sequence length
        # jax.debug.print("Weights: mean={}, median={}, min={}, max={}",
        #                jnp.mean(weights), jnp.median(weights), jnp.min(weights), jnp.max(weights))
        if use_weights:
            weighted_loss = jnp.sum(batch_losses * weights)
        else:
            weighted_loss = jnp.mean(batch_losses)
        return weighted_loss, preds

    
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, _), grads = grad_fn(state.params, weights)
    grads = jax.lax.pmean(grads, axis_name="device")
    loss = jax.lax.pmean(loss, axis_name="device")
    chex.assert_shape(loss, ())
    state = state.apply_gradients(grads=grads)
    global_norm = optax.global_norm(grads)
    diagnostics['grad_norm'] = global_norm 
    diagnostics['is_grad_clipped'] = global_norm > clip_max_norm 
    return loss, state, diagnostics


def eval_step(state: TrainState, data: Array, targets: Array, attention_mask: Array) -> Array:
    preds = state.apply_fn({"params": state.params}, data, targets, attention_mask, training=False)
    return preds


def _init_log(bsln_preds: Preds, n_dims: int) -> dict:
    log = {"train/step": [], "train/lr": [], "train/loss": [], "eval/step": []}
    for _task_name, _task_preds in bsln_preds.items():
        log[f"eval/{_task_name}"] = {}
        for _bsln_name, _bsln_preds in _task_preds.items():
            log[f"eval/{_task_name}"][f"Transformer | {_bsln_name}"] = []
            log[f"eval/{_task_name}"][f"Transformer | {_bsln_name} (RelErr)"] = []
            if _bsln_name != "True":
                # Use per-sample errors for consistency with main training loop
                _errs_per_sample = jnp.mean(jnp.square(_bsln_preds - _task_preds["True"]), axis=1) / n_dims
                _rel_errs_per_sample = jnp.mean(jnp.square(_bsln_preds - _task_preds["True"]) / (jnp.square(_task_preds["True"]) + 1), axis=1)
                log[f"eval/{_task_name}"][f"{_bsln_name} | True"] = _errs_per_sample.tolist()
                log[f"eval/{_task_name}"][f"{_bsln_name} | True (RelErr)"] = _rel_errs_per_sample.tolist()
    return log

def update_log_with_diagnostics(log: dict, diagnostics: dict) -> None:
    main_key = "train"
    for subkey, value in diagnostics.items():
        if isinstance(value, dict):
            for subsubkey, subvalue in value.items():
                try:
                    log[f"{main_key}/{subkey}/{subsubkey}"].append(u.to_float(subvalue))
                except KeyError:
                    log[f"{main_key}/{subkey}/{subsubkey}"] = [u.to_float(subvalue)]
        else:
            try:
                log[f"{main_key}/{subkey}"].append(u.to_float(value))
            except KeyError:
                log[f"{main_key}/{subkey}"] = [u.to_float(value)]

def train(config: ConfigDict) -> None:
    # Setup train experiment with Hydra output directory
    hydra_cfg = HydraConfig.get()
    exp_dir = Path(hydra_cfg.runtime.output_dir)
    exp_name = f"train_{u.get_hash(config)}"
    
    logging.info(f"Train Experiment\nNAME: {exp_name}\nOUTPUT_DIR: {exp_dir}\n")
    
    add_seed = config.get("add_seed", 0)
    for key, value in config.items():
        if isinstance(value, ConfigDict):
            for sub_key, sub_value in value.items():
                if "seed" in sub_key:
                    logging.info(f"Updated {key}.{sub_key} to {sub_value + add_seed}")
                    config[key][sub_key] += add_seed
    
    # Validate config 
    assert config.model.n_points == config.task.n_max_points, "Model n_points must match Task n_max_points"
    assert config.task.n_points <= config.task.n_max_points, "Task n_points must be less than or equal to Task n_max_points"
    assert config.eval.eval_n_points <= config.task.n_max_points, "Eval n_points must be less than or equal to Task n_points"
    if config.model.name == "SingleSeqTransformer":
        assert config.task.n_dims == config.model.n_out, "Task n_dims must match Model n_outs"

    # Config is already saved by Hydra, but save our version too  
    config_file = exp_dir / "config.json"
    with open(config_file, "w") as f:
        serializable_config = u._convert_for_json(config)
        f.write(json.dumps(serializable_config, indent=2))

    # Model, optimizer and lr schedule
    model = get_model(**config.model, dtype=jnp.dtype(config.dtype))
    #logging.info(u.tabulate_model(model, config.task.n_dims, config.model.n_points, config.task.batch_size))
    params, dropout_rng = initialize(model, config)
    clip_max_norm = config.training.get("clip_max_norm", float("inf"))
    if "clip_max_norm" not in config.training:
        config.training.clip_max_norm = clip_max_norm
    tx, lr = get_optimizer_and_lr_schedule(**config.training, params=params)
    logging.info("Initialized Model, Optimizer and LR Schedule")

    # Train state
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = jax_utils.replicate(state)
    dropout_rngs = jr.split(dropout_rng, jax.local_device_count())
    logging.info("Initialized TrainState")

    # Data samplers
    train_task = get_task(**config.task, dtype=jnp.dtype(config.dtype))
    j_sample_train_batch = (get_sharded_batch_sampler(train_task))
    j_samplers_eval_batch = {
        get_task_name(task): (get_sharded_batch_sampler(task))
        for task in train_task.get_default_eval_tasks(**config.eval)
    }
    logging.info("Initialized Data Samplers")

    # Steps
    p_train_step = jax.pmap(
            train_step,
            axis_name="device",
            donate_argnums=0,
            # Static args: T, alpha0, T_ramp_ratio, use_weights, clip_max_norm
            static_broadcasted_argnums=(7, 8, 9, 10, 11, 12)
            )
    p_eval_step = jax.pmap(eval_step, axis_name="device")
    logging.info("Pmap'd Steps")

    use_weights = config.task.use_weights
    alpha0 = config.training.get("alpha0", 0.5)
    T_ramp_ratio = config.training.get("T_ramp_ratio", 0.)

    # Create eval results directory
    eval_results_dir = exp_dir / "eval_results"
    eval_results_dir.mkdir(exist_ok=True)

    # Evaluate baselines
    logging.info("Evaluate Baselines...")
    bsln_preds = get_bsln_preds(train_task, j_samplers_eval_batch, config.eval.n_samples, config.eval.batch_size)

    logging.info("=== Base evaluation ===")
    eval_tensors = {}
    for _task_name, _task_preds in bsln_preds.items():
        logging.info(f"Task: {_task_name}")
        # Find true baseline
        _bsln_name2 = None
        _bsln_preds2 = None
        for name, preds in _task_preds.items():
            if name == "True":
                _bsln_name2 = name
                _bsln_preds2 = preds
                break
        assert _bsln_name2 is not None, "True baseline not found in predictions"

        for _bsln_name1, _bsln_preds1 in _task_preds.items():
            _errs = error_per_sample_and_seq_pos(_bsln_preds1, _bsln_preds2) / config.task.n_dims
            _rel_errs = relative_error(_bsln_preds1, _bsln_preds2)
            chex.assert_shape(_errs, (config.eval.n_samples, config.eval.eval_n_points))
            chex.assert_shape(_rel_errs, (config.eval.n_samples, config.eval.eval_n_points))

            avg_errs = jnp.mean(_errs, axis=0)  # Mean over samples
            avg_rel_errs = jnp.mean(_rel_errs, axis=0)
            chex.assert_shape(avg_errs, (config.eval.eval_n_points,))
            chex.assert_shape(avg_rel_errs, (config.eval.eval_n_points,))

            std_errs = jnp.std(_errs, axis=0)  # Std over samples
            std_rel_errs = jnp.std(_rel_errs, axis=0)
            chex.assert_shape(std_errs, (config.eval.eval_n_points,))
            chex.assert_shape(std_rel_errs, (config.eval.eval_n_points,))

            avg_err = _errs.mean().item()
            avg_rel_err = _rel_errs.mean().item()
            
            # Convert to numpy for safetensors (JAX arrays need to be converted)
            _errs_np = jnp.asarray(avg_errs)
            _rel_errs_np = jnp.asarray(avg_rel_errs)
            
            # Store in tensors dict with safe key names
            safe_task_name = _task_name.replace(" ", "_").replace(".", "_")
            safe_bsln_name1 = _bsln_name1.replace(" ", "_").replace(".", "_")
            safe_bsln_name2 = _bsln_name2.replace(" ", "_").replace(".", "_")
            tensor_key_mse = f"{safe_task_name}_{safe_bsln_name1}_vs_{safe_bsln_name2}_MSE"
            tensor_key_rel = f"{safe_task_name}_{safe_bsln_name1}_vs_{safe_bsln_name2}_RelErr"
            
            eval_tensors[tensor_key_mse] = _errs_np
            eval_tensors[tensor_key_rel] = _rel_errs_np

            tensor_key_std_mse = f"{safe_task_name}_{safe_bsln_name1}_vs_{safe_bsln_name2}_MSE_Std"
            tensor_key_std_rel = f"{safe_task_name}_{safe_bsln_name1}_vs_{safe_bsln_name2}_RelErr_Std"
            eval_tensors[tensor_key_std_mse] = jnp.asarray(std_errs)
            eval_tensors[tensor_key_std_rel] = jnp.asarray(std_rel_errs)
            
            # Continue with original logging
            # log[f"eval/{_task_name}"][f"Transformer | {_bsln_name}"].append(_errs.tolist())
            # log[f"eval/{_task_name}"][f"Transformer | {_bsln_name} (RelErr)"].append(_rel_errs.tolist())

            logging.info(f"  { _bsln_name1 } vs { _bsln_name2 }: MSE={avg_err:.6f}, RelErr={avg_rel_err:.6f}")
    
    # Save evaluation results as safetensor file
    eval_step_file = eval_results_dir / f"baseline_eval_step.safetensors"
    save_file(eval_tensors, eval_step_file)
    logging.info(f"Saved evaluation results to: {eval_step_file}")

    # Loggers
    log = _init_log(bsln_preds, config.task.n_dims)

    # Setup checkpoint manager
    ckpt_mngr = ocp.CheckpointManager(exp_dir)
    
    # Training loop
    logging.info("Start Train Loop")
    train_losses = []
    epoch_size = max(1, config.eval.every)
    last_epoch_time = time.time()
    
    for i in range(1, config.training.total_steps + 1):
        # Train step
        data, _, weights, targets, attention_mask = j_sample_train_batch(i)

        loss, state, diagnostics = (
                p_train_step(
                    state,
                    data,
                    weights,
                    targets,
                    attention_mask,
                    dropout_rngs,
                    jnp.full(jax.local_device_count(), i, dtype=jnp.int32),
                    config.training.total_steps,
                    alpha0,
                    T_ramp_ratio,
                    use_weights,
                    clip_max_norm,
                    config.training.normalize_weights
                    )
                )
        loss = jax_utils.unreplicate(loss)
        train_losses.append(loss.item())
        log["train/step"].append(i)
        log["train/lr"].append(float(lr(i)))
        log["train/loss"].append(loss.item())
        update_log_with_diagnostics(log, diagnostics)

        # diagnostics = jax.tree.map(u.to_float, diagnostics)
        # yaml_diagnostics = yaml.dump(diagnostics)
        # print(f"Step {i} Diagnostics:\n{yaml_diagnostics}")

        # Evaluate
        if i % config.eval.every == 0 or i == config.training.total_steps:
            # Log time taken for the last epoch 
            t = time.time() - last_epoch_time

            # Calculate and print average training loss over last epoch
            recent_losses = train_losses[-epoch_size:]
            avg_train_loss = sum(recent_losses) / len(recent_losses)

            recent_clips = log["train/is_grad_clipped"][-epoch_size:]
            avg_clips = sum(recent_clips) / len(recent_clips)

            recent_ess = log["train/final/ess"][-epoch_size:]
            avg_ess = sum(recent_ess) / len(recent_ess)

            recent_p995 = log["train/soft_clipped/P99.5"][-epoch_size:]
            avg_p995 = sum(recent_p995) / len(recent_p995)

            recent_median = log["train/soft_clipped/median"][-epoch_size:]
            avg_median = sum(recent_median) / len(recent_median)

            recent_kl_original = log["train/original/kl_from_uniform"][-epoch_size:]
            avg_kl_original = sum(recent_kl_original) / len(recent_kl_original)

            recent_kl_final = log["train/final/kl_from_uniform"][-epoch_size:]
            avg_kl_final = sum(recent_kl_final) / len(recent_kl_final)
            
            # Log step and lr
            logging.info(f"Step: {i} [{t:.2f}s] | Train Loss (last {len(recent_losses)} steps): {avg_train_loss:.3f} | Clips: {avg_clips * 100:.2f}% | LR: {float(lr(i)):.6f}")
            logging.info(f"ESS: {avg_ess:.6f} | P99.5/Med.: {avg_p995 /  avg_median:.2f} | KL Orig.: {avg_kl_original:.6f} | KL Final: {avg_kl_final:.6f}")
            
            # Evaluate model
            eval_preds = get_model_preds(
                state, p_eval_step, j_samplers_eval_batch, config.eval.n_samples, config.eval.batch_size
            )
            
            # Prepare tensors for safetensors saving
            eval_tensors = {}
            
            # Log and print all evaluation metrics
            log["eval/step"].append(i)
            logging.info("=== Evaluation Metrics ===")
            for _task_name, _task_preds in bsln_preds.items():
                logging.info(f"Task: {_task_name}")
                for _bsln_name, _bsln_preds in _task_preds.items():
                    _errs = error_per_sample_and_seq_pos(eval_preds[_task_name]["Transformer"], _bsln_preds) / config.task.n_dims
                    _rel_errs = relative_error(eval_preds[_task_name]["Transformer"], _bsln_preds)
                    chex.assert_shape(_errs, (config.eval.n_samples, config.eval.eval_n_points))
                    chex.assert_shape(_rel_errs, (config.eval.n_samples, config.eval.eval_n_points))

                    avg_errs = jnp.mean(_errs, axis=0)  # Mean over samples
                    avg_rel_errs = jnp.mean(_rel_errs, axis=0)
                    chex.assert_shape(avg_errs, (config.eval.eval_n_points,))
                    chex.assert_shape(avg_rel_errs, (config.eval.eval_n_points,))

                    std_errs = jnp.std(_errs, axis=0)  # Std over samples
                    std_rel_errs = jnp.std(_rel_errs, axis=0)
                    chex.assert_shape(std_errs, (config.eval.eval_n_points,))
                    chex.assert_shape(std_rel_errs, (config.eval.eval_n_points,))

                    avg_err = _errs.mean().item()
                    avg_rel_err = _rel_errs.mean().item()
                    
                    # Convert to numpy for safetensors (JAX arrays need to be converted)
                    _errs_np = jnp.asarray(avg_errs)
                    _rel_errs_np = jnp.asarray(avg_rel_errs)
                    
                    # Store in tensors dict with safe key names
                    safe_task_name = _task_name.replace(" ", "_").replace(".", "_")
                    safe_bsln_name = _bsln_name.replace(" ", "_").replace(".", "_")
                    tensor_key_mse = f"{safe_task_name}_Transformer_vs_{safe_bsln_name}_MSE"
                    tensor_key_rel = f"{safe_task_name}_Transformer_vs_{safe_bsln_name}_RelErr"
                    
                    eval_tensors[tensor_key_mse] = _errs_np
                    eval_tensors[tensor_key_rel] = _rel_errs_np

                    tensor_key_std_mse = f"{safe_task_name}_Transformer_vs_{safe_bsln_name}_MSE_Std"
                    tensor_key_std_rel = f"{safe_task_name}_Transformer_vs_{safe_bsln_name}_RelErr_Std"
                    eval_tensors[tensor_key_std_mse] = jnp.asarray(std_errs)
                    eval_tensors[tensor_key_std_rel] = jnp.asarray(std_rel_errs)
                    
                    # Continue with original logging
                    # log[f"eval/{_task_name}"][f"Transformer | {_bsln_name}"].append(_errs.tolist())
                    # log[f"eval/{_task_name}"][f"Transformer | {_bsln_name} (RelErr)"].append(_rel_errs.tolist())

                    logging.info(f"  Transformer vs {_bsln_name}: MSE={avg_err:.6f}, RelErr={avg_rel_err:.6f}")
            
            # Save evaluation results as safetensor file
            eval_step_file = eval_results_dir / f"eval_step_{i:06d}.safetensors"
            save_file(eval_tensors, eval_step_file)
            logging.info(f"Saved evaluation results to: {eval_step_file}")

            # Save logs to Hydra output directory
            with open(exp_dir / "log.json", "w") as f:
                json.dump(log, f, indent=2)
            logging.info("Saved logs to Hydra output directory")

            # Checkpoint - save to Hydra output directory
            # ckpt_mngr.save(i, args=ocp.args.StandardSave(jax_utils.unreplicate(state)))
            logging.info("=========================")

            # Reset last epoch time
            last_epoch_time = time.time()

    # Save logs to Hydra output directory
    with open(exp_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Wrap up
    ckpt_mngr.wait_until_finished()
    jr.normal(jr.PRNGKey(0)).block_until_ready()
    return None
