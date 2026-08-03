#!/usr/bin/env python3
"""
Task shift analysis functionality.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import jax
from jax import jit
import equinox as eqx
from functools import partial
import yaml
from scipy.optimize import curve_fit
from tqdm import tqdm

from loading import load_log_with_safetensors


def extract_task_shift_distance(task_name: str) -> float:
    """Extract shift distance from task name.
    
    Args:
        task_name: Task name like "Test tasks" or "Fixed task 0.25"
    
    Returns:
        float: Shift distance (0.0 for Test tasks, X.Y for Fixed task X.Y)
    """
    if task_name == "Test tasks":
        return 0.0
    elif task_name.startswith("Fixed task "):
        try:
            return float(task_name.replace("Fixed task ", ""))
        except ValueError:
            return float('inf')  # Unknown tasks go to end
    else:
        return float('inf')  # Unknown tasks go to end


def icl_power_law(k, D, alpha, C):
    """Power law function: D/(k+1)^alpha + C"""
    return C / ((k + 1) ** alpha)


def normalize_error_values(values):
    """Normalize error values to handle both old and new logging formats.
    
    Args:
        values: Either list of scalars (old format) or list of lists (new format)
    
    Returns:
        List of scalars (averaged over samples if needed)
    """
    if not isinstance(values,jnp.ndarray):
        # Convert to numpy array for easier handling
        values =jnp.array(values)
    if values.ndim == 2:
        # New format: average over batch dimension for each position
        return jnp.mean(values, axis=0)
    elif values.ndim == 1:
        return values
    else:
        # Old format: already scalars
        return values


@jit
def compute_min_mean_end_mse_over_context(mse_values: jnp.ndarray) -> tuple[float, float, float]:
    """Compute min, mean, and end MSE over context length (JIT compiled).
    
    Args:
        mse_values: MSE values over context positions
        
    Returns:
        tuple: (min_mse_excluding_first, mean_mse_all, end_mse)
    """
    # Skip first position (index 0) for min MSE as per original code
    min_mse = jnp.min(mse_values[1:]) if len(mse_values) > 1 else mse_values[0]
    mean_mse = jnp.mean(mse_values)
    end_mse = mse_values[-1]  # MSE at last context position
    return min_mse, mean_mse, end_mse


@jit  
def compute_auc_trapz(x_values: jnp.ndarray, y_values: jnp.ndarray) -> float:
    """Compute area under curve using trapezoidal rule (JIT compiled).
    
    Args:
        x_values: X coordinates (shift distances), must be sorted
        y_values: Y coordinates (MSE values)
        
    Returns:
        float: Area under the curve
    """
    return y_values[-1]
    # return jnp.trapezoid(y_values, x=x_values)


def compute_best_auc_for_baseline(log: dict, baseline_type: str) -> float:
    """Compute the best AUC (minimal mean MSE AUC) for a given baseline type.
    
    Args:
        log: The log dictionary
        baseline_type: Either 'Ridge' or 'True' to specify which baseline to use
        
    Returns:
        float: The minimal mean MSE AUC value, or float('inf') if computation fails
    """
    # Reuse the existing logic from extract_min_mse_params_for_baseline
    eval_steps = log.get("eval/step", [])
    
    # Extract evaluation metrics for all steps
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    # Find tasks that match our criteria and baseline
    task_data = {}
    for task_name, metrics in eval_metrics.items():
        if task_name == "Test tasks" or task_name.startswith("Fixed task"):
            selected_metric = None
            for metric_name, values in metrics.items():
                if f"Transformer | {baseline_type}" in metric_name and "(RelErr)" not in metric_name and values is not None:
                    selected_metric = (metric_name, values)
                    break
            
            if selected_metric:
                metric_name, values = selected_metric
                shift_distance = extract_task_shift_distance(task_name)
                task_data[task_name] = (shift_distance, values)
    
    if not task_data:
        return float('inf')
    
    # Sort tasks by shift distance for consistent ordering
    sorted_tasks = sorted(task_data.items(), key=lambda x: x[1][0])
    task_names = [task_name for task_name, _ in sorted_tasks]
    shift_distances = jnp.array([shift_dist for _, (shift_dist, _) in sorted_tasks])
    
    # Collect MSE data for all steps and tasks
    num_steps = len(eval_steps)
    num_tasks = len(sorted_tasks)
    
    all_min_mse =np.zeros((num_steps, num_tasks))
    all_mean_mse =np.zeros((num_steps, num_tasks))
    all_end_mse =np.zeros((num_steps, num_tasks))


    for task_idx, (task_name, (shift_dist, values)) in enumerate(sorted_tasks):
        assert num_steps == len(values), "Mismatch in number of evaluation steps"
        for step_idx in range(num_steps):
            if step_idx < len(values):
                mse_values = normalize_error_values(values[step_idx])
                if mse_values is not None and len(mse_values) > 0:
                    mse_jax = jnp.array(mse_values)
                    min_mse, mean_mse, end_mse = compute_min_mean_end_mse_over_context(mse_jax)
                    all_min_mse[step_idx, task_idx] = float(min_mse)
                    all_mean_mse[step_idx, task_idx] = float(mean_mse)
                    all_end_mse[step_idx, task_idx] = float(end_mse)
                else:
                    all_min_mse[step_idx, task_idx] = float('inf')
                    all_mean_mse[step_idx, task_idx] = float('inf')
                    all_end_mse[step_idx, task_idx] = float('inf')
            else:
                all_min_mse[step_idx, task_idx] = float('inf')
                all_mean_mse[step_idx, task_idx] = float('inf')
                all_end_mse[step_idx, task_idx] = float('inf')
    
    # Convert to JAX arrays for optimized computation
    all_min_mse_jax = jnp.array(all_min_mse)
    all_mean_mse_jax = jnp.array(all_mean_mse)
    all_end_mse_jax = jnp.array(all_end_mse)
    
    # Find best step and return the minimal mean AUC
    def compute_step_auc(step_idx):
        mean_log_mse = jnp.log(all_min_mse_jax[step_idx])
        return compute_auc_trapz(shift_distances, mean_log_mse)
    
    step_aucs = jax.vmap(compute_step_auc)(jnp.arange(num_steps))
    best_step, min_auc = jnp.argmin(step_aucs), float(jnp.min(step_aucs))

    # Build new log by removing everything except the best step
    new_log = {"eval/step": [eval_steps[int(best_step)]]}
    for task_name in task_names:
        new_log[f"eval/{task_name}"] = {}
        for metric_name, values in eval_metrics[task_name].items():
                new_log[f"eval/{task_name}"][metric_name] = [values[int(best_step)]]
    print("Eval steps:", new_log["eval/step"])

    
    return new_log, min_auc

def extract_power_law_params(log: dict) -> dict:
    """Extract power law parameters (alpha, C) for all tasks from log data.
    
    Returns:
        dict: {task_name: (alpha, C, r_squared)} for successfully fitted tasks
    """
    eval_steps = log.get("eval/step", [])
    if not eval_steps:
        return {}
    
    # Extract evaluation metrics for the final step
    final_step_idx = -1
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    results = {}
    
    for task_name, metrics in eval_metrics.items():
        # Look for preferred metric (Ridge) first, then fallback to True
        preferred_metric = None
        fallback_metric = None
        
        for metric_name, values in metrics.items():
            if "(RelErr)" not in metric_name and values is not None:
                if "Transformer | Ridge" in metric_name:
                    preferred_metric = (metric_name, values)
                elif "Transformer | True" in metric_name:
                    fallback_metric = (metric_name, values)
        
        # Use preferred metric if available, otherwise fallback
        selected_metric = preferred_metric or fallback_metric
        
        if selected_metric:
            metric_name, values = selected_metric
            # Get MSE values for final step
            mse_values = normalize_error_values(values[final_step_idx])
            if mse_values is None or len(mse_values) < 3:
                continue
                
            k =jnp.arange(len(mse_values))  # Context lengths: 0, 1, 2, ...
            
            try:
                # Fit the power law curve
                initial_guess = [0., 1.0, mse_values[0]]
                
                popt, pcov = curve_fit(icl_power_law, k, mse_values, p0=initial_guess, 
                                     bounds=([0, 0, 0], [jnp.inf,jnp.inf,jnp.inf]), maxfev=5000)
                
                D_fit, alpha_fit, C_fit = popt
                
                # Compute R-squared
                y_pred = icl_power_law(k, D_fit, alpha_fit, C_fit)
                ss_res =jnp.sum((mse_values - y_pred) ** 2)
                ss_tot =jnp.sum((mse_values -jnp.mean(mse_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                results[task_name] = (alpha_fit, C_fit, r_squared)
                
            except Exception as e:
                # Skip failed fits
                continue
    
    return results


def extract_swept_params(multirun_path: Path) -> list:
    """Extract swept parameter names from multirun.yaml file.
    
    Args:
        multirun_path: Path to the multirun directory
    
    Returns:
        list: List of parameter paths that were swept (e.g., ['task.distrib_param'])
    """
    multirun_yaml_path = multirun_path / "multirun.yaml"
    
    if not multirun_yaml_path.exists():
        return []
    
    try:
        with open(multirun_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract swept parameters from hydra config
        params = config.get('hydra', {}).get('sweeper', {}).get('params', {})
        
        if not params:
            return []
        
        # Return list of parameter paths
        return list(params.keys())
        
    except Exception as e:
        print(f"Warning: Could not parse multirun.yaml: {e}")
        return []


def get_param_value_from_config(config: dict, param_path: str):
    """Get parameter value from config using dotted path.
    
    Args:
        config: Configuration dictionary
        param_path: Dotted parameter path (e.g., 'task.distrib_param')
    
    Returns:
        Parameter value or None if not found
    """
    keys = param_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return None


def group_runs_by_other_params(run_data: list, optimize_params: list, all_swept_params: list) -> dict:
    """Group runs by non-optimized parameters.
    
    Args:
        run_data: List of run info dicts with 'config', 'log', 'path', etc.
        optimize_params: List of parameters to optimize (e.g., ['task.n_tasks', 'train.clip_max_norm'])  
        all_swept_params: List of all swept parameters from multirun.yaml
        
    Returns:
        dict: {other_params_tuple: [runs] averaged over add_seed}
    """
    other_params = [p for p in all_swept_params if p not in optimize_params and p != 'add_seed']

    
    groups = {}
    for run in run_data:
        # Create tuple of "other" parameter values for grouping
        other_values = []
        for param in other_params:
            value = get_param_value_from_config(run['config'], param)
            other_values.append((param, value))
        
        other_key = tuple(other_values)
        if other_key not in groups:
            groups[other_key] = []
        groups[other_key].append(run)

    for other_key, runs in groups.items():
        print(f"Grouping {len(runs)} runs for other params: {other_key}")
        new_group = {}
        for run in runs:
            # create tuple of "optimize" parameter values for grouping
            optimize_values = []
            for param in optimize_params:
                value = get_param_value_from_config(run['config'], param)
                optimize_values.append((param, value))
            optimize_key = tuple(optimize_values)
            if optimize_key not in new_group:
                new_group[optimize_key] = []
            new_group[optimize_key].append(run)
        for optimize_key, runs in new_group.items():
            print(f"  Optimize params: {optimize_key} with {len(runs)} runs")
        groups[other_key] = average_over_seed(new_group)

    return groups

def average_over_seed(run_groups: dict) -> list:
    import jax
    import pathlib

    @eqx.filter_jit
    def mean(a, axis=None):
        return jnp.exp(jnp.mean(jnp.log(a), axis=axis))

    @eqx.filter_jit
    def old_mean_stack(args):
        new_args =jnp.stack([a for a in args], axis=0)
        return mean(new_args, axis=0)

    @eqx.filter_jit
    def mean_stack(args):
        ret = jnp.zeros_like(args[0])
        for a in args:
            ret = ret + jnp.log(a)
        return jnp.exp(ret / len(args))

    def avg_func(*args):
        if isinstance(args[0], (str, pathlib.Path)):
            #print(f"Got string/path with {len(args)} args")
            return args[0]
        elif isinstance(args[0], (int, float)):
            #print(f"Got scalar with {len(args)} args")
            return mean(jnp.array(args))
        elif isinstance(args[0], list):
            #print(f"Got lists with {len(args[0])} elements and {len(args)} args")
            new_args = [jnp.array(a) for a in args]
            shapes = [a.shape for a in new_args]
            s = shapes[0]
            if not all([sh == s for sh in shapes]):
                print(f"Warning: Inconsistent shapes for averaging: {shapes}, trying to recover by keeping {s}")
                new_args = [a for a in new_args if a.shape == s]
            return mean_stack(new_args)
        elif isinstance(args[0],jnp.ndarray):
            #print(f"Got jax arrays with shape {args[0].shape} and {len(args)} args")
            shapes = [a.shape for a in args]
            s = shapes[0]
            if not all([sh == s for sh in shapes]):
                print(f"Warning: Inconsistent shapes for averaging: {shapes}, trying to recover by keeping {s}")
                args = [a for a in args if a.shape == s]
            return mean_stack(args)
        else:
            raise ValueError(f"Unsupported type for averaging: {type(args[0])}")
    @eqx.filter_jit
    def std(a, axis=None):
        return jnp.exp(jnp.std(jnp.log(a), axis=axis))
    @eqx.filter_jit
    def old_std_stack(args):
        new_args =jnp.stack([a for a in args], axis=0)
        return std(new_args, axis=0)

    @eqx.filter_jit
    def std_stack(args):
        ret = jnp.zeros_like(args[0])
        mean = jnp.log(mean_stack(args))
        for a in args:
            new_a = jnp.log(a)
            ret = ret + (new_a - mean) ** 2
        return jnp.exp(jnp.sqrt(ret / len(args)))

    def std_func(*args):
        if isinstance(args[0], (str, pathlib.Path)):
            ret = args[0]
        elif isinstance(args[0], (int, float)):
            ret = std(jnp.array(args))
        elif isinstance(args[0], list):
            new_args = [jnp.array(a) for a in args]
            shapes = [a.shape for a in new_args]
            s = shapes[0]
            if not all([sh == s for sh in shapes]):
                print(f"Warning: Inconsistent shapes for averaging: {shapes}, trying to recover by keeping {s}")
                new_args = [a for a in new_args if a.shape == s]
            ret =  std_stack(new_args)
        elif isinstance(args[0],jax.Array):
            shapes = [a.shape for a in args]
            s = shapes[0]
            if not all([sh == s for sh in shapes]):
                print(f"Warning: Inconsistent shapes for averaging: {shapes}, trying to recover by keeping {s}")
                args = [a for a in args if a.shape == s]
            ret = std_stack(args)
        else:
            raise ValueError(f"Unsupported type for std computation: {type(args[0])}")
        # print(f"std_func: {args} -> {ret}") 
        return ret

    new_runs = []
    FAST=False
    for optimize_key, runs in run_groups.items():
        if not FAST:
            new_log = {}
            for task_name, metrics in tqdm(runs[0]['log'].items()):
                if task_name == "eval/step":
                    new_log[task_name] = runs[0]['log']["eval/step"]
                    continue
                new_log[task_name] = {}
                print(f"For {task_name}, averaging metrics: {list(metrics.keys())}")
                for metric_name, values in tqdm(metrics.items(), leave=False, desc=f"Processing {task_name}"):
                    if "Std" in metric_name or "RelErr" in metric_name:
                        continue
                    metric_values = []
                    for run in runs:
                        if task_name in run['log'] and metric_name in run['log'][task_name]:
                            metric_values.append(run['log'][task_name][metric_name])
                    new_log[task_name][metric_name] = avg_func(*metric_values)
                    new_log[task_name][f"{metric_name}_Std"] = std_func(*metric_values)
            res = runs[0].copy()
            res['log'] = new_log
        else:
            res = runs[0]
        new_runs.append(res)

    return new_runs

def find_best_param_combination_by_auc(run_group: list, optimize_params: list, baseline_type: str, cached_aucs: dict = None) -> tuple:
    """Find the best parameter combination within a group of runs based on AUC.
    
    Args:
        run_group: List of runs that share the same "other" parameters
        optimize_params: List of parameters to optimize over
        baseline_type: 'Ridge' or 'True' for MSE baseline type
        cached_aucs: Optional dict of {run_name: auc_value} for performance optimization
        
    Returns:
        tuple: (best_run, best_auc, best_param_values) or (None, float('inf'), {}) if no valid runs
    """
    best_run = None
    best_auc = float('inf')
    best_param_values = {}
    
    for run in run_group:
        # Extract parameter values for this run
        param_values = {}
        for param in optimize_params:
            param_values[param] = get_param_value_from_config(run['config'], param)
        
        # Get AUC from cache or compute it
        if cached_aucs and run['name'] in cached_aucs:
            min_auc = cached_aucs[run['name']]
        else:
            # Compute the minimal AUC for this run using the helper function
            log = run['log']
            updated_log, min_auc = compute_best_auc_for_baseline(log, baseline_type)
            run['log'] = updated_log  # Update log to only contain best step
            
            # Cache the result for future use
            if cached_aucs is not None:
                cached_aucs[run['name']] = min_auc
        
        if min_auc == float('inf'):  # No valid data
            continue
        
        # Update best if this is better
        if min_auc < best_auc:
            best_auc = min_auc
            best_run = run
            best_param_values = param_values
                
    return best_run, best_auc, best_param_values


def format_parameter_legend(other_params_tuples: list, best_param_values: dict, max_length: int = 50) -> str:
    """Create a formatted legend name with smart truncation.
    
    Args:
        other_params_tuples: List of (param_name, param_value) tuples for non-optimized params
        best_param_values: Dict of {param_name: param_value} for optimized params
        max_length: Maximum allowed length for the legend string
        
    Returns:
        str: Formatted legend name with smart truncation
    """
    max_length = 1_000_000
    def shorten_param_name(param_name: str) -> str:
        """Convert parameter name to shorter form."""
        parts = param_name.split('.')
        if len(parts) >= 2:
            # Use last two parts for clarity (e.g., "task.n_tasks" -> "task.n_tasks", "train.optimizer.lr" -> "optimizer.lr")
            return '.'.join(parts[-2:]) if len(parts) > 2 else param_name
        return param_name
    
    # Format other parameters
    other_parts = []
    for param, value in other_params_tuples:
        short_name = shorten_param_name(param)
        other_parts.append(f"{short_name}={value}")
    
    # Format optimized parameters
    opt_parts = []
    for param, value in best_param_values.items():
        short_name = shorten_param_name(param)
        opt_parts.append(f"{short_name}={value}")
    
    # Combine parts
    other_str = ", ".join(other_parts)
    opt_str = ", ".join(opt_parts)
    
    if other_str:
        full_legend = f"{other_str} | BEST({opt_str})"
    else:
        full_legend = f"BEST({opt_str})"
    
    # Apply smart truncation if needed
    if len(full_legend) <= max_length:
        return full_legend
    
    # Try truncation strategies
    # Strategy 1: Truncate parameter values
    if len(full_legend) > max_length:
        def truncate_value(value_str, max_val_len=8):
            if len(str(value_str)) > max_val_len:
                return f"{str(value_str)[:max_val_len-3]}..."
            return str(value_str)
        
        # Re-format with truncated values
        other_parts_short = []
        for param, value in other_params_tuples:
            short_name = shorten_param_name(param)
            other_parts_short.append(f"{short_name}={truncate_value(value)}")
        
        opt_parts_short = []
        for param, value in best_param_values.items():
            short_name = shorten_param_name(param)
            opt_parts_short.append(f"{short_name}={truncate_value(value)}")
        
        other_str_short = ", ".join(other_parts_short)
        opt_str_short = ", ".join(opt_parts_short)
        
        if other_str_short:
            truncated_legend = f"{other_str_short} | BEST({opt_str_short})"
        else:
            truncated_legend = f"BEST({opt_str_short})"
        
        if len(truncated_legend) <= max_length:
            return truncated_legend
    
    # Strategy 2: Use ellipsis at the end
    return full_legend[:max_length-3] + "..."


def create_run_display_names(multirun_path: Path, run_subdirs: list) -> dict:
    """Create display names for runs based on their actual parameter values.
    
    Args:
        multirun_path: Path to the multirun directory
        run_subdirs: List of run subdirectory names (e.g., ['0', '1'])
    
    Returns:
        dict: {run_index: display_name} mapping based on actual parameter values
    """
    # Get swept parameters from multirun.yaml
    swept_params = extract_swept_params(multirun_path)
    
    if not swept_params:
        return {}
    
    display_names = {}
    
    for subdir in run_subdirs:
        config_path = multirun_path / subdir / "config.json"
        
        if not config_path.exists():
            continue
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Build display name from parameter values
            name_parts = []
            for param_path in swept_params:
                value = get_param_value_from_config(config, param_path)
                if value is not None:
                    param_name = param_path.split('.')[-1]  # Get last part of dotted path
                    name_parts.append(f"{param_name}={value}")
            
            if name_parts:
                display_names[int(subdir)] = ", ".join(name_parts)
            
        except Exception as e:
            print(f"Warning: Could not read config for run {subdir}: {e}")
            continue
    
    return display_names


def find_valid_multirun_subdirs(multirun_path: Path, return_paths: bool = False) -> list:
    """Find all valid subdirectories in a multirun that have either log.json or safetensor files.
    
    Args:
        multirun_path: Path to the multirun directory
        return_paths: If True, return Path objects; if False, return strings (default)
        
    Returns:
        list: List of subdirectory names (as strings) or Path objects, sorted numerically
    """
    subdirs = []
    for subdir in multirun_path.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            has_log = (subdir / "log.json").exists()
            has_safetensors = (subdir / "eval_results").exists() and any((subdir / "eval_results").glob("eval_step_*.safetensors"))
            if has_log or has_safetensors:
                if return_paths:
                    subdirs.append(subdir)
                else:
                    subdirs.append(subdir.name)
    
    # Sort numerically
    if return_paths:
        subdirs.sort(key=lambda x: int(x.name))
    else:
        subdirs.sort(key=int)
    return subdirs


def plot_task_shift_analysis(run_paths: list, output_dir: Path = None, run_labels: list = None, optimize_params: list = None):
    """Plot alpha and C parameters vs task shift for multiple runs.
    
    Args:
        run_paths: List of Path objects pointing to runs or multirun subdirs
        output_dir: Directory to save plots (optional)
        run_labels: Custom labels for runs (optional)
        optimize_params: List of parameters to optimize over (e.g., ['task.n_tasks', 'train.clip_max_norm'])
    """
    if not run_paths:
        print("No run paths provided for task shift analysis")
        return
    
    # Collect data from all runs
    data = {}  # {run_label: [(task_center, alpha, C, r_squared, task_name), ...]}
    
    for i, run_path in enumerate(run_paths):
        run_path = Path(run_path)
        
        # Determine run label
        if run_labels and i < len(run_labels):
            run_label = run_labels[i]
        else:
            run_label = run_path.name
        
        # Check if this is a multirun directory or single run
        if (run_path / "multirun.yaml").exists():
            # This is a multirun directory
            subdirs = find_valid_multirun_subdirs(run_path, return_paths=True)
            
            if optimize_params:
                # Parameter optimization mode - group runs and find best parameter combinations
                swept_params = extract_swept_params(run_path)
                if not swept_params:
                    print(f"Warning: No swept parameters found in {run_path}/multirun.yaml")
                    continue
                
                # Validate that optimize_params exist in swept_params
                invalid_params = [param for param in optimize_params if param not in swept_params]
                if invalid_params:
                    print(f"Error: Parameter optimization requested for parameters not swept in multirun: {invalid_params}")
                    print(f"Available swept parameters: {swept_params}")
                    continue
                
                # Collect all run data
                all_runs = []
                for subdir in subdirs:
                    config_path = subdir / "config.json"
                    if not config_path.exists():
                        continue
                    
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    log = load_log_with_safetensors(subdir)
                    
                    all_runs.append({
                        'config': config,
                        'log': log,
                        'path': subdir,
                        'name': subdir.name
                    })
                
                # Group runs by non-optimized parameters
                run_groups = group_runs_by_other_params(all_runs, optimize_params, swept_params)
                
                # Create AUC cache for performance optimization
                auc_cache = {}
                
                # For each group, find best parameter combination
                for group_key, run_group in run_groups.items():
                    best_run, best_auc, best_param_values = find_best_param_combination_by_auc(
                        run_group, optimize_params, 'Ridge', auc_cache
                    )
                    
                    if best_run is None:
                        continue
                    
                    # Create display name for this group using smart formatting
                    group_label = format_parameter_legend(group_key, best_param_values, max_length=100)
                    
                    # Extract power law parameters for the best run
                    config = best_run['config']
                    log = best_run['log']
                    task_centers = config.get('eval', {}).get('task_centers', [])
                    power_law_params = extract_power_law_params(log)
                    
                    run_data = []
                    
                    # Add Test tasks (task center = 0)
                    if "Test tasks" in power_law_params:
                        alpha, C, r_squared = power_law_params["Test tasks"]
                        run_data.append((0.0, alpha, C, r_squared, "Test tasks"))
                    
                    # Add Fixed tasks
                    for task_center in task_centers:
                        task_name = f"Fixed task {task_center}"
                        if task_name in power_law_params:
                            alpha, C, r_squared = power_law_params[task_name]
                            run_data.append((task_center, alpha, C, r_squared, task_name))
                    
                    if run_data:
                        data[group_label] = run_data
                        
            else:
                # Standard mode - process each subrun separately
                # Extract parameter names from multirun.yaml if no custom labels provided
                if run_labels is None:
                    param_names = create_run_display_names(run_path, [subdir.name for subdir in subdirs])
                    if param_names:
                        run_labels = [param_names.get(int(subdir.name), subdir.name) for subdir in subdirs]
                
                # Process each subrun as a separate run
                for subdir_idx, subdir in enumerate(subdirs):
                    # Load config and log for this subrun
                    config_path = subdir / "config.json"
                    log_path = subdir / "log.json"
                    
                    if not config_path.exists() or not log_path.exists():
                        continue
                    
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    log = load_log_with_safetensors(subdir)
                    
                    # Extract task centers from config
                    task_centers = config.get('eval', {}).get('task_centers', [])
                    
                    # Extract power law parameters for all tasks
                    power_law_params = extract_power_law_params(log)
                    
                    # Create run data for this subrun
                    if run_labels and subdir_idx < len(run_labels):
                        # Use custom name for this subrun
                        subrun_label = run_labels[subdir_idx].strip()
                    else:
                        subrun_label = f"{run_label}-{subdir.name}"
                    run_data = []
                    
                    # Add Test tasks (task center = 0)
                    if "Test tasks" in power_law_params:
                        alpha, C, r_squared = power_law_params["Test tasks"]
                        run_data.append((0.0, alpha, C, r_squared, "Test tasks"))
                    
                    # Add Fixed tasks
                    for task_center in task_centers:
                        task_name = f"Fixed task {task_center}"
                        if task_name in power_law_params:
                            alpha, C, r_squared = power_law_params[task_name]
                            run_data.append((task_center, alpha, C, r_squared, task_name))
                    
                    if run_data:
                        data[subrun_label] = run_data
        
        elif (run_path / "log.json").exists():
            # This is a single run
            config_path = run_path / "config.json"
            log_path = run_path / "log.json"
            
            if not config_path.exists():
                continue
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            log = load_log_with_safetensors(run_path)
            
            # Extract task centers from config
            task_centers = config.get('eval', {}).get('task_centers', [])
            
            # Extract power law parameters for all tasks
            power_law_params = extract_power_law_params(log)
            
            run_data = []
            
            # Add Test tasks (task center = 0)
            if "Test tasks" in power_law_params:
                alpha, C, r_squared = power_law_params["Test tasks"]
                run_data.append((0.0, alpha, C, r_squared, "Test tasks"))
            
            # Add Fixed tasks
            for task_center in task_centers:
                task_name = f"Fixed task {task_center}"
                if task_name in power_law_params:
                    alpha, C, r_squared = power_law_params[task_name]
                    run_data.append((task_center, alpha, C, r_squared, task_name))
            
            if run_data:
                data[run_label] = run_data
        
        else:
            print(f"Warning: {run_path} is neither a valid run nor multirun directory")
            continue
    
    if not data:
        print("No valid data found for task shift analysis")
        return
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Choose colormap based on number of runs
    num_runs = len(data)
    if num_runs > 10:
        # Use colormap for many runs
        cmap = plt.get_cmap('tab20' if num_runs <= 20 else 'hsv')
        colors = [cmap(i / num_runs) for i in range(num_runs)]
    else:
        # Use discrete colors for few runs
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    max_shift = float('inf')
    
    for i, (run_label, run_data) in enumerate(data.items()):
        if not run_data:
            continue
        
        # Sort by task center
        run_data.sort(key=lambda x: x[0])
        
        task_centers = [x[0] for x in run_data if x[0] <= max_shift]
        alphas = [x[1] for x in run_data if x[0] <= max_shift]
        Cs = [x[2] for x in run_data if x[0] <= max_shift]
        
        color = colors[i] if i < len(colors) else colors[i % len(colors)]
        
        # Plot alpha vs task center
        ax1.plot(task_centers, alphas, 'o-', color=color, linewidth=2, 
                markersize=6, label=run_label)
        
        # Plot C vs task center
        ax2.plot(task_centers, Cs, 'o-', color=color, linewidth=2, 
                markersize=6, label=run_label)
    
    # Configure alpha plot
    ax1.set_xlabel("Task Center (Task Shift)")
    ax1.set_ylabel("Alpha (Power Law Exponent)")
    ax1.set_title("Alpha vs Task Shift")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Configure C plot
    ax2.set_xlabel("Task Center (Task Shift)")
    ax2.set_ylabel("C (Asymptotic Error)")
    ax2.set_title("C vs Task Shift")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        # Check if we're analyzing multiruns by looking at the first run path
        if run_paths and (run_paths[0] / "multirun.yaml").exists():
            output_dir = run_paths[0]  # Use the multirun directory
        else:
            output_dir = Path("outputs")
    output_path = output_dir / "task_shift_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Task shift analysis plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n=== Task Shift Analysis Summary ===")
    for run_label, run_data in data.items():
        if not run_data:
            continue
        print(f"\n{run_label}:")
        for task_center, alpha, C, r_squared, task_name in sorted(run_data, key=lambda x: x[0]):
            print(f"  {task_name} (center={task_center}): alpha={alpha:.4f}, C={C:.6f}, R²={r_squared:.4f}")
