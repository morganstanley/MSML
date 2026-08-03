#!/usr/bin/env python3
"""
Hyperparameter analysis functionality.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.colors import LogNorm

from loading import load_log_with_safetensors
from task_shift import (
    extract_swept_params, get_param_value_from_config, find_valid_multirun_subdirs,
    normalize_error_values
)


def hyperparam_analysis(multirun_path: Path, output_dir: Path = None):
    """Perform hyperparameter analysis: create heatmaps of Test Task MSE vs hyperparameter pairs.
    
    For each value of task.distrib_param, creates heatmaps showing average Test Task MSE
    as a function of pairs of other hyperparameters.
    
    Args:
        multirun_path: Path to the multirun directory
        output_dir: Directory to save plots (optional)
    """
    if not multirun_path.exists():
        raise FileNotFoundError(f"Multirun directory not found: {multirun_path}")
    
    # Get swept parameters
    swept_params = extract_swept_params(multirun_path)
    if not swept_params:
        print("No swept parameters found in multirun.yaml")
        return
    
    print(f"Found swept parameters: {swept_params}")
    
    # Find all valid run subdirectories
    run_subdirs = find_valid_multirun_subdirs(multirun_path)
    
    if not run_subdirs:
        raise FileNotFoundError(f"No completed runs found in multirun")
    
    # Collect data from all runs
    run_data = []  # List of dicts with parameters and MSE
    
    for subdir in run_subdirs:
        config_path = multirun_path / subdir / "config.json"
        log_path = multirun_path / subdir / "log.json"
        
        if not config_path.exists() or not log_path.exists():
            continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Use safetensor loading for faster analysis
        subdir_path = multirun_path / subdir
        log = load_log_with_safetensors(subdir_path)
        
        # Extract parameter values
        param_values = {}
        for param_path in swept_params:
            value = get_param_value_from_config(config, param_path)
            param_values[param_path] = value
        
        # Extract MSE for all tasks (final iteration, average over context length)
        task_mses = {}
        eval_steps = log.get("eval/step", [])
        if eval_steps:
            final_step_idx = -1
            
            # Look for all task metrics (Test tasks, Fixed tasks)
            for key, value in log.items():
                if key.startswith("eval/") and key != "eval/step":
                    task_name = key.split("/")[1]  # Extract task name from "eval/Task Name"
                    
                    for metric_name, metric_values in value.items():
                        if "Transformer |" in metric_name and "(RelErr)" not in metric_name and metric_values:
                            # Get MSE values for final step
                            final_mse_values = normalize_error_values(metric_values[final_step_idx])
                            if final_mse_values is not None and len(final_mse_values) > 0:
                                task_mse = np.mean(final_mse_values)
                                task_mses[task_name] = task_mse
                                break  # Take the first available MSE metric for this task
        
        if task_mses:
            # Add all task MSEs to param_values
            for task_name, mse_value in task_mses.items():
                param_values[f'{task_name}_mse'] = mse_value
            run_data.append(param_values)
    
    if not run_data:
        print("No valid task MSE data found")
        return
    
    print(f"Collected data from {len(run_data)} runs")
    
    # Group by task.distrib_param values
    distrib_param_groups = {}
    for data in run_data:
        distrib_param_val = data.get('task.distrib_param')
        if distrib_param_val is not None:
            if distrib_param_val not in distrib_param_groups:
                distrib_param_groups[distrib_param_val] = []
            distrib_param_groups[distrib_param_val].append(data)
    
    if not distrib_param_groups:
        print("No task.distrib_param found in swept parameters")
        return
    
    # Get other parameters (excluding task.distrib_param)
    other_params = [p for p in swept_params if p != 'task.distrib_param']
    
    if len(other_params) < 2:
        print(f"Need at least 2 other parameters for heatmap analysis, found: {other_params}")
        return
    
    # Create output directory
    if output_dir is None:
        output_dir = multirun_path
    heatmap_dir = output_dir / "hyperparam_heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    
    # Get all available task names from the data
    all_task_names = set()
    for data in run_data:
        for key in data.keys():
            if key.endswith('_mse'):
                task_name = key[:-4]  # Remove '_mse' suffix
                all_task_names.add(task_name)
    
    print(f"Found tasks: {sorted(all_task_names)}")
    
    # Generate heatmaps for each distrib_param value, each task, and each pair of other parameters
    for distrib_param_val, group_data in distrib_param_groups.items():
        print(f"\nProcessing distrib_param = {distrib_param_val} ({len(group_data)} runs)")
        
        # Generate all pairs of other parameters
        param_pairs = list(itertools.combinations(other_params, 2))
        
        for task_name in sorted(all_task_names):
            task_mse_key = f'{task_name}_mse'
            
            # Check if this task has data in this distrib_param group
            has_task_data = any(task_mse_key in data for data in group_data)
            if not has_task_data:
                continue
                
            print(f"  Creating heatmaps for task: {task_name}")
            
            for param1, param2 in param_pairs:
                print(f"    {param1} vs {param2}")
                
                # Extract unique values for each parameter
                param1_values = sorted(set(data[param1] for data in group_data if param1 in data))
                param2_values = sorted(set(data[param2] for data in group_data if param2 in data))
                
                if len(param1_values) < 2 or len(param2_values) < 2:
                    print(f"      Skipping: insufficient parameter variation ({len(param1_values)} x {len(param2_values)})")
                    continue
                
                # Create MSE grid
                mse_grid = np.full((len(param2_values), len(param1_values)), np.nan)
                
                # Fill grid with MSE values for this specific task
                for data in group_data:
                    if param1 in data and param2 in data and task_mse_key in data:
                        try:
                            i = param1_values.index(data[param1])
                            j = param2_values.index(data[param2])
                            mse_grid[j, i] = data[task_mse_key]
                        except ValueError:
                            continue
                
                # Check if we have enough data points
                valid_points = np.sum(~np.isnan(mse_grid))
                if valid_points < 4:
                    print(f"      Skipping: insufficient data points ({valid_points})")
                    continue
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                
                # Use log scale for MSE values
                valid_mse = mse_grid[~np.isnan(mse_grid)]
                vmin, vmax = np.min(valid_mse), np.max(valid_mse)
                
                im = plt.imshow(mse_grid, aspect='auto', origin='lower', 
                              norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis')
                
                # Set ticks and labels
                plt.xticks(range(len(param1_values)), [str(v) for v in param1_values])
                plt.yticks(range(len(param2_values)), [str(v) for v in param2_values])
                
                # Add colorbar
                cbar = plt.colorbar(im)
                cbar.set_label(f'{task_name} MSE (log scale)')
                
                # Add text annotations with MSE values
                for i in range(len(param1_values)):
                    for j in range(len(param2_values)):
                        if not np.isnan(mse_grid[j, i]):
                            text_color = 'white' if mse_grid[j, i] < np.exp(np.log(vmin) + 0.7 * (np.log(vmax) - np.log(vmin))) else 'black'
                            plt.text(i, j, f'{mse_grid[j, i]:.2e}', 
                                   ha='center', va='center', color=text_color, fontsize=8)
                
                # Labels and title
                param1_name = param1.split('.')[-1]
                param2_name = param2.split('.')[-1]
                plt.xlabel(f'{param1_name} ({param1})')
                plt.ylabel(f'{param2_name} ({param2})')
                plt.title(f'{task_name} MSE Heatmap\ndistrib_param={distrib_param_val}\n{param1_name} vs {param2_name}')
                
                plt.tight_layout()
                
                # Save plot
                safe_param1 = param1.replace('.', '_')
                safe_param2 = param2.replace('.', '_')
                safe_task_name = task_name.replace(' ', '_').replace('.', '_')
                filename = f"heatmap_{safe_task_name}_distrib_{distrib_param_val}_{safe_param1}_vs_{safe_param2}.png"
                output_path = heatmap_dir / filename
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"      Saved: {output_path}")
                
                plt.close()
    
    print(f"\nHyperparameter analysis completed. Heatmaps saved to: {heatmap_dir}")