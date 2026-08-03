#!/usr/bin/env python3
"""
Weight analysis functionality.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from loading import load_log_with_safetensors
from task_shift import normalize_error_values, find_valid_multirun_subdirs
from mean_min_best_mse import load_all_logs


def plot_weights_analysis(log: dict, run_id: str, output_dir: Path = None):
    """Plot weights diagnostics (ESS, Sum, KL divergence) vs training steps."""
    steps = log.get("train/step", [])
    if not steps:
        print("No training steps found in log")
        return
    
    # Check if weight diagnostics exist in the log
    ess_values = log.get("train/final/ess", [])
    sum_values = log.get("train/final/sum", [])
    kl_values = log.get("train/final/kl_from_uniform", [])
    
    if not ess_values and not sum_values and not kl_values:
        print("No weight diagnostics found in log. Weights analysis may not be available for this run.")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    
    # Plot ESS (Effective Sample Size)
    if ess_values and len(ess_values) == len(steps):
        ax1.plot(steps, ess_values, 'b-', linewidth=2, label='ESS')
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Effective Sample Size")
        ax1.set_title(f"ESS vs Training Step - {run_id}")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
    else:
        ax1.text(0.5, 0.5, 'ESS data not available', transform=ax1.transAxes, 
                ha='center', va='center', fontsize=12)
        ax1.set_title(f"ESS vs Training Step - {run_id}")
    
    # Plot Sum of weights
    if sum_values and len(sum_values) == len(steps):
        ax2.plot(steps, sum_values, 'g-', linewidth=2, label='Sum')
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Sum of Final Weights")
        ax2.set_title(f"Sum of Final Weights vs Training Step - {run_id}")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Sum data not available', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=12)
        ax2.set_title(f"Sum of Final Weights vs Training Step - {run_id}")
    
    # Plot KL divergence from uniform
    if kl_values and len(kl_values) == len(steps):
        ax3.plot(steps, kl_values, 'r-', linewidth=2, label='KL divergence')
        ax3.set_xlabel("Training Step")
        ax3.set_ylabel("KL Divergence from Uniform")
        ax3.set_title(f"KL Divergence from Uniform vs Training Step - {run_id}")
        ax3.grid(True, alpha=0.3)
        
        # Only set log scale if there are positive KL values
        has_positive_kl = any(kl_val > 0 for kl_val in kl_values)
        if has_positive_kl:
            ax3.set_yscale('log')
    else:
        ax3.text(0.5, 0.5, 'KL data not available', transform=ax3.transAxes, 
                ha='center', va='center', fontsize=12)
        ax3.set_title(f"KL Divergence from Uniform vs Training Step - {run_id}")
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = Path("outputs") / run_id
    output_path = output_dir / "weights_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Weights analysis plot saved to: {output_path}")
    
    plt.show()


def plot_weights_analysis_multirun(run_paths: list, output_dir: Path = None, run_labels: list = None):
    """Plot weights diagnostics comparison across multiple runs.
    
    Args:
        run_paths: List of Path objects pointing to runs or multirun subdirs
        output_dir: Directory to save plots (optional)
        run_labels: Custom labels for runs (optional)
    """
    if not run_paths:
        print("No run paths provided for weights analysis")
        return
    
    # Load all logs using existing function
    print("Loading all log files for weights analysis...")
    loaded_data = load_all_logs(run_paths, run_labels)
    
    if not loaded_data['logs']:
        print("No valid log data found")
        return
    
    # Check if any runs have weight diagnostics
    runs_with_weights = {}
    for run_label in loaded_data['run_labels']:
        log = loaded_data['logs'][run_label]
        steps = log.get("train/step", [])
        ess_values = log.get("train/final/ess", [])
        sum_values = log.get("train/final/sum", [])
        kl_values = log.get("train/final/kl_from_uniform", [])
        
        if steps and (ess_values or sum_values or kl_values):
            runs_with_weights[run_label] = {
                'steps': steps,
                'ess': ess_values if len(ess_values) == len(steps) else [],
                'sum': sum_values if len(sum_values) == len(steps) else [],
                'kl': kl_values if len(kl_values) == len(steps) else []
            }
    
    if not runs_with_weights:
        print("No weight diagnostics found in any of the runs")
        return
    
    # Create the plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    
    # Choose colormap based on number of runs
    num_runs = len(runs_with_weights)
    if num_runs > 10:
        cmap = plt.get_cmap('tab20' if num_runs <= 20 else 'hsv')
        colors = [cmap(i / num_runs) for i in range(num_runs)]
    else:
        # Use discrete colors for few runs
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    # Plot ESS for all runs
    for i, (run_label, run_data) in enumerate(runs_with_weights.items()):
        color = colors[i] if i < len(colors) else colors[i % len(colors)]
        
        if run_data['ess']:
            ax1.plot(run_data['steps'], run_data['ess'], 'o-', color=color, 
                    linewidth=2, markersize=3, label=run_label)
    
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Effective Sample Size")
    ax1.set_title("ESS vs Training Step")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot Sum for all runs
    for i, (run_label, run_data) in enumerate(runs_with_weights.items()):
        color = colors[i] if i < len(colors) else colors[i % len(colors)]
        
        if run_data['sum']:
            ax2.plot(run_data['steps'], run_data['sum'], 'o-', color=color, 
                    linewidth=2, markersize=3, label=run_label)
    
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Sum of Final Weights")
    ax2.set_title("Sum of Final Weights vs Training Step")
    ax2.grid(True, alpha=0.3)
    
    # Plot KL divergence for all runs
    for i, (run_label, run_data) in enumerate(runs_with_weights.items()):
        color = colors[i] if i < len(colors) else colors[i % len(colors)]
        
        if run_data['kl']:
            ax3.plot(run_data['steps'], run_data['kl'], 'o-', color=color, 
                    linewidth=2, markersize=3, label=run_label)
    
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("KL Divergence from Uniform")
    ax3.set_title("KL Divergence from Uniform vs Training Step")
    ax3.grid(True, alpha=0.3)
    
    # Only set log scale if there are positive KL values
    has_positive_kl = any(any(kl_val > 0 for kl_val in run_data['kl']) 
                         for run_data in runs_with_weights.values() if run_data['kl'])
    if has_positive_kl:
        ax3.set_yscale('log')
    
    # Add shared legend below plots (consistent with shift analysis)
    handles1, labels1 = ax1.get_legend_handles_labels()
    if handles1:  # Only add legend if we have data
        fig.legend(handles1, labels1, loc='lower center', ncol=min(len(labels1), 4), 
                  bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save plot
    if output_dir is None:
        # Check if we're analyzing multiruns by looking at the first run path
        if run_paths and (run_paths[0] / "multirun.yaml").exists():
            output_dir_to_use = run_paths[0]  # Use the multirun directory
        else:
            output_dir_to_use = Path("outputs")
    else:
        output_dir_to_use = output_dir
        
    output_path = output_dir_to_use / "weights_analysis_multirun.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Multirun weights analysis plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n=== Weights Analysis Summary ===")
    for run_label, run_data in runs_with_weights.items():
        print(f"\n{run_label}:")
        if run_data['ess']:
            final_ess = run_data['ess'][-1]
            print(f"  Final ESS: {final_ess:.6f}")
        if run_data['sum']:
            final_sum = run_data['sum'][-1]
            print(f"  Final Sum: {final_sum:.6f}")
        if run_data['kl']:
            final_kl = run_data['kl'][-1]
            print(f"  Final KL: {final_kl:.6f}")
