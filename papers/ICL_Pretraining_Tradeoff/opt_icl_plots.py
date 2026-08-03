import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np

from loading import load_log_with_safetensors
from task_shift import normalize_error_values, find_valid_multirun_subdirs
from mean_min_best_mse import load_all_logs, load_all_logs_with_param_optimization

def pretty_task_label(task_name: str) -> str:
    # For the color legend (tasks only)
    return task_name.replace("Fixed task", "Shift") if task_name.startswith("Fixed task") else task_name

def method_from_metric(metric_name: str) -> str:
    # "Transformer | Ridge" -> "Transformer"
    splitted = metric_name.split(" | ")[0].strip()
    if splitted == "LastValue":
        return "Last Value"
    return splitted

def format_task_name_for_display(task_name, metric_name):
    """Format task name for display in legends, replacing 'Fixed task' with 'Shifted task'."""
    if task_name.startswith("Fixed task"):
        task_name = task_name.replace("Fixed task", "Shifted task")
    splitted_metric = metric_name.split(" | ")
    name = f"{task_name} ({splitted_metric[0]})"
    return name

def plot_opt_icl_plots(run_paths: list, output_dir: Path = None, run_labels: list = None, optimize_params: list = None, ymin: float = None, ymax: float = None):
    """Plot ICL performance for multiple runs with parameter optimization."""
    print(f"Calling load_all_logs_with_param_optimization with runs: {run_paths} and optimize_params: {optimize_params}")
    all_logs = load_all_logs_with_param_optimization(run_paths, optimize_params=optimize_params, baseline_type='True', run_labels=run_labels)
    run_labels = all_logs["run_labels"]
    all_logs = all_logs["logs"]
    # metadata = all_logs["metadata"]

    for label in run_labels:
        log = all_logs[label]
        plot_icl_for_all_steps(log, run_id=label, output_dir=run_paths[0], ymin=ymin, ymax=ymax)

FONT_SIZE = 28


def plot_icl_for_all_steps(log: dict, run_id: str, output_dir: Path = None, ymin: float = None, ymax: float = None):
    """Plot ICL performance for all evaluation steps and save each step as a separate file."""
    eval_steps = log.get("eval/step", [])
    if not eval_steps:
        print("No evaluation steps found in log")
        return
    
    # Extract evaluation metrics
    eval_metrics = {}
    for key, value in log.items():
        if key.startswith("eval/") and key != "eval/step":
            task_name = key.split("/")[1]
            if task_name not in eval_metrics:
                eval_metrics[task_name] = {}
            for metric_name, metric_values in value.items():
                eval_metrics[task_name][metric_name] = metric_values
    
    # Check which baselines are available
    ridge_available = False
    true_available = False
    
    for task_name, metrics in eval_metrics.items():
        for metric_name in metrics.keys():
            if "Transformer | Ridge" in metric_name:
                ridge_available = True
            if "Transformer | True" in metric_name:
                true_available = True
    
    if not ridge_available and not true_available:
        print("No Transformer baseline metrics found in log")
        return
    

    print(f"Eval steps: {eval_steps}")
    eval_steps = [int(step) for step in eval_steps]
    # Helper function to create plots for a specific baseline
    def create_icl_plots_for_baseline(baseline_type: str, baseline_suffix: str):
        # Create output directories for ICL plots
        base_output_dir = output_dir / run_id
            
        icl_mse_dir = base_output_dir / f"icl_plots_mse_{baseline_suffix}"
        icl_rel_err_dir = base_output_dir / f"icl_plots_rel_err_{baseline_suffix}"
        icl_mse_dir.mkdir(exist_ok=True, parents=True)
        icl_rel_err_dir.mkdir(exist_ok=True, parents=True)
        
        # Colors for different tasks
        n_curves = len(eval_metrics)

        task_names = [t for t in list(eval_metrics.keys()) if "Test tasks" not in t]
        shifts = [float(t.split(" ")[-1]) for t in task_names]
        max_shift = max(abs(s) for s in shifts)
        n_tasks = len(task_names)
        min_n_tasks_for_colorbar = 5
        if len(task_names) >= min_n_tasks_for_colorbar:
            norm = mcolors.Normalize(vmin=0, vmax=max_shift)
            cmap = cm.get_cmap("viridis")
            task_to_color = {t: cmap(norm(s)) for t, s in zip(task_names, shifts)} 
        else:
            colors_cmap = cm.get_cmap('viridis', len(task_names))
            task_to_color = {t: colors_cmap(i) for i, t in enumerate(task_names)}

        # gather all methods that match this baseline (exclude RelErr and Std)
        all_methods = sorted({
            method_from_metric(mn)
            for metrics in eval_metrics.values()
            for mn in metrics.keys()
            if (f" | {baseline_type}" in mn) and ("(RelErr)" not in mn) and ("Std" not in mn) and ("True" not in method_from_metric(mn)) and ("Corrected" not in method_from_metric(mn))})

        if len(all_methods) == 1:
            all_methods.append("Ridge")

        #linestyles_cycle = ['--', '-.', ':', '-']
        linestyles_cycle = [
            (0, (1, 3)),         # dotted:  . . . . .
            (0, (5, 5)),         # dashed:  ─ ─ ─ ─
            "-",                 # solid:   ─────────
        ]
        linestyles_cycle = linestyles_cycle[-len(all_methods):]  # ensure we have enough styles 
        method_to_style = {m: linestyles_cycle[i % len(linestyles_cycle)] for i, m in enumerate(all_methods)}
        # -------------------------------------------------------
        
        # Generate plots for each evaluation step
        for step_idx, eval_step in enumerate(eval_steps):
            fig, ax = plt.subplots(figsize=(14, 8))
            if ymin is not None and ymax is not None:
                ax.set_ylim(ymin, ymax)

            for task_name, metrics in eval_metrics.items():
                for metric_name, values in metrics.items():
                    # keep your filters
                    if (((f" | {baseline_type}" in metric_name) or (metric_name == f"{baseline_type} | Ridge"))
                        and "(RelErr)" not in metric_name
                        and "Std" not in metric_name
                        and values is not None and step_idx < len(values)):

                        # optional skips that used to rely on the combined label
                        if "(True)" in metric_name or "Test tasks" in task_name:
                            continue

                        if metric_name == f"{baseline_type} | Ridge":
                            method = "Ridge"
                        else:
                            method = method_from_metric(metric_name)
                        if method not in all_methods:
                            continue
                        color  = task_to_color[task_name]
                        style  = method_to_style.get(method, '-')  # default fallback

                        # MSE by position (already normalized in your helper)
                        mse_by_position = normalize_error_values(values[step_idx])
                        std_series = metrics.get(f"{metric_name}_Std")
                        if mse_by_position.ndim == 0:
                            new_metric_name = f"{baseline_type} | {method}"
                            print(f"Careful: metric {metric_name} for task {task_name} at step {eval_step} is a scalar, trying to recover with {new_metric_name}")
                            mse_by_position = normalize_error_values(metrics[new_metric_name][step_idx])
                            std_series = metrics.get(f"{new_metric_name}_Std")
                            if mse_by_position.ndim == 0:
                                print(f"Warning: still a scalar, skipping this metric.")
                                continue
                        positions = list(range(1, len(mse_by_position) + 1))

                        # ARMA burn-in handling (apply to both mean and std arrays)
                        if "ARMA" in metric_name:
                            burn_in = 8
                            positions = positions[burn_in:]
                            mse_by_position = mse_by_position[burn_in:]

                        # plot the line without a label; we'll build legends manually
                        ax.plot(positions, mse_by_position,
                                color=color, linestyle=style, linewidth=2)

                        # fill_between with correctly sliced std (fixes the step_idx indexing gotcha)
                        if std_series is not None and step_idx < len(std_series):
                            std_by_pos = np.array(std_series[step_idx])
                            if "ARMA" in task_name:
                                std_by_pos = std_by_pos[burn_in:]
                            # avoid shape mismatches
                            if len(std_by_pos) == len(mse_by_position):
                                lower = mse_by_position / std_by_pos
                                upper = mse_by_position * std_by_pos
                                ax.fill_between(positions, lower, upper, color=color, alpha=0.2)

            ax.set_xlabel("Context Length", fontsize=FONT_SIZE)
            ax.set_ylabel(f"MSE", fontsize=FONT_SIZE)
            ax.set_title(f"ICL MSE vs Context Length", fontsize=FONT_SIZE)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_frame_on(False)
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
            ax.minorticks_off()

            # ---- two separate legends: colors (tasks) and linestyles (methods) ----
            style_handles = [Line2D([0], [0], color='black', lw=1, linestyle=method_to_style[m]) for m in all_methods]
            style_labels  = list(all_methods)

            if len(task_names) < min_n_tasks_for_colorbar:
                color_handles = [Line2D([0], [0], color=task_to_color[t], lw=3) for t in task_names]
                color_labels  = [pretty_task_label(t) for t in task_names] 


                # side-by-side, overlaid, n
                leg_tasks = ax.legend(
                    color_handles, color_labels,
                    # title="Tasks",
                    loc="upper right",
                    bbox_to_anchor=(0.95, 0.98),    # left legend (x,y) in axes coords
                    bbox_transform=ax.transAxes,
                    frameon=False,
                    fontsize=22,
                    handlelength=1.5,
                    labelspacing=0.3,
                    borderaxespad=0.0,
                )
                ax.add_artist(leg_tasks)
                leg_tasks.set_zorder(10)
                leg_methods = ax.legend(
                        style_handles, style_labels,
                        # title="Methods",
                        loc="upper right",
                        bbox_to_anchor=(0.77, 0.98),    # right legend (adjust x to taste)
                        bbox_transform=ax.transAxes,
                        fontsize=22,
                        frameon=False,
                        handlelength=1.5,
                        labelspacing=0.3,
                        borderaxespad=0.0,
                    )
                leg_methods.set_zorder(10)
            else:
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])

                cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Shift", fontsize=FONT_SIZE)
                cbar.ax.tick_params(labelsize=16)
                leg_methods = ax.legend(
                        style_handles, style_labels,
                        # title="Methods",
                        loc="upper right",
                        bbox_to_anchor=(0.95, 0.98),    # right legend (adjust x to taste)
                        bbox_transform=ax.transAxes,
                        fontsize=22,
                        frameon=False,
                        handlelength=1.5,
                        labelspacing=0.3,
                        borderaxespad=0.0,
                    )
                leg_methods.set_zorder(10)

   

                     # ----------------------------------------------------------------------

            fig.tight_layout()
            output_path = icl_mse_dir / f"icl_step_{eval_step:04d}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            output_path = icl_mse_dir / f"icl_step_{baseline_suffix}_{eval_step:04d}.pdf"
            fig.savefig(output_path, bbox_inches='tight')
            # Print ylims
            print(f"Step {eval_step}: ylims = {ax.get_ylim()}")
            plt.close(fig)
         
        
        print(f"ICL MSE plots ({baseline_type} baseline) for {len(eval_steps)} steps saved to: {icl_mse_dir}")
    
    # Create plots for available baselines
    if ridge_available and False:
        create_icl_plots_for_baseline('Ridge', 'ridge')
    
    if true_available:
        create_icl_plots_for_baseline('True', 'true')

