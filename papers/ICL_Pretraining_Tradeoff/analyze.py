#!/usr/bin/env python3
"""
Analysis script for training logs.

Usage:
    python analyze.py [run_id]
    python analyze.py 2025-08-06_12-24-25
"""
import argparse
from pathlib import Path

from loading import get_most_recent_run, get_most_recent_multirun, load_log_with_safetensors
from task_shift import plot_task_shift_analysis
from mean_min_best_mse import plot_min_mse_analysis
from weight_analysis import plot_weights_analysis, plot_weights_analysis_multirun
from icl_plots import plot_icl_for_all_steps
from training_analysis import plot_training_loss, fit_mse_curves_and_compute_metrics, print_summary
from hyperparam_analysis import hyperparam_analysis
from opt_icl_plots import plot_opt_icl_plots


def analyze_multirun(multirun_id: str, custom_names: list = None):
    """Analyze a single multirun experiment with individual run analysis."""
    multirun_path = Path("outputs/multirun") / multirun_id
    
    if not multirun_path.exists():
        print(f"Multirun directory not found: {multirun_path}")
        return
    
    print(f"Analyzing multirun: {multirun_id}")
    print(f"Path: {multirun_path}")
    
    # Find valid run subdirs
    from task_shift import find_valid_multirun_subdirs
    run_subdirs = find_valid_multirun_subdirs(multirun_path)
    
    if not run_subdirs:
        print("No valid runs found in multirun")
        return
    
    print(f"Found {len(run_subdirs)} valid runs")
    
    # Create run output directory for individual analyses
    multirun_output_dir = multirun_path
    individual_runs_dir = multirun_output_dir / "individual_run_analyses"
    individual_runs_dir.mkdir(exist_ok=True)
    
    # Extract swept parameters and create display names
    from task_shift import extract_swept_params, create_run_display_names
    swept_params = extract_swept_params(multirun_path)
    print(f"Swept parameters: {swept_params}")
    
    # Create display names for runs
    param_names = create_run_display_names(multirun_path, run_subdirs)
    
    # Analyze each run individually  
    for i, subdir in enumerate(run_subdirs):
        subdir_path = multirun_path / subdir
        
        # Create display name for this run
        if custom_names and i < len(custom_names):
            run_display_id = custom_names[i].strip()
        elif param_names and int(subdir) in param_names:
            run_display_id = param_names[int(subdir)]
        else:
            run_display_id = f"{multirun_id}-{subdir}"
        
        print(f"\n--- Analyzing run {subdir}: {run_display_id} ---")
        
        # Load log for this run
        try:
            log = load_log_with_safetensors(subdir_path)
        except Exception as e:
            print(f"Failed to load run {subdir}: {e}")
            continue
        
        # Create output directory for this run's plots
        run_output_dir = individual_runs_dir / f"run_{subdir}"
        run_output_dir.mkdir(exist_ok=True)
        
        # Perform individual analysis
        print_summary(log, run_display_id)
        fit_mse_curves_and_compute_metrics(log, run_display_id)
        
        # Generate plots  
        plot_training_loss(log, run_display_id, run_output_dir)


def parse_multirun_args(multirun_arg, run_id_arg):
    """Parse multirun arguments to extract custom names and multirun ID."""
    if multirun_arg is True:
        # No custom names, just use most recent multirun
        return run_id_arg, None
    elif multirun_arg and "," in multirun_arg:
        # Custom names provided, run_id should be the multirun_id
        custom_names = [name.strip() for name in multirun_arg.split(',')]
        return run_id_arg, custom_names
    else:
        return run_id_arg, None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training logs and plot metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py                    # Analyze most recent run
  python analyze.py 2025-08-06_12-24-25   # Analyze specific run
  python analyze.py --multirun         # Analyze most recent multirun
  python analyze.py --multirun 2025-08-11_11-45-46   # Analyze specific multirun
  python analyze.py --multirun "GPT-2,Transformer,LSTM" 2025-08-11_11-45-46   # Custom names
  python analyze.py --multirun --shift-analysis 2025-08-11_11-45-46   # Task shift analysis
  python analyze.py --multirun --shift-analysis "task.n_tasks,train.clip_max_norm" 2025-08-11_11-45-46   # With parameter optimization
  python analyze.py --multirun --hyperparam-analysis 2025-08-11_11-45-46   # Hyperparameter analysis
        """
    )
    parser.add_argument(
        'run_id', 
        nargs='?', 
        help='Run ID to analyze (e.g., 2025-08-06_12-24-25). If not provided, uses most recent run.'
    )
    parser.add_argument(
        '--multirun',
        nargs='?',
        const=True,
        help='Analyze a multirun experiment instead of a single run. Optionally provide comma-separated names for runs (e.g., "name0,name1,name2")'
    )
    parser.add_argument(
        '--shift-analysis',
        nargs='?',
        const=True,
        help='Perform task shift analysis (alpha and C vs task centers). Optionally specify comma-separated parameters to optimize (e.g., "task.n_tasks,train.clip_max_norm")'
    )
    parser.add_argument(
        '--hyperparam-analysis',
        action='store_true',
        help='Perform hyperparameter analysis with heatmaps of MSE vs hyperparameter pairs for each distrib_param value'
    )
    parser.add_argument(
        '--weights',
        action='store_true',
        help='Analyze final weights evolution: ESS, Sum, and KL divergence from uniform vs training steps'
    )
    parser.add_argument(
        '--min-mse-analysis',
        nargs='?',
        const=True,
        help='Perform minimum MSE analysis across multiple runs. Optionally specify comma-separated parameters to optimize (e.g., "task.n_tasks,train.clip_max_norm")'
    )
    parser.add_argument(
        '--opt-icl-plots',
        nargs='?',
        const=True,
        help='Perform ICL plots analysis with parameter optimization. Optionally specify comma-separated parameters to optimize (e.g., "task.n_tasks,train.clip_max_norm")'
    )
    parser.add_argument(
        '--icl-plots',
        action='store_true',
        help='Generate ICL plots for all evaluation steps (MSE and RelErr vs context length)'
    )

    parser.add_argument(
        '--ymin',
        nargs='?',
        type=float,
        default=None,
        help='Set minimum y-axis limit for loss plots',
        )
    parser.add_argument(
        '--ymax',
        nargs='?',
        type=float,
        default=None,
        help='Set maximum y-axis limit for loss plots',
        )

    args = parser.parse_args()
    
    # Handle shift analysis mode
    if args.shift_analysis is not None:
        if args.shift_analysis is True:
            optimize_params = None  # No parameter optimization
        else:
            optimize_params = [param.strip() for param in args.shift_analysis.split(',')]
        
        if args.multirun is not None:
            multirun_id, custom_names = parse_multirun_args(args.multirun, args.run_id)
            if not multirun_id:
                try:
                    multirun_id = get_most_recent_multirun()
                    print(f"Using most recent multirun: {multirun_id}")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    return
            
            # Prepare run paths for task shift analysis
            multirun_path = Path("outputs/multirun") / multirun_id
            if multirun_path.exists():
                plot_task_shift_analysis([multirun_path], run_labels=custom_names, optimize_params=optimize_params)
            else:
                print(f"Multirun directory not found: {multirun_path}")
        else:
            # Multiple single runs for task shift analysis
            if args.run_id:
                run_ids = [args.run_id]
                run_paths = [Path("outputs") / rid for rid in run_ids]
            else:
                try:
                    multirun_id = get_most_recent_multirun()
                    print(f"Using most recent multirun: {multirun_id}")
                    run_paths = [Path("outputs/multirun") / multirun_id]
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    return
            
            plot_task_shift_analysis(run_paths, optimize_params=optimize_params)
        return
    
    # Handle min MSE analysis mode  
    if args.min_mse_analysis is not None:
        if args.min_mse_analysis is True:
            optimize_params = None  # No parameter optimization
        else:
            optimize_params = [param.strip() for param in args.min_mse_analysis.split(',')]
        
        if args.multirun is not None:
            multirun_id, custom_names = parse_multirun_args(args.multirun, args.run_id)
            if not multirun_id:
                try:
                    run_ids = [get_most_recent_run()]
                    print(f"Using most recent run: {run_ids[0]}")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    return
            
            # Prepare run paths for min MSE analysis
            if multirun_id:
                multirun_path = Path("outputs/multirun") / multirun_id
                if multirun_path.exists():
                    plot_min_mse_analysis([multirun_path], run_labels=custom_names, optimize_params=optimize_params)
                else:
                    print(f"Multirun directory not found: {multirun_path}")
            else:
                run_paths = [Path("outputs") / rid for rid in run_ids]
                plot_min_mse_analysis(run_paths, optimize_params=optimize_params)
        else:
            print("Min MSE analysis requires --multirun option")
        return

    if args.opt_icl_plots is not None:
        if args.opt_icl_plots is True:
            optimize_params = None  # No parameter optimization
        else:
            optimize_params = [param.strip() for param in args.opt_icl_plots.split(',')]
        
        if args.multirun is not None:
            multirun_id, custom_names = parse_multirun_args(args.multirun, args.run_id)
            if not multirun_id:
                try:
                    run_ids = [get_most_recent_run()]
                    print(f"Using most recent run: {run_ids[0]}")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    return
            
            # Prepare run paths for min MSE analysis
            if multirun_id:
                multirun_path = Path("outputs/multirun") / multirun_id
                if multirun_path.exists():
                    plot_opt_icl_plots([multirun_path], run_labels=custom_names, optimize_params=optimize_params, ymin=args.ymin, ymax=args.ymax)
                else:
                    print(f"Multirun directory not found: {multirun_path}")
            else:
                run_paths = [Path("outputs") / rid for rid in run_ids]
                plot_opt_icl_plots(run_paths, optimize_params=optimize_params)
        else:
            print("Opt ICL analysis requires --multirun option")
        return

    # Handle weights analysis mode
    if args.weights:
        if args.multirun is not None:
            multirun_id, custom_names = parse_multirun_args(args.multirun, args.run_id)
            if not multirun_id:
                try:
                    multirun_id = get_most_recent_multirun()
                    print(f"Using most recent multirun: {multirun_id}")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    return
            
            multirun_path = Path("outputs/multirun") / multirun_id
            if multirun_path.exists():
                plot_weights_analysis_multirun([multirun_path], run_labels=custom_names)
            else:
                print(f"Multirun directory not found: {multirun_path}")
        else:
            # Single run weights analysis
            if args.run_id:
                run_id = args.run_id
            else:
                try:
                    run_id = get_most_recent_run()
                    print(f"Using most recent run: {run_id}")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    return
            
            try:
                run_path = Path("outputs") / run_id
                log = load_log_with_safetensors(run_path)
                plot_weights_analysis(log, run_id)
            except Exception as e:
                print(f"Error loading run: {e}")
        return
        
    # Handle hyperparameter analysis mode
    if args.hyperparam_analysis:
        if args.multirun is not None:
            multirun_id, _ = parse_multirun_args(args.multirun, args.run_id)
            if not multirun_id:
                try:
                    multirun_id = get_most_recent_multirun()
                    print(f"Using most recent multirun: {multirun_id}")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    return
            
            multirun_path = Path("outputs/multirun") / multirun_id
            hyperparam_analysis(multirun_path)
        else:
            print("Hyperparameter analysis requires a multirun. Use --multirun option.")
        return
    
    # Handle ICL plots mode
    if args.icl_plots:
        if args.multirun is not None:
            # ICL plots for multirun - need to specify which subrun
            multirun_id, custom_names = parse_multirun_args(args.multirun, args.run_id)
            if not multirun_id:
                try:
                    multirun_id = get_most_recent_multirun()
                    print(f"Using most recent multirun: {multirun_id}")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    return
            
            # For multirun ICL plots, we need to ask which subrun to plot
            multirun_path = Path("outputs/multirun") / multirun_id
            if not multirun_path.exists():
                print(f"Multirun directory not found: {multirun_path}")
                return
            
            # Find valid run subdirs
            from task_shift import find_valid_multirun_subdirs
            run_subdirs = find_valid_multirun_subdirs(multirun_path)
            
            if not run_subdirs:
                print("No valid runs found in multirun")
                return
            
            print(f"Found {len(run_subdirs)} runs in multirun {multirun_id}")
            print(f"Generating ICL plots for all {len(run_subdirs)} runs...")
            
            # Generate ICL plots for all runs in the multirun
            for i, subdir in enumerate(run_subdirs):
                run_path = multirun_path / subdir
                run_display_id = f"{multirun_id}-{subdir}"
                
                print(f"  Processing run {i+1}/{len(run_subdirs)}: {run_display_id}")
                
                try:
                    log = load_log_with_safetensors(run_path)
                    # Pass the correct output directory for multirun subruns
                    plot_icl_for_all_steps(log, run_display_id, output_dir=run_path)
                    print(f"    ✓ ICL plots saved for run {subdir}")
                except Exception as e:
                    print(f"    ✗ Error loading run {subdir}: {e}")
                    continue
            
            print(f"\nCompleted ICL plots generation for {len(run_subdirs)} runs in multirun {multirun_id}")
        else:
            # Single run ICL plots
            if args.run_id:
                run_id = args.run_id
            else:
                try:
                    run_id = get_most_recent_run()
                    print(f"Using most recent run: {run_id}")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                    return
            
            try:
                run_path = Path("outputs") / run_id
                log = load_log_with_safetensors(run_path)
                plot_icl_for_all_steps(log, run_id)
            except Exception as e:
                print(f"Error loading run for ICL plots: {e}")
        return
    
    # Handle regular analysis (single run or multirun individual analysis)
    if args.multirun is not None:
        multirun_id, custom_names = parse_multirun_args(args.multirun, args.run_id)
        if not multirun_id:
            try:
                multirun_id = get_most_recent_multirun()
                print(f"Using most recent multirun: {multirun_id}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
        
        analyze_multirun(multirun_id, custom_names)
    else:
        # Single run analysis
        if args.run_id:
            run_id = args.run_id
        else:
            try:
                run_id = get_most_recent_run()
                print(f"Using most recent run: {run_id}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
        
        # Use safetensor loading for single runs too
        run_path = Path("outputs") / run_id
        log = load_log_with_safetensors(run_path)
        print_summary(log, run_id)
        fit_mse_curves_and_compute_metrics(log, run_id)
        plot_training_loss(log, run_id)


if __name__ == "__main__":
    main()
