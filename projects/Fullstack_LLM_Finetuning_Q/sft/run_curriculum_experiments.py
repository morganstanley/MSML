#!/usr/bin/env python3
"""
Curriculum Learning SFT Experiments

This script runs comprehensive SFT experiments with:
1. Different learning rates (high, low)
2. Different curriculum strategies (difficulty, task_type, mixed, progressive)
3. Regular training (baseline)

Usage:
    python run_curriculum_experiments.py --base_model Qwen/Qwen2.5-7B-Instruct --output_dir sft_experiments/
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Comprehensive SFT Curriculum Experiments")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model name or path")
    
    # Experiment arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory for all experiments")
    parser.add_argument("--max_steps", type=int, default=300, help="Total training steps for regular training")
    parser.add_argument("--steps_per_phase", type=int, default=100, help="Training steps per curriculum phase")
    
    # Data arguments
    parser.add_argument("--sft_data_dir", type=str, default="../SFT_Data", help="SFT data directory")
    parser.add_argument("--curriculum_base_dir", type=str, default="curriculum_data", help="Base directory for curriculum data")
    
    # Experiment selection
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline (regular) training experiments")
    parser.add_argument("--skip_curriculum", action="store_true", help="Skip curriculum experiments")
    parser.add_argument("--only_difficulty", action="store_true", help="Only run difficulty-based curriculum")
    
    # Misc arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device for training")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    return parser.parse_args()

def create_curriculum_data(sft_data_dir: str, curriculum_base_dir: str):
    """Create curriculum data using the curriculum organizer."""
    
    logger.info("*** Creating Curriculum Data ***")
    
    curriculum_strategies = ["difficulty", "task_type", "mixed", "progressive"]
    
    for strategy in curriculum_strategies:
        output_dir = os.path.join(curriculum_base_dir, strategy)
        
        if os.path.exists(output_dir):
            logger.info(f"Curriculum data for {strategy} already exists, skipping...")
            continue
        
        logger.info(f"Creating {strategy} curriculum...")
        
        cmd = [
            sys.executable, "curriculum_organizer.py",
            "--data_dir", sft_data_dir,
            "--strategy", strategy,
            "--output_dir", output_dir
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Created {strategy} curriculum successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create {strategy} curriculum: {e}")
            logger.error(f"Error: {e.stderr}")
            raise

def run_baseline_experiments(args) -> List[Dict]:
    """Run baseline (regular) SFT training with different learning rates."""
    
    logger.info("*** Running Baseline SFT Experiments ***")
    
    baseline_experiments = [
        {"name": "baseline_low_lr", "learning_rate": 1e-5, "description": "Baseline with low learning rate"},
        {"name": "baseline_high_lr", "learning_rate": 5e-5, "description": "Baseline with high learning rate"},
        {"name": "baseline_medium_lr", "learning_rate": 2e-5, "description": "Baseline with medium learning rate"},
    ]
    
    results = []
    
    for exp in baseline_experiments:
        logger.info(f"Running {exp['name']}: {exp['description']}")
        
        output_dir = os.path.join(args.output_dir, exp['name'])
        
        cmd = [
            sys.executable, "run_sft.py",
            "--base_model", args.base_model,
            "--output_dir", output_dir,
            "--max_steps", str(args.max_steps),
            "--learning_rate", str(exp['learning_rate']),
            "--experiment_name", exp['name']
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                check=True,
                capture_output=True,
                text=True
            )
            
            exp_result = {
                'experiment_name': exp['name'],
                'experiment_type': 'baseline',
                'learning_rate': exp['learning_rate'],
                'max_steps': args.max_steps,
                'output_dir': output_dir,
                'status': 'completed',
                'description': exp['description']
            }
            
            logger.info(f"✓ {exp['name']} completed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {exp['name']} failed: {e}")
            exp_result = {
                'experiment_name': exp['name'],
                'experiment_type': 'baseline',
                'status': 'failed',
                'error': str(e)
            }
        
        results.append(exp_result)
    
    return results

def run_curriculum_experiments(args) -> List[Dict]:
    """Run curriculum learning experiments."""
    
    logger.info("*** Running Curriculum Learning Experiments ***")
    
    if args.only_difficulty:
        curriculum_strategies = ["difficulty"]
    else:
        curriculum_strategies = ["difficulty", "task_type", "mixed", "progressive"]
    
    learning_rates = [
        {"name": "low_lr", "value": 1e-5},
        {"name": "high_lr", "value": 5e-5}
    ]
    
    experiments = []
    for strategy in curriculum_strategies:
        for lr_config in learning_rates:
            experiments.append({
                "name": f"curriculum_{strategy}_{lr_config['name']}",
                "strategy": strategy,
                "learning_rate": lr_config['value'],
                "description": f"Curriculum {strategy} with {lr_config['name']} ({lr_config['value']})"
            })
    
    results = []
    
    for exp in experiments:
        logger.info(f"Running {exp['name']}: {exp['description']}")
        
        curriculum_dir = os.path.join(args.curriculum_base_dir, exp['strategy'])
        output_dir = os.path.join(args.output_dir, exp['name'])
        
        if not os.path.exists(curriculum_dir):
            logger.error(f"Curriculum directory not found: {curriculum_dir}")
            results.append({
                'experiment_name': exp['name'],
                'experiment_type': 'curriculum',
                'status': 'failed',
                'error': f"Curriculum directory not found: {curriculum_dir}"
            })
            continue
        
        cmd = [
            sys.executable, "train_sft_curriculum.py",
            "--base_model", args.base_model,
            "--curriculum_dir", curriculum_dir,
            "--output_dir", output_dir,
            "--learning_rate", str(exp['learning_rate']),
            "--steps_per_phase", str(args.steps_per_phase),
            "--experiment_name", exp['name']
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                check=True,
                capture_output=True,
                text=True
            )
            
            exp_result = {
                'experiment_name': exp['name'],
                'experiment_type': 'curriculum',
                'curriculum_strategy': exp['strategy'],
                'learning_rate': exp['learning_rate'],
                'steps_per_phase': args.steps_per_phase,
                'curriculum_dir': curriculum_dir,
                'output_dir': output_dir,
                'status': 'completed',
                'description': exp['description']
            }
            
            logger.info(f"✓ {exp['name']} completed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {exp['name']} failed: {e}")
            exp_result = {
                'experiment_name': exp['name'],
                'experiment_type': 'curriculum',
                'curriculum_strategy': exp['strategy'],
                'status': 'failed',
                'error': str(e)
            }
        
        results.append(exp_result)
    
    return results

def create_experiment_summary(all_results: List[Dict], output_dir: str):
    """Create a comprehensive experiment summary."""
    
    logger.info("*** Creating Experiment Summary ***")
    
    # Organize results by type
    baseline_results = [r for r in all_results if r.get('experiment_type') == 'baseline']
    curriculum_results = [r for r in all_results if r.get('experiment_type') == 'curriculum']
    
    # Count successes and failures
    total_experiments = len(all_results)
    successful_experiments = len([r for r in all_results if r.get('status') == 'completed'])
    failed_experiments = total_experiments - successful_experiments
    
    summary = {
        'experiment_overview': {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'failed_experiments': failed_experiments,
            'success_rate': successful_experiments / total_experiments if total_experiments > 0 else 0,
            'timestamp': datetime.now().isoformat()
        },
        'baseline_experiments': {
            'total': len(baseline_results),
            'successful': len([r for r in baseline_results if r.get('status') == 'completed']),
            'results': baseline_results
        },
        'curriculum_experiments': {
            'total': len(curriculum_results),
            'successful': len([r for r in curriculum_results if r.get('status') == 'completed']),
            'results': curriculum_results
        },
        'all_results': all_results
    }
    
    # Save summary
    summary_file = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary to console
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total Experiments: {total_experiments}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {failed_experiments}")
    print(f"Success Rate: {successful_experiments/total_experiments:.1%}")
    
    if baseline_results:
        print(f"\nBaseline Experiments: {len(baseline_results)}")
        for result in baseline_results:
            status = "✓" if result.get('status') == 'completed' else "✗"
            print(f"  {status} {result['experiment_name']} (LR: {result.get('learning_rate', 'N/A')})")
    
    if curriculum_results:
        print(f"\nCurriculum Experiments: {len(curriculum_results)}")
        for result in curriculum_results:
            status = "✓" if result.get('status') == 'completed' else "✗"
            strategy = result.get('curriculum_strategy', 'N/A')
            lr = result.get('learning_rate', 'N/A')
            print(f"  {status} {result['experiment_name']} ({strategy}, LR: {lr})")
    
    print(f"\nResults saved to: {summary_file}")
    print("="*80)
    
    logger.info(f"Experiment summary saved to: {summary_file}")

def main():
    """Main experiment orchestrator"""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("*** Starting Comprehensive SFT Curriculum Experiments ***")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    try:
        # Create curriculum data
        if not args.skip_curriculum:
            create_curriculum_data(args.sft_data_dir, args.curriculum_base_dir)
        
        # Run baseline experiments
        if not args.skip_baseline:
            baseline_results = run_baseline_experiments(args)
            all_results.extend(baseline_results)
        
        # Run curriculum experiments
        if not args.skip_curriculum:
            curriculum_results = run_curriculum_experiments(args)
            all_results.extend(curriculum_results)
        
        # Create summary
        create_experiment_summary(all_results, args.output_dir)
        
        logger.info("*** All Experiments Completed ***")
        
    except Exception as e:
        logger.error(f"Experiment pipeline failed: {e}")
        # Still try to save partial results
        if all_results:
            create_experiment_summary(all_results, args.output_dir)
        raise

if __name__ == "__main__":
    main() 