#!/usr/bin/env python3
"""
Simple Pretraining Script

This script runs pretraining with different data types and training methods:
1. Supports multiple data types (raw, filtered, described_filtered)
2. Supports multiple training methods (LoRA, QLoRA, full fine-tuning)
3. Trains the model with checkpoints

Usage:
    python run_pretraining.py --model_name Qwen/Qwen2.5-7B-Instruct --data_type raw --training_method lora --max_steps 500
"""

import os
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run pretraining with different data types and methods")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model to pretrain")
    
    # Data arguments
    parser.add_argument("--data_type", type=str, default="raw", 
                       choices=["raw", "filtered", "described_filtered", "licensed"],
                       help="Type of dataset to use (raw, filtered, described_filtered, or licensed)")
    
    # Training method arguments
    parser.add_argument("--training_method", type=str, default="lora", 
                       choices=["lora", "qlora", "full"],
                       help="Training method: lora, qlora, or full fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Training arguments
    parser.add_argument("--max_steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="pretraining_results", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name (default: auto-generated)")
    
    # Misc arguments
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()

def get_data_files(data_type):
    """Get the train and eval data files based on data type."""
    data_files = {
        "raw": ("raw_train.jsonl", "raw_test.jsonl"),
        "filtered": ("filtered_train.jsonl", "filtered_test.jsonl"),
        "described_filtered": ("described_filtered_train.jsonl", "described_filtered_test.jsonl"),
        "licensed": ("kdb_license_processed/train_license_kdbsite.jsonl", "kdb_license_processed/eval_license_kdbsite.jsonl")
    }
    
    train_file, eval_file = data_files[data_type]
    
    # Check if files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Eval file not found: {eval_file}")
    
    return train_file, eval_file

def run_training(args, train_output_dir):
    """Run the pretraining script."""
    logger.info("Starting pretraining...")
    
    # Get data files based on data type
    try:
        train_file, eval_file = get_data_files(args.data_type)
        logger.info(f"Using {args.data_type} data: train={train_file}, eval={eval_file}")
    except FileNotFoundError as e:
        logger.error(f"Data file error: {e}")
        return False
    
    # Construct training command
    train_cmd = [
        "python", "train_pretrain.py",
        "--model_name_or_path", args.model_name,
        "--output_dir", train_output_dir,
        "--train_file", train_file,
        "--eval_file", eval_file,
        "--max_steps", str(args.max_steps),
        "--save_steps", str(args.save_steps),
        "--learning_rate", str(args.learning_rate),
        "--per_device_train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--warmup_steps", str(max(1, args.max_steps // 10)),  # 10% warmup
    ]
    
    # Add training method arguments
    if args.training_method == "lora":
        train_cmd.extend([
            "--use_lora",
            "--lora_rank", str(args.lora_rank),
            "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", str(args.lora_dropout)
        ])
    elif args.training_method == "qlora":
        train_cmd.extend([
            "--use_qlora",
            "--lora_rank", str(args.lora_rank),
            "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", str(args.lora_dropout)
        ])
    # For full fine-tuning, no additional flags needed
    
    if args.wandb:
        train_cmd.extend(["--report_to", "wandb"])
        train_cmd.extend(["--wandb_project", "q-pretraining"])
        if args.experiment_name:
            train_cmd.extend(["--wandb_name", args.experiment_name])
    
    logger.info(f"Training command: {' '.join(train_cmd)}")
    
    try:
        result = subprocess.run(
            train_cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        logger.info("Training completed successfully")
        if args.verbose:
            logger.info(f"Training output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False



def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate experiment name if not provided
    if not args.experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = args.model_name.replace("/", "_")
        args.experiment_name = f"pretrain_{model_name_safe}_{args.data_type}_{args.training_method}_{args.max_steps}steps_{timestamp}"
    
    logger.info("Starting Q pretraining")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data type: {args.data_type}")
    logger.info(f"Training method: {args.training_method}")
    if args.training_method in ["lora", "qlora"]:
        logger.info(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Save every: {args.save_steps} steps")
    
    # Set up output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_output_dir = str(output_dir)
    
    # Run training
    if not run_training(args, train_output_dir):
        logger.error("Training failed. Exiting.")
        return 1
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {train_output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 


"""
Example usage: 
python run_pretraining.py --data_type raw --training_method full --max_steps 500 --wandb
python run_pretraining.py --data_type filtered --training_method lora --max_steps 1000 
python run_pretraining.py --data_type described_filtered --training_method qlora --learning_rate 5e-5
python run_pretraining.py --data_type licensed --training_method lora --max_steps 1000 --wandb
"""