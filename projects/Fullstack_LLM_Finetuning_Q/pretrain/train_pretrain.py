#!/usr/bin/env python3
"""
Pretraining Script for Q Language Models

This script handles pretraining of language models on Q code with support for
frequent checkpointing and evaluation at each checkpoint.

Usage:
    python train_pretrain.py --model_name Qwen/Qwen3-32B --max_steps 100 --save_steps 5
"""

import os
import sys
import logging
import argparse
from typing import Dict, Optional, List, Union
import json
import torch
import numpy as np
from datasets import load_dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    Trainer,
    BitsAndBytesConfig,
    EvalPrediction,
    TrainerCallback
)
from trl import SFTTrainer, SFTConfig

import wandb
from peft import LoraConfig

logger = logging.getLogger(__name__)

class CheckpointEvaluationCallback(TrainerCallback):
    """Custom callback to run evaluation at each checkpoint save."""
    
    def __init__(self, eval_script_path: str, test_problems: int = 5):
        self.eval_script_path = eval_script_path
        self.test_problems = test_problems
        self.evaluation_results = []
    
    def on_save(self, args, state, control, **kwargs):
        """Run evaluation when a checkpoint is saved."""
        if state.is_world_process_zero:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(checkpoint_dir):
                logger.info(f"Running evaluation for checkpoint at step {state.global_step}")
                # We'll implement this evaluation call in the main script
                pass

class JSONLoggerCallback(TrainerCallback):
    """Custom callback to log trainer internal logs to JSON"""
    def __init__(self, output_file="training_logs.json"):
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.output_file = output_file
        self.logs = []
        with open(self.output_file, "w") as f:
            json.dump([], f)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            if logs is not None:
                log_entry = logs.copy()
                log_entry['step'] = state.global_step
                self.logs.append(log_entry)
                try:
                    with open(self.output_file, "w") as f:
                        json.dump(self.logs, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to write JSON log: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Q language model")
    
    # Dataset arguments
    parser.add_argument(
        "--train_file", 
        type=str, 
        default="q_clean_dataset/pretrain.jsonl",
        help="Path to the training dataset JSONL file"
    )
    parser.add_argument(
        "--eval_file", 
        type=str, 
        default="q_clean_dataset/pretrain_eval.jsonl",
        help="Path to the evaluation dataset JSONL file"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pretrain_output",
        help="Directory to save model checkpoints and logs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU for training"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU for evaluation"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-6,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay to apply to all layers except bias and LayerNorm"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Total number of training steps to perform"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every X steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=5,
        help="Evaluate every X steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5,
        help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of steps used for a linear warmup"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA (Low-Rank Adaptation)"
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true", 
        help="Whether to use QLoRA (4-bit quantized LoRA)"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    
    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb", "tensorboard"],
        help="Report platform to use"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str,
        default="q-pretraining",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="",
        help="WandB run name (default: auto-generated)"
    )
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    
    # Set up wandb if requested
    if args.report_to == "wandb":
        if os.environ.get("LOCAL_RANK", "0") == "0":
            logger.info("Initializing Weights & Biases logging")
            run_name = args.wandb_name or f"q-pretrain-{args.max_steps}-{args.learning_rate}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "model": args.model_name_or_path,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
                    "max_steps": args.max_steps,
                    "max_seq_length": args.max_seq_length,
                    "save_steps": args.save_steps,
                }
            )
            wandb.config.update(vars(args))
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load datasets
    logger.info(f"Loading training dataset from {args.train_file}")
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    
    logger.info(f"Loading evaluation dataset from {args.eval_file}")
    eval_dataset = load_dataset("json", data_files=args.eval_file, split="train")
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Detect data format (instruction vs pure text)
    sample_data = train_dataset[0]
    has_prompt_completion = "prompt" in sample_data and "completion" in sample_data
    has_text_only = "text" in sample_data and not has_prompt_completion
    
    if has_text_only:
        data_format = "text"
        dataset_text_field = "text"
        logger.info("Detected pure text format (pretraining data)")
    elif has_prompt_completion:
        data_format = "instruction"
        dataset_text_field = None  # Use default instruction format
        logger.info("Detected instruction-completion format")
    else:
        raise ValueError(f"Unknown data format. Expected 'text' field or 'prompt'+'completion' fields. Got: {list(sample_data.keys())}")
    
    # Log dataset samples if using wandb
    if args.report_to == "wandb" and os.environ.get("LOCAL_RANK", "0") == "0":
        if data_format == "instruction":
            wandb.Table(
                columns=["prompt", "completion"],
                data=[[example["prompt"][:200] + "...", example["completion"][:200] + "..."] 
                      for example in train_dataset.select(range(min(3, len(train_dataset))))]
            )
        else:  # text format
            wandb.Table(
                columns=["text"],
                data=[[example["text"][:400] + "..."] 
                      for example in train_dataset.select(range(min(3, len(train_dataset))))]
            )
    
    # Load the tokenizer
    logger.info(f"Loading tokenizer for {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Set up PEFT config if using LoRA
    peft_config = None
    if args.use_lora or args.use_qlora:
        logger.info(f"Setting up {'QLoRA' if args.use_qlora else 'LoRA'} configuration")
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            modules_to_save=["lm_head", "embed_tokens"],
            task_type="CAUSAL_LM",
        )
    
    # Set up model kwargs
    model_kwargs = dict(
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
    )
    
    # Add specific configurations for 14B model to handle sliding window attention
    if "14B" in args.model_name_or_path:
        model_kwargs["use_cache"] = False  # Disable cache for gradient checkpointing compatibility
        model_kwargs["attn_implementation"] = "eager"  # Use eager attention instead of sdpa
    
    # Add quantization config for QLoRA
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quantization_config
    
    # Set up training arguments  
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,  # Keep all checkpoints
        load_best_model_at_end=False,
        report_to=args.report_to,
        remove_unused_columns=True,
        push_to_hub=False,
        warmup_steps=args.warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
        ddp_find_unused_parameters=False,  # Optimize DDP performance
        dataloader_num_workers=0,  # Reduce memory usage
        logging_dir=os.path.join(args.output_dir, "logs"),
        max_seq_length=args.max_seq_length,
        model_init_kwargs=model_kwargs,
        eos_token=tokenizer.eos_token,
        dataset_text_field=dataset_text_field if dataset_text_field else "text"  # Add dataset_text_field here
    )
    
    # Initialize the SFT trainer
    if dataset_text_field:
        logger.info(f"Using dataset_text_field: {dataset_text_field}")
    else:
        logger.info("Using default instruction format (prompt/completion)")
        
    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[JSONLoggerCallback(output_file=os.path.join(args.output_dir, "training_logs.json"))]
    )
    
    # Start training
    logger.info("*** Starting pretraining ***")
    train_result = trainer.train()
    
    # Save the final model
    logger.info("*** Saving final model ***")
    trainer.save_model(args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("*** Pretraining completed ***")
    
    # Run final evaluation
    logger.info("*** Running final evaluation ***")
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)
    
    # Finish wandb run
    if args.report_to == "wandb" and os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.finish()

if __name__ == "__main__":
    main() 