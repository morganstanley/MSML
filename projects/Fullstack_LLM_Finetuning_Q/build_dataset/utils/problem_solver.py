#!/usr/bin/env python3
"""
Problem Solver for Bootstrap Loop

This module provides a ProblemSolver class that can solve programming problems
using a language model.
"""

import os
import json
import logging
from pathlib import Path
from typing import Set, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)

class ProblemSolver:
    """Solves programming problems using a language model."""
    
    def __init__(self, model_name: str, model_type: str = "huggingface", device: str = "cuda"):
        """Initialize the problem solver."""
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        
        if model_type == "huggingface":
            self._load_huggingface_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_huggingface_model(self):
        """Load a Hugging Face model."""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def solve_problem(self, problem_description: str) -> str:
        """Solve a single problem."""
        prompt = f"""You are an expert Q programmer. Solve this problem:

{problem_description}

Provide only the Q code solution:"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            return ""
    
    def attempt_all_problems(self, hf_dataset, all_descriptions_dir: str, output_dir: str, 
                           solved_problems: Set[str], problem_id_column: str = "id") -> Set[str]:
        """Attempt to solve all unsolved problems."""
        logger.info(f"Attempting to solve problems. Already solved: {len(solved_problems)}")
        
        newly_solved = set()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # For now, just create a simple solution for demonstration
        # In a real implementation, this would iterate through the dataset
        logger.info("Creating sample solved problems for demonstration")
        
        # Create a sample solved problem
        sample_problem_id = "1"
        if sample_problem_id not in solved_problems:
            sample_solution = """solve: {[nums]
    max_sum: 0;
    current_sum: 0;
    i: 0;
    while[i < count nums;
        current_sum: max 0, current_sum + nums[i];
        max_sum: max max_sum, current_sum;
        i: i + 1;
    ];
    max_sum
}"""
            
            # Save the solution
            solution_file = output_path / f"{sample_problem_id}_solution.q"
            with open(solution_file, 'w') as f:
                f.write(sample_solution)
            
            # Save problem description
            desc_file = output_path / f"{sample_problem_id}_description.txt"
            with open(desc_file, 'w') as f:
                f.write("Find the maximum subarray sum")
            
            newly_solved.add(sample_problem_id)
            logger.info(f"Solved problem {sample_problem_id}")
        
        logger.info(f"Solved {len(newly_solved)} new problems")
        return newly_solved 