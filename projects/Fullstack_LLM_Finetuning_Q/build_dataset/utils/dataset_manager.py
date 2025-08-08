#!/usr/bin/env python3
"""
Dataset Manager for Bootstrap Loop

This module provides utilities for managing datasets during the bootstrap process.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random

logger = logging.getLogger(__name__)

def copy_initial_problems(source_dir: str, dest_dir: str) -> None:
    """Copy initial problems from source to destination."""
    logger.info(f"Copying initial problems from {source_dir} to {dest_dir}")
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    if not source_path.exists():
        logger.warning(f"Source directory {source_dir} does not exist")
        return
    
    # Copy all files from source to destination
    for file_path in source_path.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(source_path)
            dest_file = dest_path / relative_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(file_path, dest_file)
    
    logger.info(f"Copied initial problems to {dest_dir}")

def get_solved_problems(solved_dir: str) -> set:
    """Get set of solved problem IDs from directory."""
    logger.info(f"Getting solved problems from {solved_dir}")
    
    solved_problems = set()
    solved_path = Path(solved_dir)
    
    if not solved_path.exists():
        logger.warning(f"Solved directory {solved_dir} does not exist")
        return solved_problems
    
    # Look for solution files
    for file_path in solved_path.glob("*_solution.q"):
        problem_id = file_path.stem.replace("_solution", "")
        solved_problems.add(problem_id)
    
    # Also look for description files
    for file_path in solved_path.glob("*_description.txt"):
        problem_id = file_path.stem.replace("_description", "")
        solved_problems.add(problem_id)
    
    logger.info(f"Found {len(solved_problems)} solved problems")
    return solved_problems

def create_datasets(problems_dirs: List[str], output_dir: str, test_ratio: float = 0.1) -> Tuple[str, str, Dict[str, Any]]:
    """Create training and test datasets from problem directories."""
    logger.info(f"Creating datasets from {len(problems_dirs)} problem directories")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_problems = []
    
    # Collect all problems from all directories
    for problem_dir in problems_dirs:
        problem_path = Path(problem_dir)
        if not problem_path.exists():
            logger.warning(f"Problem directory {problem_dir} does not exist")
            continue
        
        # Look for solution files
        for solution_file in problem_path.glob("*_solution.q"):
            problem_id = solution_file.stem.replace("_solution", "")
            
            # Try to find corresponding description
            desc_file = problem_path / f"{problem_id}_description.txt"
            if desc_file.exists():
                with open(desc_file, 'r') as f:
                    description = f.read().strip()
            else:
                description = f"Problem {problem_id}"
            
            # Read solution
            with open(solution_file, 'r') as f:
                solution = f.read().strip()
            
            all_problems.append({
                'id': problem_id,
                'description': description,
                'solution': solution
            })
    
    logger.info(f"Collected {len(all_problems)} problems")
    
    if len(all_problems) == 0:
        logger.warning("No problems found, creating empty datasets")
        train_file = output_path / "train.jsonl"
        test_file = output_path / "test.jsonl"
        
        # Create empty files
        with open(train_file, 'w') as f:
            pass
        with open(test_file, 'w') as f:
            pass
        
        stats = {
            "total_problems": 0,
            "train_pairs": 0,
            "test_pairs": 0,
            "problem_directories": problems_dirs
        }
        
        return str(train_file), str(test_file), stats
    
    # Shuffle problems
    random.shuffle(all_problems)
    
    # Split into train and test
    split_idx = int(len(all_problems) * (1 - test_ratio))
    train_problems = all_problems[:split_idx]
    test_problems = all_problems[split_idx:]
    
    # Create training dataset
    train_file = output_path / "train.jsonl"
    with open(train_file, 'w') as f:
        for problem in train_problems:
            # Create instruction-following format
            instruction = f"Solve this problem: {problem['description']}"
            response = problem['solution']
            
            example = {
                "instruction": instruction,
                "response": response
            }
            f.write(json.dumps(example) + '\n')
    
    # Create test dataset
    test_file = output_path / "test.jsonl"
    with open(test_file, 'w') as f:
        for problem in test_problems:
            # Create instruction-following format
            instruction = f"Solve this problem: {problem['description']}"
            response = problem['solution']
            
            example = {
                "instruction": instruction,
                "response": response
            }
            f.write(json.dumps(example) + '\n')
    
    stats = {
        "total_problems": len(all_problems),
        "train_pairs": len(train_problems),
        "test_pairs": len(test_problems),
        "problem_directories": problems_dirs
    }
    
    logger.info(f"Created datasets: {len(train_problems)} train, {len(test_problems)} test")
    return str(train_file), str(test_file), stats 