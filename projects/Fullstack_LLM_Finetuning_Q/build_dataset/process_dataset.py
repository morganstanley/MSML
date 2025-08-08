#!/usr/bin/env python3
"""
Script to process the LeetCode dataset and organize it into subdirectories.
Creates initial_dataset/ with subdirectories for each problem.
"""

import os
import json
from datasets import load_dataset
from pathlib import Path

def process_dataset():
    """Process the entire LeetCode dataset and organize into subdirectories."""
    print("Loading dataset: newfacade/LeetCodeDataset")
    
    # Load the dataset
    dataset = load_dataset("newfacade/LeetCodeDataset")
    
    # Use the train split (main split)
    train_data = dataset['train']
    
    print(f"Processing {len(train_data)} entries...")
    
    # Create the initial_dataset directory
    initial_dataset_dir = Path("initial_dataset")
    initial_dataset_dir.mkdir(exist_ok=True)
    
    for i, entry in enumerate(train_data):
        # Create subdirectory name using question_id and task_id
        question_id = entry['question_id']
        task_id = entry['task_id']
        subdir_name = f"{question_id}_{task_id}"
        
        # Create the subdirectory
        subdir_path = initial_dataset_dir / subdir_name
        subdir_path.mkdir(exist_ok=True)
        
        # Save the complete entry as JSON (convert non-serializable objects to strings)
        json_path = subdir_path / "entry.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(entry, f, indent=2, ensure_ascii=False, default=str)
        
        # Save completion as sol.py
        sol_path = subdir_path / "sol.py"
        with open(sol_path, 'w', encoding='utf-8') as f:
            f.write(entry['completion'])
        
        # Save query as problem_description.txt
        problem_desc_path = subdir_path / "problem_description.txt"
        with open(problem_desc_path, 'w', encoding='utf-8') as f:
            f.write(entry['query'])
        
        # Extract first three assert lines from test and save as test_cases.txt
        test_content = entry['test']
        assert_lines = []
        
        # Split test content into lines and find assert statements
        for line in test_content.split('\n'):
            stripped_line = line.strip()
            if stripped_line.startswith('assert '):
                assert_lines.append(stripped_line)
                if len(assert_lines) == 5:  # Only take first 5
                    break
        
        # Save the first five assert lines
        test_cases_path = subdir_path / "test_cases.txt"
        with open(test_cases_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(assert_lines))
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(train_data)} entries...")
    
    print(f"\nCompleted processing {len(train_data)} entries!")
    print(f"All data saved to: {initial_dataset_dir.absolute()}")

if __name__ == "__main__":
    process_dataset() 