#!/usr/bin/env python3

import os
import json
from datasets import load_dataset
from pathlib import Path
import shutil

def ensure_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_jsonl(data, filepath):
    """Save data as JSONL file."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def save_file(content, filepath):
    """Save content to file, creating directories if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)

def rebuild_from_structured(structured_dataset, base_dir):
    """Rebuild problem directories from structured dataset."""
    
    # Process train and test splits
    for split_name in ['train', 'test']:
        if split_name not in structured_dataset:
            continue
            
        split_data = structured_dataset[split_name]
        
        for problem in split_data:
            # Create problem directory
            problem_dir = os.path.join(base_dir, split_name, problem['problem_name'])
            ensure_directory(problem_dir)
            
            # Save entry.json
            entry_data = {
                "task_id": problem['problem_id'],
                "question_id": problem['leetcode_id'],
                "difficulty": problem['difficulty'],
                "tags": problem['tags'],
                "starter_code": problem['metadata']['starter_code'],
                "completion": problem['metadata']['completion'],
                "entry_point": problem['metadata']['entry_point'],
                "estimated_date": problem['metadata']['estimated_date'],
                "prompt": problem['metadata']['prompt'],
                "test": problem['metadata']['test'],
                "input_output": problem['metadata']['input_output']
            }
            save_file(json.dumps(entry_data, indent=2), os.path.join(problem_dir, "entry.json"))
            
            # Save problem description
            if problem['problem_description']:
                save_file(problem['problem_description'], os.path.join(problem_dir, "problem_description.txt"))
            
            # Save solutions
            if problem['python_solution']:
                save_file(problem['python_solution'], os.path.join(problem_dir, "python_sol.py"))
            if problem['q_solution']:
                save_file(problem['q_solution'], os.path.join(problem_dir, "q_sol.q"))
            
            # Save test cases
            for test_case in problem['test_cases']:
                test_num = test_case['test_case_number']
                
                if test_case['python_test_code']:
                    save_file(test_case['python_test_code'], 
                            os.path.join(problem_dir, f"test_case_{test_num}.py"))
                
                if test_case['q_test_code']:
                    save_file(test_case['q_test_code'],
                            os.path.join(problem_dir, f"q_test_case_{test_num}.q"))
                
                if test_case['python_expected_output']:
                    save_file(test_case['python_expected_output'],
                            os.path.join(problem_dir, f"python_exact_ans_test_case_{test_num}.txt"))
                
                if test_case['q_expected_output']:
                    save_file(test_case['q_expected_output'],
                            os.path.join(problem_dir, f"q_exact_ans_test_case_{test_num}.txt"))
                
                if test_case['correct_answer']:
                    save_file(test_case['correct_answer'],
                            os.path.join(problem_dir, f"test_case_{test_num}_correct_ans.txt"))

def rebuild_sft_data(base_repo_name="bhogan/sft-python-q-problems", output_dir="SFT_Data"):
    """Download datasets from Hugging Face Hub and rebuild SFT_Data directory."""
    
    print(f"🚀 Downloading datasets from Hugging Face Hub...")
    
    # Clear and recreate output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    ensure_directory(output_dir)
    
    # Download structured dataset
    print(f"📥 Downloading structured dataset...")
    structured_dataset = load_dataset(base_repo_name)
    
    # Download SFT dataset
    print(f"📥 Downloading SFT dataset...")
    sft_dataset = load_dataset(f"{base_repo_name}-sft")
    
    # Download metadata dataset
    print(f"📥 Downloading metadata dataset...")
    metadata_dataset = load_dataset(f"{base_repo_name}-metadata")
    
    # Rebuild problem directories from structured dataset
    print(f"📂 Rebuilding problem directories...")
    rebuild_from_structured(structured_dataset, output_dir)
    
    # Save JSONL files
    print(f"💾 Saving JSONL files...")
    
    # Save train/test JSONL files
    if 'train' in sft_dataset:
        save_jsonl(sft_dataset['train'], os.path.join(output_dir, "train.jsonl"))
    if 'test' in sft_dataset:
        save_jsonl(sft_dataset['test'], os.path.join(output_dir, "test.jsonl"))
    
    # Save metadata JSONL files
    if 'train' in metadata_dataset:
        save_jsonl(metadata_dataset['train'], os.path.join(output_dir, "train_metadata.jsonl"))
    if 'test' in metadata_dataset:
        save_jsonl(metadata_dataset['test'], os.path.join(output_dir, "test_metadata.jsonl"))
    
    # Save no_test_case JSONL files
    if 'train_no_tests' in sft_dataset:
        save_jsonl(sft_dataset['train_no_tests'], os.path.join(output_dir, "no_test_case_train.jsonl"))
    if 'test_no_tests' in sft_dataset:
        save_jsonl(sft_dataset['test_no_tests'], os.path.join(output_dir, "no_test_case_test.jsonl"))
    
    # Save no_test_case metadata JSONL files
    if 'train_no_tests' in metadata_dataset:
        save_jsonl(metadata_dataset['train_no_tests'], os.path.join(output_dir, "no_test_case_train_metadata.jsonl"))
    if 'test_no_tests' in metadata_dataset:
        save_jsonl(metadata_dataset['test_no_tests'], os.path.join(output_dir, "no_test_case_test_metadata.jsonl"))
    
    print(f"✅ Successfully rebuilt SFT_Data directory!")
    print(f"\nDirectory structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == "__main__":
    # Ask for the repository name
    default_repo = "bhogan/sft-python-q-problems"
    repo_name = input(f"\n🤗 Enter Hugging Face repository name [{default_repo}]: ").strip()
    if not repo_name:
        repo_name = default_repo
    
    # Ask for output directory
    default_dir = "SFT_Data"
    output_dir = input(f"📁 Enter output directory [{default_dir}]: ").strip()
    if not output_dir:
        output_dir = default_dir
    
    # Rebuild the data
    rebuild_sft_data(repo_name, output_dir) 