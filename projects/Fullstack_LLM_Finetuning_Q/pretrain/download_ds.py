#!/usr/bin/env python3
"""
Script to download kdb+ license dataset from Hugging Face and save as JSONL files.
"""

import json
import os
from datasets import load_dataset
import argparse

def download_dataset(repo_name, username=None):
    """Download dataset from Hugging Face."""
    if username:
        full_repo_name = f"{username}/{repo_name}"
    else:
        full_repo_name = repo_name
    
    print(f"Downloading dataset from {full_repo_name}...")
    
    try:
        dataset = load_dataset(full_repo_name)
        print("Successfully downloaded dataset!")
        return dataset
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def save_to_jsonl(dataset, output_dir):
    """Save dataset splits to JSONL files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in dataset.items():
        output_file = os.path.join(output_dir, f"{split_name}_license_kdbsite.jsonl")
        
        print(f"Saving {split_name} split to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in split_data:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Saved {len(split_data)} examples to {output_file}")

def print_dataset_info(dataset):
    """Print information about the downloaded dataset."""
    print("\nDataset Information:")
    print("=" * 50)
    
    for split_name, split_data in dataset.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Number of examples: {len(split_data)}")
        print(f"  Features: {split_data.features}")
        
        # Show a sample
        if len(split_data) > 0:
            print(f"  Sample text (first 200 chars): {split_data[0]['text'][:200]}...")
    
    print("\n" + "=" * 50)

def main():
    parser = argparse.ArgumentParser(description='Download kdb+ license dataset from Hugging Face')
    parser.add_argument('--repo-name', 
                       default='kdb-license-dataset',
                       help='Repository name for the dataset')
    parser.add_argument('--username', 
                       help='Hugging Face username (optional)')
    parser.add_argument('--output-dir', 
                       default='downloaded_kdb_dataset',
                       help='Output directory for JSONL files')
    parser.add_argument('--show-info', 
                       action='store_true',
                       help='Show detailed dataset information')
    
    args = parser.parse_args()
    
    # Download dataset
    dataset = download_dataset(args.repo_name, args.username)
    
    if dataset is None:
        print("Failed to download dataset. Exiting.")
        return
    
    # Print dataset info
    print_dataset_info(dataset)
    
    # Save to JSONL files
    save_to_jsonl(dataset, args.output_dir)
    
    print(f"\nDataset saved to: {args.output_dir}")
    print("Files created:")
    for split_name in dataset.keys():
        filename = f"{split_name}_license_kdbsite.jsonl"
        filepath = os.path.join(args.output_dir, filename)
        print(f"  - {filepath}")
    
    if args.show_info:
        print("\nDetailed Dataset Information:")
        print("=" * 50)
        for split_name, split_data in dataset.items():
            print(f"\n{split_name.upper()} Split Details:")
            print(f"  Dataset type: {type(split_data)}")
            print(f"  Column names: {split_data.column_names}")
            print(f"  Number of rows: {len(split_data)}")
            print(f"  Features: {split_data.features}")
            
            # Show more detailed sample
            if len(split_data) > 0:
                sample = split_data[0]
                print(f"  Sample data:")
                for key, value in sample.items():
                    if isinstance(value, str):
                        print(f"    {key}: {value[:100]}...")
                    else:
                        print(f"    {key}: {value}")

if __name__ == "__main__":
    main() 