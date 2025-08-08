#!/usr/bin/env python3
"""
Script to check final_dataset for problems with Q test files that don't call solve function.
Can also delete problematic problem folders.
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

def check_q_files_for_solve(problem_dir: Path) -> Tuple[bool, List[str]]:
    """
    Check if all Q test files in a problem directory contain 'solve'.
    
    Returns:
        (all_valid, invalid_files): bool indicating if all files are valid, 
                                   list of invalid file names
    """
    q_test_files = list(problem_dir.glob("q_test_case_*.q"))
    
    if not q_test_files:
        return False, ["No Q test files found"]
    
    invalid_files = []
    
    for q_file in q_test_files:
        try:
            with open(q_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "solve" not in content:
                invalid_files.append(q_file.name)
                
        except Exception as e:
            invalid_files.append(f"{q_file.name} (read error: {e})")
    
    return len(invalid_files) == 0, invalid_files

def check_final_dataset(final_dir: Path, verbose: bool = False) -> Tuple[int, int, List[str]]:
    """
    Check all problems in final_dataset directory.
    
    Returns:
        (total_problems, invalid_problems, invalid_problem_names)
    """
    if not final_dir.exists():
        print(f"ERROR: {final_dir} does not exist!")
        return 0, 0, []
    
    problem_dirs = [d for d in final_dir.iterdir() if d.is_dir()]
    total_problems = len(problem_dirs)
    invalid_problems = 0
    invalid_problem_names = []
    
    print(f"Checking {total_problems} problems in {final_dir}...")
    print()
    
    for problem_dir in sorted(problem_dirs):
        is_valid, invalid_files = check_q_files_for_solve(problem_dir)
        
        if not is_valid:
            invalid_problems += 1
            invalid_problem_names.append(problem_dir.name)
            
            if verbose:
                print(f"❌ {problem_dir.name}:")
                for invalid_file in invalid_files:
                    print(f"   - {invalid_file}")
            else:
                print(f"❌ {problem_dir.name} (invalid Q test files)")
        else:
            if verbose:
                print(f"✅ {problem_dir.name}")
    
    return total_problems, invalid_problems, invalid_problem_names

def delete_invalid_problems(final_dir: Path, invalid_problem_names: List[str], dry_run: bool = False):
    """
    Delete invalid problem directories.
    
    Args:
        final_dir: Path to final_dataset directory
        invalid_problem_names: List of problem directory names to delete
        dry_run: If True, only print what would be deleted without actually deleting
    """
    if not invalid_problem_names:
        print("No invalid problems to delete.")
        return
    
    print(f"\n{'DRY RUN: ' if dry_run else ''}Deleting {len(invalid_problem_names)} invalid problem directories...")
    
    deleted_count = 0
    failed_count = 0
    
    for problem_name in invalid_problem_names:
        problem_path = final_dir / problem_name
        
        if not problem_path.exists():
            print(f"⚠️  {problem_name} - Directory not found, skipping")
            continue
        
        try:
            if dry_run:
                print(f"🗑️  Would delete: {problem_path}")
            else:
                shutil.rmtree(problem_path)
                print(f"🗑️  Deleted: {problem_name}")
            deleted_count += 1
            
        except Exception as e:
            print(f"❌ Failed to delete {problem_name}: {e}")
            failed_count += 1
    
    if not dry_run:
        print(f"\nDeletion complete:")
        print(f"  Successfully deleted: {deleted_count}")
        print(f"  Failed to delete: {failed_count}")
    else:
        print(f"\nDry run complete - {deleted_count} directories would be deleted")

def main():
    parser = argparse.ArgumentParser(description='Check and clean final_dataset for invalid Q test files')
    parser.add_argument('--final-dir', type=str, default='final_dataset',
                        help='Path to final_dataset directory (default: final_dataset)')
    parser.add_argument('--delete', action='store_true',
                        help='Delete invalid problem directories')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting (only with --delete)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed information about each problem')
    args = parser.parse_args()
    
    final_dir = Path(args.final_dir)
    
    # Check the dataset
    total_problems, invalid_problems, invalid_problem_names = check_final_dataset(final_dir, args.verbose)
    
    if total_problems == 0:
        return
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"FINAL DATASET CHECK SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems: {total_problems}")
    print(f"Invalid problems: {invalid_problems}")
    print(f"Valid problems: {total_problems - invalid_problems}")
    print(f"Invalid percentage: {invalid_problems/total_problems*100:.1f}%")
    
    # if invalid_problems > 0:
    #     print(f"\nInvalid problems (Q test files don't call solve):")
    #     for i, problem_name in enumerate(invalid_problem_names, 1):
    #         print(f"  {i:3d}. {problem_name}")
    
    # Handle deletion if requested
    if args.delete:
        if invalid_problems == 0:
            print("\nNo invalid problems to delete.")
        else:
            print(f"\n{'='*60}")
            if args.dry_run:
                print("DRY RUN MODE - NO FILES WILL BE DELETED")
            else:
                print("DELETION MODE - FILES WILL BE PERMANENTLY DELETED")
            print(f"{'='*60}")
            
            if not args.dry_run:
                confirm = input(f"Are you sure you want to delete {invalid_problems} problem directories? (yes/no): ")
                if confirm.lower() != 'yes':
                    print("Deletion cancelled.")
                    return
            
            delete_invalid_problems(final_dir, invalid_problem_names, args.dry_run)
    
    elif invalid_problems > 0:
        print(f"\nTo delete invalid problems, run with --delete flag:")
        print(f"  python {__file__} --delete")
        print(f"  python {__file__} --delete --dry-run  # to see what would be deleted")

if __name__ == "__main__":
    main() 