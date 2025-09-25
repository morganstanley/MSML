"""
Dataset loaders for Q code RL training.

This module provides data loading utilities for RL training on Q programming problems.
Supports loading from SFT_Data directory structure with train/test splits.

Environment Variables:
    SFT_DATA_DIR: Path to SFT_Data directory (auto-detected if not set)
"""
import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Iterator
import logging
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

def collate_fn(batch):
    """
    Custom collate function to ensure consistent batch structure for distributed training.
    
    Args:
        batch: List of problem dictionaries
        
    Returns:
        First (and only) item from the batch since batch_size=1
    """
    if len(batch) == 1:
        return batch[0]  # Return the single problem dict directly
    else:
        logger.warning(f"Unexpected batch size: {len(batch)}, expected 1")
        return batch[0]  # Return first item anyway

class QCodeDataset(Dataset):
    """
    PyTorch Dataset for Q code generation problems.
    Compatible with Accelerate and DataLoader.
    """
    
    def __init__(self, problems: List[Dict], shuffle: bool = True, seed: int = 42):
        """
        Initialize the dataset.
        
        Args:
            problems: List of problem dictionaries
            shuffle: Whether to shuffle the data
            seed: Random seed for shuffling
        """
        self.problems = problems
        self.shuffle = shuffle
        self.seed = seed
        
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.problems)
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        return self.problems[idx]

class QCodeDataLoader:
    """
    DataLoader for Q code generation problems from SFT_Data directory.
    Yields problems one by one and shuffles on each new epoch.
    """
    
    def __init__(self, problems: List[Dict], shuffle: bool = True, seed: int = 42):
        """
        Initialize the dataloader.
        
        Args:
            problems: List of problem dictionaries
            shuffle: Whether to shuffle the data
            seed: Random seed for shuffling
        """
        self.problems = problems
        self.shuffle = shuffle
        self.seed = seed
        self.current_idx = 0
        
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.problems)
    
    def __len__(self):
        return len(self.problems)
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Dict[str, str]:
        if self.current_idx >= len(self.problems):
            self.current_idx = 0
            if self.shuffle:
                random.shuffle(self.problems)
        
        problem = self.problems[self.current_idx]
        self.current_idx += 1
        return problem
    
    def reset(self):
        """Reset the dataloader to the beginning."""
        self.current_idx = 0
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.problems)

def load_file_content(file_path: str) -> str:
    """Load and return file content as string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

def load_problems_from_directory(directory: Path) -> List[Dict]:
    """
    Load all solved problems from a directory where each problem is in its own sub-folder.
    
    Args:
        directory: Path to directory (e.g., SFT_Data/train)
        
    Returns:
        List of problem dictionaries
    """
    problems = []
    
    if not directory.is_dir():
        logger.error(f"Provided path is not a directory: {directory}")
        return []
        
    for problem_dir in directory.iterdir():
        if not problem_dir.is_dir():
            continue
        
        problem_name = problem_dir.name
        
        try:
            # The description file is now known to be `problem_description.txt`
            description_file = problem_dir / "problem_description.txt"
            if not description_file.exists():
                logger.warning(f"Skipping {problem_name}: no description file found (problem_description.txt).")
                continue
            
            description = load_file_content(str(description_file))
            if not description:
                logger.warning(f"Skipping {problem_name}: empty description file.")
                continue

            # Load all test cases using the new format
            tests = []
            test_case_files = sorted(problem_dir.glob("q_test_case_*.q"))
            
            for test_case_file in test_case_files:
                test_idx = test_case_file.stem.replace("q_test_case_", "")
                
                # Use new exact answer format
                q_exact_answer_file = problem_dir / f"q_exact_ans_test_case_{test_idx}.txt"
                
                if not q_exact_answer_file.exists():
                    logger.warning(f"Skipping test {test_idx} for {problem_name}: missing q_exact_ans_test_case_{test_idx}.txt file.")
                    continue
                
                test_code = load_file_content(str(test_case_file))
                expected_output = load_file_content(str(q_exact_answer_file))

                if test_code and expected_output is not None:  # Allow empty string as valid expected output
                    tests.append({
                        "test_idx": int(test_idx),
                        "test_code": test_code,
                        "expected_output": expected_output
                    })

            if not tests:
                logger.warning(f"Skipping {problem_name}: no valid test cases found.")
                continue

            problems.append({
                'id': problem_name,
                'description': description,
                'tests': tests
            })
            logger.debug(f"Loaded problem {problem_name} with {len(tests)} test cases.")

        except Exception as e:
            logger.error(f"Error loading problem {problem_name}: {e}", exc_info=True)
            continue
            
    logger.info(f"Loaded {len(problems)} complete problems from {directory}")
    return problems

def get_q_code_dataloaders(
    sft_data_dir: str = None,
    test_split: float = None,
    seed: int = 42,
    max_train_problems: int = None,
    max_test_problems: int = None
) -> Tuple[DataLoader, QCodeDataLoader]:
    """
    Get train and test dataloaders for Q code problems from SFT_Data directory.
    
    This function loads from pre-split train/test directories in SFT_Data,
    ignoring test_split parameter since data is already split.
    
    Args:
        sft_data_dir: Path to SFT_Data directory (auto-detected if None)
        test_split: IGNORED - data is pre-split in train/test directories
        seed: Random seed for shuffling
        max_train_problems: Maximum number of training problems to load
        max_test_problems: Maximum number of test problems to load
        
    Returns:
        train_loader: Training dataloader (PyTorch DataLoader for Accelerate)
        test_loader: Test dataloader (QCodeDataLoader for evaluation)
    """
    # Auto-detect SFT_Data directory if not provided
    if sft_data_dir is None:
        # Check environment variable first
        sft_data_dir = os.environ.get("SFT_DATA_DIR")
        if sft_data_dir:
            sft_data_dir = Path(sft_data_dir)
        else:
            # Look for SFT_Data in parent directory (relative to rl/rl_training/)
            current_dir = Path(__file__).parent.parent.parent
            sft_data_dir = current_dir / "SFT_Data"
    else:
        sft_data_dir = Path(sft_data_dir)
    
    if not sft_data_dir.exists():
        raise ValueError(f"SFT_Data directory not found: {sft_data_dir}")
    
    train_dir = sft_data_dir / "train"
    test_dir = sft_data_dir / "test"
    
    if not train_dir.exists():
        raise ValueError(f"Training directory not found: {train_dir}")
    if not test_dir.exists():
        raise ValueError(f"Test directory not found: {test_dir}")
    
    logger.info(f"Loading RL training data from {sft_data_dir}")
    
    # Load training problems
    logger.info(f"Loading training problems from {train_dir}")
    train_problems = load_problems_from_directory(train_dir)
    
    # Limit training problems if specified
    if max_train_problems and len(train_problems) > max_train_problems:
        random.seed(seed)
        train_problems = random.sample(train_problems, max_train_problems)
        logger.info(f"Limited training set to {max_train_problems} problems")
    
    # Load test problems
    logger.info(f"Loading test problems from {test_dir}")
    test_problems = load_problems_from_directory(test_dir)
    
    # Limit test problems if specified
    if max_test_problems and len(test_problems) > max_test_problems:
        random.seed(seed)
        test_problems = random.sample(test_problems, max_test_problems)
        logger.info(f"Limited test set to {max_test_problems} problems")
    
    if not train_problems:
        raise ValueError(f"No training problems found in {train_dir}")
    if not test_problems:
        raise ValueError(f"No test problems found in {test_dir}")
    
    logger.info(f"Loaded {len(train_problems)} training problems and {len(test_problems)} test problems")
    
    if test_split is not None:
        logger.warning(f"test_split parameter ({test_split}) ignored - using pre-split train/test directories")
    
    # Create datasets
    train_dataset = QCodeDataset(train_problems, shuffle=True, seed=seed)
    
    # Create PyTorch dataloader for training (needed for Accelerate)
    # batch_size=1 because each GPU will get a different problem automatically via Accelerate
    # Use custom collate_fn to ensure consistent batch structure
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Keep original QCodeDataLoader for test set (evaluation doesn't need to be distributed)
    test_loader = QCodeDataLoader(test_problems, shuffle=False, seed=seed)
    
    return train_loader, test_loader

def create_rl_prompt(description: str, use_reasoning_format: bool = True) -> str:
    """
    Create an RL training prompt for Q code generation.
    
    Args:
        description: Problem description
        use_reasoning_format: Whether to encourage reasoning/answer format
        
    Returns:
        Formatted prompt string
    """
    if use_reasoning_format:
        return f"""Write a Q solve function that solves the following problem. Use a structured approach with reasoning and answer.

Problem:
{description}

Please provide your response in this format:
<reasoning>
[Your step-by-step reasoning about how to solve this problem]
</reasoning>

<answer>
[Your Q solve function code]
</answer>

Make sure your Q solve function is complete and executable."""
    else:
        return f"""Write a Q solve function that solves the following problem:

{description}

Output ONLY the Q solve function. Do not include any other text, explanations, or a test harness."""

def get_rl_training_stats(train_loader, test_loader) -> Dict:
    """
    Get statistics about the RL training dataset.
    
    Args:
        train_loader: Training data loader (PyTorch DataLoader or QCodeDataLoader)
        test_loader: Test data loader (QCodeDataLoader)
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'train_size': len(train_loader),
        'test_size': len(test_loader),
        'total_size': len(train_loader) + len(test_loader),
        'train_problems': [],
        'test_problems': []
    }
    
    # Sample some problem IDs for logging
    # Handle both PyTorch DataLoader and QCodeDataLoader for train_loader
    if hasattr(train_loader, 'problems') and len(train_loader.problems) > 0:
        sample_size = min(10, len(train_loader.problems))
        stats['train_problems'] = [p['id'] for p in train_loader.problems[:sample_size]]
    elif hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'problems') and len(train_loader.dataset.problems) > 0:
        sample_size = min(10, len(train_loader.dataset.problems))
        stats['train_problems'] = [p['id'] for p in train_loader.dataset.problems[:sample_size]]
    
    # test_loader is always QCodeDataLoader
    if len(test_loader.problems) > 0:
        sample_size = min(10, len(test_loader.problems))
        stats['test_problems'] = [p['id'] for p in test_loader.problems[:sample_size]]
    
    return stats 