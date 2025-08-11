#!/usr/bin/env python3
"""
Curriculum Learning Data Organizer

This script organizes SFT training data for curriculum learning strategies:
1. Difficulty-based curriculum (Easy → Medium → Hard)
2. Task-type curriculum (similar tasks grouped together)
3. Mixed strategies
4. Tag-based curriculum (programming concepts grouped together)

Usage:
    python curriculum_organizer.py --strategy tag_based --output_dir curriculum_data/
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_data(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load training data and metadata."""
    train_file = Path(data_dir) / "train.jsonl"
    metadata_file = Path(data_dir) / "train_metadata.jsonl"
    
    # Load training examples
    train_data = []
    with open(train_file, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    # Load metadata
    metadata = []
    with open(metadata_file, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))
    
    logger.info(f"Loaded {len(train_data)} training examples and {len(metadata)} metadata entries")
    return train_data, metadata

def analyze_data_distribution(metadata: List[Dict]) -> Dict[str, Any]:
    """Analyze the distribution of difficulties and task types."""
    difficulty_counts = Counter(item['difficulty'] for item in metadata)
    task_type_counts = Counter(item['task_type'] for item in metadata)
    
    # Cross-tabulation
    cross_tab = defaultdict(lambda: defaultdict(int))
    for item in metadata:
        cross_tab[item['difficulty']][item['task_type']] += 1
    
    analysis = {
        'total_examples': len(metadata),
        'difficulty_distribution': dict(difficulty_counts),
        'task_type_distribution': dict(task_type_counts),
        'cross_tabulation': {diff: dict(tasks) for diff, tasks in cross_tab.items()}
    }
    
    logger.info("Data Distribution:")
    logger.info(f"  Total examples: {analysis['total_examples']}")
    logger.info(f"  Difficulties: {analysis['difficulty_distribution']}")
    logger.info(f"  Task types: {analysis['task_type_distribution']}")
    
    return analysis

def create_difficulty_curriculum(train_data: List[Dict], metadata: List[Dict]) -> List[Tuple[str, List[Dict]]]:
    """Create curriculum organized by difficulty: Easy → Medium → Hard."""
    # Group by difficulty
    difficulty_groups = defaultdict(list)
    
    for example, meta in zip(train_data, metadata):
        difficulty_groups[meta['difficulty']].append(example)
    
    # Define curriculum order
    curriculum_order = ['Easy', 'Medium', 'Hard']
    curriculum = []
    
    for difficulty in curriculum_order:
        if difficulty in difficulty_groups:
            examples = difficulty_groups[difficulty]
            curriculum.append((f"difficulty_{difficulty.lower()}", examples))
            logger.info(f"Difficulty {difficulty}: {len(examples)} examples")
    
    return curriculum

def create_task_type_curriculum(train_data: List[Dict], metadata: List[Dict]) -> List[Tuple[str, List[Dict]]]:
    """Create curriculum organized by task type."""
    # Group by task type
    task_groups = defaultdict(list)
    
    for example, meta in zip(train_data, metadata):
        task_groups[meta['task_type']].append(example)
    
    # Order task types (you can customize this ordering)
    task_order = ['description_to_q', 'python_to_q', 'q_to_python', 'pytest_to_qtest']
    curriculum = []
    
    for task_type in task_order:
        if task_type in task_groups:
            examples = task_groups[task_type]
            curriculum.append((f"task_{task_type}", examples))
            logger.info(f"Task {task_type}: {len(examples)} examples")
    
    return curriculum

def create_tag_based_curriculum(train_data: List[Dict], metadata: List[Dict]) -> List[Tuple[str, List[Dict]]]:
    """Create curriculum organized by programming concept tags."""
    
    # Define tag groups based on programming concepts
    tag_groups = {
        "basic_structures": {
            "name": "Basic Data Structures",
            "tags": ["Array", "String", "Hash Table", "Stack", "Queue"],
            "description": "Fundamental data structures and basic operations"
        },
        "algorithms": {
            "name": "Core Algorithms", 
            "tags": ["Sorting", "Binary Search", "Two Pointers", "Sliding Window", "Greedy"],
            "description": "Essential algorithmic techniques and search methods"
        },
        "dynamic_programming": {
            "name": "Dynamic Programming & Optimization",
            "tags": ["Dynamic Programming", "Memoization", "Backtracking", "Recursion"],
            "description": "Optimization problems and recursive solutions"
        },
        "graph_tree": {
            "name": "Graph & Tree Algorithms",
            "tags": ["Tree", "Graph", "Depth-First Search", "Breadth-First Search", "Union Find", "Binary Tree"],
            "description": "Graph theory and tree-based algorithms"
        },
        "math_logic": {
            "name": "Mathematical & Logic Problems",
            "tags": ["Math", "Bit Manipulation", "Number Theory", "Geometry", "Combinatorics", "Probability and Statistics"],
            "description": "Mathematical computations and logical reasoning"
        },
        "advanced_structures": {
            "name": "Advanced Data Structures",
            "tags": ["Heap (Priority Queue)", "Segment Tree", "Binary Indexed Tree", "Trie", "Ordered Set", "Monotonic Stack", "Monotonic Queue"],
            "description": "Complex data structures and specialized algorithms"
        }
    }
    
    # Group examples by tag categories
    curriculum_groups = defaultdict(list)
    
    for example, meta in zip(train_data, metadata):
        example_tags = set(meta['tags'])
        
        # Find which group this example belongs to (prioritize by order)
        assigned = False
        for group_id, group_info in tag_groups.items():
            group_tags = set(group_info["tags"])
            if example_tags.intersection(group_tags):
                curriculum_groups[group_id].append(example)
                assigned = True
                break
        
        # If no group matches, assign to basic_structures as fallback
        if not assigned:
            curriculum_groups["basic_structures"].append(example)
    
    # Create curriculum in order
    curriculum = []
    group_order = ["basic_structures", "algorithms", "dynamic_programming", "graph_tree", "math_logic", "advanced_structures"]
    
    for group_id in group_order:
        if group_id in curriculum_groups:
            examples = curriculum_groups[group_id]
            group_name = tag_groups[group_id]["name"]
            curriculum.append((f"tags_{group_id}", examples))
            logger.info(f"Tag Group '{group_name}': {len(examples)} examples")
    
    return curriculum

def create_mixed_curriculum(train_data: List[Dict], metadata: List[Dict]) -> List[Tuple[str, List[Dict]]]:
    """Create mixed curriculum: Easy tasks of each type, then Medium, then Hard."""
    # Group by both difficulty and task type
    groups = defaultdict(lambda: defaultdict(list))
    
    for example, meta in zip(train_data, metadata):
        groups[meta['difficulty']][meta['task_type']].append(example)
    
    curriculum = []
    difficulty_order = ['Easy', 'Medium', 'Hard']
    
    for difficulty in difficulty_order:
        if difficulty in groups:
            # Combine all task types for this difficulty
            combined_examples = []
            for task_type, examples in groups[difficulty].items():
                combined_examples.extend(examples)
            
            curriculum.append((f"mixed_{difficulty.lower()}", combined_examples))
            logger.info(f"Mixed {difficulty}: {len(combined_examples)} examples")
    
    return curriculum

def create_progressive_curriculum(train_data: List[Dict], metadata: List[Dict]) -> List[Tuple[str, List[Dict]]]:
    """Create progressive curriculum: gradually introduce harder problems."""
    # Group by difficulty
    difficulty_groups = defaultdict(list)
    
    for example, meta in zip(train_data, metadata):
        difficulty_groups[meta['difficulty']].append(example)
    
    curriculum = []
    
    # Phase 1: Only Easy
    if 'Easy' in difficulty_groups:
        curriculum.append(("phase1_easy_only", difficulty_groups['Easy']))
    
    # Phase 2: Easy + Medium  
    if 'Easy' in difficulty_groups and 'Medium' in difficulty_groups:
        combined = difficulty_groups['Easy'] + difficulty_groups['Medium']
        curriculum.append(("phase2_easy_medium", combined))
    
    # Phase 3: All difficulties
    all_examples = []
    for difficulty in ['Easy', 'Medium', 'Hard']:
        if difficulty in difficulty_groups:
            all_examples.extend(difficulty_groups[difficulty])
    curriculum.append(("phase3_all", all_examples))
    
    for phase_name, examples in curriculum:
        logger.info(f"{phase_name}: {len(examples)} examples")
    
    return curriculum

def save_curriculum_data(curriculum: List[Tuple[str, List[Dict]]], output_dir: str):
    """Save curriculum phases as separate JSONL files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for phase_name, examples in curriculum:
        phase_file = output_path / f"{phase_name}.jsonl"
        with open(phase_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        logger.info(f"Saved {len(examples)} examples to {phase_file}")

def main():
    parser = argparse.ArgumentParser(description="Organize SFT data for curriculum learning")
    parser.add_argument("--data_dir", type=str, default="../SFT_Data", help="SFT data directory")
    parser.add_argument("--strategy", type=str, required=True, 
                       choices=["difficulty", "task_type", "mixed", "progressive", "tag_based"],
                       help="Curriculum strategy")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for curriculum data")
    
    args = parser.parse_args()
    
    logger.info(f"Creating {args.strategy} curriculum from {args.data_dir}")
    
    # Load data
    train_data, metadata = load_training_data(args.data_dir)
    
    # Analyze distribution
    analysis = analyze_data_distribution(metadata)
    
    # Create curriculum based on strategy
    if args.strategy == "difficulty":
        curriculum = create_difficulty_curriculum(train_data, metadata)
    elif args.strategy == "task_type":
        curriculum = create_task_type_curriculum(train_data, metadata)
    elif args.strategy == "mixed":
        curriculum = create_mixed_curriculum(train_data, metadata)
    elif args.strategy == "progressive":
        curriculum = create_progressive_curriculum(train_data, metadata)
    elif args.strategy == "tag_based":
        curriculum = create_tag_based_curriculum(train_data, metadata)
    
    # Save curriculum data
    save_curriculum_data(curriculum, args.output_dir)
    
    # Save analysis
    analysis_file = Path(args.output_dir) / "curriculum_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Curriculum creation completed. Strategy: {args.strategy}")
    logger.info(f"Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 