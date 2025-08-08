# Two-Phase Evaluation System

A robust evaluation system for Q language models that separates generation and testing for better reliability and analysis.

## Overview

The evaluation system consists of two distinct phases:

1. **Generation Phase** (`generate_completions.py`)
   - Generates N completions per problem
   - Supports multiple model types
   - Handles various task formats

2. **Evaluation Phase** (`evaluate_completions.py`)
   - Tests all completions against test cases
   - Calculates comprehensive metrics
   - Generates detailed reports

## Directory Structure

```
eval/
├── generate_completions.py   # Generation phase
├── evaluate_completions.py   # Evaluation phase
├── run_full_evaluation.py    # Combined pipeline
├── llm_interface.py         # Model interface
└── output/
    ├── completions/         # Generated solutions
    ├── results/            # Test results
    └── results.json        # Final metrics
```

## Usage

### Option 1: Full Pipeline (Recommended)

```bash
# Basic evaluation with 8 completions
python run_full_evaluation.py \
    --model_path Qwen/Qwen2.5-14B-Instruct \
    --output_dir results \
    --n_completions 8

# Specific tasks only
python run_full_evaluation.py \
    --model_path /path/to/model \
    --output_dir results \
    --n_completions 4 \
    --tasks description_to_q \
    --max_workers 20
```

### Option 2: Separate Phases

1. **Generate Completions**:
   ```bash
   python generate_completions.py \
       --model_path /path/to/model \
       --output_dir results \
       --n_completions 4
   ```

2. **Evaluate Completions**:
   ```bash
   python evaluate_completions.py \
       --test_dir test_problems \
       --output_dir results
   ```

## Metrics

The system calculates several metrics:

1. **Pass@k Metrics**
   - Empirical pass@k (k=1,2,4,6,8,16)
   - Probabilistic pass@k
   - Valid problem counts

2. **Test Case Coverage**
   - Total test cases passed
   - Pass rate per problem
   - Overall pass rate

3. **Task-Specific Metrics**
   - Success rates by task type
   - Problem-specific statistics
   - Completion quality metrics

## Output Structure

```
output_dir/
├── completions/                    # Phase 1 output
│   ├── problem1_description_to_q/
│   │   ├── completion_1.q
│   │   ├── completion_2.q
│   │   └── ... (up to N)
│   └── problem2_python_to_q/
│       ├── completion_1.q
│       └── ...
├── results/                        # Phase 2 output
│   ├── problem1_description_to_q/
│   │   ├── completion_1_results.json
│   │   ├── completion_2_results.json
│   │   └── ...
│   └── ...
└── results.json                    # Final metrics
```

## Configuration

Key configuration options:

1. **Model Settings**
   - `--model_path`: Model to evaluate
   - `--n_completions`: Completions per problem
   - `--max_new_tokens`: Generation length

2. **Task Selection**
   - `--tasks`: Specific tasks to evaluate
   - `--max_problems`: Problem count limit
   - `--test_dir`: Test problem directory

3. **System Settings**
   - `--max_workers`: Parallel workers
   - `--timeout`: Execution timeout
   - `--force`: Skip missing completion check

## Supported Models

The system supports various model types:

1. **Local Models**
   - HuggingFace models
   - vLLM-served models
   - Custom model implementations

2. **API Models**
   - OpenAI (GPT-4, GPT-3.5)
   - Anthropic (Claude)
   - Google (Gemini)
   - xAI models

## Error Handling

The system includes robust error handling:

1. **Generation Phase**
   - Model API failures
   - Token limit exceeded
   - Invalid generations

2. **Evaluation Phase**
   - Syntax errors
   - Runtime errors
   - Timeout handling
   - Resource management

## Extending the System

To add new features:

1. **New Model Support**
   - Extend `llm_interface.py`
   - Add model-specific handling
   - Update generation logic

2. **New Metrics**
   - Add to `evaluate_completions.py`
   - Update results schema
   - Document in README

3. **New Task Types**
   - Add task-specific evaluation
   - Update problem loading
   - Add test cases

## Troubleshooting

Common issues and solutions:

1. **Missing Completions**
   - Check model availability
   - Verify API keys
   - Use `--force` flag if needed

2. **Evaluation Failures**
   - Check Q interpreter setup
   - Verify test case format
   - Adjust timeout settings

3. **Performance Issues**
   - Adjust `max_workers`
   - Check resource usage
   - Monitor API rate limits 