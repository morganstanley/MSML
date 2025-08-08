# Dataset Building Pipeline

This component handles the creation and processing of the Q programming language dataset, including an iterative bootstrap loop for continuous model improvement.

## Overview

The dataset building pipeline consists of several stages:

1. **Initial Collection** (`process_dataset.py`)
   - Downloads LeetCode problems and solutions
   - Organizes into structured format
   - Creates initial validation dataset

2. **Python to Q Conversion** (`convert_to_q.py`)
   - Uses LLMs to convert Python solutions to Q
   - Supports multiple LLM providers:
     - OpenAI (GPT-4, GPT-3.5)
     - Anthropic (Claude)
     - Local models
   - Includes validation and error checking

3. **Final Verification** (`final_q_verify.py`)
   - Validates Q code solutions
   - Runs test cases
   - Filters invalid or incorrect solutions

4. **Bootstrap Loop** (`bootstrap_loop.py`)
   - Iterative model improvement process
   - Solves problems → Trains model → Evaluates → Repeat
   - Continuous dataset expansion

## Directory Structure

```
build_dataset/
├── process_dataset.py     # Initial dataset processing
├── convert_to_q.py        # Python to Q conversion
├── final_q_verify.py      # Solution verification
├── bootstrap_loop.py      # Iterative improvement loop
├── pull_ds.py            # Dataset downloading
├── validate_dataset.py    # Dataset validation
├── check_final_ds.py     # Final dataset checking
└── output/
    ├── initial_dataset/   # Raw processed data
    ├── validated_dataset/ # Verified solutions
    ├── final_dataset/     # Final cleaned dataset
    └── new_iterations/    # Bootstrap iteration outputs
```

## Bootstrap Loop

The bootstrap loop implements an iterative improvement process:

### How It Works

1. **Problem Solving**: Uses current best model to solve unsolved problems
2. **Dataset Creation**: Combines all solved problems into training data
3. **Model Training**: Fine-tunes model on expanded dataset
4. **Evaluation**: Compares base vs trained model performance
5. **Iteration**: Repeats until target number of problems solved

### Bootstrap Loop Usage

```bash
python bootstrap_loop.py \
  --base-model Qwen/Qwen3-32B \
  --max-iterations 30 \
  --train-steps 1000 \
  --learning-rate 2e-5 \
  --target-problems 100 \
  --debug
```

### Key Parameters

- `--base-model`: Starting model for training
- `--max-iterations`: Maximum bootstrap iterations
- `--train-steps`: Training steps per iteration
- `--learning-rate`: Learning rate for fine-tuning
- `--target-problems`: Target number of problems to solve
- `--debug`: Debug mode (one problem, one iteration)

### Bootstrap Output Structure

```
new_iterations/
├── iteration_1/
│   ├── solved_problems/     # Problems solved in this iteration
│   ├── evaluation/          # Model evaluation results
│   ├── model/              # Trained model checkpoint
│   ├── metadata.json       # Iteration metadata
│   └── dataset_stats.json  # Dataset creation stats
├── iteration_2/
│   └── ...
└── ...
```

### Training Integration

The bootstrap loop calls the SFT training script (`sft/run_sft.py`) for each iteration:

```bash
python sft/run_sft.py \
  --base_model [base_model] \
  --train_file [train_file] \
  --eval_file [eval_file] \
  --output_dir [model_dir] \
  --max_steps [training_steps] \
  --learning_rate [learning_rate] \
  --experiment_name [wandb_name]
```

## Dataset Conversion

### Python to Q Conversion

```bash
python convert_to_q.py \
  --llm_to_use gpt4 \
  --max-files 100 \
  --retries 3 \
  --difficulty Medium
```

### LLM Options

- `gpt`: OpenAI GPT-4/3.5
- `claude`: Anthropic Claude
- `local`: Local Hugging Face model

### Conversion Process

1. **Load Python Solution**: Reads original Python code
2. **LLM Translation**: Converts Python to Q using LLM
3. **Test Case Generation**: Creates Q test cases
4. **Validation**: Runs test cases to verify correctness
5. **Retry Logic**: Multiple attempts with different strategies

## Dataset Verification

### Final Verification

```bash
python final_q_verify.py \
  --input-dir final_dataset \
  --output-dir verified_dataset \
  --max-problems 100
```

### Verification Criteria

- Syntax validity
- Test case success
- Execution time limits
- Code quality metrics

## Usage Examples

### 1. Basic Dataset Creation

```bash
# Process initial dataset
python process_dataset.py

# Convert Python to Q
python convert_to_q.py --llm_to_use gpt4

# Verify solutions
python final_q_verify.py
```

### 2. Bootstrap Loop

```bash
# Start bootstrap loop
python bootstrap_loop.py \
  --base-model Qwen/Qwen3-32B \
  --max-iterations 10 \
  --train-steps 500 \
  --target-problems 50
```

### 3. Resume Bootstrap Loop

```bash
# Resume from iteration 5
python bootstrap_loop.py \
  --resume-iteration 5 \
  --base-model Qwen/Qwen3-32B
```

## Configuration

The component uses the root `config.yaml` for settings:

```yaml
dataset:
  initial_dataset_dir: "initial_dataset"
  validated_dataset_dir: "validated_dataset"
  final_dataset_dir: "final_dataset"
  q_interpreter_path: "q"  # Path to Q interpreter

model:
  base_model: "Qwen/Qwen3-32B"
  max_steps: 1000
  learning_rate: 2e-5
```

## Output Format

Each problem in the final dataset contains:

```
problem_id_task_id/
├── entry.json           # Problem metadata
├── sol.py              # Original Python solution
├── sol.q               # Converted Q solution
├── problem_description.txt  # Problem description
├── test_case_1.py      # Python test case
├── test_case_1_correct_ans.txt  # Expected output
├── q_test_case_1.q     # Q test case
└── ...
```

## Monitoring and Logging

### Bootstrap Progress

- **Iteration Metadata**: Stored in `metadata.json` per iteration
- **High-Level Summaries**: Stored in `new_high_level_summary/`
- **Final Report**: Complete bootstrap summary in `final_report.json`

### Log Files

- `bootstrap.log`: Detailed bootstrap loop logs
- `q_conversion_log_[timestamp].json`: Conversion process logs
- `logging_final_dataset/`: Detailed conversion logs per problem

## Error Handling

The pipeline includes robust error handling:

- **LLM API Failures**: Retry logic with exponential backoff
- **Q Interpreter Errors**: Timeout and error detection
- **Invalid Syntax**: Automatic detection and filtering
- **Memory Management**: GPU cache clearing and garbage collection

## Troubleshooting

### Common Issues

1. **LLM API Errors**
   - Check API key configuration
   - Verify rate limits
   - Ensure proper network connectivity

2. **Q Interpreter Issues**
   - Verify Q installation
   - Check Q_INTERPRETER_PATH setting
   - Ensure proper permissions

3. **Memory Issues**
   - Adjust batch processing size
   - Use incremental processing
   - Clean up temporary files

4. **Bootstrap Loop Issues**
   - Check training script exists (`sft/run_sft.py`)
   - Verify model paths are correct
   - Monitor GPU memory usage

## Contributing

When adding new features:

1. Update the appropriate processing script
2. Add any new dependencies to requirements.txt
3. Update this README with new options/features
4. Add tests for new functionality
5. Update bootstrap loop if training process changes 