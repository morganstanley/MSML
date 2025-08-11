#!/usr/bin/env python3
"""
Full Two-Phase Evaluation Runner

Convenience script to run both generation and evaluation phases sequentially.
Handles common configurations and provides easy examples.

Usage:
    python run_full_evaluation.py --model_path /path/to/model --output_dir results --n_completions 8
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} completed successfully in {elapsed_time/60:.1f}m")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full two-phase evaluation (generation + evaluation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic vLLM evaluation (local model)
    python run_full_evaluation.py --model_path Qwen/Qwen2.5-14B-Instruct --output_dir results --n_completions 8

    # OpenAI GPT evaluation
    python run_full_evaluation.py --model_type openai --model_path gpt-4o \\
        --output_dir results --n_completions 4 --api_key your_openai_key

    # xAI Groq-4 evaluation
    python run_full_evaluation.py --model_type xai --model_path groq-4 \\
        --output_dir results --n_completions 4 --api_key your_xai_key

    # Google Gemini evaluation
    python run_full_evaluation.py --model_type google --model_path gemini-1.5-pro \\
        --output_dir results --n_completions 4 --api_key your_google_key

    # Anthropic Claude evaluation  
    python run_full_evaluation.py --model_type anthropic --model_path claude-3-5-sonnet-20241022 \\
        --output_dir results --n_completions 4 --api_key your_anthropic_key

    # HuggingFace model evaluation
    python run_full_evaluation.py --model_type huggingface --model_path /path/to/model \\
        --output_dir results --n_completions 8

    # Specific tasks with OpenAI
    python run_full_evaluation.py --model_type openai --model_path gpt-4o --output_dir results \\
        --n_completions 4 --tasks description_to_q --api_key your_key --max_workers 20

    # Full evaluation with thinking mode (works with any model type)
    python run_full_evaluation.py --model_path /path/to/model --output_dir results \\
        --n_completions 8 --thinking_mode --include_context --max_workers 15

    # Ground truth evaluation (for testing)
    python run_full_evaluation.py --model_type ground_truth --model_path dummy \\
        --output_dir results --n_completions 1
        """
    )
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Model identifier: path for local models, model name for API models")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--n_completions", type=int, required=True,
                       help="Number of completions to generate per problem")
    
    # Model type and API arguments
    parser.add_argument("--model_type", type=str, default="vllm_api",
                       choices=['openai', 'anthropic', 'vllm_api', 'huggingface', 'ground_truth', 'xai', 'google'],
                       help="Type of model provider (default: vllm_api)")
    parser.add_argument("--api_key", type=str,
                       help="API key for OpenAI, Anthropic, xAI or Google models (or use OPENAI_API_KEY/ANTHROPIC_API_KEY/XAI_API_KEY/GEMINI_API_KEY env vars)")
    
    # Generation phase arguments
    parser.add_argument("--test_dir", type=str, default="../SFT_Data/test",
                       help="Test data directory")
    parser.add_argument("--max_problems", type=int,
                       help="Maximum number of problems to process")
    parser.add_argument("--tasks", nargs='+', 
                       choices=['description_to_q', 'python_to_q', 'q_to_python'],
                       help="Tasks to evaluate")
    parser.add_argument("--include_context", action="store_true",
                       help="Include Q reference context")
    parser.add_argument("--thinking_mode", action="store_true",
                       help="Use reasoning format")
    
    # Server arguments  
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                       help="Server port")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=6000,
                       help="Maximum model context length")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size")
    
    # Worker arguments
    parser.add_argument("--max_workers", type=int, default=6,
                       help="Maximum number of worker threads/processes (default: 6 - conservative for system stability)")
    
    # Control arguments
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip generation phase (only run evaluation)")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip evaluation phase (only run generation)")
    parser.add_argument("--force_evaluate", action="store_true",
                       help="Force evaluation to run even if generation failed or was incomplete.")
    
    args = parser.parse_args()
    
    if args.skip_generation and args.skip_evaluation:
        print("Error: Cannot skip both phases!")
        return 1
    
    script_dir = Path(__file__).parent
    start_time = time.time()
    
    print("=" * 80)
    print("🎯 FULL TWO-PHASE EVALUATION")
    print("=" * 80)
    print(f"Model Type: {args.model_type}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Completions: {args.n_completions}")
    print(f"Workers: {args.max_workers}")
    if args.api_key:
        print(f"API Key: {'*' * 8} (provided)")
    if args.tasks:
        print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = True
    
    # Phase 1: Generation
    if not args.skip_generation:
        gen_cmd = [
            "python", str(script_dir / "generate_completions.py"),
            "--model_type", args.model_type,
            "--model_path", args.model_path,
            "--output_dir", args.output_dir,
            "--n_completions", str(args.n_completions),
            "--test_dir", args.test_dir,
            "--max_workers", str(args.max_workers)
        ]
        
        # Add API key if provided
        if args.api_key:
            gen_cmd.extend(["--api_key", args.api_key])
        
        # Add vLLM-specific parameters only if using vLLM
        if args.model_type == "vllm_api":
            gen_cmd.extend([
                "--host", args.host,
                "--port", str(args.port),
                "--gpu_memory_utilization", str(args.gpu_memory_utilization),
                "--max_model_len", str(args.max_model_len),
                "--tensor_parallel_size", str(args.tensor_parallel_size)
            ])
        
        if args.max_problems:
            gen_cmd.extend(["--max_problems", str(args.max_problems)])
        if args.tasks:
            gen_cmd.extend(["--tasks"] + args.tasks)
        if args.include_context:
            gen_cmd.append("--include_context")
        if args.thinking_mode:
            gen_cmd.append("--thinking_mode")
        
        success = run_command(gen_cmd, "Phase 1: Generate Completions")
    
    # Phase 2: Evaluation
    if (success or args.force_evaluate) and not args.skip_evaluation:
        if not success:  # This implies args.force_evaluate is True
            print("\n" + "⚠️" * 40)
            print("⚠️ WARNING: Generation phase failed or was incomplete.")
            print("⚠️ Forcing evaluation on available completions as requested.")
            print("⚠️" * 40 + "\n")

        eval_cmd = [
            "python", str(script_dir / "evaluate_completions.py"),
            "--output_dir", args.output_dir,
            "--test_dir", args.test_dir,
            "--max_workers", str(args.max_workers)
        ]
        
        if args.max_problems:
            eval_cmd.extend(["--max_problems", str(args.max_problems)])
        if args.tasks:
            eval_cmd.extend(["--tasks"] + args.tasks)
        if args.force_evaluate:
            eval_cmd.append("--force")

        # The success of the whole pipeline depends on both phases
        eval_success = run_command(eval_cmd, "Phase 2: Evaluate Completions")
        success = success and eval_success
    
    elif not args.skip_evaluation:
        print("\n" + "ℹ️  Skipping evaluation because generation failed. Use --force_evaluate to override.")

    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📁 Results: {args.output_dir}/results.json")
        print("=" * 80)
        return 0
    else:
        print("❌ EVALUATION FAILED!")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 