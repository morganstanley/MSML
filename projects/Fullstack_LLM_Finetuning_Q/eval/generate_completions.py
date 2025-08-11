#!/usr/bin/env python3
"""
Phase 1: Generate Completions

Starts a vLLM server and generates N completions for each test problem.
Saves completions in structured directory format with resume capability.

Usage:
    python generate_completions.py --model_path /path/to/model --output_dir results --n_completions 8
    python generate_completions.py --model_path Qwen/Qwen2.5-14B-Instruct --output_dir results --n_completions 4 --tasks description_to_q
"""

import argparse
import asyncio
import aiohttp
import json
import os
import signal
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests

from evaluation_utils import (
    load_test_problems, create_prompt, extract_code_from_response, 
    get_file_extension, get_problem_id, load_file_content, logger
)
from llm_interface import create_llm_interface


def extract_thinking_from_response(response: str) -> Optional[str]:
    """Extracts content from the <reasoning> tag."""
    start_tag = "<reasoning>"
    end_tag = "</reasoning>"
    start_idx = response.find(start_tag)
    if start_idx == -1:
        return None
    
    end_idx = response.find(end_tag, start_idx)
    if end_idx == -1:
        return None
        
    return response[start_idx + len(start_tag):end_idx].strip()


def is_infrastructure_failure(error_msg: str) -> bool:
    """Check if an error indicates infrastructure failure that should cause job exit."""
    infrastructure_keywords = [
        "INFRASTRUCTURE_FAILURE",
        "License daemon overloaded",
        "couldn't connect to license daemon",
        "Connection refused",
        "Service unavailable", 
        "Internal server error",
        "Rate limit exceeded",
        "Server overloaded"
    ]
    return any(keyword.lower() in error_msg.lower() for keyword in infrastructure_keywords)


class VLLMServerManager:
    """Manages vLLM server lifecycle."""
    
    def __init__(self, model_path: str, host: str = "127.0.0.1", port: int = 8000):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.api_base = f"http://{host}:{port}/v1"
        self.server_process = None
        
    def is_server_running(self) -> bool:
        """Check if server is already running."""
        try:
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def start_server(self, gpu_memory_utilization: float = 0.9, max_model_len: int = 4096, 
                    tensor_parallel_size: int = 1, wait_timeout: int = 300) -> bool:
        """Start the vLLM server."""
        if self.is_server_running():
            logger.info(f"✓ Server already running at {self.api_base}")
            return True
        
        logger.info(f"Starting vLLM server for {self.model_path}...")
        
        # Build command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--dtype", "bfloat16",
            "--disable-log-requests",
            "--served-model-name", "model"
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Start server process
        # log_path = os.path.join(str(self.output_dir), "vllm_server.log")
        log_path = "vllm_server.log"
        self.server_process = subprocess.Popen(
            cmd,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(os.environ, PYTHONUNBUFFERED='1')
        )
        
        # Wait for server to be ready
        logger.info("Waiting for server to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < wait_timeout:
            if self.is_server_running():
                logger.info(f"✓ Server ready at {self.api_base}")
                return True
            
            # Check if process died
            if self.server_process.poll() is not None:
                logger.error("✗ Server process died")
                return False
            
            time.sleep(2)
            print(".", end="", flush=True)
        
        logger.error(f"\n✗ Server failed to start within {wait_timeout} seconds")
        return False
    
    def stop_server(self):
        """Stop the vLLM server."""
        if self.server_process:
            logger.info("Stopping vLLM server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.info("Force killing server...")
                self.server_process.kill()
                self.server_process.wait()
            logger.info("✓ Server stopped")


class CompletionGenerator:
    """Generates completions for problems using parallel workers."""
    
    def __init__(self, model_type: str, model_name: str, api_base: str = None, api_key: str = None, 
                 test_dir: str = "../SFT_Data/test", max_workers: int = 10):
        self.model_type = model_type
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.test_dir = test_dir
        self.max_workers = max_workers
        
    def _get_llm(self):
        """Create fresh LLM interface for each thread (thread-safe)."""
        return create_llm_interface(
            self.model_type, self.model_name, 
            api_base=self.api_base, api_key=self.api_key, test_dir=self.test_dir
        )
    
    def generate_single_completion(self, problem: Dict[str, Any], completion_num: int,
                                 context: Optional[str] = None, thinking_mode: bool = False) -> Dict[str, Any]:
        """Generate a single completion for a problem."""
        response = ""  # Initialize to handle cases where generation fails
        try:
            # Create prompt
            prompt = create_prompt(problem, context, thinking_mode)
            
            # Generate response
            llm = self._get_llm()
            response = llm.generate(prompt, 
                                  problem_name=problem['problem_name'], 
                                  task_type=problem['task_type'])
            
            # Extract code
            language = "python" if problem['task_type'] == 'q_to_python' else "q"
            generated_code = extract_code_from_response(response, language, thinking_mode)
            
            # Extract thinking if in thinking mode
            thinking_process = None
            if thinking_mode:
                thinking_process = extract_thinking_from_response(response)

            # Remove exit 0; from Q code if present
            if language == "q" and generated_code.strip().endswith("exit 0;"):
                generated_code = generated_code.strip()[:-7].strip()
            
            return {
                'success': True,
                'completion_num': completion_num,
                'problem_id': get_problem_id(problem),
                'generated_code': generated_code,
                'raw_response': response,
                'thinking_process': thinking_process,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = str(e)
            if is_infrastructure_failure(error_msg):
                logger.error(f"🚨 INFRASTRUCTURE FAILURE DETECTED: {error_msg}")
                logger.error("🔄 Exiting to allow job restart and retry...")
                os._exit(1)  # Force exit entire process
            
            return {
                'success': False,
                'completion_num': completion_num,
                'problem_id': get_problem_id(problem),
                'error': error_msg,
                'raw_response': response,  # Include raw response if available
                'thinking_process': None,
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_completions_for_problems(self, problems: List[Dict[str, Any]], 
                                        n_completions: int, output_dir: Path,
                                        context: Optional[str] = None, 
                                        thinking_mode: bool = False) -> None:
        """Generate completions for all problems."""
        
        # Create completion tasks for missing completions
        tasks = []
        for problem in problems:
            problem_id = get_problem_id(problem)
            problem_dir = output_dir / "completions" / problem_id
            problem_dir.mkdir(parents=True, exist_ok=True)
            
            # Check which completions are missing
            existing_completions = set()
            for f in problem_dir.iterdir():
                if f.name.startswith("completion_") and f.suffix in ['.q', '.py']:
                    try:
                        comp_num = int(f.stem.replace('completion_', ''))
                        existing_completions.add(comp_num)
                    except ValueError:
                        continue
            
            # Add tasks for missing completions
            for i in range(1, n_completions + 1):
                if i not in existing_completions:
                    tasks.append((problem, i))
        
        if not tasks:
            logger.info("✓ All completions already exist")
            return
        
        logger.info(f"Generating {len(tasks)} missing completions using {self.max_workers} workers")
        
        # Process tasks in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self.generate_single_completion, 
                    problem, completion_num, context, thinking_mode
                ): (problem, completion_num)
                for problem, completion_num in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                problem, completion_num = future_to_task[future]
                result = future.result()
                
                completed += 1
                
                problem_id = get_problem_id(problem)

                # Save raw response for debugging to a separate top-level folder
                if 'raw_response' in result and result['raw_response']:
                    full_outputs_dir = output_dir / "full_outputs"
                    full_outputs_dir.mkdir(parents=True, exist_ok=True)
                    raw_response_file = full_outputs_dir / f"{problem_id}_full_completion_{completion_num}.txt"
                    with open(raw_response_file, 'w', encoding='utf-8') as f:
                        f.write(result['raw_response'])
                    logger.debug(f"Saved raw response to {raw_response_file}")
                
                # Save thinking process if available
                if thinking_mode and 'thinking_process' in result and result['thinking_process']:
                    thinking_dir = output_dir / "thinking" / problem_id
                    thinking_dir.mkdir(parents=True, exist_ok=True)
                    thinking_file = thinking_dir / f"completion_{completion_num}_thinking.txt"
                    with open(thinking_file, 'w', encoding='utf-8') as f:
                        f.write(result['thinking_process'])
                    logger.debug(f"Saved thinking process to {thinking_file}")


                if result['success']:
                    # Save completion to file
                    problem_dir = output_dir / "completions" / problem_id
                    ext = get_file_extension(problem['task_type'])
                    completion_file = problem_dir / f"completion_{completion_num}{ext}"
                    
                    with open(completion_file, 'w') as f:
                        f.write(result['generated_code'])
                    
                    logger.debug(f"✓ Saved completion {completion_num} for {problem_id}")
                else:
                    # Check for infrastructure failure in error message  
                    if is_infrastructure_failure(result['error']):
                        logger.error(f"🚨 INFRASTRUCTURE FAILURE DETECTED: {result['error']}")
                        logger.error("🔄 Exiting to allow job restart and retry...")
                        os._exit(1)  # Force exit entire process
                    
                    logger.error(f"✗ Failed completion {completion_num} for {result['problem_id']}: {result['error']}")
                
                # Progress update
                if completed % 10 == 0 or completed == len(tasks):
                    logger.info(f"Progress: {completed}/{len(tasks)} completions generated")


def main():
    parser = argparse.ArgumentParser(
        description="Generate completions for test problems using various LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Local vLLM server (default)
    python generate_completions.py --model_path /path/to/model --output_dir results --n_completions 8

    # OpenAI GPT models
    python generate_completions.py --model_type openai --model_path gpt-4o --output_dir results \\
        --n_completions 4 --api_key your_openai_key

    # xAI Groq-4 models
    python generate_completions.py --model_type xai --model_path groq-4 --output_dir results \\
        --n_completions 4 --api_key your_xai_key

    # Google Gemini models
    python generate_completions.py --model_type google --model_path gemini-1.5-pro --output_dir results \\
        --n_completions 4 --api_key your_google_key

    # Anthropic Claude models  
    python generate_completions.py --model_type anthropic --model_path claude-3-5-sonnet-20241022 \\
        --output_dir results --n_completions 4 --api_key your_anthropic_key

    # Local HuggingFace models
    python generate_completions.py --model_type huggingface --model_path /path/to/model \\
        --output_dir results --n_completions 8

    # Specific tasks only (works with any model type)
    python generate_completions.py --model_type openai --model_path gpt-4o --output_dir results \\
        --n_completions 4 --tasks description_to_q --api_key your_key

    # With thinking mode and context
    python generate_completions.py --model_path /path/to/model --output_dir results \\
        --n_completions 8 --thinking_mode --include_context

    # Ground truth (returns actual solutions for testing)
    python generate_completions.py --model_type ground_truth --model_path dummy \\
        --output_dir results --n_completions 1
        """
    )
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Model identifier: path for local models, model name for API models")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for completions")
    parser.add_argument("--n_completions", type=int, required=True,
                       help="Number of completions to generate per problem")
    
    # Model type and API arguments
    parser.add_argument("--model_type", type=str, default="vllm_api",
                       choices=['openai', 'anthropic', 'vllm_api', 'huggingface', 'ground_truth', 'xai', 'google'],
                       help="Type of model provider (default: vllm_api)")
    parser.add_argument("--api_key", type=str,
                       help="API key for OpenAI, Anthropic, xAI or Google models (or use OPENAI_API_KEY/ANTHROPIC_API_KEY/XAI_API_KEY/GEMINI_API_KEY env vars)")
    
    # Optional arguments
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
                       help="Use reasoning format with <reasoning> and <answer> tags")
    
    # Server arguments  
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Server port (default: 8000)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization (default: 0.9)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                       help="Maximum model context length (default: 4096)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size for multi-GPU (default: 1)")
    
    # Worker arguments
    parser.add_argument("--max_workers", type=int, default=6,
                       help="Maximum number of worker threads (default: 6 - conservative for vLLM stability)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load context if requested
    context = None
    if args.include_context:
        context_file = "Q_Examples.md"
        context = load_file_content(str(context_file))
        logger.info("Loaded Q reference context")
    
    # Load test problems
    logger.info(f"Loading test problems from {args.test_dir}")
    problems = load_test_problems(args.test_dir, args.max_problems, args.tasks)
    if not problems:
        logger.error("No test problems found!")
        return 1

    # Server management is only needed for vLLM API
    server_manager = None
    server_started_here = False
    api_base = None
    
    def signal_handler(signum, frame):
        logger.info("\nReceived interrupt signal. Cleaning up...")
        if server_started_here and server_manager:
            server_manager.stop_server()
        sys.exit(1)
    
    # Set up signal handler for all model types
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.model_type == "vllm_api":
        # Create server manager
        server_manager = VLLMServerManager(args.model_path, args.host, args.port)
        api_base = server_manager.api_base
        
    try:
        # Start server if using vLLM API and not already running
        if args.model_type == "vllm_api":
            if not server_manager.is_server_running():
                if not server_manager.start_server(
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    tensor_parallel_size=args.tensor_parallel_size
                ):
                    logger.error("✗ Failed to start vLLM server")
                    return 1
                server_started_here = True

        # Create completion generator
        model_name_for_api = "model" if args.model_type == "vllm_api" else args.model_path
        generator = CompletionGenerator(
            args.model_type, model_name_for_api, 
            api_base=api_base,
            api_key=args.api_key,
            test_dir=args.test_dir, 
            max_workers=args.max_workers
        )
        
        logger.info(f"🚀 Starting completion generation")
        logger.info(f"🤖 Model Type: {args.model_type}")
        logger.info(f"📊 Model: {args.model_path}")
        logger.info(f"📁 Output: {output_dir}")
        logger.info(f"🔢 Problems: {len(problems)}")
        logger.info(f"⚡ Completions per problem: {args.n_completions}")
        logger.info(f"👥 Max workers: {args.max_workers}")
        logger.info(f"🤔 Thinking mode: {args.thinking_mode}")
        logger.info(f"📖 Include context: {args.include_context}")
        if args.api_key:
            logger.info(f"🔑 API Key: {'*' * 8} (provided)")
        print()
        
        start_time = time.time()
        
        # Generate completions
        generator.generate_completions_for_problems(
            problems, args.n_completions, output_dir, context, args.thinking_mode
        )
        
        total_time = time.time() - start_time
        
        # Verify completion
        total_expected = len(problems) * args.n_completions
        total_generated = 0
        
        completions_dir = output_dir / "completions"
        for problem in problems:
            problem_id = get_problem_id(problem)
            problem_dir = completions_dir / problem_id
            if problem_dir.exists():
                ext = get_file_extension(problem['task_type'])
                completions = list(problem_dir.glob(f"completion_*{ext}"))
                total_generated += len(completions)
        
        logger.info(f"\n🎉 Generation completed!")
        logger.info(f"📊 Generated: {total_generated}/{total_expected} completions")
        logger.info(f"⏱️ Total time: {total_time/60:.1f}m")
        logger.info(f"⚡ Rate: {total_generated/total_time:.1f} completions/sec")
        logger.info(f"📁 Completions saved to: {completions_dir}")
        
        if total_generated == total_expected:
            logger.info("✅ All completions generated successfully!")
        else:
            logger.warning(f"⚠️ {total_expected - total_generated} completions missing")
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Error during generation: {e}")
        return 1
    finally:
        # Clean up server if we started it and using vLLM API
        if server_started_here and args.model_type == "vllm_api":
            server_manager.stop_server()


if __name__ == "__main__":
    sys.exit(main()) 