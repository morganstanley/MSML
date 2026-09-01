import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from generate import generate
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from countdown import CTDDataset
from sudoku import SudokuDataset
from mbpp import MBPPDataset
from humaneval import HumanEvalDataset

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
    "mbpp": MBPPDataset,
    "humaneval": HumanEvalDataset,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def evaluate(
    model,
    tokenizer,
    dataloader,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    dataset='gsm8k',
    save_trajectories=False,
    trajectory_dir=None,
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    batch_idx = 0
    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        if dataset in ['mbpp', 'humaneval']:
            test_lists = batch["test_list"]

        out = generate(
            model,
            input_ids,
            tokenizer,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
            save_trajectories=save_trajectories,
            trajectory_dir=trajectory_dir,
            trajectory_batch_idx=batch_idx,
        )

        generated_texts = tokenizer.batch_decode(out[:, -gen_length:], skip_special_tokens=False)
        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
                "test_list": test_lists[j] if dataset in ['mbpp', 'humaneval'] else None
            }
            for j in range(len(gt_answers))
        ]
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)
        batch_idx += 1

        # Print individual results
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth:")
            print(f"{gt_answers[idx]}")

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If we don't drop the last batch, we need to calculate the number of samples per rank.
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    init_seed(42)

    # Note: This evaluation script saves only model generations. A separate parser is used later to extract
    # predictions and calculate metrics.

    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset", type=str, choices=["gsm8k", "math", "countdown", "sudoku", "game24", "mbpp", "humaneval"], default="gsm8k"
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")
    parser.add_argument("--save_trajectories", action="store_true",
                        help="Save trajectory data at each generation step for animation")
    parser.add_argument("--trajectory_dir", type=str, default=None,
                        help="Directory to save trajectory files (default: output_dir/trajectories)")
    args = parser.parse_args()

    args.diffusion_steps = args.gen_length
    num_evals = {"gsm8k": -1, "math": -1, "countdown": 256, "sudoku": 256, "mbpp": -1, "humaneval": -1}

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
        local_rank
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.checkpoint_path:
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(
            local_rank
        )

        if dist.get_world_size() > 1:
            dist.barrier()  # Make sure all processes are ready
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            print(f"Rank {local_rank}: Parameters synchronized")

    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=True,  # prefill for all models
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    if len(args.checkpoint_path):
        model_name = args.checkpoint_path.split("/")
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        model_name = "instruct" if "Instruct" in args.model_path else "base"

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}_{args.diffusion_steps}_{dist.get_rank()}_generations.json"
    print(f"Saving generations to {filename}")
    
    # Set up trajectory directory
    if args.save_trajectories:
        if args.trajectory_dir is None:
            trajectory_dir = os.path.join(args.output_dir, "trajectories")
        else:
            trajectory_dir = args.trajectory_dir
        if dist.get_rank() == 0:
            print(f"Saving trajectories to {trajectory_dir}")
    else:
        trajectory_dir = None

    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        dataset=args.dataset,
        save_trajectories=args.save_trajectories,
        trajectory_dir=trajectory_dir,
    )

    if not args.dont_save:
        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                    },
                    "model_path": args.model_path,
                    "checkpoint_path": args.checkpoint_path,
                    "gen_length": args.gen_length,
                    "diffusion_steps": args.diffusion_steps,
                    "block_length": args.block_length,
                },
                f,
                indent=2,
            )

    cleanup_ddp()
