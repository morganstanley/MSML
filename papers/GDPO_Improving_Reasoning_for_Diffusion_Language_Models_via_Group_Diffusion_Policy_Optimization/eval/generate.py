import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
import os


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    save_trajectories=False,
    trajectory_dir=None,
    trajectory_batch_idx=None,
):
    """
    Optimized version of the generate function.
    
    Args:
        save_trajectories: If True, save trajectory data at each step for animation
        trajectory_dir: Directory to save trajectory files (only used if save_trajectories=True)
        trajectory_batch_idx: Batch index for naming trajectory files (only used if save_trajectories=True)
    """
    # Initialize trajectory storage if needed
    trajectories = None
    if save_trajectories and dist.get_rank() == 0:
        trajectories = {
            'token_sequences': [],  # List of x states at each step
            'mask_states': [],      # List of mask_index at each step
            'step_info': [],        # List of (block_num, step_in_block) tuples
            'prompt': prompt.cpu().clone(),  # Save original prompt
            'mask_id': mask_id,
            'gen_length': gen_length,
            'steps': steps,
            'block_length': block_length,
            'num_blocks': gen_length // block_length,
            'steps_per_block': max(1, steps // (gen_length // block_length)),
        }
    
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        
        # Save initial state
        if save_trajectories and dist.get_rank() == 0:
            trajectories['token_sequences'].append(x.cpu().clone())
            trajectories['mask_states'].append((x == mask_id).cpu().clone())
            trajectories['step_info'].append((-1, -1))  # Initial state
        
        for num_block in tqdm(range(num_blocks), disable=(dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # Get logits in a single forward pass
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]
                
                # Save trajectory state after each step
                if save_trajectories and dist.get_rank() == 0:
                    trajectories['token_sequences'].append(x.cpu().clone())
                    trajectories['mask_states'].append((x == mask_id).cpu().clone())
                    trajectories['step_info'].append((num_block, i))
        
        # Save trajectories to disk if requested
        if save_trajectories and dist.get_rank() == 0 and trajectory_dir is not None:
            os.makedirs(trajectory_dir, exist_ok=True)
            if trajectory_batch_idx is not None:
                filename = os.path.join(trajectory_dir, f"trajectory_batch_{trajectory_batch_idx}_rank_{dist.get_rank()}.pt")
            else:
                filename = os.path.join(trajectory_dir, f"trajectory_rank_{dist.get_rank()}.pt")
            torch.save(trajectories, filename)
        
        return x
