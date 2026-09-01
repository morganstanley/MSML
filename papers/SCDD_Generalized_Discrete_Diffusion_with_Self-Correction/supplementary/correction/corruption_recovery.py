"""Experiment 2: Corruption Recovery at Last Step.

Corrupt K tokens in clean validation text, run one SCDD denoising step
at the last-step noise level, and measure recovery/touch/damage rates.
Runs multiple batches and reports mean ± standard error.
Saves clean, corrupted, and corrected text for reviewer inspection.
"""

import argparse
import json
import itertools
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import dataloader
import diffusion as diffusion_module


def load_model(ckpt_path, device):
    import tokenizers as _tk
    _orig = _tk.Tokenizer.__setstate__
    _tk.Tokenizer.__setstate__ = lambda self, state: None
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    _tk.Tokenizer.__setstate__ = _orig

    config = checkpoint["hyper_parameters"]["config"]
    tokenizer = dataloader.get_tokenizer(config)

    model = diffusion_module.Diffusion(config, tokenizer=tokenizer)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    if model.ema and 'ema' in checkpoint:
        model.ema.load_state_dict(checkpoint['ema'])
        model.ema.move_shadow_params_to_device(device)
        print("EMA weights loaded.")
    elif model.ema:
        print("WARNING: EMA configured but not found in checkpoint.")

    model.to(device)
    return model, tokenizer, config


def load_clean_batches(config, tokenizer, seq_len, batch_size, num_batches,
                       data_cache_dir=None):
    """Load multiple batches of clean token sequences from validation set."""
    from omegaconf import OmegaConf, open_dict

    # Create a minimal copy that avoids unresolvable interpolations
    with open_dict(config):
        config.loader.eval_batch_size = batch_size
        config.loader.eval_global_batch_size = batch_size
        config.loader.batch_size = batch_size
        config.loader.global_batch_size = batch_size
        config.trainer.num_nodes = 1
        config.trainer.accumulate_grad_batches = 1
        config.trainer.devices = 1
        config.model.length = seq_len
        config.loader.num_workers = 4
        config.loader.pin_memory = True
        if data_cache_dir is not None:
            config.data.cache_dir = data_cache_dir

    _, valid_loader = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=42)

    batches = []
    for i, batch in enumerate(valid_loader):
        if i >= num_batches:
            break
        batches.append(batch["input_ids"][:, :seq_len])

    return batches


def corrupt_tokens(clean_ids, num_corrupt, vocab_size, mask_index):
    """Randomly corrupt num_corrupt tokens per sequence.

    Returns:
        corrupted_ids: (B, L) tensor with corrupted tokens
        corrupt_mask: (B, L) boolean tensor, True at corrupted positions
    """
    B, L = clean_ids.shape
    corrupted_ids = clean_ids.clone()
    corrupt_mask = torch.zeros(B, L, dtype=torch.bool)

    for i in range(B):
        positions = torch.randperm(L)[:num_corrupt]
        corrupt_mask[i, positions] = True
        random_tokens = torch.randint(0, vocab_size - 1, (num_corrupt,))
        random_tokens[random_tokens >= mask_index] += 1
        for j, pos in enumerate(positions):
            while random_tokens[j] == clean_ids[i, pos]:
                random_tokens[j] = torch.randint(0, vocab_size - 1, (1,)).item()
                if random_tokens[j] >= mask_index:
                    random_tokens[j] += 1
            corrupted_ids[i, pos] = random_tokens[j]

    return corrupted_ids, corrupt_mask


def run_last_step(model, corrupted_ids, num_steps, eps=1e-5):
    """Run one SCDD denoising step at the last-step noise level,
    then apply noise_removal (argmax)."""
    device = model.device
    x = corrupted_ids.to(device)

    timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
    dt = (1 - eps) / num_steps

    t = timesteps[num_steps - 1] * torch.ones(x.shape[0], 1, device=device)

    if model.ema:
        model.ema.store(itertools.chain(
            model.backbone.parameters(), model.noise.parameters()))
        model.ema.copy_to(itertools.chain(
            model.backbone.parameters(), model.noise.parameters()))

    model.backbone.eval()
    model.noise.eval()

    with torch.no_grad():
        x = model._scdd_update(x, t, dt)

        if model.config.sampling.noise_removal:
            t_final = timesteps[-1] * torch.ones(
                x.shape[0], 1, device=device)
            unet_conditioning = model.noise(t_final)[0]
            x = model.forward(x, unet_conditioning).argmax(dim=-1)

    if model.ema:
        model.ema.restore(itertools.chain(
            model.backbone.parameters(), model.noise.parameters()))

    model.backbone.train()
    model.noise.train()

    return x


def compute_metrics(clean_ids, corrupted_ids, corrected_ids, corrupt_mask):
    """Compute touch, recovery, and damage rates."""
    touched_at_corrupt = (corrected_ids != corrupted_ids) & corrupt_mask
    recovered = (corrected_ids == clean_ids) & corrupt_mask
    total_corrupt = corrupt_mask.sum().item()

    touch_rate = touched_at_corrupt.sum().item() / max(total_corrupt, 1)
    recovery_rate = recovered.sum().item() / max(total_corrupt, 1)

    uncorrupt_mask = ~corrupt_mask
    damaged = (corrected_ids != clean_ids) & uncorrupt_mask
    total_uncorrupt = uncorrupt_mask.sum().item()
    damage_rate = damaged.sum().item() / max(total_uncorrupt, 1)

    return {
        "touch_rate": touch_rate,
        "exact_recovery_rate": recovery_rate,
        "damage_rate": damage_rate,
        "total_corrupt": total_corrupt,
        "total_touched": touched_at_corrupt.sum().item(),
        "total_recovered": recovered.sum().item(),
        "total_damaged": damaged.sum().item(),
    }, touched_at_corrupt, recovered


def mean_stderr(values):
    n = len(values)
    m = sum(values) / n
    std = (sum((v - m) ** 2 for v in values) / n) ** 0.5
    se = std / math.sqrt(n)
    return m, se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--corrupt_counts", type=str, default="5,10,20,50")
    parser.add_argument("--nucleus_p", type=float, default=0.9)
    parser.add_argument("--data_cache_dir", type=str, default=None,
                        help="Override data.cache_dir in checkpoint config")
    parser.add_argument("--output_path", type=str,
                        default="corruption_recovery.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint_path}")
    model, tokenizer, config = load_model(args.checkpoint_path, device)
    print(f"Model: {model.config.model.name}, "
          f"parameterization={model.parameterization}")

    model.config.sampling.nucleus_p = args.nucleus_p
    model.config.model.length = args.seq_len

    corrupt_counts = [int(c) for c in args.corrupt_counts.split(",")]

    print(f"Config: num_steps={args.num_steps}, batch_size={args.batch_size}, "
          f"seq_len={args.seq_len}, nucleus_p={args.nucleus_p}, "
          f"num_batches={args.num_batches}")
    print(f"Corruption counts: {corrupt_counts}\n")

    # Load multiple batches of clean validation sequences
    print("Loading clean validation sequences...")
    clean_batches = load_clean_batches(
        config, tokenizer, args.seq_len, args.batch_size, args.num_batches,
        data_cache_dir=args.data_cache_dir)
    print(f"Loaded {len(clean_batches)} batches of {clean_batches[0].shape[0]} "
          f"sequences (length {clean_batches[0].shape[1]})\n")

    results = {
        "config": {
            "checkpoint_path": args.checkpoint_path,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "nucleus_p": args.nucleus_p,
            "num_batches": args.num_batches,
            "corrupt_counts": corrupt_counts,
        },
        "results": [],
    }

    for K in corrupt_counts:
        print(f"=== Corruption count K={K} ===")

        batch_touch_rates = []
        batch_recovery_rates = []
        batch_damage_rates = []
        batch_ppl_clean = []
        batch_ppl_corrupted = []
        batch_ppl_corrected = []
        all_samples = []

        for b_idx, clean_ids in enumerate(clean_batches):
            clean_ids = clean_ids.to(device)
            print(f"  Batch {b_idx+1}/{len(clean_batches)}")

            # Decode clean text
            clean_texts = [tokenizer.decode(row.cpu().tolist())
                           for row in clean_ids]

            # Corrupt tokens
            corrupted_ids, corrupt_mask = corrupt_tokens(
                clean_ids.cpu(), K, model.vocab_size, model.mask_index)
            corrupted_ids = corrupted_ids.to(device)
            corrupt_mask = corrupt_mask.to(device)

            corrupted_texts = [tokenizer.decode(row.cpu().tolist())
                               for row in corrupted_ids]

            # Run last-step correction
            corrected_ids = run_last_step(
                model, corrupted_ids, args.num_steps)

            corrected_texts = [tokenizer.decode(row.cpu().tolist())
                               for row in corrected_ids]

            # Compute metrics for this batch
            metrics, touched_mask, recovered_mask = compute_metrics(
                clean_ids, corrupted_ids, corrected_ids, corrupt_mask)

            batch_touch_rates.append(metrics['touch_rate'])
            batch_recovery_rates.append(metrics['exact_recovery_rate'])
            batch_damage_rates.append(metrics['damage_rate'])

            # Per-batch perplexities
            model.gen_ppl_metric.reset()
            model.compute_generative_perplexity(clean_texts)
            ppl_c = model.gen_ppl_metric.compute().cpu().item()
            batch_ppl_clean.append(ppl_c)

            model.gen_ppl_metric.reset()
            model.compute_generative_perplexity(corrupted_texts)
            ppl_cr = model.gen_ppl_metric.compute().cpu().item()
            batch_ppl_corrupted.append(ppl_cr)

            model.gen_ppl_metric.reset()
            model.compute_generative_perplexity(corrected_texts)
            ppl_co = model.gen_ppl_metric.compute().cpu().item()
            batch_ppl_corrected.append(ppl_co)

            print(f"    touch={metrics['touch_rate']:.4f}  "
                  f"recov={metrics['exact_recovery_rate']:.4f}  "
                  f"damage={metrics['damage_rate']:.4f}  "
                  f"ppl_clean={ppl_c:.2f}  "
                  f"ppl_corrupt={ppl_cr:.2f}  "
                  f"ppl_correct={ppl_co:.2f}")

            # Per-sample records
            for i in range(clean_ids.shape[0]):
                corrupt_positions = corrupt_mask[i].nonzero(
                    as_tuple=True)[0].cpu().tolist()
                touched_positions = touched_mask[i].nonzero(
                    as_tuple=True)[0].cpu().tolist()
                recovered_positions = recovered_mask[i].nonzero(
                    as_tuple=True)[0].cpu().tolist()
                all_samples.append({
                    "batch": b_idx,
                    "clean_text": clean_texts[i],
                    "corrupted_text": corrupted_texts[i],
                    "corrected_text": corrected_texts[i],
                    "corrupted_positions": corrupt_positions,
                    "touched_positions": touched_positions,
                    "recovered_positions": recovered_positions,
                })

        # Aggregate across batches
        m_touch, se_touch = mean_stderr(batch_touch_rates)
        m_recov, se_recov = mean_stderr(batch_recovery_rates)
        m_damage, se_damage = mean_stderr(batch_damage_rates)
        m_ppl_clean, se_ppl_clean = mean_stderr(batch_ppl_clean)
        m_ppl_corrupt, se_ppl_corrupt = mean_stderr(batch_ppl_corrupted)
        m_ppl_correct, se_ppl_correct = mean_stderr(batch_ppl_corrected)

        print(f"\n  K={K} summary (mean ± stderr over {len(clean_batches)} batches):")
        print(f"    Touch rate:    {m_touch:.4f} ± {se_touch:.4f}")
        print(f"    Recovery rate: {m_recov:.4f} ± {se_recov:.4f}")
        print(f"    Damage rate:   {m_damage:.4f} ± {se_damage:.4f}")
        print(f"    PPL clean:     {m_ppl_clean:.2f} ± {se_ppl_clean:.2f}")
        print(f"    PPL corrupted: {m_ppl_corrupt:.2f} ± {se_ppl_corrupt:.2f}")
        print(f"    PPL corrected: {m_ppl_correct:.2f} ± {se_ppl_correct:.2f}")
        print()

        results["results"].append({
            "corrupt_count": K,
            "touch_rate_mean": m_touch,
            "touch_rate_stderr": se_touch,
            "touch_rate_per_batch": batch_touch_rates,
            "exact_recovery_rate_mean": m_recov,
            "exact_recovery_rate_stderr": se_recov,
            "exact_recovery_rate_per_batch": batch_recovery_rates,
            "damage_rate_mean": m_damage,
            "damage_rate_stderr": se_damage,
            "damage_rate_per_batch": batch_damage_rates,
            "ppl_clean_mean": m_ppl_clean,
            "ppl_clean_stderr": se_ppl_clean,
            "ppl_clean_per_batch": batch_ppl_clean,
            "ppl_corrupted_mean": m_ppl_corrupt,
            "ppl_corrupted_stderr": se_ppl_corrupt,
            "ppl_corrupted_per_batch": batch_ppl_corrupted,
            "ppl_corrected_mean": m_ppl_correct,
            "ppl_corrected_stderr": se_ppl_correct,
            "ppl_corrected_per_batch": batch_ppl_corrected,
            "samples": all_samples,
        })

    # Summary table
    print("=" * 120)
    print(f"{'K':>5}  {'Touch':>14}  {'Recovery':>14}  {'Damage':>14}  "
          f"{'PPL clean':>14}  {'PPL corrupt':>14}  {'PPL correct':>14}")
    print("-" * 120)
    for r in results["results"]:
        print(f"{r['corrupt_count']:>5}  "
              f"{r['touch_rate_mean']:>6.4f}±{r['touch_rate_stderr']:.4f}  "
              f"{r['exact_recovery_rate_mean']:>6.4f}±{r['exact_recovery_rate_stderr']:.4f}  "
              f"{r['damage_rate_mean']:>6.4f}±{r['damage_rate_stderr']:.4f}  "
              f"{r['ppl_clean_mean']:>6.2f}±{r['ppl_clean_stderr']:.2f}  "
              f"{r['ppl_corrupted_mean']:>6.2f}±{r['ppl_corrupted_stderr']:.2f}  "
              f"{r['ppl_corrected_mean']:>6.2f}±{r['ppl_corrected_stderr']:.2f}")
    print("=" * 120)

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
