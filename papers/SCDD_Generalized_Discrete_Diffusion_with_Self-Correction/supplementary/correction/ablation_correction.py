"""Experiment 1: No-Correction Ablation.

Compare gen-ppl of SCDD with corrections enabled vs disabled.
Shows that corrections are net-beneficial for generation quality.
Reports per-batch gen-ppl with mean ± standard error.
"""

import argparse
import json
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
    return model, tokenizer


def generate_and_eval(model, tokenizer, num_steps, num_batches, label):
    """Generate num_batches of samples. Compute per-batch gen-ppl and
    return mean, stderr, per-batch values, and all text."""
    all_text = []
    per_batch_ppl = []

    for b in range(num_batches):
        print(f"  [{label}] Batch {b+1}/{num_batches}")
        samples = model.restore_model_and_sample(num_steps=num_steps)
        batch_text = []
        for row in samples:
            batch_text.append(tokenizer.batch_decode(row.unsqueeze(0))[0])
        all_text.extend(batch_text)

        # Compute gen-ppl for this batch independently
        model.gen_ppl_metric.reset()
        model.compute_generative_perplexity(batch_text)
        ppl = model.gen_ppl_metric.compute().cpu().item()
        per_batch_ppl.append(ppl)
        print(f"    batch gen-ppl: {ppl:.4f}")

    mean_ppl = sum(per_batch_ppl) / len(per_batch_ppl)
    std_ppl = (sum((p - mean_ppl) ** 2 for p in per_batch_ppl)
               / len(per_batch_ppl)) ** 0.5
    stderr_ppl = std_ppl / math.sqrt(len(per_batch_ppl))

    return mean_ppl, stderr_ppl, per_batch_ppl, all_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--nucleus_p", type=float, default=0.9)
    parser.add_argument("--output_path", type=str, default="ablation_correction.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint_path}")
    model, tokenizer = load_model(args.checkpoint_path, device)
    print(f"Model: {model.config.model.name}, parameterization={model.parameterization}")

    model.config.loader.eval_batch_size = args.batch_size
    model.config.model.length = args.seq_len
    model.config.sampling.nucleus_p = args.nucleus_p

    print(f"Config: num_steps={args.num_steps}, batch_size={args.batch_size}, "
          f"seq_len={args.seq_len}, nucleus_p={args.nucleus_p}, "
          f"num_batches={args.num_batches}\n")

    # --- Variant 1: corrections enabled (default) ---
    print("=== SCDD (full, with corrections) ===")
    model._disable_corrections = False
    mean_full, se_full, batches_full, text_full = generate_and_eval(
        model, tokenizer, args.num_steps, args.num_batches, "full")
    print(f"  Gen-PPL: {mean_full:.2f} ± {se_full:.2f}\n")

    # --- Variant 2: corrections disabled ---
    print("=== SCDD (no corrections) ===")
    model._disable_corrections = True
    mean_nocorr, se_nocorr, batches_nocorr, text_nocorr = generate_and_eval(
        model, tokenizer, args.num_steps, args.num_batches, "no-corr")
    print(f"  Gen-PPL: {mean_nocorr:.2f} ± {se_nocorr:.2f}\n")

    # --- Summary ---
    print("=" * 70)
    print(f"{'Variant':<30} {'Gen-PPL (GPT-2-large)':>30}")
    print("-" * 70)
    print(f"{'SCDD (full)':<30} {mean_full:>20.2f} ± {se_full:.2f}")
    print(f"{'SCDD (no corrections)':<30} {mean_nocorr:>20.2f} ± {se_nocorr:.2f}")
    print("=" * 70)

    results = {
        "config": {
            "checkpoint_path": args.checkpoint_path,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "nucleus_p": args.nucleus_p,
            "num_batches": args.num_batches,
            "total_samples": args.num_batches * args.batch_size,
        },
        "results": {
            "full": {
                "gen_ppl_mean": mean_full,
                "gen_ppl_stderr": se_full,
                "gen_ppl_per_batch": batches_full,
                "text_samples": text_full,
            },
            "no_corrections": {
                "gen_ppl_mean": mean_nocorr,
                "gen_ppl_stderr": se_nocorr,
                "gen_ppl_per_batch": batches_nocorr,
                "text_samples": text_nocorr,
            },
        },
    }

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
