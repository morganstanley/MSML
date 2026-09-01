import argparse
import os
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def main():
    """
    Main function to convert a Hugging Face model to a .ckpt file for evaluation.
    """
    parser = argparse.ArgumentParser(description="Convert a Hugging Face DiDi-Instruct model back to a .ckpt file.")
    parser.add_argument("--hf_repo_id", type=str, required=True, help="The repository ID on the Hugging Face Hub (e.g., 'haoyangzheng/didi-instruct-model').")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory to save the downloaded files and the final .ckpt file.")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_dir)
    if not output_dir: # Handle case where path is just a filename in the current directory
        output_dir = "."

    print(f"Preparing output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading model files from '{args.hf_repo_id}' to '{output_dir}'...")

    generator_weights_path = hf_hub_download(
        repo_id=args.hf_repo_id,
        filename="model.safetensors",
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )
    discriminator_weights_path = hf_hub_download(
        repo_id=args.hf_repo_id,
        filename="discriminator.safetensors",
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )

    print("Loading weights from downloaded files...")
    generator_state_dict = load_file(generator_weights_path)
    discriminator_state_dict = load_file(discriminator_weights_path)

    print("Reconstructing the combined state_dict with correct prefixes...")
    combined_state_dict = {}

    for key, value in generator_state_dict.items():
        combined_state_dict[f"student_ema.{key}"] = value
    print(f"  - Added {len(generator_state_dict)} keys with 'student_ema.' prefix.")

    for key, value in discriminator_state_dict.items():
        combined_state_dict[f"discriminator.{key}"] = value
    print(f"  - Added {len(discriminator_state_dict)} keys with 'discriminator.' prefix.")

    print("Assembling the final PyTorch Lightning checkpoint structure...")
    final_checkpoint = {
        'epoch': 0,
        'global_step': 0,
        'pytorch-lightning_version': '2.5.3',
        'state_dict': combined_state_dict,
    }

    ckpt_save_path = os.path.join(args.output_dir, "didi-instruct-from-hf.ckpt")
    print(f"Saving the checkpoint to: {args.output_dir}")
    torch.save(final_checkpoint, args.output_dir)

    print("\nConversion complete!")
    print(f"Downloaded intermediate files are in: '{output_dir}'")
    print(f"The final checkpoint is saved at: '{args.output_dir}'")


if __name__ == "__main__":
    main()
    