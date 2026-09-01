import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import modeling_utils

try:
    from transformers.pytorch_utils import apply_chunking_to_forward as _apply_chunking_to_forward

    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = _apply_chunking_to_forward
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single-image VQA inference demo with LLaDA-MedV."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about the image.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "XZDong123/LLaDA-MedV"),
        help="Model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--llada-v-codebase",
        default=os.environ.get("LLADA_V_CODEBASE"),
        help="Path to the LLaDA-V / llava codebase.",
    )
    parser.add_argument(
        "--conv-mode",
        default=os.environ.get("CONV_MODE", "llava_llada"),
        help="Conversation template name.",
    )
    parser.add_argument(
        "--vision-tower",
        default=os.environ.get("VISION_TOWER", "google/siglip2-so400m-patch14-384"),
        help="Vision tower override.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        help="Attention implementation.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=os.environ.get("TORCH_DTYPE", "bfloat16"),
        choices=["bfloat16", "float16"],
        help="Torch dtype for model and image tensors.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("STEPS", "128")),
        help="Number of generation steps.",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=int(os.environ.get("GEN_LENGTH", "128")),
        help="Generation length.",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=int(os.environ.get("BLOCK_LENGTH", "128")),
        help="Block length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TEMPERATURE", "0.0")),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--prefix-refresh-interval",
        type=int,
        default=int(os.environ.get("PREFIX_REFRESH_INTERVAL", "32")),
        help="Prefix refresh interval.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("THRESHOLD", "1.0")),
        help="Generation threshold.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cuda"),
        help="Torch device to use, for example cuda or cpu.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the inference result as JSON.",
    )
    return parser.parse_args()


def ensure_codebase_on_path(codebase):
    if not codebase:
        raise RuntimeError(
            "Please provide --llada-v-codebase or set LLADA_V_CODEBASE so llava can be imported."
        )

    codebase_path = Path(codebase).expanduser().resolve()
    if not codebase_path.exists():
        raise FileNotFoundError(f"LLaDA-V codebase not found: {codebase_path}")

    codebase_str = str(codebase_path)
    if codebase_str not in sys.path:
        sys.path.insert(0, codebase_str)

    return codebase_path


def get_torch_dtype(dtype_name):
    if dtype_name in (torch.bfloat16, torch.float16):
        return dtype_name
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def build_question(question):
    question = question.strip()
    return question + " Answer concisely in English."


def load_llava_modules():
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    return {
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "conv_templates": conv_templates,
        "get_model_name_from_path": get_model_name_from_path,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "load_pretrained_model": load_pretrained_model,
        "disable_torch_init": disable_torch_init,
    }


def load_model_components(args, llava_modules):
    llava_modules["disable_torch_init"]()
    model_name = "llava_llada"

    tokenizer, model, image_processor, _ = llava_modules["load_pretrained_model"](
        args.model_path,
        None,
        model_name,
        device_map="auto" if args.device == "cuda" else None,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        overwrite_config={"mm_vision_tower": args.vision_tower},
    )
    return tokenizer, model, image_processor, llava_modules["get_model_name_from_path"](args.model_path)


def build_prompt(question, conv_mode, llava_modules):
    conv = llava_modules["conv_templates"][conv_mode].copy()
    conv.append_message(
        conv.roles[0],
        llava_modules["DEFAULT_IMAGE_TOKEN"] + "\n" + question,
    )
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def run_inference(args, tokenizer, model, image_processor, model_name, llava_modules):
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = llava_modules["process_images"]([image], image_processor, model.config)
    model_dtype = get_torch_dtype(args.torch_dtype)
    image_tensor = [_image.to(dtype=model_dtype, device=args.device) for _image in image_tensor]
    image_sizes = [image.size]

    prompt = build_prompt(build_question(args.question), args.conv_mode, llava_modules)
    input_ids = llava_modules["tokenizer_image_token"](
        prompt,
        tokenizer,
        llava_modules["IMAGE_TOKEN_INDEX"],
        return_tensors="pt",
    ).unsqueeze(0).to(args.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            tokenizer=tokenizer,
            stopping_criteria=["<|eot_id|>"],
            prefix_refresh_interval=args.prefix_refresh_interval,
            threshold=args.threshold,
            temperature=args.temperature,
        )

    pred_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    result = {
        "image": str(image_path),
        "question": args.question,
        "pred_answer": pred_answer,
        "model_id": model_name,
        "model_path": args.model_path,
    }

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved output JSON to: {output_path}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    args.torch_dtype = get_torch_dtype(args.torch_dtype)
    ensure_codebase_on_path(args.llada_v_codebase)
    llava_modules = load_llava_modules()
    tokenizer, model, image_processor, model_name = load_model_components(args, llava_modules)
    run_inference(args, tokenizer, model, image_processor, model_name, llava_modules)


if __name__ == "__main__":
    main()
