import os
import json
from tqdm import tqdm
import torch
import pandas as pd
import argparse
from qwen_omni_utils import process_mm_info
from peft import PeftModel
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

DTYPE = torch.bfloat16

def load_model_and_processor(model_id, lora_path=None, gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    base_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_id, dtype=DTYPE, device_map=device)

    if lora_path:
        model = PeftModel.from_pretrained(base_model, lora_path, device_map=device)
    else:
        model = base_model

    print(f"Initialized model: {model.name_or_path}")

    model.eval()
    # speed/memory hints
    try:
        model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass

    model.config.use_cache = False
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    return model, processor

def build_conversation(processor, prompt, audio_path):
    system_template = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
    ]

    user_template = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt},
        ],
    }]

    conversation_template = system_template + user_template

    prompt_str = processor.apply_chat_template(conversation_template, tokenize=False, add_generation_prompt=True)

    audios, _, _ = process_mm_info(conversation_template, use_audio_in_video=False)

    return {
        "prompt": prompt_str,
        "audios": audios
    }

def build_conversation_mcq(processor, question, choices, audio_path):
    joined_choices = "\n".join([f"- {choice}" for choice in choices])

    prompt = f"""You are solving a multiple-choice question about an audio clip.

You will be given:
1. A QUESTION about what is happening in the audio.
2. A list of CHOICES.

Your task:
- Carefully listen to the audio.
- Read the QUESTION and CHOICES.
- Choose the single best answer.
- **Important**: Reply with ONLY the exact words and phrases in the choices.
Do NOT include any explanation or extra text.

QUESTION:
{question}

CHOICES:
{joined_choices}

Your answer:
"""

    system_template = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
    ]

    user_template = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt},
        ],
    }]

    conversation_template = system_template + user_template

    prompt_str = processor.apply_chat_template(conversation_template, tokenize=False, add_generation_prompt=True)

    audios, _, _ = process_mm_info(conversation_template, use_audio_in_video=False)

    return {
        "prompt": prompt_str,
        "audios": audios
    }

def get_model_output(model, processor, conversation):
    prompt = conversation["prompt"]
    audios = conversation["audios"]

    inputs = processor(text=[prompt], audio=audios, return_tensors="pt", padding=True).to(model.device).to(model.dtype)

    out = model.generate(
        **inputs,
        max_new_tokens=256,
        use_audio_in_video=False,
        output_scores=True,
        return_dict_in_generate=True,
        stop_strings=["<|im_end|>", "Human:", "User:"],
        tokenizer=processor.tokenizer,
        temperature=0.0,
        top_k=1,
    )
    generate_ids = out["sequences"]
    generate_ids = generate_ids[:, inputs.input_ids.shape[1]:]
    pred = processor.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    return pred

def inference(audio_root, test_json, output, lora=False, lora_path=None, gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-Omni-7B"

    if lora:
        model, processor = load_model_and_processor(model_id, lora_path=lora_path, gpu_id=gpu_id)
    else:
        model, processor = load_model_and_processor(model_id, lora_path=None, gpu_id=gpu_id)

    sr = processor.feature_extractor.sampling_rate

    with open(test_json, 'r') as f:
        samples = json.load(f)

    outputs = []

    for sample in tqdm(samples):
        audio_path = os.path.join(audio_root, sample["audio"] + ".wav")
        prompt = sample['prompt']
        conversation = build_conversation(processor, prompt, audio_path)
        pred = get_model_output(model, processor, conversation)
        sample['model_prediction'] = pred
        outputs.append(sample)

    with open(output, 'w') as fout:
        json.dump(outputs, fout, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', action='store_true', help='Use LoRA weights if set')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--audio_root', type=str, required=True, help='Root directory of audio files')
    parser.add_argument('--test_json', type=str, required=True, help='Path to test JSON file')
    parser.add_argument('--lora_path', type=str, default=None, help='Path to LoRA weights')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')

    args = parser.parse_args()

    inference(args.audio_root, args.test_json, args.output, lora=args.lora, lora_path=args.lora_path, gpu_id=args.gpu_id)

