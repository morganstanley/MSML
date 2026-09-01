import os
import tqdm
import re
import time
import pickle
import random
import argparse
import json
import sys
from PIL import Image as PILImage
import io

os.environ["FLASH_ATTENTION_2_ENABLED"] = "1"

import torch
import numpy as np
from transformers import BitsAndBytesConfig
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer
import wandb
import sacrebleu

from qwen_vl_utils import process_vision_info

from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, PaliGemmaForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, Qwen2VLProcessor,Qwen2_5_VLForConditionalGeneration
import logging
from torch.utils.data import DataLoader


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import load_vlm_model
from dataset_process import ChartDataset, PlotQADataset, ChartToTextDataset
from metrics import exact_match, relaxed_accuracy
from utils import get_vlm_output, clear_memory, format_data, select_icl_samples




seed = 2026


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

cache_dir = '/mnt/data/sanchit/hf'
os.environ['HF_HUB_CACHE'] = '/mnt/data/sanchit/hf'
os.environ['TRANSFORMERS_CACHE']= '/mnt/data/sanchit/hf'
os.environ['HF_HOME'] = '/mnt/data/sanchit/hf'

def set_all_seeds(seed: int = 2025) -> None:
    """Sets the random seed for PyTorch, NumPy, and Python's random module."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # For current GPU
        torch.cuda.manual_seed_all(seed) # For all GPUs
    
    # When running on the CuDNN backend, two further options must be set for full determinism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed to ensure consistent hashing behavior
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


set_all_seeds(seed)


def load_dataset_by_name(dataset_name, processor):
    if dataset_name == "chartqa-src":
        dataset = ChartDataset("chartqa-src", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")
    if dataset_name == "chartfc":
        dataset = ChartDataset("chartfc", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")

    elif dataset_name == "chartqapro":
        dataset = ChartDataset("chartqapro", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")

    elif dataset_name == "plotqa":
        dataset = PlotQADataset("plotqa", processor=processor)
        test_dataset = dataset.load_plotqa_dataset(split = "test")
        test_dataset = test_dataset.shuffle(seed=seed)
        test_dataset = test_dataset.select(range(5000))

    elif dataset_name == "figqa":
        dataset = ChartDataset("figqa", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")

    elif dataset_name == "charttotext":
        dataset = ChartToTextDataset("chart2text", processor=processor)
        test_dataset = dataset.load(split = "test")

    elif dataset_name == "evochart":
        dataset = ChartDataset("evochart", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")

    elif dataset_name == "chartllama":
        dataset = ChartDataset("chartllama", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")

    elif dataset_name == "chartbench":
        dataset = ChartDataset("chartbench", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test").shuffle(seed=seed).select(range(5000))

    elif dataset_name == "chartx":
        dataset = ChartDataset("chartx", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")

    return test_dataset
    
    # Load the model and processor


def call_qwen(model, processor, sample):
    system = ('''You are a vision assistant answering questions about a chart.\
               Follow the reasoning given and based on that answer in a few words.''')
    instruction = ('''Reasoning: {reasoning} Question: {question} Answer:''')
    
    
    payload = []

    for i in range(len(sample[0])):
        payload.append(
            [
                {
                "role": "system",
                "content": system
                },
                {
                "role": "user",
                "content": [
                            {"type": "image", "image": sample[0][i]}, 
                            {"type": "text", "text": instruction.format(question=sample[1][i], reasoning=sample[3][i]) + "\n"} 
                        ]
                }
            ])
        
    text = processor.apply_chat_template(payload, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(payload)
    inputs = processor(
        text=text,
        images=[image_inputs],
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated = model(**inputs, output_hidden_states=False, return_dict=True, max_new_tokens=10)
    return generated.logits[0][-1]

def collate_fn(batch):
        # for quick evals
        image_batch = []
        query_batch = []
        label_batch = []
        
        for idx in range(len(batch)):
            image_batch.append(resize_fit(batch[idx]['image']))
            query_batch.append(batch[idx]['query'])
            label_batch.append(batch[idx]['label'])

        return image_batch, query_batch, label_batch

def collate_fn_plotqa(batch):
        # for quick evals
        image_batch = []
        query_batch = []
        label_batch = []
        type_batch = []
        
        for idx in range(len(batch)):
            image_batch.append(resize_fit(batch[idx]['image']))
            query_batch.append(batch[idx]['question_string'])
            label_batch.append(batch[idx]['answer'])
            type_batch.append(batch[idx]['type'])

        return image_batch, query_batch, label_batch, type_batch


def collate_fn_chartqapro(batch):
        # for quick evals
        image_batch = []
        query_batch = []
        label_batch = []
        question_type_batch = []
        
        for idx in range(len(batch)):
            image_batch.append(resize_fit(PILImage.open(io.BytesIO(batch[idx]['image'])).convert("RGB")))
            query_batch.append(batch[idx]['Question'])
            label_batch.append(batch[idx]['Answer'][0])
            question_type_batch.append(batch[idx]['Question Type'])
        return image_batch, query_batch, label_batch

def collate_fn_chartfc(batch):
        # for quick evals
        image_batch = []
        query_batch = []
        label_batch = []
        machine_human_batch = []
        
        for idx in range(len(batch)):
            image_batch.append(resize_fit(batch[idx]['image']))
            query_batch.append(batch[idx]['question']+ ". Answer yes/no.")
            label_batch.append(batch[idx]['label'])
            machine_human_batch.append(None)

        return image_batch, query_batch, label_batch, machine_human_batch

def collate_fn_plotqa(batch):
        # for quick evals
        image_batch = []
        query_batch = []
        label_batch = []
        type_batch = []
        
        for idx in range(len(batch)):
            image_batch.append(resize_fit(batch[idx]['image']))
            query_batch.append(batch[idx]['question_string'])
            label_batch.append(batch[idx]['answer'])
            type_batch.append(batch[idx]['type'])

        return image_batch, query_batch, label_batch, type_batch


def resize_fit(img):
    MIN_PIXELS = 320 * 28 * 28            # 1 003 520
    # MAX_PIXELS = 16384 * 28 * 28           # 12 843 776
    MAX_PIXELS = 320 * 28 * 28           # 12 843 776
    # MAX_PIXELS = 960 * 28 * 28           # 12 843 776

    img = img.convert("RGB")
    w, h = img.size
    p = w * h
    if MIN_PIXELS <= p <= MAX_PIXELS:
        return img
    tgt_p  = max(min(p, MAX_PIXELS), MIN_PIXELS)
    scale  = (tgt_p / p) ** 0.5
    new_wh = (int(w * scale), int(h * scale))
    return img.resize(new_wh, PILImage.BICUBIC)

def main():
    processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", padding_side="left", trust_remote_code=True, cache_dir = cache_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto" ,revision='main', attn_implementation = "flash_attention_2", trust_remote_code=True, cache_dir = cache_dir)


    # dsets = ["chartqa-src", "plotqa", "chartfc", "evochart", "chartqapro", "chartbench"]
    # dsets = ["chartqapro", "chartbench"]
    dsets = ["chartqa-src"]


    for dset in dsets:
        print("Processing dataset:", dset)
        test_dataset = load_dataset_by_name(dset, processor)
        if dset in ["plotqa"]:
            test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn = collate_fn_plotqa, shuffle = False)
        elif dset in ["chartqa-src","evochart","chartbench"]:
            test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn = collate_fn, shuffle = False)
        elif dset in ["chartqapro"]:
            test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn = collate_fn_chartqapro, shuffle = False)
        elif dset in ["chartfc"]:
            test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn = collate_fn_chartfc, shuffle = False)

        with open("gt_classify/"+dset+"-2026.json", "r") as f:
            gt_info = json.load(f)

        with open("logs/cot-hard-"+dset+".out", "r") as f:
            cot_info = f.read()

        with open("logs/grpo-hard-"+dset+".out", "r") as f:
            grpo_info = f.read()

        with open("logs/sft-hard-"+dset+".out", "r") as f:
            sft_info = f.read()

        cot_info = cot_info.split("#####")
        grpo_info = grpo_info.split("#####")
        sft_info = sft_info.split("#####")


        entity = 0

        deltas = {"cot":[], "grpo":[], "sft":[]}

        for batch in tqdm.tqdm(test_dataloader):
            if entity < 1250:
                entity += 1
                continue
            index = processor.tokenizer.encode(batch[2][0].split(" ")[0], add_special_tokens=False)[0]
            sample = [batch[0],batch[1],batch[2],[""]]
            output = call_qwen(model, processor, sample)
            # print("No rationales:", torch.softmax(output, dim=-1)[index].item())
            nr = torch.softmax(output, dim=-1)[index].item()
            
            # Capture GT
            # gt_rsn = gt_info[entity].split("### Reasoning: ")[-1].strip().split("### Type:")[0].strip()

            # Capture COT
            cot_rsn = cot_info[entity].split("</table>")[-1].strip()
            sample = [batch[0],batch[1],batch[2],[cot_rsn]]
            output = call_qwen(model, processor, sample)
            # print("GRPO:", torch.softmax(output, dim=-1)[index].item())
            cot_p = torch.softmax(output, dim=-1)[index].item()

            # Capture GRPO
            grpo_rsn = grpo_info[entity].split("</table>")[-1].strip()
            sample = [batch[0],batch[1],batch[2],[grpo_rsn]]
            output = call_qwen(model, processor, sample)
            # print("GRPO:", torch.softmax(output, dim=-1)[index].item())
            grpo_p = torch.softmax(output, dim=-1)[index].item()

            # Capture SFT
            sft_rsn = sft_info[entity].split("</table>")[-1].strip()
            sample = [batch[0],batch[1],batch[2],[sft_rsn]]
            output = call_qwen(model, processor, sample)
            # print("SFT:", torch.softmax(output, dim=-1)[index].item())
            sft_p = torch.softmax(output, dim=-1)[index].item()

            deltas["cot"].append(cot_p - nr)
            deltas["grpo"].append(grpo_p - nr)
            deltas["sft"].append(sft_p - nr)

            # print("COT Delta", cot_p - nr, "GRPO Delta", grpo_p - nr, "SFT Delta", sft_p - nr)

            entity += 1

            # if entity < 1250:
            #     continue


        print("Dataset:", dset, "COT Mean Delta:", np.mean(deltas["cot"]), "GRPO Mean Delta:", np.mean(deltas["grpo"]), "SFT Mean Delta:", np.mean(deltas["sft"]))
    
main()