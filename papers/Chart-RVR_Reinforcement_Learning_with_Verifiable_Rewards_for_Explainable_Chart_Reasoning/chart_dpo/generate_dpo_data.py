import os
import torch
import pickle
import json
from tqdm import tqdm

from datasets import Dataset

from models import load_vlm_model
from dataset_process import ChartDataset


import random

from metrics import exact_match, relaxed_accuracy
from utils import get_vlm_output, clear_memory, format_data

model, processor = load_vlm_model("qwen-7b")

system_message = """You are an OCR engine.

Extract every visible character **inside the plotted area** exactly as it appears.

1. Preserve original line breaks and the reading order (top-to-bottom, left-to-right).  
2. **Ignore** chart titles, axis names, legends, gridlines, and any decorative text.  
3. Focus only on numbers, data labels, and in-chart annotations.  
4. Return **only** the raw text – no summaries, no markdown, no extra tokens."""

def run_vlm_ocr(ocr_model, image):
    messages = []
    for img in image:
        messages.append([
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img.resize((200, int(img.height * (200 / img.width))))},
                ]
            },
        ])
    out = ocr_model(text=messages, max_new_tokens=10)
    # breakpoint()
    # out = out[]
    out = [ott[0]['generated_text'][2]['content'].strip().split() for ott in out]
    return out


def single_inference():
    with open("./pref-data/base-qwen-7b-new-run", "rb") as f:
        pref_data = pickle.load(f)


    ocr_pipe = load_vlm_model("generic-ocr")

    formatted_data = []
    for row in tqdm(pref_data):
        row = [{'image':row[0],'query':row[1],'label':row[2], 'human_or_machine':row[3]}, row[-1]]
        img = row[0]['image']
        query = row[0]['query']
        label = row[0]['label'][0]
        human = row[0]['human_or_machine']
        pred = row[1]
        formatted_data.append({
            # "image": Image.open(img),
            "image": img,
            "question": query,
            "label": label,
            "human_or_machine": human,
            "predicted": pred
        })
        all_entity = run_vlm_ocr(ocr_pipe, row[0]['image'])
        for entity in all_entity:
            if entity.lower() != row[0]['label'][0].lower():
                img = row[0]['image']
                query = row[0]['query']
                label = row[0]['label'][0]
                human = row[0]['human_or_machine']
                pred = entity
                formatted_data.append({
                    # "image": Image.open(img),
                    "image": img,
                    "question": query,
                    "label": label,
                    "human_or_machine": human,
                    "predicted": pred
            })
    # Create Hugging Face Dataset
    # logging.info("Pref Data Processed")
    # breakpoint()
    # pickle.dump(formatted_data, open("./pref-data/base-qwen-7b-dataset-hardneg.pkl", "wb"))
    exit(0)

    # dataset = pickle.load(open("./pref-data/base-llava-1.6-sft-dataset-hardneg.pkl", "rb"))
    formatted_data = pickle.load(open("./pref-data/base-qwen-7b-dataset-hardneg.pkl", "rb"))
    breakpoint()
    dataset = Dataset.from_list(formatted_data)
    pref_dataset = dataset.map(pref_format, num_proc=32)
    pref_dataset.save_to_disk("./pref-data/base-qwen-7b-hf-dset-hard-negatives")


def pref_format(example):
    # Prepare the input for the chat template
    prompt = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": example["question"]}],
        },
    ]
    chosen = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["label"]}],
        },
    ]
    rejected = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["predicted"]}],
        },
    ]
    # Apply the chat template
    prompt = processor.apply_chat_template(prompt, tokenize=False)
    chosen = processor.apply_chat_template(chosen, tokenize=False)
    rejected = processor.apply_chat_template(rejected, tokenize=False)
    # Resize the image to ensure it fits within the maximum allowable
    # size of the processor to prevent OOM errors.
    # max_size = self.processor.image_processor.size["longest_edge"]
    # example["image"].thumbnail((max_size, max_size))
    return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}


def remove_special(text):
    """Remove special characters from the text."""
    # Remove any non-alphanumeric characters except spaces
    return ''.join(char for char in text if char.isalnum() or char.isspace()).strip()

def flush_batch(batch_imgs, batch_rows, ocr_pipe):
    """Run OCR on the queued images and populate `formatted_data`."""
    if not batch_imgs:
        return
    # 1. OCR inference (batched)
    batch_entities = run_vlm_ocr(ocr_pipe, batch_imgs)  # List[List[str]]
    formatted_data = []
    # 2. Post-process each item in the batch
    for row, entities in zip(batch_rows, batch_entities):
        img_path   = row[0]['image']
        query      = row[0]['query']
        gt_label   = row[0]['label'][0]
        human_flag = row[0]['human_or_machine']
        pref_pred  = row[1]

        # original preference prediction
        formatted_data.append({
            "image": img_path,
            "question": query,
            "label": gt_label,
            "human_or_machine": human_flag,
            "predicted": pref_pred,
        })

        # OCR-derived negatives
        for ent in entities:
            if ent.lower() != gt_label.lower():
                formatted_data.append({
                    "image": img_path,
                    "question": query,
                    "label": gt_label,
                    "human_or_machine": human_flag,
                    "predicted": remove_special(ent),
                })

    return formatted_data

    # breakpoint()

def main():
    with open("./pref-data/base-qwen-7b-new-run", "rb") as f:
        pref_data = pickle.load(f)


    ocr_pipe  = load_vlm_model("generic-ocr")
    BATCH_SIZE = 512                     # ← tune for your hardware

    batch_imgs, batch_rows = [], []

    formatted_data = []  # Store all formatted data

    for row in tqdm(pref_data, desc="Batched OCR"):
        row = [{'image':row[0],'query':row[1],'label':row[2], 'human_or_machine':row[3]}, row[-1]]
        batch_imgs.append(row[0]['image'])
        batch_rows.append(row)

        if len(batch_imgs) == BATCH_SIZE:
            formatted_data.extend(flush_batch(batch_imgs, batch_rows, ocr_pipe))
            batch_imgs, batch_rows = [], []

    # flush the final, possibly incomplete batch
    flush_batch(batch_imgs, batch_rows, ocr_pipe)
    print(len(formatted_data))

    breakpoint()
    dataset = Dataset.from_list(formatted_data)
    pref_dataset = dataset.map(pref_format, num_proc=32)
    pref_dataset.save_to_disk("./pref-data/base-qwen-7b-hf-dset-hard-negatives")


main()