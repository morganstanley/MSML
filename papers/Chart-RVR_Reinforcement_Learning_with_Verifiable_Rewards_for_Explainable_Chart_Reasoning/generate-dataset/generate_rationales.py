import os
import re
import time
import glob
import tqdm
import torch
import requests
import pickle
import numpy as np
import json
from PIL import Image
from PIL import Image as PILImage
import queue
import random
import base64
import io
from openai import OpenAI
import warnings
import argparse
from PIL import Image

from datasets import Dataset, load_dataset, load_from_disk, Image, DatasetDict, concatenate_datasets
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, PaliGemmaForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, Qwen2VLProcessor,Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from torch.utils.data import DataLoader

from dataset_process import ChartDataset, PlotQADataset, ChartToTextDataset

seed = 2026
random.seed(seed)

from datasets import Dataset, load_dataset, load_from_disk, Image, DatasetDict, concatenate_datasets

warnings.filterwarnings("ignore")


cache_dir = '/mnt/data/sanchit/hf'
os.environ['HF_HUB_CACHE'] = '/mnt/data/sanchit/hf'
os.environ['TRANSFORMERS_CACHE']= '/mnt/data/sanchit/hf'
os.environ['HF_HOME'] = '/mnt/data/sanchit/hf'



def format_prompt(question, answer): 
    prefix = ('''Format:\
                ### Question: {question} \
                ### Reasoning:  \
                step-1: <reasoning step 1> \ 
                step-2: <reasoning step 2> \ 
                ... \
                step-n: <reasoning step n>\
                ### Answer: {answer}''')
    
    return prefix.format(question=question, answer=answer)

def call_chatgpt(question, answer, img=None, temp=0.7):       
    if img:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        b64 = base64.b64encode(png_bytes).decode("ascii")

        content = [
        {"role": "system", "content": '''You are helping me answer questions on charts. \
                                      You have to look both at the chart picture and the question. \
                                      Think step by step in as many small steps as required to answer the question based on the chart.\
                                      Lastly, also try to predict the type of chart out of the following:\
                                      "bar", "pie", "grouped_bar", "histogram","line","scatter","bubble",\
                                      "line_dual_axis","scatter_matrix","stacked_bar","multi_line" \
                                      Format:\
                                      ### Question: <question> \
                                      ### Reasoning:  \
                                      step-1: <reasoning step 1> \ 
                                      step-2: <reasoning step 2> \ 
                                      ... \
                                      step-n: <reasoning step n>\
                                      ### Answer: <answer> \
                                      ### Type: <type of chart> \
                                       '''},
        {"role": "user", "content": [{"type":"text","text":format_prompt(question,answer)}, {"type": "image_url","image_url": {"url": f"data:image/png;base64,{b64}",}}]},
        ]
    else:
        print("No image found!!")
        exit(0)

    client = OpenAI(
    api_key="")
    
    try:
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=content,
        temperature=temp)
    except:
        print("Error")
        return "Error in response - rerun"

    
    while True:
        try:
            ans = response.choices[0].message.content
            return ans
        except:
            continue

def call_qwen(model, processor, sample):
    system = '''You are helping me answer questions on charts. \
                You have to look both at the chart picture and the question. \
                The question and the answer will be provided to you. \
                First you have to recover the table data from the chart image in JSON format.\
                For the chart image, output only a JSON object with: 
                "columns": list of column headers, 
                "rows": list-of-lists, one per data row
                No prose, no comments.
                1. Respond with **only** a JSON object inside a ```json code fence.
                2. The JSON must use exactly this schema:
                    {
                        "columns": [...],
                        "rows": [...]
                    }
                3. Do NOT output HTML, Markdown, or commentary. Any deviation gets zero reward.
                Next, think step by step in as many small steps as required to answer the question based on the chart.\
                Lastly, also predict the type of chart out of the following:\
                "line", "bar", "stacked bar", "pie", "histogram", "scatterplot", "area", "stacked area", "bubble", "treemap" \         
                Format:\
                ### Question: <question> \
                ### Answer: <answer> \
                ### Table: <json table> \
                ### Reasoning:  \
                <step-1>: Provide a description of reasoning
                <step-2>: Gather ALL the appropriate data from the chart
                <step-3>: Break down the query into smaller parts and verify each part with the data
                ...
                <step-n>: Do the final calculation or reasoning to derive the answer
                <step-n+1>: VERIFY the final answer is correct for no halluciantions
                ### Type: <type of chart> \
                '''
    # system = '''You are helping me answer questions on charts. \
    #             You have to look both at the chart picture and the question. \
    #             The question and the answer will be provided to you. \
    #             You have to recover the table data from the chart image in JSON format.\
    #             For the chart image, output only a JSON object with: 

    #             "columns": list of column headers, 
    #             "rows": list-of-lists, one per data row
    #             No prose, no comments.

    #             1. Respond with **only** a JSON object inside a ```json code fence.
    #             2. The JSON must use exactly this schema:
    #                 {
    #                     "columns": [...],
    #                     "rows": [...]
    #                 }
    #             3. Do NOT output HTML, Markdown, or commentary. Any deviation gets zero reward.

    #             Format:\
    #             ### Question: <question> \
    #             ### Answer: <answer> \
    #             ### Reasoning:  \
    #             step-1: <reasoning step 1> \ 
    #             step-2: <reasoning step 2> \ 
    #             ... \
    #             step-n: <reasoning step n>\
    #             ### Type: <type of chart> \
    #             '''
    
    instruction = ('''
            ### Question: {question} \
            ### Answer: {answer} \
             ''')
    
    suffix = """Be concise and precise. Do not add any additional information."""
    
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
                            {"type": "text", "text": instruction.format(question=sample[1][i], answer=sample[2][i]) + "\n" + suffix} 
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
    generated_ids = model.generate(**inputs, max_new_tokens=800)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    out = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
        
    return out

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
            image_batch.append(batch[idx]['image'])
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
            image_batch.append(batch[idx]['image'])
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
            image_batch.append(batch[idx]['image'])
            query_batch.append(batch[idx]['question_string'])
            label_batch.append(batch[idx]['answer'])
            type_batch.append(batch[idx]['type'])

        return image_batch, query_batch, label_batch, type_batch


def resize_fit(img):
    MIN_PIXELS = 320 * 28 * 28            # 1 003 520
    # MAX_PIXELS = 16384 * 28 * 28           # 12 843 776
    MAX_PIXELS = 640 * 28 * 28           # 12 843 776
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

    method = "qwen"
    if method == "qwen":
        from transformers import pipeline
        # "Qwen/Qwen2.5-VL-72B-Instruct"

        processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", padding_side="left", trust_remote_code=True, cache_dir = cache_dir)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto" ,revision='main', attn_implementation = "flash_attention_2", trust_remote_code=True, cache_dir = cache_dir)

    # train_dataset = load_from_disk(str(cache_dir+"/chartqa_dataset_src"))["train"]
    # dpath = str(cache_dir+"/grpo-custom-dataset-"+str(seed))
    # dpath = str(cache_dir+"/grpo-custom-dataset-large-"+str(seed))
    # train_dataset = load_from_disk(dpath)

    rationales = []
    image_list = [] # to mantatain the order of images

    # breakpoint()
    for dataname in ["chartbench"]:
        dataset = ChartDataset(dataname, processor=processor)
        train_dataset = dataset.load_chart_dataset(split = "test").shuffle(seed=seed).select(range(5000))
        # dataset = PlotQADataset("plotqa", processor=processor)
        # test_dataset = dataset.load_plotqa_dataset(split = "test")
        # test_dataset = test_dataset.shuffle(seed=seed)
        # train_dataset = test_dataset.select(range(5000))

        train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn = collate_fn, shuffle = False)


        for sample in tqdm.tqdm(train_dataloader):
            if method == "chatgpt":
                out = call_chatgpt(sample["query"], sample["label"], sample["image"]) 
            if method == "qwen":
                out = call_qwen(model, processor, sample)
            # print(out)
            rationales.extend(out)
            image_list.extend(sample[0])
            # with open("./rationales_llm/json_tables_qwen72b_chartqa.json", "w") as f:
            #     json.dump(rationales, f, indent=4)

            # with open("./rationales_llm/rationales-"+str(seed)+".json", "w") as f:
            #     json.dump(rationales, f, indent=4)

            # with open("./rationales_llm/rationales-large-"+str(seed)+".json", "w") as f:
            #     json.dump(rationales, f, indent=4)

            with open("./gt_classify/"+dataname+"-"+str(seed)+".json", "w") as f:
                json.dump(rationales, f, indent=4)

            
    
    # with open("./rationales_llm/image_list_rationales-"+str(seed)+".pkl", "wb") as f:
    #     pickle.dump(image_list, f)
        


main()