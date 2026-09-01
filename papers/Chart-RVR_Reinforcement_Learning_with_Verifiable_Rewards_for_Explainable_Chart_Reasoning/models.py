import os
import pathlib
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torch.nn.parallel import DataParallel
import torch.nn.functional as F

os.environ["FLASH_ATTENTION_2_ENABLED"] = "1"

from datasets import Dataset, load_dataset, load_from_disk

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, PaliGemmaForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, Qwen2VLProcessor,Qwen2_5_VLForConditionalGeneration
from transformers import pipeline

from transformers import DonutProcessor, VisionEncoderDecoderModel

from transformers import pipeline

import argparse
from tqdm import tqdm
import os
import json
import numpy as np
# from openai import OpenAI
import warnings
import pickle
from undecorated import undecorated
from types import MethodType
from PIL import Image




# Caching
cache_dir = '/mnt/data/sanchit/hf'
os.environ['HF_HUB_CACHE'] = '/mnt/data/sanchit/hf'
os.environ['TRANSFORMERS_CACHE']= '/mnt/data/sanchit/hf'
os.environ['HF_HOME'] = '/mnt/data/sanchit/hf'



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vlm_model(model_type, mode = "eval"):
    # For smaller models (CLIP based)
   
    if model_type == "pali-3b":
        # model_name = "google/paligemma2-3b-mix-448"
        model_name = "ahmed-masry/chartgemma"
        processor = AutoProcessor.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)

    if model_type == "pali-10b":
        model_name = "google/paligemma2-10b-mix-448"
        processor = AutoProcessor.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)
        model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir)


    # For medium models (below 8b)
    if model_type == "llava-1.6":
        model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name, device_map="auto",cache_dir = cache_dir, attn_implementation = "flash_attention_2", torch_dtype=torch.bfloat16)
    if model_type == "qwen-7b":
        processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto",cache_dir = cache_dir, attn_implementation = "flash_attention_2", torch_dtype="auto")
    if model_type == "qwen-2b":
        # device map = "auto" gives wrong device error on qwen2vl-2b #TODO
        processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, cache_dir = cache_dir, padding_side="left", use_fast=True)
        if mode == "grpo":
            model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map=None, revision='main', trust_remote_code=True, cache_dir = cache_dir)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto",revision='main', attn_implementation = "flash_attention_2", trust_remote_code=True, cache_dir = cache_dir)

    if model_type == "qwen2-5-3b":
        processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left", trust_remote_code=True, cache_dir = cache_dir)
        if mode == "grpo":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map=None, revision='main', trust_remote_code=True, cache_dir = cache_dir)
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto",revision='main', attn_implementation = "flash_attention_2", trust_remote_code=True, cache_dir = cache_dir)

    if model_type == "qwen2-2b-base":
        processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B", padding_side="left", trust_remote_code=True, cache_dir = cache_dir)
        if mode == "grpo":
            model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B", torch_dtype="auto", device_map=None, revision='main', trust_remote_code=True, cache_dir = cache_dir)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B", torch_dtype="auto", device_map="auto",revision='main', attn_implementation = "flash_attention_2", trust_remote_code=True, cache_dir = cache_dir)

    # if model_type == "qwen2-2b-base":
    #     processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B", padding_side="left", trust_remote_code=True, cache_dir = cache_dir)
    #     if mode == "grpo":
    #         model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B", torch_dtype="auto", device_map=None, revision='main', trust_remote_code=True, cache_dir = cache_dir)
    #     else:
    #         model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B", torch_dtype="auto", device_map="auto",revision='main', attn_implementation = "flash_attention_2", trust_remote_code=True, cache_dir = cache_dir)


    if model_type == "qwen2-5-7b":
        processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", padding_side="left", trust_remote_code=True, cache_dir = cache_dir)
        if mode == "grpo":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map=None, revision='main', trust_remote_code=True, cache_dir = cache_dir)
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto",revision='main', attn_implementation = "flash_attention_2", trust_remote_code=True, cache_dir = cache_dir)

    if model_type == "internvl-8b":
        processor = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2_5-8B", device_map="auto",cache_dir = cache_dir, trust_remote_code=True,torch_dtype=torch.bfloat16)
        model = AutoModel.from_pretrained("OpenGVLab/InternVL2_5-8B", device_map="auto",cache_dir = cache_dir, trust_remote_code=True,torch_dtype=torch.bfloat16)



    # Chart specific models
    if model_type == "unichart-chartqa":
        model_name = "ahmed-masry/unichart-chartqa-960"
        model = VisionEncoderDecoderModel.from_pretrained(model_name).cuda()
        processor = DonutProcessor.from_pretrained(model_name)
        # default_cfg = model.config_class()
        # for k, v in default_cfg.to_dict().items():
        #     model.config.__dict__.setdefault(k, v)
        out_dir = pathlib.Path(cache_dir) / model_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        model.save_pretrained(out_dir, safe_serialization=True)   # .safetensors + new config
        processor.save_pretrained(out_dir)
        model = VisionEncoderDecoderModel.from_pretrained(out_dir, torch_dtype="auto")
        processor = AutoTokenizer.from_pretrained(out_dir)


    if model_type == "chart-gemma":
        model_type = "ahmed-masry/chartgemma"
        model = AutoModelForImageTextToText.from_pretrained(model_type, device_map="auto",cache_dir = cache_dir)
        processor = AutoProcessor.from_pretrained(model_type)
    


    # OCR models (pipelines)
    if model_type == "generic-ocr":
        pipe = pipeline("image-text-to-text", model="Qwen/Qwen2-VL-2B-Instruct")
        return pipe

    # model.eval()
    # torch.cuda.empty_cache()

    return model, processor




    
    