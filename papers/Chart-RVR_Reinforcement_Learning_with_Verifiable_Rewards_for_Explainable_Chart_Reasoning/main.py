import os
import tqdm
import re
import time
import pickle
import random
import argparse
import json
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



from models import load_vlm_model
from dataset_process import ChartDataset, PlotQADataset, ChartToTextDataset


from metrics import exact_match, relaxed_accuracy
from utils import get_vlm_output, clear_memory, format_data, select_icl_samples

import logging, os


seed = 2026


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

cache_dir = '/mnt/data/sanchit/hf'
os.environ['HF_HUB_CACHE'] = '/mnt/data/sanchit/hf'
os.environ['TRANSFORMERS_CACHE']= '/mnt/data/sanchit/hf'
os.environ['HF_HOME'] = '/mnt/data/sanchit/hf'

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for VLM evaluation pipeline")

    parser.add_argument('--mode', type=str, choices=["eval", "sft", "dpo", "ppo", "grpo"], required=True, help='Run mode')
    parser.add_argument('--vlm-name', type=str, required=True, help='Name of the vision-language model')
    parser.add_argument('--sft-lora', type=bool, required=False, default=False, help='Use LORA adapters for SFT and location')
    parser.add_argument('--dpo-lora', type=bool, required=False, default=False, help='Use LORA adapters for DPO and location')
    parser.add_argument('--grpo-lora', type=bool, required=False, default=False, help='Use LORA adapters for GRPO and location')
    parser.add_argument('--dataset-name', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--dataset-split', type=str, required=False, default = "test", help='Dataset split')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--cot', type=bool, default=False, help='To use CoT or not')
    parser.add_argument('--icl', type=bool, default=False, help='To use ICL examples for inference or not')
    parser.add_argument('--hard', type=bool, default=False, help='To use hard example trained models for inference')

    return parser.parse_args()

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


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # TODO: Load the model and processor for chart specific models (they dont work with batched inputs)
    model, processor = load_vlm_model(args.vlm_name, args.mode)
    logging.info("Loaded model and processor")

    if args.mode == "sft" or args.mode == "dpo" or args.mode == "grpo":
        # Set the PEFT 
        if "qwen" in args.vlm_name:
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            target_modules=["q_proj", "v_proj"]
        elif "intern" in args.vlm_name:
            target_modules=["wqkv", "wo", "w1", "w2", "w3"]
        else:
            target_modules = None # This targets all possible layers/modules (llava type models, maybe pali #TODO check)

    if args.dataset_name == "chartqa":
        dataset = ChartDataset("chartqa", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")
        if args.mode == "sft" or args.icl:
            train_dataset = dataset.load_chart_dataset(split = "train")
        if args.mode == "sft":
            eval_dataset = dataset.load_chart_dataset(split = "val") 

    if args.dataset_name == "chartqa-src":
        dataset = ChartDataset("chartqa-src", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")
        if args.mode in ["sft", "dpo", "grpo"] or args.icl:
            train_dataset = dataset.load_chart_dataset(split = "train")
        if args.mode in ["sft", "dpo", "grpo"]:
            eval_dataset = dataset.load_chart_dataset(split = "val") 

    if args.dataset_name == "chartfc":
        dataset = ChartDataset("chartfc", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")

    elif args.dataset_name == "chartqapro":
        dataset = ChartDataset("chartqapro", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")
        if args.icl:
            train_dataset = dataset.load_chart_dataset(split = "train")

    elif args.dataset_name == "plotqa":
        dataset = PlotQADataset("plotqa", processor=processor)
        test_dataset = dataset.load_plotqa_dataset(split = "test")
        test_dataset = test_dataset.shuffle(seed=seed)
        test_dataset = test_dataset.select(range(5000))

        if args.icl:
            train_dataset = dataset.load_plotqa_dataset(split = "train")
            train_dataset = train_dataset.shuffle(seed=seed)
            train_dataset = train_dataset.select(range(50000))

    elif args.dataset_name == "figqa":
        dataset = ChartDataset("figqa", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")
        if args.icl:
            train_dataset = dataset.load_chart_dataset(split = "train")

    elif args.dataset_name == "charttotext":
        dataset = ChartToTextDataset("chart2text", processor=processor)
        test_dataset = dataset.load(split = "test")
        if args.icl:
            train_dataset = dataset.load(split = "train")

    elif args.dataset_name == "evochart":
        dataset = ChartDataset("evochart", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")

    elif args.dataset_name == "chartllama":
        dataset = ChartDataset("chartllama", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")

    elif args.dataset_name == "chartbench":
        dataset = ChartDataset("chartbench", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test").shuffle(seed=seed).select(range(5000))

    elif args.dataset_name == "chartx":
        dataset = ChartDataset("chartx", processor=processor)
        test_dataset = dataset.load_chart_dataset(split = "test")
    
    # Load the model and processor
    if args.mode == "eval":
        blocks = 4 if args.cot else 1 # 1 is Direct Prompting, 2 is typical COT, 4 is COT, 4 is GRPO COT 
        logging.info("Blocks:", blocks)
        # SFT adapter
        if args.sft_lora:
            # sft_model_path = "/mnt/home/sanchit/Qwen2-VL-Finetune/output/sft-"+str(seed)+"/"
            sft_model_path = "/mnt/home/sanchit/Qwen2-VL-Finetune/output/q2-large-sft-"+str(seed)+"/"
            from transformers import  Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(sft_model_path, device_map="auto", local_files_only=True)
            model.eval()
            logging.info("Loaded model with SFT adapters from {}".format(sft_model_path))

        # DPO adapter
        if args.dpo_lora:
            # adapter_path = "qwen-7b-train-chartqa-dpo-no-sft-v1-hardneg-beta-0.3-test/checkpoint-7000"
            # adapter_path = "qwen-7b-train-chartqa-dpo-no-sft-v1-hardneg-beta-0.3-ipo/checkpoint-7000"
            # adapter_path = "qwen-7b-train-chartqa-dpo-no-sft-v1-hardneg-beta-0.3-hinge/checkpoint-7000"
            # adapter_path = "qwen-7b-train-chartqa-dpo-no-sft-v1-hardneg-beta-0.3-sppo_hard/checkpoint-7000"
            # adapter_path = "qwen-7b-train-chartqa-dpo-no-sft-v1-hardneg-beta-0.3-apo_zero/checkpoint-7000"
            # adapter_path = args.dpo_lora

            # adapter_path = "qwen-7b-train-chartqa-dpo-no-sft-v2-ocr-tabular-ocr-qwen-data-high-caphinge/checkpoint-15000"
            # adapter_path = "qwen-2b-train-chartqa-dpo-no-sft-v2-ocr-tabular-ocr-qwen-data-high-caphinge/checkpoint-8000"
            
            
            # adapter_path = "qwen-7b-train-chartqa-dpo-base-ocr-tabular-rationaleshinge/checkpoint-21018"
            # adapter_path = "qwen-7b-train-chartqa-dpo-base-ocr-tabular-rationales-unicharthinge/checkpoint-28500"

            # adapter_path = "qwen-7b-train-chartqa-dpo-no-sft-v1-tabularhinge/checkpoint-14490"

            # Data gen
            # adapter_path = "qwen-2b-train-chartqa-dpo-base-ocr-tabular-rationaleshinge/checkpoint-21018"
            # Unichart tables
            adapter_path = "qwen-2b-train-chartqa-dpo-base-ocr-tabular-rationales-unicharthinge/checkpoint-65000"
        
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

            logging.info("Loaded model with DPO adapters from {}".format(adapter_path))
        
        if args.grpo_lora:
            from transformers import  Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
            if not args.hard:
                grpo_path = "/mnt/home/sanchit/rl-chart/grpo-start-ckpts/qwen2-5-3b-prm-"+str(seed)+"/checkpoint-4000/"
            else:
                grpo_path = "/mnt/home/sanchit/rl-chart/grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-"+str(seed)+"/checkpoint-22000/"
            
            grpo_path = "/mnt/home/sanchit/rl-chart/grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-"+str(seed)+"/checkpoint-22000/"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(grpo_path, device_map="auto", local_files_only=True)
            model.eval()

            logging.info("Loaded model with GRPO adapters from {}".format(grpo_path))


        em_correct, ra_correct, tot_bleuscore, total = 0, 0, 0, 0

        aug_correct, aug_total = 0, 0
        human_correct, human_total = 0, 0


        wrongs = []
        pref_data = []
        
        logging.info("Starting evaluation on {} samples".format(len(test_dataset)))
        logging.info("Using CoT: {}".format(args.cot))


        loader = dataset.create_loader(test_dataset, bsz=1)

        if args.icl:
            path = "./icl-examples/"+args.dataset_name+"-icl_samples.pkl"
            if os.path.exists(path):
                icl_samples = pickle.load(open(path, 'rb')) # load in the ICL examples : list of [img, text, answer]
                logging.info("Loaded ICL examples from {}".format(path))
            else:
                icl_samples = select_icl_samples(train_dataset, k=5)
                logging.info("Selected ICL examples")
                pickle.dump(icl_samples, open(path, 'wb'))
        
        idx = 0
        for batch in tqdm.tqdm(loader):
            if args.icl:
                pred, rationale = get_vlm_output(model, processor, batch[0], batch[1], args.cot, icl_samples, blocks)
            else:
                if args.vlm_name in ["qwen-2b"]:
                    logging.info("Using explicit model device setting")
                    pred, rationale = get_vlm_output(model, processor, batch[0], batch[1], args.cot, model_device="cuda", blocks = blocks)
                else:
                    logging.info("Using automatic model device setting")
                    pred, rationale = get_vlm_output(model, processor, batch[0], batch[1], args.cot, blocks = blocks)
            # print(batch)
            if args.vlm_name in ["chart-gemma"]:
                print(pred[0])
                print(batch[2][0])
                print("#####")
            else:
                # print(pred)
                # print(batch[2])
                print(rationale[0])
                print("#####")
            # breakpoint()
            # TODO: Understand why this takes so long (and optimize)
            bleu_score = 0
            # for i in range(len(pred)):
            #     bleu_score += sacrebleu.corpus_bleu([batch[2][i]], [[pred[i]]]).score
            tot_bleuscore += bleu_score
            # Exact match
            em_correct += exact_match(batch[2], pred)
            # Realaxed accuracy
            if args.dataset_name == "chartqapro":
                ra = relaxed_accuracy(batch[2],pred, True)
            else:
                ra = relaxed_accuracy(batch[2],pred)

            if args.dataset_name == "chartqa" or args.dataset_name == "chartqa-src":
                # Augmented accuracy
                for i in range(len(batch[3])):
                    if batch[3][i]==1:
                        aug_total += 1
                        if ra[1][i]==1:
                            aug_correct += 1
                        

                # Human accuracy
                for i in range(len(batch[3])):
                    if batch[3][i]==0:
                        human_total += 1
                        if ra[1][i]==1:
                            human_correct += 1


            ra_correct += ra[0]

            total += len(batch[0])

            # print("EM: ", em_correct/total)
            # print("Relaxed Accuracy: ", ra_correct/total)
            
            # print("Aug Accuracy:", aug_correct/aug_total ) if aug_total>0 else print("Aug Accuracy: No aug samples")
            # print("Human Accuracy:", human_correct/human_total ) if human_total>0 else print("Human Accuracy: No human samples")
            # print("Average Accuracy:", 0.5*(aug_correct/aug_total + human_correct/human_total)) if aug_total>0 and human_total>0 else print("Average Accuracy: NA")

            # for img in batch[0]:
            #     img.save("./junkimage/"+str(idx)+".png")
            #     idx+=1
            # breakpoint()
            # print("BLEU Score: ", tot_bleuscore/total)
            # print(batch)
        
    if args.mode == "sft":
        os.environ["WANDB_CONSOLE"] = "wrap" 
        wandb.init(project="chartrl", entity="chartrl")

        logging.info("Loaded model, train and eval datasets")
        logging.info("Using:", model.config._attn_implementation)

        bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_use_double_quant=True, 
                    bnb_4bit_quant_type="nf4", 
                    bnb_4bit_compute_dtype=torch.bfloat16)
        
        
        peft_config = LoraConfig(
                    lora_alpha=256,
                    lora_dropout=0.05,
                    r=256,
                    bias="none",
                    target_modules = target_modules,
                    task_type="CAUSAL_LM",)


        peft_model = get_peft_model(model, peft_config)
        print(peft_model.print_trainable_parameters())

        logging.info("Loaded model+LORA adapters")

        training_args = SFTConfig(
            output_dir=args.vlm_name+"-sft-train"+args.dataset_name,  # Directory to save the model
            num_train_epochs=3,  # Number of training epochs
            per_device_train_batch_size=4,  # Batch size for training
            per_device_eval_batch_size=4,  # Batch size for evaluation
            gradient_accumulation_steps=4,  # Steps to accumulate gradients
            gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
            # Optimizer and scheduler settings
            lr_scheduler_type="linear",
            optim="adamw_torch_fused",  # Optimizer type
            learning_rate=5e-6,  # Learning rate for training
            # Logging and evaluation
            logging_steps=10,  # Steps interval for logging
            eval_steps=500,  # Steps interval for evaluation
            eval_strategy="steps",  # Strategy for evaluation
            save_strategy="steps",  # Strategy for saving the model
            save_steps=500,  # Steps interval for saving
            metric_for_best_model="eval_loss",  # Metric to evaluate the best model
            greater_is_better=False,  # Whether higher metric values are better
            load_best_model_at_end=True,  # Load the best model after training
            # Mixed precision and gradient settings
            bf16=True,  # Use bfloat16 precision
            tf32=True,  # Use TensorFloat-32 precision
            max_grad_norm=0.3,  # Maximum norm for gradient clipping
            # warmup_ratio=0.2,  # Ratio of total steps for warmup
            warmup_steps=1000,  # Number of warmup steps
            # Hub and reporting
            push_to_hub=False,  # Whether to push model to Hugging Face Hub
            report_to="wandb" ,  # Reporting tool for tracking metrics
            # Gradient checkpointing settings
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
            # Dataset configuration
            dataset_text_field="",  # Text field in dataset
            dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
            # max_seq_length=1024  # Maximum sequence length for input
        )

        training_args.remove_unused_columns = False  # Keep unused columns in dataset
        # training_args.dataloader_num_workers = 4  # Number of workers for data loading

        logging.info("Training arguments set up")

        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # data_collator=dataset.train_collate_fn_plotqa,
            data_collator=dataset.train_collate_fn_chartqa,
            # data_collator=dataset.train_collate_fn_charttotext,
            peft_config=peft_config,
            # tokenizer=processor.tokenizer,
            tokenizer=processor,

        )

        
        logging.info("SFT Trainer initialized")
        trainer.train()
        logging.info("Training completed, saving output model")
        trainer.save_model(training_args.output_dir)
        logging.info("Model saved to {}".format(training_args.output_dir))


    if args.mode == "dpo":
        os.environ["WANDB_CONSOLE"] = "wrap" 
        wandb.init(project="chartrl", entity="chartrl")

        if args.sft_lora:
            # adapter_path = "llava-1.6-train-plotqa-v0"
            # adapter_path = "/mnt/home/sanchit/Qwen2-VL-Finetune/output/chartqa_lora-v0"
            # adapter_path = "/mnt/home/sanchit/Qwen2-VL-Finetune/output/chartqa_lora-v0"
            # adapter_path = "/mnt/home/sanchit/Qwen2-VL-Finetune/output/chartqa_lora-v0"
            adapter_path = "llava-1.6-train-chartqa"
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            logging.info("Loaded model with SFT adapters from {}".format(adapter_path))

        pref_dataset = dataset.load_pref_data()
        eval_dataset = dataset.load_pref_data().select(range(512))

        lossname = "hinge"
        logging.info("Loaded model, pref and eval datasets")
        logging.info("Using: "+lossname)
                    
        dpo_peft_config = LoraConfig(
                    lora_alpha=256,
                    lora_dropout=0.05,
                    r=256,
                    bias="none",
                    target_modules = target_modules
                    )
        peft_model = get_peft_model(model, dpo_peft_config)
        print(peft_model.print_trainable_parameters())
        peft_model.config.use_cache = False
        logging.info("Loaded model+LORA adapters")

        training_args = DPOConfig(
        # output_dir=model_name+"-train-chartqa-dpo-sft-v1-hardneg-beta-0.3-test",  # Directory to save the model
        # output_dir=args.vlm_name+"-train-chartqa-dpo-base-ocr-tabular-rationales-unichart"+lossname,  # Directory to save the model
        # output_dir=args.vlm_name+"-dpo-sft-v2-ocr-tabular-high-cap-"+lossname,  # Directory to save the model
        output_dir=args.vlm_name+"random-test"+lossname,  # Directory to save the model
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        dataset_num_proc=32,  # tokenization will use 32 processes
        dataloader_num_workers=32,  # data loading will use 32 workers
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=1000,
        beta=0.3,  # DPO beta parameter (default: 0.1)
        loss_type = lossname, # try different stuff
        )
        trainer = DPOTrainer(
            peft_model,
            ref_model=None,  # not needed when using peft
            args=training_args,
            train_dataset=pref_dataset,
            eval_dataset=eval_dataset,
            processing_class = processor,
            peft_config=dpo_peft_config,
        )

        trainer.train()


    if args.mode == "grpo":
        # Setup and imports
        os.environ["WANDB_CONSOLE"] = "wrap" 
        wandb.init(project="chartrl", entity="chartrl")

        from trl import (GRPOConfig, GRPOTrainer, get_peft_config)
        from grpo_utils import format_reward,\
             accuracy_reward,\
             length_think_reward,\
             num_token_reward,\
             chart_type_reward,\
             table_style_reward,\
             process_style_reward


        blocks = 4

        def _resize_up(img):
            from PIL import Image as PILImage
            MIN_PIXELS = 320 * 28 * 28            # 1 003 520
            # MAX_PIXELS = 16384 * 28 * 28           # 12 843 776
            MAX_PIXELS = 320 * 28 * 28           # 12 843 776 (about 500x500)

            img = img.convert("RGB")
            w, h = img.size
            p = w * h
            if MIN_PIXELS <= p <= MAX_PIXELS:
                return img
            tgt_p  = max(min(p, MAX_PIXELS), MIN_PIXELS)
            scale  = (tgt_p / p) ** 0.5
            new_wh = (int(w * scale), int(h * scale))
            return img.resize(new_wh, PILImage.BICUBIC)

        def _grpo_format_data(example):
                from prompts import SYSTEM_PROMPT_TEMPLATES
                SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATES[blocks]
                conversation = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": example["query"]},
                        ],
                    },
                    # {"role": "assistant", "content": [{"type": "text", "text": "<think>"}]},
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, truncation=False)

                return {
                    "prompt": prompt,
                    "image": _resize_up(example["image"]),
                }

        custom = True

        if custom == True:
            grpo_dataset_path = "/grpo-custom-dataset-large-"+str(seed)
            if os.path.exists(str(cache_dir+grpo_dataset_path)):
                import json
                from datasets import load_from_disk
                train_dataset = load_from_disk(str(cache_dir+grpo_dataset_path))
                logging.info("Loaded train dataset from cache")

            else:
                ### Messy - fix TODO
                all_datasets = []
                dataset = ChartDataset("chartqa-src", processor=processor, blocks=blocks)
                train_dataset = dataset.load_chart_dataset(split = "train")
                # Only hard examples!
                train_dataset = train_dataset.filter(lambda example: example["human_or_machine"] == 0)
                all_datasets.append(train_dataset)
                dataset = ChartDataset("figqa", processor=processor, blocks=blocks)
                train_dataset = dataset.load_chart_dataset(split = "train")
                all_datasets.append(train_dataset)
                dataset = PlotQADataset("plotqa", processor=processor, blocks=blocks)
                train_dataset = dataset.load_plotqa_dataset(split = "train")
                all_datasets.append(train_dataset)
                dataset = ChartDataset("chartfc", processor=processor, blocks=blocks)
                train_dataset = dataset.load_chart_dataset(split = "train")
                all_datasets.append(train_dataset)

                all_datasets[0] = all_datasets[0].remove_columns('human_or_machine')
                all_datasets[2] = all_datasets[2].remove_columns('image_index')
                all_datasets[2] = all_datasets[2].remove_columns('qid')
                all_datasets[2] = all_datasets[2].remove_columns('answer_id')
                all_datasets[2] = all_datasets[2].remove_columns('type')
                all_datasets[2] = all_datasets[2].remove_columns('question_id')
                all_datasets[2] = all_datasets[2].rename_column('question_string', 'query')
                all_datasets[3] = all_datasets[3].rename_column('question','query')

                def _collapse_no(example):
                    is_none_label = example["label"] == None 
                    is_none_ans =  example["answer"] == None
                    # Harmonise both columns
                    example["label"]  = example["answer"] if is_none_label else example["label"]
                    return example

                select_all_datasets = []
                for i in range(len(all_datasets)):
                    if i==0:
                        select_all_datasets.append(all_datasets[i].shuffle(seed=seed))
                    else:
                        select_all_datasets.append(all_datasets[i].shuffle(seed=seed).select(range(2000)))
                all_datasets = select_all_datasets
                # all_datasets = [dst.shuffle(seed=seed).select(range(1000)) for dst in all_datasets]
                all_datasets = all_datasets[:1]+ all_datasets[2:]
                from datasets import concatenate_datasets
                train_dataset = concatenate_datasets(all_datasets)
                train_dataset = train_dataset.map(_collapse_no, num_proc=32)
                train_dataset = train_dataset.remove_columns('answer')
                
                grpo_train_dataset = train_dataset.map(_grpo_format_data,num_proc=32,load_from_cache_file=False)#.shuffle(seed=seed)#.select(range(5000))
                print(grpo_train_dataset[0]['prompt'])
                logging.info("Created Dataset and saved to disk")
                grpo_train_dataset.save_to_disk(str(cache_dir+grpo_dataset_path))
                exit(0)
       

            with open("./rationales_llm/rationales-large-"+str(seed)+".json", "r") as f:
                rationales = json.load(f)
                logging.info("Loaded rationales+chart types from json: "+str(len(rationales)))            
            
            errors = 0

            def _add_keys(example, idx):
                global errors
                rsn = rationales[idx].split("### Reasoning: ")[-1].strip().split("### Type: ")[0]
                # rsn = rationales[idx].split("### Type: ")[0].split("### Reasoning: ")[-1].strip()
                if "### Type: " in rationales[idx]:
                    typ = rationales[idx].split("### Type: ")[-1].strip()
                else:
                    typ = "bar"
                
                try:
                    tab_string = rationales[idx].split("```json")[-1].split("```")[0].split("### Reasoning")[0].strip()
                    tab_string = tab_string.replace('\n', '')
                    tab = None
                    # Repair
                    if not tab_string.endswith("}"):
                        tab_string += "}"
                    try:
                        tab = json.loads(tab_string, parse_int=str, parse_float=str, parse_constant=str)
                    except:
                        errors+=1
                        pass

                except:
                    errors+=1
                    # print("Error in table loading", tab_string)

                print(idx)

                # Very important check!!
                if tab is not None:
                    for i in range(len(tab["columns"])):
                        tab["columns"][i] = str(tab["columns"][i])
                    for i in range(len(tab["rows"])):
                        for j in range(len(tab["rows"][i])):
                            tab["rows"][i][j] = str(tab["rows"][i][j])


                example["reasoning"], example["chart_type"], example["table"] = rsn, typ, tab
                return example
            

            grpo_train_dataset = train_dataset.map(_add_keys, with_indices=True, num_proc=1, load_from_cache_file=False)
            print(errors)
            # Eval dataset is different from custom but to compare benchmarks we need this
            # grpo_dataset_path = "/grpo-chartqa-with-type-3-longer-thinking"
            dataset = ChartDataset("evochart", processor=processor, blocks=blocks)
            eval_dataset = dataset.load_chart_dataset(split = "test")
            eval_dataset = eval_dataset.map(_grpo_format_data, num_proc=32,load_from_cache_file=False).select(range(200))
            grpo_eval_dataset = eval_dataset.map(_add_keys, with_indices=True, num_proc=32, load_from_cache_file=False)

            logging.info("Loaded model and datasets")
            logging.info("Last train sample:", grpo_train_dataset[-1])
            # logging.info("First train sample:", grpo_eval_dataset[-1])
            # breakpoint()



        else:
            grpo_dataset_path = "/grpo-chartqa-with-type-3-longer-thinking"
            if not os.path.exists(cache_dir+grpo_dataset_path):
                dataset = ChartDataset("chartqa-src", processor=processor, blocks=blocks)
                train_dataset = dataset.load_chart_dataset(split = "train")
                eval_dataset = dataset.load_chart_dataset(split = "val")
                grpo_train_dataset = train_dataset.map(_grpo_format_data,num_proc=32,load_from_cache_file=False)
                grpo_eval_dataset = eval_dataset.map(_grpo_format_data,num_proc=32,load_from_cache_file=False).select(range(200))
                
                with open("./rationales_llm/rationales-chartqa-data-only_qwen72b.json", "r") as f:
                    rationales = json.load(f)

                logging.info("Loaded rationales+chart types from json")
                def _add_keys(example, idx):
                    rsn = rationales[idx].split("### Type: ")[0].split("### Reasoning: ")[-1].strip()
                    typ = rationales[idx].split("### Type: ")[-1].strip()
                    # tab = rationales[idx].strip()
                    example["reasoning"], example["chart_type"] = rsn, typ
                    # example["reasoning"], example["chart_type"], example ["table"]= rsn, typ, tab
                    return example
                grpo_train_dataset = grpo_train_dataset.map(_add_keys, with_indices=True, num_proc=32,load_from_cache_file=False)
                grpo_eval_dataset = grpo_eval_dataset.map(_add_keys, with_indices=True, num_proc=32,load_from_cache_file=False)
                grpo_train_dataset.save_to_disk(str(cache_dir+grpo_dataset_path))
                grpo_eval_dataset.save_to_disk(str(cache_dir+grpo_dataset_path+"-eval"))
            else:
                from datasets import load_from_disk
                grpo_train_dataset = load_from_disk(str(cache_dir+grpo_dataset_path)).select(range(1000))
                grpo_eval_dataset = load_from_disk(str(cache_dir+grpo_dataset_path+"-eval"))
                
                

        # start from an SFT checkpoint
        # sft_model_path = "/mnt/home/sanchit/Qwen2-VL-Finetune/output/sft-2.5-3b-chartqa-rationales/checkpoint-200"
        # from transformers import  Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(sft_model_path)
        # logging.info("Loaded model with SFT adapters from {}".format(sft_model_path))

        training_args = GRPOConfig(
        # output_dir=args.vlm_name+"grpo-answer-think-preappend",  # Directory to save the model
        # output_dir="full-chartqa-vanilla",  # Directory to save the model
        # output_dir = "grpo-start-ckpts/"+args.vlm_name+"-prm-"+str(seed),  # Directory to save the model
        output_dir = "grpo-start-ckpts/"+args.vlm_name+"-prm-large-train-v2-"+str(seed),  # Directory to save the model
        # output_dir = "grpo-test/from-sft-"+args.vlm_name+"-format-accuracy-length-longer-1000samp",  # Directory to save the model
        # output_dir = "test-lim-data",
        # output_dir = "prm",
        bf16=True,
        remove_unused_columns = False,
        per_device_train_batch_size=2,
        num_train_epochs=4,
        logging_steps=50,
        max_prompt_length = 4096,
        eval_strategy="steps",
        eval_steps=500,
        max_completion_length = 768,
        num_generations = 4,
        # learning_rate = 8e-7,
        # beta = 0.2,
        )

        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            reward_funcs=[format_reward, accuracy_reward, length_think_reward, num_token_reward, chart_type_reward, table_style_reward, process_style_reward],
            train_dataset=grpo_train_dataset,
            eval_dataset=grpo_eval_dataset,  
            processing_class=processor,
        )
        

        trainer.train()

                    

