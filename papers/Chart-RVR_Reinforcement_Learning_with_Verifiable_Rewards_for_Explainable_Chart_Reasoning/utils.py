import os
import tqdm
import re
import time
import gc
import random
from PIL import Image

from prompts import SYSTEM_PROMPT_TEMPLATES

seed = 2025
random.seed(seed)

import torch

# for processing qwen image 
from qwen_vl_utils import process_vision_info
# for processinf internvl image
from internvl_utils import get_conv_template, load_image



def get_vlm_output(model, processor, image, query, cot = False, icl_samples=None, model_device = None, blocks=2):
    max_new_tokens = 768 if cot else 100 # To speed up inference when not cot
    rationale = None # only used when cot=True

    # For ICL
    # if icl_samples is not None:
    #     icl_prompt_prefix = get_icl_prompt_process() 

    # For batched inference
    query =  [get_prompt_temp(q,cot) for q in query]
    messages = []
    for q,im in zip(query,image):
        # breakpoint()
        if "llava" in model.__class__.__name__.lower():
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text":q},
                    ]
                },
            ])

        elif "qwen" in model.__class__.__name__.lower():            
            if cot:
                SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATES[blocks]
            else:
                SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATES[1]
            
            messages.append([
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resize_fit(im)},
                    {"type": "text", "text": q},
                ],
            },
            ])

        elif "gemma" in model.__class__.__name__.lower():  
            messages.append("program of thought: " + q)

        # messages.append([
            # {
            #     "role": "system",
            #     "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            # },
            # {
            #     "role": "user",
            #     "content": [
            #         {"type": "image", "image": im},
            #         {"type": "text", "text": q},
            #     ],
            # },
            # {
            #     "role": "assistant", 
            #     "content": [{"type": "text", "text": "<think> "}]
            # },
            # ])

        elif "internvl" in model.__class__.__name__.lower():
            messages.append("<image>\n " + q)

    

    # Every model has a different way of processing the input and decoding. This is messy but works. #TODO
    if "llava" in model.__class__.__name__.lower():
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_text = processor.batch_decode(output, skip_special_tokens=True)
        out_text = [o.split("[/INST]")[-1].strip() for o in out_text]
        
        if cot:
            pred_text = [out.split("Answer:")[-1].strip() for out in out_text]
            rationale = [out.split("Reason:")[-1].split("Answer:")[0].strip("\n") for out in out_text]
            out_text = pred_text

    elif "gemma" in model.__class__.__name__.lower():
        # prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        prompt = messages
        # prompt = [q for q in query]
        image = [img.convert("RGB") for img in image]
        inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        # inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample = False)
        out_text = processor.batch_decode(output[:,input_len:], skip_special_tokens=True)
        # breakpoint()
        pattern = re.compile(r'^\s*print\(\s*([\'"])(.*?)\1\s*\)\s*$')
        # out_text = [pattern.match(o) for o in out_text]
        # out_text = [m.group(2) if m else "" for m in out_text]
        # print(out_text)
        # if cot:
        #     pred_text = [out.split("### Answer:")[-1].strip() for out in out_text]
        #     rationale = [out.split("### Reason:")[-1].split("### Answer")[0].strip("\n") for out in out_text]
        #     out_text = pred_text

        if cot: 
            pred_text = [out.split("<answer>")[-1].strip().split("</answer>")[0].strip() for out in out_text]
            rationale = [out.split("<think>")[-1].strip().split("</think>")[0].strip("\n") for out in out_text]
            out_text = pred_text


    elif "qwen" in model.__class__.__name__.lower():
        # breakpoint()
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print(text)
        # text = processor.apply_chat_template(messages, add_generation_prompt=False, continue_final_message=True, truncation=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=text,
            images=[image_inputs],
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Used .to(model.device) as default but it causes issues with qwen2vl-2b models (investigate) #TODO
        if model_device is not None:
            inputs = inputs.to("cuda")
            print("Moved inputs to cuda")
        else:
            inputs = inputs.to(model.device)
            print("Moved inputs to device")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        out_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # breakpoint()
        # print("raw:", out_text)
        if cot: 
            pred_text = [out.split("<answer>")[-1].strip().split("</answer>")[0].strip() for out in out_text]
            rationale = [out.split("<think>")[-1].strip().split("</think>")[0].strip("\n") for out in out_text]
            out_text = pred_text


    elif "internvl" in model.__class__.__name__.lower():
        # Constants
        history=None
        history = [] if history is None else history
        num_patches_list=None 
        IMG_START_TOKEN='<img>' 
        IMG_END_TOKEN='</img>'
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
        # Loading images in list and sanity
        pixel_values = [load_image(img) for img in image]
        if num_patches_list is None:
            num_patches_list = [pxl.shape[0] for pxl in pixel_values]
        # print(f"Mismatch! - check inference function internvl")
        assert len(pixel_values) == sum(num_patches_list)
        # Setting up model specific things
        img_context_token_id = processor.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id
        
        query = []
        for question in messages:
            template = get_conv_template(model.template)
            template.system_message = model.system_message
            eos_token_id = processor.convert_tokens_to_ids(template.sep.strip())
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query.append(template.get_prompt())

        # Verified up until here (the prompts are formatted correctly)

        for i,num_patches in enumerate(num_patches_list):
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
            query[i] = query[i].replace('<image>', image_tokens, 1)


        generation_config = {}
        model_inputs = processor(query, return_tensors='pt',padding=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)#.to(torch.bfloat16)
        attention_mask = model_inputs['attention_mask'].to(device)#.to(torch.bfloat16)
        pixel_values = torch.stack(pixel_values, dim=1).squeeze(0).to(device).to(torch.bfloat16)
        generation_config['eos_token_id'] = eos_token_id
        generation_config['do_sample'] = False

       
        outputs = model.generate(pixel_values=pixel_values,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                **generation_config,
                                max_new_tokens=max_new_tokens)
        
        out_text = processor.batch_decode(outputs, skip_special_tokens=True)
        # breakpoint()
        if cot:
            pred_text = [out.split("### Answer:")[-1].strip() for out in out_text]
            rationale = [out.split("### Reason:")[-1].split("### Answer")[0].strip("\n") for out in out_text]
            out_text = pred_text

    return out_text, rationale

def get_prompt_temp(q, cot=False):
    if cot:
        # prefix = ( "Look at the chart. Think step-by-step based on the question and chart. \n \
        #         Generate a reasoning chain first and output it as Reason. \n \
        #         In the next line, answer the question with a single word only without units or symbols.\n \
        #         Format:  \
        #         ### Question: {question} \
        #         ### Reason: \
        #         step-1 ... \
        #         step-2 ... \
        #         ....\
        #         step-n ... \
        #         ### Answer: \n")
        prefix = ("{question}")
        
        prefix = prefix.format(question=q)
    else:
        # answer en 
        # prefix = "%s Answer the question with a single word. Answer: "%q #used lmms-lab eval

        # For weird PaliGemma models (they require an instruction format of form "answer en")
        # prefix = "{question}\n Please try to answer the question with short words or phrases if possible.".format(question=q)
        prefix = "{question}".format(question=q)

    return prefix


def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def format_data(sample):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"][0]}],
        },
    ]


def select_icl_samples(train_dataset, k=5):
    # Select k random samples from the training dataset for in-context learning TODO!
    # Here, we simply select the first k samples for reproducibility
    select_list = random.sample(range(len(train_dataset)), k)
    selected_samples = []
    for i in select_list:
        selected_samples.append(train_dataset[i])
    return selected_samples











  # elif "internvl" in model.__class__.__name__.lower():
    #     from internvl_utils import build_transform, find_closest_aspect_ratio, dynamic_preprocess, load_image, get_conv_template
    #     generation_config = {"num_beams":1,"max_new_tokens":100}
    #     question =  "<image> "+ get_prompt_temp(query[0])
    #     pixel_values = load_image(image[0])
    #     history=None 
    #     return_history=False
    #     num_patches_list=None 
    #     IMG_START_TOKEN='<img>' 
    #     IMG_END_TOKEN='</img>'
    #     IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

    #     if history is None and pixel_values is not None and '<image>' not in question:
    #         question = '<image>\n' + question

    #     if num_patches_list is None:
    #         num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    #     assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    #     img_context_token_id = processor.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    #     model.img_context_token_id = img_context_token_id


    #     template = get_conv_template(model.template)
    #     template.system_message = model.system_message
    #     eos_token_id = processor.convert_tokens_to_ids(template.sep.strip())

    #     history = [] if history is None else history
    #     for (old_question, old_answer) in history:
    #         template.append_message(template.roles[0], old_question)
    #         template.append_message(template.roles[1], old_answer)
    #     template.append_message(template.roles[0], question)
    #     template.append_message(template.roles[1], None)
    #     query = template.get_prompt()


    #     for num_patches in num_patches_list:
    #         image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
    #         query = query.replace('<image>', image_tokens, 1)

    #     model_inputs = processor(query, return_tensors='pt')
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     input_ids = model_inputs['input_ids'].to(device)
    #     attention_mask = model_inputs['attention_mask'].to(device)
    #     generation_config['eos_token_id'] = eos_token_id

        
    #     outputs = model.generate(
    #         pixel_values=pixel_values,
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         **generation_config
    #     )

    #     out_text = processor.batch_decode(
    #         outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #     )

    # elif 'visionencoderdecodermodel' in model.__class__.__name__.lower():
    #     input_prompt = '<chartqa> ' + query[0] + ' <s_answer>'
    #     print(input_prompt)
    #     decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    #     pixel_values = processor(image[0].convert("RGB"), return_tensors="pt").pixel_values
    #     outputs = model.generate(
    #         pixel_values.to(model.device),
    #         decoder_input_ids=decoder_input_ids.to(model.device),
    #         max_length=model.decoder.config.max_position_embeddings,
    #         early_stopping=True,
    #         pad_token_id=processor.tokenizer.pad_token_id,
    #         eos_token_id=processor.tokenizer.eos_token_id,
    #         use_cache=True,
    #         num_beams=4,
    #         bad_words_ids=[[processor.tokenizer.unk_token_id]],
    #         return_dict_in_generate=True,)

    #     sequence = processor.batch_decode(outputs.sequences)[0]
    #     sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    #     sequence = sequence.split("<s_answer>")[1].strip()

    #     out_text = sequence



# For OOM and efficiency - it is possible that some results change slightly from reported numbers
def resize_fit(img):
    MIN_PIXELS = 480 * 28 * 28            # 1 003 520
    # MAX_PIXELS = 16384 * 28 * 28           # 12 843 776
    MAX_PIXELS = 480 * 28 * 28           # 12 843 776
    # MAX_PIXELS = 960 * 28 * 28           # 12 843 776

    img = img.convert("RGB")
    w, h = img.size
    p = w * h
    if MIN_PIXELS <= p <= MAX_PIXELS:
        return img
    tgt_p  = max(min(p, MAX_PIXELS), MIN_PIXELS)
    scale  = (tgt_p / p) ** 0.5
    new_wh = (int(w * scale), int(h * scale))
    return img.resize(new_wh, Image.BICUBIC)