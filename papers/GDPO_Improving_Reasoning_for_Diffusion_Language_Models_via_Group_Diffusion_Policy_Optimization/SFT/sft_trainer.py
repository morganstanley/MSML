import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist


class dLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")
        outputs = model(**inputs)
        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()})
        loss = unscaled_loss / t
        loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
        return loss if not return_outputs else (loss, outputs)


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()
        batch["input_ids"] = noisy_batch.long()
        return batch


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""


# from https://github.com/NEUIR/PC-Sampler/blob/master/src/template.py
def get_code_prompt(task, func_name_pars):
    return f'''Role: You are a professional Python coding assistant
Task: Complete the follow function implementation strictly and clearly without any additional comments or explanations.
{task}
the function name and the parameters is {func_name_pars}
Again, implement the above function strictly without any additional comments, explanations, unit tests, assertions, prints, examples in the code.
'''

def preprocess_dataset(data, tokenizer, max_length, test_split=0.01, data_name='s1K'):
    preprocessed_data = []
    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        if "KodCode" in data_name:
            question = get_code_prompt(data[i]['question'], data[i]['func_name_pars'])
            trajectory = "```python\n" + data[i]["solution"].rstrip() + "\n```"
        else:
            question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
            trajectory = f"<reasoning>{data[i]['thinking_trajectories'][0]}</reasoning>\n<answer>{data[i]['attempt']}</answer>"
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": trajectory}]
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
        tokenized_input = tokenizer(
            inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)
        num_tokens = tokenized_input.shape[0]
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            }
        )

    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data



# keep only rows with a non-empty test_info and required keys
def has_test_info(x):
    ti = x.get("test_info")
    return (
        isinstance(ti, list)
        and len(ti) > 0
        and isinstance(ti[0], dict)
        and "function_name" in ti[0]
        and "parameter_list" in ti[0]
    )


def get_KodCode_V1_SFT_R1(split: str = "train"):
    dataset = load_dataset("KodCode/KodCode-V1-SFT-R1", split=split)
    dataset = dataset.filter(has_test_info)
    dataset = dataset.map(lambda x: {'func_name_pars': x['test_info'][0]['function_name'] + x['test_info'][0]['parameter_list'] })

    return dataset.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": (get_code_prompt(x['question'], x['func_name_pars'])),
                }
            ],
            "solution": x["solution"],
        }
    )
