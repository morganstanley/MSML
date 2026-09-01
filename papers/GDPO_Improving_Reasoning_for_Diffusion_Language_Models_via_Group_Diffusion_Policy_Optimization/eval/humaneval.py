import re
import pandas as pd
from gsm8k import GSM8KDataset
from datasets import Dataset as HFDataset
from datasets import load_dataset
import os
from parsers import Parser


HumanEval_SYSTEM_PROMPT = "You are an expert Python programmer, and here is your task "


class HumanEvalDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=HumanEval_SYSTEM_PROMPT,
        subsample=256,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)

    def load_test_dataset(self):
        """Load the Sudoku dataset from the CSV file."""
        dataset = load_dataset("openai/openai_humaneval", split='test')

        def extract_humaneval_test_cases(test_str: str, entry_point):
            pattern = r'^\s*(assert\s+.*)'
            cases = re.findall(pattern, test_str, re.MULTILINE)
            cases = [re.sub(r"\bassert candidate\(", f"assert {entry_point}(", a) for a in cases]
            return cases
        self.dataset = dataset.map(lambda x: {"test_cases": extract_humaneval_test_cases(x["test"], x["entry_point"])})

        print("Loaded Testing HumanEval dataset with {} examples".format(len(self.dataset)))

    def format_humaneval_grid(self, humaneval_str):
        return humaneval_str

    def create_prompt(self, input_text, test_list):
        # Format similar to your chat function
        if self.num_examples > 0:
            prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        else:
            prompt = input_text

        str_pass_test = f"Your code should pass these tests:\n\n" + "\n".join(test_list)
        messages = [{"role": "user", "content": prompt + '\n\n' + str_pass_test}]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            return user_input + "<reasoning>"
        else:
            return user_input

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        text = self.dataset[self.subsample[idx].item()]["prompt"]
        test_list = self.dataset[self.subsample[idx].item()]["test_cases"]
        code = self.dataset[self.subsample[idx].item()]["canonical_solution"]

        question = f"You are an expert Python programmer, and here is your task {text}"

        prompt = self.create_prompt(question, test_list)
        return prompt, question, code, test_list

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        test_list = [item[3] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids

        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts, "test_list": test_list}
