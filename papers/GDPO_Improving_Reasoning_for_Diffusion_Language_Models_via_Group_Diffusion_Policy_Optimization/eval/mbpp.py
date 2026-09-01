import re
import pandas as pd
from gsm8k import GSM8KDataset
from datasets import Dataset as HFDataset
import os
from parsers import Parser


MBPP_SYSTEM_PROMPT = "You are an expert Python programmer, and here is your task "



class MBPPDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=MBPP_SYSTEM_PROMPT,
        subsample=256,
    ):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.mbpp_file_path = f"{cur_path}/../dataset/mbpp/mbpp.jsonl"
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)

    def load_test_dataset(self):
        """Load the Sudoku dataset from the CSV file."""
        df = pd.read_json(self.mbpp_file_path, lines=True)
        self.dataset = HFDataset.from_pandas(df)
        print("Loaded Testing MBPP dataset with {} examples".format(len(self.dataset)))

    def format_mbpp_grid(self, mbpp_str):
        return mbpp_str

    def create_prompt(self, input_text, test_list):
        # Format similar to your chat function
        if self.num_examples > 0:
            prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        else:
            prompt = input_text

        str_pass_test = f"Your code should pass these tests:\n\n{test_list[0]}\n{test_list[1]}\n{test_list[2]}\n"
        messages = [{"role": "user", "content": prompt + '\n\n' + str_pass_test}]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            return user_input + "<reasoning>"
        else:
            return user_input

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        text = self.dataset[self.subsample[idx].item()]["text"]
        test_list = self.dataset[self.subsample[idx].item()]["test_list"]
        code = self.dataset[self.subsample[idx].item()]["code"]

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
