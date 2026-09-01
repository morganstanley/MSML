import os
import json
from parsers import Parser, evaluate_equation, validate_equation
from gsm8k import GSM8KDataset
import warnings

CTD_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think step-by-step. After reasoning, provide only your final expression inside \\boxed"
    + "{}"
    + " tags without including an equals sign or the target number. For example: \\boxed{a + b * c}"
    + """Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
)


class CTDDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=CTD_SYSTEM_PROMPT,
        subsample=256,
    ):
        if num_examples > 0:
            warnings.warn("num_examples must be 0 for Countdown dataset. Overriding num_examples to 0.")
        super().__init__(
            tokenizer,
            0,
            add_reasoning,
            system_prompt,
            subsample,
        )  # num_examples = always 0

    def load_test_dataset(self):
        self.dataset = []
        cur_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{cur_path}/../dataset/countdown_cd3_test.jsonl", "r") as f:
            for line in f:
                self.dataset.append(json.loads(line))
        print(len(self.dataset), "examples loaded")

    def __getitem__(self, idx):
        target = int(self.dataset[self.subsample[idx].item()]["output"])
        numbers_str = self.dataset[self.subsample[idx].item()]["input"]
        numbers = [int(num) for num in numbers_str.split(",")]
        question = f"Numbers: {numbers}\nTarget: {target}"
        prompt = self.create_prompt(question)
        return prompt, question, (numbers, target)
