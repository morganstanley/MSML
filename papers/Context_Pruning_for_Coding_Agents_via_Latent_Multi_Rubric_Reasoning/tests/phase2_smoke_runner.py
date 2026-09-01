import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train.core.rubric import RUBRIC_DIMENSIONS
from train.train_llm.train import (
    CodePruneDataset,
    DictData,
    TokenScorer,
    collate_fn,
    compute_combined_loss,
    infer_num_objectives_from_data,
)


class TinyCharTokenizer:
    def __init__(self, texts):
        self.pad_token_id = 0
        self.cls_token_id = None
        self.sep_token_id = None
        self.token_to_id = {"<pad>": 0, "yes": 1, "no": 2}
        self.id_to_token = {0: "<pad>", 1: "yes", 2: "no"}
        for text in texts:
            for ch in text:
                self._add_token(ch)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    def __len__(self):
        return len(self.token_to_id)

    def encode(self, text, add_special_tokens=False):
        return [self._add_token(ch) for ch in text]

    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_offsets_mapping=False,
    ):
        input_ids = self.encode(text, add_special_tokens=add_special_tokens)
        output = {"input_ids": input_ids}
        if return_offsets_mapping:
            output["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return output

    def convert_tokens_to_ids(self, token):
        return self.token_to_id[token]


def load_items(input_file: Path, max_items: int):
    items = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if len(items) >= max_items:
                break
            item = json.loads(line)
            if not item.get("rubric_scores"):
                continue
            items.append(DictData(**item))
    return items


def build_tokenizer(items):
    prefix = (
        '<|im_start|>system\nJudge whether the Document meets the requirements '
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    instruction = "Given a query, judge if the document(code) is related to query."
    texts = [prefix, suffix, instruction]
    for item in items:
        texts.extend([item.query, item.code])
    return TinyCharTokenizer(texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    torch.manual_seed(7)

    input_file = Path(args.input_file)
    items = load_items(input_file, args.max_items)
    if not items:
        raise SystemExit("No rubric-labeled rows found in input file")

    tokenizer = build_tokenizer(items)
    num_objectives = infer_num_objectives_from_data(items, fallback=4)
    objective_names = RUBRIC_DIMENSIONS[:num_objectives]
    if len(objective_names) < num_objectives:
        objective_names = objective_names + [
            f"objective_{idx}"
            for idx in range(len(objective_names), num_objectives)
        ]

    dataset = CodePruneDataset(
        items,
        tokenizer,
        max_length=args.max_length,
        instruction="Given a query, judge if the document(code) is related to query.",
        compute_class_ratio=False,
        num_objectives=num_objectives,
        objective_names=objective_names,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    reference_batch = next(iter(loader))

    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=96,
        max_position_embeddings=max(args.max_length + 64, 1024),
    )
    model = TokenScorer(
        model_name="unused",
        tokenizer=tokenizer,
        bottleneck=24,
        dropout=0.1,
        num_finetune_layers=0,
        num_fusion_layers=1,
        num_heads=4,
        use_multi_layer_fusion=False,
        compression_head_type="ffn",
        num_objectives=num_objectives,
        use_moe_gating=True,
        gating_type="softmax",
        objective_names=objective_names,
        load_pretrained_backbone=False,
        backbone_config=config,
        attn_implementation=None,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=5e-3,
    )
    objective_weights = [1.0] + [0.7] * max(0, num_objectives - 1)

    with torch.no_grad():
        initial_outputs = model(
            input_ids=reference_batch["input_ids"],
            attention_mask=reference_batch["attention_mask"],
        )
        initial_gate_mean = initial_outputs["gating_weights"].mean(dim=(0, 1)).cpu()
        initial_loss, _ = compute_combined_loss(
            model,
            reference_batch,
            lambda_score=0.0,
            compression_loss_type="focal",
            focal_alpha=0.5,
            focal_gamma=2.0,
            lambda_rubric=0.6,
            objective_weights=objective_weights,
        )

    completed_steps = 0
    while completed_steps < args.steps:
        for batch in loader:
            optimizer.zero_grad()
            loss, _ = compute_combined_loss(
                model,
                batch,
                lambda_score=0.0,
                compression_loss_type="focal",
                focal_alpha=0.5,
                focal_gamma=2.0,
                lambda_rubric=0.6,
                objective_weights=objective_weights,
            )
            if not loss.requires_grad:
                raise SystemExit(
                    "Smoke loss has no gradient; increase --max-length so code tokens are retained"
                )
            loss.backward()
            optimizer.step()
            completed_steps += 1
            if completed_steps >= args.steps:
                break

    with torch.no_grad():
        final_outputs = model(
            input_ids=reference_batch["input_ids"],
            attention_mask=reference_batch["attention_mask"],
        )
        final_gate_mean = final_outputs["gating_weights"].mean(dim=(0, 1)).cpu()
        final_loss, final_logs = compute_combined_loss(
            model,
            reference_batch,
            lambda_score=0.0,
            compression_loss_type="focal",
            focal_alpha=0.5,
            focal_gamma=2.0,
            lambda_rubric=0.6,
            objective_weights=objective_weights,
        )

    if math.isnan(final_loss.item()):
        raise SystemExit("Final loss is NaN")

    gate_delta = torch.max(torch.abs(final_gate_mean - initial_gate_mean)).item()
    print(
        json.dumps(
            {
                "input_file": str(input_file),
                "num_items": len(items),
                "num_objectives": num_objectives,
                "initial_loss": initial_loss.item(),
                "final_loss": final_loss.item(),
                "initial_gate_mean": initial_gate_mean.tolist(),
                "final_gate_mean": final_gate_mean.tolist(),
                "gate_delta_max": gate_delta,
                "final_rubric_loss": final_logs["rubric_loss"],
                "final_aggregate_loss": final_logs["aggregate_loss"],
            },
            indent=2,
        )
    )

    if final_loss.item() >= initial_loss.item():
        raise SystemExit("Smoke training did not reduce loss")
    if gate_delta <= 1e-4:
        raise SystemExit("Smoke training did not move gating weights")


if __name__ == "__main__":
    main()
