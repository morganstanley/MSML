import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "swe-pruner" / "src"))

from train.core.rubric import RUBRIC_DIMENSIONS
from train.train_llm.train import (
    CodePruneDataset,
    DEFAULT_ACTIVE_OBJECTIVES,
    DictData,
    RUBRIC_POSITIVE_THRESHOLD,
    TokenScorer as TrainTokenScorer,
    collate_fn,
    compute_gate_regularization_loss,
    compute_combined_loss,
    compute_rubric_loss,
    project_rubric_scores,
    stratified_split_indices,
    summarize_label_coverage,
)
from swe_pruner.model_structure import TokenScorer as PackagedTokenScorer


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

    def convert_ids_to_tokens(self, token_ids):
        return [self.id_to_token[token_id] for token_id in token_ids]


def build_synthetic_items():
    return [
        DictData(
            query="Where is the return value computed?",
            code="def add(x, y):\n    total = x + y\n    return total",
            kept_frags=[1, 3],
            score=0.95,
            rubric_schema=RUBRIC_DIMENSIONS,
            rubric_scores=[
                [0.95, 1.0, 0.60, 0.70],
                [0.35, 0.25, 0.95, 0.60],
                [0.95, 0.25, 0.80, 0.90],
            ],
        ),
        DictData(
            query="Which import and function set the path?",
            code="import os\n\ndef build_path(name):\n    return os.path.join('/tmp', name)",
            kept_frags=[1, 3, 4],
            score=0.90,
            rubric_schema=RUBRIC_DIMENSIONS,
            rubric_scores=[
                [0.70, 0.90, 1.00, 0.50],
                [0.00, 0.00, 0.00, 0.00],
                [0.85, 1.00, 0.60, 0.80],
                [0.90, 0.35, 0.95, 0.85],
            ],
        ),
    ]


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


def build_split_items():
    return [
        DictData(
            query="semantic a",
            code="a = 1",
            kept_frags=[1],
            score=0.9,
            rubric_schema=RUBRIC_DIMENSIONS,
            rubric_scores=[[0.9, 0.0, 0.0, 0.0]],
        ),
        DictData(
            query="semantic b",
            code="b = 2",
            kept_frags=[1],
            score=0.9,
            rubric_schema=RUBRIC_DIMENSIONS,
            rubric_scores=[[0.9, 0.0, 0.0, 0.0]],
        ),
        DictData(
            query="syntax a",
            code="def foo():\n    pass",
            kept_frags=[1],
            score=0.9,
            rubric_schema=RUBRIC_DIMENSIONS,
            rubric_scores=[[0.0, 1.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0]],
        ),
        DictData(
            query="syntax b",
            code="if True:\n    pass",
            kept_frags=[1],
            score=0.9,
            rubric_schema=RUBRIC_DIMENSIONS,
            rubric_scores=[[0.0, 1.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0]],
        ),
        DictData(
            query="dependency a",
            code="import os\nos.getcwd()",
            kept_frags=[2],
            score=0.9,
            rubric_schema=RUBRIC_DIMENSIONS,
            rubric_scores=[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.8, 0.0]],
        ),
        DictData(
            query="dependency b",
            code="x = helper()",
            kept_frags=[1],
            score=0.9,
            rubric_schema=RUBRIC_DIMENSIONS,
            rubric_scores=[[0.0, 0.0, 1.0, 0.0]],
        ),
        DictData(
            query="context a",
            code="line1\nline2",
            kept_frags=[1],
            score=0.9,
            rubric_schema=RUBRIC_DIMENSIONS,
            rubric_scores=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.6]],
        ),
        DictData(
            query="context b",
            code="line1\nline2",
            kept_frags=[2],
            score=0.9,
            rubric_schema=RUBRIC_DIMENSIONS,
            rubric_scores=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.6]],
        ),
    ]


class Phase2MultiObjectiveTest(unittest.TestCase):
    def test_crf_rubric_threshold_treats_point_five_as_positive(self):
        recorded = {}

        class RecordingCRF:
            def __call__(self, emissions, tags, mask, reduction="mean"):
                recorded["tags"] = tags.detach().cpu().clone()
                return emissions.sum() * 0.0 + torch.tensor(
                    1.0, device=emissions.device, dtype=emissions.dtype
                )

        actual_model = SimpleNamespace(
            compression_head_type="crf",
            compression_head=SimpleNamespace(crf_layers=[RecordingCRF()]),
        )
        rubric_token_logits = torch.zeros((1, 2, 1), dtype=torch.float32)
        compression_emissions = torch.zeros((1, 2, 1, 2), dtype=torch.float32)
        rubric_labels = torch.tensor(
            [[[RUBRIC_POSITIVE_THRESHOLD], [0.0]]],
            dtype=torch.float32,
        )
        valid_mask = torch.tensor([[True, True]], dtype=torch.bool)

        loss = compute_rubric_loss(
            actual_model=actual_model,
            rubric_token_logits=rubric_token_logits,
            compression_emissions=compression_emissions,
            rubric_labels=rubric_labels,
            valid_mask=valid_mask,
            device=torch.device("cpu"),
            compression_loss_type="focal",
            focal_alpha=0.5,
            focal_gamma=2.0,
            use_sample_level_aggregation=True,
            objective_weights=[1.0],
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(recorded["tags"], torch.tensor([[1, 0]])))

    def test_dataset_emits_rubric_token_labels(self):
        items = build_synthetic_items()
        tokenizer = build_tokenizer(items)
        dataset = CodePruneDataset(
            items,
            tokenizer,
            max_length=512,
            instruction="Given a query, judge if the document(code) is related to query.",
            compute_class_ratio=False,
            num_objectives=4,
            objective_names=RUBRIC_DIMENSIONS,
        )

        sample = dataset[0]
        self.assertEqual(sample["rubric_labels"].shape, (512, 4))
        self.assertEqual(sample["token_labels"].shape[0], 512)
        self.assertTrue((sample["rubric_labels"] != -100).any().item())
        self.assertTrue((sample["token_labels"] != -100).any().item())

    def test_rubric_projection_drops_syntax_dimension(self):
        items = build_synthetic_items()
        projected = project_rubric_scores(
            items[0].rubric_scores,
            items[0].rubric_schema,
            DEFAULT_ACTIVE_OBJECTIVES,
        )
        self.assertEqual(projected[0], [0.95, 0.60, 0.70])

        tokenizer = build_tokenizer(items)
        dataset = CodePruneDataset(
            items,
            tokenizer,
            max_length=512,
            instruction="Given a query, judge if the document(code) is related to query.",
            compute_class_ratio=False,
            num_objectives=3,
            objective_names=DEFAULT_ACTIVE_OBJECTIVES,
        )
        sample = dataset[0]
        self.assertEqual(sample["rubric_labels"].shape, (512, 3))
        self.assertFalse((sample["rubric_labels"] == 1.0).any().item())

    def test_packaged_model_returns_multi_objective_outputs(self):
        items = build_synthetic_items()
        tokenizer = build_tokenizer(items)
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=1024,
        )
        model = PackagedTokenScorer(
            model_name="unused",
            tokenizer=tokenizer,
            bottleneck=16,
            dropout=0.1,
            num_fusion_layers=1,
            num_heads=4,
            use_multi_layer_fusion=False,
            compression_head_type="ffn",
            num_objectives=4,
            use_moe_gating=True,
            gating_type="softmax",
            objective_names=RUBRIC_DIMENSIONS,
            load_pretrained_backbone=False,
            backbone_config=config,
            attn_implementation=None,
        )

        input_ids = torch.tensor([[1, 2, 3, 4, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(outputs["token_logits"].shape, (1, 5))
        self.assertEqual(outputs["rubric_token_logits"].shape, (1, 5, 4))
        self.assertEqual(outputs["gating_weights"].shape, (1, 5, 4))
        gate_sums = outputs["gating_weights"].sum(dim=-1)
        self.assertTrue(torch.allclose(gate_sums[0, :4], torch.ones(4), atol=1e-5))

    def test_packaged_crf_model_returns_fused_emissions(self):
        items = build_synthetic_items()
        tokenizer = build_tokenizer(items)
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=1024,
        )
        model = PackagedTokenScorer(
            model_name="unused",
            tokenizer=tokenizer,
            bottleneck=16,
            dropout=0.1,
            num_fusion_layers=1,
            num_heads=4,
            use_multi_layer_fusion=False,
            compression_head_type="crf",
            num_objectives=3,
            use_moe_gating=True,
            gating_type="softmax",
            use_final_crf=True,
            objective_names=DEFAULT_ACTIVE_OBJECTIVES,
            load_pretrained_backbone=False,
            backbone_config=config,
            attn_implementation=None,
        )

        input_ids = torch.tensor([[1, 2, 3, 4, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(outputs["compression_emissions"].shape, (1, 5, 3, 2))
        self.assertEqual(outputs["fused_emissions"].shape, (1, 5, 2))
        fused_logits = outputs["fused_emissions"][..., 1] - outputs["fused_emissions"][..., 0]
        self.assertTrue(torch.allclose(outputs["token_logits"], fused_logits, atol=1e-6))

    def test_gate_regularization_penalizes_collapsed_routing(self):
        valid_mask = torch.tensor([[True, True]], dtype=torch.bool)
        uniform_gates = torch.tensor(
            [[[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]],
            dtype=torch.float32,
        )
        collapsed_gates = torch.tensor(
            [[[0.98, 0.01, 0.01, 0.0], [0.99, 0.01, 0.0, 0.0]]],
            dtype=torch.float32,
        )

        uniform_loss, uniform_entropy = compute_gate_regularization_loss(
            uniform_gates,
            valid_mask,
        )
        collapsed_loss, collapsed_entropy = compute_gate_regularization_loss(
            collapsed_gates,
            valid_mask,
        )

        self.assertLess(uniform_loss.item(), collapsed_loss.item())
        self.assertGreater(uniform_entropy.item(), collapsed_entropy.item())
        self.assertAlmostEqual(uniform_entropy.item(), math.log(4), places=4)

    def test_stratified_split_preserves_objective_coverage(self):
        items = build_split_items()
        train_indices, val_indices = stratified_split_indices(
            items,
            train_split=0.5,
            seed=13,
            objective_names=RUBRIC_DIMENSIONS,
        )

        self.assertEqual(len(train_indices), 4)
        self.assertEqual(len(val_indices), 4)

        val_summary = summarize_label_coverage(items, val_indices, RUBRIC_DIMENSIONS)
        self.assertEqual(val_summary["rows"], 4)
        for objective_name in RUBRIC_DIMENSIONS:
            self.assertGreaterEqual(
                val_summary["objective_positive_rows"][objective_name],
                1,
            )

    def test_smoke_training_reduces_loss_and_moves_gates(self):
        torch.manual_seed(7)

        base_items = build_synthetic_items()
        items = base_items * 4
        tokenizer = build_tokenizer(items)
        dataset = CodePruneDataset(
            items,
            tokenizer,
            max_length=512,
            instruction="Given a query, judge if the document(code) is related to query.",
            compute_class_ratio=False,
            num_objectives=4,
            objective_names=RUBRIC_DIMENSIONS,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        reference_batch = next(iter(loader))

        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=96,
            max_position_embeddings=1024,
        )
        model = TrainTokenScorer(
            model_name="unused",
            tokenizer=tokenizer,
            bottleneck=24,
            dropout=0.1,
            num_finetune_layers=0,
            num_fusion_layers=1,
            num_heads=4,
            use_multi_layer_fusion=False,
            compression_head_type="ffn",
            num_objectives=4,
            use_moe_gating=True,
            gating_type="softmax",
            objective_names=RUBRIC_DIMENSIONS,
            load_pretrained_backbone=False,
            backbone_config=config,
            attn_implementation=None,
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=5e-3,
        )

        with torch.no_grad():
            initial_outputs = model(
                input_ids=reference_batch["input_ids"],
                attention_mask=reference_batch["attention_mask"],
            )
            initial_gate_mean = (
                initial_outputs["gating_weights"].mean(dim=(0, 1)).cpu().clone()
            )
            initial_loss, initial_logs = compute_combined_loss(
                model,
                reference_batch,
                lambda_score=0.0,
                compression_loss_type="focal",
                focal_alpha=0.5,
                focal_gamma=2.0,
                lambda_rubric=0.6,
                objective_weights=[1.0, 0.7, 0.7, 0.5],
            )

        for _ in range(12):
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
                    objective_weights=[1.0, 0.7, 0.7, 0.5],
                )
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            final_outputs = model(
                input_ids=reference_batch["input_ids"],
                attention_mask=reference_batch["attention_mask"],
            )
            final_gate_mean = (
                final_outputs["gating_weights"].mean(dim=(0, 1)).cpu().clone()
            )
            final_loss, final_logs = compute_combined_loss(
                model,
                reference_batch,
                lambda_score=0.0,
                compression_loss_type="focal",
                focal_alpha=0.5,
                focal_gamma=2.0,
                lambda_rubric=0.6,
                objective_weights=[1.0, 0.7, 0.7, 0.5],
            )

        self.assertFalse(math.isnan(final_loss.item()))
        self.assertLess(final_loss.item(), initial_loss.item())
        self.assertGreater(
            torch.max(torch.abs(final_gate_mean - initial_gate_mean)).item(),
            1e-4,
        )
        self.assertGreaterEqual(final_logs["rubric_loss"], 0.0)

    def test_multi_objective_crf_combined_loss_runs_with_final_crf(self):
        items = build_synthetic_items()
        tokenizer = build_tokenizer(items)
        dataset = CodePruneDataset(
            items,
            tokenizer,
            max_length=512,
            instruction="Given a query, judge if the document(code) is related to query.",
            compute_class_ratio=False,
            num_objectives=3,
            objective_names=DEFAULT_ACTIVE_OBJECTIVES,
        )
        batch = collate_fn([dataset[0], dataset[1]])

        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=1024,
        )
        model = TrainTokenScorer(
            model_name="unused",
            tokenizer=tokenizer,
            bottleneck=16,
            dropout=0.1,
            num_finetune_layers=0,
            num_fusion_layers=1,
            num_heads=4,
            use_multi_layer_fusion=False,
            compression_head_type="crf",
            num_objectives=3,
            use_moe_gating=True,
            gating_type="softmax",
            use_final_crf=True,
            objective_names=DEFAULT_ACTIVE_OBJECTIVES,
            load_pretrained_backbone=False,
            backbone_config=config,
            attn_implementation=None,
        )

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        self.assertEqual(outputs["compression_emissions"].shape[-2:], (3, 2))
        self.assertEqual(outputs["fused_emissions"].shape[-1], 2)

        loss, logs = compute_combined_loss(
            model,
            batch,
            lambda_score=0.0,
            compression_loss_type="focal",
            focal_alpha=0.5,
            focal_gamma=2.0,
            lambda_rubric=0.6,
            objective_weights=[1.0, 0.7, 0.5],
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(logs["aggregate_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
