import json
import sys
from pathlib import Path
from typing import Optional

import torch
import typer
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((Path(__file__).resolve().parent / "src")))

from swe_pruner.configuration import SwePrunerConfig
from swe_pruner.swepruner import SwePrunerForCodeCompression


app = typer.Typer(
    help="Export a training checkpoint into a serving-ready SwePruner model directory."
)


def _upgrade_legacy_state_dict(state_dict):
    upgraded = {}
    for key, value in state_dict.items():
        if key == "word_embeddings":
            continue
        if key == "compression_head.crf.transitions":
            upgraded["compression_head.crf_layers.0.transitions"] = value
            continue
        if key == "compression_head.crf.start_transitions":
            upgraded["compression_head.crf_layers.0.start_transitions"] = value
            continue
        if key == "compression_head.crf.end_transitions":
            upgraded["compression_head.crf_layers.0.end_transitions"] = value
            continue
        upgraded[key] = value
    return upgraded


@app.command()
def main(
    checkpoint_dir: Path = typer.Option(
        REPO_ROOT / "llm_experiments" / "swe-pruner-py",
        "--checkpoint-dir",
        help="Training output directory containing best_model.pt and model_config.json",
    ),
    backbone_model: Path = typer.Option(
        REPO_ROOT / "hf_models" / "Qwen3-Reranker-0.6B ",
        "--backbone-model",
        help="Local backbone model directory used to supply tokenizer and backbone weights",
    ),
    output_dir: Path = typer.Option(
        REPO_ROOT / "runtime_models" / "swe-pruner-qwen-local",
        "--output-dir",
        help="Serving-ready output directory",
    ),
) -> None:
    weights_path = checkpoint_dir / "best_model.pt"
    config_path = checkpoint_dir / "model_config.json"

    if not backbone_model.exists():
        raise FileNotFoundError(f"Missing local backbone at {backbone_model}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing checkpoint weights at {weights_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint config at {config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    config_json = json.loads(config_path.read_text())

    config = SwePrunerConfig(
        backbone_model_name_or_path=str(backbone_model),
        bottleneck=config_json.get("bottleneck", 256),
        dropout=config_json.get("dropout", 0.4),
        num_fusion_layers=config_json.get("num_fusion_layers", 1),
        num_heads=config_json.get("num_heads", 8),
        use_multi_layer_fusion=config_json.get("use_multi_layer_fusion", True),
        early_layer_ratio=config_json.get("early_layer_ratio", 0.25),
        middle_layer_ratio=config_json.get("middle_layer_ratio", 0.5),
        compression_head_type=config_json.get("compression_head_type", "crf"),
        compression_loss_type=config_json.get("compression_loss_type", "focal"),
        num_objectives=config_json.get("num_objectives", 1),
        use_moe_gating=config_json.get("use_moe_gating", False),
        gating_type=config_json.get("gating_type", "softmax"),
        use_final_crf=config_json.get("use_final_crf", False),
        objective_names=config_json.get("objective_names"),
    )

    model = SwePrunerForCodeCompression(config)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = _upgrade_legacy_state_dict(state_dict)

    missing_keys, unexpected_keys = model.model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "embedding_layer.weight",
        "gating_network.0.weight",
        "gating_network.0.bias",
        "gating_network.1.weight",
        "gating_network.1.bias",
        "gating_network.4.weight",
        "gating_network.4.bias",
    }
    real_missing = [key for key in missing_keys if key not in allowed_missing]
    if real_missing or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch. missing={real_missing}, unexpected={unexpected_keys}"
        )

    model.save_pretrained(str(output_dir), safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained(str(backbone_model), trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    typer.echo(str(output_dir))


if __name__ == "__main__":
    app()
