import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "swe-pruner" / "src"))

from swe_pruner.prune_wrapper import PruneRequest, SwePrunerForCodePruning


def main() -> None:
    model_path = REPO_ROOT / "runtime_models" / "swe-pruner-qwen-local"
    model = SwePrunerForCodePruning.from_pretrained(str(model_path))
    request = PruneRequest(
        query="Where is the basename call?",
        code=(
            "import os\n"
            "def helper(path):\n"
            "    base = os.path.basename(path)\n"
            "    return base\n"
            "value = 1"
        ),
        threshold=0.5,
    )
    response = model.prune(request)
    print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
    print("device", model._device)
    print("score", round(response.score, 6))
    print("kept", response.kept_frags)
    print(response.pruned_code)


if __name__ == "__main__":
    main()
