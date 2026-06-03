#!/usr/bin/env python3
"""
Convert consistency model checkpoint to DARTS-compatible format.

This removes misleading src.* module metadata that causes unpickling errors
when loading in darts-nextgen environments.

Usage (run from consistency_model_distillation_for_sr3/):
    python convert_checkpoint.py \\
        /path/to/original.ckpt \\
        /path/to/output.ckpt
"""

import sys
import torch
from pathlib import Path

# Add repo root to path so src.* imports resolve
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.consistency_model.model_new import ConsistencyWavelet as Consistency
from src.configuration import Config as config


def convert_checkpoint_for_darts(
    original_ckpt_path: str,
    output_ckpt_path: str,
    device: torch.device = torch.device("cpu"),
):
    """
    Load checkpoint using original src.* imports and re-save as a clean
    state_dict dict with no embedded pickle class references.
    """
    print(f"\n  Converting checkpoint:\n    {original_ckpt_path}")

    torch.serialization.add_safe_globals([config])
    model = Consistency.load_from_checkpoint(
        original_ckpt_path,
        config=config,
        bins_min=10,
        bins_max=150,
        loss_func="HybridWavelet",
        use_ema=True,
        map_location=device,
        strict=False,
        weights_only=False,
    )
    model.eval()

    state_dict = model.state_dict()
    hyperparams = dict(model.hparams) if hasattr(model, "hparams") else {}

    clean_ckpt = {
        "state_dict": state_dict,
        "hyper_parameters": hyperparams,
    }

    Path(output_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean_ckpt, output_ckpt_path)

    print(f"    ✓ Saved to:\n      {output_ckpt_path}")
    print(f"    State dict keys: {len(state_dict)}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    original_path = sys.argv[1]
    output_path = sys.argv[2]

    if not Path(original_path).exists():
        print(f"❌ Original checkpoint not found: {original_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    convert_checkpoint_for_darts(original_path, output_path, device)
    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()
