"""Model initialization utilities."""

from typing import Dict, Any, Optional
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


def create_model_and_tokenizer(
    config: Dict[str, Any],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    device: torch.device | None = None,
    checkpoint_path: Optional[str] = None,
) -> tuple:
    """
    Create model and tokenizer from configuration or checkpoint.

    Args:
        config: Configuration dictionary containing model settings.
        label2id: Mapping from label strings to integer IDs.
        id2label: Mapping from label IDs to label strings.
        device: Device to use (None = auto-detect).
        checkpoint_path: Optional path to checkpoint directory to load from.

    Returns:
        Tuple of (model, tokenizer, device).
    """
    model_cfg = config["model"]
    backbone = model_cfg.get("backbone", "distilbert-base-uncased")
    tokenizer_name = model_cfg.get("tokenizer", backbone)

    # Load from checkpoint if provided and valid
    if checkpoint_path:
        checkpoint_dir = Path(checkpoint_path)
        if checkpoint_dir.exists() and checkpoint_dir.is_dir():
            # Validate checkpoint has required files
            config_file = checkpoint_dir / "config.json"
            model_files = [
                checkpoint_dir / "pytorch_model.bin",
                checkpoint_dir / "model.safetensors",
                checkpoint_dir / "model.bin",
            ]
            
            if config_file.exists() and any(f.exists() for f in model_files):
                print(f"Loading model and tokenizer from checkpoint: {checkpoint_path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
                    model = AutoModelForTokenClassification.from_pretrained(
                        checkpoint_dir,
                        num_labels=len(label2id),
                        id2label=id2label,
                        label2id=label2id,
                    )
                    
                    # Validate label mappings match (warn if mismatch)
                    checkpoint_config = model.config
                    checkpoint_label2id = getattr(checkpoint_config, "label2id", {})
                    if checkpoint_label2id and checkpoint_label2id != label2id:
                        print(
                            f"Warning: Label mappings differ between checkpoint and config. "
                            f"Checkpoint: {checkpoint_label2id}, Config: {label2id}. "
                            f"Using config labels."
                        )
                    
                    if device is None:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    
                    return model, tokenizer, device
                except Exception as e:
                    print(
                        f"Warning: Failed to load from checkpoint {checkpoint_path}: {e}. "
                        f"Falling back to backbone {backbone}."
                    )
            else:
                print(
                    f"Warning: Checkpoint directory {checkpoint_path} exists but doesn't contain "
                    f"valid model files. Falling back to backbone {backbone}."
                )
        else:
            print(
                f"Warning: Checkpoint path {checkpoint_path} doesn't exist. "
                f"Falling back to backbone {backbone}."
            )
    
    # Fallback: Create new model from backbone (existing behavior)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model = AutoModelForTokenClassification.from_pretrained(
        backbone,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device
