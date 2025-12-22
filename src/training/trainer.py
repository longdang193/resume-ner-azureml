"""Training loop utilities."""

from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

from .data import ResumeNERDataset, build_label_list
from .model import create_model_and_tokenizer
from .evaluator import evaluate_model
from .cv_utils import load_fold_splits, get_fold_data
from .distributed import RunContext, create_run_context

# Default constants (can be overridden via config)
DEFAULT_VAL_SPLIT_DIVISOR = 10
DEFAULT_DEBERTA_MAX_BATCH_SIZE = 8
DEFAULT_WARMUP_STEPS_DIVISOR = 10


def prepare_data_loaders(
    config: Dict[str, Any],
    dataset: Dict[str, Any],
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
    train_indices: Optional[List[int]] = None,
    val_indices: Optional[List[int]] = None,
    use_all_data: bool = False,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Prepare training and validation data loaders.

    Args:
        config: Configuration dictionary.
        dataset: Dataset dictionary with "train" and "validation" keys.
        tokenizer: Tokenizer instance.
        label2id: Mapping from label strings to integer IDs.
        train_indices: Optional list of indices for training subset (for CV).
        val_indices: Optional list of indices for validation subset (for CV).
        use_all_data: If True, use all data for training without validation split.

    Returns:
        Tuple of (train_loader, val_loader). val_loader is None if use_all_data=True.
    """
    # Get original data before any modifications
    original_train_data = dataset.get("train", [])
    val_data = dataset.get("validation", [])

    # Handle fold-based CV: validation indices refer to original train_data
    if val_indices is not None:
        # For k-fold CV, validation data comes from original train_data using val_indices
        # val_indices are indices into the original train_data before splitting
        val_data = [original_train_data[i] for i in val_indices]

    # Handle fold-based CV: use provided indices for training data
    train_data = original_train_data
    if train_indices is not None:
        train_data = [original_train_data[i] for i in train_indices]
    elif use_all_data:
        # Final training: use all data, no validation split
        val_data = []
    elif not val_data:
        # Fallback: only split if not using CV and not using all data
        val_split_divisor = config["training"].get(
            "val_split_divisor", DEFAULT_VAL_SPLIT_DIVISOR)
        val_data = train_data[: max(1, len(train_data) // val_split_divisor)]

    model_cfg = config["model"]
    max_length = model_cfg.get("preprocessing", {}).get("max_length", 128)
    train_cfg = config["training"]
    batch_size = train_cfg.get("batch_size", 8)

    backbone = model_cfg.get("backbone", "distilbert-base-uncased")
    deberta_max_batch_size = config["training"].get(
        "deberta_max_batch_size", DEFAULT_DEBERTA_MAX_BATCH_SIZE)
    if "deberta" in backbone.lower() and batch_size > deberta_max_batch_size:
        batch_size = deberta_max_batch_size

    train_ds = ResumeNERDataset(train_data, tokenizer, max_length, label2id)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=data_collator
    )

    # Create validation loader only if validation data exists
    if val_data:
        val_ds = ResumeNERDataset(val_data, tokenizer, max_length, label2id)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator
        )
    else:
        val_loader = None

    return train_loader, val_loader


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: Dict[str, Any],
    total_steps: int,
) -> tuple:
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: Model to optimize.
        config: Configuration dictionary.
        total_steps: Total number of training steps.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    train_cfg = config["training"]
    lr = train_cfg.get("learning_rate", 2e-5)
    wd = train_cfg.get("weight_decay", 0.0)
    warmup_steps = train_cfg.get("warmup_steps", 0)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup_steps_divisor = config["training"].get(
        "warmup_steps_divisor", DEFAULT_WARMUP_STEPS_DIVISOR)
    max_warmup_steps = total_steps // warmup_steps_divisor
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(warmup_steps, max_warmup_steps),
        num_training_steps=total_steps,
    )

    return optimizer, scheduler, max_grad_norm


def run_training_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    max_grad_norm: float,
    context: RunContext,
) -> None:
    """
    Run the training loop for specified epochs.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        epochs: Number of training epochs.
        max_grad_norm: Maximum gradient norm for clipping.
        device: Device to run training on.
    """
    model.train()
    device = context.device
    for _ in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: Path,
) -> None:
    """
    Save model and tokenizer checkpoint.

    Args:
        model: Trained model.
        tokenizer: Tokenizer instance.
        output_dir: Directory to save checkpoint.
    """
    checkpoint_path = output_dir / "checkpoint"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)


def train_model(
    config: Dict[str, Any],
    dataset: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, float]:
    """
    Train a token classification model and return evaluation metrics.

    Args:
        config: Configuration dictionary.
        dataset: Dataset dictionary.
        output_dir: Directory for outputs and checkpoints.

    Returns:
        Dictionary of evaluation metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    label_list = build_label_list(data_cfg)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    model, tokenizer, device = create_model_and_tokenizer(
        config, label2id, id2label
    )

    # Resolve run context from centralized distributed config and hardware.
    # For now this always returns a single-process context; DDP support will
    # be added in training.distributed without changing trainer logic.
    from .config import resolve_distributed_config

    dist_cfg = resolve_distributed_config(config)
    context = create_run_context(dist_cfg)

    # Handle k-fold CV: load fold splits if specified
    train_indices = None
    val_indices = None
    use_all_data = config["training"].get("use_all_data", False)

    fold_idx = config["training"].get("fold_idx")
    fold_splits_file = config["training"].get("fold_splits_file")

    if fold_idx is not None and fold_splits_file:
        # Load fold splits and extract current fold
        splits, _ = load_fold_splits(Path(fold_splits_file))
        if fold_idx < 0 or fold_idx >= len(splits):
            raise ValueError(
                f"Fold index {fold_idx} out of range. "
                f"Expected 0 to {len(splits) - 1}"
            )
        train_indices, val_indices = splits[fold_idx]

    train_loader, val_loader = prepare_data_loaders(
        config, dataset, tokenizer, label2id,
        train_indices=train_indices,
        val_indices=val_indices,
        use_all_data=use_all_data,
    )

    train_cfg = config["training"]
    epochs = max(1, train_cfg.get("epochs", 1))
    total_steps = epochs * max(1, len(train_loader))
    optimizer, scheduler, max_grad_norm = create_optimizer_and_scheduler(
        model, config, total_steps
    )

    run_training_loop(
        model,
        train_loader,
        optimizer,
        scheduler,
        epochs,
        max_grad_norm,
        context,
    )

    # Evaluate only if validation loader exists
    if val_loader is not None:
        metrics = evaluate_model(model, val_loader, context.device, id2label)
    else:
        # No validation set (final training on all data)
        # Return empty metrics or training-only metrics
        metrics = {"note": "No validation set - training on all data"}

    save_checkpoint(model, tokenizer, output_dir)

    return metrics
