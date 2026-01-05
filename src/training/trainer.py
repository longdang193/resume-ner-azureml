"""Training loop utilities."""

from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
from .checkpoint_loader import resolve_checkpoint_path

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
    context: RunContext | None = None,
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

    # Choose sampler based on run context
    if context is not None and context.distributed:
        train_sampler: Optional[DistributedSampler] = DistributedSampler(
            train_ds,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            collate_fn=data_collator,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )

    # Create validation loader only if validation data exists
    if val_data:
        val_ds = ResumeNERDataset(val_data, tokenizer, max_length, label2id)

        if context is not None and context.distributed:
            val_sampler: Optional[DistributedSampler] = DistributedSampler(
                val_ds,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=False,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                sampler=val_sampler,
                shuffle=False,
                collate_fn=data_collator,
            )
        else:
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=data_collator,
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
    for epoch in range(epochs):
        # Ensure each epoch sees a different shard order in distributed mode
        if (
            context.distributed
            and hasattr(train_loader, "sampler")
            and hasattr(train_loader.sampler, "set_epoch")
        ):
            train_loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

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
    is_best_trial: bool = False,
    save_only_best: bool = False,
) -> None:
    """
    Save model and tokenizer checkpoint.

    Args:
        model: Trained model (may be wrapped in DDP).
        tokenizer: Tokenizer instance.
        output_dir: Directory to save checkpoint.
        is_best_trial: Whether this trial is the best so far.
        save_only_best: If True, only save when is_best_trial is True.
    """
    # Check if we should skip saving
    if save_only_best and not is_best_trial:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Skipping checkpoint save (save_only_best=True, is_best_trial=False)"
        )
        return

    checkpoint_path = output_dir / "checkpoint"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    # Unwrap DDP model if needed (DDP wraps the model in .module)
    unwrapped_model = model.module if isinstance(model, DDP) else model
    unwrapped_model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)


def train_model(
    config: Dict[str, Any],
    dataset: Dict[str, Any],
    output_dir: Path,
    context: RunContext | None = None,
    tracker: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Train a token classification model and return evaluation metrics.

    Args:
        config: Configuration dictionary.
        dataset: Dataset dictionary.
        output_dir: Directory for outputs and checkpoints.
        context: Optional distributed training context.
        tracker: Optional MLflowTrainingTracker instance for logging.

    Returns:
        Dictionary of evaluation metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract backbone for run naming
    backbone = config.get("model", {}).get("backbone", "unknown")
    if "-" in backbone:
        backbone = backbone.split("-")[0]

    # Determine training type and checkpoint path
    checkpoint_config = config.get("training", {}).get("checkpoint", {})
    checkpoint_path = None
    training_type = "final"

    # Check if checkpoint should be loaded
    if checkpoint_config:
        backbone_for_resolve = backbone
        run_id_for_resolve = config.get("training", {}).get("run_id")
        resolved_path = resolve_checkpoint_path(
            config=config,
            backbone=backbone_for_resolve,
            run_id=run_id_for_resolve,
        )
        if resolved_path:
            checkpoint_path = str(resolved_path)
            # Checkpoint loading is part of final training workflow
            training_type = "final"

    # Start MLflow tracking run if tracker provided
    # Try to use systematic naming if context information is available
    run_name = None

    # First, check if MLFLOW_RUN_NAME is set (from notebook/environment)
    import os
    env_run_name = os.environ.get("MLFLOW_RUN_NAME")
    if env_run_name:
        run_name = env_run_name

    # If not set, try to build systematic name from config
    if not run_name:
        try:
            from orchestration.naming_centralized import create_naming_context
            from orchestration.jobs.tracking.mlflow_naming import build_mlflow_run_name
            from shared.platform_detection import detect_platform

            # Try to extract fingerprints from config if available
            spec_fp = config.get("fingerprints", {}).get("spec_fp")
            exec_fp = config.get("fingerprints", {}).get("exec_fp")
            variant = config.get("fingerprints", {}).get("variant", 1)

            # Infer config_dir from output_dir if possible
            config_dir = None
            if output_dir:
                # Try to find config directory: go up from outputs/ to root, then to config/
                current = output_dir
                for _ in range(5):  # Limit search depth
                    if current.name == "outputs" and (current.parent / "config").exists():
                        config_dir = current.parent / "config"
                        break
                    current = current.parent
                    if not current or current == current.parent:  # Reached root
                        break

            # Build NamingContext if we have required fields
            if spec_fp and exec_fp:
                training_context = create_naming_context(
                    process_type="final_training",
                    model=backbone,
                    spec_fp=spec_fp,
                    exec_fp=exec_fp,
                    environment=detect_platform(),
                    variant=variant,
                )
                run_name = build_mlflow_run_name(training_context, config_dir)
        except Exception as e:
            # If systematic naming fails, fallback to manual construction
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                f"Could not build systematic run name: {e}, using fallback")

    # Fallback to manual construction if systematic naming didn't work
    if not run_name:
        run_id = config.get("training", {}).get("run_id", "unknown")
        run_name = f"{backbone}_{run_id}"

    # We'll use the tracker context manager later, after training completes

    data_cfg = config["data"]
    label_list = build_label_list(data_cfg)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    # Resolve or reuse run context from centralized distributed config + hardware.
    from .config import resolve_distributed_config

    if context is None:
        dist_cfg = resolve_distributed_config(config)
        context = create_run_context(dist_cfg)

    model, tokenizer, _ = create_model_and_tokenizer(
        config, label2id, id2label, device=context.device, checkpoint_path=checkpoint_path
    )

    # Wrap model with DDP when running in distributed mode.
    if context.distributed:
        model = DDP(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )

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
        config,
        dataset,
        tokenizer,
        label2id,
        train_indices=train_indices,
        val_indices=val_indices,
        use_all_data=use_all_data,
        context=context,
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

    # Evaluate only if validation loader exists and on main process.
    if val_loader is not None and context.is_main_process():
        metrics = evaluate_model(model, val_loader, context.device, id2label)
    elif val_loader is not None:
        # Non-main processes skip evaluation; metrics not used by callers.
        metrics = {}
    else:
        # No validation set (final training on all data)
        metrics = {"note": "No validation set - training on all data"}

    # Log to MLflow if tracker provided and on main process
    if tracker and context.is_main_process():
        try:
            import json
            import random

            # Use context manager for training run
            with tracker.start_training_run(
                run_name=run_name,
                backbone=backbone,
                training_type=training_type,
            ):
                # Log training parameters
                data_config = config.get("data", {})
                random_seed = config.get("training", {}).get(
                    "seed") or random.randint(0, 2**31 - 1)

                tracker.log_training_parameters(
                    config=config,
                    data_config=data_config,
                    source_checkpoint=checkpoint_path,
                    data_strategy=data_strategy,
                    random_seed=random_seed,
                )

                # Log metrics (extract per_entity if available)
                metrics_copy = metrics.copy() if isinstance(metrics, dict) else {}
                per_entity_metrics = metrics_copy.pop(
                    "per_entity", None) if isinstance(metrics_copy, dict) else None
                tracker.log_training_metrics(
                    metrics=metrics_copy,
                    per_entity_metrics=per_entity_metrics,
                )

                # Save metrics.json for artifact logging
                metrics_json_path = output_dir / "metrics.json"
                if isinstance(metrics, dict):
                    # Use original metrics dict (with per_entity if it exists)
                    with open(metrics_json_path, 'w') as f:
                        json.dump(metrics, f, indent=2)

                # Log artifacts
                tracker.log_training_artifacts(
                    checkpoint_dir=output_dir,
                    metrics_json_path=metrics_json_path if metrics_json_path.exists() else None,
                )
        except Exception as e:
            from shared.logging_utils import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Could not log training results to MLflow: {e}")

    # Only main process should write checkpoints to avoid file collisions.
    if context.is_main_process():
        # Read checkpoint saving parameters from config or environment
        import os
        checkpoint_config = config.get("training", {}).get("checkpoint", {})
        save_only_best = checkpoint_config.get("save_only_best", False)

        # Check if this trial is best (from environment variable set by HPO)
        # Note: This is set AFTER trial completes, so for HPO we'll handle cleanup separately
        # For now, we'll save the checkpoint and let HPO code handle deletion if needed
        is_best_trial_env = os.environ.get(
            "IS_BEST_TRIAL", "false").lower() == "true"
        is_best_trial = is_best_trial_env

        save_checkpoint(
            model,
            tokenizer,
            output_dir,
            is_best_trial=is_best_trial,
            save_only_best=save_only_best,
        )

    return metrics
