"""Configuration loading and building utilities."""

from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import argparse

from common.shared.yaml_utils import load_yaml


def load_config_file(config_dir: Path, filename: str) -> Dict[str, Any]:
    """
    Load configuration file from directory.

    Args:
        config_dir: Directory containing configuration files.
        filename: Name of the configuration file.

    Returns:
        Dictionary containing configuration data.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    config_path = config_dir / filename
    return load_yaml(config_path)


def build_training_config(args: argparse.Namespace, config_dir: Path) -> Dict[str, Any]:
    """
    Build training configuration from files and command-line arguments.

    Args:
        args: Parsed command-line arguments.
        config_dir: Directory containing configuration files.

    Returns:
        Dictionary containing merged configuration.
    """
    # Check if CHECKPOINT_PATH is set (for checkpoint loading)
    import os
    has_checkpoint = bool(os.environ.get("CHECKPOINT_PATH"))
    
    # Load base training config
    train_config = load_config_file(config_dir, "train.yaml")
    
    if has_checkpoint:
        # Checkpoint loading is handled via CHECKPOINT_PATH environment variable
        # The checkpoint path is resolved by the training script or orchestration layer
        checkpoint_config = {}
        if checkpoint_config:
            merged_training["checkpoint"] = checkpoint_config
        
        train_config_dict = merged_training
        base_train_config = train_config  # For distributed config
    else:
        # Standard training config
        train_config = load_config_file(config_dir, "train.yaml")
        train_config_dict = train_config.get("training", {}).copy()
        base_train_config = train_config
    
    model_config = load_config_file(config_dir, f"model/{args.backbone}.yaml")
    data_config = load_config_file(config_dir, "data/resume_v1.yaml")
    
    config = {
        "data": data_config,
        "model": model_config,
        "training": train_config_dict,
        # Expose distributed section (if present) at top level so orchestration
        # and training logic can consume it without hard-coding defaults.
        "distributed": base_train_config.get("distributed", {}).copy(),
        "_config_dir": config_dir,  # Store for checkpoint resolution
    }
    
    _apply_argument_overrides(args, config)
    
    return config


@dataclass
class ResolvedDistributedConfig:
    """Resolved distributed training configuration.

    This is a thin, centralized representation of DDP-related knobs, derived
    from YAML config plus (later) environment detection. It deliberately does
    not perform any torch.distributed calls; those belong in a dedicated
    distributed helper module.
    """

    enabled: bool
    backend: str
    world_size: Optional[int]
    init_method: str
    timeout_seconds: int


def resolve_distributed_config(config: Dict[str, Any]) -> ResolvedDistributedConfig:
    """Resolve the distributed config section into a simple dataclass.

    Args:
        config: Top-level training config returned by build_training_config.

    Returns:
        ResolvedDistributedConfig with basic, YAML-driven settings.
    """
    dist_cfg = (config.get("distributed") or {}).copy()

    enabled = bool(dist_cfg.get("enabled", False))
    backend = dist_cfg.get("backend", "nccl")
    world_size_raw = dist_cfg.get("world_size", "auto")
    init_method = dist_cfg.get("init_method", "env://")
    timeout_seconds = int(dist_cfg.get("timeout_seconds", 1800))

    world_size: Optional[int]
    if isinstance(world_size_raw, int):
        world_size = world_size_raw
    else:
        # 'auto' or any non-int value means: decide based on hardware later.
        world_size = None

    return ResolvedDistributedConfig(
        enabled=enabled,
        backend=backend,
        world_size=world_size,
        init_method=init_method,
        timeout_seconds=timeout_seconds,
    )


def _apply_argument_overrides(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Apply command-line argument overrides to configuration."""
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.random_seed is not None:
        config["training"]["random_seed"] = args.random_seed
    if args.early_stopping_enabled is not None:
        enabled = args.early_stopping_enabled.lower() == "true"
        config["training"]["early_stopping"]["enabled"] = enabled
    if args.use_combined_data is not None:
        config["data"]["use_combined_data"] = args.use_combined_data.lower() == "true"
    if args.fold_idx is not None:
        config["training"]["fold_idx"] = args.fold_idx
    if args.fold_splits_file is not None:
        config["training"]["fold_splits_file"] = args.fold_splits_file
    if args.k_folds is not None:
        config["training"]["k_folds"] = args.k_folds
    if args.use_all_data is not None:
        config["training"]["use_all_data"] = args.use_all_data.lower() == "true"

