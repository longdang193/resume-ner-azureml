"""Shared configuration merging utilities."""

import argparse
from typing import Any, Dict, List, Optional


def merge_configs_with_precedence(
    base: Dict[str, Any],
    overrides: Dict[str, Any],
    precedence: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Merge config dictionaries with precedence rules.
    
    This function performs a deep merge where override values take precedence
    over base values. Nested dictionaries are merged recursively.
    
    Args:
        base: Base configuration dictionary
        overrides: Override configuration dictionary
        precedence: Optional precedence order (default: overrides > base)
                   Currently not used but reserved for future enhancement
    
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs_with_precedence(result[key], value, precedence)
        else:
            # Override with new value (only if not None)
            if value is not None:
                result[key] = value
    
    return result


def apply_argument_overrides(
    args: argparse.Namespace,
    config: Dict[str, Any]
) -> None:
    """
    Apply command-line argument overrides to configuration.
    
    Extracted from training/config.py::_apply_argument_overrides()
    Supports: learning_rate, batch_size, dropout, weight_decay, epochs,
    random_seed, early_stopping_enabled, use_combined_data, fold_idx,
    fold_splits_file, k_folds, use_all_data
    
    Args:
        args: Parsed command-line arguments namespace
        config: Configuration dictionary to modify in-place
    """
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

