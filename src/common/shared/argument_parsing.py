"""
@meta
name: shared_argument_parsing
type: utility
domain: shared
responsibility:
  - Provide shared argument parsing utilities for CLI scripts
  - Add common arguments (config-dir, backbone, hyperparameters)
inputs:
  - ArgumentParser instances
outputs:
  - Configured parsers
tags:
  - utility
  - shared
  - cli
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Shared argument parsing utilities for CLI scripts."""

import argparse
from pathlib import Path
from typing import Optional


def add_config_dir_argument(parser: argparse.ArgumentParser) -> None:
    """Add --config-dir argument to parser."""
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Path to configuration directory",
    )


def add_backbone_argument(parser: argparse.ArgumentParser) -> None:
    """Add --backbone argument to parser."""
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="Model backbone name (e.g., 'distilbert', 'deberta')",
    )


def add_training_hyperparameter_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common training hyperparameter arguments to parser."""
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate override",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout override",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay override",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs override",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed",
    )


def add_training_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Add training data-related arguments to parser."""
    parser.add_argument(
        "--early-stopping-enabled",
        type=str,
        default=None,
        help="Enable early stopping ('true'/'false')",
    )
    parser.add_argument(
        "--use-combined-data",
        type=str,
        default=None,
        help="Use combined train+validation ('true'/'false')",
    )
    parser.add_argument(
        "--use-all-data",
        type=str,
        default=None,
        help="Use all data for training without validation split ('true'/'false')",
    )


def add_cross_validation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add cross-validation related arguments to parser."""
    parser.add_argument(
        "--fold-idx",
        type=int,
        default=None,
        help="Fold index (0 to k-1) for cross-validation training",
    )
    parser.add_argument(
        "--fold-splits-file",
        type=str,
        default=None,
        help="Path to JSON file containing fold splits",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=None,
        help="Number of folds for cross-validation",
    )


def add_api_server_arguments(parser: argparse.ArgumentParser) -> None:
    """Add API server configuration arguments to parser."""
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port number",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )


def validate_config_dir(config_dir: str) -> Path:
    """
    Validate and return config directory path.
    
    Args:
        config_dir: Path to configuration directory.
    
    Returns:
        Path object for config directory.
    
    Raises:
        FileNotFoundError: If config directory does not exist.
    """
    config_path = Path(config_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Config directory not found: {config_path}")
    return config_path

