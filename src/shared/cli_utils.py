"""Common CLI utilities for argument parsing and validation."""

import argparse
import sys
from pathlib import Path
from typing import Tuple


def add_model_path_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add common model path arguments to an argument parser.
    
    Args:
        parser: ArgumentParser instance to add arguments to.
    """
    parser.add_argument(
        "--onnx-model",
        type=str,
        required=True,
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (for tokenizer and config)",
    )


def validate_model_paths(onnx_path: Path, checkpoint_dir: Path) -> Tuple[Path, Path]:
    """
    Validate that model paths exist and return Path objects.
    
    Args:
        onnx_path: Path to ONNX model file.
        checkpoint_dir: Path to checkpoint directory.
    
    Returns:
        Tuple of (onnx_path, checkpoint_dir) as Path objects.
    
    Raises:
        FileNotFoundError: If either path does not exist.
    """
    onnx_path = Path(onnx_path)
    checkpoint_dir = Path(checkpoint_dir)
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    return onnx_path, checkpoint_dir


def parse_model_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    """
    Parse and validate model paths from command-line arguments.
    
    Args:
        args: Parsed command-line arguments with onnx_model and checkpoint attributes.
    
    Returns:
        Tuple of (onnx_path, checkpoint_dir) as Path objects.
    
    Raises:
        FileNotFoundError: If either path does not exist.
    """
    onnx_path = Path(args.onnx_model)
    checkpoint_dir = Path(args.checkpoint)
    return validate_model_paths(onnx_path, checkpoint_dir)

