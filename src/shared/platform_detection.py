"""Platform detection utilities for Colab, Kaggle, and local environments."""

import os
from pathlib import Path
from typing import Optional


def detect_platform() -> str:
    """
    Detect execution platform: 'colab', 'kaggle', 'azure', or 'local'.
    
    Returns:
        Platform identifier string: 'colab', 'kaggle', 'azure', or 'local'
    """
    # Check for Google Colab
    if "COLAB_GPU" in os.environ or "COLAB_TPU" in os.environ:
        return "colab"
    
    # Check for Kaggle
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    
    # Check for Azure ML
    if "AZURE_ML_RUN_ID" in os.environ or "AZURE_ML_OUTPUT_DIR" in os.environ:
        return "azure"
    
    # Default to local
    return "local"


def resolve_checkpoint_path(base_path: Path, relative_path: str) -> Path:
    """
    Resolve checkpoint path with platform-specific optimizations.
    
    For Colab: Prefers Drive mount path if available (/content/drive/MyDrive/...)
    For Kaggle: Uses /kaggle/working/ (automatically persisted)
    For Local: Uses provided base path
    
    Args:
        base_path: Base path for checkpoint storage
        relative_path: Relative path from base (e.g., "hpo/distilbert/study.db")
    
    Returns:
        Resolved absolute Path for checkpoint storage
    """
    platform = detect_platform()
    
    if platform == "colab":
        # Check if Google Drive is mounted
        drive_path = Path("/content/drive/MyDrive")
        if drive_path.exists() and drive_path.is_dir():
            # Use Drive for persistence across sessions
            # Create a checkpoint directory in Drive
            checkpoint_base = drive_path / "resume-ner-checkpoints"
            return checkpoint_base / relative_path
        else:
            # Fallback to /content/ if Drive not mounted
            return Path("/content") / relative_path
    
    elif platform == "kaggle":
        # Kaggle outputs in /kaggle/working/ are automatically persisted
        # If base_path is already under /kaggle/working, use it
        base_str = str(base_path)
        if base_str.startswith("/kaggle/working"):
            return base_path / relative_path
        else:
            # Otherwise, use /kaggle/working as base
            return Path("/kaggle/working") / relative_path
    
    else:
        # Local: use provided base path
        return base_path / relative_path






