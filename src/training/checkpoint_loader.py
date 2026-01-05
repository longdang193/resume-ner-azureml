"""Checkpoint path resolution and validation utilities."""

import os
from pathlib import Path
from typing import Dict, Any, Optional


def validate_checkpoint(checkpoint_path: Path) -> bool:
    """
    Validate that a checkpoint directory exists and contains model files.
    
    Args:
        checkpoint_path: Path to checkpoint directory.
    
    Returns:
        True if checkpoint is valid, False otherwise.
    """
    if not checkpoint_path.exists():
        return False
    
    if not checkpoint_path.is_dir():
        return False
    
    # Check for required model files (config.json and model files)
    config_file = checkpoint_path / "config.json"
    # Check for either pytorch_model.bin, model.safetensors, or model.bin
    model_files = [
        checkpoint_path / "pytorch_model.bin",
        checkpoint_path / "model.safetensors",
        checkpoint_path / "model.bin",
    ]
    
    has_config = config_file.exists()
    has_model = any(f.exists() for f in model_files)
    
    return has_config and has_model


def resolve_checkpoint_path(
    config: Dict[str, Any],
    previous_cache_path: Optional[Path] = None,
    backbone: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Optional[Path]:
    """
    Resolve checkpoint path from config, environment, or cache.
    
    Priority order:
    1. CHECKPOINT_PATH environment variable
    2. config["training"]["checkpoint"]["source_path"] (with pattern resolution)
    3. previous_cache_path (if provided, loads from cache JSON)
    
    Args:
        config: Configuration dictionary containing checkpoint settings.
        previous_cache_path: Optional path to previous training cache JSON file.
        backbone: Optional backbone name for pattern resolution (e.g., "distilbert").
        run_id: Optional run ID for pattern resolution (e.g., "20251227_220407").
    
    Returns:
        Resolved checkpoint directory Path, or None if no checkpoint should be loaded.
    """
    # Priority 1: Environment variable
    env_checkpoint = os.environ.get("CHECKPOINT_PATH")
    if env_checkpoint:
        checkpoint_path = Path(env_checkpoint)
        if validate_checkpoint(checkpoint_path):
            return checkpoint_path.resolve()
        # If invalid, continue to next priority
    
    # Priority 2: Config file
    checkpoint_config = config.get("training", {}).get("checkpoint", {})
    if checkpoint_config:
        source_path = checkpoint_config.get("source_path")
        if source_path:
            # Resolve patterns like {backbone}_{run_id}
            if backbone and run_id:
                source_path = source_path.replace("{backbone}", backbone)
                source_path = source_path.replace("{run_id}", run_id)
            elif backbone:
                source_path = source_path.replace("{backbone}", backbone)
            
            # Resolve relative paths (relative to project root or config dir)
            checkpoint_path = Path(source_path)
            if not checkpoint_path.is_absolute():
                # Try relative to current working directory (project root)
                checkpoint_path = Path.cwd() / checkpoint_path
                if not checkpoint_path.exists():
                    # Try relative to config directory
                    config_dir = Path(config.get("_config_dir", Path.cwd() / "config"))
                    checkpoint_path = config_dir.parent / source_path
            
            checkpoint_path = checkpoint_path.resolve()
            if validate_checkpoint(checkpoint_path):
                return checkpoint_path
    
    # Priority 3: Previous training cache
    if previous_cache_path and previous_cache_path.exists():
        try:
            import json
            with open(previous_cache_path, "r") as f:
                cache_data = json.load(f)
            
            output_dir = cache_data.get("output_dir")
            if output_dir:
                checkpoint_path = Path(output_dir) / "checkpoint"
                checkpoint_path = checkpoint_path.resolve()
                if validate_checkpoint(checkpoint_path):
                    return checkpoint_path
        except (json.JSONDecodeError, KeyError, OSError):
            # Cache file invalid or unreadable, continue
            pass
    
    # No valid checkpoint found
    return None

