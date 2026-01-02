"""Checkpoint manager for HPO study persistence."""

from pathlib import Path
from typing import Any, Dict, Optional

from shared.platform_detection import detect_platform, resolve_checkpoint_path


def resolve_storage_path(
    output_dir: Path,
    checkpoint_config: Dict[str, Any],
    backbone: str,
    study_name: Optional[str] = None,
) -> Optional[Path]:
    """
    Resolve checkpoint storage path with platform awareness.
    
    Args:
        output_dir: Base output directory for HPO trials
        checkpoint_config: Checkpoint configuration from HPO config
        backbone: Model backbone name (for placeholder substitution)
        study_name: Optional resolved study name (for {study_name} placeholder)
    
    Returns:
        Resolved Path for checkpoint storage, or None if checkpointing disabled
    """
    # Check if checkpointing is enabled
    enabled = checkpoint_config.get("enabled", False)
    if not enabled:
        return None
    
    # Get storage path from config or use default
    storage_path_template = checkpoint_config.get(
        "storage_path",
        f"{{backbone}}/study.db"  # Default: relative to output_dir
    )
    
    # Replace placeholders in order: {backbone} first, then {study_name}
    storage_path_str = storage_path_template.replace("{backbone}", backbone)
    if study_name:
        storage_path_str = storage_path_str.replace("{study_name}", study_name)
    
    # Resolve with platform-specific optimizations
    platform = detect_platform()
    storage_path = resolve_checkpoint_path(output_dir, storage_path_str)
    
    # Ensure parent directory exists
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    return storage_path


def get_storage_uri(storage_path: Optional[Path]) -> Optional[str]:
    """
    Convert storage path to Optuna storage URI.
    
    Args:
        storage_path: Path to SQLite database file, or None for in-memory
    
    Returns:
        Optuna storage URI string (e.g., "sqlite:///path/to/study.db"), or None
    """
    if storage_path is None:
        return None
    
    # Convert to absolute path and use 3 slashes for absolute paths
    abs_path = storage_path.resolve()
    return f"sqlite:///{abs_path}"


from pathlib import Path
from typing import Any, Dict, Optional

from shared.platform_detection import detect_platform, resolve_checkpoint_path


def resolve_storage_path(
    output_dir: Path,
    checkpoint_config: Dict[str, Any],
    backbone: str,
    study_name: Optional[str] = None,
) -> Optional[Path]:
    """
    Resolve checkpoint storage path with platform awareness.
    
    Args:
        output_dir: Base output directory for HPO trials
        checkpoint_config: Checkpoint configuration from HPO config
        backbone: Model backbone name (for placeholder substitution)
        study_name: Optional resolved study name (for {study_name} placeholder)
    
    Returns:
        Resolved Path for checkpoint storage, or None if checkpointing disabled
    """
    # Check if checkpointing is enabled
    enabled = checkpoint_config.get("enabled", False)
    if not enabled:
        return None
    
    # Get storage path from config or use default
    storage_path_template = checkpoint_config.get(
        "storage_path",
        f"{{backbone}}/study.db"  # Default: relative to output_dir
    )
    
    # Replace placeholders in order: {backbone} first, then {study_name}
    storage_path_str = storage_path_template.replace("{backbone}", backbone)
    if study_name:
        storage_path_str = storage_path_str.replace("{study_name}", study_name)
    
    # Resolve with platform-specific optimizations
    platform = detect_platform()
    storage_path = resolve_checkpoint_path(output_dir, storage_path_str)
    
    # Ensure parent directory exists
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    return storage_path


def get_storage_uri(storage_path: Optional[Path]) -> Optional[str]:
    """
    Convert storage path to Optuna storage URI.
    
    Args:
        storage_path: Path to SQLite database file, or None for in-memory
    
    Returns:
        Optuna storage URI string (e.g., "sqlite:///path/to/study.db"), or None
    """
    if storage_path is None:
        return None
    
    # Convert to absolute path and use 3 slashes for absolute paths
    abs_path = storage_path.resolve()
    return f"sqlite:///{abs_path}"

