"""Helper functions for HPO sweep orchestration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from common.shared.logging_utils import get_logger

logger = get_logger(__name__)


def generate_run_id() -> str:
    """
    Generate unique run ID (timestamp-based) to prevent overwriting on reruns.

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_checkpoint_storage(
    output_dir: Path,
    checkpoint_config: Optional[Dict[str, Any]],
    backbone: str,
    study_name: Optional[str] = None,
    restore_from_drive: Optional[Callable[[Path], bool]] = None,
) -> Tuple[Optional[Path], Optional[str], bool]:
    """
    Set up checkpoint storage and determine if resuming.

    Args:
        output_dir: Base output directory.
        checkpoint_config: Checkpoint configuration dictionary.
        backbone: Model backbone name.
        study_name: Optional resolved study name (for {study_name} placeholder).
        restore_from_drive: Optional function to restore checkpoint from Drive if missing.
                          Function should take a Path and return bool (True if restored).

    Returns:
        Tuple of (storage_path, storage_uri, should_resume).
    """
    # Lazy import to avoid circular dependency
    from training.hpo.checkpoint.storage import get_storage_uri, resolve_storage_path

    checkpoint_config = checkpoint_config or {}
    storage_path = resolve_storage_path(
        output_dir=output_dir,
        checkpoint_config=checkpoint_config,
        backbone=backbone,
        study_name=study_name,
    )
    storage_uri = get_storage_uri(storage_path)

    # If local checkpoint missing and restore_from_drive provided, attempt restore
    if storage_path is not None and not storage_path.exists() and restore_from_drive is not None:
        try:
            restored = restore_from_drive(storage_path)
            if restored:
                logger.info(
                    f"Restored HPO checkpoint from Drive: {storage_path}")
            else:
                logger.debug(
                    f"Drive backup not found for checkpoint: {storage_path}")
        except Exception as e:
            logger.warning(f"Failed to restore checkpoint from Drive: {e}")

    # Determine if we should resume
    auto_resume = (
        checkpoint_config.get("auto_resume", True)
        if checkpoint_config.get("enabled", False)
        else False
    )
    should_resume = (
        auto_resume
        and storage_path is not None
        and storage_path.exists()
    )

    return storage_path, storage_uri, should_resume


def create_study_name(
    backbone: str,
    run_id: str,
    should_resume: bool,
    checkpoint_config: Optional[Dict[str, Any]] = None,
    hpo_config: Optional[Dict[str, Any]] = None,
    run_mode: Optional[str] = None,
    root_dir: Optional[Path] = None,
    config_dir: Optional[Path] = None,
) -> str:
    """
    Create Optuna study name (simplified version - no automatic variant computation).
    
    Simple approach: Use study_name from config as-is, or default to hpo_{backbone}.
    User controls versioning by explicitly specifying study_name (e.g., hpo_distilbert_v2).

    Args:
        backbone: Model backbone name.
        run_id: Unique run ID.
        should_resume: Whether resuming from checkpoint.
        checkpoint_config: Optional checkpoint configuration dictionary.
        hpo_config: Optional HPO configuration dictionary.
        run_mode: Optional run mode (unused, kept for compatibility).
        root_dir: Optional project root directory (unused, kept for compatibility).
        config_dir: Optional config directory (unused, kept for compatibility).

    Returns:
        Study name string (from config or default).
    """
    checkpoint_config = checkpoint_config or {}
    hpo_config = hpo_config or {}
    checkpoint_enabled = checkpoint_config.get("enabled", False)
    
    # Check for custom study_name in checkpoint config first, then HPO config
    study_name_template = checkpoint_config.get("study_name") or hpo_config.get("study_name")

    if study_name_template:
        # Use custom study name as-is (replace {backbone} placeholder if present)
        study_name = study_name_template.replace("{backbone}", backbone)
        return study_name

    # Default behavior when no custom study_name is provided
    if checkpoint_enabled:
        # When checkpointing is enabled, always use consistent name so it can be resumed
        # This allows future runs to resume even if the checkpoint file doesn't exist yet
        return f"hpo_{backbone}"
    elif should_resume:
        # Use base name to resume existing study (for backward compatibility)
        return f"hpo_{backbone}"
    else:
        # Use unique name for fresh start (only when checkpointing is disabled)
        return f"hpo_{backbone}_{run_id}"


def find_study_variants(
    output_dir: Path,
    backbone: str,
) -> list[str]:
    """
    Find all study variants for a given backbone.
    
    Uses shared variants.py module (DRY).
    
    Scans output directory for study folders matching pattern:
    - hpo_{backbone} (variant 1, implicit)
    - hpo_{backbone}_v1, hpo_{backbone}_v2, etc.
    
    Args:
        output_dir: HPO output directory (backbone-level directory).
        backbone: Model backbone name.
    
    Returns:
        List of variant names (study folder names).
    """
    base_name = f"hpo_{backbone}"
    variants = []
    
    if not output_dir.exists():
        return variants
    
    for item in output_dir.iterdir():
        if not item.is_dir():
            continue
        
        folder_name = item.name
        if folder_name == base_name:
            variants.append(base_name)
        elif folder_name.startswith(f"{base_name}_v"):
            variants.append(folder_name)
    
    return sorted(variants)


def create_mlflow_run_name(
    backbone: str,
    run_id: str,
    study_name: Optional[str] = None,
    should_resume: bool = False,
    checkpoint_enabled: bool = False,
) -> str:
    """
    Create MLflow run name for HPO sweep.

    Args:
        backbone: Model backbone name.
        run_id: Unique run ID.
        study_name: Optional study name (used when checkpointing is enabled).
        should_resume: Whether resuming from checkpoint.
        checkpoint_enabled: Whether checkpointing is enabled.

    Returns:
        MLflow run name string.
    """
    # When checkpointing is enabled, always use study_name (for both new and resumed runs)
    if checkpoint_enabled and study_name:
        return study_name
    elif should_resume and study_name:
        # When resuming without checkpointing, use study_name
        return study_name
    else:
        # For new runs without checkpointing, use unique name with run_id
        return f"hpo_{backbone}_{run_id}"
