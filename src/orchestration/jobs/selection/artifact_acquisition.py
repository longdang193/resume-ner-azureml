"""
Artifact acquisition utilities for best model selection.

This module provides robust checkpoint acquisition with local-first priority,
checkpoint validation, and graceful handling of Azure ML compatibility issues.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import os
import tarfile
import mlflow
import json
import shutil

from shared.logging_utils import get_logger

logger = get_logger(__name__)


def _extract_tar_gz(tar_path: Path, extract_to: Optional[Path] = None) -> Path:
    """
    Extract a tar.gz file and return the path to the extracted directory.

    Args:
        tar_path: Path to the tar.gz file
        extract_to: Directory to extract to (defaults to same directory as tar file)

    Returns:
        Path to the extracted directory containing checkpoint files
    """
    if extract_to is None:
        extract_to = tar_path.parent

    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        if members:
            tar.extractall(path=extract_to)

            # If archive has a single root directory, return that
            root_names = {Path(m.name).parts[0] for m in members if m.name}
            if len(root_names) == 1:
                root_name = list(root_names)[0]
                extracted_path = extract_to / root_name
                if extracted_path.exists():
                    return extracted_path

            return extract_to

    return extract_to


def _validate_checkpoint(checkpoint_path: Path) -> bool:
    """
    Validate checkpoint integrity by checking for essential files.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        True if checkpoint appears valid, False otherwise
    """
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        return False

    # Check for common checkpoint files (PyTorch/HuggingFace)
    essential_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "model.bin",
        "pytorch_model.bin.index.json",
    ]

    has_model_file = any((checkpoint_path / fname).exists()
                         for fname in essential_files)
    has_config = (checkpoint_path / "config.json").exists()

    return has_model_file or has_config


def _find_checkpoint_in_directory(directory: Path) -> Optional[Path]:
    """
    Search for a valid checkpoint directory within the given directory.

    Args:
        directory: Directory to search in

    Returns:
        Path to valid checkpoint directory, or None if not found
    """
    if not directory.is_dir():
        return None

    # Check if directory itself contains checkpoint files
    if _validate_checkpoint(directory):
        return directory

    # Check for checkpoint subdirectory
    checkpoint_subdir = directory / "checkpoint"
    if checkpoint_subdir.exists() and checkpoint_subdir.is_dir() and _validate_checkpoint(checkpoint_subdir):
        return checkpoint_subdir

    # Search recursively for any directory with checkpoint files
    for item in directory.rglob("*"):
        if item.is_dir() and _validate_checkpoint(item):
            return item

    return None


def _build_checkpoint_dir(
    root_dir: Path,
    config_dir: Path,
    environment: str,
    backbone: str,
    artifact_run_id: str,
    study_key_hash: Optional[str] = None,
    trial_key_hash: Optional[str] = None,
) -> Path:
    """
    Build systematic checkpoint directory path using centralized paths config.

    Uses stable naming based on (study_key_hash, trial_key_hash) when available,
    falling back to run_id for backward compatibility.

    Path structure:
      - Preferred: outputs/best_model_selection/{environment}/{backbone}/sel_{study_hash[:8]}_{trial_hash[:8]}/
      - Fallback: outputs/best_model_selection/{environment}/{backbone}/run_{run_id[:8]}/

    Args:
        root_dir: Project root directory
        config_dir: Config directory
        environment: Execution environment (local, colab, kaggle)
        backbone: Model backbone name
        artifact_run_id: MLflow run ID (for fallback)
        study_key_hash: Study key hash (preferred for stable naming)
        trial_key_hash: Trial key hash (preferred for stable naming)

    Returns:
        Path to checkpoint directory
    """
    from orchestration.paths import resolve_output_path

    base_dir = resolve_output_path(
        root_dir, config_dir, "best_model_selection") / environment / backbone

    if study_key_hash and trial_key_hash:
        return base_dir / f"sel_{study_key_hash[:8]}_{trial_key_hash[:8]}"

    return base_dir / f"run_{artifact_run_id[:8]}"


def _find_checkpoint_in_drive_by_hash(
    drive_hpo_dir: Path,
    study_key_hash: str,
    trial_key_hash: str,
) -> Optional[Path]:
    """
    Find checkpoint in Drive by scanning Drive directory structure directly.
    This avoids restoring the entire HPO directory structure.

    Args:
        drive_hpo_dir: Drive path to HPO backbone directory
        study_key_hash: Target study key hash
        trial_key_hash: Target trial key hash

    Returns:
        Path to checkpoint directory in Drive, or None if not found
    """
    if not drive_hpo_dir.exists():
        return None

    # Scan study folders in Drive
    for study_folder in drive_hpo_dir.iterdir():
        if not study_folder.is_dir() or study_folder.name.startswith("trial_"):
            continue

        # Scan trial folders
        for trial_dir in study_folder.iterdir():
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue

            # Read trial metadata from Drive
            trial_meta_path = trial_dir / "trial_meta.json"
            if not trial_meta_path.exists():
                continue

            try:
                with open(trial_meta_path, "r") as f:
                    meta = json.load(f)

                # Match by hashes
                if (meta.get("study_key_hash") == study_key_hash and
                        meta.get("trial_key_hash") == trial_key_hash):
                    # Found match! Get checkpoint path (prefer refit, else best CV fold)
                    refit_checkpoint = trial_dir / "refit" / "checkpoint"
                    if refit_checkpoint.exists():
                        return refit_checkpoint

                    # Check CV folds
                    cv_dir = trial_dir / "cv"
                    if cv_dir.exists():
                        fold_dirs = [d for d in cv_dir.iterdir()
                                     if d.is_dir() and d.name.startswith("fold")]
                        if fold_dirs:
                            # Return first fold checkpoint (or implement best fold selection)
                            for fold_dir in sorted(fold_dirs):
                                fold_checkpoint = fold_dir / "checkpoint"
                                if fold_checkpoint.exists():
                                    return fold_checkpoint

                    # Fallback
                    checkpoint = trial_dir / "checkpoint"
                    if checkpoint.exists():
                        return checkpoint
            except Exception:
                continue

    return None


def _get_azure_ml_info(config_dir: Path, root_dir: Path, tracking_uri: str) -> tuple[str, str]:
    """
    Extract Azure ML workspace name and resource group from config files.

    Returns:
        Tuple of (workspace_name, resource_group)
    """
    workspace_name = "<workspace-name>"
    resource_group = "<resource-group>"

    try:
        from shared.yaml_utils import load_yaml
        from shared.mlflow_setup import _load_env_file
        import re

        config_dir_path = Path(config_dir) if isinstance(
            config_dir, str) else config_dir

        # Get workspace name from config
        try:
            mlflow_config_path = config_dir_path / "mlflow.yaml"
            if mlflow_config_path.exists():
                mlflow_config = load_yaml(mlflow_config_path)
                workspace_name = mlflow_config.get("azure_ml", {}).get(
                    "workspace_name", "<workspace-name>")
        except Exception:
            pass

        # Get resource group - try multiple sources
        resource_group = os.getenv("AZURE_RESOURCE_GROUP") or ""
        if not resource_group:
            # Try loading from config.env file
            possible_paths = [
                root_dir / "config.env",
                config_dir_path / "config.env",
                config_dir_path.parent / "config.env",
                Path.cwd() / "config.env",
            ]
            for config_env_path in possible_paths:
                if config_env_path.exists():
                    env_vars = _load_env_file(config_env_path)
                    resource_group = env_vars.get("AZURE_RESOURCE_GROUP", "")
                    if resource_group:
                        resource_group = resource_group.strip('"\'')
                        break

        # Try extracting from tracking URI
        if not resource_group and tracking_uri and "azureml://" in tracking_uri:
            patterns = [
                r'/resourceGroups/([^/]+)/',
                r'resourceGroups/([^/]+)',
                r'resourceGroup=([^&]+)',
            ]
            for pattern in patterns:
                rg_match = re.search(pattern, tracking_uri)
                if rg_match:
                    resource_group = rg_match.group(1)
                    break

        # Try config files
        if not resource_group:
            try:
                infra_config_path = config_dir_path / "infrastructure.yaml"
                if infra_config_path.exists():
                    infra_config = load_yaml(infra_config_path)
                    rg_config = infra_config.get(
                        "azure", {}).get("resource_group", "")
                    if rg_config.startswith("${") and rg_config.endswith("}"):
                        env_var = rg_config[2:-1]
                        resource_group = os.getenv(env_var, "")
                    else:
                        resource_group = rg_config or ""
            except Exception:
                pass

        if not resource_group:
            resource_group = "<resource-group>"
        if not workspace_name or workspace_name == "":
            workspace_name = "<workspace-name>"

    except Exception:
        pass

    return workspace_name, resource_group


def acquire_best_model_checkpoint(
    best_run_info: Dict[str, Any],
    root_dir: Path,
    config_dir: Path,
    acquisition_config: Dict[str, Any],
    selection_config: Dict[str, Any],
    platform: str,
    restore_from_drive: Optional[Callable[[Path, bool], bool]] = None,
    drive_store: Optional[Any] = None,
    in_colab: bool = False,
) -> Path:
    """
    Acquire checkpoint using local-first fallback strategy.

    Priority order (from config):
    1. Local disk (by config + backbone) - PREFERRED to avoid Azure ML issues
    2. Drive restore (Colab only) - scans Drive metadata, restores only checkpoint
    3. MLflow download

    Args:
        best_run_info: Dictionary with best run information (must include study_key_hash, trial_key_hash, run_id, backbone)
        root_dir: Project root directory
        config_dir: Config directory
        acquisition_config: Artifact acquisition configuration
        selection_config: Best model selection configuration
        platform: Platform name (local, colab, kaggle)
        restore_from_drive: Optional function to restore from Drive backup
        drive_store: Optional DriveBackupStore instance for direct Drive access
        in_colab: Whether running in Google Colab

    Returns:
        Path to validated checkpoint directory

    Raises:
        ValueError: If all fallback strategies fail
    """
    run_id = best_run_info["run_id"]
    study_key_hash = best_run_info.get("study_key_hash")
    trial_key_hash = best_run_info.get("trial_key_hash")
    backbone = best_run_info.get("backbone", "unknown")
    experiment_name = best_run_info.get("experiment_name", "unknown")

    priority = acquisition_config["priority"]
    checkpoint_path = None

    logger.info(
        f"Acquiring checkpoint for run {run_id[:8]}... (backbone={backbone}, platform={platform})")
    logger.info(f"Priority order: {priority}")
    logger.info(
        f"Drive restore available: restore_from_drive={restore_from_drive is not None}, drive_store={drive_store is not None}, in_colab={in_colab}")

    # Strategy 1: Local disk selection (using hashes from best_model)
    if "local" in priority and checkpoint_path is None:
        logger.info("Strategy 1: Attempting local disk selection...")
        try:
            from orchestration.jobs.local_selection_v2 import find_trial_checkpoint_by_hash

            if study_key_hash and trial_key_hash:
                hpo_output_dir = root_dir / "outputs" / "hpo" / platform / backbone
                found_path = find_trial_checkpoint_by_hash(
                    hpo_backbone_dir=hpo_output_dir,
                    study_key_hash=study_key_hash,
                    trial_key_hash=trial_key_hash,
                )

                if found_path:
                    found_path = Path(found_path)
                    logger.info(
                        f"Strategy 1: Found checkpoint locally at {found_path}")

                    if acquisition_config["local"].get("validate", True):
                        if _validate_checkpoint(found_path):
                            # Copy to best_model_selection directory for consistency
                            target_path = _build_checkpoint_dir(
                                root_dir, config_dir, platform, backbone, run_id,
                                study_key_hash=study_key_hash, trial_key_hash=trial_key_hash
                            )
                            if found_path != target_path:
                                target_path.mkdir(parents=True, exist_ok=True)
                                if target_path.exists():
                                    shutil.rmtree(target_path)
                                shutil.copytree(found_path, target_path)
                            checkpoint_path = target_path
                            logger.info(
                                f"Strategy 1: Successfully acquired checkpoint from local disk: {checkpoint_path}")
                    else:
                        if found_path.exists():
                            # Copy to best_model_selection directory for consistency
                            target_path = _build_checkpoint_dir(
                                root_dir, config_dir, platform, backbone, run_id,
                                study_key_hash=study_key_hash, trial_key_hash=trial_key_hash
                            )
                            if found_path != target_path:
                                target_path.mkdir(parents=True, exist_ok=True)
                                if target_path.exists():
                                    shutil.rmtree(target_path)
                                shutil.copytree(found_path, target_path)
                            checkpoint_path = target_path
                            logger.info(
                                f"Strategy 1: Successfully acquired checkpoint from local disk: {checkpoint_path}")
                else:
                    logger.info(
                        f"Strategy 1: Checkpoint not found locally (study_key_hash={study_key_hash[:8] if study_key_hash else None}..., trial_key_hash={trial_key_hash[:8] if trial_key_hash else None}...)")

        except Exception as e:
            logger.warning(f"Strategy 1 failed: {type(e).__name__}: {e}")

    # Strategy 2: Drive restore (Colab only) - optimized to scan Drive metadata and restore only checkpoint
    drive_strategy_enabled = (
        "drive" in priority and
        restore_from_drive is not None and
        in_colab and
        acquisition_config.get("drive", {}).get("enabled", False) and
        checkpoint_path is None
    )

    logger.info(f"Strategy 2: Drive restore - enabled={drive_strategy_enabled} "
                f"(drive_in_priority={'drive' in priority}, restore_from_drive={restore_from_drive is not None}, "
                f"in_colab={in_colab}, drive_enabled={acquisition_config.get('drive', {}).get('enabled', False)}, "
                f"checkpoint_path={checkpoint_path is None})")

    if drive_strategy_enabled:
        logger.info("Strategy 2: Attempting Drive restore...")
        try:
            # OPTIMIZATION: If drive_store is available, scan Drive directly to find checkpoint path
            # This avoids restoring the entire HPO directory structure
            if study_key_hash and trial_key_hash and drive_store:
                logger.info(
                    "Strategy 2: Using optimized Drive scanning (drive_store available)")
                hpo_output_dir = root_dir / "outputs" / "hpo" / platform / backbone
                logger.info(
                    f"Strategy 2: Scanning Drive HPO directory: {hpo_output_dir}")

                # Compute Drive path for HPO directory
                try:
                    drive_hpo_dir = drive_store.drive_path_for(hpo_output_dir)
                    logger.info(f"Strategy 2: Drive HPO path: {drive_hpo_dir}")
                except ValueError as e:
                    drive_hpo_dir = None
                    logger.warning(
                        f"Strategy 2: Could not compute Drive path: {e}")

                if drive_hpo_dir:
                    # Try using find_trial_checkpoint_by_hash directly on Drive path first (fastest)
                    # This works because Drive is mounted and accessible as a normal filesystem
                    if drive_hpo_dir.exists():
                        logger.info(
                            f"Strategy 2: Drive HPO directory exists, scanning for checkpoint...")
                        from orchestration.jobs.local_selection_v2 import find_trial_checkpoint_by_hash
                        
                        # Use the same function that works locally, but on Drive path
                        drive_checkpoint_path = find_trial_checkpoint_by_hash(
                            hpo_backbone_dir=drive_hpo_dir,
                            study_key_hash=study_key_hash,
                            trial_key_hash=trial_key_hash,
                        )
                        
                        if drive_checkpoint_path:
                            drive_checkpoint_path = Path(drive_checkpoint_path)

                        if drive_checkpoint_path and drive_checkpoint_path.exists():
                            logger.info(
                                f"Strategy 2: Found checkpoint in Drive at {drive_checkpoint_path}")
                            # Found checkpoint in Drive! Copy to best_model_selection cache dir
                            local_checkpoint_path = _build_checkpoint_dir(
                                root_dir, config_dir, platform, backbone, run_id,
                                study_key_hash=study_key_hash, trial_key_hash=trial_key_hash
                            )
                            logger.info(
                                f"Strategy 2: Copying checkpoint to {local_checkpoint_path}")

                            # Direct copy from Drive path to local destination
                            local_checkpoint_path.mkdir(
                                parents=True, exist_ok=True)
                            if local_checkpoint_path.exists():
                                shutil.rmtree(local_checkpoint_path)
                            shutil.copytree(drive_checkpoint_path,
                                            local_checkpoint_path)

                            # Validate restored checkpoint
                            if acquisition_config["drive"].get("validate", True):
                                if _validate_checkpoint(local_checkpoint_path):
                                    checkpoint_path = local_checkpoint_path
                                    logger.info(
                                        f"Strategy 2: Successfully restored and validated checkpoint from Drive: {checkpoint_path}")
                                else:
                                    logger.warning(
                                        f"Strategy 2: Restored checkpoint failed validation: {local_checkpoint_path}")
                            else:
                                if local_checkpoint_path.exists():
                                    checkpoint_path = local_checkpoint_path
                                    logger.info(
                                        f"Strategy 2: Successfully restored checkpoint from Drive: {checkpoint_path}")
                        else:
                            logger.info(
                                f"Strategy 2: Checkpoint not found in Drive HPO directory scan (study_key_hash={study_key_hash[:8] if study_key_hash else None}..., trial_key_hash={trial_key_hash[:8] if trial_key_hash else None}...)")
                    else:
                        logger.info(
                            f"Strategy 2: Drive HPO directory does not exist: {drive_hpo_dir}")
                        # Try scanning broader Drive structure - maybe checkpoints are in a different location
                        logger.info(
                            f"Strategy 2: Attempting broader Drive scan from backup root...")
                        try:
                            backup_root = drive_store.backup_root
                            # Try scanning outputs/hpo from backup root
                            backup_hpo_base = backup_root / "outputs" / "hpo"
                            if backup_hpo_base.exists():
                                logger.info(
                                    f"Strategy 2: Found outputs/hpo in Drive, scanning all backbones...")
                                # Scan all platforms and backbones
                                for platform_dir in backup_hpo_base.iterdir():
                                    if not platform_dir.is_dir():
                                        continue
                                    for backbone_dir in platform_dir.iterdir():
                                        if not backbone_dir.is_dir():
                                            continue
                                        drive_checkpoint_path = _find_checkpoint_in_drive_by_hash(
                                            backbone_dir, study_key_hash, trial_key_hash
                                        )
                                        if drive_checkpoint_path and drive_checkpoint_path.exists():
                                            logger.info(
                                                f"Strategy 2: Found checkpoint in Drive at {drive_checkpoint_path} (scanned from {backbone_dir})")
                                            # Found checkpoint! Copy to best_model_selection cache dir
                                            local_checkpoint_path = _build_checkpoint_dir(
                                                root_dir, config_dir, platform, backbone, run_id,
                                                study_key_hash=study_key_hash, trial_key_hash=trial_key_hash
                                            )
                                            logger.info(
                                                f"Strategy 2: Copying checkpoint to {local_checkpoint_path}")

                                            # Direct copy from Drive path to local destination
                                            local_checkpoint_path.mkdir(
                                                parents=True, exist_ok=True)
                                            if local_checkpoint_path.exists():
                                                shutil.rmtree(
                                                    local_checkpoint_path)
                                            shutil.copytree(drive_checkpoint_path,
                                                            local_checkpoint_path)

                                            # Validate restored checkpoint
                                            if acquisition_config["drive"].get("validate", True):
                                                if _validate_checkpoint(local_checkpoint_path):
                                                    checkpoint_path = local_checkpoint_path
                                                    logger.info(
                                                        f"Strategy 2: Successfully restored and validated checkpoint from Drive: {checkpoint_path}")
                                                    break
                                            else:
                                                if local_checkpoint_path.exists():
                                                    checkpoint_path = local_checkpoint_path
                                                    logger.info(
                                                        f"Strategy 2: Successfully restored checkpoint from Drive: {checkpoint_path}")
                                                    break
                                    if checkpoint_path:
                                        break
                                if not checkpoint_path:
                                    logger.info(
                                        f"Strategy 2: Checkpoint not found in broader Drive scan")
                            else:
                                logger.info(
                                    f"Strategy 2: outputs/hpo directory does not exist in Drive backup root: {backup_hpo_base}")
                        except Exception as e:
                            logger.warning(
                                f"Strategy 2: Broader Drive scan failed: {type(e).__name__}: {e}")
                else:
                    logger.info(
                        f"Strategy 2: Could not compute Drive HPO path")
            else:
                logger.info(
                    f"Strategy 2: Skipping optimized scan (study_key_hash={study_key_hash is not None}, trial_key_hash={trial_key_hash is not None}, drive_store={drive_store is not None})")

            # Fallback: Try best_model_selection directory structure (if checkpoint was manually moved)
            # This works even if drive_store is None (uses restore_from_drive wrapper)
            if checkpoint_path is None:
                logger.info(
                    "Strategy 2: Trying fallback - restore from best_model_selection directory structure")
                local_checkpoint_path = _build_checkpoint_dir(
                    root_dir, config_dir, platform, backbone, run_id,
                    study_key_hash=study_key_hash, trial_key_hash=trial_key_hash
                )
                logger.info(
                    f"Strategy 2: Attempting to restore from Drive path: {local_checkpoint_path}")

                if restore_from_drive(local_checkpoint_path, is_directory=True):
                    logger.info(
                        f"Strategy 2: Restore function returned True, validating checkpoint...")
                    if acquisition_config["drive"].get("validate", True):
                        if _validate_checkpoint(local_checkpoint_path):
                            checkpoint_path = local_checkpoint_path
                            logger.info(
                                f"Strategy 2: Successfully restored and validated checkpoint from Drive (fallback): {checkpoint_path}")
                        else:
                            logger.warning(
                                f"Strategy 2: Restored checkpoint failed validation (fallback): {local_checkpoint_path}")
                    else:
                        if local_checkpoint_path.exists():
                            checkpoint_path = local_checkpoint_path
                            logger.info(
                                f"Strategy 2: Successfully restored checkpoint from Drive (fallback): {checkpoint_path}")
                else:
                    logger.info(
                        f"Strategy 2: Fallback restore returned False (checkpoint not found in Drive)")

        except Exception as e:
            logger.warning(
                f"Strategy 2 failed: {type(e).__name__}: {e}", exc_info=True)

    # Strategy 3: MLflow download
    mlflow_strategy_enabled = (
        "mlflow" in priority and
        acquisition_config.get("mlflow", {}).get("enabled", False) and
        checkpoint_path is None
    )

    logger.info(f"Strategy 3: MLflow download - enabled={mlflow_strategy_enabled} "
                f"(mlflow_in_priority={'mlflow' in priority}, mlflow_enabled={acquisition_config.get('mlflow', {}).get('enabled', False)}, "
                f"checkpoint_path={checkpoint_path is None})")

    if mlflow_strategy_enabled:
        logger.info("Strategy 3: Attempting MLflow download...")
        try:
            checkpoint_dir = _build_checkpoint_dir(
                root_dir, config_dir, platform, backbone, run_id,
                study_key_hash=study_key_hash, trial_key_hash=trial_key_hash
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            import mlflow as mlflow_func
            from mlflow.tracking import MlflowClient

            tracking_uri = mlflow_func.get_tracking_uri()
            is_azure_ml = tracking_uri and "azureml://" in tracking_uri

            # List artifacts to find checkpoint
            client = MlflowClient()
            checkpoint_artifact_path = None

            try:
                artifacts = client.list_artifacts(run_id=run_id)
                artifact_paths = [artifact.path for artifact in artifacts]

                # Look for checkpoint in artifact paths
                for path in artifact_paths:
                    if "checkpoint" in path.lower():
                        if path == "checkpoint" or path == "checkpoint/":
                            checkpoint_artifact_path = path
                            break
                        elif checkpoint_artifact_path is None:
                            checkpoint_artifact_path = path

                if checkpoint_artifact_path is None:
                    checkpoint_artifact_path = "checkpoint"
            except Exception:
                checkpoint_artifact_path = "checkpoint"

            # Download artifacts
            local_path = client.download_artifacts(
                run_id=run_id,
                path=checkpoint_artifact_path,
                dst_path=str(checkpoint_dir)
            )

            downloaded_path = Path(local_path)

            # Check for and extract tar.gz files
            if downloaded_path.is_file() and downloaded_path.suffixes == ['.tar', '.gz']:
                downloaded_path = _extract_tar_gz(downloaded_path)
            elif downloaded_path.is_dir():
                tar_files = list(downloaded_path.glob("*.tar.gz")) + \
                    list(downloaded_path.glob("*.tgz"))
                if tar_files:
                    downloaded_path = _extract_tar_gz(
                        tar_files[0], extract_to=downloaded_path)

            # Find checkpoint in downloaded/extracted directory
            if downloaded_path.is_dir():
                found_checkpoint = _find_checkpoint_in_directory(
                    downloaded_path)
                if found_checkpoint:
                    downloaded_path = found_checkpoint
                else:
                    # Validate the directory itself
                    if not _validate_checkpoint(downloaded_path):
                        raise ValueError(
                            "Downloaded checkpoint failed validation - no valid checkpoint files found")

            # Final validation
            if acquisition_config["mlflow"].get("validate", True):
                if not _validate_checkpoint(downloaded_path):
                    raise ValueError("Downloaded checkpoint failed validation")

            checkpoint_path = downloaded_path
            logger.info(
                f"Strategy 3: Successfully downloaded checkpoint from MLflow: {checkpoint_path}")

        except Exception as e:
            logger.warning(f"Strategy 3 failed: {type(e).__name__}: {e}")

    # All strategies failed - check for manually placed checkpoint
    if checkpoint_path is None:
        checkpoint_dir = _build_checkpoint_dir(
            root_dir, config_dir, platform, backbone, run_id,
            study_key_hash=study_key_hash, trial_key_hash=trial_key_hash
        )
        manual_checkpoint_path = checkpoint_dir / "checkpoint"

        if manual_checkpoint_path.exists() and any(manual_checkpoint_path.iterdir()):
            if _validate_checkpoint(manual_checkpoint_path):
                checkpoint_path = manual_checkpoint_path

    # Backup to Drive if in Colab and checkpoint was successfully acquired
    backup_enabled = (
        checkpoint_path is not None and
        in_colab and
        drive_store is not None and
        acquisition_config.get("drive", {}).get("enabled", False)
    )

    logger.info(f"Drive backup: enabled={backup_enabled} "
                f"(checkpoint_path={checkpoint_path is not None}, in_colab={in_colab}, "
                f"drive_store={drive_store is not None}, drive_enabled={acquisition_config.get('drive', {}).get('enabled', False)})")

    if backup_enabled:
        logger.info("Backing up checkpoint to Drive...")
        try:
            # Ensure checkpoint is in best_model_selection directory structure
            target_backup_path = _build_checkpoint_dir(
                root_dir, config_dir, platform, backbone, run_id,
                study_key_hash=study_key_hash, trial_key_hash=trial_key_hash
            )
            logger.info(f"Backup target path: {target_backup_path}")

            # If checkpoint is not already in target location, copy it there
            if checkpoint_path != target_backup_path:
                logger.info(
                    f"Copying checkpoint from {checkpoint_path} to {target_backup_path}")
                target_backup_path.mkdir(parents=True, exist_ok=True)
                if target_backup_path.exists():
                    shutil.rmtree(target_backup_path)
                shutil.copytree(checkpoint_path, target_backup_path)
                checkpoint_path = target_backup_path

            # Backup to Drive
            logger.info(f"Backing up checkpoint to Drive...")
            result = drive_store.backup(checkpoint_path, expect="dir")
            if result.ok:
                logger.info(
                    f"Successfully backed up checkpoint to Drive: {result.dst}")
            else:
                logger.warning(f"Drive backup failed: {result.reason}")

        except Exception as e:
            logger.warning(
                f"Drive backup failed: {type(e).__name__}: {e}", exc_info=True)

    if checkpoint_path:
        logger.info(f"Checkpoint acquisition successful: {checkpoint_path}")
        return checkpoint_path

    # Generate error message
    tracking_uri = mlflow.get_tracking_uri() or ""
    is_azure_ml = "azureml://" in tracking_uri

    error_msg = (
        f"\n[ERROR] Could not acquire checkpoint for run {run_id[:8]}...\n"
        f"   Experiment: {experiment_name}\n"
        f"   Backbone: {backbone}\n"
        f"\n[TRIED] Strategies attempted:\n"
        f"   1. Local disk selection (by config + backbone)\n"
        f"   2. Drive restore (Colab only)\n"
        f"   3. MLflow download\n"
        f"\n[SOLUTIONS] In order of preference:\n"
        f"\n1. **Use Local Disk** (if checkpoint exists locally):\n"
        f"   - Ensure checkpoint is at: \"{root_dir / 'outputs' / 'hpo' / platform / backbone}\"\n"
        f"   - Verify 'local' is first in artifact_acquisition.yaml priority\n"
        f"   - Re-run this cell\n"
    )

    if is_azure_ml:
        error_msg += (
            f"\n2. **Manual download from Azure ML Studio**:\n"
            f"   - Go to: https://ml.azure.com\n"
            f"   - Navigate to: Experiments → {experiment_name} → Run {run_id[:8]}...\n"
            f"   - Download the 'best_trial_checkpoint' artifact (or 'checkpoint' if available)\n"
            f"   - Extract if it's a .tar.gz file\n"
            f"   - Place the checkpoint folder at: \"{manual_checkpoint_path}\"\n"
            f"   - Re-run this cell\n"
        )
    else:
        error_msg += (
            f"\n2. **Manual download from MLflow UI**:\n"
            f"   - Go to MLflow tracking UI (check tracking URI: {tracking_uri})\n"
            f"   - Navigate to: Experiment '{experiment_name}' → Run {run_id[:8]}...\n"
            f"   - Download the 'best_trial_checkpoint' artifact (or 'checkpoint' if available)\n"
            f"   - Extract if it's a .tar.gz file\n"
            f"   - Place the checkpoint folder at: \"{manual_checkpoint_path}\"\n"
            f"   - Re-run this cell\n"
        )

    raise ValueError(error_msg)
