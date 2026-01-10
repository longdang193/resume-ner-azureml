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
    from infrastructure.paths import resolve_output_path

    base_dir = resolve_output_path(
        root_dir, config_dir, "best_model_selection") / environment / backbone

    if study_key_hash and trial_key_hash:
        # Extract 8-char hashes for path construction (consistent with token expansion)
        study8 = study_key_hash[:8] if len(study_key_hash) >= 8 else study_key_hash
        trial8 = trial_key_hash[:8] if len(trial_key_hash) >= 8 else trial_key_hash
        return base_dir / f"sel_{study8}_{trial8}"

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
        from common.shared.yaml_utils import load_yaml
        from common.shared.mlflow_setup import _load_env_file
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

    # Strategy 1: Local disk selection (using hashes from best_model)
    if "local" in priority and checkpoint_path is None:
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

        except Exception:
            pass

    # Strategy 2: Drive restore (Colab only) - optimized to scan Drive metadata and restore only checkpoint
    if "drive" in priority and restore_from_drive and in_colab and acquisition_config["drive"]["enabled"] and checkpoint_path is None:
        try:
            if study_key_hash and trial_key_hash and drive_store:
                # OPTIMIZATION: Scan Drive directly to find checkpoint path
                # This avoids restoring the entire HPO directory structure
                hpo_output_dir = root_dir / "outputs" / "hpo" / platform / backbone

                # Compute Drive path for HPO directory
                try:
                    drive_hpo_dir = drive_store.drive_path_for(hpo_output_dir)
                except ValueError:
                    drive_hpo_dir = None

                if drive_hpo_dir and drive_hpo_dir.exists():
                    drive_checkpoint_path = _find_checkpoint_in_drive_by_hash(
                        drive_hpo_dir, study_key_hash, trial_key_hash
                    )

                    if drive_checkpoint_path and drive_checkpoint_path.exists():
                        # Found checkpoint in Drive! Copy to best_model_selection cache dir
                        local_checkpoint_path = _build_checkpoint_dir(
                            root_dir, config_dir, platform, backbone, run_id,
                            study_key_hash=study_key_hash, trial_key_hash=trial_key_hash
                        )

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
                        else:
                            if local_checkpoint_path.exists():
                                checkpoint_path = local_checkpoint_path

            # Fallback: Try best_model_selection directory structure (if checkpoint was manually moved)
            if checkpoint_path is None:
                local_checkpoint_path = _build_checkpoint_dir(
                    root_dir, config_dir, platform, backbone, run_id,
                    study_key_hash=study_key_hash, trial_key_hash=trial_key_hash
                )

                if restore_from_drive(local_checkpoint_path, is_directory=True):
                    if acquisition_config["drive"].get("validate", True):
                        if _validate_checkpoint(local_checkpoint_path):
                            checkpoint_path = local_checkpoint_path
                    else:
                        if local_checkpoint_path.exists():
                            checkpoint_path = local_checkpoint_path

        except Exception:
            pass

    # Strategy 3: MLflow download
    if "mlflow" in priority and acquisition_config["mlflow"]["enabled"] and checkpoint_path is None:
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
                    # Resolve and validate the found checkpoint path exists
                    found_checkpoint = Path(found_checkpoint).resolve()
                    if found_checkpoint.exists() and found_checkpoint.is_dir():
                        downloaded_path = found_checkpoint
                    else:
                        raise ValueError(
                            f"Found checkpoint path does not exist: {found_checkpoint}\n"
                            f"This may indicate an issue with the MLflow artifact structure."
                        )
                else:
                    # Validate the directory itself
                    if not _validate_checkpoint(downloaded_path):
                        raise ValueError(
                            "Downloaded checkpoint failed validation - no valid checkpoint files found")

            # Resolve path to absolute
            downloaded_path = Path(downloaded_path).resolve()

            # Ensure the checkpoint path exists and is valid
            if not downloaded_path.exists():
                raise ValueError(
                    f"Downloaded checkpoint path does not exist: {downloaded_path}\n"
                    f"This may indicate an issue with the MLflow artifact structure."
                )

            if not downloaded_path.is_dir():
                raise ValueError(
                    f"Downloaded checkpoint path is not a directory: {downloaded_path}"
                )

            # Final validation
            if acquisition_config["mlflow"].get("validate", True):
                if not _validate_checkpoint(downloaded_path):
                    raise ValueError("Downloaded checkpoint failed validation")

            checkpoint_path = downloaded_path

        except Exception as e:
            # Log the error instead of silently swallowing it
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"Failed to download checkpoint from MLflow: {e}", exc_info=True)
            # Don't set checkpoint_path - let fallback strategies try
            checkpoint_path = None

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
    # Follow the same pattern as final training: backup directly without copying first
    if checkpoint_path and in_colab and drive_store and acquisition_config.get("drive", {}).get("enabled", False):
        try:
            checkpoint_path = Path(checkpoint_path).resolve()

            # Validate checkpoint exists before attempting backup
            if not checkpoint_path.exists():
                print(f"⚠ Checkpoint path does not exist: {checkpoint_path}")
                print(
                    f"  Skipping Drive backup - checkpoint may have been moved or deleted")
            elif not checkpoint_path.is_dir():
                print(
                    f"⚠ Checkpoint path is not a directory: {checkpoint_path}")
                print(f"  Skipping Drive backup")
            else:
                # Backup to Drive (same pattern as final training)
                print(f"\n📦 Backing up best model checkpoint to Google Drive...")
                result = drive_store.backup(checkpoint_path, expect="dir")
                if result.ok:
                    print(f"✓ Successfully backed up checkpoint to Google Drive")
                    print(f"  Drive path: {result.dst}")
                else:
                    print(f"⚠ Drive backup failed: {result.reason}")
                    if result.error:
                        print(f"  Error: {result.error}")
                    print(
                        f"  Checkpoint is still available locally at: {checkpoint_path}")

        except Exception as e:
            print(f"⚠ Drive backup error: {e}")
            print(
                f"  Checkpoint is still available locally at: {checkpoint_path}")
            import traceback
            traceback.print_exc()

    if checkpoint_path:
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
