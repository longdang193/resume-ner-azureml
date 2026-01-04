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

    print(f"   [INFO] Extracting {tar_path.name} to {extract_to}...")

    with tarfile.open(tar_path, 'r:gz') as tar:
        # Get the root directory name from the archive
        members = tar.getmembers()
        if members:
            # Extract all files
            tar.extractall(path=extract_to)

            # Determine the extracted path
            # If archive has a single root directory, return that
            root_names = {Path(m.name).parts[0] for m in members if m.name}
            if len(root_names) == 1:
                root_name = list(root_names)[0]
                extracted_path = extract_to / root_name
                if extracted_path.exists():
                    print(f"   [INFO] Extracted to: {extracted_path}")
                    return extracted_path

            # Otherwise, check if extract_to itself contains checkpoint files
            if _validate_checkpoint(extract_to):
                print(
                    f"   [INFO] Extracted checkpoint files directly to: {extract_to}")
                return extract_to

            # If not, search for checkpoint subdirectory
            checkpoint_subdir = extract_to / "checkpoint"
            if checkpoint_subdir.exists() and checkpoint_subdir.is_dir() and _validate_checkpoint(checkpoint_subdir):
                print(
                    f"   [INFO] Found checkpoint in extracted 'checkpoint' subdirectory: {checkpoint_subdir}")
                return checkpoint_subdir

            # Return extract_to as fallback (will be searched by caller)
            print(
                f"   [INFO] Extracted to: {extract_to} (will search for checkpoint files)")
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
        "pytorch_model.bin",  # Older PyTorch
        "model.safetensors",  # HuggingFace safetensors
        "model.bin",  # HuggingFace bin
        "pytorch_model.bin.index.json",  # Sharded models
    ]

    # Check if any essential file exists
    has_model_file = any((checkpoint_path / fname).exists()
                         for fname in essential_files)

    # Also check for config.json (HuggingFace models)
    has_config = (checkpoint_path / "config.json").exists()

    # Valid if has model file OR config (some checkpoints might be structured differently)
    return has_model_file or has_config


def acquire_best_model_checkpoint(
    best_run_info: Dict[str, Any],
    root_dir: Path,
    config_dir: Path,
    acquisition_config: Dict[str, Any],
    selection_config: Dict[str, Any],
    platform: str,
    restore_from_drive: Optional[Callable[[Path, bool], bool]] = None,
    in_colab: bool = False,
) -> Path:
    """
    Acquire checkpoint using local-first fallback strategy.

    Priority order (from config):
    1. Local disk (by config + backbone) - PREFERRED to avoid Azure ML issues
    2. Drive restore (Colab only)
    3. MLflow download (with Azure CLI fallback instructions)

    Critical: Must use the exact identifiers from the selected MLflow run,
    not "best on disk". However, we use local selection v2 which finds the
    best trial by CV metrics, then uses refit checkpoint as artifact.

    Args:
        best_run_info: Dictionary with best run information (must include study_key_hash, trial_key_hash, run_id, backbone)
        root_dir: Project root directory
        config_dir: Config directory
        acquisition_config: Artifact acquisition configuration
        selection_config: Best model selection configuration
        platform: Platform name (local, colab, kaggle)
        restore_from_drive: Optional function to restore from Drive backup
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

    print(f"[ACQUIRE] Acquiring checkpoint for run {run_id[:8]}...")
    print(f"   Backbone: {backbone}")
    print(
        f"   Study key hash: {study_key_hash[:16] if study_key_hash else 'None'}...")
    print(
        f"   Trial key hash: {trial_key_hash[:16] if trial_key_hash else 'None'}...")

    # Use priority order from config (should be local-first)
    priority = acquisition_config["priority"]
    print(f"   Acquisition priority: {' → '.join(priority)}")

    # Strategy 1: Local disk selection (PREFERRED - avoids Azure ML issues)
    if "local" in priority:
        try:
            from orchestration.jobs.local_selection_v2 import (
                find_study_folder_by_config,
                load_best_trial_from_study_folder,
            )
            from orchestration.config_loader import load_all_configs, load_experiment_config
            from orchestration import EXPERIMENT_NAME

            print(f"\n[Strategy 1] Local disk selection...")

            # Load HPO config to get study name pattern
            experiment_config = load_experiment_config(
                config_dir, EXPERIMENT_NAME)
            all_configs = load_all_configs(experiment_config)
            hpo_config = all_configs.get("hpo", {})

            if not hpo_config:
                print("   [WARN] No HPO config found, skipping local selection")
            else:
                # Build HPO output path
                hpo_output_dir = root_dir / "outputs" / "hpo" / platform / backbone
                hpo_output_dir_quoted = f'"{hpo_output_dir}"'
                print(f"   Searching in: {hpo_output_dir_quoted}")

                if not hpo_output_dir.exists():
                    print(f"   [WARN] HPO output directory does not exist")
                else:
                    # Try to find study folder by config pattern
                    study_folder = find_study_folder_by_config(
                        backbone_dir=hpo_output_dir,
                        hpo_config=hpo_config,
                        backbone=backbone,
                    )

                    if not study_folder:
                        print(
                            f"   [WARN] No study folder found matching config pattern")
                    else:
                        print(
                            f"   [OK] Found study folder: {study_folder.name}")

                        # Load best trial from study folder (uses CV metrics for selection)
                        best_trial = load_best_trial_from_study_folder(
                            study_folder=study_folder,
                            objective_metric=selection_config["objective"]["metric"],
                        )

                        if not best_trial:
                            print(
                                f"   [WARN] No best trial found in study folder")
                        else:
                            checkpoint_path = Path(
                                best_trial["checkpoint_dir"])

                            # Validate checkpoint integrity
                            if acquisition_config["local"].get("validate", True):
                                if not _validate_checkpoint(checkpoint_path):
                                    print(
                                        f"   [WARN] Checkpoint validation failed: {checkpoint_path}")
                                    print(
                                        f"      (Missing essential model files)")
                                else:
                                    # Quote path for display (handles spaces in paths)
                                    checkpoint_path_quoted = f'"{checkpoint_path}"'
                                    print(
                                        f"   [OK] Checkpoint validated: {checkpoint_path_quoted}")
                                    return checkpoint_path
                            else:
                                # Skip validation, just check existence
                                if checkpoint_path.exists():
                                    checkpoint_path_quoted = f'"{checkpoint_path}"'
                                    print(
                                        f"   [OK] Found checkpoint: {checkpoint_path_quoted}")
                                    return checkpoint_path
                                else:
                                    print(
                                        f"   [WARN] Checkpoint path does not exist: {checkpoint_path}")
        except Exception as e:
            print(f"   [WARN] Local selection failed: {e}")
            import traceback
            print(f"      {traceback.format_exc()}")

    # Strategy 2: Drive restore (if in priority and Colab)
    if "drive" in priority and restore_from_drive and in_colab and acquisition_config["drive"]["enabled"]:
        try:
            print(f"\n[Strategy 2] Drive restore...")
            checkpoint_name = f"checkpoint_{run_id[:8]}"
            drive_folder = acquisition_config["drive"]["folder_path"]
            drive_checkpoint_path = Path(
                "/content/drive/MyDrive") / drive_folder / "checkpoints" / checkpoint_name

            if restore_from_drive(drive_checkpoint_path, is_directory=True):
                # Validate if enabled
                if acquisition_config["drive"].get("validate", True):
                    if not _validate_checkpoint(drive_checkpoint_path):
                        print(f"   [WARN] Checkpoint validation failed")
                    else:
                        drive_checkpoint_path_quoted = f'"{drive_checkpoint_path}"'
                        print(
                            f"   [OK] Restored and validated checkpoint from Drive: {drive_checkpoint_path_quoted}")
                        return drive_checkpoint_path
                else:
                    drive_checkpoint_path_quoted = f'"{drive_checkpoint_path}"'
                    print(
                        f"   [OK] Restored checkpoint from Drive: {drive_checkpoint_path_quoted}")
                    return drive_checkpoint_path
            else:
                print(f"   [WARN] Drive restore returned False")
        except Exception as e:
            print(f"   [WARN] Drive restore failed: {e}")

    # Strategy 3: MLflow download (with Azure CLI fallback instructions)
    if "mlflow" in priority and acquisition_config["mlflow"]["enabled"]:
        try:
            print(f"\n[Strategy 3] MLflow download...")
            checkpoint_dir = root_dir / "outputs" / "best_model_checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Import mlflow at function level to avoid scoping issues
            # (mlflow is already imported at module level, but Python may treat it as local
            #  if there are any imports/assignments later in the function)
            import mlflow as mlflow_func  # Use alias to avoid shadowing module-level import

            # Check if we're using Azure ML (known compatibility issues)
            tracking_uri = mlflow_func.get_tracking_uri()
            is_azure_ml = tracking_uri and "azureml://" in tracking_uri

            # Pre-extract resource group and workspace name for error messages
            # (do this early so we have the values ready if MLflow download fails)
            extracted_workspace_name = "<workspace-name>"
            extracted_resource_group = "<resource-group>"
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
                        extracted_workspace_name = mlflow_config.get(
                            "azure_ml", {}).get("workspace_name", "<workspace-name>")
                except Exception:
                    pass

                # Get resource group - try multiple sources
                extracted_resource_group = os.getenv(
                    "AZURE_RESOURCE_GROUP") or ""
                if not extracted_resource_group:
                    # Try loading from config.env file (check multiple possible locations)
                    possible_paths = [
                        root_dir / "config.env",
                        config_dir_path / "config.env",
                        config_dir_path.parent / "config.env",  # In case config_dir is config/
                        Path.cwd() / "config.env",
                    ]
                    for config_env_path in possible_paths:
                        if config_env_path.exists():
                            env_vars = _load_env_file(config_env_path)
                            extracted_resource_group = env_vars.get(
                                "AZURE_RESOURCE_GROUP", "")
                            if extracted_resource_group:
                                # Remove quotes if present (config.env has quotes)
                                extracted_resource_group = extracted_resource_group.strip(
                                    '"\'')
                                break

                # If still not found, try tracking URI extraction
                if not extracted_resource_group and tracking_uri and is_azure_ml:
                    try:
                        patterns = [
                            # Standard Azure ML format
                            r'/resourceGroups/([^/]+)/',
                            # Without trailing slash
                            r'resourceGroups/([^/]+)',
                            # Query parameter format
                            r'resourceGroup=([^&]+)',
                        ]
                        for pattern in patterns:
                            rg_match = re.search(pattern, tracking_uri)
                            if rg_match:
                                extracted_resource_group = rg_match.group(1)
                                break
                    except Exception:
                        pass

                # If still not found, try config files
                if not extracted_resource_group:
                    try:
                        infra_config_path = config_dir_path / "infrastructure.yaml"
                        if infra_config_path.exists():
                            infra_config = load_yaml(infra_config_path)
                            rg_config = infra_config.get(
                                "azure", {}).get("resource_group", "")
                            if rg_config.startswith("${") and rg_config.endswith("}"):
                                env_var = rg_config[2:-1]
                                extracted_resource_group = os.getenv(
                                    env_var, "")
                            else:
                                extracted_resource_group = rg_config or ""
                    except Exception:
                        pass

                # Set placeholder if still empty
                if not extracted_resource_group:
                    extracted_resource_group = "<resource-group>"
                if not extracted_workspace_name or extracted_workspace_name == "":
                    extracted_workspace_name = "<workspace-name>"
            except Exception:
                pass

            # Try to extract resource group from tracking URI if available
            resource_group_from_uri = None
            if is_azure_ml and tracking_uri:
                # Extract from URI format: azureml://.../resourceGroups/<rg>/.../workspaces/<ws>
                import re
                rg_match = re.search(r'/resourceGroups/([^/]+)/', tracking_uri)
                if rg_match:
                    resource_group_from_uri = rg_match.group(1)

            if is_azure_ml:
                print(
                    f"   [WARN] Azure ML detected - MLflow download may fail due to compatibility issues")
                print(
                    f"      (Known issue: azureml_artifacts_builder() got unexpected keyword 'tracking_uri')")

            # Try MLflow artifacts API
            # Use MlflowClient for better Azure ML compatibility (avoids tracking_uri issue)
            mlflow_download_success = False
            checkpoint_artifact_path = None

            try:
                from mlflow.tracking import MlflowClient

                # First, list artifacts to find the checkpoint
                client = MlflowClient()
                try:
                    artifacts = client.list_artifacts(run_id=run_id)
                    artifact_paths = [artifact.path for artifact in artifacts]

                    # Look for checkpoint in various possible locations
                    possible_checkpoint_paths = [
                        "checkpoint",
                        "checkpoint/",
                        "artifacts/checkpoint",
                        "artifacts/checkpoint/",
                    ]

                    # Also check for any path containing "checkpoint"
                    for path in artifact_paths:
                        if "checkpoint" in path.lower():
                            # Prefer exact match, then directory matches
                            if path == "checkpoint" or path == "checkpoint/":
                                checkpoint_artifact_path = path
                                break
                            elif checkpoint_artifact_path is None:
                                checkpoint_artifact_path = path

                    # If not found in list, try the standard path
                    if checkpoint_artifact_path is None:
                        checkpoint_artifact_path = "checkpoint"
                        print(
                            f"   [INFO] Checkpoint path not found in artifact list, trying: {checkpoint_artifact_path}")
                    else:
                        print(
                            f"   [INFO] Found checkpoint artifact at: {checkpoint_artifact_path}")
                        if len(artifact_paths) > 0:
                            print(
                                f"   [INFO] Available artifacts: {', '.join(artifact_paths[:10])}{'...' if len(artifact_paths) > 10 else ''}")
                except Exception as list_err:
                    # If listing fails, try the standard path anyway
                    checkpoint_artifact_path = "checkpoint"
                    print(f"   [WARN] Could not list artifacts: {list_err}")
                    print(
                        f"   [INFO] Attempting download with standard path: {checkpoint_artifact_path}")

                # Try using MlflowClient first (better Azure ML compatibility)
                # This avoids the azureml_artifacts_builder() tracking_uri error
                local_path = client.download_artifacts(
                    run_id=run_id,
                    path=checkpoint_artifact_path,
                    dst_path=str(checkpoint_dir)
                )

                # MlflowClient.download_artifacts returns the full path to the downloaded artifact
                # If it's a directory, it returns the directory path
                checkpoint_path = Path(local_path)

                print(f"   [INFO] Downloaded to: {checkpoint_path}")
                print(f"   [INFO] Is directory: {checkpoint_path.is_dir()}")

                # Check if downloaded artifact is a tar.gz file or contains tar.gz files
                if checkpoint_path.is_file() and checkpoint_path.suffixes == ['.tar', '.gz']:
                    # Single tar.gz file - extract it
                    checkpoint_path = _extract_tar_gz(checkpoint_path)
                elif checkpoint_path.is_dir():
                    # Check for tar.gz files in the directory
                    tar_files = list(checkpoint_path.glob(
                        "*.tar.gz")) + list(checkpoint_path.glob("*.tgz"))
                    if tar_files:
                        # Extract the first tar.gz file found
                        tar_file = tar_files[0]
                        print(
                            f"   [INFO] Found compressed checkpoint: {tar_file.name}")
                        extracted_path = _extract_tar_gz(
                            tar_file, extract_to=checkpoint_path)
                        # Use extracted path for further processing
                        checkpoint_path = extracted_path

                # If the path is a directory, search for checkpoint files inside it
                if checkpoint_path.is_dir():
                    # First, check if model files are directly in the downloaded directory
                    if _validate_checkpoint(checkpoint_path):
                        print(
                            f"   [INFO] Checkpoint files found directly in downloaded directory")
                    else:
                        # List contents for debugging
                        try:
                            contents = list(checkpoint_path.iterdir())
                            print(
                                f"   [INFO] Directory contents: {[item.name for item in contents[:10]]}")
                        except Exception:
                            pass

                        # Check if checkpoint subdirectory exists
                        checkpoint_subdir = checkpoint_path / "checkpoint"
                        if checkpoint_subdir.exists() and checkpoint_subdir.is_dir():
                            print(
                                f"   [INFO] Found 'checkpoint' subdirectory, checking it...")
                            if _validate_checkpoint(checkpoint_subdir):
                                checkpoint_path = checkpoint_subdir
                                print(
                                    f"   [INFO] Valid checkpoint found in 'checkpoint' subdirectory")
                            else:
                                # Try looking for checkpoint in all subdirectories recursively
                                print(
                                    f"   [INFO] Searching recursively for checkpoint files...")
                                found_valid = False
                                # Search all subdirectories
                                for item in checkpoint_path.rglob("*"):
                                    if item.is_dir() and _validate_checkpoint(item):
                                        checkpoint_path = item
                                        found_valid = True
                                        print(
                                            f"   [INFO] Found valid checkpoint in subdirectory: {checkpoint_path}")
                                        break
                                if not found_valid:
                                    # Last resort: check if any subdirectory contains model files
                                    for item in checkpoint_path.rglob("*.bin"):
                                        parent_dir = item.parent
                                        if _validate_checkpoint(parent_dir):
                                            checkpoint_path = parent_dir
                                            found_valid = True
                                            print(
                                                f"   [INFO] Found checkpoint via model file: {checkpoint_path}")
                                            break
                                    for item in checkpoint_path.rglob("*.safetensors"):
                                        parent_dir = item.parent
                                        if _validate_checkpoint(parent_dir):
                                            checkpoint_path = parent_dir
                                            found_valid = True
                                            print(
                                                f"   [INFO] Found checkpoint via safetensors file: {checkpoint_path}")
                                            break
                                    if not found_valid:
                                        print(
                                            f"   [WARN] Could not find valid checkpoint in downloaded directory")

                # Validate if enabled
                if acquisition_config["mlflow"].get("validate", True):
                    if not _validate_checkpoint(checkpoint_path):
                        print(
                            f"   [WARN] Checkpoint validation failed at: {checkpoint_path}")
                        # Try one more comprehensive search
                        if checkpoint_path.is_dir():
                            print(
                                f"   [INFO] Performing comprehensive search for checkpoint files...")
                            # Search for any directory containing model files
                            for root, dirs, files in os.walk(checkpoint_path):
                                root_path = Path(root)
                                # Check if this directory has model files
                                has_model = any(f.endswith(
                                    ('.bin', '.safetensors')) for f in files)
                                has_config = 'config.json' in files
                                if has_model or has_config:
                                    if _validate_checkpoint(root_path):
                                        checkpoint_path = root_path
                                        print(
                                            f"   [INFO] Found valid checkpoint via comprehensive search: {checkpoint_path}")
                                        break
                            else:
                                # Still not found
                                raise ValueError(
                                    "Downloaded checkpoint failed validation - no valid checkpoint files found")

                # Quote path for display (handles spaces in paths)
                local_path_quoted = f'"{checkpoint_path}"'
                print(
                    f"   [OK] Downloaded checkpoint from MLflow using MlflowClient: {local_path_quoted}")
                mlflow_download_success = True
                return checkpoint_path

            except Exception as e_client:
                # MlflowClient approach failed, try alternative method
                error_type_client = type(e_client).__name__
                error_msg_client = str(e_client)

                # Try alternative: use mlflow.artifacts as fallback (might work in some cases)
                try:
                    import mlflow.artifacts

                    # Use the discovered path or default
                    artifact_path = checkpoint_artifact_path or "checkpoint"

                    local_path = mlflow_func.artifacts.download_artifacts(
                        run_id=run_id,
                        artifact_path=artifact_path,
                        dst_path=str(checkpoint_dir)
                    )
                    checkpoint_path = Path(local_path)

                    print(f"   [INFO] Downloaded to: {checkpoint_path}")
                    print(
                        f"   [INFO] Is directory: {checkpoint_path.is_dir()}")

                    # Check if downloaded artifact is a tar.gz file or contains tar.gz files
                    if checkpoint_path.is_file() and checkpoint_path.suffixes == ['.tar', '.gz']:
                        # Single tar.gz file - extract it
                        checkpoint_path = _extract_tar_gz(checkpoint_path)
                    elif checkpoint_path.is_dir():
                        # Check for tar.gz files in the directory
                        tar_files = list(checkpoint_path.glob(
                            "*.tar.gz")) + list(checkpoint_path.glob("*.tgz"))
                        if tar_files:
                            # Extract the first tar.gz file found
                            tar_file = tar_files[0]
                            print(
                                f"   [INFO] Found compressed checkpoint: {tar_file.name}")
                            extracted_path = _extract_tar_gz(
                                tar_file, extract_to=checkpoint_path)
                            # Use extracted path for further processing
                            checkpoint_path = extracted_path

                    # Handle directory structure same as above
                    if checkpoint_path.is_dir():
                        # First, check if model files are directly in the downloaded directory
                        if _validate_checkpoint(checkpoint_path):
                            print(
                                f"   [INFO] Checkpoint files found directly in downloaded directory")
                        else:
                            # List contents for debugging
                            try:
                                contents = list(checkpoint_path.iterdir())
                                print(
                                    f"   [INFO] Directory contents: {[item.name for item in contents[:10]]}")
                            except Exception:
                                pass

                            # Check if checkpoint subdirectory exists
                            checkpoint_subdir = checkpoint_path / "checkpoint"
                            if checkpoint_subdir.exists() and checkpoint_subdir.is_dir():
                                print(
                                    f"   [INFO] Found 'checkpoint' subdirectory, checking it...")
                                if _validate_checkpoint(checkpoint_subdir):
                                    checkpoint_path = checkpoint_subdir
                                    print(
                                        f"   [INFO] Valid checkpoint found in 'checkpoint' subdirectory")
                                else:
                                    # Try looking for checkpoint in all subdirectories recursively
                                    print(
                                        f"   [INFO] Searching recursively for checkpoint files...")
                                    found_valid = False
                                    for item in checkpoint_path.rglob("*"):
                                        if item.is_dir() and _validate_checkpoint(item):
                                            checkpoint_path = item
                                            found_valid = True
                                            print(
                                                f"   [INFO] Found valid checkpoint in subdirectory: {checkpoint_path}")
                                            break
                                    if not found_valid:
                                        # Last resort: check if any subdirectory contains model files
                                        for item in checkpoint_path.rglob("*.bin"):
                                            parent_dir = item.parent
                                            if _validate_checkpoint(parent_dir):
                                                checkpoint_path = parent_dir
                                                found_valid = True
                                                print(
                                                    f"   [INFO] Found checkpoint via model file: {checkpoint_path}")
                                                break
                                        for item in checkpoint_path.rglob("*.safetensors"):
                                            parent_dir = item.parent
                                            if _validate_checkpoint(parent_dir):
                                                checkpoint_path = parent_dir
                                                found_valid = True
                                                print(
                                                    f"   [INFO] Found checkpoint via safetensors file: {checkpoint_path}")
                                                break

                    # Validate if enabled
                    if acquisition_config["mlflow"].get("validate", True):
                        if not _validate_checkpoint(checkpoint_path):
                            print(
                                f"   [WARN] Checkpoint validation failed at: {checkpoint_path}")
                            # Try one more comprehensive search
                            if checkpoint_path.is_dir():
                                print(
                                    f"   [INFO] Performing comprehensive search for checkpoint files...")
                                # Search for any directory containing model files
                                for root, dirs, files in os.walk(checkpoint_path):
                                    root_path = Path(root)
                                    # Check if this directory has model files
                                    has_model = any(f.endswith(
                                        ('.bin', '.safetensors')) for f in files)
                                    has_config = 'config.json' in files
                                    if has_model or has_config:
                                        if _validate_checkpoint(root_path):
                                            checkpoint_path = root_path
                                            print(
                                                f"   [INFO] Found valid checkpoint via comprehensive search: {checkpoint_path}")
                                            break
                                else:
                                    # Still not found
                                    raise ValueError(
                                        "Downloaded checkpoint failed validation - no valid checkpoint files found")

                    local_path_quoted = f'"{checkpoint_path}"'
                    print(
                        f"   [OK] Downloaded checkpoint from MLflow using mlflow.artifacts: {local_path_quoted}")
                    mlflow_download_success = True
                    return checkpoint_path

                except Exception as e_artifacts:
                    # Both methods failed - try downloading root artifacts as last resort
                    if checkpoint_artifact_path and checkpoint_artifact_path != ".":
                        try:
                            print(
                                f"   [INFO] Trying to download root artifacts directory...")
                            from mlflow.tracking import MlflowClient
                            client = MlflowClient()
                            local_path = client.download_artifacts(
                                run_id=run_id,
                                path=".",
                                dst_path=str(checkpoint_dir)
                            )
                            checkpoint_path = Path(local_path)

                            print(
                                f"   [INFO] Root artifacts downloaded to: {checkpoint_path}")

                            # Check for tar.gz files in root artifacts
                            if checkpoint_path.is_dir():
                                tar_files = list(checkpoint_path.rglob(
                                    "*.tar.gz")) + list(checkpoint_path.rglob("*.tgz"))
                                if tar_files:
                                    # Extract the first tar.gz file found
                                    tar_file = tar_files[0]
                                    print(
                                        f"   [INFO] Found compressed checkpoint in root artifacts: {tar_file.name}")
                                    extracted_path = _extract_tar_gz(
                                        tar_file, extract_to=tar_file.parent)
                                    # Use extracted path for further processing
                                    checkpoint_path = extracted_path

                            # Search for checkpoint in downloaded artifacts using comprehensive search
                            if checkpoint_path.is_dir():
                                # First check if valid checkpoint is directly in root
                                if _validate_checkpoint(checkpoint_path):
                                    local_path_quoted = f'"{checkpoint_path}"'
                                    print(
                                        f"   [OK] Root artifacts contain valid checkpoint: {local_path_quoted}")
                                    return checkpoint_path

                                # Look for checkpoint subdirectory or any directory with checkpoint files
                                print(
                                    f"   [INFO] Searching for checkpoint in root artifacts...")
                                found_valid = False

                                # Search for directories named "checkpoint" or containing checkpoint files
                                for item in checkpoint_path.rglob("*"):
                                    if item.is_dir() and _validate_checkpoint(item):
                                        checkpoint_path = item
                                        found_valid = True
                                        local_path_quoted = f'"{checkpoint_path}"'
                                        print(
                                            f"   [OK] Found checkpoint in root artifacts: {local_path_quoted}")
                                        return checkpoint_path

                                # If not found by name, search by file content
                                if not found_valid:
                                    for root, dirs, files in os.walk(checkpoint_path):
                                        root_path = Path(root)
                                        has_model = any(f.endswith(
                                            ('.bin', '.safetensors')) for f in files)
                                        has_config = 'config.json' in files
                                        if has_model or has_config:
                                            if _validate_checkpoint(root_path):
                                                checkpoint_path = root_path
                                                found_valid = True
                                                local_path_quoted = f'"{checkpoint_path}"'
                                                print(
                                                    f"   [OK] Found checkpoint via file search: {local_path_quoted}")
                                                return checkpoint_path

                                if not found_valid:
                                    print(
                                        f"   [WARN] No valid checkpoint found in root artifacts")
                            else:
                                # Single file downloaded - shouldn't happen but handle it
                                if _validate_checkpoint(checkpoint_path.parent):
                                    checkpoint_path = checkpoint_path.parent
                                    local_path_quoted = f'"{checkpoint_path}"'
                                    print(
                                        f"   [OK] Root artifacts contain valid checkpoint: {local_path_quoted}")
                                    return checkpoint_path
                        except Exception as e_root:
                            # Root download also failed
                            pass

                    # All methods failed - use the first error for reporting
                    e_mlflow = e_client
                    error_type = type(e_mlflow).__name__
                    error_msg_str = str(e_mlflow)

                    # MLflow download failed - provide Azure CLI fallback instructions

                # Quote paths for command-line usage (handles spaces in paths)
                checkpoint_path_quoted = f'"{checkpoint_dir / "checkpoint"}"'
                checkpoint_dir_quoted = f'"{checkpoint_dir}"'
                hpo_path_quoted = f'"{root_dir / "outputs" / "hpo" / platform / backbone}"'

                print(
                    f"   [WARN] MLflow download failed: {error_type}: {error_msg_str}")

                if is_azure_ml:
                    # Use pre-extracted values (extracted at the start of Strategy 3)
                    # If extraction failed, try one more time here as fallback
                    workspace_name = extracted_workspace_name
                    resource_group = extracted_resource_group

                    # If still placeholders, try extraction one more time (fallback)
                    if workspace_name == "<workspace-name>" or resource_group == "<resource-group>":
                        try:
                            from shared.yaml_utils import load_yaml
                            from shared.mlflow_setup import _load_env_file
                            import re
                            config_dir_path = Path(config_dir) if isinstance(
                                config_dir, str) else config_dir

                            # Get workspace name from config
                            if workspace_name == "<workspace-name>":
                                try:
                                    mlflow_config_path = config_dir_path / "mlflow.yaml"
                                    if mlflow_config_path.exists():
                                        mlflow_config = load_yaml(
                                            mlflow_config_path)
                                        workspace_name = mlflow_config.get("azure_ml", {}).get(
                                            "workspace_name", "<workspace-name>")
                                except Exception:
                                    pass

                            # Get resource group - try multiple sources
                            if resource_group == "<resource-group>":
                                resource_group = os.getenv(
                                    "AZURE_RESOURCE_GROUP") or ""
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
                                            env_vars = _load_env_file(
                                                config_env_path)
                                            resource_group = env_vars.get(
                                                "AZURE_RESOURCE_GROUP", "")
                                            if resource_group:
                                                resource_group = resource_group.strip(
                                                    '"\'')
                                                break

                                if not resource_group:
                                    # Try to extract from tracking URI
                                    if tracking_uri:
                                        try:
                                            patterns = [
                                                r'/resourceGroups/([^/]+)/',
                                                r'resourceGroups/([^/]+)',
                                                r'resourceGroup=([^&]+)',
                                            ]
                                            for pattern in patterns:
                                                rg_match = re.search(
                                                    pattern, tracking_uri)
                                                if rg_match:
                                                    resource_group = rg_match.group(
                                                        1)
                                                    break
                                        except Exception:
                                            pass

                                if not resource_group:
                                    # Try config files
                                    try:
                                        infra_config_path = config_dir_path / "infrastructure.yaml"
                                        if infra_config_path.exists():
                                            infra_config = load_yaml(
                                                infra_config_path)
                                            rg_config = infra_config.get(
                                                "azure", {}).get("resource_group", "")
                                            if rg_config.startswith("${") and rg_config.endswith("}"):
                                                env_var = rg_config[2:-1]
                                                resource_group = os.getenv(
                                                    env_var, "")
                                            else:
                                                resource_group = rg_config or ""
                                    except Exception:
                                        pass

                                if not resource_group:
                                    resource_group = "<resource-group>"
                        except Exception:
                            pass

                    # Generate command with actual values (use whatever we found, even if only one is available)
                    az_cli_cmd = f"az ml job download --name {run_id} --resource-group {resource_group} --workspace-name {workspace_name} --all --download-path {checkpoint_dir_quoted}"

                    print(f"\n   [INFO] Azure ML Compatibility Issue Detected")
                    print(f"      This is a known MLflow ↔ Azure ML version mismatch.")
                    print(f"\n   [RECOMMENDED] Use Azure CLI v2 to download:")
                    print(f"      {az_cli_cmd}")
                    if "<resource-group>" in az_cli_cmd:
                        print(f"      \n      To find resource-group:")
                        print(
                            f"      - Set env var: $env:AZURE_RESOURCE_GROUP='your-rg' (PowerShell)")
                        print(
                            f"      - Or run: az account show --query 'resourceGroup' -o tsv")
                    print(
                        f"      Then search for 'checkpoint' folder inside the downloaded directory.")
                    print(
                        f"\n   [ALTERNATIVE] Fix MLflow version compatibility:")
                    print(f"      pip uninstall -y mlflow azureml-mlflow")
                    print(
                        f"      pip install azureml-mlflow  # This will pull compatible mlflow")
                    print(f"      Then restart kernel and retry.")

                # Don't raise here - let it fall through to manual placement check
                pass

        except Exception as e:
            print(f"   [WARN] MLflow download strategy failed: {e}")
            import traceback
            print(f"      {traceback.format_exc()}")

    # All strategies failed - check for manually placed checkpoint
    checkpoint_dir = root_dir / "outputs" / "best_model_checkpoint"
    manual_checkpoint_path = checkpoint_dir / "checkpoint"

    # Quote paths for command-line usage (handles spaces in paths)
    manual_checkpoint_path_quoted = f'"{manual_checkpoint_path}"'
    hpo_path_quoted = f'"{root_dir / "outputs" / "hpo" / platform / backbone}"'
    checkpoint_dir_quoted = f'"{checkpoint_dir}"'

    print(f"\n[WARN] All automated strategies failed. Checking for manually placed checkpoint...")

    if manual_checkpoint_path.exists() and any(manual_checkpoint_path.iterdir()):
        # Validate manually placed checkpoint
        if _validate_checkpoint(manual_checkpoint_path):
            print(
                f"[OK] Found and validated manually placed checkpoint at: {manual_checkpoint_path_quoted}")
            return manual_checkpoint_path
        else:
            print(
                f"[WARN] Manually placed checkpoint found but validation failed: {manual_checkpoint_path_quoted}")

    # Provide comprehensive error message with all options
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
        f"   - Ensure checkpoint is at: {hpo_path_quoted}\n"
        f"   - Verify 'local' is first in artifact_acquisition.yaml priority\n"
        f"   - Re-run this cell\n"
        f"\n2. **Download via Azure CLI v2** (recommended for Azure ML):\n"
    )

    # Try to get workspace name and resource group from config or tracking URI
    workspace_name = "<workspace-name>"
    resource_group = "<resource-group>"

    try:
        from shared.yaml_utils import load_yaml
        from shared.mlflow_setup import _load_env_file
        import mlflow as mlflow_module
        import re

        config_dir_path = Path(config_dir) if isinstance(
            config_dir, str) else config_dir

        # Get workspace name from config
        try:
            mlflow_config_path = config_dir_path / "mlflow.yaml"
            if mlflow_config_path.exists():
                mlflow_config = load_yaml(mlflow_config_path)
                workspace_name = mlflow_config.get(
                    "azure_ml", {}).get("workspace_name", "")
                if not workspace_name:
                    workspace_name = "<workspace-name>"
        except Exception:
            pass

        # Get resource group - try multiple sources in order
        # 1. Environment variable
        resource_group = os.getenv("AZURE_RESOURCE_GROUP", "").strip()

        # 2. config.env file (check multiple possible locations)
        if not resource_group:
            possible_paths = [
                root_dir / "config.env",
                config_dir_path / "config.env",
                config_dir_path.parent / "config.env",  # In case config_dir is config/
                Path.cwd() / "config.env",
            ]
            for config_env_path in possible_paths:
                try:
                    if config_env_path.exists():
                        env_vars = _load_env_file(config_env_path)
                        resource_group = env_vars.get(
                            "AZURE_RESOURCE_GROUP", "")
                        if resource_group:
                            # Remove quotes if present (config.env has quotes)
                            resource_group = resource_group.strip('"\'')
                            if resource_group:  # Check again after stripping
                                break
                except Exception:
                    continue  # Try next path

        # 3. Extract from MLflow tracking URI
        if not resource_group:
            try:
                tracking_uri = mlflow_module.get_tracking_uri()
                if tracking_uri and "azureml://" in tracking_uri:
                    patterns = [
                        # Standard Azure ML format
                        r'/resourceGroups/([^/]+)/',
                        r'resourceGroups/([^/]+)',     # Without trailing slash
                        r'resourceGroup=([^&]+)',     # Query parameter format
                    ]
                    for pattern in patterns:
                        rg_match = re.search(pattern, tracking_uri)
                        if rg_match:
                            resource_group = rg_match.group(1)
                            break
            except Exception:
                pass

        # 4. Try config files
        if not resource_group:
            try:
                infra_config_path = config_dir_path / "infrastructure.yaml"
                if infra_config_path.exists():
                    infra_config = load_yaml(infra_config_path)
                    rg_config = infra_config.get(
                        "azure", {}).get("resource_group", "")
                    if rg_config:
                        if rg_config.startswith("${") and rg_config.endswith("}"):
                            env_var = rg_config[2:-1]
                            resource_group = os.getenv(env_var, "").strip()
                        else:
                            resource_group = rg_config.strip()
            except Exception:
                pass

        # Set placeholder if still empty
        if not resource_group:
            resource_group = "<resource-group>"
        if not workspace_name or workspace_name == "":
            workspace_name = "<workspace-name>"

    except Exception:
        # If anything fails, use placeholders
        pass

    # Generate command with actual values (use whatever we found, even if only one is available)
    az_cli_cmd = f"az ml job download --name {run_id} --resource-group {resource_group} --workspace-name {workspace_name} --all --download-path {checkpoint_dir_quoted}"

    error_msg += (
        f"   {az_cli_cmd}\n"
    )

    if "<resource-group>" in az_cli_cmd:
        error_msg += (
            f"   \n"
            f"   To find resource-group:\n"
            f"   - Set environment variable: $env:AZURE_RESOURCE_GROUP='your-resource-group' (PowerShell)\n"
            f"   - Or check Azure Portal → Your Resource Group → Overview → Resource Group name\n"
            f"   - Or run: az account show --query 'resourceGroup' -o tsv\n"
        )

    error_msg += (
        f"   \n"
        f"   Then search for 'checkpoint' folder and move it to: {manual_checkpoint_path_quoted}\n"
        f"\n3. **Manual download from Azure ML UI**:\n"
        f"   - Go to: https://ml.azure.com\n"
        f"   - Navigate to: Experiments → {experiment_name} → Run {run_id[:8]}...\n"
        f"   - Download the 'checkpoint' artifact\n"
        f"   - Place it at: {manual_checkpoint_path_quoted}\n"
        f"   - Re-run this cell\n"
        f"\n4. **Fix MLflow compatibility** (if you need MLflow download to work):\n"
        f"   pip uninstall -y mlflow azureml-mlflow\n"
        f"   pip install azureml-mlflow\n"
        f"   Restart kernel and retry\n"
    )

    raise ValueError(error_msg)
