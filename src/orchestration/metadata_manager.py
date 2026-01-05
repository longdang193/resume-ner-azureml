"""Manage persistent metadata for training and conversion stages."""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from orchestration.paths import resolve_output_path
from shared.json_cache import load_json, save_json
from shared.platform_detection import detect_platform


def get_metadata_file_path(
    root_dir: Path,
    config_dir: Path,
    training_name: str
) -> Path:
    """
    Get path to metadata file for a training name.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory (root_dir / "config").
        training_name: Stable training name (e.g., "distilbert_trial_0").

    Returns:
        Path to metadata file.
    """
    cache_dir = resolve_output_path(
        root_dir, config_dir, "cache", subcategory="final_training"
    )
    return cache_dir / f"{training_name}_metadata.json"


def load_training_metadata(
    root_dir: Path,
    config_dir: Path,
    training_name: str
) -> Optional[Dict[str, Any]]:
    """
    Load metadata for a training name.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        training_name: Stable training name.

    Returns:
        Metadata dictionary or None if not found.
    """
    metadata_file = get_metadata_file_path(root_dir, config_dir, training_name)
    return load_json(metadata_file, default=None)


def save_training_metadata(
    root_dir: Path,
    config_dir: Path,
    training_name: str,
    backbone: str,
    trial_name: str,
    trial_id: str,
    best_config_timestamp: str,
    status_updates: Dict[str, Any]
) -> Path:
    """
    Save or update training metadata.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        training_name: Stable training name.
        backbone: Model backbone name.
        trial_name: Trial name from best configuration.
        trial_id: Trial ID from best configuration.
        best_config_timestamp: Timestamp when best config was selected.
        status_updates: Dictionary of status updates by stage:
            {
                "training": {"completed": True, "checkpoint_path": "...", ...},
                "benchmarking": {"completed": True, ...},
                "conversion": {"completed": True, "onnx_model_path": "...", ...}
            }

    Returns:
        Path to saved metadata file.
    """
    metadata_file = get_metadata_file_path(root_dir, config_dir, training_name)

    # Load existing metadata or create new
    metadata = load_json(metadata_file, default={})

    # Update base info
    metadata.update({
        "training_name": training_name,
        "backbone": backbone,
        "trial_name": trial_name,
        "trial_id": trial_id,
        "best_config_timestamp": best_config_timestamp,
        "last_updated": datetime.now().isoformat(),
    })

    # Update status
    if "status" not in metadata:
        metadata["status"] = {}

    for stage, updates in status_updates.items():
        if stage not in metadata["status"]:
            metadata["status"][stage] = {}
        metadata["status"][stage].update(updates)

        # Add timestamps for completion and artifact upload
        if "completed" in updates and updates["completed"]:
            metadata["status"][stage]["completed_at"] = datetime.now().isoformat()
        if "artifacts_uploaded" in updates and updates.get("artifacts_uploaded"):
            metadata["status"][stage]["artifacts_uploaded_at"] = datetime.now(
            ).isoformat()

    save_json(metadata_file, metadata)
    return metadata_file


def is_training_complete(
    root_dir: Path,
    config_dir: Path,
    training_name: str
) -> bool:
    """
    Check if training has been completed.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        training_name: Stable training name.

    Returns:
        True if training is marked as complete, False otherwise.
    """
    metadata = load_training_metadata(root_dir, config_dir, training_name)
    if not metadata:
        return False
    return metadata.get("status", {}).get("training", {}).get("completed", False)


def are_training_artifacts_uploaded(
    root_dir: Path,
    config_dir: Path,
    training_name: str
) -> bool:
    """
    Check if training artifacts have been uploaded.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        training_name: Stable training name.

    Returns:
        True if artifacts are marked as uploaded, False otherwise.
    """
    metadata = load_training_metadata(root_dir, config_dir, training_name)
    if not metadata:
        return False
    return metadata.get("status", {}).get("training", {}).get("artifacts_uploaded", False)


def is_benchmarking_complete(
    root_dir: Path,
    config_dir: Path,
    training_name: str
) -> bool:
    """
    Check if benchmarking has been completed.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        training_name: Stable training name.

    Returns:
        True if benchmarking is marked as complete, False otherwise.
    """
    metadata = load_training_metadata(root_dir, config_dir, training_name)
    if not metadata:
        return False
    return metadata.get("status", {}).get("benchmarking", {}).get("completed", False)


def is_conversion_complete(
    root_dir: Path,
    config_dir: Path,
    training_name: str
) -> bool:
    """
    Check if model conversion has been completed.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        training_name: Stable training name.

    Returns:
        True if conversion is marked as complete, False otherwise.
    """
    metadata = load_training_metadata(root_dir, config_dir, training_name)
    if not metadata:
        return False
    return metadata.get("status", {}).get("conversion", {}).get("completed", False)


def are_conversion_artifacts_uploaded(
    root_dir: Path,
    config_dir: Path,
    training_name: str
) -> bool:
    """
    Check if conversion artifacts have been uploaded.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        training_name: Stable training name.

    Returns:
        True if artifacts are marked as uploaded, False otherwise.
    """
    metadata = load_training_metadata(root_dir, config_dir, training_name)
    if not metadata:
        return False
    return metadata.get("status", {}).get("conversion", {}).get("artifacts_uploaded", False)


def get_training_checkpoint_path(
    root_dir: Path,
    config_dir: Path,
    training_name: str
) -> Optional[Path]:
    """
    Get checkpoint path from metadata.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        training_name: Stable training name.

    Returns:
        Path to checkpoint directory or None if not found.
    """
    metadata = load_training_metadata(root_dir, config_dir, training_name)
    if not metadata:
        return None

    checkpoint_path_str = metadata.get("status", {}).get(
        "training", {}).get("checkpoint_path")
    if checkpoint_path_str:
        return Path(checkpoint_path_str)
    return None


def get_conversion_onnx_path(
    root_dir: Path,
    config_dir: Path,
    training_name: str
) -> Optional[Path]:
    """
    Get ONNX model path from metadata.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        training_name: Stable training name.

    Returns:
        Path to ONNX model file or None if not found.
    """
    metadata = load_training_metadata(root_dir, config_dir, training_name)
    if not metadata:
        return None

    onnx_path_str = metadata.get("status", {}).get(
        "conversion", {}).get("onnx_model_path")
    if onnx_path_str:
        return Path(onnx_path_str)
    return None


# ============================================================================
# Fingerprint-based metadata functions (new centralized naming system)
# ============================================================================

def save_metadata_with_fingerprints(
    root_dir: Path,
    config_dir: Path,
    context: "NamingContext",
    metadata_content: Dict[str, Any],
    status_updates: Optional[Dict[str, Any]] = None,
    **additional_metadata
) -> Path:
    """
    Save metadata with fingerprints using NamingContext (wrapper function).

    This is a convenience wrapper that builds the metadata path from the context
    and extracts fingerprint information automatically.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        context: NamingContext with all fingerprint information.
        metadata_content: Metadata content to save.
        status_updates: Optional status updates by stage.
        **additional_metadata: Additional metadata fields.

    Returns:
        Path to saved metadata file.
    """
    from .naming_centralized import build_output_path

    # Build output directory from context
    output_dir = build_output_path(root_dir, context)

    # Metadata file is always metadata.json in the output directory
    metadata_path = output_dir / "metadata.json"

    # Extract fingerprint information from context
    spec_fp = context.spec_fp
    exec_fp = context.exec_fp
    variant = context.variant
    environment = context.environment
    model = context.model

    # Extract parent HPO trial from metadata_content if available
    parent_hpo_trial = metadata_content.get(
        "trial_id") or metadata_content.get("trial_name")

    # Merge metadata_content into additional_metadata
    merged_metadata = {**additional_metadata, **metadata_content}

    # Call the low-level save function
    return _save_metadata_with_fingerprints_impl(
        metadata_path=metadata_path,
        spec_fp=spec_fp,
        exec_fp=exec_fp,
        variant=variant,
        environment=environment,
        model=model,
        parent_hpo_trial=parent_hpo_trial,
        status=status_updates,
        mlflow_info=mlflow_info,
        **merged_metadata
    )


def _save_metadata_with_fingerprints_impl(
    metadata_path: Path,
    spec_fp: Optional[str] = None,
    exec_fp: Optional[str] = None,
    variant: int = 1,
    environment: Optional[str] = None,
    model: Optional[str] = None,
    parent_hpo_trial: Optional[str] = None,
    lineage: Optional[Dict[str, Any]] = None,
    status: Optional[Dict[str, Any]] = None,
    mlflow_info: Optional[Dict[str, Any]] = None,
    **additional_metadata
) -> Path:
    """
    Internal implementation for saving metadata with fingerprints.

    Args:
        metadata_path: Path to metadata.json file.
        spec_fp: Specification fingerprint.
        exec_fp: Execution fingerprint.
        variant: Variant number (for final_training).
        environment: Execution environment (auto-detected if None).
        model: Model backbone name.
        parent_hpo_trial: Parent HPO trial identifier.
        lineage: Lineage information (e.g., {"hpo_trial": "...", "best_config_source": "..."}).
        status: Status updates by stage.
        **additional_metadata: Additional metadata fields.

    Returns:
        Path to saved metadata file.
    """
    if environment is None:
        environment = detect_platform()

    # Load existing metadata or create new
    metadata = load_json(metadata_path, default={})

    # Update fingerprint-based identity
    if spec_fp:
        metadata["spec_fp"] = spec_fp
    if exec_fp:
        metadata["exec_fp"] = exec_fp
    if variant:
        metadata["variant"] = variant
    metadata["environment"] = environment

    if model:
        metadata["model"] = model

    # Update lineage
    if lineage:
        if "lineage" not in metadata:
            metadata["lineage"] = {}
        metadata["lineage"].update(lineage)

    if parent_hpo_trial:
        metadata["parent_hpo_trial"] = parent_hpo_trial
        if "lineage" not in metadata:
            metadata["lineage"] = {}
        metadata["lineage"]["hpo_trial"] = parent_hpo_trial

    # Update status
    if status:
        if "status" not in metadata:
            metadata["status"] = {}
        for stage, updates in status.items():
            if stage not in metadata["status"]:
                metadata["status"][stage] = {}
            metadata["status"][stage].update(updates)

            # Add timestamps for completion
            if "completed" in updates and updates["completed"]:
                metadata["status"][stage]["completed_at"] = datetime.now().isoformat()
            if "artifacts_uploaded" in updates and updates.get("artifacts_uploaded"):
                metadata["status"][stage]["artifacts_uploaded_at"] = datetime.now(
                ).isoformat()

    # Add timestamps
    if "created_at" not in metadata:
        metadata["created_at"] = datetime.now().isoformat()
    metadata["last_updated"] = datetime.now().isoformat()

    # Add MLflow info if provided
    if mlflow_info:
        metadata["mlflow"] = mlflow_info

    # Add additional metadata
    metadata.update(additional_metadata)

    # Ensure directory exists
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    save_json(metadata_path, metadata)
    return metadata_path


def save_metadata_with_fingerprints(
    root_dir: Path,
    config_dir: Path,
    context: "NamingContext",
    metadata_content: Dict[str, Any],
    status_updates: Optional[Dict[str, Any]] = None,
    **additional_metadata
) -> Path:
    """
    Save metadata with fingerprints using NamingContext (wrapper function).

    This is a convenience wrapper that builds the metadata path from the context
    and extracts fingerprint information automatically.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        context: NamingContext with all fingerprint information.
        metadata_content: Metadata content to save.
        status_updates: Optional status updates by stage.
        **additional_metadata: Additional metadata fields.

    Returns:
        Path to saved metadata file.
    """
    from .naming_centralized import build_output_path

    # Build output directory from context
    output_dir = build_output_path(root_dir, context)

    # Metadata file is always metadata.json in the output directory
    metadata_path = output_dir / "metadata.json"

    # Extract fingerprint information from context
    spec_fp = context.spec_fp
    exec_fp = context.exec_fp
    variant = context.variant
    environment = context.environment
    model = context.model

    # Extract parent HPO trial from metadata_content if available
    parent_hpo_trial = metadata_content.get(
        "trial_id") or metadata_content.get("trial_name")

    # Merge metadata_content into additional_metadata
    merged_metadata = {**additional_metadata, **metadata_content}

    # Extract mlflow_info if provided in additional_metadata
    mlflow_info = merged_metadata.pop("mlflow_info", None)

    # Call the actual save function (low-level implementation)
    return _save_metadata_with_fingerprints_impl(
        metadata_path=metadata_path,
        spec_fp=spec_fp,
        exec_fp=exec_fp,
        variant=variant,
        environment=environment,
        model=model,
        parent_hpo_trial=parent_hpo_trial,
        status=status_updates,
        mlflow_info=mlflow_info,
        **merged_metadata
    )


def save_metadata_with_fingerprints_low_level(
    metadata_path: Path,
    spec_fp: Optional[str] = None,
    exec_fp: Optional[str] = None,
    variant: int = 1,
    environment: Optional[str] = None,
    model: Optional[str] = None,
    parent_hpo_trial: Optional[str] = None,
    lineage: Optional[Dict[str, Any]] = None,
    status: Optional[Dict[str, Any]] = None,
    **additional_metadata
) -> Path:
    """
    Save metadata with fingerprint-based identity (low-level function).

    This is the low-level function that actually saves the metadata.
    Use save_metadata_with_fingerprints() for the high-level wrapper.

    Args:
        metadata_path: Path to metadata.json file.
        spec_fp: Specification fingerprint.
        exec_fp: Execution fingerprint.
        variant: Variant number (for final_training).
        environment: Execution environment (auto-detected if None).
        model: Model backbone name.
        parent_hpo_trial: Parent HPO trial identifier.
        lineage: Lineage information (e.g., {"hpo_trial": "...", "best_config_source": "..."}).
        status: Status updates by stage.
        **additional_metadata: Additional metadata fields.

    Returns:
        Path to saved metadata file.
    """
    if environment is None:
        environment = detect_platform()

    # Load existing metadata or create new
    metadata = load_json(metadata_path, default={})

    # Update fingerprint-based identity
    if spec_fp:
        metadata["spec_fp"] = spec_fp
    if exec_fp:
        metadata["exec_fp"] = exec_fp
    if variant:
        metadata["variant"] = variant
    metadata["environment"] = environment

    if model:
        metadata["model"] = model

    # Update lineage
    if lineage:
        if "lineage" not in metadata:
            metadata["lineage"] = {}
        metadata["lineage"].update(lineage)

    if parent_hpo_trial:
        metadata["parent_hpo_trial"] = parent_hpo_trial
        if "lineage" not in metadata:
            metadata["lineage"] = {}
        metadata["lineage"]["hpo_trial"] = parent_hpo_trial

    # Update status
    if status:
        if "status" not in metadata:
            metadata["status"] = {}
        for stage, updates in status.items():
            if stage not in metadata["status"]:
                metadata["status"][stage] = {}
            metadata["status"][stage].update(updates)

            # Add timestamps for completion
            if "completed" in updates and updates["completed"]:
                metadata["status"][stage]["completed_at"] = datetime.now().isoformat()
            if "artifacts_uploaded" in updates and updates.get("artifacts_uploaded"):
                metadata["status"][stage]["artifacts_uploaded_at"] = datetime.now(
                ).isoformat()

    # Add timestamps
    if "created_at" not in metadata:
        metadata["created_at"] = datetime.now().isoformat()
    metadata["last_updated"] = datetime.now().isoformat()

    # Add additional metadata
    metadata.update(additional_metadata)

    # Ensure directory exists
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    save_json(metadata_path, metadata)
    return metadata_path


def load_metadata_by_fingerprints(
    metadata_path: Path,
    spec_fp: Optional[str] = None,
    exec_fp: Optional[str] = None,
    variant: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Load metadata by fingerprints.

    Args:
        metadata_path: Path to metadata.json file.
        spec_fp: Optional spec_fp to filter by.
        exec_fp: Optional exec_fp to filter by.
        variant: Optional variant to filter by.

    Returns:
        Metadata dictionary or None if not found or doesn't match filters.
    """
    metadata = load_json(metadata_path, default=None)
    if not metadata:
        return None

    # Apply filters if provided
    if spec_fp and metadata.get("spec_fp") != spec_fp:
        return None
    if exec_fp and metadata.get("exec_fp") != exec_fp:
        return None
    if variant is not None and metadata.get("variant") != variant:
        return None

    return metadata


def find_metadata_by_spec_fp(
    root_dir: Path,
    spec_fp: str,
    process_type: str = "final_training",
) -> list[Dict[str, Any]]:
    """
    Find all metadata files with matching spec_fp.

    This searches for metadata.json files in the output directories
    and returns those matching the spec_fp.

    Args:
        root_dir: Project root directory.
        spec_fp: Specification fingerprint to search for.
        process_type: Process type to search (final_training, conversion, etc.).

    Returns:
        List of metadata dictionaries matching spec_fp.
    """
    results = []
    base_path = root_dir / "outputs"

    if process_type == "final_training":
        search_path = base_path / "final_training"
    elif process_type == "conversion":
        search_path = base_path / "conversion"
    elif process_type == "hpo":
        search_path = base_path / "hpo"
    elif process_type == "benchmarking":
        search_path = base_path / "benchmarking"
    else:
        return results

    # Search for metadata.json files
    for metadata_file in search_path.rglob("metadata.json"):
        metadata = load_json(metadata_file, default=None)
        if metadata and metadata.get("spec_fp") == spec_fp:
            metadata["_metadata_path"] = str(metadata_file)
            results.append(metadata)

    return results


def update_mlflow_info_in_metadata(
    metadata_path: Path,
    mlflow_info_updates: Dict[str, Any],
) -> Path:
    """
    Update MLflow info in existing metadata.json (e.g., at run end).

    Args:
        metadata_path: Path to metadata.json file.
        mlflow_info_updates: Dictionary of MLflow info fields to update.
            Common fields: artifact_uri, ended_at, status.

    Returns:
        Path to updated metadata file.
    """
    from datetime import datetime

    # Load existing metadata
    metadata = load_json(metadata_path, default={})

    # Initialize mlflow section if needed
    if "mlflow" not in metadata:
        metadata["mlflow"] = {}

    # Update MLflow info
    metadata["mlflow"].update(mlflow_info_updates)

    # Update last_updated timestamp
    metadata["last_updated"] = datetime.now().isoformat()

    # Ensure directory exists
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Save updated metadata
    save_json(metadata_path, metadata)

    return metadata_path
