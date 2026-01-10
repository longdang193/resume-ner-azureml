from __future__ import annotations

"""
@meta
name: tracking_mlflow_artifacts
type: utility
domain: tracking
responsibility:
  - Provide safe MLflow artifact upload utilities with retry logic
  - Handle artifact upload errors gracefully
inputs:
  - Local file paths
  - Run IDs and artifact paths
outputs:
  - Upload success status
tags:
  - utility
  - tracking
  - mlflow
  - artifacts
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Safe MLflow artifact upload utilities with retry logic and error handling."""
from pathlib import Path
from typing import Optional
import tempfile
import json
import shutil

import mlflow
from common.shared.logging_utils import get_logger

from infrastructure.tracking.mlflow.utils import retry_with_backoff

logger = get_logger(__name__)

def log_artifact_safe(
    local_path: str | Path,
    artifact_path: Optional[str] = None,
    run_id: Optional[str] = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> bool:
    """
    Safely upload a single artifact to MLflow with retry logic.

    This function handles both active run (mlflow.log_artifact) and explicit
    run_id (client.log_artifact) scenarios. Errors are logged but not raised.

    Args:
        local_path: Path to the local file to upload.
        artifact_path: Optional artifact path within the run's artifact directory.
        run_id: Optional run ID. If provided, uses client.log_artifact().
                If None, uses mlflow.log_artifact() with active run.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds for exponential backoff.

    Returns:
        True if upload succeeded, False otherwise.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        logger.warning(f"Artifact file does not exist: {local_path}")
        return False

    try:
        def upload_func():
            if run_id:
                # Use explicit run_id (client.log_artifact)
                client = mlflow.tracking.MlflowClient()
                client.log_artifact(
                    run_id=run_id,
                    local_path=str(local_path),
                    artifact_path=artifact_path
                )
            else:
                # Use active run (mlflow.log_artifact)
                mlflow.log_artifact(
                    local_path=str(local_path),
                    artifact_path=artifact_path
                )

        retry_with_backoff(
            func=upload_func,
            max_retries=max_retries,
            base_delay=base_delay,
            operation_name=f"artifact upload ({local_path.name})"
        )

        logger.debug(
            f"Successfully uploaded artifact: {local_path.name} "
            f"(run_id={run_id[:12] + '...' if run_id else 'active'})"
        )
        return True

    except Exception as e:
        logger.warning(
            f"Failed to upload artifact {local_path.name}: {e}",
            exc_info=True
        )
        return False

def log_artifacts_safe(
    local_dir: str | Path,
    artifact_path: Optional[str] = None,
    run_id: Optional[str] = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> bool:
    """
    Safely upload a directory of artifacts to MLflow with retry logic.

    This function handles both active run (mlflow.log_artifacts) and explicit
    run_id (client.log_artifacts) scenarios. Errors are logged but not raised.

    Args:
        local_dir: Path to the local directory to upload.
        artifact_path: Optional artifact path within the run's artifact directory.
        run_id: Optional run ID. If provided, uses client.log_artifacts().
                If None, uses mlflow.log_artifacts() with active run.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds for exponential backoff.

    Returns:
        True if upload succeeded, False otherwise.
    """
    local_dir = Path(local_dir)
    if not local_dir.exists() or not local_dir.is_dir():
        logger.warning(f"Artifact directory does not exist: {local_dir}")
        return False

    try:
        # Count files for progress logging
        file_count = sum(1 for _ in local_dir.rglob('*') if _.is_file())
        logger.info(f"Uploading {file_count} files from {local_dir}...")

        def upload_func():
            if run_id:
                # Use explicit run_id (client.log_artifacts)
                client = mlflow.tracking.MlflowClient()
                client.log_artifacts(
                    run_id=run_id,
                    local_dir=str(local_dir),
                    artifact_path=artifact_path
                )
            else:
                # Use active run (mlflow.log_artifacts)
                mlflow.log_artifacts(
                    local_dir=str(local_dir),
                    artifact_path=artifact_path
                )

        retry_with_backoff(
            func=upload_func,
            max_retries=max_retries,
            base_delay=base_delay,
            operation_name=f"artifacts upload ({local_dir.name})"
        )

        logger.info(
            f"Successfully uploaded {file_count} files from {local_dir.name} "
            f"(run_id={run_id[:12] + '...' if run_id else 'active'})"
        )
        return True

    except Exception as e:
        logger.warning(
            f"Failed to upload artifacts from {local_dir.name}: {e}",
            exc_info=True
        )
        return False

def upload_checkpoint_archive(
    archive_path: Path,
    manifest: Optional[dict] = None,
    artifact_path: Optional[str] = None,
    run_id: Optional[str] = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
    cleanup_on_failure: bool = True,
) -> bool:
    """
    Upload a checkpoint archive and optional manifest to MLflow.

    This is a specialized helper for checkpoint uploads that includes
    manifest logging and cleanup handling.

    Args:
        archive_path: Path to the checkpoint archive file.
        manifest: Optional manifest dictionary to upload as JSON.
        artifact_path: Optional artifact path within the run's artifact directory.
        run_id: Optional run ID. If provided, uses client.log_artifact().
                If None, uses mlflow.log_artifact() with active run.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds for exponential backoff.
        cleanup_on_failure: If True, attempt to clean up archive on failure.

    Returns:
        True if upload succeeded, False otherwise.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        logger.warning(f"Checkpoint archive does not exist: {archive_path}")
        return False

    archive_size_mb = archive_path.stat().st_size / 1024 / 1024
    logger.info(
        f"Uploading checkpoint archive ({archive_size_mb:.1f}MB)..."
    )

    # Upload archive
    archive_artifact_path = artifact_path or "best_trial_checkpoint.tar.gz"
    archive_success = log_artifact_safe(
        local_path=archive_path,
        artifact_path=archive_artifact_path,
        run_id=run_id,
        max_retries=max_retries,
        base_delay=base_delay,
    )

    if not archive_success:
        if cleanup_on_failure:
            try:
                archive_path.unlink(missing_ok=True)
                logger.debug(f"Cleaned up archive after failed upload: {archive_path}")
            except Exception as e:
                logger.debug(f"Could not clean up archive: {e}")
        return False

    # Upload manifest if provided
    if manifest:
        try:
            manifest_json = json.dumps(manifest, indent=2)
            
            # Determine manifest artifact path
            # Azure ML's MLflow has a bug where it treats paths ending in .json as directories.
            # To work around this, we use a different filename for the artifact path.
            # The local file is still named manifest.json for consistency.
            if artifact_path:
                if artifact_path.endswith('.tar.gz') or artifact_path.endswith('.gz'):
                    # Archive path provided, use same base with _manifest.json
                    base = artifact_path.rsplit('.', 2)[0]  # Remove .tar.gz
                    manifest_filename = f"{base}_manifest.json"
                    manifest_artifact_filename = manifest_filename  # Use same name
                    manifest_artifact_path = None  # Upload to root
                elif '/' in artifact_path:
                    # Directory-like path with slash, use directory only
                    manifest_filename = "manifest.json"  # Local file name
                    # Use checkpoint_manifest.json to avoid Azure ML directory bug
                    manifest_artifact_filename = "checkpoint_manifest.json"
                    manifest_artifact_path = artifact_path.rstrip('/')  # Directory only
                elif not artifact_path.endswith('.json'):
                    # Plain directory name (no extension, no slash), use as directory
                    manifest_filename = "manifest.json"  # Local file name
                    # Use checkpoint_manifest.json to avoid Azure ML directory bug
                    manifest_artifact_filename = "checkpoint_manifest.json"
                    manifest_artifact_path = artifact_path  # Directory only
                else:
                    # Already a JSON filename, extract directory if any
                    manifest_filename = Path(artifact_path).name
                    # Use checkpoint_ prefix to avoid directory bug
                    manifest_artifact_filename = f"checkpoint_{manifest_filename}"
                    parent_dir = Path(artifact_path).parent
                    manifest_artifact_path = str(parent_dir) if str(parent_dir) != '.' else None
            else:
                manifest_filename = "manifest.json"  # Local file name
                manifest_artifact_filename = "best_trial_checkpoint_manifest.json"
                manifest_artifact_path = None  # Upload to root
            
            # Create temp directory and write manifest with correct filename
            # This ensures MLflow uses the correct filename instead of temp basename
            # Use a fixed temp filename to avoid Azure ML creating directories
            temp_dir = tempfile.mkdtemp(prefix="mlflow_manifest_")
            try:
                manifest_file_path = Path(temp_dir) / manifest_filename
                manifest_file_path.write_text(manifest_json, encoding='utf-8')
                
                # For Azure ML, we need to be explicit about the file path
                # Use the artifact filename (not local filename) to avoid directory creation bug
                if manifest_artifact_path:
                    # Construct full artifact path: directory/artifact_filename
                    full_artifact_path = f"{manifest_artifact_path}/{manifest_artifact_filename}"
                else:
                    # Upload to root with explicit artifact filename
                    full_artifact_path = manifest_artifact_filename
                
                # Use MLflow client directly with retry logic to have more control
                # Explicitly specify full path to avoid Azure ML creating directories
                def upload_manifest():
                    if run_id:
                        client = mlflow.tracking.MlflowClient()
                        # Upload with explicit full path to avoid directory creation issues
                        client.log_artifact(
                            run_id=run_id,
                            local_path=str(manifest_file_path),
                            artifact_path=full_artifact_path
                        )
                    else:
                        # Use active run
                        mlflow.log_artifact(
                            local_path=str(manifest_file_path),
                            artifact_path=full_artifact_path
                        )
                
                try:
                    retry_with_backoff(
                        func=upload_manifest,
                        max_retries=3,
                        base_delay=1.0,
                        operation_name=f"manifest upload ({manifest_filename})"
                    )
                    manifest_success = True
                except Exception as upload_error:
                    logger.warning(
                        f"Failed to upload manifest using direct API: {upload_error}. "
                        f"Trying fallback method..."
                    )
                    # Fallback to log_artifact_safe (uses directory only)
                    manifest_success = log_artifact_safe(
                        local_path=manifest_file_path,
                        artifact_path=manifest_artifact_path,  # Directory only for fallback
                        run_id=run_id,
                        max_retries=3,
                        base_delay=1.0,
                    )
                
                if manifest_success:
                    logger.debug(f"Uploaded checkpoint manifest.json to {full_artifact_path}")
                else:
                    logger.warning(f"Failed to upload checkpoint manifest to {full_artifact_path}")
            finally:
                # Clean up temp directory and file
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_error:
                    logger.debug(f"Could not clean up temp directory {temp_dir}: {cleanup_error}")
        except Exception as e:
            logger.warning(f"Failed to upload checkpoint manifest: {e}")

    logger.info(f"Successfully uploaded checkpoint archive: {archive_path.name}")
    return True

