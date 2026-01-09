"""MLflow utilities for artifact uploads, run lifecycle, and compatibility.

This module provides centralized utilities for MLflow operations including:
- Azure ML compatibility patches
- Safe artifact uploads with retry logic
- Run lifecycle management (creation, termination)
- URL generation

The Azure ML compatibility patch is automatically applied when this module is imported.
"""

# Import compatibility module first to auto-apply Azure ML patch
from tracking.mlflow.compatibility import apply_azureml_artifact_patch  # noqa: F401

# Export artifact utilities
from tracking.mlflow.artifacts import (
    log_artifact_safe,
    log_artifacts_safe,
    upload_checkpoint_archive,
)

# Export lifecycle utilities
from tracking.mlflow.lifecycle import (
    terminate_run_safe,
    ensure_run_terminated,
    terminate_run_with_tags,
)

# Export run creation utilities
from tracking.mlflow.runs import (
    create_child_run,
    create_run_safe,
    get_or_create_experiment,
    resolve_experiment_id,
)

# Export URL utilities
from tracking.mlflow.urls import get_mlflow_run_url

# Re-export retry_with_backoff for convenience
from tracking.mlflow.utils import retry_with_backoff

# Export setup utilities
from tracking.mlflow.setup import setup_mlflow_for_stage

__all__ = [
    # Compatibility
    "apply_azureml_artifact_patch",
    # Artifacts
    "log_artifact_safe",
    "log_artifacts_safe",
    "upload_checkpoint_archive",
    # Lifecycle
    "terminate_run_safe",
    "ensure_run_terminated",
    "terminate_run_with_tags",
    # Runs
    "create_child_run",
    "create_run_safe",
    "get_or_create_experiment",
    "resolve_experiment_id",
    # URLs
    "get_mlflow_run_url",
    # Utilities
    "retry_with_backoff",
    # Setup
    "setup_mlflow_for_stage",
]

