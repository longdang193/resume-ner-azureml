"""MLflow artifact management utilities.

This module re-exports artifact functions for backward compatibility.
"""

from __future__ import annotations

# Re-export checkpoint archive functions from manager
from infrastructure.tracking.mlflow.artifacts.manager import (
    create_checkpoint_archive,
    should_skip_file,
)

# Import artifact upload functions from the artifacts.py file (sibling file)
# Use importlib to avoid circular import issues
import importlib.util
from pathlib import Path
import logging

_artifacts_file_path = Path(__file__).parent.parent / "artifacts.py"
if _artifacts_file_path.exists():
    # Load the module with a temporary name
    _artifacts_spec = importlib.util.spec_from_file_location("tracking.mlflow.artifacts._file", _artifacts_file_path)
    if _artifacts_spec and _artifacts_spec.loader:
        _artifacts_file_module = importlib.util.module_from_spec(_artifacts_spec)
        _artifacts_spec.loader.exec_module(_artifacts_file_module)
        
        # Fix the logger name to show the correct module name
        # The logger was created with __name__ = "tracking.mlflow.artifacts._file"
        # We need to replace it with a logger that has the correct name
        if hasattr(_artifacts_file_module, 'logger'):
            old_logger = _artifacts_file_module.logger
            # Create new logger with correct name
            new_logger = logging.getLogger("tracking.mlflow.artifacts")
            # Copy configuration from old logger
            new_logger.setLevel(old_logger.level)
            # Don't copy handlers - let logging system handle that
            # Update the module's logger reference
            _artifacts_file_module.logger = new_logger
        
        log_artifact_safe = _artifacts_file_module.log_artifact_safe
        log_artifacts_safe = _artifacts_file_module.log_artifacts_safe
        upload_checkpoint_archive = _artifacts_file_module.upload_checkpoint_archive
    else:
        log_artifact_safe = None
        log_artifacts_safe = None
        upload_checkpoint_archive = None
else:
    log_artifact_safe = None
    log_artifacts_safe = None
    upload_checkpoint_archive = None

__all__ = [
    "create_checkpoint_archive",
    "should_skip_file",
    "log_artifact_safe",
    "log_artifacts_safe",
    "upload_checkpoint_archive",
]
