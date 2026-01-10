"""Shared MLflow utilities for child run creation.

This module re-exports create_child_run from infrastructure.tracking.mlflow.runs for backward compatibility.
The function has been moved to tracking.mlflow.runs as part of the MLflow utilities consolidation.
"""

from __future__ import annotations

# Re-export from new location for backward compatibility
# Use try/except to handle cases where path isn't set up during pytest collection
try:
    from infrastructure.tracking.mlflow.runs import create_child_run
except ImportError:
    # During pytest collection, path might not be set up yet
    # Will be imported when actually needed
    create_child_run = None

__all__ = ["create_child_run"]
