"""Shared MLflow utility functions.

This module re-exports MLflow utilities from tracking.mlflow for backward compatibility.
The functions have been moved to tracking.mlflow as part of the MLflow utilities consolidation.
"""

from __future__ import annotations

# Re-export get_mlflow_run_url from tracking.mlflow.urls for backward compatibility
# The function has been moved to tracking.mlflow.urls as part of the MLflow utilities consolidation.
# Use try/except to handle cases where path isn't set up during pytest collection
try:
    from tracking.mlflow.urls import get_mlflow_run_url  # noqa: F401
    from tracking.mlflow.utils import retry_with_backoff  # noqa: F401
except ImportError:
    # During pytest collection, path might not be set up yet
    # These will be imported when actually needed
    get_mlflow_run_url = None
    retry_with_backoff = None

