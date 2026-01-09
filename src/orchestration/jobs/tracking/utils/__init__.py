"""MLflow utility functions for helpers and general utilities.

This module re-exports utility functions for backward compatibility.
New code should import directly from tracking.mlflow.*
"""

from __future__ import annotations

# Re-export all utility functions from new consolidated location
# Lazy import to avoid pytest collection issues
try:
    from tracking.mlflow.runs import create_child_run
    from tracking.mlflow.urls import get_mlflow_run_url
    from tracking.mlflow.utils import retry_with_backoff
except ImportError:
    # During pytest collection, path might not be set up yet
    # Will be imported when actually needed
    create_child_run = None
    get_mlflow_run_url = None
    retry_with_backoff = None

__all__ = [
    "create_child_run",
    "get_mlflow_run_url",
    "retry_with_backoff",
]
