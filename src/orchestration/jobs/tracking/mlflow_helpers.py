"""Shared MLflow utilities for child run creation.

This module re-exports helper functions for backward compatibility.
New code should import directly from tracking.mlflow.runs
"""

from __future__ import annotations

# Re-export for backward compatibility
# Use try/except to handle cases where path isn't set up during pytest collection
try:
    from tracking.mlflow.runs import create_child_run
except ImportError:
    # During pytest collection, path might not be set up yet
    # Will be imported when actually needed
    create_child_run = None

__all__ = [
    "create_child_run",
]
