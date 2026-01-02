"""Local MLflow index cache for fast, backend-independent run retrieval.

This module re-exports all index functions for backward compatibility.
New code should import directly from orchestration.jobs.tracking.index.*
"""

from __future__ import annotations

# Re-export all index functions for backward compatibility
from orchestration.jobs.tracking.index.run_index import (
    get_mlflow_index_path,
    update_mlflow_index,
    find_in_mlflow_index,
)
from orchestration.jobs.tracking.index.version_counter import (
    get_run_name_counter_path,
    reserve_run_name_version,
    commit_run_name_version,
    cleanup_stale_reservations,
)

__all__ = [
    "get_mlflow_index_path",
    "update_mlflow_index",
    "find_in_mlflow_index",
    "get_run_name_counter_path",
    "reserve_run_name_version",
    "commit_run_name_version",
    "cleanup_stale_reservations",
]
