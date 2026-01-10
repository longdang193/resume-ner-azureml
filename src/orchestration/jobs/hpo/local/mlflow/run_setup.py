"""MLflow run setup for local HPO.

This module provides backward compatibility by re-exporting from the new location.
"""

from __future__ import annotations

# Import from new location
from hpo.tracking.setup import (
    setup_hpo_mlflow_run,
    commit_run_name_version,
)

__all__ = [
    "setup_hpo_mlflow_run",
    "commit_run_name_version",
]


