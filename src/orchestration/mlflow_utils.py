"""MLflow utility functions for experiment tracking.

DEPRECATED: This module is maintained for backward compatibility only.
Please use tracking.mlflow.setup.setup_mlflow_for_stage instead.
"""

import warnings
from typing import Optional

# Re-export from new location with deprecation warning
from tracking.mlflow.setup import setup_mlflow_for_stage

warnings.warn(
    "The 'orchestration.mlflow_utils' module is deprecated. "
    "Please use 'tracking.mlflow.setup' module instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["setup_mlflow_for_stage"]

