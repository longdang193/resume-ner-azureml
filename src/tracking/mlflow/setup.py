"""MLflow experiment setup utilities.

This module provides functions for setting up MLflow tracking for different stages.
"""

from __future__ import annotations

from typing import Optional

import mlflow


def setup_mlflow_for_stage(
    experiment_name: str,
    tracking_uri: Optional[str] = None
) -> None:
    """Setup MLflow tracking for a specific stage.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: Optional tracking URI (uses default if None).
    
    Note:
        For cross-platform tracking with Azure ML workspace support,
        consider using `shared.mlflow_setup.setup_mlflow_cross_platform()`
        instead, which provides platform-aware setup with Azure ML fallback.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


