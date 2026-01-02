"""Base tracker class with common functionality."""

from __future__ import annotations

import mlflow
from shared.logging_utils import get_logger

logger = get_logger(__name__)


class BaseTracker:
    """Base class for MLflow trackers with common experiment setup."""

    def __init__(self, experiment_name: str):
        """
        Initialize tracker.

        Args:
            experiment_name: MLflow experiment name.
        """
        self.experiment_name = experiment_name
        self._setup_experiment()

    def _setup_experiment(self) -> None:
        """Set up MLflow experiment."""
        try:
            # Check if MLflow tracking URI is already set (e.g., by notebook setup)
            current_tracking_uri = mlflow.get_tracking_uri()
            is_azure_ml = current_tracking_uri and "azureml" in current_tracking_uri.lower()

            if is_azure_ml:
                # Azure ML tracking is already configured, just set the experiment
                logger.debug(
                    f"Using existing Azure ML tracking URI: {current_tracking_uri[:50]}...")
                # Use setup_mlflow_cross_platform for consistency (it will use existing tracking URI)
                from shared.mlflow_setup import setup_mlflow_cross_platform
                setup_mlflow_cross_platform(
                    experiment_name=self.experiment_name,
                    ml_client=None,  # Tracking URI already set, will use it
                    fallback_to_local=False,  # Don't override Azure ML tracking
                )
            else:
                # No Azure ML tracking set, use cross-platform setup
                from shared.mlflow_setup import setup_mlflow_cross_platform
                setup_mlflow_cross_platform(
                    experiment_name=self.experiment_name,
                    ml_client=None,  # Will use local tracking or env vars
                    fallback_to_local=True,
                )
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
            logger.warning("Continuing without MLflow tracking...")

