"""Shared MLflow utility functions."""

from __future__ import annotations

import random
import time
from typing import Any, Callable

import mlflow
from shared.logging_utils import get_logger

logger = get_logger(__name__)


def get_mlflow_run_url(experiment_id: str, run_id: str) -> str:
    """
    Construct MLflow run URL from experiment ID and run ID.

    Handles both Azure ML and standard MLflow tracking URIs.

    Args:
        experiment_id: MLflow experiment ID.
        run_id: MLflow run ID.

    Returns:
        URL string for viewing the run in MLflow UI.
    """
    try:
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri and "azureml" in tracking_uri.lower():
            # Azure ML tracking URI format:
            # azureml://<region>.api.azureml.ms/mlflow/v2.0/subscriptions/<sub_id>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<ws_name>
            # UI URL format:
            # https://<region>.api.azureml.ms/mlflow/v2.0/subscriptions/<sub_id>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<ws_name>/#/experiments/<exp_id>/runs/<run_id>
            if tracking_uri.startswith("azureml://"):
                # Convert azureml:// to https://
                base_url = tracking_uri.replace("azureml://", "https://")
                return f"{base_url}/#/experiments/{experiment_id}/runs/{run_id}"
            else:
                # Already https:// or other format
                base_url = tracking_uri.split(
                    "/mlflow")[0] if "/mlflow" in tracking_uri else tracking_uri
                return f"{base_url}/#/experiments/{experiment_id}/runs/{run_id}"
        else:
            # Standard MLflow tracking URI
            return f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
    except Exception as e:
        logger.debug(f"Could not construct MLflow run URL: {e}")
        return f"<tracking_uri>/#/experiments/{experiment_id}/runs/{run_id}"


def retry_with_backoff(
    func: Callable,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    operation_name: str = "operation"
) -> Any:
    """
    Retry a function with exponential backoff and jitter.

    Args:
        func: Function to retry (callable that takes no arguments).
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum delay in seconds.
        operation_name: Name of operation for logging.

    Returns:
        Result of func() if successful.

    Raises:
        Exception: Original exception if all retries exhausted.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            # Detect retryable errors
            is_retryable = any(
                code in error_str
                for code in ['429', '503', '504', 'timeout', 'connection', 'temporary', 'rate limit']
            )

            if not is_retryable or attempt == max_retries - 1:
                # Not retryable or last attempt - raise original exception
                raise

            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)  # 10% jitter
            total_delay = delay + jitter

            logger.debug(
                f"Retry {attempt + 1}/{max_retries} for {operation_name} "
                f"after {total_delay:.2f}s (error: {str(e)[:100]})"
            )
            time.sleep(total_delay)

    # Should never reach here, but just in case
    raise RuntimeError(f"Retry logic exhausted for {operation_name}")

