from __future__ import annotations

"""
@meta
name: tracking_mlflow_urls
type: utility
domain: tracking
responsibility:
  - Generate MLflow run URLs for UI navigation
  - Handle Azure ML and standard MLflow tracking URIs
inputs:
  - Experiment IDs and run IDs
outputs:
  - MLflow run URLs
tags:
  - utility
  - tracking
  - mlflow
  - urls
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""MLflow URL generation utilities."""
import mlflow
from common.shared.logging_utils import get_logger

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

