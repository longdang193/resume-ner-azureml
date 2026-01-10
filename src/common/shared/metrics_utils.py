from __future__ import annotations

"""
@meta
name: shared_metrics_utils
type: utility
domain: shared
responsibility:
  - Read metrics from JSON files
  - Read metrics from MLflow runs
inputs:
  - Metrics file paths
  - MLflow run IDs
outputs:
  - Metric values
tags:
  - utility
  - shared
  - metrics
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Utilities for reading metrics from files and MLflow."""
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .logging_utils import get_logger

logger = get_logger(__name__)

def read_metrics_from_file(
    metrics_file: Path,
    objective_metric: str,
    fallback_file: Optional[Path] = None,
) -> Optional[float]:
    """
    Read a specific metric from a metrics.json file.

    Args:
        metrics_file: Path to metrics.json file
        objective_metric: Name of the metric to read
        fallback_file: Optional fallback file path if primary doesn't exist

    Returns:
        Metric value if found, None otherwise
    """
    # Try primary file first
    file_to_read = metrics_file
    if not file_to_read.exists() and fallback_file and fallback_file.exists():
        logger.warning(
            f"Metrics file not found at {metrics_file}. "
            f"Using fallback: {fallback_file}. "
            "This may read metrics from a different trial!"
        )
        file_to_read = fallback_file

    if not file_to_read.exists():
        return None

    try:
        with open(file_to_read, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            if objective_metric in metrics:
                return float(metrics[objective_metric])
            else:
                logger.warning(
                    f"Objective metric '{objective_metric}' not found in metrics file. "
                    f"Available metrics: {list(metrics.keys())}"
                )
    except Exception as e:
        logger.warning(f"Could not read metrics from {file_to_read}: {e}")

    return None

def read_all_metrics_from_file(metrics_file: Path) -> Dict[str, Any]:
    """
    Read all metrics from a metrics.json file.

    Args:
        metrics_file: Path to metrics.json file

    Returns:
        Dictionary of all metrics, or empty dict if file doesn't exist or can't be read
    """
    if not metrics_file.exists():
        return {}

    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read metrics from {metrics_file}: {e}")
        return {}

def read_metric_from_mlflow(
    experiment_name: str,
    objective_metric: str,
    run_id: Optional[str] = None,
) -> Optional[float]:
    """
    Read a metric from MLflow (from most recent run or specific run).

    Args:
        experiment_name: MLflow experiment name
        objective_metric: Name of the metric to read
        run_id: Optional specific run ID (if None, uses most recent run)

    Returns:
        Metric value if found, None otherwise
    """
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        if run_id:
            run = mlflow.get_run(run_id)
            metrics = run.data.metrics
            if objective_metric in metrics:
                return float(metrics[objective_metric])
        else:
            # Get the most recent run for this experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=1,
                    order_by=["start_time DESC"],
                )
                if not runs.empty:
                    run_id = runs.iloc[0]["run_id"]
                    run = mlflow.get_run(run_id)
                    metrics = run.data.metrics
                    if objective_metric in metrics:
                        return float(metrics[objective_metric])

        logger.warning(
            f"Objective metric '{objective_metric}' not found in MLflow"
        )
    except Exception as e:
        logger.warning(f"Could not retrieve metrics from MLflow: {e}")

    return None

