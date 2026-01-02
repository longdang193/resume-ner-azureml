"""Trial metrics reading and storage utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from shared.logging_utils import get_logger
from shared.metrics_utils import read_all_metrics_from_file, read_metric_from_mlflow
from orchestration.constants import METRICS_FILENAME

logger = get_logger(__name__)


def read_trial_metrics(
    trial_output_dir: Path,
    root_dir: Path,
    objective_metric: str,
    mlflow_experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Read all metrics from trial output directory.

    Tries multiple strategies:
    1. Read from trial-specific metrics.json
    2. Fallback to default outputs/metrics.json
    3. Fallback to MLflow (if experiment_name provided)

    Args:
        trial_output_dir: Trial-specific output directory.
        root_dir: Project root directory.
        objective_metric: Name of the objective metric (for fallback to MLflow).
        mlflow_experiment_name: Optional MLflow experiment name for fallback.

    Returns:
        Dictionary of all metrics, or empty dict if not found.
    """
    metrics_file = trial_output_dir / METRICS_FILENAME

    # If trial-specific file doesn't exist, check if metrics were saved to default location
    if not metrics_file.exists():
        default_metrics = root_dir / "outputs" / METRICS_FILENAME
        if default_metrics.exists():
            logger.warning(
                f"Trial-specific metrics not found at {metrics_file}. "
                f"Falling back to default location: {default_metrics}. "
                f"This may read metrics from a different trial!"
            )
            metrics_file = default_metrics

    # Try to read from metrics.json file first
    metrics = read_all_metrics_from_file(metrics_file)
    if metrics:
        return metrics

    # If file doesn't exist, try fallback location
    default_metrics = root_dir / "outputs" / METRICS_FILENAME
    if not metrics_file.exists() and default_metrics.exists():
        metrics = read_all_metrics_from_file(default_metrics)
        if metrics:
            return metrics

    # Log error if file doesn't exist
    if not metrics_file.exists() and not default_metrics.exists():
        logger.error(
            f"metrics.json not found at expected location: {trial_output_dir / METRICS_FILENAME}. "
            f"Trial output dir: {trial_output_dir}, Root dir: {root_dir}"
        )

    # Fallback: try to read from MLflow (only objective metric)
    if mlflow_experiment_name:
        metric_value = read_metric_from_mlflow(
            experiment_name=mlflow_experiment_name,
            objective_metric=objective_metric,
        )
        if metric_value is not None:
            return {objective_metric: metric_value}

    return {}


def store_metrics_in_trial_attributes(
    trial: Any,
    output_base_dir: Path,
    trial_number: int,
    run_id: Optional[str] = None,
    fold_splits: Optional[list] = None,
) -> None:
    """
    Store additional metrics in trial user attributes for callback display.

    Reads metrics from trial output directory and stores key metrics in trial user attributes.

    Args:
        trial: Optuna trial object.
        output_base_dir: Base output directory for all trials.
        trial_number: Trial number.
        run_id: Optional run ID for directory naming.
        fold_splits: Optional fold splits for CV (to determine output directory).
    """
    # Determine trial output directory
    run_suffix = f"_{run_id}" if run_id else ""
    if fold_splits is not None:
        # CV: read from last fold's output (fold indices are 0-based, so last is len(fold_splits)-1)
        # Fold output directory format: trial_{number}_{run_id}_fold{fold_idx}
        last_fold_idx = len(fold_splits) - 1
        trial_output_dir = output_base_dir / f"trial_{trial_number}{run_suffix}_fold{last_fold_idx}"
    else:
        # Single training: read from trial output directory
        # Format: trial_{number}_{run_id}
        trial_output_dir = output_base_dir / f"trial_{trial_number}{run_suffix}"

    metrics_file = trial_output_dir / METRICS_FILENAME

    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
                # Store key metrics in user attributes for callback
                if "macro-f1-span" in all_metrics:
                    trial.set_user_attr("macro_f1_span", float(all_metrics["macro-f1-span"]))
                if "loss" in all_metrics:
                    trial.set_user_attr("loss", float(all_metrics["loss"]))
                if "per_entity" in all_metrics and isinstance(all_metrics["per_entity"], dict):
                    entity_count = len(all_metrics["per_entity"])
                    trial.set_user_attr("entity_count", entity_count)
                    # Store average entity F1
                    entity_f1s = [
                        v.get("f1", 0.0)
                        for v in all_metrics["per_entity"].values()
                        if isinstance(v, dict) and isinstance(v.get("f1"), (int, float))
                    ]
                    if entity_f1s:
                        trial.set_user_attr("avg_entity_f1", float(sum(entity_f1s) / len(entity_f1s)))
        except Exception:
            pass  # Silently fail if we can't read metrics

