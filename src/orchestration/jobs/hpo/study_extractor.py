"""Extract configurations from Optuna studies."""

from __future__ import annotations

from typing import Any, Dict


def extract_best_config_from_study(
    study: Any,
    backbone: str,
    dataset_version: str,
    objective_metric: str = "macro-f1",
) -> Dict[str, Any]:
    """
    Extract best configuration from an Optuna study.

    Args:
        study: Completed Optuna study.
        backbone: Model backbone name.
        dataset_version: Dataset version string.
        objective_metric: Name of the objective metric (from HPO config).

    Returns:
        Configuration dictionary matching Azure ML format with:
        - backbone
        - hyperparameters
        - metrics
        - selection_criteria
    """
    if study.best_trial is None:
        raise ValueError(
            f"No completed trials found in study '{study.study_name}'")

    best_trial = study.best_trial

    # Extract hyperparameters (exclude backbone and trial_number if present)
    hyperparameters = {
        k: v
        for k, v in best_trial.params.items()
        if k not in ("backbone", "trial_number")
    }

    # Extract metrics
    metrics = {}
    if best_trial.values:
        # Optuna stores objective values in trial.values
        # We'll use the first value as the primary metric
        objective_value = best_trial.values[0] if best_trial.values else None
        if objective_value is not None:
            # Get the objective metric name from study direction
            # For now, we'll use a default name - this should match HPO config
            metrics["objective_value"] = objective_value

    direction = study.direction.name if hasattr(
        study.direction, "name") else "maximize"

    # Extract CV statistics if available
    cv_stats = {}
    if hasattr(best_trial, "user_attrs"):
        cv_mean = best_trial.user_attrs.get("cv_mean")
        cv_std = best_trial.user_attrs.get("cv_std")
        cv_fold_metrics = best_trial.user_attrs.get("cv_fold_metrics")
        if cv_mean is not None:
            cv_stats = {
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "cv_fold_metrics": cv_fold_metrics,
            }

    return {
        "trial_name": f"trial_{best_trial.number}",
        "trial_id": str(best_trial.number),
        "backbone": backbone,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "dataset_version": dataset_version,
        "selection_criteria": {
            "metric": objective_metric,
            "goal": direction,
            "best_value": objective_value if best_trial.values else None,
            "backbone": backbone,
        },
        "cv_statistics": cv_stats if cv_stats else None,
    }

