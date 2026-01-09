"""Best configuration selection from HPO sweep jobs using MLflow."""

from __future__ import annotations

from typing import Any, Dict, Optional

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job

# azureml-mlflow registers the 'azureml' URI scheme
try:
    import azureml.mlflow  # noqa: F401
except ImportError:
    pass


def _configure_mlflow(ml_client: MLClient) -> None:
    """Configure MLflow to use Azure ML workspace tracking URI."""
    from shared.mlflow_setup import setup_mlflow_cross_platform
    # Use setup_mlflow_cross_platform for consistency
    # We use a placeholder experiment name since setup_mlflow_cross_platform requires it
    # The actual experiment will be set by the caller if needed
    try:
        setup_mlflow_cross_platform(
            experiment_name="placeholder",  # Will be overridden by caller if needed
            ml_client=ml_client,
            fallback_to_local=False,
        )
    except Exception:
        # Fallback: manually set tracking URI if setup fails
        workspace = ml_client.workspaces.get(name=ml_client.workspace_name)
        tracking_uri = workspace.mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)


def _get_metrics_from_mlflow(run_id: str) -> Dict[str, float]:
    """Fetch metrics for a run from MLflow."""
    try:
        run = mlflow.get_run(run_id)
        return dict(run.data.metrics or {})
    except Exception:
        return {}


def _get_params_from_mlflow(run_id: str) -> Dict[str, Any]:
    """Fetch params for a run from MLflow."""
    try:
        run = mlflow.get_run(run_id)
        return dict(run.data.params or {})
    except Exception:
        return {}


def get_best_trial_from_sweep(
    ml_client: MLClient,
    sweep_job: Job,
    objective_metric: str,
    goal: str,
) -> tuple[Optional[Job], Optional[float]]:
    """
    Get the best trial from a sweep job based on the objective metric.

    Returns:
        Tuple of (best_trial_job, best_metric_value) or (None, None) if not found.
    """
    _configure_mlflow(ml_client)
    trials = list(ml_client.jobs.list(parent_job_name=sweep_job.name))

    best_trial = None
    best_value = None

    for trial in trials:
        if trial.status != "Completed":
            continue

        metrics = _get_metrics_from_mlflow(trial.name)
        if objective_metric not in metrics:
            continue

        value = metrics[objective_metric]
        if best_value is None:
            best_value = value
            best_trial = trial
        elif goal == "maximize" and value > best_value:
            best_value = value
            best_trial = trial
        elif goal == "minimize" and value < best_value:
            best_value = value
            best_trial = trial

    return best_trial, best_value


def extract_trial_configuration(
    ml_client: MLClient,
    trial: Job,
    dataset_version: str,
) -> Dict[str, Any]:
    """Extract configuration from a trial run using MLflow."""
    _configure_mlflow(ml_client)

    return {
        "trial_name": trial.name,
        "trial_id": trial.id,
        "backbone": trial.tags.get("backbone", "unknown"),
        "hyperparameters": _get_params_from_mlflow(trial.name),
        "metrics": _get_metrics_from_mlflow(trial.name),
        "dataset_version": trial.tags.get("data_version", dataset_version),
    }


def select_best_configuration(
    ml_client: MLClient,
    hpo_completed_jobs: Dict[str, Job],
    hpo_config: Dict[str, Any],
    dataset_version: str,
) -> Dict[str, Any]:
    """
    Select the best configuration across all backbone sweep jobs.

    Uses MLflow as the canonical source for metrics.

    Raises:
        ValueError: If no valid trials found.
    """
    objective_metric = hpo_config["objective"]["metric"]
    goal = hpo_config["objective"]["goal"]

    best_trial = None
    best_value = None
    best_backbone = None

    for backbone, sweep_job in hpo_completed_jobs.items():
        trial, value = get_best_trial_from_sweep(
            ml_client, sweep_job, objective_metric, goal
        )
        if trial is None:
            continue

        if best_value is None:
            best_value = value
            best_trial = trial
            best_backbone = backbone
        elif goal == "maximize" and value > best_value:
            best_value = value
            best_trial = trial
            best_backbone = backbone
        elif goal == "minimize" and value < best_value:
            best_value = value
            best_trial = trial
            best_backbone = backbone

    if best_trial is None:
        # Diagnostic error message
        _configure_mlflow(ml_client)
        error_parts = [
            f"No valid trials found in any sweep job.",
            f"Looking for metric '{objective_metric}' with goal '{goal}'.",
            f"Checked {len(hpo_completed_jobs)} sweep job(s): {list(hpo_completed_jobs.keys())}",
        ]
        for backbone, sweep_job in list(hpo_completed_jobs.items())[:3]:
            trials = list(ml_client.jobs.list(parent_job_name=sweep_job.name))
            completed = [t for t in trials if t.status == "Completed"]
            sample_metrics = []
            if completed:
                sample_metrics = list(_get_metrics_from_mlflow(completed[0].name).keys())[:5]
            error_parts.append(
                f"  {backbone} (sweep: {sweep_job.name}): "
                f"{len(completed)} completed. Sample metrics: {sample_metrics}"
            )
        raise ValueError("\n".join(error_parts))

    best_config = extract_trial_configuration(ml_client, best_trial, dataset_version)
    best_config["selection_criteria"] = {
        "metric": objective_metric,
        "goal": goal,
        "best_value": best_value,
        "backbone": best_backbone,
    }
    return best_config

