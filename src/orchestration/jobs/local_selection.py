"""Best configuration selection from local Optuna HPO studies."""

from __future__ import annotations

from typing import Any, Dict

import optuna


def extract_best_config_from_study(
    study: optuna.Study,
    backbone: str,
    dataset_version: str,
) -> Dict[str, Any]:
    """
    Extract best configuration from an Optuna study.

    Args:
        study: Completed Optuna study.
        backbone: Model backbone name.
        dataset_version: Dataset version string.

    Returns:
        Configuration dictionary matching Azure ML format with:
        - backbone
        - hyperparameters
        - metrics
        - selection_criteria
    """
    if study.best_trial is None:
        raise ValueError(f"No completed trials found in study '{study.study_name}'")

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

    # Get study direction to determine metric name
    direction = study.direction.name if hasattr(study.direction, "name") else "maximize"
    metric_name = "macro-f1"  # Default, should come from HPO config

    return {
        "trial_name": f"trial_{best_trial.number}",
        "trial_id": str(best_trial.number),
        "backbone": backbone,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "dataset_version": dataset_version,
        "selection_criteria": {
            "metric": metric_name,
            "goal": direction,
            "best_value": objective_value if best_trial.values else None,
            "backbone": backbone,
        },
    }


def select_best_configuration_across_studies(
    studies: Dict[str, optuna.Study],
    hpo_config: Dict[str, Any],
    dataset_version: str,
) -> Dict[str, Any]:
    """
    Select the best configuration across multiple backbone studies.

    Args:
        studies: Dictionary mapping backbone names to Optuna studies.
        hpo_config: HPO configuration dictionary (for objective metric).
        dataset_version: Dataset version string.

    Returns:
        Best configuration dictionary across all backbones.

    Raises:
        ValueError: If no valid trials found in any study.
    """
    objective_metric = hpo_config["objective"]["metric"]
    goal = hpo_config["objective"]["goal"]

    best_config = None
    best_value = None
    best_backbone = None

    for backbone, study in studies.items():
        if study.best_trial is None:
            continue

        trial_value = study.best_trial.values[0] if study.best_trial.values else None
        if trial_value is None:
            continue

        if best_value is None:
            best_value = trial_value
            best_config = extract_best_config_from_study(study, backbone, dataset_version)
            best_backbone = backbone
        elif goal == "maximize" and trial_value > best_value:
            best_value = trial_value
            best_config = extract_best_config_from_study(study, backbone, dataset_version)
            best_backbone = backbone
        elif goal == "minimize" and trial_value < best_value:
            best_value = trial_value
            best_config = extract_best_config_from_study(study, backbone, dataset_version)
            best_backbone = backbone

    if best_config is None:
        error_parts = [
            f"No valid trials found in any study.",
            f"Looking for metric '{objective_metric}' with goal '{goal}'.",
            f"Checked {len(studies)} study/studies: {list(studies.keys())}",
        ]
        for backbone, study in list(studies.items())[:3]:
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            error_parts.append(
                f"  {backbone}: {completed} completed trials"
            )
        raise ValueError("\n".join(error_parts))

    return best_config

