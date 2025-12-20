"""HPO execution service for HPO pipeline tests.

This module is responsible solely for executing HPO sweeps and returning
structured results. It contains no orchestration, presentation, or configuration logic.
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add root directory to sys.path for module imports
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from orchestration.jobs.local_sweeps import run_local_hpo_sweep

from tests.fixtures.config.test_config_loader import DEFAULT_BACKBONE


def run_hpo_sweep_for_dataset(
    dataset_path: Path,
    config_dir: Path,
    hpo_config: Dict[str, Any],
    train_config: Dict[str, Any],
    output_dir: Path,
    mlflow_experiment_name: str,
    backbone: str = DEFAULT_BACKBONE,
) -> Dict[str, Any]:
    """
    Run HPO sweep for a dataset and return results summary.

    Args:
        dataset_path: Path to dataset directory
        config_dir: Path to configuration directory
        hpo_config: HPO configuration dictionary
        train_config: Training configuration dictionary
        output_dir: Output directory for trials
        mlflow_experiment_name: MLflow experiment name
        backbone: Model backbone name

    Returns:
        Dictionary with results summary (trials, best_trial, errors, etc.)
    """
    results = {
        "dataset_path": str(dataset_path),
        "backbone": backbone,
        "trials_completed": 0,
        "trials_failed": 0,
        "best_trial": None,
        "best_value": None,
        "best_params": None,
        "errors": [],
        "success": False,
    }

    try:
        study = run_local_hpo_sweep(
            dataset_path=str(dataset_path),
            config_dir=config_dir,
            backbone=backbone,
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            mlflow_experiment_name=mlflow_experiment_name,
        )

        completed_trials = [
            t for t in study.trials if t.state.name == "COMPLETE"]
        failed_trials = [t for t in study.trials if t.state.name == "FAIL"]

        results["trials_completed"] = len(completed_trials)
        results["trials_failed"] = len(failed_trials)
        results["total_trials"] = len(study.trials)

        if completed_trials:
            results["best_trial"] = study.best_trial.number
            results["best_value"] = study.best_trial.value
            results["best_params"] = study.best_trial.params
            results["success"] = True
        else:
            results["errors"].append("No completed trials")

        for trial in failed_trials:
            if hasattr(trial, "system_attrs") and "error" in trial.system_attrs:
                results["errors"].append(
                    f"Trial {trial.number}: {trial.system_attrs['error']}"
                )

    except Exception as e:
        results["errors"].append(f"HPO sweep failed: {str(e)}")
        results["success"] = False

    return results

