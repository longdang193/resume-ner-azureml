"""Local hyperparameter optimization using Optuna."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import mlflow
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import RandomSampler
from optuna.trial import Trial

from shared.json_cache import save_json


def translate_search_space_to_optuna(
    hpo_config: Dict[str, Any], trial: Trial
) -> Dict[str, Any]:
    """
    Translate HPO config search space to Optuna trial suggestions.

    Args:
        hpo_config: HPO configuration dictionary with search_space.
        trial: Optuna trial object for suggesting values.

    Returns:
        Dictionary of hyperparameter values for this trial.
    """
    search_space = hpo_config["search_space"]
    params: Dict[str, Any] = {}

    for name, spec in search_space.items():
        p_type = spec["type"]
        if p_type == "choice":
            params[name] = trial.suggest_categorical(name, spec["values"])
        elif p_type == "uniform":
            params[name] = trial.suggest_float(
                name, float(spec["min"]), float(spec["max"])
            )
        elif p_type == "loguniform":
            params[name] = trial.suggest_float(
                name, float(spec["min"]), float(spec["max"]), log=True
            )
        else:
            raise ValueError(f"Unsupported search space type: {p_type}")

    return params


def create_optuna_pruner(hpo_config: Dict[str, Any]) -> Optional[Any]:
    """
    Create Optuna pruner from HPO config early termination policy.

    Args:
        hpo_config: HPO configuration dictionary.

    Returns:
        Optuna pruner instance or None if no early termination configured.
    """
    if "early_termination" not in hpo_config:
        return None

    et_cfg = hpo_config["early_termination"]
    policy = et_cfg.get("policy", "").lower()

    if policy == "bandit":
        # Optuna doesn't have exact bandit pruner, use MedianPruner as closest alternative
        # Bandit policy: stop if trial is worse than best by slack_factor
        # MedianPruner: stop if trial is worse than median
        return MedianPruner(
            n_startup_trials=et_cfg.get("delay_evaluation", 2),
            n_warmup_steps=et_cfg.get("evaluation_interval", 1),
        )
    elif policy == "median":
        return MedianPruner(
            n_startup_trials=et_cfg.get("delay_evaluation", 2),
            n_warmup_steps=et_cfg.get("evaluation_interval", 1),
        )
    else:
        return None


def run_training_trial(
    trial_params: Dict[str, Any],
    dataset_path: str,
    config_dir: Path,
    backbone: str,
    output_dir: Path,
    train_config: Dict[str, Any],
    mlflow_experiment_name: str,
    objective_metric: str = "macro-f1",
) -> float:
    """
    Execute a single training trial with given hyperparameters.

    Args:
        trial_params: Hyperparameters for this trial.
        dataset_path: Path to dataset directory.
        config_dir: Path to configuration directory.
        backbone: Model backbone name.
        output_dir: Output directory for checkpoint.
        train_config: Training configuration dictionary.
        mlflow_experiment_name: MLflow experiment name.
        objective_metric: Name of the objective metric to optimize.

    Returns:
        Objective metric value (e.g., macro-f1).
    """
    # Build command arguments
    root_dir = Path.cwd()
    args = [
        sys.executable,
        str(root_dir / "src" / "train.py"),
        "--data-asset",
        dataset_path,
        "--config-dir",
        str(config_dir),
        "--backbone",
        backbone,
    ]

    # Add hyperparameters from trial
    if "learning_rate" in trial_params:
        args.extend(["--learning-rate", str(trial_params["learning_rate"])])
    if "batch_size" in trial_params:
        args.extend(["--batch-size", str(trial_params["batch_size"])])
    if "dropout" in trial_params:
        args.extend(["--dropout", str(trial_params["dropout"])])
    if "weight_decay" in trial_params:
        args.extend(["--weight-decay", str(trial_params["weight_decay"])])

    # Use minimal epochs for HPO (from train config or default to 1)
    epochs = train_config.get("training", {}).get("epochs", 1)
    args.extend(["--epochs", str(epochs)])

    # Enable early stopping for HPO
    args.extend(["--early-stopping-enabled", "true"])

    # Set output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_output_dir = output_dir / f"trial_{trial_params.get('trial_number', 'unknown')}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable for output (train.py will use this)
    env = os.environ.copy()
    env["AZURE_ML_OUTPUT_checkpoint"] = str(trial_output_dir)

    # Run training (train.py will handle MLflow logging internally)
    result = subprocess.run(
        args,
        cwd=root_dir,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Trial failed with return code {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Training trial failed: {result.stderr}")

    # Try to read metrics from metrics.json file (created by train.py)
    # This is more reliable than querying MLflow
    metrics_file = trial_output_dir / "metrics.json"
    if metrics_file.exists():
        try:
            import json
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                if objective_metric in metrics:
                    return float(metrics[objective_metric])
                else:
                    print(f"Warning: Objective metric '{objective_metric}' not found in metrics.json")
                    print(f"Available metrics: {list(metrics.keys())}")
        except Exception as e:
            print(f"Warning: Could not read metrics.json: {e}")
    
    # Fallback: try to read from MLflow
    try:
        mlflow.set_experiment(mlflow_experiment_name)
        # Get the most recent run for this experiment
        experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
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
        
        print(f"Warning: Objective metric '{objective_metric}' not found in MLflow")
        return 0.0
    except Exception as e:
        print(f"Warning: Could not retrieve metrics from MLflow: {e}")
        return 0.0


def create_local_hpo_objective(
    dataset_path: str,
    config_dir: Path,
    backbone: str,
    hpo_config: Dict[str, Any],
    train_config: Dict[str, Any],
    output_base_dir: Path,
    mlflow_experiment_name: str,
    objective_metric: str = "macro-f1",
) -> Callable[[Trial], float]:
    """
    Create Optuna objective function for local HPO.

    Args:
        dataset_path: Path to dataset directory.
        config_dir: Path to configuration directory.
        backbone: Model backbone name.
        hpo_config: HPO configuration dictionary.
        train_config: Training configuration dictionary.
        output_base_dir: Base output directory for all trials.
        mlflow_experiment_name: MLflow experiment name.

    Returns:
        Objective function that takes an Optuna trial and returns metric value.
    """

    def objective(trial: Trial) -> float:
        # Sample hyperparameters
        trial_params = translate_search_space_to_optuna(hpo_config, trial)
        trial_params["backbone"] = backbone
        trial_params["trial_number"] = trial.number

        # Run training
        metric_value = run_training_trial(
            trial_params=trial_params,
            dataset_path=dataset_path,
            config_dir=config_dir,
            backbone=backbone,
            output_dir=output_base_dir,
            train_config=train_config,
            mlflow_experiment_name=mlflow_experiment_name,
            objective_metric=objective_metric,
        )

        # Report to Optuna
        trial.report(metric_value, step=0)
        return metric_value

    return objective


def run_local_hpo_sweep(
    dataset_path: str,
    config_dir: Path,
    backbone: str,
    hpo_config: Dict[str, Any],
    train_config: Dict[str, Any],
    output_dir: Path,
    mlflow_experiment_name: str,
) -> optuna.Study:
    """
    Run a local hyperparameter optimization sweep using Optuna.

    Args:
        dataset_path: Path to dataset directory.
        config_dir: Path to configuration directory.
        backbone: Model backbone name.
        hpo_config: HPO configuration dictionary.
        train_config: Training configuration dictionary.
        output_dir: Base output directory for all trials.
        mlflow_experiment_name: MLflow experiment name.

    Returns:
        Optuna study object with completed trials.
    """
    objective_metric = hpo_config["objective"]["metric"]
    goal = hpo_config["objective"]["goal"]
    direction = "maximize" if goal == "maximize" else "minimize"

    # Create pruner
    pruner = create_optuna_pruner(hpo_config)

    # Create sampler
    algorithm = hpo_config["sampling"]["algorithm"].lower()
    if algorithm == "random":
        sampler = RandomSampler()
    else:
        sampler = RandomSampler()  # Default to random

    # Create study
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=f"hpo_{backbone}",
    )

    # Get objective metric name
    objective_metric = hpo_config["objective"]["metric"]
    
    # Create objective function
    objective = create_local_hpo_objective(
        dataset_path=dataset_path,
        config_dir=config_dir,
        backbone=backbone,
        hpo_config=hpo_config,
        train_config=train_config,
        output_base_dir=output_dir,
        mlflow_experiment_name=mlflow_experiment_name,
        objective_metric=objective_metric,
    )

    # Run optimization
    max_trials = hpo_config["sampling"]["max_trials"]
    timeout_seconds = hpo_config["sampling"]["timeout_minutes"] * 60

    study.optimize(
        objective,
        n_trials=max_trials,
        timeout=timeout_seconds,
        show_progress_bar=True,
    )

    return study

