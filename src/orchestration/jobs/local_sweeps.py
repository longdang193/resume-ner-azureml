"""Local hyperparameter optimization using Optuna."""

from __future__ import annotations
from shared.json_cache import save_json

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List, Tuple

import mlflow
import numpy as np

# Lazy import optuna - only import when actually needed for local execution
# This prevents optuna from being required when using Azure ML orchestration


def _import_optuna():
    """Lazy import optuna and related modules."""
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import RandomSampler
        from optuna.trial import Trial
        return optuna, MedianPruner, RandomSampler, Trial
    except ImportError as e:
        raise ImportError(
            "optuna is required for local HPO execution. "
            "Install it with: pip install optuna"
        ) from e


def translate_search_space_to_optuna(
    hpo_config: Dict[str, Any], trial: Any, exclude_params: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Translate HPO config search space to Optuna trial suggestions.

    Args:
        hpo_config: HPO configuration dictionary with search_space.
        trial: Optuna trial object for suggesting values.
        exclude_params: Optional list of parameter names to exclude from search space.

    Returns:
        Dictionary of hyperparameter values for this trial.
    """
    search_space = hpo_config["search_space"]
    params: Dict[str, Any] = {}
    exclude_set = set(exclude_params or [])

    for name, spec in search_space.items():
        # Skip excluded parameters (e.g., "backbone" when it's fixed per study)
        if name in exclude_set:
            continue

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

    # Lazy import optuna
    _, MedianPruner, _, _ = _import_optuna()

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
    fold_idx: Optional[int] = None,
    fold_splits_file: Optional[Path] = None,
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
    # Derive project root from config_dir (config_dir is ROOT_DIR / "config")
    root_dir = config_dir.parent
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

    # Add fold-specific arguments if CV is enabled
    if fold_idx is not None:
        args.extend(["--fold-idx", str(fold_idx)])
    if fold_splits_file is not None:
        args.extend(["--fold-splits-file", str(fold_splits_file)])

    # Set output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    trial_output_dir = output_dir / \
        f"trial_{trial_params.get('trial_number', 'unknown')}{fold_suffix}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    # Create MLflow experiment if it doesn't exist
    mlflow.set_experiment(mlflow_experiment_name)

    # Set environment variables for output and MLflow (train.py will use these)
    env = os.environ.copy()
    # Note: AzureMLOutputPathResolver converts output_name to uppercase, so use CHECKPOINT
    env["AZURE_ML_OUTPUT_CHECKPOINT"] = str(trial_output_dir)
    # Also set lowercase for backward compatibility
    env["AZURE_ML_OUTPUT_checkpoint"] = str(trial_output_dir)

    # Pass MLflow tracking URI and experiment name to subprocess
    mlflow_tracking_uri = mlflow.get_tracking_uri()
    if mlflow_tracking_uri:
        env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    env["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment_name

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
    # IMPORTANT: Always check trial-specific location first to avoid reading wrong metrics
    metrics_file = trial_output_dir / "metrics.json"

    # If trial-specific file doesn't exist, check if metrics were saved to default location
    # This can happen if the platform adapter doesn't properly detect AZURE_ML_OUTPUT_checkpoint
    if not metrics_file.exists():
        # Check if train.py saved to default outputs directory (due to platform adapter behavior)
        default_metrics = Path(root_dir) / "outputs" / "metrics.json"
        if default_metrics.exists():
            # Only use this as a last resort - it might be from a different trial!
            # Print a clear warning
            print(
                f"WARNING: Trial-specific metrics not found at {metrics_file}")
            print(f"  Falling back to default location: {default_metrics}")
            print(f"  This may read metrics from a different trial!")
            metrics_file = default_metrics

    if metrics_file.exists():
        try:
            import json
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                if objective_metric in metrics:
                    metric_value = float(metrics[objective_metric])
                    # Debug: print which file was read
                    file_mtime = os.path.getmtime(metrics_file)
                    print(
                        f"Read {objective_metric}={metric_value} from {metrics_file} (modified: {file_mtime})")
                    return metric_value
                else:
                    print(
                        f"Warning: Objective metric '{objective_metric}' not found in metrics.json")
                    print(f"Available metrics: {list(metrics.keys())}")
                    print(f"Metrics file location: {metrics_file}")
        except Exception as e:
            print(
                f"Warning: Could not read metrics.json from {metrics_file}: {e}")
    else:
        print(
            f"ERROR: metrics.json not found at expected location: {trial_output_dir / 'metrics.json'}")
        print(f"  Trial output dir: {trial_output_dir}")
        print(f"  Root dir: {root_dir}")
        print(
            f"  AZURE_ML_OUTPUT_checkpoint env var: {os.environ.get('AZURE_ML_OUTPUT_checkpoint', 'NOT SET')}")

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

        print(
            f"Warning: Objective metric '{objective_metric}' not found in MLflow")
        return 0.0
    except Exception as e:
        print(f"Warning: Could not retrieve metrics from MLflow: {e}")
        return 0.0


def run_training_trial_with_cv(
    trial_params: Dict[str, Any],
    dataset_path: str,
    config_dir: Path,
    backbone: str,
    output_dir: Path,
    train_config: Dict[str, Any],
    mlflow_experiment_name: str,
    objective_metric: str,
    fold_splits: List[Tuple[List[int], List[int]]],
    fold_splits_file: Path,
) -> Tuple[float, List[float]]:
    """
    Run training trial with k-fold cross-validation.

    Args:
        trial_params: Hyperparameters for this trial.
        dataset_path: Path to dataset directory.
        config_dir: Path to configuration directory.
        backbone: Model backbone name.
        output_dir: Output directory for checkpoints.
        train_config: Training configuration dictionary.
        mlflow_experiment_name: MLflow experiment name.
        objective_metric: Name of the objective metric to optimize.
        fold_splits: List of (train_indices, val_indices) tuples for each fold.
        fold_splits_file: Path to file containing fold splits (for train.py).

    Returns:
        Tuple of (average_metric, fold_metrics) where:
        - average_metric: Average metric across all folds
        - fold_metrics: List of metrics for each fold
    """
    fold_metrics = []

    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        # Run training for this fold
        fold_metric = run_training_trial(
            trial_params=trial_params,
            dataset_path=dataset_path,
            config_dir=config_dir,
            backbone=backbone,
            output_dir=output_dir,
            train_config=train_config,
            mlflow_experiment_name=mlflow_experiment_name,
            objective_metric=objective_metric,
            fold_idx=fold_idx,
            fold_splits_file=fold_splits_file,
        )
        fold_metrics.append(fold_metric)

    # Calculate average metric
    average_metric = np.mean(fold_metrics)

    return average_metric, fold_metrics


def create_local_hpo_objective(
    dataset_path: str,
    config_dir: Path,
    backbone: str,
    hpo_config: Dict[str, Any],
    train_config: Dict[str, Any],
    output_base_dir: Path,
    mlflow_experiment_name: str,
    objective_metric: str = "macro-f1",
    k_folds: Optional[int] = None,
    fold_splits_file: Optional[Path] = None,
) -> Callable[[Any], float]:
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
    # Load or create fold splits if k-fold CV is enabled
    fold_splits = None
    if k_folds is not None and k_folds > 1:
        if fold_splits_file and fold_splits_file.exists():
            # Load existing splits
            from training.cv_utils import load_fold_splits
            fold_splits, _ = load_fold_splits(fold_splits_file)
        elif fold_splits_file:
            # Create new splits and save them
            from training.data import load_dataset
            from training.cv_utils import create_kfold_splits, save_fold_splits

            dataset = load_dataset(dataset_path)
            train_data = dataset.get("train", [])

            k_fold_config = hpo_config.get("k_fold", {})
            random_seed = k_fold_config.get("random_seed", 42)
            shuffle = k_fold_config.get("shuffle", True)

            fold_splits = create_kfold_splits(
                dataset=train_data,
                k=k_folds,
                random_seed=random_seed,
                shuffle=shuffle,
            )

            # Save splits for reproducibility
            save_fold_splits(
                fold_splits,
                fold_splits_file,
                metadata={
                    "k": k_folds,
                    "random_seed": random_seed,
                    "shuffle": shuffle,
                }
            )

    def objective(trial: Any) -> float:
        # Sample hyperparameters
        # Exclude "backbone" from search space since it's fixed per study
        trial_params = translate_search_space_to_optuna(
            hpo_config, trial, exclude_params=["backbone"])
        # Set the fixed backbone for this study
        trial_params["backbone"] = backbone
        trial_params["trial_number"] = trial.number

        # Run training with or without CV
        if fold_splits is not None:
            # Run k-fold CV
            average_metric, fold_metrics = run_training_trial_with_cv(
                trial_params=trial_params,
                dataset_path=dataset_path,
                config_dir=config_dir,
                backbone=backbone,
                output_dir=output_base_dir,
                train_config=train_config,
                mlflow_experiment_name=mlflow_experiment_name,
                objective_metric=objective_metric,
                fold_splits=fold_splits,
                fold_splits_file=fold_splits_file,
            )

            # Log CV statistics to trial user attributes
            trial.set_user_attr("cv_mean", float(average_metric))
            trial.set_user_attr("cv_std", float(np.std(fold_metrics)))
            trial.set_user_attr("cv_fold_metrics", [
                                float(m) for m in fold_metrics])

            metric_value = average_metric
        else:
            # Run single training (no CV)
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
    k_folds: Optional[int] = None,
    fold_splits_file: Optional[Path] = None,
) -> Any:
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
    # Lazy import optuna
    optuna, _, RandomSampler, _ = _import_optuna()

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
        k_folds=k_folds,
        fold_splits_file=fold_splits_file,
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
