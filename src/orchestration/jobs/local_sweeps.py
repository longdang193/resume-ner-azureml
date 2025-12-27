"""Local hyperparameter optimization using Optuna."""

from __future__ import annotations
from shared.json_cache import save_json

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List, Tuple

import mlflow
import numpy as np

from .checkpoint_manager import resolve_storage_path, get_storage_uri

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
    # Run train.py as a module to allow relative imports to work
    # This requires src/ to be in PYTHONPATH (set in env below)
    args = [
        sys.executable,
        "-m",
        "training.train",
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

    # Set output directory with unique run ID to prevent overwriting
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = trial_params.get('run_id', '')
    run_suffix = f"_{run_id}" if run_id else ""
    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    trial_output_dir = output_dir / \
        f"trial_{trial_params.get('trial_number', 'unknown')}{run_suffix}{fold_suffix}"
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

    # Pass parent run ID if we're in an active MLflow run
    # The subprocess will create a nested child run using this parent run ID
    try:
        active_run = mlflow.active_run()
        if active_run is not None:
            parent_run_id = active_run.info.run_id
            env["MLFLOW_PARENT_RUN_ID"] = parent_run_id
            # Also pass trial number for run naming
            env["MLFLOW_TRIAL_NUMBER"] = str(
                trial_params.get("trial_number", "unknown"))
            print(
                f"  [MLflow] Passing parent run ID to trial: {parent_run_id[:12]}... (trial {trial_params.get('trial_number', 'unknown')})")
        else:
            print(
                f"  [MLflow] Warning: No active MLflow run - trials will be independent runs")
    except Exception as e:
        # If MLflow is not available or no active run, continue without parent run ID
        print(f"  [MLflow] Warning: Could not get active run ID: {e}")
        pass

    # Add src directory to PYTHONPATH to allow relative imports in train.py
    src_dir = str(root_dir / "src")
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = src_dir

    # Run training (train.py will handle MLflow logging internally)
    result = subprocess.run(
        args,
        cwd=root_dir,
        env=env,
        capture_output=True,
        text=True,
    )

    # Print subprocess output for debugging (especially MLflow context manager output)
    # Check both stdout and stderr for MLflow-related messages
    mlflow_lines_found = False
    if result.stdout:
        # Filter and print relevant MLflow debug output
        for line in result.stdout.split('\n'):
            line_lower = line.lower()
            if '[mlflow' in line_lower or 'mlflow' in line_lower or 'mlflow_context' in line_lower:
                print(f"  [Subprocess STDOUT] {line}")
                mlflow_lines_found = True

    if result.stderr:
        # Also check stderr for MLflow messages
        for line in result.stderr.split('\n'):
            line_lower = line.lower()
            if '[mlflow' in line_lower or 'mlflow' in line_lower or 'mlflow_context' in line_lower:
                print(f"  [Subprocess STDERR] {line}")
                mlflow_lines_found = True

    # If no MLflow output found, print a warning and show first few lines of stderr for debugging
    if not mlflow_lines_found and result.returncode == 0:
        print(f"  [MLflow Debug] ⚠ No MLflow debug output found in subprocess")
        print(
            f"  [MLflow Debug] This may indicate the context manager isn't being called")
        if result.stderr:
            stderr_lines = result.stderr.split('\n')
            print(f"  [MLflow Debug] First 10 lines of stderr:")
            for line in stderr_lines[:10]:
                if line.strip():
                    print(f"    {line}")
        if result.stdout:
            stdout_lines = result.stdout.split('\n')
            print(f"  [MLflow Debug] First 10 lines of stdout:")
            for line in stdout_lines[:10]:
                if line.strip():
                    print(f"    {line}")

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
                    # Store full metrics for callback display (avoid duplication here)
                    # Just log that we read the objective metric
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
    run_id: Optional[str] = None,
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
            stratified = k_fold_config.get("stratified", False)

            fold_splits = create_kfold_splits(
                dataset=train_data,
                k=k_folds,
                random_seed=random_seed,
                shuffle=shuffle,
                stratified=stratified,
            )

            # Save splits for reproducibility
            save_fold_splits(
                fold_splits,
                fold_splits_file,
                metadata={
                    "k": k_folds,
                    "random_seed": random_seed,
                    "shuffle": shuffle,
                    "stratified": stratified,
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
        trial_params["run_id"] = run_id  # Pass run_id to trial functions

        # Don't create child runs in parent process - let subprocess create them
        # This avoids issues with ended runs and ensures training logs to the correct run
        # We'll pass the parent run ID and let the subprocess create nested child runs

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

        # Store additional metrics in trial user attributes for callback display
        # Try to read full metrics from the trial output directory
        # Use run_id from closure (parameter) - it was set in trial_params at line 432
        # Use closure variable directly to avoid UnboundLocalError
        run_suffix = f"_{run_id}" if run_id else ""
        if fold_splits is not None:
            # CV: read from last fold's output (fold indices are 0-based, so last is len(fold_splits)-1)
            # Fold output directory format: trial_{number}_{run_id}_fold{fold_idx}
            last_fold_idx = len(fold_splits) - 1
            trial_output_dir = output_base_dir / \
                f"trial_{trial.number}{run_suffix}_fold{last_fold_idx}"
            metrics_file = trial_output_dir / "metrics.json"
        else:
            # Single training: read from trial output directory
            # Format: trial_{number}_{run_id}
            trial_output_dir = output_base_dir / \
                f"trial_{trial.number}{run_suffix}"
            metrics_file = trial_output_dir / "metrics.json"

        if metrics_file.exists():
            try:
                import json
                with open(metrics_file, "r") as f:
                    all_metrics = json.load(f)
                    # Store key metrics in user attributes for callback
                    if "macro-f1-span" in all_metrics:
                        trial.set_user_attr("macro_f1_span", float(
                            all_metrics["macro-f1-span"]))
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
                            trial.set_user_attr("avg_entity_f1", float(
                                sum(entity_f1s) / len(entity_f1s)))
            except Exception:
                pass  # Silently fail if we can't read metrics

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
    checkpoint_config: Optional[Dict[str, Any]] = None,
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
        k_folds: Optional number of k-folds for cross-validation.
        fold_splits_file: Optional path to fold splits file.
        checkpoint_config: Optional checkpoint configuration dict with 'enabled', 
                          'storage_path', and 'auto_resume' keys.

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

    # Resolve checkpoint storage if enabled
    checkpoint_config = checkpoint_config or {}
    storage_path = resolve_storage_path(
        output_dir=output_dir,
        checkpoint_config=checkpoint_config,
        backbone=backbone,
    )
    storage_uri = get_storage_uri(storage_path)

    # Determine if we should resume
    auto_resume = checkpoint_config.get(
        "auto_resume", True) if checkpoint_config.get("enabled", False) else False
    should_resume = auto_resume and storage_path is not None and storage_path.exists()

    # Generate unique run ID (timestamp-based) to prevent overwriting on reruns
    # This is used for both study naming and MLflow run naming
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create or load study
    # Use unique study name when starting fresh to avoid conflicts
    # When resuming, use the base name to load existing study
    if should_resume:
        # Use base name to resume existing study
        study_name = f"hpo_{backbone}"
    else:
        # Use unique name for fresh start
        study_name = f"hpo_{backbone}_{run_id}"

    if should_resume:
        print(
            f"\n[HPO] Resuming optimization for {backbone} from checkpoint...")
        print(f"  Checkpoint: {storage_path}")
        try:
            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                study_name=study_name,
                storage=storage_uri,
                load_if_exists=True,
            )
            # Count completed trials
            completed_trials = len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            print(
                f"  Loaded {len(study.trials)} existing trials ({completed_trials} completed)")
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
            print(f"  Creating new study instead...")
            # Use unique study name when resume fails
            study_name = f"hpo_{backbone}_{run_id}"
            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                study_name=study_name,
                storage=storage_uri,
                load_if_exists=False,
            )
            should_resume = False
    else:
        if storage_uri:
            print(
                f"\n[HPO] Starting optimization for {backbone} with checkpointing...")
            print(f"  Checkpoint: {storage_path}")
        else:
            print(f"\n[HPO] Starting optimization for {backbone}...")
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=False,
        )

    # Get objective metric name
    objective_metric = hpo_config["objective"]["metric"]

    # Run ID was already generated above for study naming
    # Print it here for user visibility
    print(f"  Run ID: {run_id} (prevents overwriting on reruns)")

    # Set MLflow experiment (safe to call even if already set)
    try:
        mlflow.set_experiment(mlflow_experiment_name)
    except Exception as e:
        print(f"  Warning: Could not set MLflow experiment: {e}")
        print("  Continuing without MLflow tracking...")

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
        run_id=run_id,
    )

    # Create callback to display additional metrics after each trial
    def trial_complete_callback(study: Any, trial: Any) -> None:
        """Callback to print additional metrics after trial completes."""
        # Import optuna for TrialState enum
        optuna_module, _, _, _ = _import_optuna()
        if trial.state == optuna_module.trial.TrialState.COMPLETE:
            # Print clearer value with metric name (Optuna's default shows "value" which is unclear)
            # Format: metric_name=value (e.g., macro-f1=0.838824)
            print(f"  {objective_metric}={trial.value:.6f}")

            attrs = trial.user_attrs
            extra_info = []

            if "macro_f1_span" in attrs:
                extra_info.append(
                    f"macro-f1-span={attrs['macro_f1_span']:.6f}")
            if "loss" in attrs:
                extra_info.append(f"loss={attrs['loss']:.6f}")
            if "avg_entity_f1" in attrs:
                entity_count = attrs.get("entity_count", "?")
                extra_info.append(
                    f"avg_entity_f1={attrs['avg_entity_f1']:.6f} ({entity_count} entities)")

            if extra_info:
                print(f"  Additional metrics: {' | '.join(extra_info)}")

    # Calculate remaining trials
    max_trials = hpo_config["sampling"]["max_trials"]
    timeout_seconds = hpo_config["sampling"]["timeout_minutes"] * 60

    # Create MLflow parent run for HPO sweep
    mlflow_run_name = f"hpo_{backbone}_{run_id}"

    # Log MLflow tracking URI for debugging
    tracking_uri = mlflow.get_tracking_uri()
    if tracking_uri:
        # Check if it's actually Azure ML (starts with azureml://)
        if tracking_uri.lower().startswith("azureml://"):
            print(f"  ✓ Using Azure ML Workspace for MLflow tracking")
            print(f"    Tracking URI: {tracking_uri}")
        elif tracking_uri.startswith("sqlite://") or tracking_uri.startswith("file://"):
            print(f"  ⚠ Using LOCAL MLflow tracking (not Azure ML)")
            print(f"    Tracking URI: {tracking_uri}")
            print(f"    To use Azure ML, ensure:")
            print(f"      1. config/mlflow.yaml has azure_ml.enabled: true")
            print(
                f"      2. Environment variables are set: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP")
            print(
                f"      3. Azure ML SDK is installed: pip install azure-ai-ml azure-identity azureml-mlflow")
        else:
            print(f"  Using MLflow tracking: {tracking_uri}")
    else:
        print("  Warning: MLflow tracking URI not set")

    try:
        with mlflow.start_run(run_name=mlflow_run_name):
            # Log HPO parameters
            mlflow.log_param("backbone", backbone)
            mlflow.log_param("max_trials", max_trials)
            mlflow.log_param("study_name", study_name)
            mlflow.log_param("objective_metric", objective_metric)
            mlflow.log_param("checkpoint_enabled",
                             checkpoint_config.get("enabled", False))

            # Log checkpoint path (even if disabled, log None)
            if storage_path is not None:
                mlflow.log_param("checkpoint_path",
                                 str(storage_path.resolve()))
            else:
                mlflow.log_param("checkpoint_path", None)

            # Log checkpoint storage type
            mlflow.log_param("checkpoint_storage_type",
                             "sqlite" if storage_path else None)

            # Log resume status
            mlflow.log_param("resumed_from_checkpoint", should_resume)

            if should_resume:
                # Count only completed trials (not FAILED, PRUNED, etc.)
                completed_trials = len([
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ])
                remaining_trials = max(0, max_trials - completed_trials)

                if remaining_trials > 0:
                    print(
                        f"  Running {remaining_trials} more trials (already completed {completed_trials}/{max_trials})")
                    study.optimize(
                        objective,
                        n_trials=remaining_trials,
                        timeout=timeout_seconds,
                        show_progress_bar=True,
                        callbacks=[trial_complete_callback],
                    )
                else:
                    print(f"  All {max_trials} trials already completed!")
            else:
                # Run all trials
                study.optimize(
                    objective,
                    n_trials=max_trials,
                    timeout=timeout_seconds,
                    show_progress_bar=True,
                    callbacks=[trial_complete_callback],
                )

            # Log final metrics after optimization
            completed_trials = len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            mlflow.log_metric("n_trials", len(study.trials))
            mlflow.log_metric("n_completed_trials", completed_trials)

            if study.best_trial is not None and study.best_value is not None:
                # Log only the metric-specific name to avoid duplication
                # (Optuna already prints "Best value" in console output)
                mlflow.log_metric(f"best_{objective_metric}", study.best_value)
                # Log best hyperparameters
                for param_name, param_value in study.best_params.items():
                    mlflow.log_param(f"best_{param_name}", param_value)
    except Exception as e:
        # Gracefully handle MLflow failures - don't fail HPO if MLflow is unavailable
        print(f"  Warning: MLflow tracking failed: {e}")
        print("  Continuing HPO without MLflow tracking...")

        # Run optimization without MLflow context
        if should_resume:
            completed_trials = len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            remaining_trials = max(0, max_trials - completed_trials)

            if remaining_trials > 0:
                print(
                    f"  Running {remaining_trials} more trials (already completed {completed_trials}/{max_trials})")
                study.optimize(
                    objective,
                    n_trials=remaining_trials,
                    timeout=timeout_seconds,
                    show_progress_bar=True,
                    callbacks=[trial_complete_callback],
                )
            else:
                print(f"  All {max_trials} trials already completed!")
        else:
            study.optimize(
                objective,
                n_trials=max_trials,
                timeout=timeout_seconds,
                show_progress_bar=True,
                callbacks=[trial_complete_callback],
            )

    return study
