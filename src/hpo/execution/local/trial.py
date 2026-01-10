"""Trial execution for HPO training runs.

Handles subprocess execution, environment setup, and metrics reading.
Combines TrialExecutor class and run_training_trial function.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from paths import find_project_root
from shared.logging_utils import get_logger
from training.execution import (
    FoldConfig,
    MLflowConfig,
    TrialConfig,
    TrainingOptions,
    build_training_command,
    execute_training_subprocess,
    setup_training_environment,
    verify_training_environment,
)

from hpo.trial.metrics import read_trial_metrics

logger = get_logger(__name__)


class TrialExecutor:
    """Executes a single training trial via subprocess and returns metrics."""

    def __init__(
        self,
        config_dir: Path,
        mlflow_experiment_name: str,
    ):
        """
        Initialize trial executor.

        Args:
            config_dir: Path to configuration directory.
            mlflow_experiment_name: MLflow experiment name.
        """
        # Derive project root from config_dir (config_dir is ROOT_DIR / "config")
        self.root_dir = config_dir.parent
        self.config_dir = config_dir
        self.mlflow_experiment_name = mlflow_experiment_name

    def execute(
        self,
        trial_params: Dict[str, Any],
        dataset_path: str,
        backbone: str,
        output_dir: Path,
        train_config: Dict[str, Any],
        objective_metric: str = "macro-f1",
        fold_idx: Optional[int] = None,
        fold_splits_file: Optional[Path] = None,
        parent_run_id: Optional[str] = None,
    ) -> float:
        """
        Execute training trial and return objective metric value.

        Args:
            trial_params: Hyperparameters for this trial.
            dataset_path: Path to dataset directory.
            backbone: Model backbone name.
            output_dir: Output directory for checkpoint.
            train_config: Training configuration dictionary.
            objective_metric: Name of the objective metric to optimize.
            fold_idx: Optional fold index for cross-validation.
            fold_splits_file: Optional path to fold splits file.
            parent_run_id: Optional parent MLflow run ID for nested runs.

        Returns:
            Objective metric value (e.g., macro-f1).
        """
        # Find project root (parent of config_dir)
        root_dir = find_project_root(self.config_dir)
        src_dir = root_dir / "src"

        # Build command using shared infrastructure
        training_options = TrainingOptions(
            fold_idx=fold_idx,
            epochs=train_config.get("training", {}).get("hpo_epochs", 1),
            early_stopping_enabled=False,  # Disable for HPO (consistent evaluation)
        )
        args = build_training_command(
            backbone=backbone,
            dataset_path=dataset_path,
            config_dir=self.config_dir,
            hyperparameters=trial_params,
            training_options=training_options,
        )

        # Set up environment using shared infrastructure
        mlflow_config = MLflowConfig(
            experiment_name=self.mlflow_experiment_name,
            parent_run_id=parent_run_id,
            trial_number=trial_params.get("trial_number") if trial_params else None,
        )
        fold_config = (
            FoldConfig(fold_idx=fold_idx, fold_splits_file=fold_splits_file)
            if fold_idx is not None
            else None
        )
        trial_config = TrialConfig(skip_artifact_logging=True)
        env = setup_training_environment(
            root_dir=root_dir,
            src_dir=src_dir,
            output_dir=output_dir,
            mlflow_config=mlflow_config,
            fold_config=fold_config,
            trial_config=trial_config,
        )

        # Verify environment before running
        verify_training_environment(root_dir, env, logger)

        # Run training subprocess using shared infrastructure
        result = execute_training_subprocess(
            command=args,
            cwd=root_dir,
            env=env,
            logger_instance=logger,
        )

        # Read metrics from output directory
        metrics = read_trial_metrics(
            trial_output_dir=output_dir,
            root_dir=self.root_dir,
            objective_metric=objective_metric,
            mlflow_experiment_name=self.mlflow_experiment_name,
        )

        if objective_metric not in metrics:
            raise ValueError(
                f"Objective metric '{objective_metric}' not found in metrics. "
                f"Available metrics: {list(metrics.keys())}"
            )

        metric_value = metrics[objective_metric]
        logger.info(
            f"[TRIAL] Training completed. Objective metric '{objective_metric}': {metric_value}"
        )

        return metric_value


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
    parent_run_id: Optional[str] = None,
) -> float:
    """
    Execute a single training trial with given hyperparameters.

    This is a convenience wrapper around TrialExecutor.

    Args:
        trial_params: Hyperparameters for this trial.
        dataset_path: Path to dataset directory.
        config_dir: Path to configuration directory.
        backbone: Model backbone name.
        output_dir: Output directory for checkpoint.
        train_config: Training configuration dictionary.
        mlflow_experiment_name: MLflow experiment name.
        objective_metric: Name of the objective metric to optimize.
        fold_idx: Optional fold index for cross-validation.
        fold_splits_file: Optional path to fold splits file.
        parent_run_id: Optional parent MLflow run ID for nested runs.

    Returns:
        Objective metric value (e.g., macro-f1).
    """
    executor = TrialExecutor(config_dir, mlflow_experiment_name)
    return executor.execute(
        trial_params=trial_params,
        dataset_path=dataset_path,
        backbone=backbone,
        output_dir=output_dir,
        train_config=train_config,
        objective_metric=objective_metric,
        fold_idx=fold_idx,
        fold_splits_file=fold_splits_file,
        parent_run_id=parent_run_id,
    )
