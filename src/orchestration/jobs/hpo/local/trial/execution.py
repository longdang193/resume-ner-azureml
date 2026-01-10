from __future__ import annotations

"""
@meta
name: hpo_trial_execution_legacy
type: script
domain: hpo
responsibility:
  - Execute HPO training trials (legacy wrapper)
  - Handle subprocess execution
  - Set up environment variables
  - Read trial metrics
inputs:
  - Trial hyperparameters
  - Training configuration
outputs:
  - Trial metrics
tags:
  - execution
  - hpo
  - legacy
ci:
  runnable: true
  needs_gpu: true
  needs_cloud: false
lifecycle:
  status: active
"""

"""Trial execution for HPO training runs.

Handles subprocess execution, environment setup, and metrics reading.
Combines TrialExecutor class and run_training_trial function.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from common.shared.logging_utils import get_logger

# Lazy import to avoid circular dependency
def _get_read_trial_metrics():
    """Lazy import of read_trial_metrics to avoid circular dependencies."""
    from .metrics import read_trial_metrics
    return read_trial_metrics

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
        root_dir = self._find_project_root(self.config_dir)
        src_dir = root_dir / "src"

        # Build command
        args = self._build_command(
            trial_params=trial_params,
            dataset_path=dataset_path,
            backbone=backbone,
            train_config=train_config,
            fold_idx=fold_idx,
        )

        # Set up environment
        env = self._setup_environment(
            output_dir=output_dir,
            root_dir=root_dir,
            src_dir=src_dir,
            mlflow_experiment_name=self.mlflow_experiment_name,
            fold_idx=fold_idx,
            fold_splits_file=fold_splits_file,
            parent_run_id=parent_run_id,
            trial_params=trial_params,
        )

        # Run training subprocess

        # Verify the training module can be found
        training_module_path = root_dir / "src" / "training" / "__init__.py"
        if not training_module_path.exists():
            raise RuntimeError(
                f"[TRIAL] Training module not found at {training_module_path}. "
                f"Root dir: {root_dir}, src_dir: {src_dir}"
            )

        # Verify PYTHONPATH is set
        pythonpath_value = env.get("PYTHONPATH", "")
        if not pythonpath_value:
            raise RuntimeError(
                f"[TRIAL] PYTHONPATH not set in environment. "
                f"Expected: {src_dir}"
            )

        result = subprocess.run(
            args,
            cwd=root_dir,
            env=env,
            capture_output=True,
            text=True,
        )

        # Check if training succeeded
        if result.returncode != 0:
            logger.error(
                f"[TRIAL] Training failed with return code {result.returncode}")
            logger.error(f"[TRIAL] STDOUT:\n{result.stdout}")
            logger.error(f"[TRIAL] STDERR:\n{result.stderr}")
            raise RuntimeError(f"Training failed: {result.stderr}")

        # Read metrics from output directory
        read_trial_metrics = _get_read_trial_metrics()
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

    def _find_project_root(self, config_dir: Path) -> Path:
        """Find project root directory by walking up from config_dir."""
        candidate_root = config_dir.parent
        max_depth = 5
        depth = 0
        while depth < max_depth:
            if (candidate_root / "src").exists() and (candidate_root / "src" / "training").exists():
                return candidate_root
            candidate_root = candidate_root.parent
            depth += 1

        # Fallback: assume config_dir is project_root/config
        root_dir = config_dir.parent
        logger.warning(
            f"[TRIAL] Could not find project root with src/training/ directory after {max_depth} levels. "
            f"Using {root_dir} as root_dir. Config_dir: {config_dir}"
        )
        return root_dir

    def _build_command(
        self,
        trial_params: Dict[str, Any],
        dataset_path: str,
        backbone: str,
        train_config: Dict[str, Any],
        fold_idx: Optional[int] = None,
    ) -> list[str]:
        """Build command arguments for training."""
        args = [
            sys.executable,
            "-m",
            "training.train",
            "--data-asset",
            dataset_path,
            "--config-dir",
            str(self.config_dir),
            "--backbone",
            backbone,
        ]

        # Add hyperparameters
        if "learning_rate" in trial_params:
            args.extend(["--learning-rate", str(trial_params["learning_rate"])])
        if "batch_size" in trial_params:
            args.extend(["--batch-size", str(trial_params["batch_size"])])
        if "dropout" in trial_params:
            args.extend(["--dropout", str(trial_params["dropout"])])
        if "weight_decay" in trial_params:
            args.extend(["--weight-decay", str(trial_params["weight_decay"])])

        # Add fold index if k-fold CV is enabled
        if fold_idx is not None:
            args.extend(["--fold-idx", str(fold_idx)])

        # Use minimal epochs for HPO (speed optimization)
        epochs = train_config.get("training", {}).get("hpo_epochs", 1)
        args.extend(["--epochs", str(epochs)])

        # Disable early stopping for HPO (we want consistent evaluation)
        args.extend(["--early-stopping-enabled", "false"])

        return args

    def _setup_environment(
        self,
        output_dir: Path,
        root_dir: Path,
        src_dir: Path,
        mlflow_experiment_name: str,
        fold_idx: Optional[int] = None,
        fold_splits_file: Optional[Path] = None,
        parent_run_id: Optional[str] = None,
        trial_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Set up environment variables for training subprocess."""
        env = os.environ.copy()

        # Set PYTHONPATH to include src directory
        env["PYTHONPATH"] = str(src_dir)

        # Set output directory for checkpoint saving
        env["AZURE_ML_OUTPUT_CHECKPOINT"] = str(output_dir)
        env["AZURE_ML_OUTPUT_checkpoint"] = str(output_dir)

        # Set MLflow tracking
        mlflow.set_experiment(mlflow_experiment_name)
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri:
            env["MLFLOW_TRACKING_URI"] = tracking_uri
        env["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment_name

        # Set parent run ID and trial number for nested runs
        if parent_run_id:
            env["MLFLOW_PARENT_RUN_ID"] = parent_run_id

        # Set trial number for proper MLflow run naming
        if trial_params:
            trial_number = trial_params.get("trial_number")
            if trial_number is not None:
                env["MLFLOW_TRIAL_NUMBER"] = str(trial_number)

        # Disable automatic artifact logging during HPO trials
        # Only the refit checkpoint of the best trial will be logged via log_best_checkpoint
        # This prevents logging all fold checkpoints of all trials (saves storage and time)
        env["MLFLOW_SKIP_ARTIFACT_LOGGING"] = "true"

        # Set fold index if k-fold CV is enabled
        if fold_idx is not None:
            env["MLFLOW_FOLD_IDX"] = str(fold_idx)
            if fold_splits_file:
                env["MLFLOW_FOLD_SPLITS_FILE"] = str(fold_splits_file)

        return env

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
