"""Subprocess execution infrastructure for training runs.

This module provides unified functions for building training commands,
setting up execution environments, and running training subprocesses.
Used by HPO trials, refit training, and final training execution.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.shared.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingOptions:
    """Options for training execution."""

    fold_idx: Optional[int] = None
    epochs: Optional[int] = None
    early_stopping_enabled: bool = False
    use_combined_data: bool = True
    use_all_data: bool = False
    random_seed: Optional[int] = None


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""

    experiment_name: str
    tracking_uri: Optional[str] = None
    parent_run_id: Optional[str] = None
    run_id: Optional[str] = None
    trial_number: Optional[int] = None


@dataclass
class FoldConfig:
    """Cross-validation fold configuration."""

    fold_idx: int
    fold_splits_file: Optional[Path] = None


@dataclass
class TrialConfig:
    """HPO trial-specific configuration."""

    skip_artifact_logging: bool = False


def build_training_command(
    backbone: str,
    dataset_path: str | Path,
    config_dir: Path,
    hyperparameters: Dict[str, Any],
    training_options: Optional[TrainingOptions] = None,
) -> List[str]:
    """
    Build command arguments for training subprocess.

    Args:
        backbone: Model backbone name
        dataset_path: Path to dataset directory
        config_dir: Configuration directory
        hyperparameters: Dict with learning_rate, batch_size, dropout, weight_decay
        training_options: Optional TrainingOptions object with:
            - fold_idx: Fold index for cross-validation
            - epochs: Number of epochs (defaults from config)
            - early_stopping_enabled: Enable/disable early stopping
            - use_combined_data: Use combined dataset
            - use_all_data: Use all data without validation split
            - random_seed: Random seed

    Returns:
        List of command arguments for subprocess

    Examples:
        # HPO trial usage:
        options = TrainingOptions(
            fold_idx=0,
            epochs=1,
            early_stopping_enabled=False
        )
        cmd = build_training_command(
            backbone="distilbert",
            dataset_path="/data",
            config_dir=Path("config"),
            hyperparameters={"learning_rate": 2e-5, "batch_size": 16},
            training_options=options
        )

        # Refit usage:
        options = TrainingOptions(
            epochs=10,
            early_stopping_enabled=True,
            use_all_data=True
        )
        cmd = build_training_command(...)

        # Final training usage:
        options = TrainingOptions(
            epochs=5,
            early_stopping_enabled=False,
            use_combined_data=True,
            random_seed=42
        )
        cmd = build_training_command(...)
    """
    args = [
        sys.executable,
        "-m",
        "training.train",
        "--data-asset",
        str(dataset_path),
        "--config-dir",
        str(config_dir),
        "--backbone",
        backbone,
    ]

    # Add hyperparameters
    if "learning_rate" in hyperparameters:
        args.extend(["--learning-rate", str(hyperparameters["learning_rate"])])
    if "batch_size" in hyperparameters:
        args.extend(["--batch-size", str(hyperparameters["batch_size"])])
    if "dropout" in hyperparameters:
        args.extend(["--dropout", str(hyperparameters["dropout"])])
    if "weight_decay" in hyperparameters:
        args.extend(["--weight-decay", str(hyperparameters["weight_decay"])])

    # Add training options
    if training_options:
        if training_options.fold_idx is not None:
            args.extend(["--fold-idx", str(training_options.fold_idx)])

        if training_options.epochs is not None:
            args.extend(["--epochs", str(training_options.epochs)])

        args.extend(
            [
                "--early-stopping-enabled",
                str(training_options.early_stopping_enabled).lower(),
            ]
        )

        if training_options.use_all_data:
            args.extend(["--use-all-data", "true"])

        if training_options.use_combined_data is not None:
            args.extend(
                ["--use-combined-data", str(training_options.use_combined_data).lower()]
            )

        if training_options.random_seed is not None:
            args.extend(["--random-seed", str(training_options.random_seed)])

    return args


def setup_training_environment(
    root_dir: Path,
    src_dir: Path,
    output_dir: Path,
    mlflow_config: MLflowConfig,
    fold_config: Optional[FoldConfig] = None,
    trial_config: Optional[TrialConfig] = None,
) -> Dict[str, str]:
    """
    Set up environment variables for training subprocess.

    Args:
        root_dir: Project root directory
        src_dir: Source directory (root_dir / "src")
        output_dir: Output directory for checkpoints
        mlflow_config: MLflowConfig with experiment_name, tracking_uri, etc.
        fold_config: Optional FoldConfig with fold_idx and fold_splits_file
        trial_config: Optional TrialConfig with skip_artifact_logging

    Returns:
        Dictionary of environment variables
    """
    import mlflow

    env = os.environ.copy()

    # Set PYTHONPATH to include src directory
    # Handle both prepending (trial) and appending (refit/final) patterns
    src_dir_str = str(src_dir.resolve())
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        # Append to existing PYTHONPATH (refit/final pattern)
        env["PYTHONPATH"] = f"{src_dir_str}{os.pathsep}{current_pythonpath}"
    else:
        # Set as new PYTHONPATH (trial pattern)
        env["PYTHONPATH"] = src_dir_str

    # Set output directory for checkpoint saving
    env["AZURE_ML_OUTPUT_CHECKPOINT"] = str(output_dir)
    env["AZURE_ML_OUTPUT_checkpoint"] = str(output_dir)

    # Set MLflow tracking
    mlflow.set_experiment(mlflow_config.experiment_name)
    tracking_uri = mlflow_config.tracking_uri or mlflow.get_tracking_uri()
    if tracking_uri:
        env["MLFLOW_TRACKING_URI"] = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)

        # Set Azure ML artifact upload timeout if using Azure ML
        if "azureml" in tracking_uri.lower():
            env["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "600"

    env["MLFLOW_EXPERIMENT_NAME"] = mlflow_config.experiment_name

    # Set parent run ID and trial number for nested runs
    if mlflow_config.parent_run_id:
        env["MLFLOW_PARENT_RUN_ID"] = mlflow_config.parent_run_id

    if mlflow_config.run_id:
        env["MLFLOW_RUN_ID"] = mlflow_config.run_id

    if mlflow_config.trial_number is not None:
        env["MLFLOW_TRIAL_NUMBER"] = str(mlflow_config.trial_number)

    # Set trial-specific configs
    if trial_config and trial_config.skip_artifact_logging:
        env["MLFLOW_SKIP_ARTIFACT_LOGGING"] = "true"

    # Set fold configs if k-fold CV is enabled
    if fold_config:
        env["MLFLOW_FOLD_IDX"] = str(fold_config.fold_idx)
        if fold_config.fold_splits_file:
            env["MLFLOW_FOLD_SPLITS_FILE"] = str(fold_config.fold_splits_file)

    return env


def verify_training_environment(
    root_dir: Path, env: Dict[str, str], logger_instance: Optional[Any] = None
) -> None:
    """
    Verify environment is set up correctly for training subprocess.

    Args:
        root_dir: Project root directory
        env: Environment variables dictionary
        logger_instance: Optional logger instance (defaults to module logger)

    Raises:
        RuntimeError: If validation fails
    """
    log = logger_instance or logger
    src_dir = root_dir / "src"
    src_dir_str = str(src_dir.resolve())
    training_module_path = root_dir / "src" / "training" / "__init__.py"

    if not training_module_path.exists():
        raise RuntimeError(
            f"Training module not found at {training_module_path}. "
            f"Root dir: {root_dir}, src_dir: {src_dir_str}"
        )

    pythonpath_value = env.get("PYTHONPATH", "")
    if not pythonpath_value:
        raise RuntimeError(
            "PYTHONPATH is not set in environment! "
            "This will prevent the subprocess from finding the training module."
        )
    if src_dir_str not in pythonpath_value:
        raise RuntimeError(
            f"src_dir ({src_dir_str}) is not in PYTHONPATH ({pythonpath_value})! "
            "This will prevent the subprocess from finding the training module."
        )


def execute_training_subprocess(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    capture_output: bool = True,
    text: bool = True,
    logger_instance: Optional[Any] = None,
) -> subprocess.CompletedProcess:
    """
    Execute training subprocess with error handling.

    Args:
        command: Command arguments
        cwd: Working directory
        env: Environment variables
        capture_output: Capture stdout/stderr
        text: Return text output
        logger_instance: Optional logger instance (defaults to module logger)

    Returns:
        CompletedProcess result

    Raises:
        RuntimeError: If subprocess fails
    """
    log = logger_instance or logger

    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=capture_output,
        text=text,
    )

    # Check if training succeeded
    if result.returncode != 0:
        log.error(f"Training failed with return code {result.returncode}")
        log.error(f"STDOUT:\n{result.stdout}")
        log.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError(
            f"Training failed with return code {result.returncode}\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    # Filter out verbose debug messages from subprocess output
    if result.stdout:
        filtered_stdout = _filter_debug_messages(result.stdout)
        if filtered_stdout.strip():
            log.info(filtered_stdout)

    if result.stderr:
        filtered_stderr = _filter_debug_messages(result.stderr)
        if filtered_stderr.strip():
            log.warning(filtered_stderr)

    return result


def _filter_debug_messages(output: str) -> str:
    """
    Filter out verbose debug messages from subprocess output.

    Removes "Attempted to log scalar metric" debug messages and their values.

    Args:
        output: Subprocess output string

    Returns:
        Filtered output string
    """
    lines = output.split("\n")
    filtered_lines = []
    skip_next = False

    for line in lines:
        if line.strip().startswith("Attempted to log scalar metric"):
            skip_next = True  # Skip the value line that follows
            continue
        if skip_next and line.strip() and not line.strip().startswith("["):
            # Skip the value line (unless it's a log message starting with [)
            skip_next = False
            continue
        skip_next = False
        filtered_lines.append(line)

    return "\n".join(filtered_lines)

