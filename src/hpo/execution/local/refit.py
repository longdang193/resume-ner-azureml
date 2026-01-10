"""Refit training executor for HPO.

Handles refit training on full dataset using best trial hyperparameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
from shared.logging_utils import get_logger
from constants import METRICS_FILENAME
from naming import create_naming_context
from tracking.mlflow.naming import (
    build_mlflow_run_name,
    build_mlflow_tags,
    build_mlflow_run_key,
    build_mlflow_run_key_hash,
)
# Tag key imports moved to local scope where needed
from paths import find_project_root
from training.execution import (
    MLflowConfig,
    TrainingOptions,
    build_training_command,
    create_training_mlflow_run,
    execute_training_subprocess,
    setup_training_environment,
    verify_training_environment,
)
from shared.platform_detection import detect_platform

logger = get_logger(__name__)


def run_refit_training(
    best_trial: Any,
    dataset_path: str,
    config_dir: Path,
    backbone: str,
    output_dir: Path,
    train_config: Dict[str, Any],
    mlflow_experiment_name: str,
    objective_metric: str,
    hpo_parent_run_id: Optional[str] = None,
    study_key_hash: Optional[str] = None,
    study_family_hash: Optional[str] = None,
    trial_key_hash: Optional[str] = None,
    refit_protocol_fp: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[Dict[str, float], Path, Optional[str]]:
    """
    Run refit training on full training dataset using best trial hyperparameters.

    This creates a canonical checkpoint for production use, trained on the full
    training set (no validation split).

    Args:
        best_trial: Optuna trial object for the best trial.
        dataset_path: Path to dataset directory.
        config_dir: Path to configuration directory.
        backbone: Model backbone name.
        output_dir: Base output directory (trial directory will be created here).
        train_config: Training configuration dictionary.
        mlflow_experiment_name: MLflow experiment name.
        objective_metric: Name of the objective metric.
        hpo_parent_run_id: Optional HPO parent run ID (for creating refit as child run).
        study_key_hash: Optional study key hash (for grouping tags).
        trial_key_hash: Optional trial key hash (for grouping tags).
        refit_protocol_fp: Optional refit protocol fingerprint.
        run_id: Optional run ID for directory naming.

    Returns:
        Tuple of (metrics_dict, checkpoint_dir, refit_run_id) where:
        - metrics_dict: Dictionary of metrics from refit training
        - checkpoint_dir: Path to refit checkpoint directory
        - refit_run_id: MLflow run ID for refit run (if created)
    """
    logger.info(
        f"[REFIT] Starting refit training for trial {best_trial.number} "
        f"with hyperparameters: {best_trial.params}"
    )

    # Extract hyperparameters from best trial
    refit_params = {
        k: v for k, v in best_trial.params.items()
        if k not in ("backbone", "trial_number", "run_id")
    }

    # Build trial ID (same as the trial that was refit)
    trial_number = best_trial.number
    run_suffix = f"_{run_id}" if run_id else ""
    trial_id = f"trial_{trial_number}{run_suffix}"

    # Layer A: Ensure trial_id is never None/empty/whitespace
    if not trial_id or not trial_id.strip():
        trial_id = f"trial_{int(trial_number)}"
        logger.warning(
            f"[REFIT] trial_id was empty/None, auto-filled to: {trial_id!r}"
        )

    logger.info(
        f"[REFIT] Computed trial_id={trial_id!r}, run_id={run_id!r}, "
        f"trial_number={trial_number}"
    )

    # Derive project root from config_dir (needed for v2 path construction)
    root_dir = find_project_root(config_dir)

    # Create NamingContext and MLflow run for refit FIRST (needed for v2 path construction)
    # Include study_key_hash and trial_key_hash for hash-driven naming consistency
    refit_context = create_naming_context(
        process_type="hpo_refit",
        model=backbone.split("-")[0] if "-" in backbone else backbone,
        environment=detect_platform(),
        storage_env=detect_platform(),
        trial_id=trial_id,
        trial_number=trial_number,  # Add trial_number for readability
        study_key_hash=study_key_hash,  # Add study_key_hash for grouping
        trial_key_hash=trial_key_hash,  # Add trial_key_hash for grouping
    )

    # Assert: ensure trial_id is present before creating MLflow run
    assert refit_context.trial_id and refit_context.trial_id.strip(), (
        f"Refit context missing trial_id; would become *_unknown. "
        f"Computed trial_id={trial_id!r}, context.trial_id={refit_context.trial_id!r}"
    )

    # Create refit output directory using v2 pattern if hashes available
    # IMPORTANT: Do this BEFORE creating legacy directories to prevent legacy folder creation in v2 study folders
    refit_output_dir = None
    if refit_context.study_key_hash and refit_context.trial_key_hash:
        try:
            from paths import build_output_path
            # build_output_path() handles hpo_refit by appending /refit to trial path
            refit_output_dir = build_output_path(root_dir, refit_context, config_dir=config_dir)
            refit_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not construct v2 refit folder, falling back to legacy: {e}")
            refit_output_dir = None
    
    # Fallback to legacy pattern if v2 construction failed or hashes unavailable
    if refit_output_dir is None:
        # Check if we're in a v2 study folder (study-{hash})
        # If so, we need to find the v2 trial folder first, then append /refit
        study_folder_name = output_dir.name if output_dir.name.startswith("study-") else None
        if study_folder_name and refit_context.trial_key_hash:
            # We're in a v2 study folder, construct v2 trial path manually
            from naming.context_tokens import build_token_values
            tokens = build_token_values(refit_context)
            trial8 = tokens["trial8"]
            trial_base_dir = output_dir / f"trial-{trial8}"
            refit_output_dir = trial_base_dir / "refit"
            refit_output_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"build_output_path() failed but we're in v2 study folder. "
                f"Constructed v2 refit folder manually: {refit_output_dir}"
            )
        else:
            # Should not happen - we only support v2 paths
            raise RuntimeError(
                f"Cannot create refit in non-v2 study folder. Only v2 paths (study-{{hash}}) are supported. "
                f"Found study folder: {study_folder_name}"
            )

    # Build command arguments for refit training using shared infrastructure
    training_options = TrainingOptions(
        epochs=train_config.get("training", {}).get("epochs", 10),
        early_stopping_enabled=True,  # Enable for refit
        use_all_data=True,  # Refit uses full training set, no validation split
    )
    args = build_training_command(
        backbone=backbone,
        dataset_path=dataset_path,
        config_dir=config_dir,
        hyperparameters=refit_params,
        training_options=training_options,
    )

    # Set environment variables using shared infrastructure
    # Note: run_id will be set after MLflow run creation
    mlflow_config = MLflowConfig(
        experiment_name=mlflow_experiment_name,
        parent_run_id=hpo_parent_run_id,
    )
    env = setup_training_environment(
        root_dir=root_dir,
        src_dir=root_dir / "src",
        output_dir=refit_output_dir,
        mlflow_config=mlflow_config,
    )

    # Build MLflow run name and tags
    refit_run_name = build_mlflow_run_name(
        refit_context,
        config_dir=config_dir,
        root_dir=root_dir,
        output_dir=refit_output_dir,
    )

    refit_run_key = build_mlflow_run_key(
        refit_context) if refit_context else None
    refit_run_key_hash = build_mlflow_run_key_hash(
        refit_run_key) if refit_run_key else None
    refit_tags = build_mlflow_tags(
        context=refit_context,
        output_dir=refit_output_dir,
        parent_run_id=hpo_parent_run_id,
        config_dir=config_dir,
        study_key_hash=study_key_hash,
        study_family_hash=study_family_hash,
        trial_key_hash=trial_key_hash,
        refit_protocol_fp=refit_protocol_fp,
        run_key_hash=refit_run_key_hash,
    )
    refit_tags["mlflow.runType"] = "refit"
    refit_tags["mlflow.runName"] = refit_run_name
    if hpo_parent_run_id:
        refit_tags["mlflow.parentRunId"] = hpo_parent_run_id
        refit_tags["azureml.runType"] = "refit"

    # Create refit run as child of HPO parent using shared infrastructure
    refit_run_id = None
    if hpo_parent_run_id:
        try:
            refit_run_id, _ = create_training_mlflow_run(
                experiment_name=mlflow_experiment_name,
                run_name=refit_run_name,
                tags=refit_tags,
                parent_run_id=hpo_parent_run_id,
            )
        except Exception as e:
            logger.warning(f"Could not create refit MLflow run: {e}", exc_info=True)

    # Set run ID for refit training subprocess
    if refit_run_id:
        env["MLFLOW_RUN_ID"] = refit_run_id
        # CRITICAL: Also set MLFLOW_PARENT_RUN_ID so training script knows it's refit mode
        # This prevents training script from auto-ending the run (parent will terminate it)
        if hpo_parent_run_id:
            env["MLFLOW_PARENT_RUN_ID"] = hpo_parent_run_id
    elif hpo_parent_run_id:
        env["MLFLOW_PARENT_RUN_ID"] = hpo_parent_run_id
        env["MLFLOW_TRIAL_NUMBER"] = "refit"
        logger.warning(
            f"[REFIT] Refit run not created, using HPO parent as fallback. "
            f"This may create an unwanted child run."
        )

    # Verify environment before running using shared infrastructure
    verify_training_environment(root_dir, env, logger)

    # Run refit training using shared infrastructure
    result = execute_training_subprocess(
        command=args,
        cwd=root_dir,
        env=env,
        logger_instance=logger,
    )

    # Read metrics from metrics.json
    metrics = _read_refit_metrics(refit_output_dir)

    # Log metrics to MLflow refit run
    if refit_run_id:
        _log_refit_metrics_to_mlflow(
            refit_run_id=refit_run_id,
            metrics=metrics,
            refit_params=refit_params,
            config_dir=config_dir,
        )

    checkpoint_dir = refit_output_dir / "checkpoint"
    logger.info(
        f"[REFIT] Refit training completed. Metrics: {metrics.get(objective_metric, 'N/A')}, "
        f"Checkpoint: {checkpoint_dir}"
    )

    return metrics, checkpoint_dir, refit_run_id


def _read_refit_metrics(refit_output_dir: Path) -> Dict[str, float]:
    """Read metrics from refit output directory."""
    metrics_file = refit_output_dir / METRICS_FILENAME
    if not metrics_file.exists():
        logger.warning(f"[REFIT] Metrics file not found at {metrics_file}")
        return {}

    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[REFIT] Could not read metrics file: {e}")
        return {}


def _log_refit_metrics_to_mlflow(
    refit_run_id: str,
    metrics: Dict[str, Any],
    refit_params: Dict[str, Any],
    config_dir: Path,
) -> None:
    """Log metrics and parameters to MLflow refit run."""
    try:
        client = mlflow.tracking.MlflowClient()

        # Split metrics into numeric (for log_metric) and string notes (for tags)
        numeric_metrics = {}
        string_notes = {}

        for k, v in metrics.items():
            if isinstance(v, bool):
                numeric_metrics[k] = float(int(v))
            elif isinstance(v, (int, float)):
                numeric_metrics[k] = float(v)
            else:
                string_notes[k] = str(v)

        # Log numeric metrics
        for k, v in numeric_metrics.items():
            client.log_metric(refit_run_id, k, v)

        # Log string notes as tags
        for k, v in string_notes.items():
            client.set_tag(refit_run_id, f"note.{k}", v)

        # Set explicit refit tags
        from naming.mlflow.tag_keys import (
            get_refit,
            get_refit_has_validation,
        )
        refit_tag = get_refit(config_dir)
        refit_has_validation_tag = get_refit_has_validation(config_dir)
        client.set_tag(refit_run_id, refit_tag, "true")
        client.set_tag(refit_run_id, refit_has_validation_tag, "false")

        # Log hyperparameters
        for param_name, param_value in refit_params.items():
            client.log_param(refit_run_id, param_name, str(param_value))

        logger.info(
            f"[REFIT] Logged metrics to MLflow (run will be marked FINISHED after artifacts are uploaded)"
        )
    except Exception as e:
        logger.warning(f"[REFIT] Could not log metrics to MLflow: {e}")

