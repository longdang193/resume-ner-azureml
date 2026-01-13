from __future__ import annotations

"""
@meta
name: hpo_refit_execution
type: script
domain: hpo
responsibility:
  - Execute refit training on full dataset
  - Use best trial hyperparameters
  - Create canonical checkpoint for production
inputs:
  - Best trial
  - Training configuration
outputs:
  - Refit checkpoint
  - Refit metrics
tags:
  - execution
  - hpo
  - refit
ci:
  runnable: true
  needs_gpu: true
  needs_cloud: false
lifecycle:
  status: active
"""

"""Refit training executor for HPO.

Handles refit training on full dataset using best trial hyperparameters.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
from common.shared.logging_utils import get_logger
from common.constants import METRICS_FILENAME
from infrastructure.naming import create_naming_context
from infrastructure.tracking.mlflow.naming import (
    build_mlflow_run_name,
    build_mlflow_tags,
    build_mlflow_run_key,
    build_mlflow_run_key_hash,
)
# Tag key imports moved to local scope where needed
from infrastructure.paths import find_project_root
from training.execution import (
    MLflowConfig,
    TrainingOptions,
    TrialConfig,
    build_training_command,
    create_training_mlflow_run,
    execute_training_subprocess,
    setup_training_environment,
    verify_training_environment,
)
from common.shared.platform_detection import detect_platform

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

    # Follow SSOT pattern: Retrieve from tags first, then compute as fallback
    # Priority 1: Use provided study_key_hash (should have been retrieved by caller)
    # Priority 2: Retrieve study_key_hash from parent run tags (SSOT)
    computed_study_key_hash = study_key_hash
    if not computed_study_key_hash and hpo_parent_run_id:
        try:
            from infrastructure.tracking.mlflow.hash_utils import get_study_key_hash_from_run
            import mlflow
            client = mlflow.tracking.MlflowClient()
            computed_study_key_hash = get_study_key_hash_from_run(
                hpo_parent_run_id, client, config_dir
            )
            if computed_study_key_hash:
                logger.debug(
                    f"[REFIT] Retrieved study_key_hash={computed_study_key_hash[:16]}... "
                    f"from parent run tags (SSOT)"
                )
        except Exception as e:
            logger.debug(f"Could not retrieve study_key_hash from parent run: {e}")

    # Priority 1: Use provided trial_key_hash (should have been retrieved from trial run by caller - SSOT)
    # Priority 2: Compute trial_key_hash from trial parameters (fallback when trial run tags unavailable)
    computed_trial_key_hash = trial_key_hash
    if not computed_trial_key_hash and computed_study_key_hash and refit_params:
        try:
            from infrastructure.tracking.mlflow.hash_utils import compute_trial_key_hash_from_configs
            computed_trial_key_hash = compute_trial_key_hash_from_configs(
                computed_study_key_hash, refit_params, config_dir
            )
            if computed_trial_key_hash:
                logger.warning(
                    f"[REFIT] Computed trial_key_hash={computed_trial_key_hash[:16]}... "
                    f"from trial parameters (fallback - may not match trial run hash). "
                    f"Trial run tags should be used as SSOT."
                )
        except Exception as e:
            logger.debug(f"Could not compute trial_key_hash from trial parameters: {e}")

    # Create NamingContext and MLflow run for refit FIRST (needed for v2 path construction)
    # Include study_key_hash and trial_key_hash for hash-driven naming consistency
    refit_context = create_naming_context(
        process_type="hpo_refit",
        model=backbone.split("-")[0] if "-" in backbone else backbone,
        environment=detect_platform(),
        storage_env=detect_platform(),
        trial_id=trial_id,
        trial_number=trial_number,  # Add trial_number for readability
        study_key_hash=computed_study_key_hash,  # Use computed study_key_hash (from tags or configs)
        trial_key_hash=computed_trial_key_hash,  # Use computed trial_key_hash
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
            from infrastructure.paths import build_output_path
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
        is_v2_study_folder = study_folder_name and len(study_folder_name) > 7  # study-{hash} has at least 8 chars
        
        if is_v2_study_folder:
            if refit_context.trial_key_hash:
                # We're in a v2 study folder and have trial_key_hash, construct v2 trial path manually
                from infrastructure.naming.context_tokens import build_token_values
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
                # We're in a v2 study folder but don't have trial_key_hash
                # This is an error - we can't create v2 trial name without hash
                raise RuntimeError(
                    f"Cannot create refit in v2 study folder {study_folder_name} without trial_key_hash. "
                    f"study_key_hash={'present' if study_key_hash else 'missing'}, "
                    f"trial_key_hash=missing, refit_params={'present' if refit_params else 'missing'}. "
                    f"Hash computation from trial parameters may have failed."
                )
        else:
            # Should not happen - we only support v2 paths
            raise RuntimeError(
                f"Cannot create refit in non-v2 study folder. Only v2 paths (study-{{hash}}) are supported. "
                f"Found study folder: {study_folder_name or output_dir.name}"
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
    # Skip artifact logging during refit training - checkpoint will be uploaded as archive
    # by sweep_tracker.log_best_checkpoint() after refit completes
    trial_config = TrialConfig(skip_artifact_logging=True)
    env = setup_training_environment(
        root_dir=root_dir,
        src_dir=root_dir / "src",
        output_dir=refit_output_dir,
        mlflow_config=mlflow_config,
        trial_config=trial_config,
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
            study_key_hash=computed_study_key_hash,  # Use computed study_key_hash
        study_family_hash=study_family_hash,
            trial_key_hash=computed_trial_key_hash,  # Use computed trial_key_hash
        refit_protocol_fp=refit_protocol_fp,
        run_key_hash=refit_run_key_hash,
    )
    refit_tags["mlflow.runType"] = "refit"
    refit_tags["mlflow.runName"] = refit_run_name
    
    # Copy Phase 2 tags from parent run to refit run (for consistency)
    # This ensures refit runs have schema_version, fingerprints, etc. for champion selection
    if hpo_parent_run_id:
        try:
            from mlflow.tracking import MlflowClient
            from infrastructure.naming.mlflow.tags_registry import load_tags_registry
            client = MlflowClient()
            parent_run = client.get_run(hpo_parent_run_id)
            parent_tags = parent_run.data.tags
            
            # Phase 2 tags to copy from parent
            try:
                tags_registry = load_tags_registry(config_dir)
                schema_version_tag = tags_registry.key("study", "key_schema_version")
                data_fp_tag = tags_registry.key("fingerprint", "data")
                eval_fp_tag = tags_registry.key("fingerprint", "eval")
                direction_tag = tags_registry.key("objective", "direction")
                
                # Copy tags if they exist on parent
                if schema_version_tag in parent_tags:
                    refit_tags[schema_version_tag] = parent_tags[schema_version_tag]
                if data_fp_tag in parent_tags:
                    refit_tags[data_fp_tag] = parent_tags[data_fp_tag]
                if eval_fp_tag in parent_tags:
                    refit_tags[eval_fp_tag] = parent_tags[eval_fp_tag]
                if direction_tag in parent_tags:
                    refit_tags[direction_tag] = parent_tags[direction_tag]
            except Exception:
                # Fallback: use hardcoded tag names
                phase2_tag_keys = [
                    "code.study.key_schema_version",
                    "code.fingerprint.data",
                    "code.fingerprint.eval",
                    "code.objective.direction",
                ]
                for tag_key in phase2_tag_keys:
                    if tag_key in parent_tags:
                        refit_tags[tag_key] = parent_tags[tag_key]
        except Exception as e:
            logger.debug(f"Could not copy Phase 2 tags from parent run to refit run: {e}")
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
            
            # CRITICAL: Try to set linking tag immediately after run creation
            # (in case run gets finalized before _log_refit_metrics_to_mlflow is called)
            if refit_run_id and computed_trial_key_hash:
                try:
                    _link_refit_to_trial_run(
                        refit_run_id=refit_run_id,
                        trial_key_hash=computed_trial_key_hash,
                        hpo_parent_run_id=hpo_parent_run_id,
                        config_dir=config_dir,
                        trial_number=best_trial.number,
                    )
                except Exception as e:
                    logger.debug(
                        f"[REFIT] Could not set linking tag immediately after run creation: {e}. "
                        f"Will retry after training completes."
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
            trial_key_hash=computed_trial_key_hash,  # Use computed trial_key_hash for linking
            hpo_parent_run_id=hpo_parent_run_id,  # Pass for linking
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
    trial_key_hash: Optional[str] = None,
    hpo_parent_run_id: Optional[str] = None,
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
        from infrastructure.naming.mlflow.tag_keys import (
            get_refit,
            get_refit_has_validation,
        )
        from infrastructure.naming.mlflow.tags_registry import load_tags_registry
        
        tags_registry = load_tags_registry(config_dir)
        refit_tag = get_refit(config_dir)
        refit_has_validation_tag = get_refit_has_validation(config_dir)
        client.set_tag(refit_run_id, refit_tag, "true")
        client.set_tag(refit_run_id, refit_has_validation_tag, "false")
        
        # CRITICAL: Link refit run to trial run for deterministic mapping
        # This is called after training completes, but we also try immediately after run creation
        # Note: trial_number is not available here, so we pass None
        # (it was already tried immediately after run creation with trial_number)
        _link_refit_to_trial_run(
            refit_run_id=refit_run_id,
            trial_key_hash=trial_key_hash,
            hpo_parent_run_id=hpo_parent_run_id,
            config_dir=config_dir,
            trial_number=None,  # Not available in this context
        )

        # Log hyperparameters
        for param_name, param_value in refit_params.items():
            client.log_param(refit_run_id, param_name, str(param_value))

        logger.info(
            f"[REFIT] Logged metrics to MLflow (run will be marked FINISHED after artifacts are uploaded)"
        )
    except Exception as e:
        logger.warning(f"[REFIT] Could not log metrics to MLflow: {e}", exc_info=True)


def _link_refit_to_trial_run(
    refit_run_id: str,
    trial_key_hash: Optional[str],
    hpo_parent_run_id: Optional[str],
    config_dir: Path,
    trial_number: Optional[int] = None,
) -> None:
    """
    Link refit run to trial run by setting refit.of_trial_run_id tag.
    
    This function can be called multiple times safely (idempotent).
    """
    from common.shared.logging_utils import get_logger
    logger = get_logger(__name__)
    
    # Search for trial run that matches trial_key_hash
    trial_run_id = None
    logger.info(
        f"[REFIT] Attempting to link refit run {refit_run_id[:12]}... to trial run. "
        f"trial_key_hash={'present' if trial_key_hash else 'missing'}, "
        f"hpo_parent_run_id={'present' if hpo_parent_run_id else 'missing'}"
    )
    
    if trial_key_hash and hpo_parent_run_id:
        try:
            from mlflow.tracking import MlflowClient
            from infrastructure.naming.mlflow.tags_registry import load_tags_registry
            
            mlflow_client = MlflowClient()
            tags_registry = load_tags_registry(config_dir)
            trial_key_tag = tags_registry.key("grouping", "trial_key_hash")
            stage_tag = tags_registry.key("process", "stage")
                
            # Get experiment ID from parent run
            parent_run = mlflow_client.get_run(hpo_parent_run_id)
            experiment_id = parent_run.info.experiment_id
            logger.debug(
                f"[REFIT] Parent run experiment_id: {experiment_id}, "
                f"searching for trial runs with trial_key_hash={trial_key_hash[:16]}..."
            )
            
            # Strategy 1: Search for trial runs with matching trial_key_hash
            # Note: Azure ML MLflow backend doesn't support parentheses in filter strings
            # So we search for hpo_trial first, then hpo as fallback
            try:
                # Try hpo_trial first (preferred)
                filter_str_trial = f"tags.{trial_key_tag} = '{trial_key_hash}' AND tags.{stage_tag} = 'hpo_trial'"
                logger.debug(f"[REFIT] Search filter (hpo_trial): {filter_str_trial}")
                trial_runs = mlflow_client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=filter_str_trial,
                    max_results=10,
                )
                logger.debug(f"[REFIT] Found {len(trial_runs)} trial run(s) with stage='hpo_trial' matching trial_key_hash")
                
                # If no hpo_trial runs found, try hpo stage (legacy)
                if not trial_runs:
                    filter_str_hpo = f"tags.{trial_key_tag} = '{trial_key_hash}' AND tags.{stage_tag} = 'hpo'"
                    logger.debug(f"[REFIT] Search filter (hpo, fallback): {filter_str_hpo}")
                    trial_runs = mlflow_client.search_runs(
                        experiment_ids=[experiment_id],
                        filter_string=filter_str_hpo,
                        max_results=10,
                    )
                    logger.debug(f"[REFIT] Found {len(trial_runs)} trial run(s) with stage='hpo' matching trial_key_hash")
                
                # Find the trial run (prefer hpo_trial, fallback to hpo)
                for trial_run in trial_runs:
                    trial_stage = trial_run.data.tags.get(stage_tag)
                    trial_run_hash = trial_run.data.tags.get(trial_key_tag)
                    logger.debug(
                        f"[REFIT] Trial run {trial_run.info.run_id[:12]}...: "
                        f"stage={trial_stage}, trial_key_hash={trial_run_hash[:16] if trial_run_hash else 'missing'}..."
                    )
                    # Prefer hpo_trial, but accept hpo as fallback
                    if trial_stage in ("hpo_trial", "hpo"):
                        trial_run_id = trial_run.info.run_id
                        if trial_stage == "hpo_trial":
                            break  # Prefer hpo_trial
                
                if trial_run_id:
                    logger.info(
                        f"[REFIT] Found trial run {trial_run_id[:12]}... by trial_key_hash={trial_key_hash[:8]}..."
                    )
            except Exception as e:
                logger.warning(f"[REFIT] Search by trial_key_hash failed: {e}", exc_info=True)
            
            # Strategy 2: Fallback - search for child runs of HPO parent (trial runs are children)
            if not trial_run_id:
                try:
                    logger.debug(
                        f"[REFIT] Trying fallback: searching for child runs of parent {hpo_parent_run_id[:12]}..."
                    )
                    # Get all child runs of the HPO parent
                    # Note: Azure ML MLflow backend doesn't support parentheses, so search separately
                    # First try hpo_trial
                    child_runs = mlflow_client.search_runs(
                        experiment_ids=[experiment_id],
                        filter_string=f"tags.mlflow.parentRunId = '{hpo_parent_run_id}' AND tags.{stage_tag} = 'hpo_trial'",
                        max_results=100,
                    )
                    # If no hpo_trial runs, try hpo (legacy)
                    if not child_runs:
                        child_runs = mlflow_client.search_runs(
                            experiment_ids=[experiment_id],
                            filter_string=f"tags.mlflow.parentRunId = '{hpo_parent_run_id}' AND tags.{stage_tag} = 'hpo'",
                            max_results=100,
                        )
                    logger.debug(f"[REFIT] Found {len(child_runs)} child run(s) of HPO parent")
                    
                    # Filter by matching trial_key_hash first
                    for child_run in child_runs:
                        child_trial_hash = child_run.data.tags.get(trial_key_tag)
                        child_stage = child_run.data.tags.get(stage_tag)
                        logger.debug(
                            f"[REFIT] Child run {child_run.info.run_id[:12]}...: "
                            f"stage={child_stage}, trial_key_hash={child_trial_hash[:16] if child_trial_hash else 'missing'}..."
                        )
                        if child_trial_hash == trial_key_hash:
                            trial_run_id = child_run.info.run_id
                            logger.info(
                                f"[REFIT] Found trial run {trial_run_id[:12]}... by parent-child relationship "
                                f"(trial_key_hash={trial_key_hash[:8]}...)"
                            )
                            break
                    
                    # Strategy 3: If still not found and we have trial_number, search by trial number
                    # (Trial runs might not have trial_key_hash tag set, but should have trial number)
                    if not trial_run_id and trial_number is not None:
                        logger.debug(
                            f"[REFIT] Trying fallback by trial number: searching for trial {trial_number} "
                            f"among {len(child_runs)} child runs..."
                        )
                        from infrastructure.naming.mlflow.tag_keys import get_trial_number
                        trial_number_tag = get_trial_number(config_dir)
                        
                        for child_run in child_runs:
                            child_trial_num = child_run.data.tags.get(trial_number_tag)
                            child_stage = child_run.data.tags.get(stage_tag)
                            # Check if this is a trial run (not a fold run) with matching trial number
                            # Trial runs typically don't have "fold" in their name/tags
                            run_name = child_run.data.tags.get("mlflow.runName", "")
                            is_fold_run = "fold" in run_name.lower()
                            
                            logger.debug(
                                f"[REFIT] Child run {child_run.info.run_id[:12]}...: "
                                f"stage={child_stage}, trial_number={child_trial_num}, "
                                f"is_fold_run={is_fold_run}, name={run_name[:50]}..."
                            )
                            
                            # Match by trial number and ensure it's not a fold run
                            # Trial runs have naming pattern like: ..._t00_... or ..._trial_0_...
                            trial_number_in_name = f"_t{trial_number:02d}_" in run_name or f"_trial_{trial_number}_" in run_name
                            
                            if (child_trial_num == str(trial_number) or 
                                trial_number_in_name or
                                (child_trial_num is None and not is_fold_run and trial_number == 0)):
                                # Additional check: trial runs are usually parents of fold runs
                                # or have specific naming patterns (trial runs don't have "fold" in name)
                                if not is_fold_run and child_stage in ("hpo_trial", "hpo"):
                                    trial_run_id = child_run.info.run_id
                                    logger.info(
                                        f"[REFIT] Found trial run {trial_run_id[:12]}... by trial number {trial_number} "
                                        f"(name: {run_name[:50]}..., matched by: "
                                        f"{'tag' if child_trial_num == str(trial_number) else 'name pattern'})"
                                    )
                                    break
                except Exception as e:
                    logger.warning(f"[REFIT] Fallback search by parent-child relationship failed: {e}", exc_info=True)
            
            # Set linking tag if we found a trial run
            if trial_run_id:
                try:
                    refit_of_trial_tag = tags_registry.key("refit", "of_trial_run_id")
                    logger.info(
                        f"[REFIT] Setting tag {refit_of_trial_tag} = {trial_run_id} on refit run {refit_run_id[:12]}..."
                    )
                    mlflow_client.set_tag(refit_run_id, refit_of_trial_tag, trial_run_id)
                    
                    # Verify tag was set
                    verify_run = mlflow_client.get_run(refit_run_id)
                    verify_tag = verify_run.data.tags.get(refit_of_trial_tag)
                    if verify_tag == trial_run_id:
                        logger.info(
                            f"[REFIT] ✓ Successfully linked refit run {refit_run_id[:12]}... to trial run {trial_run_id[:12]}... "
                            f"(trial_key_hash={trial_key_hash[:8]}...)"
                        )
                    else:
                        logger.error(
                            f"[REFIT] ❌ Tag verification failed! Expected {trial_run_id}, got {verify_tag}. "
                            f"Tag may not have been set correctly. Run may be finalized."
                        )
                except Exception as e:
                    logger.error(
                        f"[REFIT] ❌ Failed to set linking tag: {e}",
                        exc_info=True
                    )
            else:
                logger.warning(
                    f"[REFIT] ⚠ Could not find trial run for trial_key_hash={trial_key_hash[:8]}... "
                    f"- refit run will not be linked to trial run. "
                    f"Checkpoint acquisition may fail or use fallback search."
                )
        except Exception as e:
            logger.error(
                f"[REFIT] ❌ Error linking refit run to trial run: {e}. "
                f"This may affect checkpoint acquisition later.",
                exc_info=True
            )
    elif not trial_key_hash:
        logger.warning(
            f"[REFIT] ⚠ trial_key_hash not available - cannot link refit run to trial run"
        )
    elif not hpo_parent_run_id:
        logger.warning(
            f"[REFIT] ⚠ hpo_parent_run_id not available - cannot link refit run to trial run"
        )

