from __future__ import annotations

"""
@meta
name: tracking_mlflow_sweep_tracker
type: utility
domain: tracking
responsibility:
  - Track MLflow runs for HPO sweep stage
  - Manage sweep and trial run lifecycle
inputs:
  - Sweep configurations
  - Run names and contexts
outputs:
  - MLflow run handles
tags:
  - utility
  - tracking
  - mlflow
  - tracker
  - hpo
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""MLflow tracker for sweep stage."""
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, List
import time
import tempfile
import json
import sys
import os
import re

import mlflow
# Import tracking.mlflow to ensure Azure ML compatibility patch is applied
# Use try/except to handle cases where path isn't set up during pytest collection
try:
    from infrastructure.tracking.mlflow import (  # noqa: F401
        get_mlflow_run_url,
        upload_checkpoint_archive,
        log_artifact_safe,
    )
except ImportError:
    # During pytest collection, path might not be set up yet
    # These will be imported lazily when actually needed
    pass

from common.shared.logging_utils import get_logger

from infrastructure.tracking.mlflow.types import RunHandle
from infrastructure.tracking.mlflow.naming import build_mlflow_tags, build_mlflow_run_key, build_mlflow_run_key_hash
from infrastructure.tracking.mlflow.index import update_mlflow_index
from infrastructure.tracking.mlflow.artifacts.uploader import ArtifactUploader
from infrastructure.tracking.mlflow.trackers.base_tracker import BaseTracker
from infrastructure.tracking.mlflow.utils import infer_config_dir_from_path
# Tag key imports moved to local scope where needed

logger = get_logger(__name__)

class MLflowSweepTracker(BaseTracker):
    """Tracks MLflow runs for HPO sweeps."""

    def __init__(self, experiment_name: str):
        """
        Initialize tracker.

        Args:
            experiment_name: MLflow experiment name.
        """
        super().__init__(experiment_name)

    @contextmanager
    def start_sweep_run(
        self,
        run_name: str,
        hpo_config: Dict[str, Any],
        backbone: str,
        study_name: str,
        checkpoint_config: Optional[Dict[str, Any]],
        storage_path: Optional[Any],
        should_resume: bool,
        context: Optional[Any] = None,  # NamingContext
        output_dir: Optional[Path] = None,
        group_id: Optional[str] = None,
        data_config: Optional[Dict[str, Any]] = None,
        benchmark_config: Optional[Dict[str, Any]] = None,
        train_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Start a parent MLflow run for HPO sweep.

        Args:
            run_name: Name for the parent run.
            hpo_config: HPO configuration dictionary.
            backbone: Model backbone name.
            study_name: Optuna study name.
            checkpoint_config: Checkpoint configuration.
            storage_path: Path to checkpoint storage.
            should_resume: Whether this is a resumed run.
            context: Optional NamingContext for tag-based identification.
            output_dir: Optional output directory for metadata persistence.
            group_id: Optional group/session identifier.
            data_config: Optional data configuration dictionary (for grouping tags).
            benchmark_config: Optional benchmark configuration dictionary (for grouping tags).
            train_config: Optional training configuration dictionary (for v2 hash computation).

        Yields:
            RunHandle with run information.
        """
        try:
            with mlflow.start_run(run_name=run_name) as parent_run:
                run_id = parent_run.info.run_id
                experiment_id = parent_run.info.experiment_id
                tracking_uri = mlflow.get_tracking_uri()

                # Infer config_dir from output_dir
                config_dir = infer_config_dir_from_path(output_dir)

                # Compute grouping tags using centralized utilities
                # Priority: 1) Compute v2 from configs (for new runs), 2) Fallback to v1
                from infrastructure.tracking.mlflow.hash_utils import (
                    compute_study_key_hash_v2,
                )
                
                study_key_hash = None
                study_family_hash = None
                
                # Compute v2 hash using fingerprints (requires data_config, hpo_config, train_config)
                if hpo_config and data_config and train_config and context and context.model:
                    study_key_hash = compute_study_key_hash_v2(
                        data_config, hpo_config, train_config, context.model, config_dir
                            )
                    if study_key_hash:
                            logger.info(
                                f"[START_SWEEP_RUN] Computed v2 grouping hashes: "
                            f"study_key_hash={study_key_hash[:16]}..."
                        )
                
                # Always compute family hash (v1, doesn't depend on train_config)
                if not study_family_hash and hpo_config and data_config:
                    try:
                        from infrastructure.tracking.mlflow.naming import (
                            build_hpo_study_family_key,
                            build_hpo_study_family_hash,
                        )
                        study_family_key = build_hpo_study_family_key(
                            data_config, hpo_config, benchmark_config
                        )
                        study_family_hash = build_hpo_study_family_hash(
                            study_family_key)
                        if study_key_hash:
                            logger.info(
                                f"[START_SWEEP_RUN] Computed grouping hashes: "
                                f"study_key_hash={study_key_hash[:16]}..., "
                                f"study_family_hash={study_family_hash[:16]}..."
                            )
                    except Exception as e:
                        logger.warning(
                            f"[START_SWEEP_RUN] Could not compute study_family_hash: {e}",
                            exc_info=True
                        )

                # Build RunHandle (compute run_key_hash before building tags)
                run_key = build_mlflow_run_key(context) if context else None
                run_key_hash = build_mlflow_run_key_hash(
                    run_key) if run_key else None

                # Build and set tags atomically
                tags = build_mlflow_tags(
                    context=context,
                    output_dir=output_dir,
                    group_id=group_id,
                    config_dir=config_dir,
                    study_key_hash=study_key_hash,
                    study_family_hash=study_family_hash,
                    trial_key_hash=None,  # Trial key hash is computed in trial runs
                    run_key_hash=run_key_hash,  # Pass run_key_hash for cleanup matching
                )

                mlflow.set_tags(tags)

                handle = RunHandle(
                    run_id=run_id,
                    run_key=run_key or "",
                    run_key_hash=run_key_hash or "",
                    experiment_id=experiment_id,
                    experiment_name=self.experiment_name,
                    tracking_uri=tracking_uri,
                    artifact_uri=parent_run.info.artifact_uri,
                    study_key_hash=study_key_hash,
                    study_family_hash=study_family_hash,
                )

                # Update local index
                if run_key_hash:
                    try:
                        root_dir = output_dir.parent.parent if output_dir else Path.cwd()
                        update_mlflow_index(
                            root_dir=root_dir,
                            run_key_hash=run_key_hash,
                            run_id=run_id,
                            experiment_id=experiment_id,
                            tracking_uri=tracking_uri,
                            config_dir=config_dir,
                        )
                    except Exception as e:
                        logger.debug(f"Could not update MLflow index: {e}")

                self._log_sweep_metadata(
                    hpo_config, backbone, study_name, checkpoint_config, storage_path, should_resume, output_dir=output_dir
                )
                logger.info(
                    f"[START_SWEEP_RUN] Yielding RunHandle. run_id={run_id[:12]}...")
                yield handle
                logger.info(
                    f"[START_SWEEP_RUN] Context manager exiting normally. run_id={run_id[:12]}...")
        except Exception as e:
            import traceback
            logger.error(f"[START_SWEEP_RUN] MLflow tracking failed: {e}")
            logger.error(
                f"[START_SWEEP_RUN] Traceback: {traceback.format_exc()}")
            logger.warning("Continuing HPO without MLflow tracking...")
            # Yield a dummy context manager that does nothing
            from contextlib import nullcontext
            with nullcontext():
                yield None

    def _log_sweep_metadata(
        self,
        hpo_config: Dict[str, Any],
        backbone: str,
        study_name: str,
        checkpoint_config: Optional[Dict[str, Any]],
        storage_path: Optional[Any],
        should_resume: bool,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Log sweep metadata to MLflow."""
        objective_metric = hpo_config["objective"]["metric"]
        goal = hpo_config.get("objective", {}).get("goal", "maximize")
        max_trials = hpo_config["sampling"]["max_trials"]

        # Infer config_dir from output_dir
        config_dir = infer_config_dir_from_path(output_dir)

        # Mark parent run as sweep job for Azure ML UI
        from infrastructure.naming.mlflow.tag_keys import (
            get_azureml_run_type,
            get_azureml_sweep,
            get_mlflow_run_type,
        )
        azureml_run_type = get_azureml_run_type(config_dir)
        mlflow_run_type = get_mlflow_run_type(config_dir)
        azureml_sweep = get_azureml_sweep(config_dir)
        mlflow.set_tag(azureml_run_type, "sweep")
        mlflow.set_tag(mlflow_run_type, "sweep")
        mlflow.set_tag(azureml_sweep, "true")

        # Log primary metric and goal for Azure ML UI to identify best trial
        mlflow.log_param("primary_metric", objective_metric)
        mlflow.log_param("objective_goal", goal)

        # Log HPO parameters
        mlflow.log_param("backbone", backbone)
        mlflow.log_param("max_trials", max_trials)
        mlflow.log_param("study_name", study_name)
        mlflow.log_param("objective_metric", objective_metric)
        mlflow.log_param("checkpoint_enabled",
                         checkpoint_config.get("enabled", False) if checkpoint_config else False)

        # Log checkpoint path (even if disabled, log None)
        if storage_path is not None:
            mlflow.log_param("checkpoint_path", str(storage_path.resolve()))
        else:
            mlflow.log_param("checkpoint_path", None)

        # Log checkpoint storage type
        mlflow.log_param(
            "checkpoint_storage_type", "sqlite" if storage_path else None
        )

        # Log resume status
        mlflow.log_param("resumed_from_checkpoint", should_resume)

    def log_final_metrics(
        self,
        study: Any,
        objective_metric: str,
        parent_run_id: str,
        run_name: Optional[str] = None,
        should_resume: bool = False,
        hpo_output_dir: Optional[Path] = None,
        backbone: Optional[str] = None,
        run_id: Optional[str] = None,
        fold_splits: Optional[List] = None,
        hpo_config: Optional[Dict[str, Any]] = None,
        child_runs_map: Optional[List] = None,
        upload_checkpoint: bool = True,
        output_dir: Optional[Path] = None,
        config_dir: Optional[Path] = None,
    ) -> None:
        """
        Log final metrics and best trial information to parent run.

        Args:
            upload_checkpoint: If True, upload checkpoint immediately. If False, 
                             defer checkpoint upload (e.g., until after refit completes).
        """
        logger.info(
            f"[LOG_FINAL_METRICS] Starting log_final_metrics for "
            f"parent_run_id={parent_run_id[:12] if parent_run_id else 'None'}..."
        )

        # Infer output_dir if not provided
        if output_dir is None:
            output_dir = hpo_output_dir
        
        # Infer config_dir if not provided
        if config_dir is None:
            # Use output_dir (or hpo_output_dir) for config inference
            config_dir = infer_config_dir_from_path(output_dir)

        try:
            # Count completed trials
            try:
                import optuna
                completed_trials = len(
                    [
                        t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                )
                logger.info(
                    f"[LOG_FINAL_METRICS] Found {completed_trials} completed trials "
                    f"out of {len(study.trials)} total"
                )
            except ImportError:
                completed_trials = len(study.trials)
                logger.warning(
                    f"[LOG_FINAL_METRICS] Could not import optuna, "
                    f"counting all {completed_trials} trials as completed"
                )

            logger.info(
                f"[LOG_FINAL_METRICS] Logging metrics: "
                f"n_trials={len(study.trials)}, "
                f"n_completed_trials={completed_trials}"
            )
            mlflow.log_metric("n_trials", len(study.trials))
            mlflow.log_metric("n_completed_trials", completed_trials)

            if study.best_trial is not None and study.best_value is not None:
                logger.info(
                    f"[LOG_FINAL_METRICS] Logging best trial metrics: "
                    f"{objective_metric}={study.best_value}"
                )

                mlflow.log_metric(
                    f"best_{objective_metric}", study.best_value
                )

                logger.info(
                    f"[LOG_FINAL_METRICS] Logging best hyperparameters: "
                    f"{study.best_params}"
                )
                for param_name, param_value in study.best_params.items():
                    mlflow.log_param(f"best_{param_name}", param_value)

                logger.info(
                    "[LOG_FINAL_METRICS] Finding and logging best trial run ID..."
                )
                self._log_best_trial_id(
                    study,
                    parent_run_id,
                    run_name,
                    should_resume,
                    cached_child_runs=child_runs_map,
                    output_dir=output_dir,
                )
                logger.info(
                    "[LOG_FINAL_METRICS] Completed best trial ID logging"
                )

                log_checkpoint = (
                    upload_checkpoint
                    and hpo_config
                    and hpo_config.get("mlflow", {})
                    .get("log_best_checkpoint", True)
                )
                logger.info(
                    f"[LOG_FINAL_METRICS] Checkpoint logging enabled: {log_checkpoint} "
                    f"(upload_checkpoint={upload_checkpoint})"
                )

                if log_checkpoint:
                    if hpo_output_dir and backbone:
                        logger.info(
                            "[LOG_FINAL_METRICS] Logging best trial checkpoint..."
                        )
                        try:
                            self._log_best_trial_checkpoint(
                                study=study,
                                hpo_output_dir=hpo_output_dir,
                                backbone=backbone,
                                run_id=run_id,
                                fold_splits=fold_splits,
                            )
                            logger.info(
                                "[LOG_FINAL_METRICS] Successfully logged "
                                "best trial checkpoint"
                            )
                        except Exception as e:
                            logger.warning(
                                "[LOG_FINAL_METRICS] Could not log best trial "
                                f"checkpoint to MLflow: {e}"
                            )
                    else:
                        logger.info(
                            "[LOG_FINAL_METRICS] Skipping checkpoint logging: "
                            f"hpo_output_dir={hpo_output_dir}, backbone={backbone}"
                        )
                else:
                    logger.info(
                        "[LOG_FINAL_METRICS] Checkpoint logging deferred "
                        f"(will upload after refit if available)"
                    )

                # Set refit_planned tag if refit is enabled
                if hpo_config and hpo_config.get("refit", {}).get("enabled", False):
                    from infrastructure.naming.mlflow.tag_keys import get_hpo_refit_planned
                    refit_planned_tag = get_hpo_refit_planned(config_dir)
                    mlflow.set_tag(refit_planned_tag, "true")
            else:
                logger.info(
                    "[LOG_FINAL_METRICS] No best trial to log "
                    f"(best_trial={study.best_trial}, "
                    f"best_value={study.best_value if hasattr(study, 'best_value') else 'N/A'})"
                )

            logger.info(
                "[LOG_FINAL_METRICS] Completed log_final_metrics successfully"
            )

        except Exception as e:
            import traceback
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            logger.warning(
                f"[LOG_FINAL_METRICS] Error in log_final_metrics: {error_msg}"
            )
            logger.debug(
                f"[LOG_FINAL_METRICS] Full traceback: {error_traceback}"
            )

            if "'Run' object has no attribute 'run_id'" in error_msg:
                logger.warning(
                    "[LOG_FINAL_METRICS] This error indicates a Run object is "
                    "missing 'run_id' attribute. Please check MLflow version "
                    f"compatibility. Error: {e}"
                )

    def _extract_trial_number(self, run: Any) -> Optional[int]:
        """
        Extract trial number from a run using multiple strategies.

        Tries:
        1. trial_number tag (primary)
        2. code.trial_number tag (alternative)
        3. code.trial tag (alternative)
        4. Parse from run name (e.g., "trial_0_..." -> 0)

        Args:
            run: MLflow Run object

        Returns:
            Trial number as int, or None if not found
        """
        # Strategy 1: Check primary trial_number tag
        trial_num_tag = run.data.tags.get("trial_number")
        if trial_num_tag:
            try:
                return int(trial_num_tag)
            except (ValueError, TypeError):
                pass

        # Strategy 2: Check alternative tag keys
        # Try registry keys first, then fallback to legacy
        config_dir = None  # Could be inferred from context if available
        from infrastructure.naming.mlflow.tag_keys import (
            get_hpo_trial_number,
            get_legacy_trial_number,
        )
        hpo_trial_number_tag = get_hpo_trial_number(config_dir)
        legacy_trial_number_tag = get_legacy_trial_number(config_dir)
        for tag_key in [hpo_trial_number_tag, legacy_trial_number_tag, "code.trial"]:
            tag_value = run.data.tags.get(tag_key)
            if tag_value:
                try:
                    return int(tag_value)
                except (ValueError, TypeError):
                    pass

        # Strategy 3: Parse from run name (fallback)
        run_name = run.data.tags.get(
            "mlflow.runName") or run.info.run_name or ""
        match = re.search(r"trial[_-](\d+)", run_name)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, TypeError):
                pass

        return None

    def _log_best_trial_id(
        self,
        study: Any,
        parent_run_id: str,
        run_name: Optional[str] = None,
        should_resume: bool = False,
        cached_child_runs: Optional[List] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Find and log the best trial's MLflow run ID.
        """
        try:
            client = mlflow.tracking.MlflowClient()
            active_run = mlflow.active_run()
            if active_run is None:
                raise ValueError("No active MLflow run")

            experiment_id = active_run.info.experiment_id
            best_trial_number = study.best_trial.number
            best_run_id = None
            trial_to_run_id = {}

            # Search for best trial run using study_key_hash + trial_number + parentRunId
            # This is more reliable than searching by mlflow.parentRunId alone, which doesn't work
            # due to MLflow's parent-child relationship limitations with nested runs
            try:
                # Get study_key_hash from parent run tags
                parent_run = client.get_run(parent_run_id)
                study_key_hash = parent_run.data.tags.get("code.study_key_hash")
                
                if study_key_hash:
                    # Search for runs with matching study_key_hash and trial_number
                    filter_str = (
                        f"tags.code.study_key_hash = '{study_key_hash}' AND "
                        f"tags.trial_number = '{best_trial_number}'"
                    )
                    candidate_runs = client.search_runs(
                        experiment_ids=[experiment_id],
                        filter_string=filter_str,
                        max_results=100,
                    )
                    
                    # Filter results to only include runs that belong to this parent
                    # (to avoid matching trial runs from other sweeps with same study_key_hash)
                    matching_runs = []
                    for run in candidate_runs:
                        run_parent_id = run.data.tags.get("mlflow.parentRunId")
                        if run_parent_id == parent_run_id:
                            trial_num = self._extract_trial_number(run)
                            if trial_num == best_trial_number:
                                matching_runs.append(run)
                    
                    if len(matching_runs) > 0:
                        # Use the first matching run
                        best_run = matching_runs[0]
                        best_run_id = best_run.info.run_id
                        logger.info(
                            f"Found best trial {best_trial_number} "
                            f"(study_key_hash + trial_number + parentRunId): {best_run_id[:12]}..."
                        )
                        # Log it
                        config_dir = infer_config_dir_from_path(output_dir)
                        from infrastructure.naming.mlflow.tag_keys import (
                            get_hpo_best_trial_number,
                            get_hpo_best_trial_run_id,
                        )
                        best_trial_run_id_tag = get_hpo_best_trial_run_id(config_dir)
                        best_trial_number_tag = get_hpo_best_trial_number(config_dir)
                        mlflow.log_param("best_trial_run_id", best_run_id)
                        mlflow.set_tag(best_trial_run_id_tag, best_run_id)
                        mlflow.set_tag(best_trial_number_tag, str(best_trial_number))
                    else:
                        logger.warning(
                            f"Could not find MLflow run ID for best trial {best_trial_number}. "
                            f"Found {len(candidate_runs)} runs with matching study_key_hash and trial_number, "
                            f"but none belong to parent {parent_run_id[:12]}... "
                            f"This may be a timing issue - trial runs may not be created/committed yet."
                        )
                else:
                    logger.warning(
                        f"Could not find MLflow run ID for best trial {best_trial_number}. "
                        f"Parent run {parent_run_id[:12]}... missing study_key_hash tag. "
                        f"This may be a timing issue - parent run tags may not be set yet."
                    )
            except Exception as search_error:
                logger.warning(
                    f"Could not find MLflow run ID for best trial {best_trial_number}. "
                    f"Search failed: {search_error}. "
                    f"This may be a timing issue - trial runs may not be created/committed yet."
                )

        except Exception as e:
            error_msg = str(e)
            if "'Run' object has no attribute 'run_id'" in error_msg:
                logger.warning(
                    f"Could not retrieve best trial run ID: {e}. "
                    f"Use run.info.run_id instead of run.run_id"
                )
            else:
                logger.warning(
                    f"Could not retrieve best trial run ID: {e}"
                )

    def _log_best_trial_checkpoint(
        self,
        study: Any,
        hpo_output_dir: Path,
        backbone: str,
        run_id: Optional[str] = None,
        fold_splits: Optional[List] = None,
    ) -> None:
        """
        Log best trial checkpoint to MLflow parent run.

        Args:
            study: Completed Optuna study.
            hpo_output_dir: Path to HPO output directory containing trial checkpoints.
            backbone: Model backbone name.
            run_id: HPO run identifier for directory resolution.
            fold_splits: Optional list of fold splits (if k-fold CV used).
        """
        if study.best_trial is None:
            logger.warning("No best trial found, skipping checkpoint logging")
            return

        best_trial_number = study.best_trial.number

        # Resolve checkpoint directory
        # Note: hpo_output_dir may already include backbone (if passed from notebook)
        # or may be the base directory. Check if backbone is already in the path.
        # Single training: {hpo_output_dir}/{backbone}/trial_{number}_{run_id}/checkpoint/
        # K-fold CV: {hpo_output_dir}/{backbone}/trial_{number}_{run_id}_fold{last_idx}/checkpoint/

        # Check if hpo_output_dir already includes backbone (e.g., outputs/hpo/distilbert)
        if hpo_output_dir.name == backbone:
            # hpo_output_dir already includes backbone
            base_dir = hpo_output_dir
        else:
            # hpo_output_dir is base directory (e.g., outputs/hpo), add backbone
            base_dir = hpo_output_dir / backbone

        # Try to find checkpoint directory
        # Priority: 1) refit/checkpoint (preferred), 2) cv/foldN/checkpoint, 3) legacy structure
        # When resuming, run_id changes, so we need to search for the actual checkpoint
        checkpoint_dir = None
        import glob

        # Strategy 1: Search for refit checkpoint (preferred - canonical checkpoint)
        if run_id:
            run_suffix = f"_{run_id}"
            refit_pattern = str(
                base_dir / f"trial_{best_trial_number}{run_suffix}" / "refit" / "checkpoint")
            refit_matches = glob.glob(refit_pattern)
            if refit_matches:
                checkpoint_dir = Path(refit_matches[0])

        # If not found, search for any refit checkpoint
        if not checkpoint_dir or not checkpoint_dir.exists():
            refit_pattern = str(
                base_dir / f"trial_{best_trial_number}_*" / "refit" / "checkpoint")
            refit_matches = glob.glob(refit_pattern)
            if refit_matches:
                checkpoint_dir = Path(refit_matches[0])

        # Strategy 2: If no refit checkpoint, try CV fold checkpoints (new structure: cv/foldN/checkpoint)
        if not checkpoint_dir or not checkpoint_dir.exists():
            if fold_splits is not None and len(fold_splits) > 0:
                # K-fold CV: use last fold's checkpoint
                last_fold_idx = len(fold_splits) - 1
                # First try with the provided run_id
                if run_id:
                    run_suffix = f"_{run_id}"
                    checkpoint_dir = (
                        base_dir /
                        f"trial_{best_trial_number}{run_suffix}" /
                        "cv" /
                        f"fold{last_fold_idx}" /
                        "checkpoint"
                    )

                # If not found, search for any trial directory matching the trial number and fold
                if not checkpoint_dir or not checkpoint_dir.exists():
                    pattern = str(
                        base_dir / f"trial_{best_trial_number}_*" / "cv" / f"fold{last_fold_idx}" / "checkpoint")
                    matches = glob.glob(pattern)
                    if matches:
                        checkpoint_dir = Path(matches[0])
            else:
                # Single training (no CV)
                # First try with the provided run_id
                if run_id:
                    run_suffix = f"_{run_id}"
                    checkpoint_dir = (
                        base_dir /
                        f"trial_{best_trial_number}{run_suffix}" /
                        "checkpoint"
                    )

                # If not found, search for any trial directory matching the trial number
                if not checkpoint_dir or not checkpoint_dir.exists():
                    pattern = str(
                        base_dir / f"trial_{best_trial_number}_*" / "checkpoint")
                    matches = glob.glob(pattern)
                    if matches:
                        checkpoint_dir = Path(matches[0])

        # Strategy 3: Legacy structure fallback (old format: trial_N_foldK/checkpoint)
        if not checkpoint_dir or not checkpoint_dir.exists():
            if fold_splits is not None and len(fold_splits) > 0:
                last_fold_idx = len(fold_splits) - 1
                # Legacy: trial_N_*_foldK/checkpoint
                pattern = str(
                    base_dir / f"trial_{best_trial_number}_*_fold{last_fold_idx}" / "checkpoint")
                matches = glob.glob(pattern)
                if matches:
                    checkpoint_dir = Path(matches[0])
                else:
                    # Try without run_id suffix
                    checkpoint_dir = (
                        base_dir /
                        f"trial_{best_trial_number}_fold{last_fold_idx}" /
                        "checkpoint"
                    )
            else:
                # Legacy: trial_N/checkpoint
                checkpoint_dir = (
                    base_dir /
                    f"trial_{best_trial_number}" /
                    "checkpoint"
                )

        if not checkpoint_dir or not checkpoint_dir.exists():
            logger.warning(
                f"Best trial checkpoint not found for trial {best_trial_number}. "
                f"Searched in: {base_dir}. "
                f"Skipping MLflow checkpoint logging."
            )
            return

        try:
            # Log checkpoint directory as artifact to MLflow
            # Prefer best trial's child run over parent run for artifact upload

            # Get the parent run ID from the current active run
            parent_run_id_for_artifacts = None
            best_trial_run_id = None

            try:
                active_run = mlflow.active_run()
                if active_run and hasattr(active_run, 'info') and hasattr(active_run.info, 'run_id'):
                    parent_run_id_for_artifacts = active_run.info.run_id

                    # Try to get the best trial's child run ID from MLflow tags on the parent run
                    try:
                        client = mlflow.tracking.MlflowClient()
                        parent_run_data = client.get_run(
                            parent_run_id_for_artifacts)
                        if parent_run_data and parent_run_data.data and parent_run_data.data.tags:
                            tags = parent_run_data.data.tags
                            best_trial_run_id = tags.get("best_trial_run_id")
                    except Exception:
                        pass

                    # Fallback: If best_trial_run_id not found in tags, search child runs by trial number
                    if not best_trial_run_id:
                        try:
                            client = mlflow.tracking.MlflowClient()
                            experiment_id = active_run.info.experiment_id
                            # Search for child runs of the parent run with matching trial number
                            child_runs = client.search_runs(
                                experiment_ids=[experiment_id],
                                filter_string=f"tags.mlflow.parentRunId = '{parent_run_id_for_artifacts}' AND (tags.trial_number = '{best_trial_number}' OR tags.optuna.trial.number = '{best_trial_number}')",
                                max_results=1
                            )
                            if child_runs:
                                best_trial_run_id = child_runs[0].info.run_id
                        except Exception:
                            pass
            except Exception:
                pass

            # Use ArtifactUploader for unified checkpoint upload
            artifact_logged = False
            try:
                active_run = mlflow.active_run()
                if not active_run:
                    raise ValueError(
                        "No active MLflow run for artifact logging")

                # Defensive check: ensure active_run has info.run_id
                if not hasattr(active_run, 'info') or not hasattr(active_run.info, 'run_id'):
                    raise ValueError(
                        "Active MLflow run does not have 'info.run_id' attribute")

                # Infer config_dir from checkpoint_dir
                config_dir = infer_config_dir_from_path(checkpoint_dir)
                
                # Use ArtifactUploader for unified checkpoint upload
                uploader = ArtifactUploader(
                    run_id=None,  # Use active run
                    stage=None,  # HPO doesn't have a specific stage config
                    config_dir=config_dir,
                )
                
                logger.info("Uploading checkpoint archive...")
                artifact_logged = uploader.upload_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    artifact_path="best_trial_checkpoint.tar.gz",
                    trial_number=best_trial_number,
                )
                
            except Exception as archive_error:
                error_type = type(archive_error).__name__
                error_msg = str(archive_error)
                logger.warning(
                    f"Failed to upload checkpoint archive: {error_type}: {error_msg}"
                )
                artifact_logged = False
                raise

            # Mark study as complete if checkpoint was successfully uploaded
            if artifact_logged:
                try:
                    from datetime import datetime
                    study.set_user_attr("hpo_complete", "true")
                    study.set_user_attr("checkpoint_uploaded", "true")
                    study.set_user_attr(
                        "completion_timestamp", datetime.now().isoformat())
                    study.set_user_attr("best_trial_number",
                                        str(best_trial_number))
                    logger.info(
                        f"Marked study as complete with checkpoint uploaded (best trial: {best_trial_number})"
                    )
                except Exception as attr_error:
                    logger.warning(
                        f"Could not mark study as complete: {attr_error}"
                    )

            # Log fallback message if upload failed
            if not artifact_logged:
                logger.info(
                    f"Checkpoint for best trial {best_trial_number} is available locally at: {checkpoint_dir}. "
                    f"MLflow artifact logging was not successful, but the checkpoint can be accessed directly."
                )
        except Exception as e:
            # Don't fail HPO if checkpoint logging fails
            logger.warning(f"Could not log checkpoint to MLflow: {e}")
            logger.info(
                f"Checkpoint is available locally at: {checkpoint_dir}")

    def log_best_checkpoint(
        self,
        study: Any,
        hpo_output_dir: Path,
        backbone: str,
        run_id: Optional[str] = None,
        fold_splits: Optional[List] = None,
        prefer_checkpoint_dir: Optional[Path] = None,
        refit_ok: Optional[bool] = None,
        parent_run_id: Optional[str] = None,
        refit_run_id: Optional[str] = None,
    ) -> None:
        """
        Log best trial checkpoint to MLflow run.

        This method can be called after refit to upload the refit checkpoint,
        or as a fallback to upload the best CV checkpoint.

        Args:
            study: Completed Optuna study.
            hpo_output_dir: Path to HPO output directory containing trial checkpoints.
            backbone: Model backbone name.
            run_id: HPO run identifier for directory resolution.
            fold_splits: Optional list of fold splits (if k-fold CV used).
            prefer_checkpoint_dir: Preferred checkpoint directory (e.g., refit checkpoint).
                                  If provided and exists, use it. Otherwise, fallback to CV checkpoint.
            refit_ok: Whether refit completed successfully. Used for tagging.
            parent_run_id: MLflow parent run ID (fallback if refit_run_id not provided).
            refit_run_id: Optional refit run ID. If provided, artifacts will be uploaded to refit run.
        """
        if study.best_trial is None:
            logger.warning("No best trial found, skipping checkpoint logging")
            return

        best_trial_number = study.best_trial.number

        # Get run ID - prefer refit run, then parent run
        # Refit checkpoint should be uploaded to refit run (not parent)
        # Do NOT rely on mlflow.active_run() when refit_run_id is provided
        run_id_to_use = refit_run_id if refit_run_id else parent_run_id
        if not run_id_to_use:
            # Only fall back to active_run if neither refit_run_id nor parent_run_id provided
            active_run = mlflow.active_run()
            if active_run:
                run_id_to_use = active_run.info.run_id
            else:
                raise ValueError(
                    "No active MLflow run and no parent_run_id/refit_run_id provided. "
                    "Cannot log checkpoint artifacts."
                )

        client = mlflow.tracking.MlflowClient()

        # Set refit tags
        if refit_ok is not None:
            try:
                client.set_tag(
                    run_id_to_use,
                    "code.refit_completed",
                    "true" if refit_ok else "false"
                )
            except Exception as e:
                logger.warning(f"Could not set refit_completed tag: {e}")

        # Determine checkpoint directory
        checkpoint_dir = None
        if prefer_checkpoint_dir and prefer_checkpoint_dir.exists():
            checkpoint_dir = prefer_checkpoint_dir
            logger.info(
                f"[LOG_BEST_CHECKPOINT] Using preferred checkpoint directory: {checkpoint_dir}"
            )
        else:
            # Fallback to standard checkpoint search (reuse existing logic)
            logger.info(
                "[LOG_BEST_CHECKPOINT] Falling back to standard checkpoint search")
            self._log_best_trial_checkpoint(
                study=study,
                hpo_output_dir=hpo_output_dir,
                backbone=backbone,
                run_id=run_id,
                fold_splits=fold_splits,
            )
            return

        # Infer config_dir from checkpoint_dir
        config_dir = infer_config_dir_from_path(checkpoint_dir)

        # Use ArtifactUploader for unified checkpoint upload
        try:
            logger.info("Uploading checkpoint archive to MLflow...")

            # Always upload to the target run (refit_run_id if available, otherwise parent_run_id)
            # Uploading to refit run ensures it can be marked as FINISHED after artifact upload
            if refit_run_id:
                logger.info(
                    f"Uploading checkpoint to refit run {refit_run_id[:12]} "
                    f"(child of parent {parent_run_id[:12] if parent_run_id else 'N/A'})"
                )
            else:
                logger.info(
                    f"Uploading checkpoint to parent run {parent_run_id[:12] if parent_run_id else 'N/A'}"
                )
            
            # Use ArtifactUploader for unified checkpoint upload
            uploader = ArtifactUploader(
                run_id=run_id_to_use,
                stage=None,  # HPO doesn't have a specific stage config
                config_dir=config_dir,
            )
            
            artifact_logged = uploader.upload_checkpoint(
                checkpoint_dir=checkpoint_dir,
                artifact_path="best_trial_checkpoint.tar.gz",
                trial_number=best_trial_number,
            )

            # Note: Detailed upload message (file count, size) is already logged by uploader.upload_checkpoint()
            # This message confirms the overall operation completed
            if artifact_logged:
                logger.info(
                    f"Checkpoint upload completed successfully for trial {best_trial_number}"
                )

                # Update artifact.available tag to "true" after successful checkpoint upload
                try:
                    from infrastructure.naming.mlflow.tags_registry import load_tags_registry
                    tags_registry = load_tags_registry(config_dir)
                    artifact_tag = tags_registry.key("artifact", "available")
                    
                    # Update on the run where artifacts were uploaded
                    client.set_tag(run_id_to_use, artifact_tag, "true")
                    logger.info(
                        f"[LOG_BEST_CHECKPOINT] Updated {artifact_tag}=true on run {run_id_to_use[:12]}... "
                        f"after successful checkpoint upload"
                    )
                    
                    # Also update on parent run (where Phase 2 tags are set)
                    # This ensures parent run has artifact.available=true for champion selection
                    if parent_run_id and run_id_to_use != parent_run_id:
                        client.set_tag(parent_run_id, artifact_tag, "true")
                        logger.debug(
                            f"[LOG_BEST_CHECKPOINT] Updated {artifact_tag}=true on parent run {parent_run_id[:12]}... "
                            f"(artifacts uploaded to child run)"
                        )
                except Exception as tag_error:
                    logger.warning(
                        f"[LOG_BEST_CHECKPOINT] Could not update artifact.available tag: {tag_error}"
                    )
            else:
                logger.warning("Checkpoint upload returned False (may have been skipped)")

            # Mark study as complete after successful checkpoint upload
            try:
                from datetime import datetime
                study.set_user_attr("hpo_complete", "true")
                study.set_user_attr("checkpoint_uploaded", "true")
                study.set_user_attr(
                    "completion_timestamp", datetime.now().isoformat())
                study.set_user_attr("best_trial_number",
                                    str(best_trial_number))
                logger.info(
                    f"Marked study as complete with checkpoint uploaded (best trial: {best_trial_number})"
                )
            except Exception as attr_error:
                logger.warning(
                    f"Could not mark study as complete: {attr_error}"
                )

        except Exception as e:
            logger.error(f"Failed to upload checkpoint via MLflow: {e}")
            raise

    def log_tracking_info(self) -> None:
        """Log MLflow tracking URI information for user visibility."""
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri:
            # Check if it's actually Azure ML (starts with azureml://)
            if tracking_uri.lower().startswith("azureml://"):
                logger.info("Using Azure ML Workspace for MLflow tracking")
                logger.debug(f"Tracking URI: {tracking_uri}")
            elif tracking_uri.startswith("sqlite://") or tracking_uri.startswith("file://"):
                logger.warning("Using LOCAL MLflow tracking (not Azure ML)")
                logger.debug(f"Tracking URI: {tracking_uri}")
                logger.info(
                    "To use Azure ML, ensure:\n"
                    "  1. config/mlflow.yaml has azure_ml.enabled: true\n"
                    "  2. Environment variables are set: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP\n"
                    "  3. Azure ML SDK is installed: pip install azure-ai-ml azure-identity azureml-mlflow"
                )
            else:
                logger.info(f"Using MLflow tracking: {tracking_uri}")
        else:
            logger.warning("MLflow tracking URI not set")
