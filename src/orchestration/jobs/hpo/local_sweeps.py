
"""Local hyperparameter optimization using Optuna."""

from __future__ import annotations
import logging
from shared.json_cache import save_json
from shared.logging_utils import get_logger

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List, Tuple

import mlflow
import numpy as np

from .local.checkpoint.manager import resolve_storage_path, get_storage_uri
from .search_space import translate_search_space_to_optuna
from ..tracking.mlflow_tracker import MLflowSweepTracker
from .hpo_helpers import (
    generate_run_id,
    setup_checkpoint_storage,
    create_study_name,
    create_mlflow_run_name,
)

# Import from extracted modules
from .local.optuna.integration import import_optuna as _import_optuna, create_optuna_pruner
import optuna
from .local.trial.execution import run_training_trial
from .local.cv.orchestrator import run_training_trial_with_cv
from .local.refit.executor import run_refit_training
from .local.checkpoint.cleanup import CheckpointCleanupManager
from .local.trial.metrics import store_metrics_in_trial_attributes
from .local.trial.run_manager import create_trial_run_no_cv, finalize_trial_run_no_cv
from .local.trial.callback import create_trial_callback
from .local.study.manager import StudyManager
from .local.mlflow.run_setup import setup_hpo_mlflow_run, commit_run_name_version
from .local.mlflow.cleanup import cleanup_interrupted_runs

logger = get_logger(__name__)

# Suppress Optuna's verbose output to reduce log clutter
logging.getLogger("optuna").setLevel(logging.WARNING)

# Re-export for backward compatibility
__all__ = [
    "run_training_trial",
    "run_training_trial_with_cv",
    "create_optuna_pruner",
    "create_local_hpo_objective",
    "run_refit_training",
    "run_local_hpo_sweep",
]


# Functions are now imported from extracted modules above
# Keeping this comment for clarity - old definitions removed


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
    data_config: Optional[Dict[str, Any]] = None,
    benchmark_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Callable[[Any], float], Callable[[], None]]:
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
    logger.debug(
        f"[CV Setup] k_folds={k_folds}, fold_splits_file={fold_splits_file}")
    if k_folds is not None and k_folds > 1:
        if fold_splits_file and fold_splits_file.exists():
            # Load existing splits
            logger.debug(
                f"[CV Setup] Loading existing fold splits from {fold_splits_file}")
            from training.cv_utils import load_fold_splits
            fold_splits, _ = load_fold_splits(fold_splits_file)
            logger.debug(
                f"[CV Setup] Loaded {len(fold_splits) if fold_splits else 0} fold splits")
        else:
            # Create new splits and save them (even if fold_splits_file is None, auto-generate path)
            logger.debug(f"[CV Setup] Creating new {k_folds}-fold splits")
            from training.data import load_dataset
            from training.cv_utils import create_kfold_splits, save_fold_splits

            try:
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

                # Auto-generate fold_splits_file path if not provided
                if fold_splits_file is None:
                    # Use a default location in the output directory
                    fold_splits_file = output_base_dir / "fold_splits.json"

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
                logger.info(
                    f"[CV Setup] ✓ Created {len(fold_splits)} fold splits and saved to {fold_splits_file}")
            except Exception as e:
                logger.error(
                    f"[CV Setup] Failed to create fold splits: {e}", exc_info=True)
                raise
    else:
        logger.debug(
            f"[CV Setup] CV disabled: k_folds={k_folds} (must be > 1 to enable CV)")

    # Capture run_id in closure to avoid UnboundLocalError
    captured_run_id = run_id

    # Initialize checkpoint cleanup manager
    cleanup_manager = CheckpointCleanupManager(
        output_base_dir=output_base_dir,
        hpo_config=hpo_config,
        run_id=run_id,
        fold_splits=fold_splits,
    )

    def objective(trial: Any) -> float:
        # Sample hyperparameters
        # Exclude "backbone" from search space since it's fixed per study
        trial_params = translate_search_space_to_optuna(
            hpo_config, trial, exclude_params=["backbone"])
        # Set the fixed backbone for this study
        trial_params["backbone"] = backbone
        trial_params["trial_number"] = trial.number
        # Pass run_id to trial functions (use captured value from outer scope)
        trial_params["run_id"] = captured_run_id

        # Don't create child runs in parent process - let subprocess create them
        # This avoids issues with ended runs and ensures training logs to the correct run
        # We'll pass the parent run ID and let the subprocess create nested child runs

        # Get HPO parent run ID for nested structure (trial -> folds)
        hpo_parent_run_id = None
        study_key_hash = None
        study_family_hash = None
        try:
            active_run = mlflow.active_run()
            if active_run:
                hpo_parent_run_id = active_run.info.run_id
                # Get grouping tags from parent run
                try:
                    client = mlflow.tracking.MlflowClient()
                    parent_run = client.get_run(hpo_parent_run_id)
                    study_key_hash = parent_run.data.tags.get(
                        "code.study_key_hash")
                    study_family_hash = parent_run.data.tags.get(
                        "code.study_family_hash")
                except Exception as e:
                    logger.debug(
                        f"Could not get grouping tags from parent run: {e}")
        except Exception:
            pass

        # Run training with or without CV
        logger.info(
            f"[Trial {trial.number}] fold_splits={'present' if fold_splits is not None else 'None'}, using {'CV' if fold_splits is not None else 'non-CV'} path")
        if fold_splits is not None:
            # Run k-fold CV with nested structure
            logger.info(
                f"[Trial {trial.number}] Running {len(fold_splits)}-fold CV")
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
                hpo_parent_run_id=hpo_parent_run_id,  # Pass HPO parent to create trial run
                study_key_hash=study_key_hash,  # Pass grouping tags
                study_family_hash=study_family_hash,
                data_config=data_config,  # Pass configs for study_key reconstruction
                hpo_config=hpo_config,
                benchmark_config=benchmark_config,
            )

            # Log CV statistics to trial user attributes
            trial.set_user_attr("cv_mean", float(average_metric))
            trial.set_user_attr("cv_std", float(np.std(fold_metrics)))
            trial.set_user_attr("cv_fold_metrics", [
                                float(m) for m in fold_metrics])

            metric_value = average_metric
        else:
            # Run single training (no CV)
            # Create trial-level run as child of HPO parent for consistency
            trial_run_id_for_no_cv = create_trial_run_no_cv(
                trial_params=trial_params,
                config_dir=config_dir,
                output_base_dir=output_base_dir,
                hpo_parent_run_id=hpo_parent_run_id,
            )

            # Create v2 trial folder for non-CV trials (similar to CV trials)
            trial_output_dir = output_base_dir
            trial_key_hash = None
            if study_key_hash:
                try:
                    from orchestration.jobs.tracking.mlflow_naming import (
                        build_hpo_trial_key,
                        build_hpo_trial_key_hash,
                    )
                    # Extract hyperparameters (exclude metadata fields)
                    hyperparameters = {
                        k: v for k, v in trial_params.items()
                        if k not in ("backbone", "trial_number", "run_id")
                    }
                    trial_key = build_hpo_trial_key(
                        study_key_hash, hyperparameters)
                    trial_key_hash = build_hpo_trial_key_hash(trial_key)
                    # Use token expansion for consistency
                    from naming.context_tokens import build_token_values
                    from naming.context import NamingContext
                    temp_context = NamingContext(
                        process_type="hpo",
                        model=backbone.split("-")[0] if "-" in backbone else backbone,
                        environment=detect_platform(),
                        trial_key_hash=trial_key_hash
                    )
                    tokens = build_token_values(temp_context)
                    trial8 = tokens["trial8"]

                    # Create trial-{hash} folder in study folder
                    trial_output_dir = output_base_dir / f"trial-{trial8}"
                    trial_output_dir.mkdir(parents=True, exist_ok=True)

                    # Create trial_meta.json for easy lookup
                    import json
                    study_folder_name = output_base_dir.name
                    trial_meta = {
                        "study_key_hash": study_key_hash,
                        "trial_key_hash": trial_key_hash,
                        "trial_number": trial.number,
                        "study_name": study_folder_name,
                        "run_id": captured_run_id,
                        "created_at": datetime.now().isoformat(),
                    }
                    trial_meta_path = trial_output_dir / "trial_meta.json"
                    with open(trial_meta_path, "w") as f:
                        json.dump(trial_meta, f, indent=2)

                    logger.debug(
                        f"Created v2 trial folder for non-CV trial {trial.number}: {trial_output_dir.name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not create v2 trial folder for non-CV trial {trial.number}: {e}. "
                        f"Using study folder directly."
                    )
                    # Fallback to study folder if trial folder creation fails
                    trial_output_dir = output_base_dir

            metric_value = run_training_trial(
                trial_params=trial_params,
                dataset_path=dataset_path,
                config_dir=config_dir,
                backbone=backbone,
                output_dir=trial_output_dir,  # Use trial folder instead of study folder
                train_config=train_config,
                mlflow_experiment_name=mlflow_experiment_name,
                objective_metric=objective_metric,
                parent_run_id=trial_run_id_for_no_cv if trial_run_id_for_no_cv else hpo_parent_run_id,
            )

            # Mark trial run as FINISHED after training completes (no CV case)
            if trial_run_id_for_no_cv:
                finalize_trial_run_no_cv(trial_run_id_for_no_cv, trial.number)

        # Store additional metrics in trial user attributes for callback display
        # Resolve output_base_dir to actual path (Drive if checkpoints are in Drive)
        # This ensures we look in the same location where CV orchestrator created trial folders
        resolved_output_base_dir = output_base_dir
        if fold_splits is not None:
            # For CV trials, trial folders are created in Drive (if checkpoints are in Drive)
            # Check if Drive path exists and use that instead of local path
            try:
                import os
                if "COLAB_GPU" in os.environ or "COLAB_TPU" in os.environ:
                    # Construct Drive path: /content/drive/MyDrive/resume-ner-azureml/...
                    # output_base_dir is like: /content/resume-ner-azureml/outputs/hpo/colab/distilbert/study-584922ce
                    # Drive path would be: /content/drive/MyDrive/resume-ner-azureml/outputs/hpo/colab/distilbert/study-584922ce
                    drive_path = Path("/content/drive/MyDrive") / \
                        output_base_dir.relative_to("/content")
                    if drive_path.exists():
                        resolved_output_base_dir = drive_path
                        logger.debug(
                            f"[Metrics] Using Drive path: {resolved_output_base_dir}")
                    else:
                        logger.debug(
                            f"[Metrics] Drive path does not exist: {drive_path}. Using local path: {output_base_dir}")
            except Exception as e:
                logger.debug(
                    f"[Metrics] Could not resolve Drive path: {e}. Using original path.")

        store_metrics_in_trial_attributes(
            trial=trial,
            output_base_dir=resolved_output_base_dir,
            trial_number=trial.number,
            run_id=captured_run_id,
            fold_splits=fold_splits,
        )

        # Report to Optuna
        trial.report(metric_value, step=0)

        # Handle checkpoint cleanup
        cleanup_manager.register_trial_checkpoint(trial.number)
        early_return = cleanup_manager.handle_trial_completion(
            trial, metric_value)
        if early_return is not None:
            return early_return

        return metric_value

    def cleanup_non_best_checkpoints() -> None:
        """Final cleanup: delete all non-best checkpoints after HPO completes, preserving refit checkpoints."""
        cleanup_manager.final_cleanup()

    return objective, cleanup_non_best_checkpoints


# run_refit_training is now imported from refit_executor module above


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
    restore_from_drive: Optional[Callable[[Path], bool]] = None,
    data_config: Optional[Dict[str, Any]] = None,
    benchmark_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Run a local hyperparameter optimization sweep using Optuna.

    Args:
        dataset_path: Path to dataset directory.
        config_dir: Path to configuration directory (project config, not HPO output config).
        backbone: Model backbone name.
        hpo_config: HPO configuration dictionary.
        train_config: Training configuration dictionary.
        output_dir: Base output directory for all trials.
        mlflow_experiment_name: MLflow experiment name.
        k_folds: Optional number of k-folds for cross-validation.
        fold_splits_file: Optional path to fold splits file.
        checkpoint_config: Optional checkpoint configuration dict with 'enabled',
                          'storage_path', and 'auto_resume' keys.
        restore_from_drive: Optional function to restore checkpoint from Drive if missing.
                          Function should take a Path and return bool (True if restored).
        data_config: Optional data configuration dictionary (for grouping tags).
        benchmark_config: Optional benchmark configuration dictionary (for grouping tags).

    Returns:
        Optuna study object with completed trials.
    """
    # Store original project config_dir (never overwrite with outputs/hpo/config)
    project_config_dir = config_dir

    # Validate that config_dir exists and is the project config (not HPO output config)
    if not project_config_dir.exists():
        raise FileNotFoundError(
            f"Project config_dir not found: {project_config_dir}. "
            f"Ensure you're passing the project root config directory, not outputs/hpo/config"
        )

    # Generate unique run ID
    run_id = generate_run_id()

    # Compute study_key_hash EARLY (before creating study) to enable v2 folder creation
    # This allows us to create study.db in the v2 folder from the start, avoiding legacy folder creation
    study_key_hash = None
    if data_config and hpo_config:
        try:
            from orchestration.jobs.tracking.naming.hpo_keys import (
                build_hpo_study_key,
                build_hpo_study_key_hash,
            )
            study_key = build_hpo_study_key(
                data_config=data_config,
                hpo_config=hpo_config,
                model=backbone,
                benchmark_config=benchmark_config,
            )
            study_key_hash = build_hpo_study_key_hash(study_key)
        except Exception as e:
            logger.warning(f"✗ Could not compute study_key_hash early: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")

    # Create study manager and load/create study
    study_manager = StudyManager(
        backbone=backbone,
        hpo_config=hpo_config,
        checkpoint_config=checkpoint_config,
        restore_from_drive=restore_from_drive,
    )

    # Create v2 folder and use it for study.db (study_key_hash required)
    v2_study_folder = None
    if study_key_hash:
        try:
            from paths import load_paths_config, apply_env_overrides, resolve_output_path
            from shared.platform_detection import detect_platform

            # Derive root_dir by walking up from output_dir until we find "outputs" directory
            root_dir = None
            current = output_dir
            while current.parent != current:  # Stop at filesystem root
                if current.name == "outputs":
                    root_dir = current.parent
                    break
                current = current.parent

            if root_dir is None:
                # Fallback: try to find project root by looking for config directory
                root_dir = Path.cwd()
                for candidate in [Path.cwd(), Path.cwd().parent]:
                    if (candidate / "config").exists():
                        root_dir = candidate
                        break

            config_dir = root_dir / "config"
            hpo_base = resolve_output_path(root_dir, config_dir, "hpo")
            paths_config = load_paths_config(config_dir)
            storage_env = detect_platform()
            paths_config = apply_env_overrides(paths_config, storage_env)

            pattern = paths_config.get("patterns", {}).get("hpo_v2", "")
            if pattern:
                # Use token expansion for consistency
                from naming.context_tokens import build_token_values
                from naming.context import NamingContext
                temp_context = NamingContext(
                    process_type="hpo",
                    model=backbone.split("-")[0] if "-" in backbone else backbone,
                    environment=detect_platform(),
                    study_key_hash=study_key_hash
                )
                tokens = build_token_values(temp_context)
                study8 = tokens["study8"]
                model = backbone.split("-")[0] if "-" in backbone else backbone

                v2_study_folder = hpo_base / \
                    storage_env / model / f"study-{study8}"
                v2_study_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(
                f"Could not create v2 study folder early, will use legacy: {e}")

    # Create or load study - pass v2_study_folder if available to use for study.db
    study, study_name, storage_path, storage_uri, should_resume = (
        study_manager.create_or_load_study(
            output_dir, run_id, v2_study_folder=v2_study_folder)
    )

    # Check if HPO is already complete (early return)
    if should_resume:
        user_attrs = study.user_attrs if hasattr(study, "user_attrs") else {}
        hpo_complete = user_attrs.get(
            "hpo_complete", "false").lower() == "true"
        checkpoint_uploaded = (
            user_attrs.get("checkpoint_uploaded", "false").lower() == "true"
        )
        if hpo_complete and checkpoint_uploaded:
            return study

    # Get objective metric name
    objective_metric = study_manager.objective_metric

    # Use v2 folder (study.db was created there)
    if not v2_study_folder or not v2_study_folder.exists():
        raise RuntimeError(
            f"V2 study folder not created. Expected: {v2_study_folder}. "
            f"Only v2 paths (study-{{hash}}) are supported."
        )
    study_folder = v2_study_folder
    study_output_dir = study_folder

    # Check if checkpointing is enabled (needed for setup_hpo_mlflow_run)
    checkpoint_enabled = checkpoint_config is not None and checkpoint_config.get(
        "enabled", False
    )

    # Create NamingContext for HPO parent run (study_key_hash already computed earlier)
    # Pass the computed study_key_hash to avoid recomputation and ensure it's set in the frozen context
    hpo_parent_context, mlflow_run_name = setup_hpo_mlflow_run(
        backbone=backbone,
        study_name=study_name,
        output_dir=output_dir,
        run_id=run_id,
        should_resume=should_resume,
        checkpoint_enabled=checkpoint_enabled,
        data_config=data_config,
        hpo_config=hpo_config,
        benchmark_config=benchmark_config,
        study_key_hash=study_key_hash,  # Pass pre-computed hash to avoid recomputation
    )

    # Write .active_study.json marker for fast lookup in selection code
    try:
        from orchestration.jobs.local_selection_v2 import write_active_study_marker
        backbone_dir = study_folder.parent
        write_active_study_marker(
            backbone_dir=backbone_dir,
            study_folder=study_folder,
            study_name=study_name,
            study_key_hash=None,  # Will be updated later when MLflow run is created
        )
        logger.debug(
            f"Wrote .active_study.json marker to {backbone_dir / '.active_study.json'}")
    except Exception as e:
        logger.debug(f"Could not write .active_study.json marker: {e}")

    # Run ID was already generated above for study naming
    # Print it here for user visibility
    logger.debug(f"Run ID: {run_id} (prevents overwriting on reruns)")

    # Load fold splits for logging (if k-fold CV is enabled)
    fold_splits_for_logging = None
    if k_folds is not None and k_folds > 1 and fold_splits_file and fold_splits_file.exists():
        try:
            from training.cv_utils import load_fold_splits
            fold_splits_for_logging, _ = load_fold_splits(fold_splits_file)
            logger.debug(
                f"Loaded {len(fold_splits_for_logging)} fold splits for logging")
        except Exception as e:
            logger.warning(f"Could not load fold splits for logging: {e}")

    # MLflow experiment setup is handled by the tracker (MLflowSweepTracker)
    # which reads from mlflow.yaml config. No need to set it up here.

    # Create objective function and cleanup function
    logger.info(
        f"[HPO Setup] k_folds={k_folds}, fold_splits_file={fold_splits_file}, k_fold config: {hpo_config.get('k_fold', {})}")
    objective, cleanup_checkpoints = create_local_hpo_objective(
        dataset_path=dataset_path,
        config_dir=config_dir,
        backbone=backbone,
        hpo_config=hpo_config,
        train_config=train_config,
        output_base_dir=study_output_dir,  # Use study folder instead of base output_dir
        mlflow_experiment_name=mlflow_experiment_name,
        objective_metric=objective_metric,
        k_folds=k_folds,
        fold_splits_file=fold_splits_file,
        run_id=run_id,
        data_config=data_config,
        benchmark_config=benchmark_config,
    )

    # Create callback factory to capture parent_run_id

    # Calculate remaining trials
    max_trials = hpo_config["sampling"]["max_trials"]
    timeout_seconds = hpo_config["sampling"]["timeout_minutes"] * 60

    # Cleanup stale reservations from crashed processes
    try:
        from orchestration.jobs.tracking.mlflow_index import cleanup_stale_reservations
        root_dir = output_dir.parent.parent if output_dir else Path.cwd()
        config_dir = output_dir.parent.parent / "config" if output_dir else None
        cleaned_count = cleanup_stale_reservations(
            root_dir, config_dir, stale_minutes=30)
        if cleaned_count > 0:
            logger.info(
                f"Cleaned up {cleaned_count} stale run name reservations")
    except Exception as e:
        logger.debug(f"Could not cleanup stale reservations: {e}")

    # Use v2 study folder
    study_folder = study_output_dir

    # IMPORTANT: Do NOT move study.db after the study object has been created.
    # The Optuna study object maintains a connection to the database at the original location.
    # Moving the file breaks this connection and causes "no such table: studies" errors.
    # Instead, ensure the database stays where it was created (storage_path location).
    # The v2 folder structure is used for trial outputs, but the database can remain
    # in the legacy location for backward compatibility and to avoid breaking the study connection.
    if storage_path and storage_path.parent != study_folder:
        logger.debug(
            f"Database at {storage_path} is in different location than study folder {study_folder}. "
            f"Keeping database in original location to maintain Optuna study connection."
        )

    # Use study folder as base for trials (so trials are created inside study folder)

    tracker = MLflowSweepTracker(mlflow_experiment_name)
    tracker.log_tracking_info()

    parent_run_id = None
    parent_run_handle = None

    try:
        with tracker.start_sweep_run(
            run_name=mlflow_run_name,
            context=hpo_parent_context,
            output_dir=output_dir,
            hpo_config=hpo_config,
            backbone=backbone,
            study_name=study_name,
            checkpoint_config=checkpoint_config,
            storage_path=storage_path,
            should_resume=should_resume,
            data_config=data_config,
            benchmark_config=benchmark_config,
        ) as parent_run:

            parent_run_handle = parent_run
            parent_run_id = parent_run.run_id if parent_run else None

            # Commit reserved version if auto-increment was used
            commit_run_name_version(
                parent_run_id=parent_run_id,
                hpo_parent_context=hpo_parent_context,
                mlflow_run_name=mlflow_run_name,
                output_dir=output_dir,
            )

            # Update .active_study.json marker with study_key_hash from MLflow run
            try:
                from orchestration.jobs.local_selection_v2 import write_active_study_marker
                import mlflow
                client = mlflow.tracking.MlflowClient()
                parent_run = client.get_run(parent_run_id)
                study_key_hash = parent_run.data.tags.get(
                    "code.study_key_hash")
                if study_key_hash:
                    backbone_dir = study_folder.parent
                    write_active_study_marker(
                        backbone_dir=backbone_dir,
                        study_folder=study_folder,
                        study_name=study_name,
                        study_key_hash=study_key_hash,
                    )
                    logger.debug(
                        f"Updated .active_study.json marker with study_key_hash")
            except Exception as e:
                logger.debug(
                    f"Could not update .active_study.json marker with study_key_hash: {e}")

            # Cleanup: Tag interrupted runs from previous sessions
            parent_to_children = cleanup_interrupted_runs(
                parent_run_id=parent_run_id,
                hpo_parent_context=hpo_parent_context,
                mlflow_experiment_name=mlflow_experiment_name,
                mlflow_run_name=mlflow_run_name,
                output_dir=output_dir,
                hpo_config=hpo_config,
            )
            # Extract child_runs_map for reuse in log_final_metrics
            child_runs_map = None
            if parent_to_children:
                tracking_uri = mlflow.get_tracking_uri()
                experiment = mlflow.get_experiment_by_name(
                    mlflow_experiment_name)
                if experiment:
                    child_runs_map = parent_to_children.get(parent_run_id, [])

            trial_callback = create_trial_callback(
                objective_metric, parent_run_id)

            if should_resume:
                completed_trials = len(
                    [
                        t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                )
                remaining_trials = max(0, max_trials - completed_trials)

                if remaining_trials > 0:
                    study.optimize(
                        objective,
                        n_trials=remaining_trials,
                        timeout=timeout_seconds,
                        show_progress_bar=True,
                        callbacks=[trial_callback],
                    )
            else:
                study.optimize(
                    objective,
                    n_trials=max_trials,
                    timeout=timeout_seconds,
                    show_progress_bar=True,
                    callbacks=[trial_callback],
                )

            if parent_run_id and parent_run_handle:
                try:
                    # Pass child_runs_map if available from cleanup
                    # Log metrics immediately, but defer checkpoint upload until after refit
                    tracker.log_final_metrics(
                        study=study,
                        objective_metric=objective_metric,
                        parent_run_id=parent_run_id,
                        run_name=mlflow_run_name,
                        should_resume=should_resume,
                        hpo_output_dir=output_dir,
                        backbone=backbone,
                        run_id=run_id,
                        fold_splits=fold_splits_for_logging,
                        hpo_config=hpo_config,
                        child_runs_map=child_runs_map if 'child_runs_map' in locals() else None,
                        upload_checkpoint=False,  # Defer checkpoint upload until after refit
                    )
                except Exception as e:
                    logger.error(
                        f"[HPO] Error in log_final_metrics: {e}"
                    )
                    import traceback
                    logger.error(traceback.format_exc())

            # MOVE REFIT INSIDE THE 'with' BLOCK so parent run stays open until refit completes
            # Run refit training if enabled
            refit_config = hpo_config.get("refit", {})
            refit_enabled = refit_config.get(
                "enabled", True)  # Default: enabled

            if refit_enabled and study.best_trial is not None:
                try:
                    logger.info(
                        f"[REFIT] Starting refit training for best trial {study.best_trial.number}"
                    )

                    # Get grouping tags and compute trial_key_hash
                    # Priority: 1) recompute from configs (most reliable), 2) get from parent tags
                    refit_study_key_hash = None
                    refit_study_family_hash = None
                    refit_trial_key_hash = None

                    # Primary: compute grouping hashes from configs (ensures canonicalization matches)
                    if data_config and hpo_config:
                        try:
                            from orchestration.jobs.tracking.mlflow_naming import (
                                build_hpo_study_key,
                                build_hpo_study_family_key,
                                build_hpo_study_key_hash,
                                build_hpo_study_family_hash,
                            )
                            study_key = build_hpo_study_key(
                                data_config, hpo_config, backbone, benchmark_config
                            )
                            refit_study_key_hash = build_hpo_study_key_hash(
                                study_key)
                            study_family_key = build_hpo_study_family_key(
                                data_config, hpo_config, benchmark_config
                            )
                            refit_study_family_hash = build_hpo_study_family_hash(
                                study_family_key)
                            logger.debug(
                                f"[REFIT] Computed grouping hashes from configs: "
                                f"study_key_hash={refit_study_key_hash[:16]}..., "
                                f"study_family_hash={refit_study_family_hash[:16]}..."
                            )
                        except Exception as e:
                            logger.debug(
                                f"Could not compute grouping hashes from configs for refit: {e}")

                    # Fallback: get from parent run tags (for backward compatibility)
                    if (not refit_study_key_hash or not refit_study_family_hash) and parent_run_id:
                        try:
                            # Use mlflow from top-level import (line 15)
                            client = mlflow.tracking.MlflowClient()
                            parent_run = client.get_run(parent_run_id)
                            if not refit_study_key_hash:
                                refit_study_key_hash = parent_run.data.tags.get(
                                    "code.study_key_hash")
                                if refit_study_key_hash:
                                    logger.debug(
                                        f"[REFIT] Retrieved study_key_hash from parent tags: {refit_study_key_hash[:16]}..."
                                    )
                            if not refit_study_family_hash:
                                refit_study_family_hash = parent_run.data.tags.get(
                                    "code.study_family_hash")
                                if refit_study_family_hash:
                                    logger.debug(
                                        f"[REFIT] Retrieved study_family_hash from parent tags: {refit_study_family_hash[:16]}..."
                                    )
                        except Exception as e:
                            logger.debug(
                                f"Could not get grouping tags from parent run for refit: {e}")

                    # Compute trial_key_hash from best trial hyperparameters and study_key_hash
                    if refit_study_key_hash and study.best_trial:
                        try:
                            from orchestration.jobs.tracking.mlflow_naming import (
                                build_hpo_trial_key,
                                build_hpo_trial_key_hash,
                            )
                            # Extract hyperparameters from best trial
                            hyperparameters = {
                                k: v for k, v in study.best_trial.params.items()
                                if k not in ("backbone", "trial_number", "run_id")
                            }
                            trial_key = build_hpo_trial_key(
                                refit_study_key_hash, hyperparameters)
                            refit_trial_key_hash = build_hpo_trial_key_hash(
                                trial_key)
                            logger.info(
                                f"[REFIT] Computed trial_key_hash={refit_trial_key_hash[:16]}... "
                                f"from study_key_hash and best trial hyperparameters"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not compute trial_key_hash for refit: {e}", exc_info=True)

                    # Compute refit protocol fingerprint
                    from orchestration.jobs.tracking.mlflow_naming import compute_refit_protocol_fp
                    refit_protocol_fp = compute_refit_protocol_fp(
                        data_config=data_config if data_config else {},
                        train_config=train_config,
                    )

                    # Run refit training
                    # Debug: Verify function signature before calling (helps diagnose module reload issues)
                    try:
                        import inspect
                        sig = inspect.signature(run_refit_training)
                        logger.debug(
                            f"[REFIT] run_refit_training signature: {sig}, "
                            f"module: {run_refit_training.__module__}, "
                            f"file: {run_refit_training.__code__.co_filename}, "
                            f"line: {run_refit_training.__code__.co_firstlineno}"
                        )
                        if "study_family_hash" not in sig.parameters:
                            logger.error(
                                f"[REFIT] CRITICAL: run_refit_training does not have 'study_family_hash' parameter! "
                                f"This suggests module reload/shadowing issue. "
                                f"Signature: {sig}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[REFIT] Could not inspect run_refit_training signature: {e}")

                    refit_ok = False
                    refit_checkpoint_dir = None
                    refit_metrics = None
                    refit_run_id = None

                    try:
                        refit_metrics, refit_checkpoint_dir, refit_run_id = run_refit_training(
                            best_trial=study.best_trial,
                            dataset_path=dataset_path,
                            config_dir=project_config_dir,  # Use original project config, not HPO output config
                            backbone=backbone,
                            output_dir=study_output_dir,  # Use study folder instead of base output_dir
                            train_config=train_config,
                            mlflow_experiment_name=mlflow_experiment_name,
                            objective_metric=objective_metric,
                            hpo_parent_run_id=parent_run_id,
                            study_key_hash=refit_study_key_hash,
                            study_family_hash=refit_study_family_hash,
                            trial_key_hash=refit_trial_key_hash,
                            refit_protocol_fp=refit_protocol_fp,
                            run_id=run_id,
                        )
                        refit_ok = True
                        logger.info(
                            f"[REFIT] Refit training completed. Metrics: {refit_metrics}, "
                            f"Checkpoint: {refit_checkpoint_dir}, Run ID: {refit_run_id[:12] if refit_run_id else 'None'}..."
                        )

                        # Mark training as done (before upload) - helps diagnose where it died
                        if refit_run_id:
                            try:
                                client = mlflow.tracking.MlflowClient()
                                client.set_tag(
                                    refit_run_id, "code.refit_training_done", "true")
                            except Exception as tag_error:
                                logger.debug(
                                    f"Could not set refit_training_done tag: {tag_error}")

                        # Link back from parent: add tags and metrics pointing to refit run
                        if parent_run_id and refit_run_id:
                            try:
                                client = mlflow.tracking.MlflowClient()
                                # Add refit run ID tag to parent
                                client.set_tag(
                                    parent_run_id, "code.refit_run_id", refit_run_id)
                                client.set_tag(
                                    parent_run_id, "code.refit_completed", "true")

                                # Copy headline refit metrics to parent for quick browsing
                                if refit_metrics and objective_metric in refit_metrics:
                                    try:
                                        metric_value = refit_metrics[objective_metric]
                                        if isinstance(metric_value, (int, float)):
                                            client.log_metric(
                                                parent_run_id,
                                                f"refit_{objective_metric}",
                                                metric_value
                                            )
                                    except Exception as metric_error:
                                        logger.debug(
                                            f"Could not log refit metric to parent: {metric_error}")
                            except Exception as link_error:
                                logger.warning(
                                    f"Could not link refit run to parent: {link_error}")
                    except Exception as refit_error:
                        refit_ok = False
                        logger.warning(
                            f"[REFIT] Refit training failed: {refit_error}. Continuing without refit checkpoint.",
                            exc_info=True
                        )

                    # Upload checkpoint after refit (prefer refit checkpoint, fallback to CV checkpoint)
                    # Upload to refit run (not parent) since refit checkpoint belongs to refit run
                    # Only log if mlflow.log_best_checkpoint is enabled
                    log_best_checkpoint = hpo_config.get("mlflow", {}).get("log_best_checkpoint", False)
                    if log_best_checkpoint and (refit_run_id or parent_run_id):
                        upload_succeeded = False
                        upload_error = None
                        try:
                            tracker.log_best_checkpoint(
                                study=study,
                                hpo_output_dir=output_dir,
                                backbone=backbone,
                                run_id=run_id,
                                fold_splits=fold_splits_for_logging if 'fold_splits_for_logging' in locals() else None,
                                prefer_checkpoint_dir=refit_checkpoint_dir,
                                refit_ok=refit_ok,
                                parent_run_id=parent_run_id,
                                refit_run_id=refit_run_id,  # Upload to refit run
                            )
                            upload_succeeded = True
                        except Exception as e:
                            upload_error = e
                            logger.error(
                                f"[HPO] Error in log_best_checkpoint: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                    else:
                        if not log_best_checkpoint:
                            logger.info(
                                "[HPO] Skipping checkpoint logging (mlflow.log_best_checkpoint=false or not set)")
                        upload_succeeded = True  # Skipped intentionally, not a failure
                        upload_error = None

                    # Mark refit run as FINISHED (success) or FAILED (error) after upload attempt
                    # Only terminate if it's still RUNNING (safety check)
                    # This must run regardless of whether log_best_checkpoint was enabled
                    if refit_run_id:
                        try:
                            client = mlflow.tracking.MlflowClient()
                            # Check current status before terminating
                            run = client.get_run(refit_run_id)
                            current_status = run.info.status
                            logger.info(
                                f"[REFIT] Refit run {refit_run_id[:12]}... current status: {current_status}, upload_succeeded: {upload_succeeded}")
                            if current_status == "RUNNING":
                                if upload_succeeded:
                                    # Success: mark as FINISHED with tag
                                    client.set_tag(
                                        refit_run_id, "code.refit_artifacts_uploaded", "true")
                                    client.set_terminated(
                                        refit_run_id, status="FINISHED")
                                    logger.info(
                                        f"[REFIT] ✓ Artifacts uploaded and run marked as FINISHED: {refit_run_id[:12]}...")
                                else:
                                    # Failure: mark as FAILED with error tag
                                    client.set_tag(
                                        refit_run_id, "code.refit_artifacts_uploaded", "false")
                                    error_msg = str(upload_error)[
                                        :200] if upload_error else "Unknown error"
                                    client.set_tag(
                                        refit_run_id, "code.refit_error", error_msg)
                                    client.set_terminated(
                                        refit_run_id, status="FAILED")
                                    logger.warning(
                                        f"[REFIT] Artifact upload failed, run marked as FAILED: {refit_run_id[:12]}... (error: {error_msg})")
                            else:
                                # Run already terminated (shouldn't happen, but handle gracefully)
                                logger.info(
                                    f"[REFIT] Refit run {refit_run_id[:12]}... already has status {current_status}, skipping termination. Expected RUNNING.")
                        except Exception as term_error:
                            logger.error(
                                f"[REFIT] Could not mark refit run status: {term_error}", exc_info=True)
                            # Try one more time with a simpler approach
                            try:
                                client = mlflow.tracking.MlflowClient()
                                client.set_terminated(refit_run_id, status="FINISHED")
                                logger.info(f"[REFIT] Fallback: Marked refit run {refit_run_id[:12]}... as FINISHED")
                            except Exception as fallback_error:
                                logger.error(
                                    f"[REFIT] Fallback termination also failed: {fallback_error}")
                    else:
                        logger.warning("[REFIT] No refit_run_id available to mark as FINISHED")
                except Exception as e:
                    logger.warning(
                        f"[REFIT] Refit training failed: {e}. Continuing without refit checkpoint.",
                        exc_info=True
                    )
                    # Ensure refit run is marked as FINISHED even if there was an error
                    if refit_run_id:
                        try:
                            client = mlflow.tracking.MlflowClient()
                            run = client.get_run(refit_run_id)
                            if run.info.status == "RUNNING":
                                client.set_terminated(refit_run_id, status="FINISHED")
                                logger.info(f"[REFIT] Marked refit run {refit_run_id[:12]}... as FINISHED after error")
                        except Exception as term_error2:
                            logger.error(f"[REFIT] Could not mark refit run as FINISHED after error: {term_error2}")
            elif not refit_enabled:
                logger.info("[REFIT] Refit training is disabled in config")
            elif study.best_trial is None:
                logger.warning(
                    "[REFIT] No best trial found, skipping refit training")

            # Run final cleanup AFTER refit training completes (if it ran)
            # This ensures cleanup happens in the normal MLflow flow, not just in exception handler
            try:
                cleanup_checkpoints()
            except Exception as e:
                logger.warning(
                    f"Error during final checkpoint cleanup: {e}"
                )

    except Exception as e:
        logger.warning(f"MLflow tracking failed: {e}")
        logger.warning("Continuing HPO without MLflow tracking...")

        trial_callback = create_trial_callback(objective_metric, None)

        study.optimize(
            objective,
            n_trials=max_trials,
            timeout=timeout_seconds,
            show_progress_bar=True,
            callbacks=[trial_callback],
        )

        try:
            cleanup_checkpoints()
        except Exception as e:
            logger.warning(
                f"Error during final checkpoint cleanup: {e}"
            )

    return study
