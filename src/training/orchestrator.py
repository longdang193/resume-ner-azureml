"""
@meta
name: training_orchestrator
type: script
domain: training
responsibility:
  - Orchestrate training execution
  - Set up MLflow tracking
  - Handle distributed training context
  - Manage training run lifecycle
inputs:
  - Training configuration
  - Training arguments
outputs:
  - Trained model checkpoint
  - Training metrics (via MLflow)
tags:
  - orchestration
  - training
  - mlflow
ci:
  runnable: true
  needs_gpu: true
  needs_cloud: false
lifecycle:
  status: active
"""

"""Training orchestration logic."""

import os
import argparse
from pathlib import Path

from training.config import build_training_config, resolve_distributed_config
from data.loaders import load_dataset
from training.trainer import train_model
from training.logging import log_metrics
from training.utils import set_seed
from training.distributed import (
    create_run_context,
    init_process_group_if_needed,
)
from infrastructure.platform.adapters import get_platform_adapter
from common.shared.argument_parsing import validate_config_dir


def log_training_parameters(config: dict, logging_adapter) -> None:
    """Log training parameters using platform adapter."""
    params = {
        "learning_rate": config["training"].get("learning_rate"),
        "batch_size": config["training"].get("batch_size"),
        "dropout": config["model"].get("dropout"),
        "weight_decay": config["training"].get("weight_decay"),
        "epochs": config["training"].get("epochs"),
        "backbone": config["model"].get("backbone"),
    }
    logging_adapter.log_params(
        {k: v for k, v in params.items() if v is not None})


def run_training(args: argparse.Namespace, prebuilt_config: dict | None = None) -> None:
    """
    Run a single training process (rank-agnostic).

    This function is used for both single-process training and each rank in
    a DDP run. It is intentionally unaware of world_size; DDP setup is
    handled via `training.distributed`.

    Args:
        args: Parsed command-line arguments.
        prebuilt_config: Optional pre-built configuration dictionary.
    """
    config_dir = validate_config_dir(args.config_dir)

    config = prebuilt_config or build_training_config(args, config_dir)

    # Optionally offset random seed by rank in distributed runs.
    rank_env = os.getenv("RANK")
    if rank_env is not None and "training" in config:
        try:
            rank = int(rank_env)
        except ValueError:
            rank = 0
        base_seed = config["training"].get("random_seed")
        if base_seed is not None:
            config["training"]["random_seed"] = int(base_seed) + rank

    # Resolve distributed config, create run context, and initialize process
    # group if needed (DDP). Single-process runs will get a SingleProcessContext.
    dist_cfg = resolve_distributed_config(config)
    context = create_run_context(dist_cfg)
    init_process_group_if_needed(context)

    seed = config["training"].get("random_seed")
    set_seed(seed)

    dataset = load_dataset(args.data_asset)

    # Get platform adapter for output paths, logging, and MLflow context
    platform_adapter = get_platform_adapter(
        default_output_dir=Path("./outputs"))
    output_resolver = platform_adapter.get_output_path_resolver()
    logging_adapter = platform_adapter.get_logging_adapter()
    mlflow_context = platform_adapter.get_mlflow_context_manager()

    # Resolve output directory using platform adapter
    output_dir = output_resolver.resolve_output_path(
        "checkpoint", default=Path("./outputs")
    )
    output_dir = output_resolver.ensure_output_directory(output_dir)

    # CRITICAL: Set up MLflow BEFORE using context manager
    # This ensures tracking URI and experiment are set, and child runs are created correctly
    import sys
    
    # Import azureml.mlflow early to register the 'azureml' URI scheme before MLflow initializes
    # This must happen before mlflow is imported to ensure the scheme is registered
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri and "azureml" in tracking_uri.lower():
        try:
            import azureml.mlflow  # noqa: F401
        except ImportError:
            # If azureml.mlflow is not available, fallback to local tracking
            # This is expected in some environments and the code handles it gracefully
            print(
                "  [Training] INFO: azureml.mlflow not available, but Azure ML URI detected. "
                "Falling back to local tracking. (This is normal if azureml-mlflow is not installed)",
                file=sys.stderr, flush=True)
            # Override with local tracking URI
            from common.shared.mlflow_setup import _get_local_tracking_uri
            tracking_uri = _get_local_tracking_uri()
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            # Clear Azure ML run IDs - they won't exist in local SQLite database
            # This forces creation of a new run in local tracking
            if "MLFLOW_RUN_ID" in os.environ:
                old_run_id = os.environ.pop("MLFLOW_RUN_ID")
                print(
                    f"  [Training] Cleared Azure ML run ID {old_run_id[:12]}... (will create new run in local tracking)",
                    file=sys.stderr, flush=True)
            if "MLFLOW_USE_RUN_ID" in os.environ:
                os.environ.pop("MLFLOW_USE_RUN_ID")
    
    import mlflow
    
    # Set tracking URI from environment variable (CRITICAL for subprocesses)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(
            f"  [Training] Set MLflow tracking URI: {tracking_uri[:50]}...", file=sys.stderr, flush=True)

        # Set Azure ML artifact upload timeout if using Azure ML and not already set
        if "azureml" in tracking_uri.lower():
            if "AZUREML_ARTIFACTS_DEFAULT_TIMEOUT" not in os.environ:
                os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "600"
                print(
                    f"  [Training] Set AZUREML_ARTIFACTS_DEFAULT_TIMEOUT=600 for artifact uploads",
                    file=sys.stderr, flush=True)

    # Set experiment from environment variable
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
        print(
            f"  [Training] Set MLflow experiment: {experiment_name}", file=sys.stderr, flush=True)

    # Check if we should use an existing run (for refit) or create a child run (for HPO trials)
    use_run_id = os.environ.get(
        "MLFLOW_RUN_ID") or os.environ.get("MLFLOW_USE_RUN_ID")
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
    trial_number = os.environ.get("MLFLOW_TRIAL_NUMBER", "unknown")
    fold_idx = os.environ.get("MLFLOW_FOLD_IDX")

    # Track whether we started a run directly (needed for cleanup)
    started_run_directly = False
    # Track if we started an existing run (refit mode) - don't end it here
    started_existing = False

    if use_run_id:
        # Check if this is final training (no parent_run_id) vs refit mode (has parent_run_id)
        # For final training: start run actively so artifacts can be logged
        # For refit mode: don't start run (keep it RUNNING for parent to manage)
        is_final_training = parent_run_id is None

        if is_final_training:
            # Final training: start the run actively so artifacts can be logged
            print(
                f"  [Training] Using existing run: {use_run_id[:12]}... (final training)",
                file=sys.stderr,
                flush=True,
            )
            mlflow.start_run(run_id=use_run_id)
            started_run_directly = True
            started_existing = False  # Not refit mode - we'll end the run normally
            print(
                "  [Training] ✓ Started run for artifact logging",
                file=sys.stderr,
                flush=True,
            )
        else:
            # Refit mode: don't start an active run context - use client API instead
            # This prevents MLflow from auto-ending the run when subprocess exits
            print(
                f"  [Training] Using existing run: {use_run_id[:12]}... (refit mode)",
                file=sys.stderr,
                flush=True,
            )
            # Don't start an active run - we'll log via client API instead
            # This keeps the run RUNNING until parent process explicitly terminates it
            started_run_directly = False  # Not using active run
            started_existing = True       # Mark as existing run (refit mode)
            print(
                "  [Training] ✓ Will log to existing run via client API (run stays RUNNING)",
                file=sys.stderr,
                flush=True,
            )

    elif parent_run_id:
        # Try to build systematic run name using naming policy
        run_name = None
        trial_display = f"trial {trial_number}"
        if fold_idx is not None:
            trial_display = f"trial {trial_number}, fold {fold_idx}"

        # Set environment variables for potential use by mlflow_context fallback
        os.environ["MLFLOW_TRIAL_NUMBER"] = str(trial_number)
        if fold_idx is not None:
            os.environ["MLFLOW_FOLD_IDX"] = str(fold_idx)

        # Initialize variables for fallback use
        study_key_hash = None
        model = None

        # Try to build systematic name using naming policy
        try:
            from infrastructure.naming import create_naming_context
            from infrastructure.tracking.mlflow.naming import build_mlflow_run_name
            from common.shared.platform_detection import detect_platform

            # Try to get study_key_hash and model from parent run
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                parent_run = client.get_run(parent_run_id)
                study_key_hash = parent_run.data.tags.get(
                    "code.study_key_hash")
                model = parent_run.data.tags.get("code.model")
            except Exception:
                pass

            # Infer config_dir from environment or current directory
            config_dir = Path(os.environ.get(
                "CONFIG_DIR", Path.cwd() / "config"))

            # Determine process type: hpo_trial_fold if fold_idx, otherwise hpo_trial
            process_type = "hpo_trial_fold" if fold_idx is not None else "hpo_trial"

            # Create context if we have minimum required info
            if model or study_key_hash:
                fold_context = create_naming_context(
                    process_type="hpo",
                    model=model or "unknown",
                    environment=detect_platform(),
                    trial_id=f"trial_{trial_number}",
                    trial_number=trial_number,
                    fold_idx=fold_idx,
                    study_key_hash=study_key_hash,
                )
                run_name = build_mlflow_run_name(
                    fold_context, config_dir=config_dir)
        except Exception as e:
            # If systematic naming fails, will use fallback below
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                f"Could not build systematic run name: {e}, using fallback")

        # Fallback to policy-like deterministic format if systematic naming didn't work
        if not run_name:
            from common.shared.platform_detection import detect_platform
            env = detect_platform()
            model_name = model or "unknown"

            # Try to get study_hash short version
            study_hash_short = "unknown"
            if study_key_hash:
                study_hash_short = study_key_hash[:8]

            # Build policy-like name
            if fold_idx is not None:
                # Use hpo_trial_fold pattern
                trial_num_str = f"t{str(trial_number).zfill(2)}"
                run_name = f"{env}_{model_name}_hpo_trial_study-{study_hash_short}_{trial_num_str}_fold{fold_idx}"
            else:
                # Use hpo_trial pattern
                trial_num_str = f"t{str(trial_number).zfill(2)}"
                run_name = f"{env}_{model_name}_hpo_trial_study-{study_hash_short}_{trial_num_str}"

        print(
            f"  [Training] Creating child run with parent: {parent_run_id[:12]}... ({trial_display})",
            file=sys.stderr,
            flush=True,
        )

        # Get experiment ID from environment or parent run
        experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
        if not experiment_id:
            # Get from parent run
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                parent_run_info = client.get_run(parent_run_id)
                experiment_id = parent_run_info.info.experiment_id
                print(
                    f"  [Training] Using parent's experiment ID: {experiment_id}", file=sys.stderr, flush=True)
            except Exception as e:
                print(
                    f"  [Training] Warning: Could not get parent run info: {e}", file=sys.stderr, flush=True)

        if experiment_id:
            # Create child run with parent tag and Azure ML-specific tags
            # These tags help Azure ML UI recognize this as a trial and display metrics/parameters
            tags = {
                "mlflow.parentRunId": parent_run_id,
                "azureml.runType": "trial",  # Mark as trial for Azure ML UI
                "azureml.trial": "true",  # Azure ML-specific tag for trials
                # Store trial number as tag
                "trial_number": str(trial_number),
            }
            # Add fold index to tags if k-fold CV is enabled
            if fold_idx is not None:
                tags["fold_idx"] = str(fold_idx)

            # CRITICAL: Set mlflow.runName tag (required for proper run name display in Azure ML)
            # Without this, Azure ML may show "trial_unknown" instead of the actual run name
            tags["mlflow.runName"] = run_name

            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                run = client.create_run(
                    experiment_id=experiment_id,
                    tags=tags,
                    run_name=run_name
                )
                print(
                    f"  [Training] ✓ Created child run: {run.info.run_id[:12]}...", file=sys.stderr, flush=True)
                # Use this child run instead of creating a new one
                mlflow.start_run(run_id=run.info.run_id)
                started_run_directly = True
                print(f"  [Training] ✓ Started child run",
                      file=sys.stderr, flush=True)
            except Exception as e:
                print(
                    f"  [Training] Error creating child run: {e}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc()
                # Fallback to independent run
                mlflow.start_run(run_name=run_name)
                started_run_directly = True
        else:
            # Fallback to independent run
            mlflow.start_run(run_name=run_name)
            started_run_directly = True
    else:
        # No parent run ID - use context manager as normal
        context_mgr = mlflow_context.get_context()
        context_mgr.__enter__()

    try:
        if context.is_main_process():
            log_training_parameters(config, logging_adapter)
        metrics = train_model(config, dataset, output_dir, context=context)
        if context.is_main_process():
            log_metrics(output_dir, metrics, logging_adapter)
    finally:
        # End the run if we started it directly
        # EXCEPTION: For refit runs (started_existing=True), we don't have an active run
        # so there's nothing to end - the parent process will mark it FINISHED after artifact upload
        if started_existing:
            # Refit mode: No active run to end - run stays RUNNING until parent terminates it
            print(f"  [Training] Refit run remains RUNNING (will be marked FINISHED after artifacts)",
                  file=sys.stderr, flush=True)
        elif started_run_directly:
            # Non-refit mode: End the run normally
            mlflow.end_run()
            if parent_run_id:
                print(f"  [Training] Ended child run",
                      file=sys.stderr, flush=True)
            else:
                print(f"  [Training] Ended independent run",
                      file=sys.stderr, flush=True)
        else:
            # Use context manager's exit
            context_mgr.__exit__(None, None, None)
