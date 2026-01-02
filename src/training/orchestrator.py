
"""Training orchestration logic."""

import os
import argparse
from pathlib import Path

from training.config import build_training_config, resolve_distributed_config
from training.data import load_dataset
from training.trainer import train_model
from training.logging import log_metrics
from training.utils import set_seed
from training.distributed import (
    create_run_context,
    init_process_group_if_needed,
)
from platform_adapters import get_platform_adapter
from shared.argument_parsing import validate_config_dir


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
    import mlflow
    import sys

    # Set tracking URI from environment variable (CRITICAL for subprocesses)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(
            f"  [Training] Set MLflow tracking URI: {tracking_uri[:50]}...", file=sys.stderr, flush=True)

    # Set experiment from environment variable
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
        print(
            f"  [Training] Set MLflow experiment: {experiment_name}", file=sys.stderr, flush=True)

    # Check if we should use an existing run (for refit) or create a child run (for HPO trials)
    use_run_id = os.environ.get("MLFLOW_RUN_ID") or os.environ.get("MLFLOW_USE_RUN_ID")
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
    trial_number = os.environ.get("MLFLOW_TRIAL_NUMBER", "unknown")
    fold_idx = os.environ.get("MLFLOW_FOLD_IDX")
    
    # Track whether we started a run directly (needed for cleanup)
    started_run_directly = False

    if use_run_id:
        # Use existing run directly (for refit training)
        # NEVER create a child - log directly to the specified run
        print(
            f"  [Training] Using existing run: {use_run_id[:12]}... (refit mode)", 
            file=sys.stderr, flush=True
        )
        try:
            # Ensure no active run before starting (avoid "Run is already active" errors)
            if mlflow.active_run():
                mlflow.end_run()
            mlflow.start_run(run_id=use_run_id)
            started_run_directly = True
            print(f"  [Training] ✓ Started existing run", file=sys.stderr, flush=True)
        except Exception as e:
            print(
                f"  [Training] Error starting existing run: {e}", file=sys.stderr, flush=True
            )
            import traceback
            traceback.print_exc()
            # Fallback to independent run
            mlflow.start_run(run_name=f"refit_fallback_{use_run_id[:8]}")
            started_run_directly = True
    elif parent_run_id:
        # Construct unique run name: include fold index if k-fold CV is enabled
        if fold_idx is not None:
            run_name = f"trial_{trial_number}_fold{fold_idx}"
            trial_display = f"trial {trial_number}, fold {fold_idx}"
        else:
            run_name = f"trial_{trial_number}"
            trial_display = f"trial {trial_number}"
        
        print(
            f"  [Training] Creating child run with parent: {parent_run_id[:12]}... ({trial_display})", file=sys.stderr, flush=True)
        # Create child run explicitly using tracking client
        client = mlflow.tracking.MlflowClient()

        # Get experiment ID
        if experiment_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id if experiment else None
        else:
            experiment_id = None

        if not experiment_id:
            # Get from parent run
            try:
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
        if started_run_directly:
            mlflow.end_run()
            if use_run_id:
                print(f"  [Training] Ended existing run (refit mode)", file=sys.stderr, flush=True)
            elif parent_run_id:
                print(f"  [Training] Ended child run", file=sys.stderr, flush=True)
            else:
                print(f"  [Training] Ended independent run", file=sys.stderr, flush=True)
        else:
            # Use context manager's exit
            context_mgr.__exit__(None, None, None)
