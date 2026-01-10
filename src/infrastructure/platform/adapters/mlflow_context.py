"""
@meta
name: platform_mlflow_context
type: utility
domain: platform_adapters
responsibility:
  - Manage MLflow context for different platforms
  - Handle Azure ML and local MLflow run lifecycle
inputs:
  - Platform identifiers
outputs:
  - MLflow context managers
tags:
  - utility
  - platform_adapters
  - mlflow
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""MLflow context management for different platforms."""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any


class MLflowContextManager(ABC):
    """Abstract interface for MLflow context management."""

    @abstractmethod
    def get_context(self) -> AbstractContextManager[Any]:
        """
        Get the MLflow context manager for this platform.

        Returns:
            Context manager that handles MLflow run lifecycle.
        """
        pass


class AzureMLMLflowContextManager(MLflowContextManager):
    """MLflow context manager for Azure ML jobs.

    Azure ML automatically creates an MLflow run context for each job.
    We should NOT call mlflow.start_run() when running in Azure ML, as it creates
    a nested/separate run, causing metrics to be logged to the wrong run.
    """

    def get_context(self) -> AbstractContextManager[Any]:
        """Return a no-op context manager (Azure ML handles MLflow automatically)."""
        from contextlib import nullcontext
        return nullcontext()


class LocalMLflowContextManager(MLflowContextManager):
    """MLflow context manager for local execution."""

    def get_context(self) -> AbstractContextManager[Any]:
        """Return MLflow start_run context manager for local execution."""
        import mlflow
        import os
        import sys
        from contextlib import contextmanager

        # CRITICAL DEBUG: Print immediately to confirm this method is being called
        print("=" * 80, file=sys.stderr, flush=True)
        print("  [MLflow Context Manager] get_context() CALLED",
              file=sys.stderr, flush=True)
        print("=" * 80, file=sys.stderr, flush=True)
        print("  [MLflow Context Manager] get_context() CALLED", flush=True)

        # Set up MLflow from environment variables (CRITICAL for subprocesses)
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")

        if tracking_uri or experiment_name:
            from shared.mlflow_setup import setup_mlflow_cross_platform
            try:
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                    print(
                        f"  [MLflow Context] Set tracking URI from env: {tracking_uri[:50]}...", file=sys.stderr, flush=True)
                    print(
                        f"  [MLflow Context] Set tracking URI from env: {tracking_uri[:50]}...", flush=True)
                else:
                    print(f"  [MLflow Context] WARNING: MLFLOW_TRACKING_URI not set in environment!",
                          file=sys.stderr, flush=True)
                    print(
                        f"  [MLflow Context] WARNING: MLFLOW_TRACKING_URI not set in environment!", flush=True)

                if experiment_name:
                    setup_mlflow_cross_platform(
                        experiment_name=experiment_name,
                        ml_client=None,  # Will use local tracking or env vars
                        fallback_to_local=True,
                    )
                    print(
                        f"  [MLflow Context] Set experiment from env: {experiment_name}", file=sys.stderr, flush=True)
                    print(
                        f"  [MLflow Context] Set experiment from env: {experiment_name}", flush=True)
            except Exception as e:
                print(
                    f"  [MLflow Context] Warning: Could not set up MLflow: {e}", file=sys.stderr, flush=True)

        # Debug: Print all MLflow-related environment variables
        mlflow_child = os.environ.get("MLFLOW_CHILD_RUN_ID")
        mlflow_parent = os.environ.get("MLFLOW_PARENT_RUN_ID")
        mlflow_trial = os.environ.get("MLFLOW_TRIAL_NUMBER")
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

        # Always print MLflow environment variables for debugging
        # Print to both stdout and stderr to ensure visibility
        import sys
        debug_msg = f"  [MLflow Context] Environment variables:"
        print(debug_msg, file=sys.stderr, flush=True)
        print(debug_msg, flush=True)  # Also print to stdout

        mlflow_run_name = os.environ.get("MLFLOW_RUN_NAME")
        
        for var_name, var_value in [
            ("MLFLOW_CHILD_RUN_ID", mlflow_child),
            ("MLFLOW_PARENT_RUN_ID", mlflow_parent),
            ("MLFLOW_TRIAL_NUMBER", mlflow_trial),
            ("MLFLOW_EXPERIMENT_NAME", mlflow_experiment),
            ("MLFLOW_TRACKING_URI", mlflow_tracking_uri),
            ("MLFLOW_RUN_NAME", mlflow_run_name),
        ]:
            if var_name == "MLFLOW_TRACKING_URI" and var_value:
                display_value = f"{var_value[:50]}..."
            elif var_name in ["MLFLOW_CHILD_RUN_ID", "MLFLOW_PARENT_RUN_ID"] and var_value:
                display_value = f"{var_value[:12]}..."
            else:
                display_value = var_value if var_value else 'None'

            msg = f"    {var_name}: {display_value}"
            print(msg, file=sys.stderr, flush=True)
            print(msg, flush=True)  # Also print to stdout

        # Check if we should use an existing child run ID (created in parent process)
        # This is the preferred method - child run was created with nested=True in parent
        child_run_id = os.environ.get("MLFLOW_CHILD_RUN_ID")
        if child_run_id:
            # Use the existing child run that was created in the parent process
            # This ensures proper parent-child relationship with Azure ML Workspace
            print(f"  [MLflow] Resuming child run: {child_run_id[:8]}...")

            @contextmanager
            def existing_child_run_context():
                # Resume the child run - MLflow allows resuming ended runs
                mlflow.start_run(run_id=child_run_id)
                try:
                    yield
                finally:
                    mlflow.end_run()
            return existing_child_run_context()

        # Check if we should create a nested child run (for HPO trials)
        parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
        trial_number = os.environ.get("MLFLOW_TRIAL_NUMBER", "unknown")
        if parent_run_id:
            print(
                f"  [MLflow] Creating child run with parent: {parent_run_id[:12]}... (trial {trial_number})")
            # Use shared child run creation function
            from orchestration.jobs.tracking.mlflow_helpers import create_child_run  # TODO: Move to tracking.mlflow
            return create_child_run(
                parent_run_id=parent_run_id,
                trial_number=trial_number,
                experiment_name=experiment_name,
            )
        else:
            # Create an independent run
            # Check for custom run name from environment variable
            run_name = os.environ.get("MLFLOW_RUN_NAME")
            if run_name:
                print(f"  [MLflow] Creating run with name: {run_name}", file=sys.stderr, flush=True)
                print(f"  [MLflow] Creating run with name: {run_name}", flush=True)
                # Use MLflow client to create run with explicit name
                # This ensures the run name is set correctly
                client = mlflow.tracking.MlflowClient()
                experiment_id = None
                if experiment_name:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    if experiment:
                        experiment_id = experiment.experiment_id
                
                if experiment_id:
                    # Create run with explicit name using client
                    run = client.create_run(
                        experiment_id=experiment_id,
                        run_name=run_name
                    )
                    print(f"  [MLflow] Created run: {run.info.run_id[:12]}... with name: {run_name}", file=sys.stderr, flush=True)
                    # Use context manager to start the created run
                    @contextmanager
                    def named_run_context():
                        mlflow.start_run(run_id=run.info.run_id)
                        try:
                            yield
                        finally:
                            mlflow.end_run()
                    return named_run_context()
                else:
                    # Fallback to standard start_run with name
                    return mlflow.start_run(run_name=run_name)
            else:
                # Deterministic fallback — avoid MLflow auto-generated names
                fallback_bits = []
                
                # Try to get process type
                process_type = os.environ.get("PROCESS_TYPE", "run")
                fallback_bits.append(process_type)
                
                # Try to get run ID (shortened)
                run_id = os.environ.get("MLFLOW_RUN_ID", "")
                if run_id:
                    fallback_bits.append(run_id[:8])
                else:
                    # Try to get from other sources
                    run_id_alt = os.environ.get("MLFLOW_CHILD_RUN_ID") or os.environ.get("MLFLOW_PARENT_RUN_ID")
                    if run_id_alt:
                        fallback_bits.append(run_id_alt[:8])
                    else:
                        fallback_bits.append("noid")
                
                # Try to get trial number
                trial_number = os.environ.get("MLFLOW_TRIAL_NUMBER")
                if trial_number:
                    fallback_bits.append(f"t{trial_number}")
                
                # Try to get fold index
                fold_idx = os.environ.get("MLFLOW_FOLD_IDX")
                if fold_idx:
                    fallback_bits.append(f"fold{fold_idx}")
                
                # Build deterministic fallback name
                run_name = "_".join(fallback_bits)
                
                # Sanitize: replace problematic characters
                run_name = run_name.replace("/", "_").replace("\\", "_").replace(":", "_")
                
                print(f"  [MLflow] No MLFLOW_RUN_NAME set, using fallback name: {run_name}", file=sys.stderr, flush=True)
                print(f"  [MLflow] No MLFLOW_RUN_NAME set, using fallback name: {run_name}", flush=True)
            
            return mlflow.start_run(run_name=run_name)
