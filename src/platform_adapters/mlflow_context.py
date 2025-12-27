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

        # Set tracking URI from environment variable if provided (CRITICAL for subprocesses)
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
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

        # Set experiment from environment variable if provided
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            print(
                f"  [MLflow Context] Set experiment from env: {experiment_name}", file=sys.stderr, flush=True)
            print(
                f"  [MLflow Context] Set experiment from env: {experiment_name}", flush=True)

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

        for var_name, var_value in [
            ("MLFLOW_CHILD_RUN_ID", mlflow_child),
            ("MLFLOW_PARENT_RUN_ID", mlflow_parent),
            ("MLFLOW_TRIAL_NUMBER", mlflow_trial),
            ("MLFLOW_EXPERIMENT_NAME", mlflow_experiment),
            ("MLFLOW_TRACKING_URI", mlflow_tracking_uri),
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
            # Create a child run explicitly using the tracking client
            # This works even in subprocesses since we have the parent_run_id

            @contextmanager
            def child_run_context():
                client = mlflow.tracking.MlflowClient()
                tracking_uri = mlflow.get_tracking_uri()
                is_azure_ml = tracking_uri and "azureml" in tracking_uri.lower()

                # Get experiment ID
                if experiment_name:
                    experiment = mlflow.get_experiment_by_name(experiment_name)
                    experiment_id = experiment.experiment_id if experiment else None
                else:
                    # Try to get from active run if available
                    active_run = mlflow.active_run()
                    experiment_id = active_run.info.experiment_id if active_run else None

                if not experiment_id:
                    # Fallback: create independent run if we can't determine experiment
                    print(
                        f"  [MLflow] Warning: Could not determine experiment ID, creating independent run")
                    with mlflow.start_run(run_name=f"trial_{trial_number}") as run:
                        yield run
                    return

                # Verify parent run exists and get its info
                # CRITICAL: Use the parent run's experiment ID to ensure they're in the same experiment
                try:
                    parent_run_info = client.get_run(parent_run_id)
                    print(
                        f"  [MLflow] ✓ Verified parent run exists: {parent_run_info.info.run_id[:12]}...")
                    print(
                        f"  [MLflow]   Parent run status: {parent_run_info.info.status}")
                    print(
                        f"  [MLflow]   Parent run experiment: {parent_run_info.info.experiment_id}")
                    # CRITICAL: Use the parent run's experiment ID to ensure they're in the same experiment
                    # This is especially important for Azure ML Workspace
                    if parent_run_info.info.experiment_id:
                        experiment_id = parent_run_info.info.experiment_id
                        print(
                            f"  [MLflow]   Using parent's experiment ID: {experiment_id}")
                    else:
                        print(
                            f"  [MLflow]   ⚠ Warning: Parent run has no experiment ID")
                except Exception as e:
                    print(
                        f"  [MLflow] ⚠ Warning: Could not verify parent run: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue anyway - might still work

                # Create a child run with the parent_run_id tag
                # MLflow uses the mlflow.parentRunId tag to establish parent-child relationships
                # For Azure ML Workspace, we need to ensure the tag is set correctly
                run_name = f"trial_{trial_number}"

                # IMPORTANT: For Azure ML Workspace, try using MLflow's start_run with nested=True
                # But since we're in a subprocess, we need to use the tracking client
                # The mlflow.parentRunId tag is the standard way, but Azure ML might need special handling

                # Try creating the run with the parent tag
                # For Azure ML Workspace, ensure we're using the correct experiment ID from parent
                try:
                    # Create run with parent tag - this should establish the relationship
                    # Use the parent's experiment ID to ensure they're in the same experiment
                    run = client.create_run(
                        experiment_id=experiment_id,
                        tags={"mlflow.parentRunId": parent_run_id},
                        run_name=run_name
                    )
                    print(
                        f"  [MLflow] ✓ Created child run: {run.info.run_id[:12]}...")
                    print(
                        f"  [MLflow]   Child run experiment: {run.info.experiment_id}")
                    print(
                        f"  [MLflow]   Parent run ID in tags: {parent_run_id[:12]}...")

                    # For Azure ML, also try setting additional tags that might help
                    if is_azure_ml:
                        # Set additional Azure ML specific tags if needed
                        client.set_tag(run.info.run_id,
                                       "mlflow.runName", run_name)
                        print(f"  [MLflow]   Set additional Azure ML tags")
                except Exception as e:
                    print(
                        f"  [MLflow] ⚠ Error creating child run with tag: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback: create run without tag, then set it
                    run = client.create_run(
                        experiment_id=experiment_id,
                        run_name=run_name
                    )
                    # Set parent tag after creation
                    client.set_tag(run.info.run_id,
                                   "mlflow.parentRunId", parent_run_id)
                    print(
                        f"  [MLflow] ✓ Created child run and set parent tag: {run.info.run_id[:12]}...")

                # Start the run using the created run_id
                # The parent-child relationship should be established via the tag
                try:
                    mlflow.start_run(run_id=run.info.run_id)
                    print(
                        f"  [MLflow] ✓ Started child run: {run.info.run_id[:12]}... (parent: {parent_run_id[:12]}...)")
                except Exception as e:
                    print(f"  [MLflow] ⚠ Error starting child run: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback: create new run
                    with mlflow.start_run(run_name=run_name) as fallback_run:
                        yield fallback_run
                    return

                # Verify the parent tag is set and the relationship is established
                try:
                    current_run = mlflow.active_run()
                    if current_run and current_run.info.run_id == run.info.run_id:
                        # Get the run info to verify tags
                        run_info = client.get_run(run.info.run_id)
                        parent_tag = run_info.data.tags.get(
                            "mlflow.parentRunId")
                        if parent_tag == parent_run_id:
                            print(
                                f"  [MLflow] ✓ Verified parent-child relationship established")
                            if is_azure_ml:
                                print(
                                    f"  [MLflow]   Using Azure ML Workspace - parent-child should appear in UI")
                        else:
                            print(
                                f"  [MLflow] ⚠ Warning: Parent tag mismatch!")
                            print(
                                f"  [MLflow]     Expected: {parent_run_id[:12]}...")
                            print(
                                f"  [MLflow]     Got: {parent_tag[:12] if parent_tag else 'None'}...")
                            # Try to fix it
                            client.set_tag(run.info.run_id,
                                           "mlflow.parentRunId", parent_run_id)
                            print(f"  [MLflow]   Re-set parent tag")
                except Exception as e:
                    # Log but don't fail - the tag was set during creation
                    print(
                        f"  [MLflow] ⚠ Warning: Could not verify parent run tag: {e}")
                    import traceback
                    traceback.print_exc()
                try:
                    yield
                finally:
                    mlflow.end_run()
                    print(
                        f"  [MLflow] ✓ Ended child run: {run.info.run_id[:12]}...")

            return child_run_context()
        else:
            # Create an independent run
            return mlflow.start_run()
