"""Shared MLflow utilities for child run creation.

Consolidates child run creation logic from mlflow_context.py and training/orchestrator.py.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Optional

import mlflow
from shared.logging_utils import get_logger

logger = get_logger(__name__)


@contextmanager
def create_child_run(
    parent_run_id: str,
    trial_number: str,
    experiment_name: Optional[str] = None,
    additional_tags: Optional[dict[str, str]] = None,
) -> Any:
    """
    Create and start a child MLflow run for HPO trials.

    This function consolidates the child run creation logic that was duplicated
    between mlflow_context.py and training/orchestrator.py.

    Args:
        parent_run_id: ID of the parent MLflow run.
        trial_number: Trial number identifier (e.g., "0", "1", "unknown").
        experiment_name: Optional experiment name. If not provided, will try to get from parent run.
        additional_tags: Optional additional tags to set on the child run.

    Yields:
        Active MLflow run context.

    Note:
        The child run is automatically ended when the context exits.
    """
    client = mlflow.tracking.MlflowClient()
    tracking_uri = mlflow.get_tracking_uri()
    is_azure_ml = tracking_uri and "azureml" in tracking_uri.lower()

    # Get experiment ID - CRITICAL: Get from parent run FIRST to ensure same experiment
    # This matches the original working code logic
    experiment_id = None

    # First, try to get from parent run (most reliable for Azure ML)
    try:
        parent_run_info = client.get_run(parent_run_id)
        experiment_id = parent_run_info.info.experiment_id
        logger.info(
            f"Using parent's experiment ID: {experiment_id} (parent: {parent_run_id[:12]}...)")
    except Exception as e:
        logger.error(f"Could not get parent run info: {e}")
        logger.error(f"Parent run ID: {parent_run_id}")
        import traceback
        logger.error(traceback.format_exc())

    # Fallback: try to get from experiment name
    if not experiment_id and experiment_name:
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id if experiment else None
        except Exception as e:
            logger.debug(f"Could not get experiment by name: {e}")

    # Fallback: try to get from active run
    if not experiment_id:
        try:
            active_run = mlflow.active_run()
            if active_run:
                experiment_id = active_run.info.experiment_id
        except Exception:
            pass

    # If still no experiment_id, we can't create a proper child run
    if not experiment_id:
        logger.error(
            f"Could not determine experiment ID for child run! "
            f"Parent: {parent_run_id[:12]}..., Trial: {trial_number}, Experiment name: {experiment_name}")
        logger.error(
            "This will create an independent run instead of a child run!")
        # Still try to create a run, but it won't be nested
        with mlflow.start_run(run_name=f"trial_{trial_number}") as run:
            # Try to set parent tag anyway
            try:
                mlflow.set_tag("mlflow.parentRunId", parent_run_id)
                mlflow.set_tag("trial_number", str(trial_number))
                logger.warning("Set parent tag on independent run as fallback")
            except Exception as e:
                logger.warning(f"Could not set parent tag: {e}")
            yield run
        return

    # Build tags for child run
    tags = {
        "mlflow.parentRunId": parent_run_id,
        "trial_number": str(trial_number),
    }

    # Add Azure ML-specific tags if using Azure ML
    if is_azure_ml:
        tags["azureml.runType"] = "trial"
        tags["azureml.trial"] = "true"
        # Azure ML also uses this tag for parent-child relationships
        tags["mlflow.runName"] = f"trial_{trial_number}"

    # Add any additional tags
    if additional_tags:
        tags.update(additional_tags)

    run_name = f"trial_{trial_number}"

    # For Azure ML, double-check we're using the parent's experiment ID
    # This ensures child runs appear nested in Azure ML UI
    if is_azure_ml:
        try:
            parent_run_info = client.get_run(parent_run_id)
            # Always use parent's experiment ID for Azure ML
            if parent_run_info.info.experiment_id != experiment_id:
                logger.info(
                    f"Using parent's experiment ID for Azure ML: {parent_run_info.info.experiment_id}"
                )
                experiment_id = parent_run_info.info.experiment_id
        except Exception as e:
            logger.warning(f"Could not verify parent run for Azure ML: {e}")

    # Create child run with parent tag
    try:
        run = client.create_run(
            experiment_id=experiment_id,
            tags=tags,
            run_name=run_name
        )
        logger.info(
            f"Created child run: {run.info.run_id[:12]}... (parent: {parent_run_id[:12]}..., experiment: {experiment_id})")
        logger.info(f"Child run tags: {tags}")

        # Verify the run was created with correct tags
        created_run_info = client.get_run(run.info.run_id)
        actual_parent_tag = created_run_info.data.tags.get(
            "mlflow.parentRunId")
        actual_trial_tag = created_run_info.data.tags.get("trial_number")
        logger.info(
            f"Verified child run tags - parent: {actual_parent_tag[:12] if actual_parent_tag else 'None'}..., "
            f"trial_number: {actual_trial_tag}"
        )

        # For Azure ML, verify the parent tag was set correctly
        if is_azure_ml:
            run_info = client.get_run(run.info.run_id)
            parent_tag = run_info.data.tags.get("mlflow.parentRunId")
            if parent_tag != parent_run_id:
                logger.warning(
                    f"Parent tag mismatch! Expected {parent_run_id[:12]}..., got {parent_tag[:12] if parent_tag else 'None'}...")
                # Force set it
                client.set_tag(run.info.run_id,
                               "mlflow.parentRunId", parent_run_id)
                logger.info("Re-set parent tag for Azure ML")
    except Exception as e:
        logger.warning(f"Error creating child run with tag: {e}")
        # Fallback: create run without tag, then set it
        run = client.create_run(
            experiment_id=experiment_id,
            run_name=run_name
        )
        # Set parent tag after creation
        for tag_key, tag_value in tags.items():
            try:
                client.set_tag(run.info.run_id, tag_key, tag_value)
            except Exception as tag_error:
                logger.warning(f"Could not set tag {tag_key}: {tag_error}")
        logger.info(
            f"Created child run and set tags: {run.info.run_id[:12]}...")

    # Start the run using the created run_id
    try:
        mlflow.start_run(run_id=run.info.run_id)
        logger.info(
            f"Started child run: {run.info.run_id[:12]}... (parent: {parent_run_id[:12]}...)")

        # Final verification for Azure ML
        if is_azure_ml:
            current_run = mlflow.active_run()
            if current_run:
                run_info = client.get_run(run.info.run_id)
                parent_tag = run_info.data.tags.get("mlflow.parentRunId")
                if parent_tag == parent_run_id:
                    logger.info(
                        f"✓ Verified parent-child relationship for Azure ML")
                else:
                    logger.warning(
                        f"⚠ Parent tag verification failed! "
                        f"Expected: {parent_run_id[:12]}..., Got: {parent_tag[:12] if parent_tag else 'None'}..."
                    )
    except Exception as e:
        logger.warning(f"Error starting child run: {e}")
        # Fallback: create new run
        with mlflow.start_run(run_name=run_name) as fallback_run:
            yield fallback_run
        return

    try:
        yield
    finally:
        mlflow.end_run()
        logger.debug(f"Ended child run: {run.info.run_id[:12]}...")

