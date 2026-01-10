from __future__ import annotations

"""
@meta
name: tracking_mlflow_lifecycle
type: utility
domain: tracking
responsibility:
  - Manage MLflow run lifecycle with safe termination
  - Handle run status checks and error handling
inputs:
  - Run IDs
  - Termination status and tags
outputs:
  - Termination success status
tags:
  - utility
  - tracking
  - mlflow
  - lifecycle
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""MLflow run lifecycle management utilities.

This module provides safe run termination functions with status checks
and error handling.
"""
from typing import Optional, Dict

import mlflow
from common.shared.logging_utils import get_logger

logger = get_logger(__name__)

def terminate_run_safe(
    run_id: str,
    status: str = "FINISHED",
    tags: Optional[Dict[str, str]] = None,
    check_status: bool = True,
) -> bool:
    """
    Safely terminate an MLflow run with optional status check and tag setting.

    This function checks the run status before termination if requested,
    sets tags if provided, and handles errors gracefully.

    Args:
        run_id: MLflow run ID to terminate.
        status: Termination status ("FINISHED", "FAILED", "KILLED").
        tags: Optional dictionary of tags to set before termination.
        check_status: If True, check current status before terminating.
                     Only terminates if status is "RUNNING".

    Returns:
        True if termination succeeded, False otherwise.
    """
    try:
        client = mlflow.tracking.MlflowClient()

        # Check current status if requested
        if check_status:
            try:
                run = client.get_run(run_id)
                current_status = run.info.status
                if current_status != "RUNNING":
                    logger.info(
                        f"Run {run_id[:12]}... already has status {current_status}, "
                        f"skipping termination (expected RUNNING)"
                    )
                    return True  # Already terminated, consider it success
            except Exception as e:
                logger.warning(
                    f"Could not check run status for {run_id[:12]}...: {e}"
                )
                # Continue with termination attempt

        # Set tags before termination if provided
        if tags:
            try:
                for tag_key, tag_value in tags.items():
                    client.set_tag(run_id, tag_key, tag_value)
                logger.debug(
                    f"Set tags on run {run_id[:12]}...: {list(tags.keys())}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not set tags on run {run_id[:12]}...: {e}"
                )
                # Continue with termination even if tag setting failed

        # Terminate the run
        client.set_terminated(run_id, status=status)
        logger.info(
            f"Successfully terminated run {run_id[:12]}... with status {status}"
        )
        return True

    except Exception as e:
        logger.warning(
            f"Failed to terminate run {run_id[:12]}...: {e}",
            exc_info=True
        )
        return False

def ensure_run_terminated(
    run_id: str,
    expected_status: str = "FINISHED",
) -> bool:
    """
    Ensure a run is terminated with the expected status.

    Checks current status and terminates only if still RUNNING.
    Handles already-terminated runs gracefully.

    Args:
        run_id: MLflow run ID to check/terminate.
        expected_status: Expected termination status ("FINISHED", "FAILED", "KILLED").

    Returns:
        True if run is terminated (or was already terminated), False otherwise.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        current_status = run.info.status

        if current_status == "RUNNING":
            # Run is still running, terminate it
            return terminate_run_safe(
                run_id=run_id,
                status=expected_status,
                check_status=False,  # Already checked
            )
        else:
            # Run is already terminated
            if current_status == expected_status:
                logger.debug(
                    f"Run {run_id[:12]}... already terminated with "
                    f"expected status {expected_status}"
                )
            else:
                logger.info(
                    f"Run {run_id[:12]}... already terminated with status "
                    f"{current_status} (expected {expected_status})"
                )
            return True

    except Exception as e:
        logger.warning(
            f"Could not ensure run {run_id[:12]}... is terminated: {e}",
            exc_info=True
        )
        return False

def terminate_run_with_tags(
    run_id: str,
    status: str,
    tags: Dict[str, str],
) -> bool:
    """
    Convenience function to set tags and terminate a run.

    This is a common pattern: set tags then terminate.

    Args:
        run_id: MLflow run ID to terminate.
        status: Termination status ("FINISHED", "FAILED", "KILLED").
        tags: Dictionary of tags to set before termination.

    Returns:
        True if termination succeeded, False otherwise.
    """
    return terminate_run_safe(
        run_id=run_id,
        status=status,
        tags=tags,
        check_status=True,
    )

