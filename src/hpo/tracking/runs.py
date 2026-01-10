"""Trial run creation utilities for HPO.

Handles MLflow run creation for non-CV trials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from common.shared.logging_utils import get_logger

# Lazy import to avoid pytest collection issues
try:
    from infrastructure.tracking.mlflow import terminate_run_safe
except ImportError:
    # During pytest collection, path might not be set up yet
    # Will be imported when actually needed
    terminate_run_safe = None

logger = get_logger(__name__)


def create_trial_run_no_cv(
    trial_params: Dict[str, Any],
    config_dir: Path,
    output_base_dir: Path,
    hpo_parent_run_id: Optional[str] = None,
) -> Optional[str]:
    """
    Create MLflow trial run for non-CV case.

    Creates a trial-level run as child of HPO parent for consistency.
    This ensures trial runs are properly linked and can be found later.

    Args:
        trial_params: Trial parameters including trial_number, run_id, backbone.
        config_dir: Configuration directory.
        output_base_dir: Base output directory.
        hpo_parent_run_id: Optional HPO parent run ID.

    Returns:
        Trial run ID if created, None otherwise.
    """
    if not hpo_parent_run_id:
        return None

    try:
        client = mlflow.tracking.MlflowClient()
        active_run = mlflow.active_run()
        if not active_run:
            return None

        experiment_id = active_run.info.experiment_id
        trial_number = trial_params.get("trial_number", "unknown")

        # Build systematic run name using NamingContext
        run_name = None
        try:
            from infrastructure.naming import create_naming_context
            from infrastructure.tracking.mlflow.naming import (
                build_mlflow_run_name,
                build_mlflow_tags,
            )
            from common.shared.platform_detection import detect_platform

            # Extract backbone short name
            backbone_full = trial_params.get("backbone", "unknown")
            backbone_short = (
                backbone_full.split("-")[0] if "-" in backbone_full else backbone_full
            )

            # Derive grouping hashes from parent run if available
            study_key_hash = None
            study_family_hash = None
            try:
                parent_run = client.get_run(hpo_parent_run_id)
                study_key_hash = parent_run.data.tags.get("code.study_key_hash")
                study_family_hash = parent_run.data.tags.get("code.study_family_hash")
            except Exception:
                pass

            # Build trial_id from trial_number and run_id if available
            run_id = trial_params.get("run_id")
            if run_id:
                trial_id = f"trial_{trial_number}_{run_id}"
            else:
                trial_id = f"trial_{trial_number}"

            # Create NamingContext for HPO trial
            # Note: study_key_hash is retrieved from parent run above (lines 70-74)
            # Use explicit trial_number from Optuna (robust, no string parsing)
            trial_number_int = trial_params.get("trial_number")
            trial_context = create_naming_context(
                process_type="hpo",
                model=backbone_short,
                environment=detect_platform(),
                storage_env=detect_platform(),
                stage="hpo_trial",
                trial_id=trial_id,
                trial_number=trial_number_int,  # Explicit Optuna trial number
                study_key_hash=study_key_hash,  # Retrieved from parent run tags
                trial_key_hash=None,  # Not computed for non-CV trials
            )

            # Build systematic run name
            run_name = build_mlflow_run_name(trial_context, config_dir)

            # Build tags including project identity tags
            trial_tags = build_mlflow_tags(
                context=trial_context,
                output_dir=output_base_dir,
                config_dir=config_dir,
                study_key_hash=study_key_hash,
                study_family_hash=study_family_hash,
            )
            # Merge with trial-specific tags
            trial_tags.update(
                {
                    "mlflow.parentRunId": hpo_parent_run_id,
                    "azureml.runType": "trial",
                    "azureml.trial": "true",
                    "trial_number": str(trial_number),
                }
            )
        except Exception as e:
            logger.warning(
                f"Could not build systematic run name and tags (no CV): {e}, using fallback"
            )
            # Fallback to simple name
            run_name = f"trial_{trial_number}"
            # Fallback to minimal tags - still try to get project name from config
            try:
                from infrastructure.tracking.mlflow.config_loader import (
                    get_naming_config,
                )

                naming_config = get_naming_config(config_dir)
                project_name = naming_config.get("project_name", "resume-ner")
            except Exception:
                project_name = "resume-ner"
            trial_tags = {
                "mlflow.parentRunId": hpo_parent_run_id,
                "azureml.runType": "trial",
                "azureml.trial": "true",
                "trial_number": str(trial_number),
                "code.project": project_name,  # Always include project identity
            }

        # Ensure run_name is never None (MLflow will auto-generate random names if None)
        if not run_name:
            logger.warning(f"[TRIAL_RUN_NO_CV] run_name is None, using fallback for trial {trial_number}")
            run_name = f"trial_{trial_number}"
        
        # Create trial run as child of HPO parent
        trial_run = client.create_run(
            experiment_id=experiment_id, tags=trial_tags, run_name=run_name
        )
        trial_run_id = trial_run.info.run_id

        # DO NOT mark trial run as FINISHED here - it should remain RUNNING until training completes
        # The run will be used by training subprocess and marked as FINISHED after training completes
        logger.info(
            f"[TRIAL_RUN_NO_CV] Created trial run (no CV): {trial_run_id[:12]}... "
            f"(trial {trial_number}). Run remains RUNNING until training completes."
        )
        return trial_run_id
    except Exception as e:
        logger.warning(f"Could not create trial run (no CV): {e}")
        return None


def finalize_trial_run_no_cv(trial_run_id: str, trial_number: int) -> None:
    """
    Mark trial run as FINISHED after training completes (no CV case).

    Args:
        trial_run_id: Trial run ID to finalize.
        trial_number: Trial number for logging.
    """
    if not trial_run_id:
        return

    try:
        logger.info(
            f"[TRIAL_RUN_NO_CV] Training completed. Marking trial run "
            f"{trial_run_id[:12]}... as FINISHED (trial {trial_number})"
        )
        terminate_run_safe(trial_run_id, status="FINISHED", check_status=True)
        logger.info(
            f"[TRIAL_RUN_NO_CV] Successfully marked trial run "
            f"{trial_run_id[:12]}... as FINISHED"
        )
    except Exception as e:
        logger.warning(f"[TRIAL_RUN_NO_CV] Could not mark trial run as FINISHED: {e}")

