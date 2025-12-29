"""MLflow tracking utilities for HPO sweeps.

Handles parent run creation, child run tracking, and best trial identification.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, List
import time

import mlflow
from shared.logging_utils import get_logger

logger = get_logger(__name__)


class MLflowSweepTracker:
    """Tracks MLflow runs for HPO sweeps."""

    def __init__(self, experiment_name: str):
        """
        Initialize sweep tracker.

        Args:
            experiment_name: MLflow experiment name.
        """
        self.experiment_name = experiment_name
        self._setup_experiment()

    def _setup_experiment(self) -> None:
        """Set up MLflow experiment."""
        try:
            # Check if MLflow tracking URI is already set (e.g., by notebook setup)
            current_tracking_uri = mlflow.get_tracking_uri()
            is_azure_ml = current_tracking_uri and "azureml" in current_tracking_uri.lower()

            if is_azure_ml:
                # Azure ML tracking is already configured, just set the experiment
                logger.debug(
                    f"Using existing Azure ML tracking URI: {current_tracking_uri[:50]}...")
                # Use setup_mlflow_cross_platform for consistency (it will use existing tracking URI)
                from shared.mlflow_setup import setup_mlflow_cross_platform
                setup_mlflow_cross_platform(
                    experiment_name=self.experiment_name,
                    ml_client=None,  # Tracking URI already set, will use it
                    fallback_to_local=False,  # Don't override Azure ML tracking
                )
            else:
                # No Azure ML tracking set, use cross-platform setup
                from shared.mlflow_setup import setup_mlflow_cross_platform
                setup_mlflow_cross_platform(
                    experiment_name=self.experiment_name,
                    ml_client=None,  # Will use local tracking or env vars
                    fallback_to_local=True,
                )
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
            logger.warning("Continuing without MLflow tracking...")

    @contextmanager
    def start_sweep_run(
        self,
        run_name: str,
        hpo_config: Dict[str, Any],
        backbone: str,
        study_name: str,
        checkpoint_config: Dict[str, Any],
        storage_path: Optional[Any],
        should_resume: bool,
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

        Yields:
            Active MLflow run context.
        """
        try:
            with mlflow.start_run(run_name=run_name) as parent_run:
                self._log_sweep_metadata(
                    hpo_config, backbone, study_name, checkpoint_config, storage_path, should_resume
                )
                yield parent_run
        except Exception as e:
            logger.warning(f"MLflow tracking failed: {e}")
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
        checkpoint_config: Dict[str, Any],
        storage_path: Optional[Any],
        should_resume: bool,
    ) -> None:
        """Log sweep metadata to MLflow."""
        objective_metric = hpo_config["objective"]["metric"]
        goal = hpo_config.get("objective", {}).get("goal", "maximize")
        max_trials = hpo_config["sampling"]["max_trials"]

        # Mark parent run as sweep job for Azure ML UI
        mlflow.set_tag("azureml.runType", "sweep")
        mlflow.set_tag("mlflow.runType", "sweep")
        mlflow.set_tag("azureml.sweep", "true")

        # Log primary metric and goal for Azure ML UI to identify best trial
        mlflow.log_param("primary_metric", objective_metric)
        mlflow.log_param("objective_goal", goal)

        # Log HPO parameters
        mlflow.log_param("backbone", backbone)
        mlflow.log_param("max_trials", max_trials)
        mlflow.log_param("study_name", study_name)
        mlflow.log_param("objective_metric", objective_metric)
        mlflow.log_param("checkpoint_enabled",
                         checkpoint_config.get("enabled", False))

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
    ) -> None:
        """
        Log final metrics and best trial information to parent run.

        Args:
            study: Completed Optuna study.
            objective_metric: Name of the objective metric.
            parent_run_id: ID of the parent MLflow run.
            run_name: Optional run name (used when resuming to search across all parent runs).
            should_resume: Whether this is a resumed run.
            hpo_output_dir: Path to HPO output directory containing trial checkpoints.
            backbone: Model backbone name (e.g., "distilbert").
            run_id: HPO run identifier for directory resolution.
            fold_splits: Optional list of fold splits (if k-fold CV used).
            hpo_config: HPO configuration dictionary containing MLflow settings.
        """
        # Use optuna module to check trial state properly
        try:
            import optuna
            completed_trials = len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
        except ImportError:
            # Fallback: count all trials as completed if we can't check state
            completed_trials = len(study.trials)

        mlflow.log_metric("n_trials", len(study.trials))
        mlflow.log_metric("n_completed_trials", completed_trials)

        if study.best_trial is not None and study.best_value is not None:
            # Log only the metric-specific name to avoid duplication
            mlflow.log_metric(f"best_{objective_metric}", study.best_value)

            # Log best hyperparameters
            for param_name, param_value in study.best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)

            # Find and log the best trial's MLflow run ID
            self._log_best_trial_id(
                study, parent_run_id, run_name, should_resume)

            # Log best trial checkpoint to MLflow if configured
            if hpo_config and hpo_config.get("mlflow", {}).get("log_best_checkpoint", True):
                if hpo_output_dir and backbone:
                    try:
                        self._log_best_trial_checkpoint(
                            study=study,
                            hpo_output_dir=hpo_output_dir,
                            backbone=backbone,
                            run_id=run_id,
                            fold_splits=fold_splits,
                        )
                    except Exception as e:
                        # Don't fail HPO if checkpoint logging fails
                        logger.warning(
                            f"Could not log best trial checkpoint to MLflow: {e}")

    def _log_best_trial_id(self, study: Any, parent_run_id: str, run_name: Optional[str] = None, should_resume: bool = False) -> None:
        """
        Find and log the best trial's MLflow run ID.

        Args:
            study: Completed Optuna study.
            parent_run_id: ID of the parent MLflow run.
            run_name: Optional run name (used when resuming to search across all parent runs).
            should_resume: Whether this is a resumed run.
        """
        try:
            client = mlflow.tracking.MlflowClient()
            active_run = mlflow.active_run()
            if active_run is None:
                raise ValueError("No active MLflow run")

            experiment_id = active_run.info.experiment_id

            # When resuming, search for child runs from ALL parent runs with the same name
            # This allows finding trials from previous (now FAILED) parent runs
            if should_resume and run_name:
                logger.debug(
                    f"Resuming: Searching for child runs from all parent runs with name '{run_name}'"
                )
                # First, find all parent runs with this name
                all_parent_runs = client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=f"attributes.run_name = '{run_name}'",
                    max_results=100,
                )
                parent_run_ids = [run.info.run_id for run in all_parent_runs]
                logger.debug(
                    f"Found {len(parent_run_ids)} parent run(s) with name '{run_name}'"
                )

                # Search for child runs from all these parent runs
                all_runs = []
                for parent_id in parent_run_ids:
                    child_runs = client.search_runs(
                        experiment_ids=[experiment_id],
                        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
                        max_results=1000,
                    )
                    all_runs.extend(child_runs)
                    logger.debug(
                        f"Found {len(child_runs)} child runs for parent {parent_id[:12]}..."
                    )
            else:
                # Normal case: search for child runs of current parent
                logger.debug(
                    f"Searching for child runs with parent: {parent_run_id[:12]}... in experiment: {experiment_id}")

            # Search with retry to handle cases where runs are still being committed
            import time
            max_retries = 3
            retry_delay = 1.0  # seconds
            
            # Get the best trial number we're looking for
            best_trial_number = study.best_trial.number
            best_run_id = None
            trial_to_run_id = {}
            
            for attempt in range(max_retries):
                # Re-search on each retry to get latest runs
                if should_resume and run_name:
                    # Re-search from all parent runs with this name
                    all_runs = []
                    for parent_id in parent_run_ids:
                        child_runs = client.search_runs(
                            experiment_ids=[experiment_id],
                            filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
                            max_results=1000,
                        )
                        all_runs.extend(child_runs)
                else:
                    # Normal case: search for child runs of current parent
                    all_runs = client.search_runs(
                        experiment_ids=[experiment_id],
                        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
                        max_results=1000,
                    )
                
                logger.info(
                    f"Found {len(all_runs)} total child runs for best trial search (attempt {attempt + 1}/{max_retries})")

                # Map trial numbers to run IDs
                trial_to_run_id = {}
                for run in all_runs:
                    trial_num_tag = run.data.tags.get("trial_number")
                    parent_tag = run.data.tags.get("mlflow.parentRunId")
                    logger.debug(
                        f"Child run {run.info.run_id[:12]}... - trial_number: {trial_num_tag}, "
                        f"parent: {parent_tag[:12] if parent_tag else 'None'}..."
                    )
                    if trial_num_tag:
                        try:
                            trial_num = int(trial_num_tag)
                            # Only keep the first occurrence if there are duplicates
                            if trial_num not in trial_to_run_id:
                                trial_to_run_id[trial_num] = run.info.run_id
                        except (ValueError, TypeError):
                            pass

                # Get the best trial's run ID
                best_run_id = trial_to_run_id.get(best_trial_number)
                
                # If we found the best trial, break early
                if best_run_id:
                    logger.debug(
                        f"Found best trial {best_trial_number} run ID: {best_run_id[:12]}..."
                    )
                    break
                
                # If this is not the last attempt, wait and retry
                if attempt < max_retries - 1:
                    logger.debug(
                        f"Best trial {best_trial_number} not found yet. "
                        f"Available: {sorted(trial_to_run_id.keys())}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)

            if best_run_id:
                # Log best trial run ID as both parameter and tag for Azure ML UI
                mlflow.log_param("best_trial_run_id", best_run_id)
                mlflow.set_tag("best_trial_run_id", best_run_id)
                mlflow.set_tag("best_trial_number", str(best_trial_number))
                logger.info(
                    f"Best trial: {best_trial_number} (run ID: {best_run_id[:12]}...)"
                )
            else:
                logger.warning(
                    f"Could not find MLflow run ID for best trial {best_trial_number}. "
                    f"Available trial numbers: {sorted(trial_to_run_id.keys())}"
                )
        except Exception as e:
            # Don't fail if we can't find the best trial run ID
            logger.warning(f"Could not retrieve best trial run ID: {e}")

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
        # When resuming, run_id changes, so we need to search for the actual checkpoint
        checkpoint_dir = None

        if fold_splits is not None and len(fold_splits) > 0:
            # K-fold CV: use last fold's checkpoint
            last_fold_idx = len(fold_splits) - 1
            # First try with the provided run_id
            if run_id:
                run_suffix = f"_{run_id}"
                checkpoint_dir = (
                    base_dir /
                    f"trial_{best_trial_number}{run_suffix}_fold{last_fold_idx}" /
                    "checkpoint"
                )

            # If not found, search for any trial directory matching the trial number and fold
            if not checkpoint_dir or not checkpoint_dir.exists():
                import glob
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
            # Single training
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
                import glob
                pattern = str(
                    base_dir / f"trial_{best_trial_number}_*" / "checkpoint")
                matches = glob.glob(pattern)
                if matches:
                    checkpoint_dir = Path(matches[0])
                else:
                    # Try without run_id suffix
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
                if active_run:
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

            # Check if we're using Azure ML
            tracking_uri = mlflow.get_tracking_uri()
            is_azure_ml = tracking_uri and "azureml" in tracking_uri.lower()

            artifact_logged = False

            if is_azure_ml:
                try:
                    active_run = mlflow.active_run()
                    if not active_run:
                        raise ValueError(
                            "No active MLflow run for artifact logging")

                    # Prefer best trial's child run over parent run for artifact upload
                    run_id_to_use = best_trial_run_id or active_run.info.run_id

                    run_id = run_id_to_use

                    logger.info(
                        "Uploading best trial checkpoint to Azure ML...")
                    try:
                        from azureml.core import Run as AzureMLRun
                        from azureml.core import Workspace
                        import os

                        # Get Azure ML workspace - try multiple methods
                        workspace = None

                        # Method 1: Try from config.json (if available)
                        try:
                            workspace = Workspace.from_config()
                        except Exception as ws_error1:

                            # Method 2: Try from environment variables
                            subscription_id = os.environ.get(
                                "AZURE_SUBSCRIPTION_ID")
                            resource_group = os.environ.get(
                                "AZURE_RESOURCE_GROUP")
                            workspace_name = os.environ.get(
                                "AZURE_WORKSPACE_NAME", "resume-ner-ws")  # Default from codebase

                            if subscription_id and resource_group:
                                try:
                                    workspace = Workspace(
                                        subscription_id=subscription_id,
                                        resource_group=resource_group,
                                        workspace_name=workspace_name
                                    )
                                except Exception:
                                    pass

                            # Method 3: Try loading from config.env file (same as mlflow_setup.py)
                            if not workspace:
                                try:
                                    # Path is already imported at module level
                                    project_root = Path.cwd()
                                    # Try to find config.env in project root or parent
                                    config_env_paths = [
                                        project_root / "config.env",
                                        project_root.parent / "config.env",
                                    ]

                                    for config_env_path in config_env_paths:
                                        if config_env_path.exists():
                                            # Simple .env parser
                                            env_vars = {}
                                            with open(config_env_path, "r", encoding="utf-8") as f:
                                                for line in f:
                                                    line = line.strip()
                                                    if not line or line.startswith("#"):
                                                        continue
                                                    if "=" in line:
                                                        key, value = line.split(
                                                            "=", 1)
                                                        key = key.strip()
                                                        value = value.strip().strip('"').strip("'")
                                                        env_vars[key] = value

                                            subscription_id = env_vars.get(
                                                "AZURE_SUBSCRIPTION_ID") or subscription_id
                                            resource_group = env_vars.get(
                                                "AZURE_RESOURCE_GROUP") or resource_group
                                            workspace_name = env_vars.get(
                                                "AZURE_WORKSPACE_NAME", workspace_name)

                                            if subscription_id and resource_group:
                                                workspace = Workspace(
                                                    subscription_id=subscription_id,
                                                    resource_group=resource_group,
                                                    workspace_name=workspace_name
                                                )
                                                break
                                except Exception:
                                    pass

                            # Method 4: Try parsing from MLflow tracking URI
                            if not workspace:
                                try:
                                    tracking_uri = mlflow.get_tracking_uri()
                                    # Azure ML tracking URI format: azureml://<region>.api.azureml.ms/mlflow/v2.0/subscriptions/<sub_id>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<ws_name>
                                    if "azureml" in tracking_uri.lower() and "/subscriptions/" in tracking_uri:
                                        import re
                                        sub_match = re.search(
                                            r'/subscriptions/([^/]+)', tracking_uri)
                                        rg_match = re.search(
                                            r'/resourceGroups/([^/]+)', tracking_uri)
                                        ws_match = re.search(
                                            r'/workspaces/([^/]+)', tracking_uri)

                                        if sub_match and rg_match and ws_match:
                                            workspace = Workspace(
                                                subscription_id=sub_match.group(
                                                    1),
                                                resource_group=rg_match.group(
                                                    1),
                                                workspace_name=ws_match.group(
                                                    1)
                                            )
                                except Exception:
                                    pass

                            if not workspace:
                                raise RuntimeError(
                                    f"Could not load Azure ML workspace. Tried: config.json, environment variables, config.env, and tracking URI parsing. "
                                    f"Last error: {ws_error1}"
                                )

                        # Get experiment ID and run name from MLflow run
                        experiment_id = active_run.info.experiment_id
                        run_name = active_run.info.run_name

                        # Get the Azure ML experiment
                        try:
                            experiment = workspace.experiments[experiment_id]
                        except Exception as exp_err1:
                            # Try to find experiment by name
                            try:
                                experiment_name = mlflow.get_experiment(
                                    experiment_id).name
                                experiment = workspace.experiments[experiment_name]
                            except Exception as exp_err2:
                                raise ValueError(
                                    f"Could not find Azure ML experiment for MLflow experiment_id={experiment_id}. "
                                    f"Errors: by_id={exp_err1}, by_name={exp_err2}"
                                )

                        # Get the Azure ML run - extract run ID from MLflow artifact URI
                        # This is the most reliable method for Azure ML
                        azureml_run = None
                        try:
                            mlflow_client = mlflow.tracking.MlflowClient()
                            mlflow_run_data = mlflow_client.get_run(run_id)

                            # Extract Azure ML run ID from artifact URI
                            # Format: azureml://.../runs/<run_id>/artifacts
                            if mlflow_run_data.info.artifact_uri and "azureml://" in mlflow_run_data.info.artifact_uri and "/runs/" in mlflow_run_data.info.artifact_uri:
                                import re
                                run_id_match = re.search(
                                    r'/runs/([^/]+)', mlflow_run_data.info.artifact_uri)
                                if run_id_match:
                                    azureml_run_id_from_uri = run_id_match.group(
                                        1)
                                    azureml_run = workspace.get_run(
                                        azureml_run_id_from_uri)
                        except Exception as uri_err:
                            logger.debug(
                                f"Failed to get Azure ML run from artifact URI: {uri_err}")

                        if not azureml_run:
                            # Fallback: try to get parent run's Azure ML run ID from artifact URI
                            if parent_run_id_for_artifacts:
                                try:
                                    parent_mlflow_run = mlflow.tracking.MlflowClient().get_run(
                                        parent_run_id_for_artifacts)
                                    if parent_mlflow_run.info.artifact_uri and "azureml://" in parent_mlflow_run.info.artifact_uri and "/runs/" in parent_mlflow_run.info.artifact_uri:
                                        import re
                                        run_id_match = re.search(
                                            r'/runs/([^/]+)', parent_mlflow_run.info.artifact_uri)
                                        if run_id_match:
                                            parent_azureml_run_id = run_id_match.group(
                                                1)
                                            azureml_run = workspace.get_run(
                                                parent_azureml_run_id)
                                            logger.warning(
                                                f"Could not find best trial's Azure ML run, "
                                                f"uploading checkpoint to current parent run instead"
                                            )
                                except Exception:
                                    pass

                            if not azureml_run:
                                logger.warning(
                                    f"Could not find Azure ML run for MLflow run_id={run_id}. "
                                    f"This may happen when resuming from a previous session. "
                                    f"Best trial checkpoint is available locally at: {checkpoint_dir}"
                                )
                                artifact_logged = False
                                azureml_run = None

                        # Upload artifacts using Azure ML Run's upload_file method
                        if not azureml_run:
                            # Already logged warning above, just skip upload
                            pass
                        else:
                            files_uploaded = 0
                            total_files = sum(len(files)
                                              for _, _, files in os.walk(checkpoint_dir))

                            # Check run status - completed runs don't accept uploads
                            run_status = azureml_run.get_status()
                            if run_status in ["Completed", "Failed", "Canceled"]:
                                # Try to get the best trial's child run if parent is completed
                                try:
                                    child_runs = list(
                                        azureml_run.get_children())
                                    for child_run in child_runs:
                                        child_tags = {}
                                        try:
                                            child_tags = child_run.get_tags() if hasattr(child_run, 'get_tags') else {}
                                        except Exception:
                                            pass
                                        child_name = child_run.name if hasattr(
                                            child_run, 'name') else str(child_run.id)

                                        trial_tag = child_tags.get('trial_number') or child_tags.get(
                                            'trial') or child_tags.get('optuna.trial.number')
                                        if (trial_tag and str(trial_tag) == str(best_trial_number)) or \
                                           (f"trial_{best_trial_number}" in child_name.lower() or f"trial-{best_trial_number}" in child_name.lower()):
                                            child_status = child_run.get_status()
                                            if child_status not in ["Completed", "Failed", "Canceled"]:
                                                azureml_run = child_run
                                                break
                                except Exception:
                                    pass

                            for root, dirs, files in os.walk(checkpoint_dir):
                                for file in files:
                                    file_path = Path(root) / file
                                    file_path = file_path.resolve()

                                    if not file_path.exists():
                                        continue

                                    # Try artifact paths: prefer subdirectory, fallback to filename only
                                    artifact_paths = [
                                        f"best_trial_checkpoint/{file}",
                                        file,
                                    ]

                                    for artifact_path in artifact_paths:
                                        try:
                                            azureml_run.upload_file(
                                                name=artifact_path, path_or_stream=str(file_path))
                                            files_uploaded += 1
                                            break
                                        except Exception as upload_error:
                                            error_msg = str(
                                                upload_error).lower()
                                            # If artifact already exists, that's okay - count as uploaded
                                            if "already exists" in error_msg or "resource conflict" in error_msg:
                                                files_uploaded += 1
                                                break
                                            # Otherwise, try next path
                                            continue

                            if files_uploaded > 0:
                                artifact_logged = True
                                logger.info(
                                    f"Uploaded best trial checkpoint: {files_uploaded}/{total_files} files for trial {best_trial_number}"
                                )
                            else:
                                raise RuntimeError(
                                    f"No files could be uploaded via Azure ML Run (attempted {total_files} files)")
                    except ImportError as import_error:
                        logger.warning(
                            f"Azure ML SDK (azureml.core) not available: {import_error}")
                        artifact_logged = False
                    except Exception as azureml_error:
                        error_type = type(azureml_error).__name__
                        error_msg = str(azureml_error)
                        logger.warning(
                            f"Failed to upload checkpoint to Azure ML: {error_type}: {error_msg}")
                        artifact_logged = False
                except Exception as outer_error:
                    # Handle errors from the outer try block (e.g., no active run)
                    logger.warning(
                        f"Could not upload checkpoint: {outer_error}")
                    artifact_logged = False
            else:
                # For non-Azure ML backends, try standard methods
                try:
                    mlflow.log_artifacts(str(checkpoint_dir),
                                         artifact_path="best_trial_checkpoint")
                    artifact_logged = True
                    logger.info(
                        f"Logged best trial checkpoint to MLflow: trial {best_trial_number} "
                        f"(path: {checkpoint_dir})"
                    )
                except Exception as e:
                    # Try client method as fallback
                    try:
                        client = mlflow.tracking.MlflowClient()
                        run_id_to_use = parent_run_id_for_artifacts
                        if not run_id_to_use:
                            active_run = mlflow.active_run()
                            if active_run:
                                run_id_to_use = active_run.info.run_id

                        if run_id_to_use:
                            client.log_artifacts(
                                run_id_to_use,
                                str(checkpoint_dir),
                                artifact_path="best_trial_checkpoint"
                            )
                            artifact_logged = True
                            logger.info(
                                f"Logged best trial checkpoint to MLflow using client: trial {best_trial_number} "
                                f"(path: {checkpoint_dir})"
                            )
                        else:
                            raise ValueError("No MLflow run ID available")
                    except Exception as client_error:
                        logger.warning(
                            f"MLflow artifact logging failed: {client_error}")
                        artifact_logged = False

            # Mark study as complete if checkpoint was successfully uploaded
            if artifact_logged:
                try:
                    from datetime import datetime
                    study.set_user_attr("hpo_complete", "true")
                    study.set_user_attr("checkpoint_uploaded", "true")
                    study.set_user_attr("completion_timestamp", datetime.now().isoformat())
                    study.set_user_attr("best_trial_number", str(best_trial_number))
                    logger.info(
                        f"Marked study as complete with checkpoint uploaded (best trial: {best_trial_number})"
                    )
                except Exception as attr_error:
                    logger.warning(
                        f"Could not mark study as complete: {attr_error}"
                    )
            
            # Log fallback message if upload failed
            if not artifact_logged:
                if is_azure_ml:
                    logger.info(
                        f"Best trial checkpoint for trial {best_trial_number} is available locally at: {checkpoint_dir}. "
                        f"Azure ML artifact upload was not successful, but the checkpoint can be accessed directly."
                    )
                else:
                    logger.info(
                        f"Checkpoint for best trial {best_trial_number} is available locally at: {checkpoint_dir}. "
                        f"MLflow artifact logging was not successful, but the checkpoint can be accessed directly."
                    )
        except Exception as e:
            # Don't fail HPO if checkpoint logging fails
            logger.warning(f"Could not log checkpoint to MLflow: {e}")
            logger.info(
                f"Checkpoint is available locally at: {checkpoint_dir}")

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


class MLflowBenchmarkTracker:
    """Tracks MLflow runs for benchmarking stage."""

    def __init__(self, experiment_name: str):
        """
        Initialize benchmark tracker.

        Args:
            experiment_name: MLflow experiment name.
        """
        self.experiment_name = experiment_name
        self._setup_experiment()

    def _setup_experiment(self) -> None:
        """Set up MLflow experiment."""
        try:
            current_tracking_uri = mlflow.get_tracking_uri()
            is_azure_ml = current_tracking_uri and "azureml" in current_tracking_uri.lower()

            if is_azure_ml:
                logger.debug(
                    f"Using existing Azure ML tracking URI: {current_tracking_uri[:50]}...")
                from shared.mlflow_setup import setup_mlflow_cross_platform
                setup_mlflow_cross_platform(
                    experiment_name=self.experiment_name,
                    ml_client=None,
                    fallback_to_local=False,
                )
            else:
                from shared.mlflow_setup import setup_mlflow_cross_platform
                setup_mlflow_cross_platform(
                    experiment_name=self.experiment_name,
                    ml_client=None,
                    fallback_to_local=True,
                )
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
            logger.warning("Continuing without MLflow tracking...")

    @contextmanager
    def start_benchmark_run(
        self,
        run_name: str,
        backbone: str,
        benchmark_source: str = "final_training",
    ):
        """
        Start a MLflow run for benchmarking.

        Args:
            run_name: Name for the benchmark run.
            backbone: Model backbone name.
            benchmark_source: Source of benchmark ("hpo_trial" or "final_training").

        Yields:
            Active MLflow run context.
        """
        try:
            with mlflow.start_run(run_name=run_name) as benchmark_run:
                # Set tags
                mlflow.set_tag("benchmark_source", benchmark_source)
                mlflow.set_tag("benchmarked_model", backbone)
                mlflow.set_tag("mlflow.runType", "benchmark")
                yield benchmark_run
        except Exception as e:
            logger.warning(f"MLflow tracking failed: {e}")
            logger.warning(
                "Continuing benchmarking without MLflow tracking...")
            from contextlib import nullcontext
            with nullcontext():
                yield None

    def log_benchmark_results(
        self,
        batch_sizes: List[int],
        iterations: int,
        warmup_iterations: int,
        max_length: int,
        device: Optional[str],
        benchmark_json_path: Path,
        benchmark_data: Dict[str, Any],
    ) -> None:
        """
        Log benchmark results to MLflow.

        Args:
            batch_sizes: List of batch sizes tested.
            iterations: Number of iterations.
            warmup_iterations: Warmup iterations.
            max_length: Max sequence length.
            device: Device used (cuda/cpu).
            benchmark_json_path: Path to benchmark.json file.
            benchmark_data: Parsed benchmark results dictionary.
        """
        try:
            # Log parameters
            mlflow.log_param("benchmark_batch_sizes", str(batch_sizes))
            mlflow.log_param("benchmark_iterations", iterations)
            mlflow.log_param("benchmark_warmup_iterations", warmup_iterations)
            mlflow.log_param("benchmark_max_length", max_length)
            mlflow.log_param("benchmark_device", device or "auto")

            # Log per-batch-size metrics
            # Benchmark JSON format: {"batch_1": {...}, "batch_8": {...}, ...}
            for batch_size in batch_sizes:
                batch_key = f"batch_{batch_size}"
                if batch_key in benchmark_data:
                    batch_results = benchmark_data[batch_key]
                    if "mean_ms" in batch_results:
                        mlflow.log_metric(
                            f"latency_batch_{batch_size}_ms", batch_results["mean_ms"])
                    if "median_ms" in batch_results:
                        mlflow.log_metric(
                            f"latency_batch_{batch_size}_p50_ms", batch_results["median_ms"])
                    if "p95_ms" in batch_results:
                        mlflow.log_metric(
                            f"latency_batch_{batch_size}_p95_ms", batch_results["p95_ms"])
                    if "p99_ms" in batch_results:
                        mlflow.log_metric(
                            f"latency_batch_{batch_size}_p99_ms", batch_results["p99_ms"])
                    # Note: std, min, max not currently in benchmark output, but structure supports them

            # Log throughput - calculate from batch results or use overall throughput
            # For now, we'll use the largest batch size's throughput as the overall metric
            max_batch_size = max(batch_sizes) if batch_sizes else None
            if max_batch_size:
                max_batch_key = f"batch_{max_batch_size}"
                if max_batch_key in benchmark_data:
                    batch_results = benchmark_data[max_batch_key]
                    if "throughput_docs_per_sec" in batch_results:
                        mlflow.log_metric(
                            "throughput_samples_per_sec", batch_results["throughput_docs_per_sec"])

            # Log artifact - use Azure ML SDK for Azure ML, standard MLflow for others
            if benchmark_json_path.exists():
                tracking_uri = mlflow.get_tracking_uri()
                is_azure_ml = tracking_uri and "azureml" in tracking_uri.lower()
                
                if is_azure_ml:
                    # Use Azure ML SDK for artifact upload (same approach as HPO checkpoint logging)
                    try:
                        active_run = mlflow.active_run()
                        if not active_run:
                            raise ValueError("No active MLflow run for artifact logging")
                        
                        run_id = active_run.info.run_id
                        
                        from azureml.core import Run as AzureMLRun
                        from azureml.core import Workspace
                        import os
                        
                        # Get Azure ML workspace (same logic as HPO checkpoint logging)
                        workspace = None
                        try:
                            workspace = Workspace.from_config()
                        except Exception:
                            subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
                            resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
                            workspace_name = os.environ.get("AZURE_WORKSPACE_NAME", "resume-ner-ws")
                            
                            if subscription_id and resource_group:
                                try:
                                    workspace = Workspace(
                                        subscription_id=subscription_id,
                                        resource_group=resource_group,
                                        workspace_name=workspace_name
                                    )
                                except Exception:
                                    pass
                        
                        if workspace:
                            # Get Azure ML run ID from MLflow artifact URI
                            mlflow_client = mlflow.tracking.MlflowClient()
                            mlflow_run_data = mlflow_client.get_run(run_id)
                            
                            azureml_run = None
                            if mlflow_run_data.info.artifact_uri and "azureml://" in mlflow_run_data.info.artifact_uri and "/runs/" in mlflow_run_data.info.artifact_uri:
                                import re
                                run_id_match = re.search(r'/runs/([^/]+)', mlflow_run_data.info.artifact_uri)
                                if run_id_match:
                                    azureml_run_id_from_uri = run_id_match.group(1)
                                    azureml_run = workspace.get_run(azureml_run_id_from_uri)
                            
                            if azureml_run:
                                # Upload artifact using Azure ML SDK
                                file_path = benchmark_json_path.resolve()
                                azureml_run.upload_file(
                                    name="benchmark.json",
                                    path_or_stream=str(file_path)
                                )
                                logger.debug(f"Uploaded benchmark.json to Azure ML run {azureml_run.id}")
                            else:
                                logger.warning("Could not find Azure ML run for benchmark artifact upload")
                        else:
                            logger.warning("Could not get Azure ML workspace for benchmark artifact upload")
                    except Exception as azureml_err:
                        logger.warning(f"Failed to upload benchmark artifact to Azure ML: {azureml_err}")
                else:
                    # Use standard MLflow for non-Azure ML backends
                    mlflow.log_artifact(str(benchmark_json_path),
                                        artifact_path="benchmark.json")
        except Exception as e:
            logger.warning(f"Could not log benchmark results to MLflow: {e}")


class MLflowTrainingTracker:
    """Tracks MLflow runs for final training stage."""

    def __init__(self, experiment_name: str):
        """
        Initialize training tracker.

        Args:
            experiment_name: MLflow experiment name.
        """
        self.experiment_name = experiment_name
        self._setup_experiment()

    def _setup_experiment(self) -> None:
        """Set up MLflow experiment."""
        try:
            current_tracking_uri = mlflow.get_tracking_uri()
            is_azure_ml = current_tracking_uri and "azureml" in current_tracking_uri.lower()

            if is_azure_ml:
                logger.debug(
                    f"Using existing Azure ML tracking URI: {current_tracking_uri[:50]}...")
                from shared.mlflow_setup import setup_mlflow_cross_platform
                setup_mlflow_cross_platform(
                    experiment_name=self.experiment_name,
                    ml_client=None,
                    fallback_to_local=False,
                )
            else:
                from shared.mlflow_setup import setup_mlflow_cross_platform
                setup_mlflow_cross_platform(
                    experiment_name=self.experiment_name,
                    ml_client=None,
                    fallback_to_local=True,
                )
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
            logger.warning("Continuing without MLflow tracking...")

    @contextmanager
    def start_training_run(
        self,
        run_name: str,
        backbone: str,
        training_type: str = "final",
    ):
        """
        Start a MLflow run for training.

        Args:
            run_name: Name for the training run.
            backbone: Model backbone name.
            training_type: Type of training ("final" or "continued").

        Yields:
            Active MLflow run context.
        """
        try:
            with mlflow.start_run(run_name=run_name) as training_run:
                # Set tags
                mlflow.set_tag("mlflow.runType", "training")
                mlflow.set_tag("training_type", training_type)
                yield training_run
        except Exception as e:
            logger.warning(f"MLflow tracking failed: {e}")
            logger.warning("Continuing training without MLflow tracking...")
            from contextlib import nullcontext
            with nullcontext():
                yield None

    def log_training_parameters(
        self,
        config: Dict[str, Any],
        data_config: Optional[Dict[str, Any]] = None,
        source_checkpoint: Optional[str] = None,
        data_strategy: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Log training parameters to MLflow.

        Args:
            config: Training configuration dictionary.
            data_config: Data configuration dictionary (optional).
            source_checkpoint: Path to checkpoint used for continued training (optional).
            data_strategy: Data strategy for continued training (optional).
            random_seed: Random seed used (optional).
        """
        try:
            training_config = config.get("training", {})

            # Training config parameters
            if "learning_rate" in training_config:
                mlflow.log_param(
                    "learning_rate", training_config["learning_rate"])
            if "batch_size" in training_config:
                mlflow.log_param("batch_size", training_config["batch_size"])
            if "dropout" in training_config:
                mlflow.log_param("dropout", training_config["dropout"])
            if "weight_decay" in training_config:
                mlflow.log_param(
                    "weight_decay", training_config["weight_decay"])
            if "epochs" in training_config:
                mlflow.log_param("epochs", training_config["epochs"])

            # Backbone from model config
            model_config = config.get("model", {})
            if "backbone" in model_config:
                mlflow.log_param("backbone", model_config["backbone"])

            # Data config
            if data_config:
                if "data_version" in data_config:
                    mlflow.log_param(
                        "data_version", data_config["data_version"])
                if "dataset_path" in data_config:
                    mlflow.log_param(
                        "dataset_path", data_config["dataset_path"])

            # Additional parameters
            training_type = "continued" if source_checkpoint else "final"
            mlflow.log_param("training_type", training_type)

            if source_checkpoint:
                mlflow.log_param("source_checkpoint", source_checkpoint)
            if data_strategy:
                mlflow.log_param("data_strategy", data_strategy)
            if random_seed is not None:
                mlflow.log_param("random_seed", random_seed)
        except Exception as e:
            logger.warning(f"Could not log training parameters to MLflow: {e}")

    def log_training_metrics(
        self,
        metrics: Dict[str, float],
        per_entity_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """
        Log training metrics to MLflow.

        Args:
            metrics: Dictionary of metrics (macro-f1, loss, etc.).
            per_entity_metrics: Optional dictionary of per-entity metrics.
        """
        try:
            # Main metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Per-entity metrics
            if per_entity_metrics:
                for entity, entity_metrics in per_entity_metrics.items():
                    for metric_type, metric_value in entity_metrics.items():
                        mlflow.log_metric(
                            f"{entity}_{metric_type}", metric_value)
        except Exception as e:
            logger.warning(f"Could not log training metrics to MLflow: {e}")

    def log_training_artifacts(
        self,
        checkpoint_dir: Path,
        metrics_json_path: Optional[Path] = None,
    ) -> None:
        """
        Log training artifacts to MLflow.

        Args:
            checkpoint_dir: Directory containing checkpoint files.
            metrics_json_path: Optional path to metrics.json file.
        """
        try:
            tracking_uri = mlflow.get_tracking_uri()
            is_azure_ml = tracking_uri and "azureml" in tracking_uri.lower()
            
            if is_azure_ml:
                # Use Azure ML SDK for artifact upload (same approach as HPO checkpoint logging)
                try:
                    active_run = mlflow.active_run()
                    if not active_run:
                        raise ValueError("No active MLflow run for artifact logging")
                    
                    run_id = active_run.info.run_id
                    
                    from azureml.core import Run as AzureMLRun
                    from azureml.core import Workspace
                    import os
                    
                    # Get Azure ML workspace (same logic as HPO checkpoint logging)
                    workspace = None
                    try:
                        workspace = Workspace.from_config()
                    except Exception:
                        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
                        resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
                        workspace_name = os.environ.get("AZURE_WORKSPACE_NAME", "resume-ner-ws")
                        
                        if subscription_id and resource_group:
                            try:
                                workspace = Workspace(
                                    subscription_id=subscription_id,
                                    resource_group=resource_group,
                                    workspace_name=workspace_name
                                )
                            except Exception:
                                pass
                    
                    if workspace:
                        # Get Azure ML run ID from MLflow artifact URI
                        mlflow_client = mlflow.tracking.MlflowClient()
                        mlflow_run_data = mlflow_client.get_run(run_id)
                        
                        azureml_run = None
                        if mlflow_run_data.info.artifact_uri and "azureml://" in mlflow_run_data.info.artifact_uri and "/runs/" in mlflow_run_data.info.artifact_uri:
                            import re
                            run_id_match = re.search(r'/runs/([^/]+)', mlflow_run_data.info.artifact_uri)
                            if run_id_match:
                                azureml_run_id_from_uri = run_id_match.group(1)
                                azureml_run = workspace.get_run(azureml_run_id_from_uri)
                        
                        if azureml_run:
                            # Upload checkpoint directory files
                            if checkpoint_dir.exists():
                                import os as os_module
                                files_uploaded = 0
                                for root, dirs, files in os_module.walk(checkpoint_dir):
                                    for file in files:
                                        file_path = Path(root) / file
                                        file_path = file_path.resolve()
                                        
                                        if not file_path.exists():
                                            continue
                                        
                                        # Create artifact path relative to checkpoint_dir
                                        rel_path = file_path.relative_to(checkpoint_dir)
                                        artifact_path = f"checkpoint/{rel_path}"
                                        
                                        try:
                                            azureml_run.upload_file(
                                                name=artifact_path,
                                                path_or_stream=str(file_path)
                                            )
                                            files_uploaded += 1
                                        except Exception as upload_err:
                                            error_msg = str(upload_err).lower()
                                            # If artifact already exists, that's okay
                                            if "already exists" in error_msg or "resource conflict" in error_msg:
                                                files_uploaded += 1
                                            else:
                                                logger.debug(f"Failed to upload {file_path.name}: {upload_err}")
                                
                                if files_uploaded > 0:
                                    logger.debug(f"Uploaded {files_uploaded} checkpoint files to Azure ML run {azureml_run.id}")
                            
                            # Upload metrics.json if provided
                            if metrics_json_path and metrics_json_path.exists():
                                file_path = metrics_json_path.resolve()
                                try:
                                    azureml_run.upload_file(
                                        name="metrics.json",
                                        path_or_stream=str(file_path)
                                    )
                                    logger.debug(f"Uploaded metrics.json to Azure ML run {azureml_run.id}")
                                except Exception as upload_err:
                                    error_msg = str(upload_err).lower()
                                    if "already exists" not in error_msg and "resource conflict" not in error_msg:
                                        logger.warning(f"Failed to upload metrics.json: {upload_err}")
                        else:
                            logger.warning("Could not find Azure ML run for training artifact upload")
                    else:
                        logger.warning("Could not get Azure ML workspace for training artifact upload")
                except Exception as azureml_err:
                    logger.warning(f"Failed to upload training artifacts to Azure ML: {azureml_err}")
            else:
                # Use standard MLflow for non-Azure ML backends
                # Log checkpoint directory
                if checkpoint_dir.exists():
                    mlflow.log_artifacts(str(checkpoint_dir),
                                         artifact_path="checkpoint")

                # Log metrics.json if provided
                if metrics_json_path and metrics_json_path.exists():
                    mlflow.log_artifact(str(metrics_json_path),
                                        artifact_path="metrics.json")
        except Exception as e:
            logger.warning(f"Could not log training artifacts to MLflow: {e}")


class MLflowConversionTracker:
    """Tracks MLflow runs for model conversion stage."""

    def __init__(self, experiment_name: str):
        """
        Initialize conversion tracker.

        Args:
            experiment_name: MLflow experiment name.
        """
        self.experiment_name = experiment_name
        self._setup_experiment()

    def _setup_experiment(self) -> None:
        """Set up MLflow experiment."""
        try:
            current_tracking_uri = mlflow.get_tracking_uri()
            is_azure_ml = current_tracking_uri and "azureml" in current_tracking_uri.lower()

            if is_azure_ml:
                logger.debug(
                    f"Using existing Azure ML tracking URI: {current_tracking_uri[:50]}...")
                from shared.mlflow_setup import setup_mlflow_cross_platform
                setup_mlflow_cross_platform(
                    experiment_name=self.experiment_name,
                    ml_client=None,
                    fallback_to_local=False,
                )
            else:
                from shared.mlflow_setup import setup_mlflow_cross_platform
                setup_mlflow_cross_platform(
                    experiment_name=self.experiment_name,
                    ml_client=None,
                    fallback_to_local=True,
                )
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
            logger.warning("Continuing without MLflow tracking...")

    @contextmanager
    def start_conversion_run(
        self,
        run_name: str,
        conversion_type: str,
        source_training_run: Optional[str] = None,
    ):
        """
        Start a MLflow run for model conversion.

        Args:
            run_name: Name for the conversion run.
            conversion_type: Type of conversion ("onnx_int8" or "onnx_fp32").
            source_training_run: Run ID of training that produced checkpoint (optional).

        Yields:
            Active MLflow run context.
        """
        try:
            with mlflow.start_run(run_name=run_name) as conversion_run:
                # Set tags
                mlflow.set_tag("conversion_type", conversion_type)
                mlflow.set_tag("mlflow.runType", "conversion")
                if source_training_run:
                    mlflow.set_tag("source_training_run", source_training_run)
                yield conversion_run
        except Exception as e:
            logger.warning(f"MLflow tracking failed: {e}")
            logger.warning("Continuing conversion without MLflow tracking...")
            from contextlib import nullcontext
            with nullcontext():
                yield None

    def log_conversion_parameters(
        self,
        checkpoint_path: str,
        conversion_target: str,
        quantization: str,
        opset_version: int,
        backbone: str,
    ) -> None:
        """
        Log conversion parameters to MLflow.

        Args:
            checkpoint_path: Path to source checkpoint.
            conversion_target: Target format ("onnx_int8" or "onnx_fp32").
            quantization: Quantization type ("int8" or "none").
            opset_version: ONNX opset version.
            backbone: Model backbone name.
        """
        try:
            mlflow.log_param("conversion_source", checkpoint_path)
            mlflow.log_param("conversion_target", conversion_target)
            mlflow.log_param("quantization", quantization)
            mlflow.log_param("onnx_opset_version", opset_version)
            mlflow.log_param("conversion_backbone", backbone)
        except Exception as e:
            logger.warning(
                f"Could not log conversion parameters to MLflow: {e}")

    def log_conversion_results(
        self,
        conversion_success: bool,
        onnx_model_path: Optional[Path],
        original_checkpoint_size: Optional[float] = None,
        smoke_test_passed: Optional[bool] = None,
        conversion_log_path: Optional[Path] = None,
    ) -> None:
        """
        Log conversion results to MLflow.

        Args:
            conversion_success: Whether conversion succeeded.
            onnx_model_path: Path to converted ONNX model.
            original_checkpoint_size: Size of original checkpoint in MB (optional).
            smoke_test_passed: Whether smoke test passed (optional).
            conversion_log_path: Path to conversion log file (optional).
        """
        try:
            # Log metrics
            mlflow.log_metric("conversion_success",
                              1 if conversion_success else 0)

            if onnx_model_path and onnx_model_path.exists():
                # Calculate model size
                model_size_mb = onnx_model_path.stat().st_size / (1024 * 1024)
                mlflow.log_metric("onnx_model_size_mb", model_size_mb)

                # Calculate compression ratio if original size provided
                if original_checkpoint_size:
                    compression_ratio = original_checkpoint_size / model_size_mb
                    mlflow.log_metric("compression_ratio", compression_ratio)

            if smoke_test_passed is not None:
                mlflow.log_metric("smoke_test_passed",
                                  1 if smoke_test_passed else 0)

            # Log artifacts - use Azure ML SDK for Azure ML, standard MLflow for others
            tracking_uri = mlflow.get_tracking_uri()
            is_azure_ml = tracking_uri and "azureml" in tracking_uri.lower()
            
            if is_azure_ml:
                # Use Azure ML SDK for artifact upload (same approach as HPO checkpoint logging)
                try:
                    active_run = mlflow.active_run()
                    if not active_run:
                        raise ValueError("No active MLflow run for artifact logging")
                    
                    run_id = active_run.info.run_id
                    
                    from azureml.core import Run as AzureMLRun
                    from azureml.core import Workspace
                    import os
                    
                    # Get Azure ML workspace (same logic as HPO checkpoint logging)
                    workspace = None
                    try:
                        workspace = Workspace.from_config()
                    except Exception:
                        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
                        resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
                        workspace_name = os.environ.get("AZURE_WORKSPACE_NAME", "resume-ner-ws")
                        
                        if subscription_id and resource_group:
                            try:
                                workspace = Workspace(
                                    subscription_id=subscription_id,
                                    resource_group=resource_group,
                                    workspace_name=workspace_name
                                )
                            except Exception:
                                pass
                    
                    if workspace:
                        # Get Azure ML run ID from MLflow artifact URI
                        mlflow_client = mlflow.tracking.MlflowClient()
                        mlflow_run_data = mlflow_client.get_run(run_id)
                        
                        azureml_run = None
                        if mlflow_run_data.info.artifact_uri and "azureml://" in mlflow_run_data.info.artifact_uri and "/runs/" in mlflow_run_data.info.artifact_uri:
                            import re
                            run_id_match = re.search(r'/runs/([^/]+)', mlflow_run_data.info.artifact_uri)
                            if run_id_match:
                                azureml_run_id_from_uri = run_id_match.group(1)
                                azureml_run = workspace.get_run(azureml_run_id_from_uri)
                        
                        if azureml_run:
                            # Upload ONNX model if available
                            if onnx_model_path and onnx_model_path.exists():
                                artifact_name = onnx_model_path.name
                                file_path = onnx_model_path.resolve()
                                try:
                                    azureml_run.upload_file(
                                        name=artifact_name,
                                        path_or_stream=str(file_path)
                                    )
                                    logger.debug(f"Uploaded {artifact_name} to Azure ML run {azureml_run.id}")
                                except Exception as upload_err:
                                    error_msg = str(upload_err).lower()
                                    if "already exists" not in error_msg and "resource conflict" not in error_msg:
                                        logger.warning(f"Failed to upload {artifact_name}: {upload_err}")
                            
                            # Upload conversion log if available
                            if conversion_log_path and conversion_log_path.exists():
                                file_path = conversion_log_path.resolve()
                                try:
                                    azureml_run.upload_file(
                                        name="conversion_log.txt",
                                        path_or_stream=str(file_path)
                                    )
                                    logger.debug(f"Uploaded conversion_log.txt to Azure ML run {azureml_run.id}")
                                except Exception as upload_err:
                                    error_msg = str(upload_err).lower()
                                    if "already exists" not in error_msg and "resource conflict" not in error_msg:
                                        logger.warning(f"Failed to upload conversion_log.txt: {upload_err}")
                        else:
                            logger.warning("Could not find Azure ML run for conversion artifact upload")
                    else:
                        logger.warning("Could not get Azure ML workspace for conversion artifact upload")
                except Exception as azureml_err:
                    logger.warning(f"Failed to upload conversion artifacts to Azure ML: {azureml_err}")
            else:
                # Use standard MLflow for non-Azure ML backends
                if onnx_model_path and onnx_model_path.exists():
                    artifact_name = onnx_model_path.name
                    # Add retry logic for artifact upload (handles SSL/network timeouts)
                    max_retries = 3
                    retry_delay = 2  # seconds
                    for attempt in range(max_retries):
                        try:
                            mlflow.log_artifact(str(onnx_model_path),
                                                artifact_path=artifact_name)
                            logger.debug(f"Successfully uploaded {artifact_name} to MLflow")
                            break
                        except Exception as upload_err:
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                                logger.warning(
                                    f"Failed to upload {artifact_name} (attempt {attempt + 1}/{max_retries}): {upload_err}. "
                                    f"Retrying in {wait_time}s..."
                                )
                                time.sleep(wait_time)
                            else:
                                logger.warning(f"Failed to upload {artifact_name} after {max_retries} attempts: {upload_err}")

                if conversion_log_path and conversion_log_path.exists():
                    # Add retry logic for log upload
                    max_retries = 3
                    retry_delay = 2  # seconds
                    for attempt in range(max_retries):
                        try:
                            mlflow.log_artifact(str(conversion_log_path),
                                                artifact_path="conversion_log.txt")
                            logger.debug("Successfully uploaded conversion_log.txt to MLflow")
                            break
                        except Exception as upload_err:
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                                logger.warning(
                                    f"Failed to upload conversion_log.txt (attempt {attempt + 1}/{max_retries}): {upload_err}. "
                                    f"Retrying in {wait_time}s..."
                                )
                                time.sleep(wait_time)
                            else:
                                logger.warning(f"Failed to upload conversion_log.txt after {max_retries} attempts: {upload_err}")
        except Exception as e:
            logger.warning(f"Could not log conversion results to MLflow: {e}")
