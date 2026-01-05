"""MLflow tracker for training stage."""

from __future__ import annotations

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
from shared.logging_utils import get_logger

from orchestration.jobs.tracking.mlflow_types import RunHandle
from orchestration.jobs.tracking.mlflow_naming import build_mlflow_tags, build_mlflow_run_key, build_mlflow_run_key_hash
from orchestration.jobs.tracking.mlflow_index import update_mlflow_index
from orchestration.jobs.tracking.utils.mlflow_utils import get_mlflow_run_url, retry_with_backoff
from orchestration.jobs.tracking.artifacts.manager import create_checkpoint_archive
from orchestration.jobs.tracking.trackers.base_tracker import BaseTracker

logger = get_logger(__name__)


class MLflowTrainingTracker(BaseTracker):
    """Tracks MLflow runs for final training stage."""

    def __init__(self, experiment_name: str):
        """
        Initialize tracker.

        Args:
            experiment_name: MLflow experiment name.
        """
        super().__init__(experiment_name)


    @contextmanager
    def start_training_run(
        self,
        run_name: str,
        backbone: str,
        training_type: str = "final",
        context: Optional[Any] = None,  # NamingContext
        output_dir: Optional[Path] = None,
        parent_run_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ):
        """
        Start a MLflow run for training.

        Args:
            run_name: Name for the training run.
            backbone: Model backbone name.
            training_type: Type of training ("final" or "continued").
            context: Optional NamingContext for tag-based identification.
            output_dir: Optional output directory for metadata persistence.
            parent_run_id: Optional parent MLflow run ID.
            group_id: Optional group/session identifier.

        Yields:
            RunHandle with run information.
        """
        try:
            with mlflow.start_run(run_name=run_name) as training_run:
                run_id = training_run.info.run_id
                experiment_id = training_run.info.experiment_id
                tracking_uri = mlflow.get_tracking_uri()

                # Infer config_dir from output_dir
                config_dir = None
                if output_dir:
                    root_dir_for_config = output_dir.parent.parent if output_dir.parent.name == "outputs" else output_dir.parent.parent.parent
                    config_dir = root_dir_for_config / "config" if root_dir_for_config else None

                # Build and set tags atomically
                tags = build_mlflow_tags(
                    context=context,
                    output_dir=output_dir,
                    parent_run_id=parent_run_id,
                    group_id=group_id,
                    config_dir=config_dir,
                )
                # Add training-specific tags
                tags["mlflow.runType"] = "training"
                tags["training_type"] = training_type
                mlflow.set_tags(tags)

                # Build RunHandle
                run_key = build_mlflow_run_key(context) if context else None
                run_key_hash = build_mlflow_run_key_hash(
                    run_key) if run_key else None

                handle = RunHandle(
                    run_id=run_id,
                    run_key=run_key or "",
                    run_key_hash=run_key_hash or "",
                    experiment_id=experiment_id,
                    experiment_name=self.experiment_name,
                    tracking_uri=tracking_uri,
                    artifact_uri=training_run.info.artifact_uri,
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

                # Save MLflow run info to metadata.json if context and output_dir are available
                if context and output_dir and config_dir:
                    try:
                        from orchestration.metadata_manager import save_metadata_with_fingerprints

                        # Build root_dir from output_dir
                        root_dir = output_dir.parent.parent if output_dir.parent.name == "outputs" else output_dir.parent.parent.parent
                        if not root_dir or not root_dir.exists():
                            root_dir = Path.cwd()

                        # Save MLflow run information to metadata
                        mlflow_info = {
                            "run_id": run_id,
                            "experiment_id": experiment_id,
                            "tracking_uri": tracking_uri,
                            "run_key": run_key,
                            "run_key_hash": run_key_hash,
                        }

                        save_metadata_with_fingerprints(
                            root_dir=root_dir,
                            config_dir=config_dir,
                            context=context,
                            metadata_content={"mlflow": mlflow_info},
                        )
                        logger.debug(
                            f"Saved MLflow run info to metadata for run {run_id[:12]}...")
                    except Exception as e:
                        logger.debug(
                            f"Could not save MLflow run info to metadata: {e}")

                yield handle
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
                        raise ValueError(
                            "No active MLflow run for artifact logging")

                    run_id = active_run.info.run_id

                    from azureml.core import Run as AzureMLRun
                    from azureml.core import Workspace
                    import os

                    # Get Azure ML workspace (same logic as HPO checkpoint logging)
                    workspace = None
                    try:
                        workspace = Workspace.from_config()
                    except Exception:
                        subscription_id = os.environ.get(
                            "AZURE_SUBSCRIPTION_ID")
                        resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
                        workspace_name = os.environ.get(
                            "AZURE_WORKSPACE_NAME", "resume-ner-ws")

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
                            run_id_match = re.search(
                                r'/runs/([^/]+)', mlflow_run_data.info.artifact_uri)
                            if run_id_match:
                                azureml_run_id_from_uri = run_id_match.group(1)
                                azureml_run = workspace.get_run(
                                    azureml_run_id_from_uri)

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
                                        rel_path = file_path.relative_to(
                                            checkpoint_dir)
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
                                                logger.debug(
                                                    f"Failed to upload {file_path.name}: {upload_err}")

                                if files_uploaded > 0:
                                    logger.debug(
                                        f"Uploaded {files_uploaded} checkpoint files to Azure ML run {azureml_run.id}")

                            # Upload metrics.json if provided
                            if metrics_json_path and metrics_json_path.exists():
                                file_path = metrics_json_path.resolve()
                                try:
                                    azureml_run.upload_file(
                                        name="metrics.json",
                                        path_or_stream=str(file_path)
                                    )
                                    logger.debug(
                                        f"Uploaded metrics.json to Azure ML run {azureml_run.id}")
                                except Exception as upload_err:
                                    error_msg = str(upload_err).lower()
                                    if "already exists" not in error_msg and "resource conflict" not in error_msg:
                                        logger.warning(
                                            f"Failed to upload metrics.json: {upload_err}")
                        else:
                            logger.warning(
                                "Could not find Azure ML run for training artifact upload")
                    else:
                        logger.warning(
                            "Could not get Azure ML workspace for training artifact upload")
                except Exception as azureml_err:
                    logger.warning(
                        f"Failed to upload training artifacts to Azure ML: {azureml_err}")
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



