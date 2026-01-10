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
from common.shared.logging_utils import get_logger

from orchestration.jobs.tracking.mlflow_types import RunHandle
from orchestration.jobs.tracking.mlflow_naming import build_mlflow_tags, build_mlflow_run_key, build_mlflow_run_key_hash
from orchestration.jobs.tracking.mlflow_index import update_mlflow_index
from orchestration.jobs.tracking.utils.mlflow_utils import retry_with_backoff
# Lazy import to avoid pytest collection issues
try:
    from infrastructure.tracking.mlflow import get_mlflow_run_url
except ImportError:
    # During pytest collection, path might not be set up yet
    get_mlflow_run_url = None
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
        # Infer config_dir from output_dir BEFORE creating run
        config_dir = None
        if output_dir:
            root_dir_for_config = output_dir.parent.parent if output_dir.parent.name == "outputs" else output_dir.parent.parent.parent
            config_dir = root_dir_for_config / "config" if root_dir_for_config else None

        # Check if tracking is enabled for training stage BEFORE creating run
        from orchestration.jobs.tracking.config.loader import get_tracking_config
        tracking_config = get_tracking_config(config_dir=config_dir, stage="training")
        if not tracking_config.get("enabled", True):
            logger.info("[Training Tracker] MLflow tracking disabled for training stage (tracking.training.enabled=false)")
            from contextlib import nullcontext
            with nullcontext():
                yield None
            return

        try:
            with mlflow.start_run(run_name=run_name) as training_run:
                run_id = training_run.info.run_id
                experiment_id = training_run.info.experiment_id
                tracking_uri = mlflow.get_tracking_uri()

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
                        from infrastructure.metadata.training import save_metadata_with_fingerprints

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
            # Infer config_dir to check tracking config
            config_dir = None
            if checkpoint_dir:
                current = checkpoint_dir.parent
                while current.parent != current:
                    if current.name == "outputs":
                        root_dir_for_config = current.parent
                        config_dir = root_dir_for_config / "config" if root_dir_for_config else None
                        break
                    current = current.parent
            
            from orchestration.jobs.tracking.config.loader import get_tracking_config
            tracking_config = get_tracking_config(config_dir=config_dir, stage="training")
            
            # Use MLflow for artifact upload (works for both Azure ML and non-Azure ML backends)
            from infrastructure.tracking.mlflow import log_artifacts_safe, log_artifact_safe
            # Log checkpoint directory if enabled
            if tracking_config.get("log_checkpoint", True) and checkpoint_dir.exists():
                log_artifacts_safe(
                    local_dir=checkpoint_dir,
                    artifact_path="checkpoint",
                    run_id=None,  # Use active run
                )
            elif not tracking_config.get("log_checkpoint", True):
                logger.debug("[Training Tracker] Checkpoint logging disabled (tracking.training.log_checkpoint=false)")

            # Log metrics.json if provided and enabled
            if tracking_config.get("log_metrics_json", True) and metrics_json_path and metrics_json_path.exists():
                log_artifact_safe(
                    local_path=metrics_json_path,
                    artifact_path="metrics.json",
                    run_id=None,  # Use active run
                )
            elif not tracking_config.get("log_metrics_json", True):
                logger.debug("[Training Tracker] Metrics JSON logging disabled (tracking.training.log_metrics_json=false)")
        except Exception as e:
            logger.warning(f"Could not log training artifacts to MLflow: {e}")



