"""MLflow tracker for benchmark stage."""

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


class MLflowBenchmarkTracker(BaseTracker):
    """Tracks MLflow runs for benchmarking stage."""

    def __init__(self, experiment_name: str):
        """
        Initialize tracker.

        Args:
            experiment_name: MLflow experiment name.
        """
        super().__init__(experiment_name)


    @contextmanager
    def start_benchmark_run(
        self,
        run_name: str,
        backbone: str,
        benchmark_source: str = "final_training",
        context: Optional[Any] = None,  # NamingContext
        output_dir: Optional[Path] = None,
        parent_run_id: Optional[str] = None,
        group_id: Optional[str] = None,
        study_key_hash: Optional[str] = None,
        trial_key_hash: Optional[str] = None,
    ):
        """
        Start a MLflow run for benchmarking.

        Args:
            run_name: Name for the benchmark run.
            backbone: Model backbone name.
            benchmark_source: Source of benchmark ("hpo_trial" or "final_training").
            context: Optional NamingContext for tag-based identification.
            output_dir: Optional output directory for metadata persistence.
            parent_run_id: Optional parent MLflow run ID.
            group_id: Optional group/session identifier.
            study_key_hash: Optional study key hash from HPO trial (for grouping tags).
            trial_key_hash: Optional trial key hash from HPO trial (for grouping tags).

        Yields:
            RunHandle with run information.
        """
        try:
            with mlflow.start_run(run_name=run_name) as benchmark_run:
                run_id = benchmark_run.info.run_id
                experiment_id = benchmark_run.info.experiment_id
                tracking_uri = mlflow.get_tracking_uri()

                # Infer config_dir from output_dir
                config_dir = None
                if output_dir:
                    root_dir_for_config = output_dir.parent.parent if output_dir.parent.name == "outputs" else output_dir.parent.parent.parent
                    config_dir = root_dir_for_config / "config" if root_dir_for_config else None

                logger.info(
                    f"[START_BENCHMARK_RUN] Building tags: context={context}, "
                    f"study_key_hash={study_key_hash[:16] if study_key_hash else None}..., "
                    f"trial_key_hash={trial_key_hash[:16] if trial_key_hash else None}..., "
                    f"context.model={context.model if context else None}, "
                    f"context.process_type={context.process_type if context else None}"
                )

                # Compute run_key_hash for benchmark run (needed for cleanup matching)
                from orchestration.jobs.tracking.mlflow_naming import (
                    build_mlflow_run_key,
                    build_mlflow_run_key_hash,
                )
                benchmark_run_key = build_mlflow_run_key(
                    context) if context else None
                benchmark_run_key_hash = build_mlflow_run_key_hash(
                    benchmark_run_key) if benchmark_run_key else None

                # Build and set tags atomically
                tags = build_mlflow_tags(
                    context=context,
                    output_dir=output_dir,
                    parent_run_id=parent_run_id,
                    group_id=group_id,
                    config_dir=config_dir,
                    study_key_hash=study_key_hash,
                    trial_key_hash=trial_key_hash,
                    run_key_hash=benchmark_run_key_hash,  # For cleanup matching
                )

                logger.info(
                    f"[START_BENCHMARK_RUN] Built tags, grouping tags present: "
                    f"study_key_hash={'code.study_key_hash' in tags}, "
                    f"trial_key_hash={'code.trial_key_hash' in tags}, "
                    f"code.model={tags.get('code.model')}, "
                    f"code.stage={tags.get('code.stage')}"
                )
                # Add benchmark-specific tags
                tags["benchmark_source"] = benchmark_source
                tags["benchmarked_model"] = backbone
                tags["mlflow.runType"] = "benchmark"
                mlflow.set_tags(tags)

                # Commit reserved version if auto-increment was used
                if context and run_name:
                    try:
                        import re
                        from orchestration.jobs.tracking.mlflow_index import commit_run_name_version
                        from orchestration.jobs.tracking.mlflow_naming import (
                            build_mlflow_run_key,
                            build_mlflow_run_key_hash,
                            build_counter_key,
                        )
                        from orchestration.jobs.tracking.config.loader import (
                            get_naming_config,
                            get_auto_increment_config,
                        )

                        # Check if auto-increment was used (run name has version suffix like _1, _2, etc.)
                        version_match = re.search(r'_(\d+)$', run_name)
                        if version_match:
                            version = int(version_match.group(1))
                            logger.info(
                                f"[Benchmark Commit] Found version {version} in run name '{run_name}'"
                            )

                            # Check if auto-increment is enabled for benchmarking
                            auto_inc_config = get_auto_increment_config(
                                config_dir, "benchmarking")
                            if auto_inc_config.get("enabled_for_process", False):
                                # Rebuild counter_key from context
                                run_key = build_mlflow_run_key(context)
                                run_key_hash = build_mlflow_run_key_hash(
                                    run_key)
                                naming_config = get_naming_config(config_dir)
                                counter_key = build_counter_key(
                                    naming_config.get(
                                        "project_name", "resume-ner"),
                                    "benchmarking",
                                    run_key_hash,
                                    context.environment or "",
                                )

                                # Infer root_dir from output_dir
                                root_dir = None
                                if output_dir:
                                    current = output_dir
                                    while current.name != "outputs" and current.parent != current:
                                        current = current.parent
                                    root_dir = current.parent if current.name == "outputs" else output_dir.parent.parent.parent

                                if root_dir is None:
                                    root_dir = Path.cwd()

                                logger.info(
                                    f"[Benchmark Commit] Committing version {version} for run {run_id[:12]}..., "
                                    f"counter_key={counter_key[:50]}..."
                                )

                                commit_run_name_version(
                                    counter_key, run_id, version, root_dir, config_dir
                                )
                                logger.info(
                                    f"[Benchmark Commit] ✓ Successfully committed version {version} for benchmark run {run_id[:12]}..."
                                )
                            else:
                                logger.info(
                                    f"[Benchmark Commit] Auto-increment not enabled for benchmarking, skipping commit"
                                )
                        else:
                            logger.info(
                                f"[Benchmark Commit] No version pattern found in run name '{run_name}', "
                                f"skipping commit (auto-increment may not have been used)"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[Benchmark Commit] ✗ Could not commit reserved version for benchmark run: {e}",
                            exc_info=True
                        )

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
                    artifact_uri=benchmark_run.info.artifact_uri,
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

                yield handle
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
                            resource_group = os.environ.get(
                                "AZURE_RESOURCE_GROUP")
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
                                    azureml_run_id_from_uri = run_id_match.group(
                                        1)
                                    azureml_run = workspace.get_run(
                                        azureml_run_id_from_uri)

                            if azureml_run:
                                # Upload artifact using Azure ML SDK
                                file_path = benchmark_json_path.resolve()
                                azureml_run.upload_file(
                                    name="benchmark.json",
                                    path_or_stream=str(file_path)
                                )
                                logger.debug(
                                    f"Uploaded benchmark.json to Azure ML run {azureml_run.id}")
                            else:
                                logger.warning(
                                    "Could not find Azure ML run for benchmark artifact upload")
                        else:
                            logger.warning(
                                "Could not get Azure ML workspace for benchmark artifact upload")
                    except Exception as azureml_err:
                        logger.warning(
                            f"Failed to upload benchmark artifact to Azure ML: {azureml_err}")
                else:
                    # Use standard MLflow for non-Azure ML backends
                    mlflow.log_artifact(str(benchmark_json_path),
                                        artifact_path="benchmark.json")
        except Exception as e:
            logger.warning(f"Could not log benchmark results to MLflow: {e}")



