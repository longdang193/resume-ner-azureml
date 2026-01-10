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
        hpo_trial_run_id: Optional[str] = None,
        hpo_refit_run_id: Optional[str] = None,
        hpo_sweep_run_id: Optional[str] = None,
    ):
        """
        Start a MLflow run for benchmarking.

        Run names are UI sugar only. All parent-child relationships are tracked via tags
        (`code.lineage.*`). Never parse run names for logic.

        IMPORTANT: On AzureML, `parentRunId` field cannot be set for runs created in separate
        processes (subprocesses). Parent-child relationships are tracked exclusively via
        `code.lineage.*` tags, which are the authoritative source of truth.

        Args:
            run_name: Name for the benchmark run.
            backbone: Model backbone name.
            benchmark_source: Source of benchmark ("hpo_trial" or "final_training").
            context: Optional NamingContext for tag-based identification.
            output_dir: Optional output directory for metadata persistence.
            parent_run_id: Optional parent MLflow run ID (legacy, for backward compatibility).
            group_id: Optional group/session identifier.
            study_key_hash: Optional study key hash from HPO trial (for grouping tags).
            trial_key_hash: Optional trial key hash from HPO trial (for grouping tags).
            hpo_trial_run_id: Optional HPO trial run ID (CV trial run, lineage parent).
            hpo_refit_run_id: Optional HPO refit run ID (refit run, artifact parent).
            hpo_sweep_run_id: Optional HPO sweep run ID (HPO parent, optional).

        Yields:
            RunHandle with run information.
        """
        try:
            # Infer config_dir from output_dir
            config_dir = None
            if output_dir:
                # Derive project root correctly by finding "outputs" directory
                root_dir_for_config = None
                current = output_dir
                while current.parent != current:  # Stop at filesystem root
                    if current.name == "outputs":
                        root_dir_for_config = current.parent
                        break
                    current = current.parent
                
                if root_dir_for_config is None:
                    # Fallback: try to find project root by looking for config directory
                    root_dir_for_config = Path.cwd()
                    for candidate in [Path.cwd(), Path.cwd().parent]:
                        if (candidate / "config").exists():
                            root_dir_for_config = candidate
                            break
                
                config_dir = root_dir_for_config / "config" if root_dir_for_config else None

            # Check if tracking is enabled for benchmark stage
            from orchestration.jobs.tracking.config.loader import get_tracking_config
            tracking_config = get_tracking_config(config_dir=config_dir, stage="benchmark")
            if not tracking_config.get("enabled", True):
                logger.info("[Benchmark Tracker] MLflow tracking disabled for benchmark stage (tracking.benchmark.enabled=false)")
                from contextlib import nullcontext
                with nullcontext():
                    yield None
                return

            # Validate run IDs are UUIDs (not timestamps) BEFORE creating run
            import re
            uuid_pattern = re.compile(
                r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                re.IGNORECASE
            )

            valid_trial_run_id = hpo_trial_run_id if (
                hpo_trial_run_id and uuid_pattern.match(hpo_trial_run_id)) else None
            valid_refit_run_id = hpo_refit_run_id if (
                hpo_refit_run_id and uuid_pattern.match(hpo_refit_run_id)) else None
            valid_sweep_run_id = hpo_sweep_run_id if (
                hpo_sweep_run_id and uuid_pattern.match(hpo_sweep_run_id)) else None

            # Determine parent run ID for lineage (priority: trial > refit > sweep)
            lineage_parent_run_id = valid_trial_run_id or valid_refit_run_id or valid_sweep_run_id
            parent_kind = (
                "trial" if valid_trial_run_id
                else ("refit" if valid_refit_run_id
                      else ("sweep" if valid_sweep_run_id else ""))
            )

            with mlflow.start_run(run_name=run_name) as benchmark_run:
                run_id = benchmark_run.info.run_id
                experiment_id = benchmark_run.info.experiment_id
                tracking_uri = mlflow.get_tracking_uri()

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

                # Use pre-computed lineage_parent_run_id and parent_kind from before run creation

                # Get tag keys from registry (using centralized helpers)
                from orchestration.jobs.tracking.naming.tag_keys import (
                    get_lineage_hpo_refit_run_id,
                    get_lineage_hpo_sweep_run_id,
                    get_lineage_hpo_trial_run_id,
                    get_lineage_parent_training_run_id,
                )
                # Note: config_dir was already inferred from output_dir at line 104, don't overwrite it
                lineage_hpo_trial_run_id_tag = get_lineage_hpo_trial_run_id(config_dir)
                lineage_hpo_refit_run_id_tag = get_lineage_hpo_refit_run_id(config_dir)
                lineage_hpo_sweep_run_id_tag = get_lineage_hpo_sweep_run_id(config_dir)
                lineage_parent_training_run_id_tag = get_lineage_parent_training_run_id(config_dir)
                
                # Set explicit lineage tags using code.lineage.* namespace (only valid UUIDs)
                if valid_trial_run_id:
                    tags[lineage_hpo_trial_run_id_tag] = valid_trial_run_id
                if valid_refit_run_id:
                    tags[lineage_hpo_refit_run_id_tag] = valid_refit_run_id
                if valid_sweep_run_id:
                    tags[lineage_hpo_sweep_run_id_tag] = valid_sweep_run_id

                # Canonical parent tags (only if we have a valid parent)
                if lineage_parent_run_id:
                    tags[lineage_parent_training_run_id_tag] = lineage_parent_run_id
                    # parent_kind is not in registry, keep as-is for now
                    tags["code.lineage.parent_kind"] = parent_kind
                    logger.info(
                        f"[START_BENCHMARK_RUN] Set lineage tags: parent_run_id={lineage_parent_run_id[:12]}..., "
                        f"parent_kind={parent_kind}"
                    )
                else:
                    logger.warning(
                        f"[START_BENCHMARK_RUN] No valid parent run IDs found. "
                        f"Received: trial={hpo_trial_run_id[:20] if hpo_trial_run_id else None}, "
                        f"refit={hpo_refit_run_id[:20] if hpo_refit_run_id else None}, "
                        f"sweep={hpo_sweep_run_id[:20] if hpo_sweep_run_id else None}. "
                        f"These may be timestamps or None - will query MLflow if trial_key_hash available."
                    )

                # Add benchmark-specific tags
                tags["benchmark_source"] = benchmark_source
                tags["benchmarked_model"] = backbone
                tags["mlflow.runType"] = "benchmark"
                # Set run name as tag (MLflow version compatibility)
                tags["mlflow.runName"] = run_name
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
                                    while current.parent != current:  # Stop at filesystem root
                                        if current.name == "outputs":
                                            root_dir = current.parent
                                            break
                                        current = current.parent

                                if root_dir is None:
                                    # Fallback: try to find project root by looking for config directory
                                    root_dir = Path.cwd()
                                    for candidate in [Path.cwd(), Path.cwd().parent]:
                                        if (candidate / "config").exists():
                                            root_dir = candidate
                                            break

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
                        # Derive project root correctly by finding "outputs" directory
                        root_dir = None
                        if output_dir:
                            current = output_dir
                            while current.parent != current:  # Stop at filesystem root
                                if current.name == "outputs":
                                    root_dir = current.parent
                                    break
                                current = current.parent
                        
                        if root_dir is None:
                            # Fallback: try to find project root by looking for config directory
                            root_dir = Path.cwd()
                            for candidate in [Path.cwd(), Path.cwd().parent]:
                                if (candidate / "config").exists():
                                    root_dir = candidate
                                    break
                        
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

            # Log artifact using MLflow (works for both Azure ML and non-Azure ML backends)
            # Check if artifact logging is enabled
            config_dir = None
            if benchmark_json_path:
                # Try to infer config_dir from benchmark_json_path
                current = benchmark_json_path.parent
                while current.parent != current:
                    if current.name == "outputs":
                        root_dir_for_config = current.parent
                        config_dir = root_dir_for_config / "config" if root_dir_for_config else None
                        break
                    current = current.parent
            
            from orchestration.jobs.tracking.config.loader import get_tracking_config
            from infrastructure.tracking.mlflow import log_artifact_safe
            tracking_config = get_tracking_config(config_dir=config_dir, stage="benchmark")
            if tracking_config.get("log_artifacts", True) and benchmark_json_path.exists():
                log_artifact_safe(
                    local_path=benchmark_json_path,
                    artifact_path="benchmark.json",
                    run_id=None,  # Use active run
                )
            elif not tracking_config.get("log_artifacts", True):
                logger.debug("[Benchmark Tracker] Artifact logging disabled (tracking.benchmark.log_artifacts=false)")
        except Exception as e:
            logger.warning(f"Could not log benchmark results to MLflow: {e}")
