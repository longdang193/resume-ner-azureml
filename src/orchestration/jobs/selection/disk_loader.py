"""Load trial data from disk-based HPO outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from shared.logging_utils import get_logger
from orchestration.paths import parse_hpo_path_v2, is_v2_path

logger = get_logger(__name__)


def load_benchmark_speed_score(trial_dir: Path) -> Optional[float]:
    """
    Load speed score from benchmark.json if available.

    Args:
        trial_dir: Path to trial directory containing benchmark.json.

    Returns:
        Latency in milliseconds (batch_size=1 mean), or None if not available.
    """
    benchmark_file = trial_dir / "benchmark.json"

    if not benchmark_file.exists():
        return None

    try:
        with open(benchmark_file, "r") as f:
            benchmark = json.load(f)

        # Extract batch_1 mean latency
        batch_1_data = benchmark.get("batch_1", {})
        if isinstance(batch_1_data, dict) and "mean_ms" in batch_1_data:
            return float(batch_1_data["mean_ms"])

        return None
    except Exception as e:
        logger.debug(f"Could not read benchmark file {benchmark_file}: {e}")
        return None


def load_best_trial_from_disk(
    hpo_output_dir: Path,
    backbone: str,
    objective_metric: str = "macro-f1",
) -> Optional[Dict[str, Any]]:
    """
    Load best trial configuration from saved HPO outputs on disk.

    Works by reading metrics.json files from trial directories.
    This allows selection even after notebook restart.

    Supports both v2 paths (study-{study8}/trial-{trial8}) and legacy paths (trial_N).

    Args:
        hpo_output_dir: Path to HPO outputs directory (e.g., outputs/hpo).
        backbone: Model backbone name.
        objective_metric: Name of the objective metric to optimize.

    Returns:
        Dictionary with best trial info, or None if no trials found.
    """
    # Handle case where hpo_output_dir already includes backbone (e.g., outputs/hpo/local/distilbert)
    if hpo_output_dir.name == backbone:
        backbone_dir = hpo_output_dir
    else:
        backbone_dir = hpo_output_dir / backbone

    if not backbone_dir.exists():
        return None

    best_metric = None
    best_trial_dir = None
    best_trial_name = None

    # Collect all trial directories (support both v2 and legacy paths)
    trial_dirs = []

    # First, check for v2 paths: study-{study8}/trial-{trial8}
    for study_dir in backbone_dir.iterdir():
        if not study_dir.is_dir():
            continue

        # Check if this is a v2 study folder (study-{hash})
        if study_dir.name.startswith("study-") and len(study_dir.name) > 7:
            # Look for trial folders inside study folder
            for trial_dir in study_dir.iterdir():
                if trial_dir.is_dir() and trial_dir.name.startswith("trial-") and len(trial_dir.name) > 7:
                    trial_dirs.append(trial_dir)
        # Also check for legacy study folders (hpo_{backbone}_*)
        elif study_dir.name.startswith(f"hpo_{backbone}_"):
            # Legacy study folder - look for trial_N directories inside
            for trial_dir in study_dir.iterdir():
                if trial_dir.is_dir() and trial_dir.name.startswith("trial_"):
                    trial_dirs.append(trial_dir)

    # Also check for legacy direct trial directories (trial_N directly under backbone_dir)
    for trial_dir in backbone_dir.iterdir():
        if trial_dir.is_dir() and trial_dir.name.startswith("trial_") and trial_dir not in trial_dirs:
            trial_dirs.append(trial_dir)

    # Find all trial directories
    for trial_dir in trial_dirs:

        # Check for metrics.json
        metrics_file = trial_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        # Read metrics
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            if objective_metric in metrics:
                metric_value = metrics[objective_metric]

                # For k-fold CV, we want the average across folds
                # Check if this is a fold-specific trial or aggregated
                if "_fold" in trial_dir.name:
                    # This is a fold-specific trial - we'll aggregate later
                    continue

                if best_metric is None or metric_value > best_metric:
                    best_metric = metric_value
                    best_trial_dir = trial_dir
                    best_trial_name = trial_dir.name
        except Exception as e:
            logger.warning(f"Could not read {metrics_file}: {e}")
            continue

    # If no aggregated trials found, try to aggregate from fold-specific trials
    if best_trial_dir is None:
        # Group by trial base (e.g., trial_0_20251223_123456_fold0, trial_0_20251223_123456_fold1 -> trial_0_20251223_123456)
        # Handles both old format (trial_N_foldX) and new format (trial_N_RUNID_foldX)
        trial_groups = {}
        # Use the same trial_dirs collection we built earlier
        for trial_dir in trial_dirs:

            metrics_file = trial_dir / "metrics.json"
            if not metrics_file.exists():
                continue

            # Extract trial base (e.g., "trial_0" or "trial_0_20251223_123456" from "trial_0_20251223_123456_fold0")
            # Handle both old format (trial_N_foldX) and new format (trial_N_RUNID_foldX)
            if "_fold" in trial_dir.name:
                trial_base = trial_dir.name.split("_fold")[0]
            else:
                # No fold suffix, use full name
                trial_base = trial_dir.name

            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                if objective_metric in metrics:
                    if trial_base not in trial_groups:
                        trial_groups[trial_base] = []
                    trial_groups[trial_base].append({
                        "metric": metrics[objective_metric],
                        "trial_dir": trial_dir,
                    })
            except Exception:
                continue

        # Find best trial (highest average across folds)
        for trial_base, fold_metrics in trial_groups.items():
            avg_metric = sum(m["metric"]
                             for m in fold_metrics) / len(fold_metrics)

            if best_metric is None or avg_metric > best_metric:
                best_metric = avg_metric
                # Use first fold's directory as representative
                best_trial_dir = fold_metrics[0]["trial_dir"]
                best_trial_name = trial_base

    if best_trial_dir is None:
        return None

    # Load metrics
    metrics_file = best_trial_dir / "metrics.json"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # Try to read trial_meta.json from trial root (not subfolders) for run_id
    trial_run_id = None
    trial_meta_path = best_trial_dir / "trial_meta.json"
    if trial_meta_path.exists():
        try:
            import re
            with open(trial_meta_path, "r") as f:
                trial_meta = json.load(f)
            if "run_id" in trial_meta:
                run_id_from_meta = trial_meta["run_id"]
                # Validate: MLflow run IDs are UUIDs (e.g., "0caed3f7-bf68-4b52-a1bb-bc53b7344ae4")
                # Skip if it looks like a timestamp (e.g., "20260105_114431")
                uuid_pattern = re.compile(
                    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                    re.IGNORECASE
                )
                if uuid_pattern.match(run_id_from_meta):
                    trial_run_id = run_id_from_meta
                    logger.debug(
                        f"Read valid trial_run_id (UUID) from trial_meta.json: {trial_run_id[:12]}..."
                    )
                else:
                    logger.debug(
                        f"Skipping run_id from trial_meta.json (not a UUID): {run_id_from_meta}"
                    )
        except Exception as e:
            logger.debug(f"Could not read trial_meta.json: {e}")

    result = {
        "backbone": backbone,
        "trial_name": best_trial_name,
        "trial_dir": str(best_trial_dir),
        "accuracy": best_metric,
        "metrics": metrics,
    }

    # Add trial_run_id if found (don't assume sweep_run_id or refit_run_id exist)
    if trial_run_id:
        result["trial_run_id"] = trial_run_id

    return result
