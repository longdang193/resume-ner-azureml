"""Load trial data from disk-based HPO outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from shared.logging_utils import get_logger

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

    Args:
        hpo_output_dir: Path to HPO outputs directory (e.g., outputs/hpo).
        backbone: Model backbone name.
        objective_metric: Name of the objective metric to optimize.

    Returns:
        Dictionary with best trial info, or None if no trials found.
    """
    backbone_dir = hpo_output_dir / backbone

    if not backbone_dir.exists():
        return None

    best_metric = None
    best_trial_dir = None
    best_trial_name = None

    # Find all trial directories
    for trial_dir in backbone_dir.iterdir():
        if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
            continue

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
        for trial_dir in backbone_dir.iterdir():
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue

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

    return {
        "backbone": backbone,
        "trial_name": best_trial_name,
        "trial_dir": str(best_trial_dir),
        "accuracy": best_metric,
        "metrics": metrics,
    }

