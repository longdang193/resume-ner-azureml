"""Best configuration selection from local Optuna HPO studies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# Model speed characteristics (parameter count as proxy for inference speed)
MODEL_SPEED_SCORES = {
    "distilbert": 1.0,   # ~66M parameters (baseline)
    "deberta": 2.79,     # ~184M parameters (~2.79x slower)
}


def _import_optuna():
    """Lazy import optuna - only import when actually needed for local execution."""
    try:
        import optuna
        return optuna
    except ImportError as e:
        raise ImportError(
            "optuna is required for local HPO execution. "
            "Install it with: pip install optuna"
        ) from e


def extract_best_config_from_study(
    study: Any,
    backbone: str,
    dataset_version: str,
    objective_metric: str = "macro-f1",
) -> Dict[str, Any]:
    """
    Extract best configuration from an Optuna study.

    Args:
        study: Completed Optuna study.
        backbone: Model backbone name.
        dataset_version: Dataset version string.
        objective_metric: Name of the objective metric (from HPO config).

    Returns:
        Configuration dictionary matching Azure ML format with:
        - backbone
        - hyperparameters
        - metrics
        - selection_criteria
    """
    if study.best_trial is None:
        raise ValueError(
            f"No completed trials found in study '{study.study_name}'")

    best_trial = study.best_trial

    # Extract hyperparameters (exclude backbone and trial_number if present)
    hyperparameters = {
        k: v
        for k, v in best_trial.params.items()
        if k not in ("backbone", "trial_number")
    }

    # Extract metrics
    metrics = {}
    if best_trial.values:
        # Optuna stores objective values in trial.values
        # We'll use the first value as the primary metric
        objective_value = best_trial.values[0] if best_trial.values else None
        if objective_value is not None:
            # Get the objective metric name from study direction
            # For now, we'll use a default name - this should match HPO config
            metrics["objective_value"] = objective_value

    direction = study.direction.name if hasattr(
        study.direction, "name") else "maximize"

    # Extract CV statistics if available
    cv_stats = {}
    if hasattr(best_trial, "user_attrs"):
        cv_mean = best_trial.user_attrs.get("cv_mean")
        cv_std = best_trial.user_attrs.get("cv_std")
        cv_fold_metrics = best_trial.user_attrs.get("cv_fold_metrics")
        if cv_mean is not None:
            cv_stats = {
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "cv_fold_metrics": cv_fold_metrics,
            }

    return {
        "trial_name": f"trial_{best_trial.number}",
        "trial_id": str(best_trial.number),
        "backbone": backbone,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "dataset_version": dataset_version,
        "selection_criteria": {
            "metric": objective_metric,
            "goal": direction,
            "best_value": objective_value if best_trial.values else None,
            "backbone": backbone,
        },
        "cv_statistics": cv_stats if cv_stats else None,
    }


def select_best_configuration_across_studies(
    studies: Optional[Dict[str, Any]] = None,
    hpo_config: Dict[str, Any] = None,
    dataset_version: str = None,
    hpo_output_dir: Optional[Path] = None,
    accuracy_threshold: Optional[float] = None,
    use_relative_threshold: bool = True,
    min_accuracy_gain: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Select the best configuration across multiple backbone studies.

    Supports both in-memory Optuna studies and disk-based selection from saved logs.
    Implements accuracy-speed tradeoff with configurable threshold.

    Args:
        studies: Dictionary mapping backbone names to Optuna studies (optional).
        hpo_config: HPO configuration dictionary (for objective metric).
        dataset_version: Dataset version string.
        hpo_output_dir: Path to HPO outputs directory for disk-based selection (optional).
        accuracy_threshold: Threshold for accuracy-speed tradeoff (optional).
        use_relative_threshold: If True, threshold is relative to best accuracy (default: True).
        min_accuracy_gain: Minimum accuracy gain to justify slower model (optional).

    Returns:
        Best configuration dictionary across all backbones.

    Raises:
        ValueError: If no valid trials found in any study or on disk.
    """
    # If no studies but hpo_output_dir provided, use disk-based selection
    if (studies is None or len(studies) == 0) and hpo_output_dir is not None:
        return select_best_from_disk(
            hpo_output_dir=hpo_output_dir,
            hpo_config=hpo_config,
            dataset_version=dataset_version,
            accuracy_threshold=accuracy_threshold,
            use_relative_threshold=use_relative_threshold,
            min_accuracy_gain=min_accuracy_gain,
        )

    # Otherwise, use Optuna-based selection (enhanced with accuracy-speed tradeoff)
    optuna = _import_optuna()

    objective_metric = hpo_config["objective"]["metric"]
    goal = hpo_config["objective"]["goal"]

    # Get config from hpo_config if not provided
    selection_config = hpo_config.get("selection", {})
    if accuracy_threshold is None:
        accuracy_threshold = selection_config.get("accuracy_threshold")
    if use_relative_threshold is None:
        use_relative_threshold = selection_config.get(
            "use_relative_threshold", True)
    if min_accuracy_gain is None:
        min_accuracy_gain = selection_config.get("min_accuracy_gain")

    # Collect all candidates with their metrics and speed scores
    candidates = []
    for backbone, study in studies.items():
        if study.best_trial is None:
            continue

        trial_value = study.best_trial.values[0] if study.best_trial.values else None
        if trial_value is None:
            continue

        config = extract_best_config_from_study(
            study, backbone, dataset_version, objective_metric)

        # Try to load benchmark data if hpo_output_dir is provided
        speed_score = MODEL_SPEED_SCORES.get(backbone, 10.0)
        speed_data_source = "parameter_proxy"
        benchmark_latency = None

        if hpo_output_dir is not None:
            # Try to find trial directory and load benchmark
            trial_number = study.best_trial.number
            trial_dir = hpo_output_dir / backbone / f"trial_{trial_number}"
            benchmark_latency = load_benchmark_speed_score(trial_dir)

            if benchmark_latency is not None:
                speed_score = benchmark_latency
                speed_data_source = "benchmark"

        candidates.append({
            "backbone": backbone,
            "accuracy": trial_value,
            "config": config,
            "speed_score": speed_score,  # Raw latency or proxy score
            "speed_data_source": speed_data_source,
            "benchmark_latency_ms": benchmark_latency,
        })

    if not candidates:
        error_parts = [
            f"No valid trials found in any study.",
            f"Looking for metric '{objective_metric}' with goal '{goal}'.",
            f"Checked {len(studies)} study/studies: {list(studies.keys())}",
        ]
        for backbone, study in list(studies.items())[:3]:
            completed = len([t for t in study.trials if t.state ==
                            optuna.trial.TrialState.COMPLETE])
            error_parts.append(
                f"  {backbone}: {completed} completed trials"
            )
        raise ValueError("\n".join(error_parts))

    # Normalize speed scores relative to fastest model
    raw_speed_scores = [c["speed_score"] for c in candidates]
    fastest_speed = min(raw_speed_scores)

    for candidate in candidates:
        # Normalize: fastest model gets 1.0, others are relative multiples
        candidate["speed_score"] = candidate["speed_score"] / fastest_speed

    # Sort by accuracy (descending)
    candidates.sort(key=lambda x: x["accuracy"], reverse=True)
    best_candidate = candidates[0]
    best_accuracy = best_candidate["accuracy"]

    # Determine effective threshold
    if accuracy_threshold is not None:
        if use_relative_threshold:
            effective_threshold = best_accuracy * accuracy_threshold
        else:
            effective_threshold = accuracy_threshold
    else:
        effective_threshold = None

    # Apply accuracy-speed tradeoff if threshold is set
    selected = best_candidate
    selection_reason = f"Best accuracy ({best_accuracy:.4f})"

    if accuracy_threshold is not None and len(candidates) > 1:
        # Find fastest candidate within threshold
        faster_candidates = [
            c for c in candidates[1:]
            if c["speed_score"] < best_candidate["speed_score"]
        ]

        for candidate in faster_candidates:
            accuracy_diff = best_accuracy - candidate["accuracy"]

            # Check threshold
            within_threshold = accuracy_diff <= effective_threshold

            # Check minimum gain
            meets_min_gain = True
            if min_accuracy_gain is not None:
                if use_relative_threshold:
                    relative_gain = accuracy_diff / best_accuracy
                    meets_min_gain = relative_gain >= min_accuracy_gain
                else:
                    meets_min_gain = accuracy_diff >= min_accuracy_gain

            if within_threshold and meets_min_gain:
                selected = candidate
                selection_reason = (
                    f"Accuracy within threshold ({accuracy_threshold:.1%} "
                    f"{'relative' if use_relative_threshold else 'absolute'}), "
                    f"preferring faster model ({candidate['backbone']}). "
                    f"Accuracy diff: {accuracy_diff:.4f}"
                )
                break

    # Build result with enhanced metadata
    result = selected["config"]
    result["selection_criteria"]["selection_strategy"] = (
        "accuracy_first_with_threshold" if accuracy_threshold else "accuracy_only"
    )
    result["selection_criteria"]["reason"] = selection_reason

    if accuracy_threshold is not None:
        result["selection_criteria"]["accuracy_threshold"] = accuracy_threshold
        result["selection_criteria"]["use_relative_threshold"] = use_relative_threshold
        result["selection_criteria"]["all_candidates"] = [
            {
                "backbone": c["backbone"],
                "accuracy": c["accuracy"],
                "speed_score": c["speed_score"],
                "speed_data_source": c.get("speed_data_source", "parameter_proxy"),
                "benchmark_latency_ms": c.get("benchmark_latency_ms"),
            }
            for c in candidates
        ]

        # Add speed_data_source for selected model
        result["selection_criteria"]["speed_data_source"] = selected.get(
            "speed_data_source", "parameter_proxy")
        if len(candidates) > 1:
            result["selection_criteria"]["accuracy_diff_from_best"] = (
                best_accuracy - selected["accuracy"]
            )

    return result


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
    except Exception:
        # Return None if benchmark file can't be read or parsed
        return None


def load_best_trial_from_disk(
    hpo_output_dir: Path,
    backbone: str,
    objective_metric: str = "macro-f1",
    hpo_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load best trial configuration from saved HPO outputs on disk.

    Works by reading metrics.json files from trial directories.
    This allows selection even after notebook restart.

    Args:
        hpo_output_dir: Path to HPO outputs directory (e.g., outputs/hpo).
        backbone: Model backbone name.
        objective_metric: Name of the objective metric to optimize.
        hpo_config: Optional HPO config for config-aware study folder discovery.

    Returns:
        Dictionary with best trial info, or None if no trials found.
    """
    backbone_dir = hpo_output_dir / backbone

    if not backbone_dir.exists():
        return None
    
    # Try new structure first (if hpo_config provided): study folders with config-aware matching
    if hpo_config is not None:
        try:
            from orchestration.jobs.local_selection_v2 import (
                find_study_folder_by_config,
                load_best_trial_from_study_folder,
            )
            study_folder = find_study_folder_by_config(
                backbone_dir, hpo_config, backbone
            )
            if study_folder:
                return load_best_trial_from_study_folder(
                    study_folder, objective_metric
                )
        except Exception:
            # Fall through to old structure if v2 functions fail
            pass

    best_metric = None
    best_trial_dir = None
    best_trial_name = None
    best_checkpoint_type = None  # "refit" or "fold"

    # Find all trial directories
    for trial_dir in backbone_dir.iterdir():
        if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
            continue

        # NEW STRUCTURE: Check for refit checkpoint first (preferred)
        refit_dir = trial_dir / "refit"
        refit_metrics_file = refit_dir / "metrics.json"
        if refit_metrics_file.exists():
            try:
                with open(refit_metrics_file, "r") as f:
                    metrics = json.load(f)

                if objective_metric in metrics:
                    metric_value = metrics[objective_metric]
                    if best_metric is None or metric_value > best_metric:
                        best_metric = metric_value
                        best_trial_dir = refit_dir
                        best_trial_name = trial_dir.name
                        best_checkpoint_type = "refit"
            except Exception as e:
                print(f"Warning: Could not read {refit_metrics_file}: {e}")
                continue

        # OLD STRUCTURE: Check for metrics.json at trial root (non-CV or old structure)
        metrics_file = trial_dir / "metrics.json"
        if metrics_file.exists():
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

                    # Only use this if we haven't found a refit checkpoint for this trial
                    # or if this metric is better than existing refit
                    if best_checkpoint_type != "refit" or best_trial_dir is None or best_trial_dir.parent != trial_dir:
                        if best_metric is None or metric_value > best_metric:
                            best_metric = metric_value
                            best_trial_dir = trial_dir
                            best_trial_name = trial_dir.name
                            best_checkpoint_type = "fold"  # or "single" for non-CV
            except Exception as e:
                print(f"Warning: Could not read {metrics_file}: {e}")
                continue

    # If no aggregated trials found, try to aggregate from fold-specific trials
    # NEW STRUCTURE: Check cv/foldN/ directories
    # OLD STRUCTURE: Check trial_<n>_fold<k>/ directories
    if best_trial_dir is None:
        # Group by trial number
        trial_groups = {}
        for trial_dir in backbone_dir.iterdir():
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue

            # NEW STRUCTURE: Check cv/foldN/ subdirectories
            cv_dir = trial_dir / "cv"
            if cv_dir.exists():
                for fold_dir in cv_dir.iterdir():
                    if not fold_dir.is_dir() or not fold_dir.name.startswith("fold"):
                        continue

                    metrics_file = fold_dir / "metrics.json"
                    if not metrics_file.exists():
                        continue

                    trial_base = trial_dir.name
                    try:
                        with open(metrics_file, "r") as f:
                            metrics = json.load(f)

                        if objective_metric in metrics:
                            if trial_base not in trial_groups:
                                trial_groups[trial_base] = []
                            trial_groups[trial_base].append({
                                "metric": metrics[objective_metric],
                                "trial_dir": fold_dir,  # Use fold directory
                                "trial_base_dir": trial_dir,  # Keep reference to trial base
                            })
                    except Exception:
                        continue

            # OLD STRUCTURE: Check trial_<n>_fold<k>/ directories (backward compatibility)
            if "_fold" in trial_dir.name:
                metrics_file = trial_dir / "metrics.json"
                if not metrics_file.exists():
                    continue

                # Extract trial number (e.g., "trial_0" from "trial_0_fold0")
                trial_base = trial_dir.name.split("_fold")[0]

                try:
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)

                    if objective_metric in metrics:
                        if trial_base not in trial_groups:
                            trial_groups[trial_base] = []
                        trial_groups[trial_base].append({
                            "metric": metrics[objective_metric],
                            "trial_dir": trial_dir,
                            "trial_base_dir": None,  # Old structure, no base dir
                        })
                except Exception:
                    continue

        # Find best trial (highest average across folds)
        for trial_base, fold_metrics in trial_groups.items():
            avg_metric = sum(m["metric"]
                             for m in fold_metrics) / len(fold_metrics)

            if best_metric is None or avg_metric > best_metric:
                best_metric = avg_metric
                # Use fold with best individual metric (not first fold)
                best_fold = max(fold_metrics, key=lambda m: m["metric"])
                best_trial_dir = best_fold["trial_dir"]
                best_trial_name = trial_base
                best_checkpoint_type = "fold"

    if best_trial_dir is None:
        return None

    # Load metrics
    metrics_file = best_trial_dir / "metrics.json"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # Determine checkpoint directory
    # For refit: refit/checkpoint/
    # For fold: cv/foldN/checkpoint/ or trial_<n>_fold<k>/checkpoint/
    checkpoint_dir = best_trial_dir / "checkpoint"

    return {
        "backbone": backbone,
        "trial_name": best_trial_name,
        "trial_dir": str(best_trial_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_type": best_checkpoint_type or "unknown",
        "accuracy": best_metric,
        "metrics": metrics,
    }


def select_best_from_disk(
    hpo_output_dir: Path,
    hpo_config: Dict[str, Any],
    dataset_version: str,
    backbones: Optional[List[str]] = None,
    accuracy_threshold: Optional[float] = None,
    use_relative_threshold: bool = True,
    min_accuracy_gain: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Select best configuration from saved HPO outputs on disk.

    This function works entirely from saved metrics.json files,
    so it can be used even after notebook restart.

    Args:
        hpo_output_dir: Path to HPO outputs directory (e.g., outputs/hpo).
        hpo_config: HPO configuration dictionary.
        dataset_version: Dataset version string.
        backbones: List of backbone names to consider (default: all found).
        accuracy_threshold: Threshold for accuracy-speed tradeoff.
        use_relative_threshold: If True, threshold is relative to best accuracy.
        min_accuracy_gain: Minimum accuracy gain to justify slower model.

    Returns:
        Best configuration dictionary matching existing format.

    Raises:
        ValueError: If no valid HPO results found.
    """
    objective_metric = hpo_config["objective"]["metric"]

    # Get config from hpo_config if not provided
    selection_config = hpo_config.get("selection", {})
    if accuracy_threshold is None:
        accuracy_threshold = selection_config.get("accuracy_threshold")
    if use_relative_threshold is None:
        use_relative_threshold = selection_config.get(
            "use_relative_threshold", True)
    if min_accuracy_gain is None:
        min_accuracy_gain = selection_config.get("min_accuracy_gain")

    # Determine backbones to check
    if backbones is None:
        # Auto-detect from directory structure
        backbones = []
        if hpo_output_dir.exists():
            for item in hpo_output_dir.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    backbones.append(item.name)

    # Load best trial for each backbone
    candidates = []
    for backbone in backbones:
        candidate = load_best_trial_from_disk(
            hpo_output_dir, backbone, objective_metric, hpo_config=hpo_config
        )
        if candidate:
            trial_dir = Path(candidate["trial_dir"])

            # Try to load benchmark data, fall back to parameter proxy
            benchmark_latency = load_benchmark_speed_score(trial_dir)
            if benchmark_latency is not None:
                speed_score = benchmark_latency
                speed_data_source = "benchmark"
            else:
                speed_score = MODEL_SPEED_SCORES.get(backbone, 10.0)
                speed_data_source = "parameter_proxy"

            candidates.append({
                "backbone": candidate["backbone"],
                "accuracy": candidate["accuracy"],
                "trial_name": candidate["trial_name"],
                "trial_dir": candidate["trial_dir"],
                "metrics": candidate["metrics"],
                "speed_score": speed_score,  # Raw latency or proxy score
                "speed_data_source": speed_data_source,
                "benchmark_latency_ms": benchmark_latency,  # None if not available
            })

    if not candidates:
        raise ValueError(
            f"No valid HPO results found in {hpo_output_dir}. "
            f"Checked backbones: {backbones}"
        )

    # Normalize speed scores relative to fastest model
    # If using benchmark data, normalize latencies; if using proxy, already normalized
    raw_speed_scores = [c["speed_score"] for c in candidates]
    fastest_speed = min(raw_speed_scores)

    for candidate in candidates:
        # Normalize: fastest model gets 1.0, others are relative multiples
        candidate["speed_score"] = candidate["speed_score"] / fastest_speed

    # Sort by accuracy (descending)
    candidates.sort(key=lambda x: x["accuracy"], reverse=True)
    best_candidate = candidates[0]
    best_accuracy = best_candidate["accuracy"]

    # Determine effective threshold
    if accuracy_threshold is not None:
        if use_relative_threshold:
            effective_threshold = best_accuracy * accuracy_threshold
        else:
            effective_threshold = accuracy_threshold
    else:
        effective_threshold = None

    # Select best configuration
    selected = best_candidate
    selection_reason = f"Best accuracy ({best_accuracy:.4f})"

    if accuracy_threshold is not None and len(candidates) > 1:
        # Find fastest candidate within threshold
        faster_candidates = [
            c for c in candidates[1:]
            if c["speed_score"] < best_candidate["speed_score"]
        ]

        for candidate in faster_candidates:
            accuracy_diff = best_accuracy - candidate["accuracy"]

            # Check threshold
            within_threshold = accuracy_diff <= effective_threshold

            # Check minimum gain
            meets_min_gain = True
            if min_accuracy_gain is not None:
                if use_relative_threshold:
                    relative_gain = accuracy_diff / best_accuracy
                    meets_min_gain = relative_gain >= min_accuracy_gain
                else:
                    meets_min_gain = accuracy_diff >= min_accuracy_gain

            if within_threshold and meets_min_gain:
                selected = candidate
                selection_reason = (
                    f"Accuracy within threshold ({accuracy_threshold:.1%} "
                    f"{'relative' if use_relative_threshold else 'absolute'}), "
                    f"preferring faster model ({candidate['backbone']}). "
                    f"Accuracy diff: {accuracy_diff:.4f}"
                )
                break

    # Build result dictionary (matching format from Optuna-based selection)
    # Note: Hyperparameters are not available from disk, so we leave empty
    # They can be loaded separately if needed from checkpoint configs

    # Build selection criteria
    selection_criteria = {
        "metric": objective_metric,
        "goal": "maximize",
        "best_value": selected["accuracy"],
        "backbone": selected["backbone"],
        "selection_strategy": (
            "accuracy_first_with_threshold" if accuracy_threshold else "accuracy_only"
        ),
        "reason": selection_reason,
    }

    if accuracy_threshold is not None:
        selection_criteria["accuracy_threshold"] = accuracy_threshold
        selection_criteria["use_relative_threshold"] = use_relative_threshold
        selection_criteria["all_candidates"] = [
            {
                "backbone": c["backbone"],
                "accuracy": c["accuracy"],
                "speed_score": c["speed_score"],
                "speed_data_source": c.get("speed_data_source", "parameter_proxy"),
                "benchmark_latency_ms": c.get("benchmark_latency_ms"),
            }
            for c in candidates
        ]

        # Add speed_data_source for selected model
        selection_criteria["speed_data_source"] = selected.get(
            "speed_data_source", "parameter_proxy")
        if len(candidates) > 1:
            selection_criteria["accuracy_diff_from_best"] = (
                best_accuracy - selected["accuracy"]
            )

    return {
        "trial_name": selected["trial_name"],
        "trial_id": selected["trial_name"],  # Use trial name as ID
        "backbone": selected["backbone"],
        "hyperparameters": {},  # Empty - not available from disk
        "metrics": selected["metrics"],
        "dataset_version": dataset_version,
        "selection_criteria": selection_criteria,
    }
