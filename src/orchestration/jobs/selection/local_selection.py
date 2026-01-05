"""Best configuration selection from local Optuna HPO studies.

This module provides a facade for configuration selection, delegating to
specialized modules for study extraction, disk loading, and selection logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.logging_utils import get_logger

from .disk_loader import load_best_trial_from_disk, load_benchmark_speed_score
from ..errors import SelectionError
from .selection_logic import MODEL_SPEED_SCORES, SelectionLogic
from ..hpo.study_extractor import extract_best_config_from_study

logger = get_logger(__name__)


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


# Re-export for backward compatibility
__all__ = [
    "extract_best_config_from_study",
    "select_best_configuration_across_studies",
    "load_best_trial_from_disk",
    "load_benchmark_speed_score",
    "select_best_from_disk",
    "MODEL_SPEED_SCORES",
]


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
        SelectionError: If no valid trials found in any study or on disk.
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
            backbone_dir = hpo_output_dir / backbone
            # Try to find trial directory - could be trial_N, trial_N_RUNID, or trial_N_RUNID_foldX
            trial_dir = None
            if backbone_dir.exists():
                # Look for directories matching trial_{number} or trial_{number}_*
                for candidate_dir in backbone_dir.iterdir():
                    if candidate_dir.is_dir() and candidate_dir.name.startswith(f"trial_{trial_number}"):
                        # Prefer non-fold directories, but accept fold directories if that's all we have
                        if "_fold" not in candidate_dir.name:
                            trial_dir = candidate_dir
                            break
                        elif trial_dir is None:
                            trial_dir = candidate_dir
            # Fallback to old format for backward compatibility
            if trial_dir is None:
                trial_dir = backbone_dir / f"trial_{trial_number}"
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
        raise SelectionError("\n".join(error_parts))

    # Use SelectionLogic to select best configuration
    return SelectionLogic.select_best(
        candidates, accuracy_threshold, use_relative_threshold, min_accuracy_gain
    )


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
        SelectionError: If no valid HPO results found.
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
            hpo_output_dir, backbone, objective_metric
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

            # Build config dictionary matching Optuna-based format
            config = {
                "trial_name": candidate["trial_name"],
                "trial_id": candidate["trial_name"],
                "backbone": candidate["backbone"],
                "hyperparameters": {},  # Empty - not available from disk
                "metrics": candidate["metrics"],
                "dataset_version": dataset_version,
                "selection_criteria": {
                    "metric": objective_metric,
                    "goal": "maximize",
                    "best_value": candidate["accuracy"],
                    "backbone": candidate["backbone"],
                },
            }

            candidates.append({
                "backbone": candidate["backbone"],
                "accuracy": candidate["accuracy"],
                "config": config,
                "speed_score": speed_score,  # Raw latency or proxy score
                "speed_data_source": speed_data_source,
                "benchmark_latency_ms": benchmark_latency,  # None if not available
            })

    if not candidates:
        raise SelectionError(
            f"No valid HPO results found in {hpo_output_dir}. "
            f"Checked backbones: {backbones}"
        )

    # Use SelectionLogic to select best configuration
    return SelectionLogic.select_best(
        candidates, accuracy_threshold, use_relative_threshold, min_accuracy_gain
    )
