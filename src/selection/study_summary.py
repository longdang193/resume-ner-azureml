"""Utilities for displaying and summarizing HPO study results.

This module provides functions to load Optuna studies, extract trial information,
and format summaries for display in notebooks or logs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from shared.logging_utils import get_logger

from hpo.utils.paths import resolve_hpo_output_dir
from orchestration.jobs.hpo.local.checkpoint.manager import resolve_storage_path

logger = get_logger(__name__)


def extract_cv_statistics(best_trial: Any) -> Optional[Tuple[float, float]]:
    """Extract CV statistics from Optuna trial user attributes.

    Args:
        best_trial: Optuna trial object with user_attrs.

    Returns:
        Tuple of (cv_mean, cv_std) if available, else None.
    """
    if not hasattr(best_trial, "user_attrs"):
        return None
    cv_mean = best_trial.user_attrs.get("cv_mean")
    cv_std = best_trial.user_attrs.get("cv_std")
    return (cv_mean, cv_std) if cv_mean is not None else None


def get_trial_hash_info(trial_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Extract study_key_hash, trial_key_hash, and trial_number from trial_meta.json if available.

    Args:
        trial_dir: Path to trial directory containing trial_meta.json.

    Returns:
        Tuple of (study_key_hash, trial_key_hash, trial_number), or (None, None, None) if not found.
    """
    trial_meta_path = trial_dir / "trial_meta.json"
    if not trial_meta_path.exists():
        return None, None, None
    try:
        with open(trial_meta_path, "r") as f:
            meta = json.load(f)
        return (
            meta.get("study_key_hash"),
            meta.get("trial_key_hash"),
            meta.get("trial_number"),
        )
    except Exception:
        return None, None, None


def load_study_from_disk(
    backbone_name: str,
    root_dir: Path,
    environment: str,
    hpo_config: Dict[str, Any],
) -> Optional[Any]:
    """Load Optuna study from disk if not in memory.

    Args:
        backbone_name: Model backbone name (e.g., "distilbert").
        root_dir: Project root directory.
        environment: Platform environment (e.g., "local", "colab").
        hpo_config: HPO configuration dictionary.

    Returns:
        Optuna study object if found, else None.
    """
    backbone_output_dir = root_dir / "outputs" / "hpo" / environment / backbone_name
    hpo_backbone_dir = resolve_hpo_output_dir(backbone_output_dir)
    if not hpo_backbone_dir.exists():
        return None

    checkpoint_config = hpo_config.get("checkpoint", {})
    study_name_template = checkpoint_config.get(
        "study_name") or hpo_config.get("study_name")
    study_name = None
    if study_name_template:
        study_name = study_name_template.replace("{backbone}", backbone_name)

    # Find v2 study folder
    from .trial_finder import find_study_folder_in_backbone_dir
    study_folder = find_study_folder_in_backbone_dir(backbone_output_dir)

    if not study_folder or not study_folder.exists():
        return None

    study_db_path = study_folder / "study.db"
    if not study_db_path.exists():
        return None

    try:
        from orchestration.jobs.hpo.local.optuna.integration import import_optuna

        optuna, _, _, _ = import_optuna()
    except ImportError:
        import optuna

    try:
        storage_uri = f"sqlite:///{study_db_path.resolve()}"
        study = optuna.load_study(
            study_name=study_folder.name, storage=storage_uri)
        return study
    except Exception as e:
        logger.debug(f"Could not load study for {backbone_name}: {e}")
        return None


def find_trial_hash_info_for_study(
    backbone_name: str,
    trial_number: int,
    root_dir: Path,
    environment: str,
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Find trial hash info for a specific trial number in a study.

    Args:
        backbone_name: Model backbone name (e.g., "distilbert").
        trial_number: Optuna trial number.
        root_dir: Project root directory.
        environment: Platform environment (e.g., "local", "colab").

    Returns:
        Tuple of (study_key_hash, trial_key_hash, trial_number), or (None, None, None) if not found.
    """
    backbone_dir = root_dir / "outputs" / "hpo" / \
        environment / backbone_name.split("-")[0]
    if not backbone_dir.exists():
        return None, None, None

    for study_folder in backbone_dir.iterdir():
        if not study_folder.is_dir() or not study_folder.name.startswith("study-"):
            continue
        for trial_dir in study_folder.iterdir():
            if (
                trial_dir.is_dir()
                and trial_dir.name.startswith("trial-")
            ):
                # Check trial_meta.json for trial_number match
                trial_meta_path = trial_dir / "trial_meta.json"
                if trial_meta_path.exists():
                    try:
                        with open(trial_meta_path, "r") as f:
                            meta = json.load(f)
                        if meta.get("trial_number") == trial_number:
                            return get_trial_hash_info(trial_dir)
                    except Exception:
                        continue
    return None, None, None


def format_study_summary_line(
    backbone: str,
    num_trials: int,
    best_metric_value: float,
    objective_metric: str,
    study_key_hash: Optional[str] = None,
    trial_key_hash: Optional[str] = None,
    trial_number: Optional[int] = None,
    cv_stats: Optional[Tuple[float, float]] = None,
    from_disk: bool = False,
) -> str:
    """Format a single line summary for an HPO study.

    Args:
        backbone: Model backbone name.
        num_trials: Number of trials in the study.
        best_metric_value: Best metric value from the study.
        objective_metric: Name of the objective metric.
        study_key_hash: Optional study key hash (first 8 chars).
        trial_key_hash: Optional trial key hash (first 8 chars).
        trial_number: Optional trial number.
        cv_stats: Optional tuple of (cv_mean, cv_std).
        from_disk: Whether the study was loaded from disk.

    Returns:
        Formatted summary line string.
    """
    # Format hash info
    if study_key_hash and trial_key_hash:
        hash_info = f" [study-{study_key_hash[:8]}, trial-{trial_key_hash[:8]}, t{trial_number}]"
    elif trial_number is not None:
        hash_info = f" [t{trial_number}]"
    else:
        hash_info = ""

    disk_suffix = " (from disk)" if from_disk else ""
    line = f"📊 {backbone}: {num_trials} trials, best {objective_metric}={best_metric_value:.4f}{hash_info}{disk_suffix}"

    if cv_stats:
        cv_mean, cv_std = cv_stats
        line += f"\n   CV: {cv_mean:.4f} ± {cv_std:.4f}"

    return line


def print_study_summaries(
    hpo_studies: Optional[Dict[str, Any]],
    backbone_values: list[str],
    hpo_config: Dict[str, Any],
    root_dir: Path,
    environment: str,
) -> None:
    """Print formatted summaries for HPO studies.

    Processes both in-memory studies and studies loaded from disk,
    displaying hash-based identifiers and CV statistics when available.

    Args:
        hpo_studies: Optional dictionary mapping backbone -> Optuna study.
        backbone_values: List of backbone model names to process.
        hpo_config: HPO configuration dictionary.
        root_dir: Project root directory.
        environment: Platform environment (e.g., "local", "colab").
    """
    objective_metric = hpo_config["objective"]["metric"]
    processed_backbones = set()

    # Process in-memory studies
    if hpo_studies:
        for backbone, study in hpo_studies.items():
            if not study or not study.trials:
                continue

            processed_backbones.add(backbone)
            best_trial = study.best_trial
            cv_stats = extract_cv_statistics(best_trial)
            trial_number = best_trial.number

            # Get hash info from trial directory
            study_key_hash, trial_key_hash, _ = find_trial_hash_info_for_study(
                backbone, trial_number, root_dir, environment
            )

            summary_line = format_study_summary_line(
                backbone=backbone,
                num_trials=len(study.trials),
                best_metric_value=best_trial.value,
                objective_metric=objective_metric,
                study_key_hash=study_key_hash,
                trial_key_hash=trial_key_hash,
                trial_number=trial_number,
                cv_stats=cv_stats,
                from_disk=False,
            )
            print(summary_line)

    # Process remaining backbones from disk
    for backbone in backbone_values:
        if backbone in processed_backbones:
            continue

        study = load_study_from_disk(
            backbone, root_dir, environment, hpo_config)
        if not study or not study.trials:
            continue

        best_trial = study.best_trial
        cv_stats = extract_cv_statistics(best_trial)
        trial_number = best_trial.number

        # Get hash info
        study_key_hash, trial_key_hash, _ = find_trial_hash_info_for_study(
            backbone, trial_number, root_dir, environment
        )

        summary_line = format_study_summary_line(
            backbone=backbone,
            num_trials=len(study.trials),
            best_metric_value=best_trial.value,
            objective_metric=objective_metric,
            study_key_hash=study_key_hash,
            trial_key_hash=trial_key_hash,
            trial_number=trial_number,
            cv_stats=cv_stats,
            from_disk=True,
        )
        print(summary_line)
