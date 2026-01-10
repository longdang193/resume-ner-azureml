"""Configuration selection utilities.

This module provides functionality for selecting the best configuration from HPO results,
supporting both local (Optuna) and Azure ML selection methods.
"""

from __future__ import annotations

from .disk_loader import load_benchmark_speed_score, load_best_trial_from_disk
from .local_selection import (
    extract_best_config_from_study,
    load_best_trial_from_disk as load_best_trial,
    select_best_configuration_across_studies,
)
from .mlflow_selection import find_best_model_from_mlflow
from .artifact_acquisition import acquire_best_model_checkpoint
from .selection import select_best_configuration
from .cache import (
    compute_selection_cache_key,
    load_cached_best_model,
    save_best_model_cache,
)
from .trial_finder import (
    find_best_trials_for_backbones,
    find_study_folder_in_backbone_dir,
)
from .study_summary import (
    extract_cv_statistics,
    get_trial_hash_info,
    load_study_from_disk,
    find_trial_hash_info_for_study,
    format_study_summary_line,
    print_study_summaries,
)

# Alias for backward compatibility
select_production_configuration = select_best_configuration
from .selection_logic import MODEL_SPEED_SCORES, SelectionLogic

__all__ = [
    # Disk loading
    "load_benchmark_speed_score",
    "load_best_trial_from_disk",
    "load_best_trial",
    # Local selection
    "extract_best_config_from_study",
    "select_best_configuration_across_studies",
    # Azure ML selection
    "select_production_configuration",
    # MLflow selection
    "find_best_model_from_mlflow",
    # Artifact acquisition
    "acquire_best_model_checkpoint",
    # Trial finding
    "find_best_trials_for_backbones",
    "find_study_folder_in_backbone_dir",
    # Selection logic
    "MODEL_SPEED_SCORES",
    "SelectionLogic",
    # Cache management
    "compute_selection_cache_key",
    "load_cached_best_model",
    "save_best_model_cache",
    # Study summary
    "extract_cv_statistics",
    "get_trial_hash_info",
    "load_study_from_disk",
    "find_trial_hash_info_for_study",
    "format_study_summary_line",
    "print_study_summaries",
]






