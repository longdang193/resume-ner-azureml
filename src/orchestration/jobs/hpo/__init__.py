"""Hyperparameter Optimization (HPO) utilities.

This module provides HPO functionality for both local (Optuna) and Azure ML execution.
"""

from __future__ import annotations

from .local.checkpoint.manager import get_storage_uri, resolve_storage_path
from .hpo_helpers import (
    create_mlflow_run_name,
    create_study_name,
    generate_run_id,
    setup_checkpoint_storage,
)
from .local_sweeps import run_local_hpo_sweep, translate_search_space_to_optuna
from .search_space import (
    SearchSpaceTranslator,
    create_search_space,
    translate_search_space_to_optuna as translate_search_space,
)
from .study_extractor import extract_best_config_from_study
from .azureml.sweeps import (
    create_dry_run_sweep_job_for_backbone,
    create_hpo_sweep_job_for_backbone,
    validate_sweep_job,
)
from .local.trial.execution import TrialExecutor

__all__ = [
    # Checkpoint management
    "get_storage_uri",
    "resolve_storage_path",
    # HPO helpers
    "create_mlflow_run_name",
    "create_study_name",
    "generate_run_id",
    "setup_checkpoint_storage",
    # Local sweeps
    "run_local_hpo_sweep",
    "translate_search_space_to_optuna",
    # Search space
    "SearchSpaceTranslator",
    "create_search_space",
    "translate_search_space",
    # Study extraction
    "extract_best_config_from_study",
    # Azure ML sweeps
    "create_dry_run_sweep_job_for_backbone",
    "create_hpo_sweep_job_for_backbone",
    "validate_sweep_job",
    # Trial execution
    "TrialExecutor",
]
