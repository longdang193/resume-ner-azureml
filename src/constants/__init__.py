"""Shared constants module.

This module provides stable orchestration identifiers shared across notebooks and scripts.
"""

from .orchestration import (
    STAGE_SMOKE,
    STAGE_HPO,
    STAGE_TRAINING,
    EXPERIMENT_NAME,
    MODEL_NAME,
    PROD_STAGE,
    CONVERSION_JOB_NAME,
    METRICS_FILENAME,
    BENCHMARK_FILENAME,
    CHECKPOINT_DIRNAME,
    OUTPUTS_DIRNAME,
    MLRUNS_DIRNAME,
    DEFAULT_RANDOM_SEED,
    DEFAULT_K_FOLDS,
)

__all__ = [
    "STAGE_SMOKE",
    "STAGE_HPO",
    "STAGE_TRAINING",
    "EXPERIMENT_NAME",
    "MODEL_NAME",
    "PROD_STAGE",
    "CONVERSION_JOB_NAME",
    "METRICS_FILENAME",
    "BENCHMARK_FILENAME",
    "CHECKPOINT_DIRNAME",
    "OUTPUTS_DIRNAME",
    "MLRUNS_DIRNAME",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_K_FOLDS",
]


