"""Legacy facade for environment module.

This module provides backward compatibility by re-exporting from config.environment.
All imports from this module are deprecated.
"""

import warnings
from config.environment import (
    DEFAULT_ENVIRONMENT_NAME,
    DEFAULT_CONDA_RELATIVE_PATH,
    DEFAULT_DOCKER_IMAGE,
    WARMUP_DISPLAY_NAME,
    WARMUP_HISTORY_LIMIT,
    EnvironmentConfig,
    build_environment_config,
    load_conda_environment,
    compute_environment_hash,
    get_or_create_environment,
    create_training_environment,
    prepare_environment_image,
)

__all__ = [
    "DEFAULT_ENVIRONMENT_NAME",
    "DEFAULT_CONDA_RELATIVE_PATH",
    "DEFAULT_DOCKER_IMAGE",
    "WARMUP_DISPLAY_NAME",
    "WARMUP_HISTORY_LIMIT",
    "EnvironmentConfig",
    "build_environment_config",
    "load_conda_environment",
    "compute_environment_hash",
    "get_or_create_environment",
    "create_training_environment",
    "prepare_environment_image",
]

# Issue deprecation warning
warnings.warn(
    "Importing 'environment' from 'orchestration' is deprecated. "
    "Please import from 'config.environment' instead.",
    DeprecationWarning,
    stacklevel=2
        )
