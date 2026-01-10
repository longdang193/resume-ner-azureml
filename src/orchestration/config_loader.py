"""Legacy facade for config_loader module.

This module provides backward compatibility by re-exporting from config.loader.
All imports from this module are deprecated.
"""

import warnings
from config.loader import (
    CONFIG_HASH_LENGTH,
    ExperimentConfig,
    load_experiment_config,
    load_all_configs,
)

__all__ = [
    "CONFIG_HASH_LENGTH",
    "ExperimentConfig",
    "load_experiment_config",
    "load_all_configs",
]

# Issue deprecation warning
warnings.warn(
    "Importing 'config_loader' from 'orchestration' is deprecated. "
    "Please import from 'config.loader' instead.",
    DeprecationWarning,
    stacklevel=2
)
