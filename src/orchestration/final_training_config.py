"""Legacy facade for final_training_config module.

This module provides backward compatibility by re-exporting from config.training.
All imports from this module are deprecated.
"""

import warnings
from infrastructure.config.training import load_final_training_config

__all__ = [
    "load_final_training_config",
]

# Issue deprecation warning
warnings.warn(
    "Importing 'final_training_config' from 'orchestration' is deprecated. "
    "Please import from 'config.training' instead.",
    DeprecationWarning,
    stacklevel=2
)
