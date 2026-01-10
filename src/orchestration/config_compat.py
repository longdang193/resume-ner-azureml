"""Legacy facade for config_compat module.

This module provides backward compatibility by re-exporting from config.validation.
All imports from this module are deprecated.
    """

import warnings
from config.validation import validate_paths_and_naming_compatible

__all__ = [
    "validate_paths_and_naming_compatible",
]

# Issue deprecation warning
warnings.warn(
    "Importing 'config_compat' from 'orchestration' is deprecated. "
    "Please import from 'config.validation' instead.",
    DeprecationWarning,
    stacklevel=2
)
