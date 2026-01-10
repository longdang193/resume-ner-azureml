"""Legacy facade for conversion_config module.

This module provides backward compatibility by re-exporting from config.conversion.
All imports from this module are deprecated.
"""

import warnings
from config.conversion import load_conversion_config

__all__ = [
    "load_conversion_config",
]

# Issue deprecation warning
warnings.warn(
    "Importing 'conversion_config' from 'orchestration' is deprecated. "
    "Please import from 'config.conversion' instead.",
    DeprecationWarning,
    stacklevel=2
)
