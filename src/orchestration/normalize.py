"""Legacy facade for normalize module.

This module provides backward compatibility by re-exporting from core.normalize.
All imports from this module are deprecated.
"""

import warnings
from core.normalize import (
    normalize_for_name,
    normalize_for_path,
)

__all__ = [
    "normalize_for_name",
    "normalize_for_path",
]

# Issue deprecation warning
warnings.warn(
    "Importing 'normalize' from 'orchestration' is deprecated. "
    "Please import from 'core.normalize' instead.",
    DeprecationWarning,
    stacklevel=2
)
