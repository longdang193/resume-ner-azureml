"""Legacy facade for metadata_manager module.

This module provides backward compatibility by re-exporting from metadata.training.
All imports from this module are deprecated.
"""

import warnings
from infrastructure.metadata.training import (
    get_metadata_file_path,
    load_training_metadata,
    save_training_metadata,
    save_metadata_with_fingerprints,
    load_metadata_by_fingerprints,
)

__all__ = [
    "get_metadata_file_path",
    "load_training_metadata",
    "save_training_metadata",
    "save_metadata_with_fingerprints",
    "load_metadata_by_fingerprints",
]

# Issue deprecation warning
warnings.warn(
    "Importing 'metadata_manager' from 'orchestration' is deprecated. "
    "Please import from 'metadata.training' instead.",
    DeprecationWarning,
    stacklevel=2
)
