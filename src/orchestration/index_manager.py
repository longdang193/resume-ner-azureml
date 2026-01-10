"""Legacy facade for index_manager module.

This module provides backward compatibility by re-exporting from metadata.index.
All imports from this module are deprecated.
"""

import warnings
from infrastructure.metadata.index import (
    get_index_file_path,
    update_index,
    find_by_spec_fp,
    find_by_env,
    find_by_model,
    find_by_spec_and_env,
    get_latest_entry,
)

__all__ = [
    "get_index_file_path",
    "update_index",
    "find_by_spec_fp",
    "find_by_env",
    "find_by_model",
    "find_by_spec_and_env",
    "get_latest_entry",
]

# Issue deprecation warning
warnings.warn(
    "Importing 'index_manager' from 'orchestration' is deprecated. "
    "Please import from 'metadata.index' instead.",
    DeprecationWarning,
    stacklevel=2
)
