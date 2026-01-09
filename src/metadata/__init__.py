"""Metadata and index management module.

This module provides utilities for managing persistent metadata and index files
for training and conversion stages.
"""

from .index import (
    get_index_file_path,
    update_index,
    find_by_spec_fp,
    find_by_env,
    find_by_model,
    find_by_spec_and_env,
    get_latest_entry,
)
from .training import (
    get_metadata_file_path,
    load_training_metadata,
    save_training_metadata,
    save_metadata_with_fingerprints,
    load_metadata_by_fingerprints,
)

__all__ = [
    # Index
    "get_index_file_path",
    "update_index",
    "find_by_spec_fp",
    "find_by_env",
    "find_by_model",
    "find_by_spec_and_env",
    "get_latest_entry",
    # Training metadata
    "get_metadata_file_path",
    "load_training_metadata",
    "save_training_metadata",
    "save_metadata_with_fingerprints",
    "load_metadata_by_fingerprints",
]

