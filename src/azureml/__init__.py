"""Azure ML specific utilities module.

This module provides utilities for managing Azure ML data assets and environments.
"""

from .data_assets import (
    resolve_dataset_path,
    register_data_asset,
    ensure_data_asset_uploaded,
    build_data_asset_reference,
)

__all__ = [
    "resolve_dataset_path",
    "register_data_asset",
    "ensure_data_asset_uploaded",
    "build_data_asset_reference",
]

