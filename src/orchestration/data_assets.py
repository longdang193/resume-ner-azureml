"""Legacy facade for data_assets module.

This module provides backward compatibility by re-exporting from azureml.data_assets.
All imports from this module are deprecated.
"""

import warnings
from azureml.data_assets import (
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

# Issue deprecation warning
warnings.warn(
    "Importing 'data_assets' from 'orchestration' is deprecated. "
    "Please import from 'azureml.data_assets' instead.",
    DeprecationWarning,
    stacklevel=2
)
