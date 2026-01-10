"""Azure ML specific utilities module.

This module provides utilities for managing Azure ML data assets and environments.
"""

# Lazy imports to avoid requiring Azure SDK at module level
__all__ = [
    "resolve_dataset_path",
    "register_data_asset",
    "ensure_data_asset_uploaded",
    "build_data_asset_reference",
]


def __getattr__(name: str):
    """Lazy import for Azure ML functions."""
    if name in __all__:
        from .data_assets import (
            resolve_dataset_path,
            register_data_asset,
            ensure_data_asset_uploaded,
            build_data_asset_reference,
        )
        if name == "resolve_dataset_path":
            return resolve_dataset_path
        elif name == "register_data_asset":
            return register_data_asset
        elif name == "ensure_data_asset_uploaded":
            return ensure_data_asset_uploaded
        elif name == "build_data_asset_reference":
            return build_data_asset_reference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

