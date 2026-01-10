"""Path resolution utilities for Colab, Kaggle, and local environments.

DEPRECATED: This module is maintained for backward compatibility only.
Please use the following modules instead:
- paths.validation: validate_path_before_mkdir
- paths.drive: resolve_output_path_for_colab
- hpo.utils.paths: resolve_hpo_output_dir
"""

import warnings
from pathlib import Path
from typing import Optional, Callable

# Re-export from new locations with deprecation warnings
from infrastructure.paths.validation import validate_path_before_mkdir
from infrastructure.paths.drive import resolve_output_path_for_colab
from hpo.utils.paths import resolve_hpo_output_dir


def _deprecation_warning(name: str, new_module: str) -> None:
    """Issue deprecation warning for moved functions."""
    warnings.warn(
        f"Importing '{name}' from 'orchestration.path_resolution' is deprecated. "
        f"Please import from '{new_module}' instead.",
        DeprecationWarning,
        stacklevel=3
    )


# Issue warnings when module is imported
warnings.warn(
    "The 'orchestration.path_resolution' module is deprecated. "
    "Please use 'paths' and 'hpo.utils.paths' modules instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "validate_path_before_mkdir",
    "resolve_output_path_for_colab",
    "resolve_hpo_output_dir",
]
