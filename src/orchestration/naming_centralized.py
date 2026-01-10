"""Legacy facade for naming_centralized module (backward compatibility).

This module re-exports NamingContext, create_naming_context, and build_output_path
from the new naming and paths modules for backward compatibility.

New code should import directly from the new modules:
    from naming import NamingContext, create_naming_context
    from paths import build_output_path

Deprecation: This facade will be removed in a future release.
"""

import warnings
from pathlib import Path
from typing import Optional

# Re-export NamingContext and create_naming_context from infrastructure.naming module
from infrastructure.naming.context import (
    NamingContext,
    create_naming_context,
)

# Re-export build_output_path from infrastructure.paths module
from infrastructure.paths.resolve import build_output_path


def build_parent_training_id(spec_fp: str, exec_fp: str, variant: int = 1) -> str:
    """
    Build parent training identifier for conversion.

    This creates a string identifier that can be used as parent_training_id
    in conversion contexts. The format matches the directory structure.

    Args:
        spec_fp: Specification fingerprint.
        exec_fp: Execution fingerprint.
        variant: Variant number.

    Returns:
        Parent training identifier string.
    """
    return f"spec_{spec_fp}_exec_{exec_fp}/v{variant}"


# Add deprecation warnings when module is imported
warnings.warn(
    "orchestration.naming_centralized is deprecated. "
    "Use 'from naming import NamingContext, create_naming_context' and "
    "'from paths import build_output_path' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "NamingContext",
    "create_naming_context",
    "build_output_path",
    "build_parent_training_id",
]
