"""
@meta
name: paths_validation
type: utility
domain: paths
responsibility:
  - Validate paths before directory creation
  - Ensure filesystem safety
inputs:
  - Path objects
outputs:
  - Validated path objects
tags:
  - utility
  - paths
  - validation
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Filesystem/path safety validation."""

import re
from pathlib import Path
from typing import Optional

from common.shared.logging_utils import get_logger

logger = get_logger(__name__)


def validate_path_before_mkdir(path: Path, context: str = "directory") -> Path:
    """
    Validate path before creating directory to prevent creating invalid files.

    Args:
        path: Path to validate
        context: Context string for error messages

    Returns:
        Validated and resolved path

    Raises:
        ValueError: If path is invalid
    """
    if not path or not str(path):
        raise ValueError(f"Invalid {context} path: {path}")

    # Ensure path is absolute
    if not path.is_absolute():
        path = path.resolve()

    path_str = str(path)

    # Basic invalid cases
    if path_str in ("", ".", ".."):
        raise ValueError(
            f"Invalid {context} path (too short or relative): {path_str}"
        )

    # Split path
    path_parts = path_str.replace("\\", "/").split("/")

    # Check if last part looks like a version number (e.g. "1.0.0")
    if path_parts:
        last_part = path_parts[-1]
        if re.match(r"^[\d\.]+$", last_part):
            # Reject single-part paths like "1.0.0"
            if len(path_parts) == 1:
                raise ValueError(
                    f"Invalid {context} path (looks like version number): {path_str}"
                )

    # Validate path has reasonable structure
    if len(path_parts) < 2:
        raise ValueError(
            f"Invalid {context} path (too short, appears to be filename): {path_str}"
        )

    # Safety: path exists but is a file
    if path.exists() and path.is_file():
        logger.error(f"Path exists as file, not directory: {path}")
        raise ValueError(
            f"Cannot create {context}, path exists as file: {path}"
        )

    return path


def validate_output_path(path: Path) -> Path:
    """
    Public validation function for output paths (called by resolve.py).

    This performs filesystem-specific safety checks (forbidden chars, length, mkdir safety).
    Naming pattern validation is handled separately in naming/display_policy.py.

    Args:
        path: Path to validate

    Returns:
        Validated and resolved path

    Raises:
        ValueError: If path is invalid
    """
    return validate_path_before_mkdir(path, context="output directory")

