"""Artifact upload and checkpoint archive management.

This module re-exports artifact functions for backward compatibility.
New code should import directly from orchestration.jobs.tracking.artifacts.*
"""

from __future__ import annotations

# Re-export for backward compatibility
from orchestration.jobs.tracking.artifacts.manager import (
    create_checkpoint_archive,
    should_skip_file,
)

__all__ = [
    "create_checkpoint_archive",
    "should_skip_file",
]
