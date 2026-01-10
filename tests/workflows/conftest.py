"""Shared pytest fixtures for workflow E2E tests."""

import sys
from pathlib import Path

import pytest

# Add fixtures to path
_fixtures_path = Path(__file__).parent.parent / "fixtures"
sys.path.insert(0, str(_fixtures_path.parent))

# Import shared fixtures
from fixtures import (
    tiny_dataset,
    mock_mlflow_tracking,
    validate_path_structure,
    validate_run_name,
    validate_tags,
)

# Re-export for convenience
__all__ = [
    "tiny_dataset",
    "mock_mlflow_tracking",
    "validate_path_structure",
    "validate_run_name",
    "validate_tags",
]






