"""Final training execution module.

This module provides final training job execution functionality.
"""

from __future__ import annotations

from .executor import execute_final_training
from .lineage import extract_lineage_from_best_model
from .tags import apply_lineage_tags

__all__ = [
    "extract_lineage_from_best_model",
    "apply_lineage_tags",
    "execute_final_training",
]


