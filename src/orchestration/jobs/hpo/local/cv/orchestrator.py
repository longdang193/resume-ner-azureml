"""CV orchestrator for local HPO.

This module provides backward compatibility by re-exporting from the new location.
"""

from __future__ import annotations

# Import from new location (note: it's in cv.py, not orchestrator.py)
from hpo.execution.local.cv import (
    run_training_trial_with_cv,
)

__all__ = [
    "run_training_trial_with_cv",
]


