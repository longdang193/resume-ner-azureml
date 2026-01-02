"""Cross-validation for local HPO."""

from __future__ import annotations

from .orchestrator import run_training_trial_with_cv

__all__ = [
    "run_training_trial_with_cv",
]
