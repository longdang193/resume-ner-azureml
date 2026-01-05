"""MLflow naming utilities: run keys, hashing, tag building, and sanitization.

This module re-exports all naming functions for backward compatibility.
New code should import directly from orchestration.jobs.tracking.naming.*
"""

from __future__ import annotations

# Re-export all naming functions for backward compatibility
from orchestration.jobs.tracking.naming.run_keys import (
    build_mlflow_run_key,
    build_mlflow_run_key_hash,
    build_counter_key,
)
from orchestration.jobs.tracking.naming.run_names import (
    build_mlflow_run_name,
)
from orchestration.jobs.tracking.naming.tags import (
    build_mlflow_tags,
    sanitize_tag_value,
)
from orchestration.jobs.tracking.naming.hpo_keys import (
    build_hpo_study_key,
    build_hpo_study_key_hash,
    build_hpo_study_family_key,
    build_hpo_study_family_hash,
    build_hpo_trial_key,
    build_hpo_trial_key_hash,
)
from orchestration.jobs.tracking.naming.refit_keys import (
    compute_refit_protocol_fp,
)

__all__ = [
    "build_mlflow_run_key",
    "build_mlflow_run_key_hash",
    "build_counter_key",
    "build_mlflow_run_name",
    "build_mlflow_tags",
    "sanitize_tag_value",
    "build_hpo_study_key",
    "build_hpo_study_key_hash",
    "build_hpo_study_family_key",
    "build_hpo_study_family_hash",
    "build_hpo_trial_key",
    "build_hpo_trial_key_hash",
    "compute_refit_protocol_fp",
]
