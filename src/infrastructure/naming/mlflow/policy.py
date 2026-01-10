"""Naming policy loader and formatter for run names.

This module provides backward compatibility by re-exporting from the old location.
"""

from __future__ import annotations

# Import the entire module first to avoid circular import issues
import orchestration.jobs.tracking.naming.policy as _policy_module

# Re-export all the functions
load_naming_policy = _policy_module.load_naming_policy
format_run_name = _policy_module.format_run_name
validate_run_name = _policy_module.validate_run_name
parse_parent_training_id = _policy_module.parse_parent_training_id
validate_naming_policy = _policy_module.validate_naming_policy
normalize_value = _policy_module.normalize_value
sanitize_semantic_suffix = _policy_module.sanitize_semantic_suffix
extract_component = _policy_module.extract_component

__all__ = [
    "load_naming_policy",
    "format_run_name",
    "validate_run_name",
    "parse_parent_training_id",
    "validate_naming_policy",
    "normalize_value",
    "sanitize_semantic_suffix",
    "extract_component",
]
