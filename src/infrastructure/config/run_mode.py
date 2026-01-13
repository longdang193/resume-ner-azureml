"""Unified run mode extraction utility.

Single source of truth for run.mode extraction across all stages:
- HPO: No longer uses run.mode (uses explicit study_name and auto_resume instead)
- Final Training: Controls variant creation and checkpoint reuse
- Best Model Selection: Controls cache reuse
- Benchmarking: Independent run.mode configuration (defaults to reuse_if_exists)

This module replaces 4+ duplicate extractions throughout the codebase.

See also: `run_decision.py` for unified decision logic (should_reuse_existing, get_load_if_exists_flag)
that uses the run mode extracted by this module to make reuse vs. create new decisions.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

RunMode = Literal["reuse_if_exists", "force_new", "resume_if_incomplete"]


def get_run_mode(config: Dict[str, Any], default: RunMode = "reuse_if_exists") -> RunMode:
    """
    Extract run.mode from configuration with consistent defaults.
    
    Used across all stages:
    - HPO: No longer uses run.mode (uses explicit study_name and auto_resume instead)
    - Final Training: Controls variant creation and checkpoint reuse
    - Best Model Selection: Controls cache reuse
    - Benchmarking: Independent run.mode configuration (defaults to reuse_if_exists)
    
    Args:
        config: Configuration dictionary (e.g., from YAML)
        default: Default mode if not specified (default: "reuse_if_exists")
    
    Returns:
        Run mode string: "reuse_if_exists", "force_new", or "resume_if_incomplete"
    
    Examples:
        >>> config = {"run": {"mode": "force_new"}}
        >>> get_run_mode(config)
        'force_new'
        
        >>> config = {}  # No run.mode specified
        >>> get_run_mode(config)
        'reuse_if_exists'
    """
    return config.get("run", {}).get("mode", default)


def is_force_new(config: Dict[str, Any]) -> bool:
    """Check if run.mode is force_new."""
    return get_run_mode(config) == "force_new"


def is_reuse_if_exists(config: Dict[str, Any]) -> bool:
    """Check if run.mode is reuse_if_exists."""
    return get_run_mode(config) == "reuse_if_exists"


def is_resume_if_incomplete(config: Dict[str, Any]) -> bool:
    """Check if run.mode is resume_if_incomplete."""
    return get_run_mode(config) == "resume_if_incomplete"

