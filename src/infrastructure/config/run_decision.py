"""Unified run decision logic for all process types.

Single source of truth for determining whether to reuse existing
or create new runs based on run.mode configuration.

Used by:
- HPO: No longer uses run.mode (uses explicit study_name and auto_resume instead)
- Final Training: Checkpoint reuse vs. new training run
- Best Model Selection: Cache reuse
- Benchmarking: Independent run.mode configuration
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from infrastructure.config.run_mode import get_run_mode, is_force_new, is_reuse_if_exists

ProcessType = Literal["hpo", "final_training", "selection", "benchmarking"]


def should_reuse_existing(
    config: Dict[str, Any],
    exists: bool,
    is_complete: Optional[bool] = None,
    process_type: ProcessType = "hpo",
) -> bool:
    """
    Unified decision: Should we reuse existing or create new?
    
    Logic:
    - force_new: Always False (create new, ignore existing)
    - reuse_if_exists: True if exists (and complete if applicable)
    - resume_if_incomplete: True if exists and not complete
    
    Args:
        config: Configuration dict with run.mode
        exists: Whether the existing run/checkpoint/study exists
        is_complete: Optional completeness check (for final training)
                    None = not applicable (HPO), True/False = complete/incomplete
        process_type: Type of process (for logging/context)
    
    Returns:
        True if should reuse existing, False if should create new
    
    Examples:
        >>> # HPO: force_new → always create new
        >>> should_reuse_existing({"run": {"mode": "force_new"}}, exists=True)
        False
        
        >>> # Final Training: reuse_if_exists with complete checkpoint → reuse
        >>> should_reuse_existing({"run": {"mode": "reuse_if_exists"}}, exists=True, is_complete=True)
        True
        
        >>> # Final Training: reuse_if_exists but incomplete → create new
        >>> should_reuse_existing({"run": {"mode": "reuse_if_exists"}}, exists=True, is_complete=False)
        False
        
        >>> # HPO: reuse_if_exists with existing study → reuse
        >>> should_reuse_existing({"run": {"mode": "reuse_if_exists"}}, exists=True)
        True
    """
    run_mode = get_run_mode(config)
    
    # force_new: Always create new (highest priority)
    if is_force_new(config):
        return False
    
    # If doesn't exist, can't reuse
    if not exists:
        return False
    
    # reuse_if_exists: Reuse if exists (and complete if applicable)
    if is_reuse_if_exists(config):
        # For final training, only reuse if complete
        if process_type == "final_training" and is_complete is not None:
            return is_complete
        # For HPO and others, reuse if exists
        return True
    
    # resume_if_incomplete: Reuse if exists and NOT complete
    if run_mode == "resume_if_incomplete":
        if is_complete is not None:
            return not is_complete
        # If completeness not applicable, treat as reuse_if_exists
        return True
    
    # Default: reuse_if_exists behavior
    return True


def get_load_if_exists_flag(
    config: Dict[str, Any],
    checkpoint_enabled: bool,
    process_type: ProcessType = "hpo",
) -> bool:
    """
    Determine load_if_exists flag for Optuna/other libraries.
    
    For HPO: Used in Optuna's create_study(load_if_exists=...)
    For other processes: May be used in similar contexts
    
    Args:
        config: Configuration dict with run.mode
        checkpoint_enabled: Whether checkpointing is enabled
        process_type: Type of process
    
    Returns:
        True if should load existing if exists, False if always create new
    
    Examples:
        >>> # force_new: Never load existing
        >>> get_load_if_exists_flag({"run": {"mode": "force_new"}}, checkpoint_enabled=True)
        False
        
        >>> # reuse_if_exists: Load if checkpointing enabled
        >>> get_load_if_exists_flag({"run": {"mode": "reuse_if_exists"}}, checkpoint_enabled=True)
        True
        
        >>> # Checkpointing disabled: Can't load existing
        >>> get_load_if_exists_flag({"run": {"mode": "reuse_if_exists"}}, checkpoint_enabled=False)
        False
    """
    # If checkpointing disabled, can't load existing
    if not checkpoint_enabled:
        return False
    
    # force_new: Never load existing
    if is_force_new(config):
        return False
    
    # reuse_if_exists or default: Load if exists
    return True

