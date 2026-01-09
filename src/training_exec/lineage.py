"""Lineage extraction utilities for final training."""

from __future__ import annotations

from typing import Any, Dict


def extract_lineage_from_best_model(best_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract normalized lineage information from best_model dict.
    
    Extracts HPO lineage information (study_key_hash, trial_key_hash, run IDs)
    from the best_model dictionary returned by model selection. This information
    is used to tag final training runs with lineage tags linking them back to
    their HPO origins.
    
    Args:
        best_model: Dictionary containing best model information from model selection.
                    Expected keys:
                    - study_key_hash: Study key hash
                    - trial_key_hash: Trial key hash
                    - trial_run_id: Trial run ID
                    - run_id: Refit/artifact run ID
                    - tags: Dictionary of MLflow tags (may contain additional lineage info)
    
    Returns:
        Dictionary with normalized lineage information:
        - hpo_study_key_hash: Study key hash
        - hpo_trial_key_hash: Trial key hash
        - hpo_trial_run_id: Trial run ID
        - hpo_refit_run_id: Refit/artifact run ID
        - hpo_sweep_run_id: Sweep run ID (optional, if available in tags)
        
        None values are filtered out.
    """
    tags = best_model.get("tags", {})
    
    lineage = {
        "hpo_study_key_hash": best_model.get("study_key_hash") or tags.get("code.study_key_hash"),
        "hpo_trial_key_hash": best_model.get("trial_key_hash") or tags.get("code.trial_key_hash"),
        "hpo_trial_run_id": best_model.get("trial_run_id"),
        "hpo_refit_run_id": best_model.get("run_id"),  # Artifact run ID
    }
    
    # Optionally extract sweep run ID from tags if available
    if "code.lineage.hpo_sweep_run_id" in tags:
        lineage["hpo_sweep_run_id"] = tags["code.lineage.hpo_sweep_run_id"]
    
    # Filter out None values
    return {k: v for k, v in lineage.items() if v is not None}

