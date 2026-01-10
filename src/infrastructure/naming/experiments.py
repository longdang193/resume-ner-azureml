from __future__ import annotations

"""
@meta
name: naming_experiments
type: utility
domain: naming
responsibility:
  - Build experiment and stage names
  - Extract stage configurations
inputs:
  - Experiment configurations
  - Stage identifiers
outputs:
  - Formatted experiment names
tags:
  - utility
  - naming
  - experiments
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Experiment and stage naming helpers (dict-based interface)."""
from typing import Any, Dict, Optional, Union

def get_stage_config(experiment_cfg: Union[dict, Any], stage: str) -> Dict[str, Any]:
    """
    Return the configuration block for a given stage, if present.

    This is a thin, read-only helper around experiment config stages.
    Supports both dict interface and ExperimentConfig objects.

    Args:
        experiment_cfg: Experiment configuration dictionary with 'stages' key,
            or ExperimentConfig object with a 'stages' attribute.
        stage: Stage name to retrieve.

    Returns:
        Stage configuration dictionary, or empty dict if not found.
    """
    # Handle ExperimentConfig objects (has 'stages' attribute)
    if hasattr(experiment_cfg, "stages"):
        stages = experiment_cfg.stages
    # Handle dict interface
    else:
        stages = experiment_cfg.get("stages", {})
    
    return stages.get(stage, {}) or {}

def build_aml_experiment_name(
    experiment_name: str,
    stage: str,
    backbone: Optional[str] = None,
) -> str:
    """Build a simple, stable AML experiment name.

    This intentionally accepts only the minimum information required to build
    the name and leaves more complex context handling (full configs, hashes,
    etc.) to the caller.

    Args:
        experiment_name: Base experiment name.
        stage: Stage name (e.g., "hpo", "training").
        backbone: Optional model backbone name.

    Returns:
        Formatted experiment name.
    """
    parts = [experiment_name, stage]
    if backbone:
        parts.append(backbone)
    return "-".join(parts)

def build_mlflow_experiment_name(experiment_name: str, stage: str, backbone: str) -> str:
    """Build MLflow experiment name from components.

    Args:
        experiment_name: Base experiment name.
        stage: Stage name (e.g., "hpo", "training").
        backbone: Model backbone name (required).

    Returns:
        Formatted experiment name: "{experiment_name}-{stage}-{backbone}".
    """
    return build_aml_experiment_name(experiment_name, stage, backbone)

