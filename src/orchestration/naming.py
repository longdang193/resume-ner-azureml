from __future__ import annotations

from typing import Any, Dict


def get_stage_config(experiment_config: Any, stage_name: str) -> Dict[str, Any]:
    """Return stage configuration from the experiment YAML (or empty dict).

    This is a thin helper around ``experiment_config.stages[stage_name]`` that
    tolerates missing configuration and always returns a dictionary.
    """
    stages = getattr(experiment_config, "stages", {}) or {}
    return stages.get(stage_name, {}) or {}


def build_aml_experiment_name(
    experiment_config: Any,
    env_config: Dict[str, Any],
    stage_name: str,
    backbone: str | None = None,
) -> str:
    """Derive AML experiment name from stage config and optional backbone.

    Resolution order:
    1. ``experiment_config.stages[stage_name].aml_experiment`` when present
    2. Fallback to legacy ``env_config['logging']['experiment_name']``

    The experiment-level ``naming.include_backbone_in_experiment`` flag controls
    whether the backbone suffix is appended to the base name.
    """
    stage_cfg = get_stage_config(experiment_config, stage_name)
    base_name = stage_cfg.get("aml_experiment") or env_config["logging"]["experiment_name"]

    naming_cfg = getattr(experiment_config, "naming", {}) or {}
    include_backbone = bool(naming_cfg.get("include_backbone_in_experiment", False))

    if include_backbone and backbone:
        return f"{base_name}-{backbone}"
    return base_name





