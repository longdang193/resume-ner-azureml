from __future__ import annotations

"""
@meta
name: config_loader
type: utility
domain: config
responsibility:
  - Load experiment configuration from YAML
  - Load all domain configuration files
  - Compute configuration hashes
  - Create configuration metadata for tagging
  - Validate configuration immutability
inputs:
  - Experiment YAML files
  - Domain configuration files (data, model, train, hpo, env, benchmark)
outputs:
  - ExperimentConfig dataclass
  - Loaded configuration dictionaries
  - Configuration hashes
tags:
  - utility
  - config
  - loading
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import hashlib
import json

from common.shared.yaml_utils import load_yaml

CONFIG_HASH_LENGTH = 16

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Resolved experiment configuration.

    This object holds concrete paths to all domain configs (data/model/train/hpo/env/benchmark)
    as well as optional orchestration metadata such as stages and naming rules.
    The underlying YAML files remain the single source of truth.
    """

    name: str
    data_config: Path
    model_config: Path
    train_config: Path
    hpo_config: Path
    env_config: Path
    benchmark_config: Path
    stages: Dict[str, Any]
    naming: Dict[str, Any]

def load_experiment_config(config_root: Path, experiment_name: str) -> ExperimentConfig:
    """
    Resolve an experiment-level YAML into an ``ExperimentConfig``.

    The experiment YAML is expected at ``config_root / 'experiment' / <name>.yaml``
    and contains only relative paths into the config tree plus optional
    orchestration metadata.

    Args:
        config_root: Path to the top-level ``config`` directory.
        experiment_name: Basename (without extension) of the experiment YAML.

    Returns:
        An ``ExperimentConfig`` with fully-resolved config paths and metadata.
    """
    experiment_path = config_root / "experiment" / f"{experiment_name}.yaml"
    raw = load_yaml(experiment_path)

    def resolve(relative: str) -> Path:
        return config_root / relative

    return ExperimentConfig(
        name=raw.get("experiment_name", experiment_name),
        data_config=resolve(raw["data_config"]),
        model_config=resolve(raw["model_config"]),
        train_config=resolve(raw["train_config"]),
        hpo_config=resolve(raw["hpo_config"]),
        env_config=resolve(raw["env_config"]),
        benchmark_config=resolve(raw.get("benchmark_config", "benchmark.yaml")),
        stages=raw.get("stages", {}) or {},
        naming=raw.get("naming", {}) or {},
    )

def load_all_configs(exp_cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Load all domain configuration files referenced by an ``ExperimentConfig``.

    Args:
        exp_cfg: Resolved experiment configuration containing config paths.

    Returns:
        Dictionary keyed by domain name:
        ``data``, ``model``, ``train``, ``hpo``, ``env``, ``benchmark``.
    """
    configs = {
        "data": load_yaml(exp_cfg.data_config),
        "model": load_yaml(exp_cfg.model_config),
        "train": load_yaml(exp_cfg.train_config),
        "hpo": load_yaml(exp_cfg.hpo_config),
        "env": load_yaml(exp_cfg.env_config),
    }
    
    # Load benchmark config if it exists
    if exp_cfg.benchmark_config.exists():
        configs["benchmark"] = load_yaml(exp_cfg.benchmark_config)
    
    return configs

def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute a deterministic short hash for a single configuration dictionary.

    Args:
        config: Arbitrary configuration mapping.

    Returns:
        Hex string of length ``CONFIG_HASH_LENGTH`` suitable for versioning.
    """
    config_str = json.dumps(config, sort_keys=True)
    full_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()
    return full_hash[:CONFIG_HASH_LENGTH]

def compute_config_hashes(configs: Dict[str, Any]) -> Dict[str, str]:
    """
    Compute short hashes for all domain configs in a configuration bundle.

    Args:
        configs: Mapping from domain name (e.g. ``'data'``) to config dict.

    Returns:
        Mapping from domain name to short hash string.
    """
    return {name: compute_config_hash(cfg) for name, cfg in configs.items()}

def create_config_metadata(
    configs: Dict[str, Any],
    config_hashes: Dict[str, str],
) -> Dict[str, str]:
    """
    Build a flat metadata dictionary for tagging Azure ML jobs and models.

    Args:
        configs: Loaded configuration dictionaries by domain.
        config_hashes: Short hashes for each domain config (from ``compute_config_hashes``).

    Returns:
        Dictionary of string keys to string values suitable for Azure ML tags.
    """
    return {
        "data_config_hash": config_hashes["data"],
        "model_config_hash": config_hashes["model"],
        "train_config_hash": config_hashes["train"],
        "hpo_config_hash": config_hashes["hpo"],
        "env_config_hash": config_hashes["env"],
        "data_version": str(configs["data"].get("version")),
        "model_backbone": str(configs["model"].get("backbone")),
    }

def snapshot_configs(configs: Dict[str, Any]) -> Dict[str, str]:
    """
    Take immutable JSON snapshots of all configs for later mutation checks.

    Args:
        configs: Loaded configuration dictionaries by domain.

    Returns:
        Mapping from domain name to JSON-serialised string representation.
    """
    return {name: json.dumps(cfg, sort_keys=True) for name, cfg in configs.items()}

def validate_config_immutability(
    configs: Dict[str, Any],
    snapshots: Dict[str, str],
) -> None:
    """
    Verify that configuration dictionaries have not been mutated at runtime.

    Args:
        configs: Current configuration dictionaries by domain.
        snapshots: Original JSON snapshots from ``snapshot_configs``.

    Raises:
        ValueError: If any domain config differs from its original snapshot.
    """
    for name, current in configs.items():
        current_json = json.dumps(current, sort_keys=True)
        if current_json != snapshots[name]:
            raise ValueError(f"Config '{name}' was mutated at runtime")