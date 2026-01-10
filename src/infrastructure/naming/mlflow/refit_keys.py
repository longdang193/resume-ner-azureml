from __future__ import annotations

"""
@meta
name: naming_mlflow_refit_keys
type: utility
domain: naming
responsibility:
  - Compute refit protocol fingerprints
  - Capture refit/eval protocol for reproducibility
inputs:
  - Data and training configurations
outputs:
  - Refit protocol fingerprint hashes
tags:
  - utility
  - naming
  - mlflow
  - refit
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Refit protocol fingerprint computation."""
import json
from typing import Any, Dict, Optional

from .hpo_keys import _compute_hash_64

def compute_refit_protocol_fp(
    data_config: Dict[str, Any],
    train_config: Optional[Dict[str, Any]] = None,
    eval_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compute refit protocol fingerprint.

    Captures refit/eval protocol: dataset version, eval config, split strategy (full train).
    Ensures different refit protocols produce different fingerprints even with same hyperparameters.

    Args:
        data_config: Data configuration dict (dataset version, path).
        train_config: Optional training configuration (for eval settings).
        eval_config: Optional evaluation configuration (if separate from train_config).

    Returns:
        64-character hex hash of refit protocol fingerprint.
    """
    payload = {
        "schema_version": "1.0",
        "dataset": {
            "name": data_config.get("name", ""),
            "version": data_config.get("version", ""),
            "local_path": str(data_config.get("local_path", "")),
        },
        "refit_protocol": {
            "split_strategy": "full_train",  # Refit uses full training set
            "use_validation": False,  # Refit doesn't use validation split
        },
        "eval": {
            "metric": train_config.get("evaluation", {}).get("metric", "macro-f1") if train_config else "macro-f1",
            "postprocess": train_config.get("evaluation", {}).get("postprocess", {}) if train_config else {},
        },
    }

    # Add eval_config if provided separately
    if eval_config:
        payload["eval"].update({
            "metric": eval_config.get("metric", payload["eval"]["metric"]),
            "postprocess": eval_config.get("postprocess", payload["eval"]["postprocess"]),
        })

    payload_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return _compute_hash_64(payload_str)
