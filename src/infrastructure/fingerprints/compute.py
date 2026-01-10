from __future__ import annotations

"""
@meta
name: fingerprints_compute
type: utility
domain: fingerprints
responsibility:
  - Compute specification fingerprints (spec_fp)
  - Compute execution fingerprints (exec_fp)
  - Compute conversion fingerprints (conv_fp)
  - Compute benchmarking fingerprints (bench_fp)
inputs:
  - Configuration dictionaries
  - Git SHA and environment info
outputs:
  - 16-character hex fingerprints
tags:
  - utility
  - fingerprints
  - reproducibility
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Fingerprint computation for reproducibility and traceability."""
import hashlib
import json
from typing import Any, Dict, Optional

def _compute_hash(data: str) -> str:
    """Compute SHA256 hash and return first 16 hex characters."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]

def compute_spec_fp(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    train_config: Dict[str, Any],
    seed: int,
) -> str:
    """
    Compute specification fingerprint (spec_fp).
    
    Represents the specification of the training run:
    - Model architecture (backbone, etc.)
    - Dataset configuration
    - Training hyperparameters
    - Random seed
    
    Args:
        model_config: Model configuration dict.
        data_config: Data configuration dict.
        train_config: Training configuration dict.
        seed: Random seed.
    
    Returns:
        16-character hex fingerprint.
    """
    # Create a deterministic representation of the specification
    spec_data = {
        "model": model_config,
        "data": data_config,
        "train": train_config,
        "seed": seed,
    }
    
    # Sort keys for deterministic JSON
    spec_json = json.dumps(spec_data, sort_keys=True, default=str)
    return _compute_hash(spec_json)

def compute_exec_fp(
    git_sha: Optional[str],
    env_config: Dict[str, Any],
    include_precision: Optional[bool] = None,
    include_determinism: Optional[bool] = None,
) -> str:
    """
    Compute execution fingerprint (exec_fp).
    
    Represents the execution environment:
    - Git commit SHA
    - Environment configuration (platform, compute, etc.)
    - Optional: precision settings
    - Optional: determinism settings
    
    Args:
        git_sha: Git commit SHA (or None/unknown).
        env_config: Environment configuration dict.
        include_precision: Whether to include precision in fingerprint (optional).
        include_determinism: Whether to include determinism in fingerprint (optional).
    
    Returns:
        16-character hex fingerprint.
    """
    # Create a deterministic representation of the execution environment
    exec_data = {
        "git_sha": git_sha or "unknown",
        "env": env_config,
    }
    
    if include_precision is not None:
        exec_data["precision"] = include_precision
    if include_determinism is not None:
        exec_data["determinism"] = include_determinism
    
    # Sort keys for deterministic JSON
    exec_json = json.dumps(exec_data, sort_keys=True, default=str)
    return _compute_hash(exec_json)

def compute_conv_fp(
    parent_spec_fp: str,
    parent_exec_fp: str,
    conversion_config: Dict[str, Any],
) -> str:
    """
    Compute conversion fingerprint (conv_fp).
    
    Represents the model conversion specification:
    - Parent training fingerprints (spec_fp, exec_fp)
    - Conversion configuration (quantization, format, etc.)
    
    Args:
        parent_spec_fp: Parent training specification fingerprint.
        parent_exec_fp: Parent training execution fingerprint.
        conversion_config: Conversion configuration dict.
    
    Returns:
        16-character hex fingerprint.
    """
    # Create a deterministic representation of the conversion
    conv_data = {
        "parent_spec_fp": parent_spec_fp,
        "parent_exec_fp": parent_exec_fp,
        "conversion": conversion_config,
    }
    
    # Sort keys for deterministic JSON
    conv_json = json.dumps(conv_data, sort_keys=True, default=str)
    return _compute_hash(conv_json)

def compute_bench_fp(
    model_config: Dict[str, Any],
    benchmark_config: Dict[str, Any],
) -> str:
    """
    Compute benchmarking fingerprint (bench_fp).
    
    Represents the benchmarking specification:
    - Model configuration
    - Benchmark settings (batch sizes, iterations, etc.)
    
    Args:
        model_config: Model configuration dict.
        benchmark_config: Benchmark configuration dict.
    
    Returns:
        16-character hex fingerprint.
    """
    bench_data = {
        "model": model_config,
        "benchmark": benchmark_config,
    }
    
    bench_json = json.dumps(bench_data, sort_keys=True, default=str)
    return _compute_hash(bench_json)

def compute_hardware_fp(
    hardware_info: Dict[str, Any],
) -> str:
    """
    Compute hardware fingerprint (hardware_fp).
    
    Represents the hardware configuration:
    - GPU model, CUDA version, etc.
    
    Args:
        hardware_info: Hardware information dict.
    
    Returns:
        16-character hex fingerprint.
    """
    hardware_json = json.dumps(hardware_info, sort_keys=True, default=str)
    return _compute_hash(hardware_json)

