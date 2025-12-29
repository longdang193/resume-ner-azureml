"""Fingerprint computation for experiment identity and lineage tracking."""

from __future__ import annotations

import hashlib
import json
import platform
from typing import Dict, Any, Optional

try:
    import torch
    import transformers
    import accelerate
    import numpy
except ImportError:
    # Handle case where these aren't installed
    torch = None
    transformers = None
    accelerate = None
    numpy = None

from .config_loader import CONFIG_HASH_LENGTH


def _compute_hash(data: Dict[str, Any]) -> str:
    """Compute a deterministic short hash for a dictionary."""
    data_str = json.dumps(data, sort_keys=True)
    full_hash = hashlib.sha256(data_str.encode("utf-8")).hexdigest()
    return full_hash[:CONFIG_HASH_LENGTH]


def compute_spec_fp(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    train_config: Dict[str, Any],
    seed: Optional[int] = None,
    task_schema: Optional[Dict[str, Any]] = None,
    selection_spec: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compute a platform-independent specification fingerprint (spec_fp).
    Includes what defines the experiment scientifically.
    """
    spec_data = {
        "model_config_hash": _compute_hash(model_config),
        "data_config_hash": _compute_hash(data_config),
        "train_config_hash": _compute_hash(train_config),
        "seed": seed,
        "task_schema_hash": _compute_hash(task_schema) if task_schema else None,
        "selection_spec_hash": _compute_hash(selection_spec) if selection_spec else None,
    }
    return _compute_hash(spec_data)


def compute_exec_fp(
    git_sha: str,
    env_config: Dict[str, Any],
    precision_flags: Optional[Dict[str, Any]] = None,
    determinism_flags: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compute an execution fingerprint (exec_fp).
    Includes what can change results or comparability across platforms (toolchain identity).
    """
    dependency_versions = {}
    if torch:
        dependency_versions["torch"] = torch.__version__
    if transformers:
        dependency_versions["transformers"] = transformers.__version__
    if accelerate:
        dependency_versions["accelerate"] = accelerate.__version__
    if numpy:
        dependency_versions["numpy"] = numpy.__version__
    
    exec_data = {
        "git_sha": git_sha,
        "env_config_hash": _compute_hash(env_config),
        "dependency_versions": dependency_versions,
        "precision_flags": precision_flags,
        "determinism_flags": determinism_flags,
        "python_version": platform.python_version(),
    }
    return _compute_hash(exec_data)


def compute_hardware_fp() -> str:
    """Compute a hardware fingerprint (GPU model, CPU model, RAM class)."""
    hardware_data = {
        "processor": platform.processor(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    
    if torch and torch.cuda.is_available():
        hardware_data["gpu"] = torch.cuda.get_device_name(0)
        hardware_data["cuda_version"] = torch.version.cuda if hasattr(torch.version, 'cuda') else "N/A"
    else:
        hardware_data["gpu"] = "N/A"
        hardware_data["cuda_version"] = "N/A"
    
    return _compute_hash(hardware_data)


def compute_bench_fp(
    spec_fp: str,
    benchmark_config: Dict[str, Any],
    hardware_fp: str,
    runtime_fp: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compute a benchmark fingerprint (bench_fp).
    Includes spec_fp, benchmark config hash, hardware, and runtime.
    """
    bench_data = {
        "spec_fp": spec_fp,
        "benchmark_config_hash": _compute_hash(benchmark_config),
        "hardware_fp": hardware_fp,
        "runtime_fp": runtime_fp,
    }
    return _compute_hash(bench_data)


def compute_conv_fp(
    parent_spec_fp: str,
    parent_exec_fp: str,
    conversion_config: Dict[str, Any],
    tooling_versions: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compute a conversion fingerprint (conv_fp).
    Depends on parent training run + conversion config + toolchain.
    """
    conv_data = {
        "parent_spec_fp": parent_spec_fp,
        "parent_exec_fp": parent_exec_fp,
        "conversion_config_hash": _compute_hash(conversion_config),
        "tooling_versions": tooling_versions,
    }
    return _compute_hash(conv_data)

