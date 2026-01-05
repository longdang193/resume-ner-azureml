"""HPO-specific key building (study, trial, family)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


def _compute_hash_64(data: str) -> str:
    """
    Compute full SHA256 hash (64 hex characters).

    Args:
        data: String to hash.

    Returns:
        64-character hex hash.
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _normalize_hyperparameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize hyperparameters for deterministic hashing across platforms.

    Ensures that the same hyperparameters produce the same hash even if:
    - Float precision differs (e.g., 2.33e-05 vs 0.0000233000001)
    - String casing/whitespace differs
    - Types differ (int vs float for whole numbers)

    Args:
        params: Dictionary of hyperparameters.

    Returns:
        Normalized dictionary with canonical representations.
    """
    normalized = {}
    for key, value in sorted(params.items()):
        if isinstance(value, float):
            # Normalize floats to 12 significant figures for stability
            # This ensures 2.33e-05 and 0.0000233000001 both become the same value
            if value == 0.0:
                normalized[key] = 0.0
            else:
                # Use exponential notation with 12 significant figures
                normalized[key] = float(f"{value:.12e}")
        elif isinstance(value, (int, bool)):
            # Keep ints and bools as-is
            normalized[key] = value
        elif isinstance(value, str):
            # Normalize strings: lowercase, strip whitespace
            normalized[key] = value.lower().strip()
        else:
            # For other types, convert to string and normalize
            normalized[key] = str(value).lower().strip()
    return normalized


def build_hpo_study_key(
    data_config: Dict[str, Any],
    hpo_config: Dict[str, Any],
    model: str,
    benchmark_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build canonical study key JSON string for HPO study identity.

    This key uniquely identifies an HPO study based on:
    - Dataset configuration (name, version, path)
    - HPO search space and objective
    - Model backbone
    - Benchmark configuration (for ranking/comparison)

    The key is a JSON string that can be hashed for stable identity.

    Args:
        data_config: Data configuration dictionary.
        hpo_config: HPO configuration dictionary.
        model: Model backbone name.
        benchmark_config: Optional benchmark configuration.

    Returns:
        Canonical JSON string representing the study key.
    """
    # Extract relevant parts of data_config
    data_key = {
        "name": data_config.get("name", ""),
        "version": data_config.get("version", ""),
        "local_path": str(data_config.get("local_path", "")),
        # Include schema if available (affects data processing)
        "schema": data_config.get("schema", {}),
    }

    # Extract relevant parts of hpo_config
    hpo_key = {
        "search_space": hpo_config.get("search_space", {}),
        "objective": hpo_config.get("objective", {}),
        "k_fold": hpo_config.get("k_fold", {}),
        "sampling": {
            "algorithm": hpo_config.get("sampling", {}).get("algorithm", "random"),
        },
        "early_termination": hpo_config.get("early_termination", {}),
    }

    # Extract benchmark config for ranking (excludes hardware-specific details)
    bench_key = {}
    if benchmark_config:
        bench_config = benchmark_config.get("benchmarking", {})
        bench_key = {
            "metric": bench_config.get("metric", "macro-f1"),
            "max_length": bench_config.get("max_length", 512),
            # Exclude hardware-specific: batch_sizes, iterations, device, etc.
        }

    payload = {
        "schema_version": "1.0",
        "data": data_key,
        "hpo": hpo_key,
        "model": model.lower().strip(),
        "benchmark": bench_key,
    }

    # Use compact JSON (no spaces) for consistent hashing
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def build_hpo_study_key_hash(study_key: str) -> str:
    """
    Build hash of study key for tag storage.

    Args:
        study_key: Canonical study key JSON string.

    Returns:
        64-character hex hash.
    """
    return _compute_hash_64(study_key)


def build_hpo_study_family_key(
    data_config: Dict[str, Any],
    hpo_config: Dict[str, Any],
    benchmark_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build canonical study family key JSON string.

    A study family groups multiple studies that share:
    - Same dataset configuration
    - Same HPO search space and objective
    - Same benchmark configuration
    But different model backbones.

    This allows cross-model comparison within the same family.

    Args:
        data_config: Data configuration dictionary.
        hpo_config: HPO configuration dictionary.
        benchmark_config: Optional benchmark configuration.

    Returns:
        Canonical JSON string representing the study family key.
    """
    # Extract relevant parts (same as study_key but without model)
    data_key = {
        "name": data_config.get("name", ""),
        "version": data_config.get("version", ""),
        "local_path": str(data_config.get("local_path", "")),
        "schema": data_config.get("schema", {}),
    }

    hpo_key = {
        "search_space": hpo_config.get("search_space", {}),
        "objective": hpo_config.get("objective", {}),
        "k_fold": hpo_config.get("k_fold", {}),
        "sampling": {
            "algorithm": hpo_config.get("sampling", {}).get("algorithm", "random"),
        },
        "early_termination": hpo_config.get("early_termination", {}),
    }

    bench_key = {}
    if benchmark_config:
        bench_config = benchmark_config.get("benchmarking", {})
        bench_key = {
            "metric": bench_config.get("metric", "macro-f1"),
            "max_length": bench_config.get("max_length", 512),
        }

    payload = {
        "schema_version": "1.0",
        "data": data_key,
        "hpo": hpo_key,
        "benchmark": bench_key,
    }

    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def build_hpo_study_family_hash(study_family_key: str) -> str:
    """
    Build hash of study family key for tag storage.

    Args:
        study_family_key: Canonical study family key JSON string.

    Returns:
        64-character hex hash.
    """
    return _compute_hash_64(study_family_key)


def build_hpo_trial_key(
    study_key_hash: str,
    hyperparameters: Dict[str, Any],
) -> str:
    """
    Build canonical trial key JSON string for trial identity.

    A trial key uniquely identifies a trial within a study by combining:
    - Study key hash (identifies the study, 64 chars)
    - Normalized hyperparameters (identifies the trial within the study)

    Args:
        study_key_hash: SHA256 hash of study key (64 hex chars).
        hyperparameters: Dictionary of trial hyperparameters.

    Returns:
        Canonical JSON string representing the trial key.
    """
    # Normalize hyperparameters for deterministic hashing
    normalized_params = _normalize_hyperparameters(hyperparameters)

    payload = {
        "schema_version": "1.0",
        "study_key_hash": study_key_hash,
        "hyperparameters": normalized_params,
    }

    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def build_hpo_trial_key_hash(trial_key: str) -> str:
    """
    Build hash of trial key for tag storage.

    Args:
        trial_key: Canonical trial key JSON string.

    Returns:
        64-character hex hash.
    """
    return _compute_hash_64(trial_key)

