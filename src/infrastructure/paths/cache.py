"""
@meta
name: paths_cache
type: utility
domain: paths
responsibility:
  - Manage cache file paths based on strategy configuration
  - Support latest, index, and timestamped cache files
inputs:
  - Cache type identifiers
  - Configuration directories
outputs:
  - Cache file paths
tags:
  - utility
  - paths
  - cache
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Cache file path management."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.normalize import normalize_for_path
from .config import load_paths_config
from .resolve import resolve_output_path
from common.shared.json_cache import load_json, save_json


def get_cache_file_path(
    root_dir: Path,
    config_dir: Path,
    cache_type: str,
    filename: Optional[str] = None,
    file_type: Optional[str] = None
) -> Path:
    """
    Get cache file path based on strategy configuration.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        cache_type: Type of cache (e.g., "best_configurations", "final_training").
        filename: Optional specific filename.
        file_type: Type of file ("latest", "index", "timestamped").

    Returns:
        Path to cache file.
    """
    paths_config = load_paths_config(config_dir)
    cache_dir = resolve_output_path(
        root_dir, config_dir, "cache", subcategory=cache_type)

    if filename:
        return cache_dir / filename

    # Get strategy config
    strategy_config = paths_config.get(
        "cache_strategies", {}).get(cache_type, {})

    if file_type == "latest":
        latest_config = strategy_config.get("latest", {})
        if latest_config.get("enabled", True):
            return cache_dir / latest_config.get("filename", f"latest_{cache_type}.json")

    elif file_type == "index":
        index_config = strategy_config.get("index", {})
        if index_config.get("enabled", True):
            return cache_dir / index_config.get("filename", f"{cache_type}_index.json")

    # Default: return latest file path
    files_config = paths_config.get("files", {}).get("cache", {})
    if cache_type == "best_configurations":
        return cache_dir / files_config.get("best_config_latest", "latest_best_configuration.json")
    elif cache_type == "final_training":
        return cache_dir / files_config.get("final_training_latest", "latest_final_training_cache.json")

    return cache_dir


def get_timestamped_cache_filename(
    config_dir: Path,
    cache_type: str,
    backbone: str,
    identifier: str,
    timestamp: str,
) -> str:
    """
    Generate timestamped cache filename from pattern.
    """
    paths_config = load_paths_config(config_dir)
    norm_rules = paths_config.get("normalize_paths")

    # Get pattern based on cache type
    if cache_type == "best_configurations":
        pattern = paths_config.get("patterns", {}).get(
            "best_config_file",
            "best_config_{backbone}_{trial}_{timestamp}.json",
        )
        # Use 'identifier' placeholder for trial
        pattern = pattern.replace("{trial}", "{identifier}")

    elif cache_type == "final_training":
        pattern = paths_config.get("patterns", {}).get(
            "final_training_cache_file",
            "final_training_{backbone}_{run_id}_{timestamp}.json",
        )
        # Use 'identifier' placeholder for run_id
        pattern = pattern.replace("{run_id}", "{identifier}")

    elif cache_type == "best_model_selection":
        pattern = paths_config.get("patterns", {}).get(
            "best_model_selection_cache_file",
            "best_model_selection_{backbone}_{identifier}_{timestamp}.json",
        )
        # Pattern already uses {identifier}

    else:
        # Generic pattern
        pattern = f"{cache_type}_{{backbone}}_{{identifier}}_{{timestamp}}.json"

    # Sanitize values for filename
    if norm_rules:
        backbone_safe, _ = normalize_for_path(backbone, norm_rules)
        identifier_safe, _ = normalize_for_path(identifier, norm_rules)
    else:
        backbone_safe = backbone.replace("-", "_").replace("/", "_")
        identifier_safe = identifier.replace("-", "_").replace("/", "_")

    return pattern.format(
        backbone=backbone_safe,
        identifier=identifier_safe,
        timestamp=timestamp,
    )


def get_cache_strategy_config(
    config_dir: Path,
    cache_type: str
) -> Dict[str, Any]:
    """
    Get cache strategy configuration.

    Args:
        config_dir: Config directory.
        cache_type: Type of cache.

    Returns:
        Strategy configuration dictionary.
    """
    paths_config = load_paths_config(config_dir)
    return paths_config.get("cache_strategies", {}).get(cache_type, {
        "strategy": "dual",
        "timestamped": {"enabled": True},
        "latest": {"enabled": True},
        "index": {"enabled": True},
    })


def save_cache_with_dual_strategy(
    root_dir: Path,
    config_dir: Path,
    cache_type: str,
    data: Dict[str, Any],
    backbone: str,
    identifier: str,
    timestamp: str,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Path, Path, Path]:
    """
    Save cache using dual file strategy.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        cache_type: Type of cache ("best_configurations" or "final_training").
        data: Data to save.
        backbone: Backbone name.
        identifier: Identifier (trial name or run_id).
        timestamp: Timestamp string.
        additional_metadata: Optional additional metadata to include.

    Returns:
        Tuple of (timestamped_file_path, latest_file_path, index_file_path).
    """
    # Get cache directory
    cache_dir = resolve_output_path(
        root_dir, config_dir, "cache", subcategory=cache_type)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get strategy configuration
    strategy_config = get_cache_strategy_config(config_dir, cache_type)

    # Generate timestamped filename
    timestamped_filename = get_timestamped_cache_filename(
        config_dir, cache_type, backbone, identifier, timestamp
    )
    timestamped_file = cache_dir / timestamped_filename

    # Get latest and index file paths
    latest_file = get_cache_file_path(
        root_dir, config_dir, cache_type, file_type="latest")
    index_file = get_cache_file_path(
        root_dir, config_dir, cache_type, file_type="index")

    # Prepare data with metadata
    data_with_metadata = data.copy()
    cache_metadata = {
        "saved_at": datetime.now().isoformat(),
        "timestamped_file": timestamped_filename,
        "cache_directory": str(cache_dir),
        "strategy": strategy_config.get("strategy", "dual"),
    }
    if additional_metadata:
        cache_metadata.update(additional_metadata)

    data_with_metadata["cache_metadata"] = cache_metadata

    # Save timestamped file (if enabled)
    if strategy_config.get("timestamped", {}).get("enabled", True):
        save_json(timestamped_file, data_with_metadata)

    # Save/update latest pointer (if enabled)
    if strategy_config.get("latest", {}).get("enabled", True):
        latest_data = data_with_metadata.copy()
        if strategy_config.get("latest", {}).get("include_timestamped_ref", True):
            latest_data["cache_metadata"]["timestamped_file"] = timestamped_filename
        save_json(latest_file, latest_data)

    # Update index file (if enabled)
    if strategy_config.get("index", {}).get("enabled", True):
        index_data = load_json(index_file, default={"entries": []})

        index_entry = {
            "timestamp": timestamp,
            "backbone": backbone,
            "identifier": identifier,
            "timestamped_file": timestamped_filename,
            "latest_file": latest_file.name,
        }

        # Add full metadata if configured
        if strategy_config.get("index", {}).get("include_metadata", True):
            # Add relevant fields from data
            if "selection_criteria" in data:
                index_entry["best_value"] = data.get(
                    "selection_criteria", {}).get("best_value")
            if "metrics" in data:
                index_entry["metrics"] = {
                    "macro-f1": data["metrics"].get("macro-f1"),
                    "loss": data["metrics"].get("loss"),
                }

        index_data.setdefault("entries", []).append(index_entry)

        # Apply max entries limit
        max_entries = strategy_config.get("index", {}).get("max_entries", 20)
        if max_entries and len(index_data["entries"]) > max_entries:
            index_data["entries"] = index_data["entries"][-max_entries:]

        save_json(index_file, index_data)

    return timestamped_file, latest_file, index_file


def load_cache_file(
    root_dir: Path,
    config_dir: Path,
    cache_type: str,
    use_latest: bool = True,
    specific_timestamp: Optional[str] = None,
    specific_identifier: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Load cache file with flexible options.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        cache_type: Type of cache.
        use_latest: If True, load from latest cache.
        specific_timestamp: Load specific timestamped file.
        specific_identifier: Load by identifier (trial name or run_id).

    Returns:
        Cache data or None if not found.
    """
    cache_dir = resolve_output_path(
        root_dir, config_dir, "cache", subcategory=cache_type)
    strategy_config = get_cache_strategy_config(config_dir, cache_type)

    # Load by specific timestamp
    if specific_timestamp:
        pattern = f"*_{specific_timestamp}.json"
        matching_files = list(cache_dir.glob(pattern))
        if matching_files:
            return load_json(matching_files[0])
        return None

    # Load by identifier from index
    if specific_identifier:
        index_file = get_cache_file_path(
            root_dir, config_dir, cache_type, file_type="index")
        index_data = load_json(index_file, default={"entries": []})

        for entry in reversed(index_data.get("entries", [])):
            if entry.get("identifier") == specific_identifier:
                cache_file = cache_dir / entry["timestamped_file"]
                return load_json(cache_file)
        return None

    # Load latest (default)
    if use_latest and strategy_config.get("latest", {}).get("enabled", True):
        latest_file = get_cache_file_path(
            root_dir, config_dir, cache_type, file_type="latest")
        cache_data = load_json(latest_file, default=None)
        if cache_data:
            return cache_data

    # Fallback: Load from index (most recent)
    if strategy_config.get("index", {}).get("enabled", True):
        index_file = get_cache_file_path(
            root_dir, config_dir, cache_type, file_type="index")
        index_data = load_json(index_file, default={"entries": []})

        if index_data.get("entries"):
            latest_entry = index_data["entries"][-1]
            cache_file = cache_dir / latest_entry["timestamped_file"]
            return load_json(cache_file)

    return None

