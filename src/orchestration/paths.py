"""Centralized path resolution from config."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from shared.yaml_utils import load_yaml
from shared.json_cache import save_json, load_json


def load_paths_config(config_dir: Path) -> Dict[str, Any]:
    """
    Load paths configuration from config/paths.yaml.

    Args:
        config_dir: Configuration directory (ROOT_DIR / "config").

    Returns:
        Dictionary containing paths configuration, or defaults if file doesn't exist.
    """
    paths_config_path = config_dir / "paths.yaml"
    if paths_config_path.exists():
        return load_yaml(paths_config_path)
    else:
        # Return defaults if config doesn't exist (backward compatibility)
        return _get_default_paths()


def _get_default_paths() -> Dict[str, Any]:
    """Default paths (backward compatible)."""
    return {
        "base": {"outputs": "outputs"},
        "outputs": {
            "hpo": "hpo",
            "benchmarking": "benchmarking",
            "final_training": "final_training",
            "conversion": "conversion",
            "best_model_selection": "best_model_selection",
            "cache": "cache",
        },
        "cache": {
            "best_configurations": "best_configurations",
            "final_training": "final_training",
        },
        "files": {
            "metrics": "metrics.json",
            "benchmark": "benchmark.json",
            "checkpoint_dir": "checkpoint",
            "cache": {
                "best_config_latest": "latest_best_configuration.json",
                "best_config_index": "index.json",
                "final_training_latest": "latest_final_training_cache.json",
                "final_training_index": "final_training_index.json",
            },
        },
        "patterns": {
            "best_config_file": "best_config_{backbone}_{trial}_{timestamp}.json",
            "final_training_cache_file": "final_training_{backbone}_{run_id}_{timestamp}.json",
        },
        "cache_strategies": {
            "best_configurations": {
                "strategy": "dual",
                "timestamped": {"enabled": True},
                "latest": {"enabled": True, "filename": "latest_best_configuration.json"},
                "index": {"enabled": True, "filename": "index.json", "max_entries": 20},
            },
            "final_training": {
                "strategy": "dual",
                "timestamped": {"enabled": True},
                "latest": {"enabled": True, "filename": "latest_final_training_cache.json"},
                "index": {"enabled": True, "filename": "final_training_index.json", "max_entries": 20},
            },
        },
    }


def resolve_output_path(
    root_dir: Path,
    config_dir: Path,
    category: str,
    **kwargs
) -> Path:
    """
    Resolve output path from config.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory (root_dir / "config").
        category: Output category (e.g., "hpo", "final_training", "cache").
        **kwargs: Additional path components or pattern replacements.

    Returns:
        Resolved path.

    Examples:
        resolve_output_path(ROOT_DIR, CONFIG_DIR, "hpo")
        # -> outputs/hpo

        resolve_output_path(ROOT_DIR, CONFIG_DIR, "cache", subcategory="best_configurations")
        # -> outputs/cache/best_configurations

        resolve_output_path(ROOT_DIR, CONFIG_DIR, "final_training", 
                          backbone="distilbert", run_id="20251227_220407")
        # -> outputs/final_training/distilbert_20251227_220407
    """
    paths_config = load_paths_config(config_dir)
    base_outputs = paths_config["base"]["outputs"]

    if category == "cache" and "subcategory" in kwargs:
        # Special handling for cache subdirectories
        cache_sub = kwargs.pop("subcategory")
        cache_dir = paths_config["cache"][cache_sub]
        return root_dir / base_outputs / paths_config["outputs"]["cache"] / cache_dir

    output_dir = paths_config["outputs"].get(category, category)
    path = root_dir / base_outputs / output_dir

    # Apply pattern replacements if provided
    if kwargs:
        pattern = paths_config.get("patterns", {}).get(category, "")
        if pattern:
            for key, value in kwargs.items():
                pattern = pattern.replace(f"{{{key}}}", str(value))
            path = path / pattern
        else:
            # Append as subdirectories
            for value in kwargs.values():
                path = path / str(value)

    return path


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
    timestamp: str
) -> str:
    """
    Generate timestamped cache filename from pattern.

    Args:
        config_dir: Config directory.
        cache_type: Type of cache ("best_configurations" or "final_training").
        backbone: Backbone name.
        identifier: Identifier (trial name for best_config, run_id for final_training).
        timestamp: Timestamp string.

    Returns:
        Generated filename.
    """
    paths_config = load_paths_config(config_dir)

    # Get pattern based on cache type
    if cache_type == "best_configurations":
        pattern = paths_config.get("patterns", {}).get("best_config_file",
                                                       "best_config_{backbone}_{trial}_{timestamp}.json")
        # Use 'identifier' placeholder for trial
        pattern = pattern.replace("{trial}", "{identifier}")
    elif cache_type == "final_training":
        pattern = paths_config.get("patterns", {}).get("final_training_cache_file",
                                                       "final_training_{backbone}_{run_id}_{timestamp}.json")
        # Use 'identifier' placeholder for run_id
        pattern = pattern.replace("{run_id}", "{identifier}")
    else:
        # Generic pattern
        pattern = "{cache_type}_{backbone}_{identifier}_{timestamp}.json"
        pattern = pattern.replace("{cache_type}", cache_type)

    # Sanitize values for filename
    backbone_safe = backbone.replace('-', '_').replace('/', '_')
    identifier_safe = identifier.replace('-', '_').replace('/', '_')

    return pattern.format(
        backbone=backbone_safe,
        identifier=identifier_safe,
        timestamp=timestamp
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
    from datetime import datetime

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


def get_drive_backup_base(config_dir: Path) -> Optional[Path]:
    """
    Get base Google Drive backup directory from config.

    Args:
        config_dir: Config directory.

    Returns:
        Base Drive backup path (e.g., /content/drive/MyDrive/resume-ner-checkpoints), 
        or None if not configured.

    Examples:
        get_drive_backup_base(CONFIG_DIR)
        # -> Path("/content/drive/MyDrive/resume-ner-checkpoints")
    """
    paths_config = load_paths_config(config_dir)
    drive_config = paths_config.get("drive", {})

    if not drive_config:
        return None

    mount_point = drive_config.get("mount_point", "/content/drive")
    backup_base = drive_config.get("backup_base_dir", "resume-ner-checkpoints")

    return Path(mount_point) / "MyDrive" / backup_base


def get_drive_backup_path(
    root_dir: Path,
    config_dir: Path,
    local_path: Path
) -> Optional[Path]:
    """
    Convert local output path to Drive backup path, mirroring structure.

    Only paths within outputs/ can be backed up. The function automatically
    mirrors the exact same directory structure from outputs/ to Drive.

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        local_path: Local file or directory path to backup (must be within outputs/).

    Returns:
        Equivalent Drive backup path, or None if Drive not configured or path outside outputs/.

    Examples:
        Local: outputs/hpo/distilbert/trial_0/checkpoint/
        Drive:  /content/drive/MyDrive/resume-ner-checkpoints/outputs/hpo/distilbert/trial_0/checkpoint/

        Local: outputs/cache/best_configurations/latest_best_configuration.json
        Drive:  /content/drive/MyDrive/resume-ner-checkpoints/outputs/cache/best_configurations/latest_best_configuration.json
    """
    paths_config = load_paths_config(config_dir)
    drive_config = paths_config.get("drive", {})

    if not drive_config:
        return None

    # Get the base outputs directory
    base_outputs = paths_config["base"]["outputs"]
    outputs_dir = root_dir / base_outputs

    # Check if the local path is within outputs/
    try:
        relative_path = local_path.relative_to(outputs_dir)
    except ValueError:
        # Path is not within outputs/, can't mirror it
        return None

    # Get Drive base directory
    drive_base = get_drive_backup_base(config_dir)
    if not drive_base:
        return None

    # Build Drive path: mount_point/MyDrive/backup_base/outputs/relative_path
    drive_path = drive_base / base_outputs / relative_path

    return drive_path


# ============================================================================
# New centralized naming system (v2) - fingerprint-based paths
# ============================================================================

def resolve_output_path_v2(
    root_dir: Path,
    context: Any,  # NamingContext from naming_centralized
    base_outputs: str = "outputs"
) -> Path:
    """
    Resolve output path using new centralized naming system (v2).

    This is the new path resolution that uses fingerprint-based identity
    and environment-aware organization. The old resolve_output_path() remains
    for backward compatibility.

    Args:
        root_dir: Project root directory.
        context: NamingContext with all required information.
        base_outputs: Base outputs directory name (default: "outputs").

    Returns:
        Resolved path following new structure.

    Examples:
        # HPO
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            trial_id="trial_1_20251229_100000"
        )
        path = resolve_output_path_v2(ROOT_DIR, context)
        # -> outputs/hpo/local/distilbert/trial_1_20251229_100000/

        # Final training
        context = NamingContext(
            process_type="final_training",
            model="distilbert",
            environment="local",
            spec_fp="abc123...",
            exec_fp="xyz789...",
            variant=1
        )
        path = resolve_output_path_v2(ROOT_DIR, context)
        # -> outputs/final_training/local/distilbert/spec_abc123..._exec_xyz789.../v1/
    """
    from .naming_centralized import build_output_path

    return build_output_path(root_dir, context, base_outputs)
