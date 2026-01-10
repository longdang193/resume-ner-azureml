from __future__ import annotations

"""
@meta
name: naming_mlflow_config
type: utility
domain: naming
responsibility:
  - Load MLflow configuration from YAML with caching
  - Provide naming configuration accessors
inputs:
  - Configuration directories
outputs:
  - MLflow configuration dictionaries
tags:
  - utility
  - naming
  - mlflow
  - config
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""MLflow configuration loader for systematic naming settings."""
from pathlib import Path
from typing import Any, Dict, Optional

from common.shared.yaml_utils import load_yaml
from common.shared.logging_utils import get_logger

logger = get_logger(__name__)

# Module-level cache for loaded config (with mtime check)
_config_cache: Dict[tuple, tuple] = {}  # (config_dir, mtime) -> config

def _get_config_mtime(config_path: Path) -> float:
    """Get modification time of config file, or 0 if doesn't exist."""
    try:
        return config_path.stat().st_mtime
    except (OSError, FileNotFoundError):
        return 0.0

def load_mlflow_config(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load full MLflow config from config/mlflow.yaml with caching.
    
    Cache is invalidated when the file modification time changes.
    
    Args:
        config_dir: Path to config directory (defaults to current directory / "config").
    
    Returns:
        Full MLflow config dictionary, or empty dict if file not found.
    """
    if config_dir is None:
        config_dir = Path.cwd() / "config"
    
    config_path = config_dir / "mlflow.yaml"
    mtime = _get_config_mtime(config_path)
    cache_key = str(config_dir)
    
    # Check cache
    if cache_key in _config_cache:
        cached_mtime, cached_config = _config_cache[cache_key]
        if cached_mtime == mtime:
            return cached_config
        # Cache invalid, remove it
        del _config_cache[cache_key]
    
    # Load config
    if not config_path.exists():
        _config_cache[cache_key] = (mtime, {})
        return {}
    
    try:
        config = load_yaml(config_path)
        _config_cache[cache_key] = (mtime, config)
        return config
    except Exception as e:
        logger.warning(f"[MLflow Config] Failed to load config from {config_path}: {e}", exc_info=True)
        _config_cache[cache_key] = (mtime, {})
        return {}

def _validate_naming_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and apply defaults for naming config."""
    defaults = {
        "project_name": "resume-ner",
        "tags": {
            "max_length": 250,
            "sanitize": True,
        },
        "run_name": {
            "max_length": 100,
            "shorten_fingerprints": True,
        },
    }
    
    result = defaults.copy()
    
    # Validate and merge project_name
    if "project_name" in config:
        project_name = config["project_name"]
        if isinstance(project_name, str) and project_name:
            result["project_name"] = project_name
        else:
            logger.warning(f"Invalid project_name in config, using default: {defaults['project_name']}")
    
    # Validate and merge tags
    if "tags" in config and isinstance(config["tags"], dict):
        tags_config = config["tags"]
        if "max_length" in tags_config:
            max_length = tags_config["max_length"]
            if isinstance(max_length, int) and 0 < max_length <= 500:
                result["tags"]["max_length"] = max_length
            else:
                logger.warning(
                    f"Invalid tags.max_length in config (must be 1-500), "
                    f"using default: {defaults['tags']['max_length']}"
                )
        if "sanitize" in tags_config:
            if isinstance(tags_config["sanitize"], bool):
                result["tags"]["sanitize"] = tags_config["sanitize"]
            else:
                logger.warning(
                    f"Invalid tags.sanitize in config (must be bool), "
                    f"using default: {defaults['tags']['sanitize']}"
                )
    
    # Validate and merge run_name
    if "run_name" in config and isinstance(config["run_name"], dict):
        run_name_config = config["run_name"]
        if "max_length" in run_name_config:
            max_length = run_name_config["max_length"]
            if isinstance(max_length, int) and max_length > 0:
                result["run_name"]["max_length"] = max_length
            else:
                logger.warning(
                    f"Invalid run_name.max_length in config (must be > 0), "
                    f"using default: {defaults['run_name']['max_length']}"
                )
        if "shorten_fingerprints" in run_name_config:
            if isinstance(run_name_config["shorten_fingerprints"], bool):
                result["run_name"]["shorten_fingerprints"] = run_name_config["shorten_fingerprints"]
            else:
                logger.warning(
                    f"Invalid run_name.shorten_fingerprints in config (must be bool), "
                    f"using default: {defaults['run_name']['shorten_fingerprints']}"
                )
    
    return result

def get_naming_config(
    config_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get naming configuration with defaults.
    
    Args:
        config_dir: Path to config directory (defaults to current directory / "config").
        config: Optional pre-loaded config dict (avoids re-reading file).
    
    Returns:
        Dictionary with:
        - project_name: str
        - tags: {"max_length": int, "sanitize": bool}
        - run_name: {"max_length": int, "shorten_fingerprints": bool}
    """
    if config is None:
        config = load_mlflow_config(config_dir)
    
    naming_config = config.get("naming", {})
    return _validate_naming_config(naming_config)

def _validate_index_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and apply defaults for index config."""
    defaults = {
        "enabled": True,
        "max_entries": 1000,
        "file_name": "mlflow_index.json",
    }
    
    result = defaults.copy()
    
    # Validate enabled
    if "enabled" in config:
        if isinstance(config["enabled"], bool):
            result["enabled"] = config["enabled"]
        else:
            logger.warning(
                f"Invalid index.enabled in config (must be bool), "
                f"using default: {defaults['enabled']}"
            )
    
    # Validate max_entries
    if "max_entries" in config:
        max_entries = config["max_entries"]
        if isinstance(max_entries, int) and max_entries > 0:
            result["max_entries"] = max_entries
        else:
            logger.warning(
                f"Invalid index.max_entries in config (must be > 0), "
                f"using default: {defaults['max_entries']}"
            )
    
    # Validate file_name
    if "file_name" in config:
        file_name = config["file_name"]
        if isinstance(file_name, str) and file_name and "/" not in file_name and "\\" not in file_name:
            result["file_name"] = file_name
        else:
            logger.warning(
                f"Invalid index.file_name in config (must be valid filename), "
                f"using default: {defaults['file_name']}"
            )
    
    return result

def get_index_config(
    config_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get index configuration with defaults.
    
    Args:
        config_dir: Path to config directory (defaults to current directory / "config").
        config: Optional pre-loaded config dict (avoids re-reading file).
    
    Returns:
        Dictionary with:
        - enabled: bool
        - max_entries: int
        - file_name: str
    """
    if config is None:
        config = load_mlflow_config(config_dir)
    
    index_config = config.get("index", {})
    return _validate_index_config(index_config)

def _validate_run_finder_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and apply defaults for run_finder config."""
    defaults = {
        "strict_mode_default": True,
    }
    
    result = defaults.copy()
    
    # Validate strict_mode_default
    if "strict_mode_default" in config:
        if isinstance(config["strict_mode_default"], bool):
            result["strict_mode_default"] = config["strict_mode_default"]
        else:
            logger.warning(
                f"Invalid run_finder.strict_mode_default in config (must be bool), "
                f"using default: {defaults['strict_mode_default']}"
            )
    
    return result

def get_run_finder_config(
    config_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get run finder configuration with defaults.
    
    Args:
        config_dir: Path to config directory (defaults to current directory / "config").
        config: Optional pre-loaded config dict (avoids re-reading file).
    
    Returns:
        Dictionary with:
        - strict_mode_default: bool
    """
    if config is None:
        config = load_mlflow_config(config_dir)
    
    run_finder_config = config.get("run_finder", {})
    return _validate_run_finder_config(run_finder_config)

def _validate_auto_increment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and apply defaults for auto_increment config."""
    defaults = {
        "enabled": False,
        "processes": {
            "hpo": True,
            "benchmarking": True,
        },
        "format": "{base}.{version}",
    }
    
    result = defaults.copy()
    
    # Validate enabled
    if "enabled" in config:
        if isinstance(config["enabled"], bool):
            result["enabled"] = config["enabled"]
        else:
            logger.warning(
                f"Invalid auto_increment.enabled in config (must be bool), "
                f"using default: {defaults['enabled']}"
            )
    
    # Validate processes
    if "processes" in config and isinstance(config["processes"], dict):
        processes_config = config["processes"]
        for process_type in ["hpo", "benchmarking"]:
            if process_type in processes_config:
                if isinstance(processes_config[process_type], bool):
                    result["processes"][process_type] = processes_config[process_type]
                else:
                    logger.warning(
                        f"Invalid auto_increment.processes.{process_type} in config (must be bool), "
                        f"using default: {defaults['processes'][process_type]}"
                    )
    
    # Validate format
    if "format" in config:
        format_str = config["format"]
        if isinstance(format_str, str) and "{base}" in format_str and "{version}" in format_str:
            result["format"] = format_str
        else:
            logger.warning(
                f"Invalid auto_increment.format in config (must contain {{base}} and {{version}}), "
                f"using default: {defaults['format']}"
            )
    
    return result

def get_auto_increment_config(
    config_dir: Optional[Path] = None,
    process_type: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get auto-increment configuration with defaults.
    
    Args:
        config_dir: Path to config directory (defaults to current directory / "config").
        process_type: Optional process type ("hpo" or "benchmarking") to check if enabled for that process.
        config: Optional pre-loaded config dict (avoids re-reading file).
    
    Returns:
        Dictionary with:
        - enabled: bool (global toggle)
        - processes: {"hpo": bool, "benchmarking": bool}
        - format: str (format string with {base} and {version})
        If process_type provided, also includes:
        - enabled_for_process: bool (enabled AND enabled for specific process)
    """
    if config is None:
        config = load_mlflow_config(config_dir)
    
    naming_config = config.get("naming", {})
    run_name_config = naming_config.get("run_name", {})
    auto_inc_config = run_name_config.get("auto_increment", {})
    
    logger.info(
        f"[Auto-Increment Config] Loading from config_dir={config_dir}, "
        f"raw_auto_inc_config={auto_inc_config}"
    )
    
    result = _validate_auto_increment_config(auto_inc_config)
    
    logger.info(
        f"[Auto-Increment Config] Validated config: {result}, process_type={process_type}"
    )
    
    # If process_type provided, add enabled_for_process flag
    if process_type:
        global_enabled = result.get("enabled", False)
        process_enabled = result.get("processes", {}).get(process_type, False)
        result["enabled_for_process"] = global_enabled and process_enabled
    
    return result

def get_tracking_config(
    config_dir: Optional[Path] = None,
    stage: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get tracking configuration for a specific stage.
    
    Args:
        config_dir: Path to config directory (defaults to current directory / "config").
        stage: Stage name ("benchmark", "training", "conversion"). If None, returns all tracking config.
        config: Optional pre-loaded config dict (avoids re-reading file).
    
    Returns:
        Dictionary with tracking configuration for the stage, or all stages if stage is None.
        Defaults to enabled=True and log_*=True if not specified in config.
    """
    if config is None:
        config = load_mlflow_config(config_dir)
    
    tracking_config = config.get("tracking", {})
    
    if stage is None:
        return tracking_config
    
    stage_config = tracking_config.get(stage, {})
    
    # Apply defaults
    defaults = {
        "enabled": True,
    }
    
    # Stage-specific defaults
    if stage == "benchmark":
        defaults["log_artifacts"] = True
    elif stage == "training":
        defaults["log_checkpoint"] = True
        defaults["log_metrics_json"] = True
    elif stage == "conversion":
        defaults["log_onnx_model"] = True
        defaults["log_conversion_log"] = True
    
    result = defaults.copy()
    result.update(stage_config)
    
    return result

