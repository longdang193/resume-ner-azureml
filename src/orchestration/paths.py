"""Centralized path resolution from config."""

from typing import Dict, Optional
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from orchestration.tokens import (
    extract_placeholders,
    is_token_allowed,
    is_token_known,
)
from orchestration.normalize import normalize_for_path
from shared.yaml_utils import load_yaml
from shared.json_cache import save_json, load_json

logger = logging.getLogger(__name__)


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
        config = load_yaml(paths_config_path)
        try:
            validate_paths_config(config, paths_config_path)
        except Exception as e:
            raise RuntimeError(
                f"Invalid paths configuration in {paths_config_path}: {e}"
            ) from e
        return config
    else:
        # Return defaults if config doesn't exist (backward compatibility)
        return _get_default_paths()


def apply_env_overrides(
    paths_config: Dict[str, Any], storage_env: Optional[str]
) -> Dict[str, Any]:
    """
    Apply shallow env overrides (keyed by storage_env) to a paths config.

    Only whitelisted sections are merged (currently: base, outputs).
    Does not mutate the original config; returns a merged copy.
    """
    if not storage_env:
        return paths_config

    overrides = paths_config.get("env_overrides", {}).get(storage_env)
    if not overrides:
        return paths_config

    merged = dict(paths_config)

    if "base" in overrides and isinstance(overrides["base"], dict):
        merged_base = dict(paths_config.get("base", {}))
        merged_base.update(overrides["base"])
        merged["base"] = merged_base

    if "outputs" in overrides and isinstance(overrides["outputs"], dict):
        merged_outputs = dict(paths_config.get("outputs", {}))
        merged_outputs.update(overrides["outputs"])
        merged["outputs"] = merged_outputs

    return merged


def validate_paths_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> None:
    """
    Basic schema validation for paths.yaml.

    - v1 (no or schema_version=1): minimal checks.
    - v2 and above: fail fast on core keys and required patterns.
    """
    location = f" ({config_path})" if config_path is not None else ""
    schema_version_raw = config.get("schema_version", 1)

    try:
        schema_version = int(schema_version_raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"schema_version must be an integer, got {schema_version_raw!r}{location}"
        )

    if schema_version < 1:
        logger.warning(
            f"[paths.yaml] Unsupported schema_version={schema_version}{location}, "
            f"treating as v1 with minimal validation."
        )
        schema_version = 1

    # Core requirement: base.outputs must exist and be non-empty
    base = config.get("base")
    if not isinstance(base, dict):
        raise ValueError(
            f"[paths.yaml] 'base' section must be a mapping{location}")
    outputs_base = base.get("outputs")
    if not outputs_base or not isinstance(outputs_base, str):
        raise ValueError(
            f"[paths.yaml] 'base.outputs' must be a non-empty string{location}"
        )

    # v1: lenient beyond this
    if schema_version == 1:
        return

    # v2+: stricter checks for patterns
    patterns = config.get("patterns")
    if not isinstance(patterns, dict):
        raise ValueError(
            f"[paths.yaml] 'patterns' section must be a mapping for schema_version>=2{location}"
        )

    required_pattern_keys = [
        "final_training_v2",
        "conversion_v2",
        "best_config_v2",
        "hpo_v2",
        "benchmarking_v2",
    ]
    missing = [key for key in required_pattern_keys if key not in patterns]
    if missing:
        raise ValueError(
            f"[paths.yaml] Missing required pattern keys for schema_version>=2{location}: "
            f"{', '.join(missing)}"
        )

    # Validate placeholders for all string patterns (cache + v2)
    for pattern_name, pattern_value in patterns.items():
        if not isinstance(pattern_value, str):
            continue
        placeholders = extract_placeholders(pattern_value)
        for token in placeholders:
            if not is_token_known(token):
                if schema_version >= 2:
                    raise ValueError(
                        f"[paths.yaml] Unknown placeholder '{{{token}}}' in pattern '{pattern_name}'{location}"
                    )
                logger.warning(
                    f"[paths.yaml] Unknown placeholder '{{{token}}}' in pattern '{pattern_name}'{location}"
                )
                continue
            if not is_token_allowed(token, "path"):
                raise ValueError(
                    f"[paths.yaml] Placeholder '{{{token}}}' in pattern '{pattern_name}' "
                    f"is not allowed for path scope{location}"
                )


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
    base_outputs_path = Path(base_outputs)
    base_dir = base_outputs_path if base_outputs_path.is_absolute() else root_dir / \
        base_outputs

    if category == "cache" and "subcategory" in kwargs:
        # Special handling for cache subdirectories
        cache_sub = kwargs.pop("subcategory")
        cache_dir = paths_config["cache"][cache_sub]
        return base_dir / paths_config["outputs"]["cache"] / cache_dir

    output_dir = paths_config["outputs"].get(category, category)
    path = base_dir / output_dir

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
    norm_rules = paths_config.get("normalize_paths")
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
    base_outputs_path = Path(base_outputs)
    outputs_dir = (
        base_outputs_path if base_outputs_path.is_absolute() else root_dir / base_outputs
    )
    base_outputs_name = base_outputs_path.name if base_outputs_path.is_absolute() else base_outputs

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
    drive_path = drive_base / base_outputs_name / relative_path

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


# ============================================================================
# V2 Path Parsing and Detection Helpers
# ============================================================================


def parse_hpo_path_v2(path: Path) -> Optional[Dict[str, str]]:
    """
    Parse HPO v2 path to extract study8 and trial8 hashes.

    V2 pattern: {storage_env}/{model}/study-{study8}/trial-{trial8}
    Example: outputs/hpo/local/distilbert/study-350a79aa/trial-747428f2

    Args:
        path: Path to parse (can be full path or relative fragment).

    Returns:
        Dictionary with keys: 'study8', 'trial8', 'storage_env', 'model'
        Returns None if path doesn't match v2 pattern.
    """
    path_str = str(path)

    # Pattern to match: study-{8_char_hash}/trial-{8_char_hash}
    # Also capture storage_env and model from preceding components
    pattern = r'(?:.*/)?(?:outputs/hpo/)?([^/]+)/([^/]+)/study-([a-f0-9]{8})/trial-([a-f0-9]{8})'
    match = re.search(pattern, path_str)

    if match:
        storage_env, model, study8, trial8 = match.groups()
        return {
            'storage_env': storage_env,
            'model': model,
            'study8': study8,
            'trial8': trial8,
        }

    return None


def is_v2_path(path: Path) -> bool:
    """
    Detect if path follows v2 pattern (study-{study8}/trial-{trial8}).

    Args:
        path: Path to check.

    Returns:
        True if path matches v2 pattern, False otherwise.
    """
    path_str = str(path)
    # Check for v2 pattern: study-{8_char_hash}/trial-{8_char_hash}
    v2_pattern = r'study-[a-f0-9]{8}/trial-[a-f0-9]{8}'
    return bool(re.search(v2_pattern, path_str))


def find_study_by_hash(
    root_dir: Path,
    config_dir: Path,
    model: str,
    study_key_hash: str
) -> Optional[Path]:
    """
    Find study folder by study_key_hash using v2 pattern.

    Searches for study folder matching: outputs/hpo/{storage_env}/{model}/study-{study8}/
    where study8 = study_key_hash[:8]

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        model: Model backbone name.
        study_key_hash: Full study key hash.

    Returns:
        Path to study folder if found, None otherwise.
    """
    if not study_key_hash or len(study_key_hash) < 8:
        return None

    study8 = study_key_hash[:8]

    # Get HPO base directory
    hpo_base = resolve_output_path(root_dir, config_dir, "hpo")

    # Search for study-{study8} pattern in all storage_env directories
    for storage_env_dir in hpo_base.iterdir():
        if not storage_env_dir.is_dir():
            continue

        model_dir = storage_env_dir / model
        if not model_dir.exists():
            continue

        # Look for study-{study8} folder
        study_folder = model_dir / f"study-{study8}"
        if study_folder.exists() and study_folder.is_dir():
            return study_folder

    return None


def find_trial_by_hash(
    root_dir: Path,
    config_dir: Path,
    model: str,
    study_key_hash: str,
    trial_key_hash: str
) -> Optional[Path]:
    """
    Find trial folder by study_key_hash and trial_key_hash using v2 pattern.

    Searches for trial folder matching:
    outputs/hpo/{storage_env}/{model}/study-{study8}/trial-{trial8}/
    where study8 = study_key_hash[:8], trial8 = trial_key_hash[:8]

    Args:
        root_dir: Project root directory.
        config_dir: Config directory.
        model: Model backbone name.
        study_key_hash: Full study key hash.
        trial_key_hash: Full trial key hash.

    Returns:
        Path to trial folder if found, None otherwise.
    """
    if not study_key_hash or len(study_key_hash) < 8:
        return None
    if not trial_key_hash or len(trial_key_hash) < 8:
        return None

    study8 = study_key_hash[:8]
    trial8 = trial_key_hash[:8]

    # Get HPO base directory
    hpo_base = resolve_output_path(root_dir, config_dir, "hpo")

    # Search for trial-{trial8} pattern in study-{study8} folders
    for storage_env_dir in hpo_base.iterdir():
        if not storage_env_dir.is_dir():
            continue

        model_dir = storage_env_dir / model
        if not model_dir.exists():
            continue

        study_folder = model_dir / f"study-{study8}"
        if not study_folder.exists():
            continue

        trial_folder = study_folder / f"trial-{trial8}"
        if trial_folder.exists() and trial_folder.is_dir():
            return trial_folder

    return None
