"""
@meta
name: paths_parse
type: utility
domain: paths
responsibility:
  - Parse HPO and other output paths to extract components
  - Detect path patterns and versions
inputs:
  - Path objects
outputs:
  - Parsed path components dictionaries
tags:
  - utility
  - paths
  - parsing
ci:
  runnable: false
  needs_gpu: false
  needs_cloud: false
lifecycle:
  status: active
"""

"""Path parsing and detection helpers."""

import re
from pathlib import Path
from typing import Dict, Optional

from .config import load_paths_config
from .resolve import resolve_output_path


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

    # Check if directory exists before iterating
    if not hpo_base.exists() or not hpo_base.is_dir():
        return None

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

    # Check if directory exists before iterating
    if not hpo_base.exists() or not hpo_base.is_dir():
        return None

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

