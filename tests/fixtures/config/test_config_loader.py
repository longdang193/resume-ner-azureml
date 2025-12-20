"""Configuration loading for HPO pipeline tests.

This module is responsible solely for loading and caching test configuration
from YAML files. It provides no business logic or presentation functionality.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Config cache
_config_cache: Optional[Dict[str, Any]] = None


def _get_config_file_path(root_dir: Optional[Path] = None) -> Path:
    """
    Get path to HPO pipeline test config file.
    
    Args:
        root_dir: Project root directory (auto-detects if None)
        
    Returns:
        Path to config file
    """
    if root_dir is None:
        # Auto-detect root directory from this file's location
        this_file = Path(__file__)
        root_dir = this_file.parent.parent.parent.parent
    
    return root_dir / "config" / "test" / "hpo_pipeline.yaml"


def load_hpo_test_config(
    config_path: Optional[Path] = None,
    root_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load HPO pipeline test configuration from YAML file.
    
    Args:
        config_path: Explicit path to config file (overrides root_dir)
        root_dir: Project root directory (used if config_path not provided)
        
    Returns:
        Dictionary with test configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None:
        config_path = _get_config_file_path(root_dir)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"HPO pipeline test config not found: {config_path}\n"
            "Create config/test/hpo_pipeline.yaml with test configuration."
        )
    
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config.get("hpo_pipeline_tests", {})


def get_test_config(root_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get cached test configuration (singleton pattern).
    
    Args:
        root_dir: Project root directory (for first load only)
        
    Returns:
        Dictionary with test configuration
    """
    global _config_cache
    
    if _config_cache is None:
        _config_cache = load_hpo_test_config(root_dir=root_dir)
    
    return _config_cache


def reset_config_cache() -> None:
    """
    Reset the config cache (useful for testing).
    """
    global _config_cache
    _config_cache = None


# Constants loaded from config (with defaults)
# These are initialized on module import from config/test/hpo_pipeline.yaml
try:
    _config = get_test_config()
    _defaults = _config.get("defaults", {})
    
    DEFAULT_RANDOM_SEED = _defaults.get("random_seed", 42)
    MINIMAL_K_FOLDS = _defaults.get("minimal_k_folds", 2)
    BACKBONES_LIST = _defaults.get("backbones", ["distilbert"])
    # Derive default backbone from first item in backbones list (DRY principle)
    DEFAULT_BACKBONE = BACKBONES_LIST[0] if BACKBONES_LIST else "distilbert"
    METRIC_DECIMAL_PLACES = _defaults.get("metric_decimal_places", 4)
    SEPARATOR_WIDTH = _defaults.get("separator_width", 60)
    VERY_SMALL_VALIDATION_THRESHOLD = _defaults.get("very_small_validation_threshold", 2)
except (FileNotFoundError, Exception):
    # Fallback to defaults if config file not found or error loading
    DEFAULT_RANDOM_SEED = 42
    MINIMAL_K_FOLDS = 2
    BACKBONES_LIST = ["distilbert"]
    DEFAULT_BACKBONE = "distilbert"
    METRIC_DECIMAL_PLACES = 4
    SEPARATOR_WIDTH = 60
    VERY_SMALL_VALIDATION_THRESHOLD = 2

