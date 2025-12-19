"""Pytest configuration hooks to integrate YAML test configurations.

This module automatically loads and applies settings from config/test/*.yaml files
when pytest runs. It ensures that:
- Coverage thresholds from execution.yaml are enforced
- Environment-specific settings are applied
- Markers from execution.yaml are registered
- Test fixtures can access YAML configuration data
"""

import pytest
import sys
from pathlib import Path

# Add tests directory to path so we can import config_loader
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

try:
    from config_loader import (
        get_execution_settings,
        get_environment_settings,
        get_coverage_threshold,
    )
    YAML_CONFIG_AVAILABLE = True
except ImportError:
    YAML_CONFIG_AVAILABLE = False


def pytest_configure(config):
    """Configure pytest using YAML test configurations."""
    if not YAML_CONFIG_AVAILABLE:
        # If config_loader is not available, skip YAML configuration
        return
    
    try:
        # Load execution settings
        execution_settings = get_execution_settings()
        
        # Apply coverage thresholds if pytest-cov is available
        if config.pluginmanager.has_plugin("pytest_cov"):
            coverage_config = execution_settings.get("coverage", {})
            overall_threshold = coverage_config.get("overall_threshold", 80)
            
            # Set coverage threshold in pytest-cov
            if hasattr(config.option, "cov_fail_under"):
                if config.option.cov_fail_under is None:
                    config.option.cov_fail_under = overall_threshold
                # If user specified a threshold, use the higher of the two
                elif config.option.cov_fail_under < overall_threshold:
                    config.option.cov_fail_under = overall_threshold
            
            # Store module thresholds for later use
            module_thresholds = coverage_config.get("module_thresholds", {})
            config._yaml_module_thresholds = module_thresholds
            
            # Add module-specific thresholds as markers for reporting
            for module, threshold in module_thresholds.items():
                marker_name = f"coverage_{module}"
                if not any(marker_name in str(m) for m in config.getini("markers")):
                    config.addinivalue_line(
                        "markers",
                        f"{marker_name}: Coverage threshold for {module} module (target: {threshold}%)"
                    )
        
        # Apply environment-specific settings
        env_settings = get_environment_settings()
        
        # Apply execution settings
        execution = env_settings.get("execution", {})
        if execution.get("verbose") and not config.option.verbose:
            config.option.verbose = 1
        
        # Register markers from execution.yaml
        markers = execution_settings.get("markers", {})
        for marker_name, marker_config in markers.items():
            description = marker_config.get("description", "")
            # Check if marker already exists
            existing_markers = config.getini("markers")
            if not any(marker_name in str(m) for m in existing_markers):
                config.addinivalue_line("markers", f"{marker_name}: {description}")
        
        # Store settings in config for access by other hooks
        config._yaml_execution_settings = execution_settings
        config._yaml_env_settings = env_settings
        
    except Exception as e:
        # Don't fail if config loading fails - tests should still run
        # Just print a warning
        import warnings
        warnings.warn(f"Failed to load YAML test configuration: {e}. Tests will run with defaults.")


def pytest_collection_modifyitems(config, items):
    """Modify test items based on YAML configuration."""
    if not YAML_CONFIG_AVAILABLE:
        return
    
    try:
        env_settings = get_environment_settings()
        execution = env_settings.get("execution", {})
        
        # Apply show_capture setting if not already set
        show_capture = execution.get("show_capture", "no")
        if show_capture and hasattr(config.option, "tbstyle"):
            # Map show_capture values to pytest options
            capture_map = {
                "no": "no",
                "log": "short",
                "all": "long"
            }
            if show_capture in capture_map and not hasattr(config.option, "_tbstyle_set"):
                config.option.tbstyle = capture_map[show_capture]
                config.option._tbstyle_set = True
    except Exception:
        pass  # Ignore errors in collection modification


def pytest_report_header(config):
    """Add YAML configuration info to pytest report header."""
    if not YAML_CONFIG_AVAILABLE:
        return []
    
    try:
        execution_settings = get_execution_settings()
        coverage_config = execution_settings.get("coverage", {})
        overall_threshold = coverage_config.get("overall_threshold", 80)
        
        header = [
            f"YAML Test Configuration: Loaded from config/test/",
            f"  Overall Coverage Threshold: {overall_threshold}%",
        ]
        
        module_thresholds = coverage_config.get("module_thresholds", {})
        if module_thresholds:
            header.append("  Module Thresholds:")
            for module, threshold in sorted(module_thresholds.items()):
                header.append(f"    - {module}: {threshold}%")
        
        return header
    except Exception:
        return []

