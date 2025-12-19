"""Test configuration loader for YAML-based test settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from shared.yaml_utils import load_yaml


def _get_test_config_root() -> Path:
    """
    Get the path to the test configuration directory.

    Returns:
        Path to config/test directory relative to project root.
    """
    project_root = Path(__file__).parent.parent
    return project_root / "config" / "test"


@lru_cache(maxsize=1)
def _load_test_config(filename: str) -> Dict[str, Any]:
    """
    Load a test configuration YAML file with caching.

    Args:
        filename: Name of the YAML file (e.g., "fixtures.yaml").

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file cannot be parsed as valid YAML.
    """
    config_path = _get_test_config_root() / filename
    return load_yaml(config_path)


def get_fixture_data() -> Dict[str, Any]:
    """
    Get test fixture data from fixtures.yaml.

    Returns:
        Dictionary containing sample_data, label_mappings, entity_types, and mock_configs.
    """
    return _load_test_config("fixtures.yaml")


def get_execution_settings() -> Dict[str, Any]:
    """
    Get test execution settings from execution.yaml.

    Returns:
        Dictionary containing coverage thresholds, markers, and performance settings.
    """
    return _load_test_config("execution.yaml")


def get_mock_settings() -> Dict[str, Any]:
    """
    Get mock configuration settings from mocks.yaml.

    Returns:
        Dictionary containing tokenizer, model, and platform adapter mock settings.
    """
    return _load_test_config("mocks.yaml")


def get_environment_settings(env: Optional[str] = None) -> Dict[str, Any]:
    """
    Get environment-specific test settings from environments.yaml.

    Args:
        env: Environment name (e.g., "ci", "local"). If None, uses TEST_ENV
            environment variable or defaults to "default".

    Returns:
        Dictionary containing environment-specific settings merged with defaults.
    """
    if env is None:
        env = os.environ.get("TEST_ENV", "default")

    all_envs = _load_test_config("environments.yaml")
    default_settings = all_envs.get("default", {})
    env_settings = all_envs.get(env, {})

    merged = default_settings.copy()
    for key, value in env_settings.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    return merged


def get_coverage_threshold(module: Optional[str] = None) -> int:
    """
    Get coverage threshold for a specific module or overall.

    Args:
        module: Module name (e.g., "training", "orchestration"). If None,
            returns overall threshold.

    Returns:
        Coverage threshold percentage.
    """
    settings = get_execution_settings()
    coverage = settings.get("coverage", {})
    if module:
        module_thresholds = coverage.get("module_thresholds", {})
        return module_thresholds.get(module, coverage.get("overall_threshold", 80))
    return coverage.get("overall_threshold", 80)


def get_sample_resume_data() -> list[Dict[str, Any]]:
    """
    Get sample resume data entries.

    Returns:
        List of resume data dictionaries with text and annotations.
    """
    fixtures = get_fixture_data()
    return fixtures.get("sample_data", {}).get("resume_entries", [])


def get_label_mapping(name: str = "standard") -> Dict[str, int]:
    """
    Get label to ID mapping.

    Args:
        name: Name of the label mapping (default: "standard").

    Returns:
        Dictionary mapping label strings to integer IDs.
    """
    fixtures = get_fixture_data()
    return fixtures.get("label_mappings", {}).get(name, {})


def get_entity_types(name: str = "standard") -> list[str]:
    """
    Get entity type definitions.

    Args:
        name: Name of the entity type set (default: "standard").

    Returns:
        List of entity type strings.
    """
    fixtures = get_fixture_data()
    return fixtures.get("entity_types", {}).get(name, [])


def get_mock_config_template(name: str) -> Dict[str, Any]:
    """
    Get a mock configuration template.

    Args:
        name: Template name (e.g., "experiment", "data", "model", "training", "hpo", "env").

    Returns:
        Dictionary containing the mock configuration template.
    """
    fixtures = get_fixture_data()
    return fixtures.get("mock_configs", {}).get(name, {})


def get_tokenizer_mock_settings(tokenizer_type: str = "fast") -> Dict[str, Any]:
    """
    Get tokenizer mock settings.

    Args:
        tokenizer_type: Type of tokenizer ("fast" or "slow").

    Returns:
        Dictionary containing tokenizer mock settings.
    """
    mocks = get_mock_settings()
    return mocks.get("tokenizer", {}).get(tokenizer_type, {})


def clear_cache() -> None:
    """Clear the configuration cache. Useful for testing or reloading configs."""
    _load_test_config.cache_clear()

