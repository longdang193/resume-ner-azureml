"""pytest fixtures for HPO pipeline integration tests.

This module provides shared fixtures for configuration, paths, and test data
used across HPO pipeline integration tests.
"""

from pathlib import Path

import pytest
import yaml

from testing.fixtures.config.test_config_loader import get_test_config


@pytest.fixture(scope="session")
def root_dir() -> Path:
    """
    Project root directory fixture.
    
    Auto-detects root directory from test file location.
    """
    test_file = Path(__file__)
    return test_file.parent.parent.parent


@pytest.fixture(scope="session")
def config_dir(root_dir: Path) -> Path:
    """
    Configuration directory fixture.
    
    Args:
        root_dir: Project root directory fixture
        
    Returns:
        Path to configuration directory
    """
    return root_dir / "config"


@pytest.fixture(scope="session")
def hpo_config(config_dir: Path, root_dir: Path) -> dict:
    """
    Loaded HPO config fixture.
    
    Args:
        config_dir: Configuration directory fixture
        root_dir: Project root directory fixture
        
    Returns:
        Dictionary with HPO configuration
        
    Raises:
        FileNotFoundError: If HPO config file doesn't exist
    """
    test_config = get_test_config(root_dir)
    configs_section = test_config.get("configs", {})
    hpo_config_rel = configs_section.get("hpo_config", "hpo/smoke.yaml")
    hpo_config_path = config_dir / hpo_config_rel
    
    if not hpo_config_path.exists():
        raise FileNotFoundError(f"HPO config not found: {hpo_config_path}")
    
    with hpo_config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def train_config(config_dir: Path, root_dir: Path) -> dict:
    """
    Loaded training config fixture.
    
    Args:
        config_dir: Configuration directory fixture
        root_dir: Project root directory fixture
        
    Returns:
        Dictionary with training configuration
        
    Raises:
        FileNotFoundError: If train config file doesn't exist
    """
    test_config = get_test_config(root_dir)
    configs_section = test_config.get("configs", {})
    train_config_rel = configs_section.get("train_config", "train.yaml")
    train_config_path = config_dir / train_config_rel
    
    if not train_config_path.exists():
        raise FileNotFoundError(f"Train config not found: {train_config_path}")
    
    with train_config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def deterministic_dataset(root_dir: Path) -> Path:
    """
    Path to deterministic dataset fixture.
    
    Args:
        root_dir: Project root directory fixture
        
    Returns:
        Path to deterministic dataset directory
    """
    test_config = get_test_config(root_dir)
    datasets_section = test_config.get("datasets", {})
    deterministic_path = datasets_section.get("deterministic_path", "dataset_tiny")
    return root_dir / deterministic_path


@pytest.fixture(scope="session")
def test_output_dir(root_dir: Path) -> Path:
    """
    Test output directory fixture.
    
    Args:
        root_dir: Project root directory fixture
        
    Returns:
        Path to test output directory (creates if doesn't exist)
    """
    test_config = get_test_config(root_dir)
    output_section = test_config.get("output", {})
    output_base = output_section.get("base_dir", "outputs/hpo_tests")
    output_dir = root_dir / output_base
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(scope="session")
def mlflow_tracking_uri(root_dir: Path) -> str:
    """
    MLflow tracking URI fixture.
    
    Args:
        root_dir: Project root directory fixture
        
    Returns:
        MLflow tracking URI string
    """
    test_config = get_test_config(root_dir)
    output_section = test_config.get("output", {})
    mlflow_base = output_section.get("mlflow_dir", "mlruns")
    mlflow_dir = root_dir / mlflow_base
    mlflow_dir.mkdir(exist_ok=True)
    return mlflow_dir.as_uri()

