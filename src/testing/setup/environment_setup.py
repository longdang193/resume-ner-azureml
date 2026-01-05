"""Environment setup for HPO pipeline tests.

This module is responsible solely for loading configurations, resolving paths,
and initializing MLflow tracking. It contains no business logic, orchestration,
or presentation functionality.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import yaml

from testing.fixtures.config.test_config_loader import get_test_config


def load_configs(
    root_dir: Path,
    hpo_config_path: Optional[Path] = None,
    train_config_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load HPO and training configuration files.

    Args:
        root_dir: Project root directory
        hpo_config_path: Optional path to HPO config file
        train_config_path: Optional path to train config file

    Returns:
        Tuple of (hpo_config, train_config) dictionaries

    Raises:
        FileNotFoundError: If config files don't exist
    """
    test_config = get_test_config(root_dir)
    configs_section = test_config.get("configs", {})
    config_dir = root_dir / "config"

    if hpo_config_path is None:
        hpo_config_rel = configs_section.get("hpo_config", "hpo/smoke.yaml")
        hpo_config_path = config_dir / hpo_config_rel
    if train_config_path is None:
        train_config_rel = configs_section.get("train_config", "train.yaml")
        train_config_path = config_dir / train_config_rel

    if not hpo_config_path.exists():
        raise FileNotFoundError(f"HPO config not found: {hpo_config_path}")
    if not train_config_path.exists():
        raise FileNotFoundError(f"Train config not found: {train_config_path}")

    with hpo_config_path.open("r", encoding="utf-8") as f:
        hpo_config = yaml.safe_load(f)
    # Store config path for reference
    hpo_config["_config_path"] = str(hpo_config_path.relative_to(root_dir))

    # Apply HPO overrides from test config (if any)
    hpo_overrides = test_config.get("hpo_overrides", {})
    if hpo_overrides:
        # Override max_trials if specified in test config
        if "max_trials" in hpo_overrides and hpo_overrides["max_trials"] is not None:
            if "sampling" not in hpo_config:
                hpo_config["sampling"] = {}
            hpo_config["sampling"]["max_trials"] = hpo_overrides["max_trials"]

    with train_config_path.open("r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)
    # Store config path for reference
    train_config["_config_path"] = str(train_config_path.relative_to(root_dir))

    return hpo_config, train_config


def resolve_paths(
    root_dir: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Resolve all directory paths needed for testing.

    Args:
        root_dir: Project root directory
        output_dir: Optional test output directory

    Returns:
        Dictionary with config_dir, output_dir, and deterministic_dataset paths
    """
    test_config = get_test_config(root_dir)
    datasets_section = test_config.get("datasets", {})
    output_section = test_config.get("output", {})

    config_dir = root_dir / "config"

    if output_dir is None:
        output_base = output_section.get("base_dir", "outputs/hpo_tests")
        output_dir = root_dir / output_base

    output_dir.mkdir(parents=True, exist_ok=True)

    deterministic_path = datasets_section.get(
        "deterministic_path", "dataset_tiny")
    deterministic_dataset = root_dir / deterministic_path

    return {
        "config_dir": config_dir,
        "output_dir": output_dir,
        "deterministic_dataset": deterministic_dataset,
    }


def initialize_mlflow(root_dir: Path) -> str:
    """
    Initialize MLflow tracking and return tracking URI.

    Args:
        root_dir: Project root directory

    Returns:
        MLflow tracking URI string
    """
    test_config = get_test_config(root_dir)
    output_section = test_config.get("output", {})
    mlflow_base = output_section.get("mlflow_dir", "mlruns")
    mlflow_dir = root_dir / mlflow_base
    mlflow_dir.mkdir(exist_ok=True)
    mlflow_tracking_uri = mlflow_dir.as_uri()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    return mlflow_tracking_uri


def setup_test_environment(
    root_dir: Path,
    hpo_config_path: Optional[Path] = None,
    train_config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Setup test environment: load configs, setup paths, initialize MLflow.

    Args:
        root_dir: Project root directory
        hpo_config_path: Optional path to HPO config file
        train_config_path: Optional path to train config file
        output_dir: Optional test output directory

    Returns:
        Dictionary with config_dir, hpo_config, train_config, output_dir,
        mlflow_tracking_uri, deterministic_dataset
    """
    hpo_config, train_config = load_configs(
        root_dir=root_dir,
        hpo_config_path=hpo_config_path,
        train_config_path=train_config_path,
    )

    paths = resolve_paths(root_dir=root_dir, output_dir=output_dir)

    mlflow_tracking_uri = initialize_mlflow(root_dir=root_dir)

    return {
        "config_dir": paths["config_dir"],
        "hpo_config": hpo_config,
        "train_config": train_config,
        "output_dir": paths["output_dir"],
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "deterministic_dataset": paths["deterministic_dataset"],
    }

