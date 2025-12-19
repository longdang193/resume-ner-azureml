"""Shared pytest fixtures for all tests."""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock

import pytest

try:
    from config_loader import (
        get_sample_resume_data,
        get_label_mapping,
        get_mock_config_template,
        get_tokenizer_mock_settings,
        get_execution_settings,
        get_environment_settings,
        get_coverage_threshold,
    )
    YAML_CONFIG_AVAILABLE = True
except ImportError:
    get_sample_resume_data = None
    get_label_mapping = None
    get_mock_config_template = None
    get_tokenizer_mock_settings = None
    get_execution_settings = None
    get_environment_settings = None
    get_coverage_threshold = None
    YAML_CONFIG_AVAILABLE = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_resume_data() -> List[Dict[str, Any]]:
    """Sample resume data for testing. Loaded from YAML config if available."""
    if get_sample_resume_data is not None:
        try:
            return get_sample_resume_data()
        except Exception:
            pass

    return [
        {
            "text": "John Doe is a software engineer with 5 years of experience.",
            "annotations": [
                [0, 8, "PERSON"],
                [30, 48, "JOB_TITLE"],
                [54, 55, "EXPERIENCE"],
            ],
        },
        {
            "text": "Jane Smith worked at Microsoft from 2020 to 2023.",
            "annotations": [
                [0, 10, "PERSON"],
                [22, 31, "ORG"],
                [37, 41, "DATE"],
                [45, 49, "DATE"],
            ],
        },
        {
            "text": "Python, Java, and SQL are my main skills.",
            "annotations": [
                [0, 6, "SKILL"],
                [8, 12, "SKILL"],
                [21, 24, "SKILL"],
            ],
        },
    ]


@pytest.fixture
def sample_dataset_dict(sample_resume_data, temp_dir) -> Dict[str, Any]:
    """Create a sample dataset dictionary with train.json file."""
    train_file = temp_dir / "train.json"
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(sample_resume_data, f)

    return {
        "train": sample_resume_data,
        "validation": [],
        "path": temp_dir,
    }


@pytest.fixture
def sample_dataset_with_validation(sample_resume_data, temp_dir) -> Dict[str, Any]:
    """Create a sample dataset with both train and validation files."""
    train_data = sample_resume_data[:2]
    val_data = sample_resume_data[2:]

    train_file = temp_dir / "train.json"
    val_file = temp_dir / "validation.json"

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f)

    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_data, f)

    return {
        "train": train_data,
        "validation": val_data,
        "path": temp_dir,
    }


@pytest.fixture
def mock_configs(temp_dir) -> Dict[str, Path]:
    """Create mock configuration files. Loaded from YAML config if available."""
    config_dir = temp_dir / "config"
    config_dir.mkdir()

    if get_mock_config_template is not None:
        try:
            experiment_config = get_mock_config_template("experiment")
            data_config = get_mock_config_template("data")
            model_config = get_mock_config_template("model")
            train_config = {"training": get_mock_config_template("training")}
            hpo_config = get_mock_config_template("hpo")
            env_config = get_mock_config_template("env")
        except Exception:
            experiment_config = None
            data_config = None
            model_config = None
            train_config = None
            hpo_config = None
            env_config = None
    else:
        experiment_config = None
        data_config = None
        model_config = None
        train_config = None
        hpo_config = None
        env_config = None

    if experiment_config is None:
        experiment_config = {
            "experiment_name": "test_experiment",
            "data_config": "data/resume_tiny.yaml",
            "model_config": "model/distilbert.yaml",
            "train_config": "train.yaml",
            "hpo_config": "hpo/smoke.yaml",
            "env_config": "env/local.yaml",
        }

    if data_config is None:
        data_config = {
            "version": "1.0",
            "schema": {
                "entity_types": ["PERSON", "ORG", "JOB_TITLE", "SKILL", "DATE", "EXPERIENCE"],
            },
        }

    if model_config is None:
        model_config = {
            "backbone": "distilbert-base-uncased",
            "dropout": 0.1,
        }

    if train_config is None:
        train_config = {
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
            },
        }

    if hpo_config is None:
        hpo_config = {
            "search_space": {
                "learning_rate": {
                    "type": "loguniform",
                    "min": 1e-5,
                    "max": 5e-5,
                },
            },
            "sampling": {
                "algorithm": "random",
                "max_trials": 2,
            },
        }

    if env_config is None:
        env_config = {
            "environment_type": "local",
        }

    # Create subdirectories
    (config_dir / "data").mkdir()
    (config_dir / "model").mkdir()
    (config_dir / "hpo").mkdir()
    (config_dir / "env").mkdir()
    (config_dir / "experiment").mkdir()

    # Write config files
    configs = {
        "experiment": config_dir / "experiment" / "test_experiment.yaml",
        "data": config_dir / "data" / "resume_tiny.yaml",
        "model": config_dir / "model" / "distilbert.yaml",
        "train": config_dir / "train.yaml",
        "hpo": config_dir / "hpo" / "smoke.yaml",
        "env": config_dir / "env" / "local.yaml",
    }

    import yaml

    with open(configs["experiment"], "w") as f:
        yaml.dump(experiment_config, f)
    with open(configs["data"], "w") as f:
        yaml.dump(data_config, f)
    # Also create resume_v1.yaml as expected by build_training_config
    with open(config_dir / "data" / "resume_v1.yaml", "w") as f:
        yaml.dump(data_config, f)
    with open(configs["model"], "w") as f:
        yaml.dump(model_config, f)
    with open(configs["train"], "w") as f:
        yaml.dump(train_config, f)
    with open(configs["hpo"], "w") as f:
        yaml.dump(hpo_config, f)
    with open(configs["env"], "w") as f:
        yaml.dump(env_config, f)

    return {
        "root": config_dir,
        **configs,
    }


@pytest.fixture
def mock_fast_tokenizer():
    """Mock a fast tokenizer with offset mapping support. Settings loaded from YAML if available."""
    tokenizer = MagicMock()
    tokenizer.is_fast = True

    if get_tokenizer_mock_settings is not None:
        try:
            settings = get_tokenizer_mock_settings("fast")
            default_max_length = settings.get("default_max_length", 512)
            cls_token_id = settings.get("cls_token_id", 101)
            sep_token_id = settings.get("sep_token_id", 102)
        except Exception:
            default_max_length = 512
            cls_token_id = 101
            sep_token_id = 102
    else:
        default_max_length = 512
        cls_token_id = 101
        sep_token_id = 102

    def tokenize(text, **kwargs):
        tokens = text.split()
        max_length = kwargs.get("max_length", default_max_length)
        truncation = kwargs.get("truncation", False)

        if truncation and len(tokens) > max_length - 2:
            tokens = tokens[:max_length - 2]

        token_ids = list(range(len(tokens)))

        offset_mapping = []
        start = 0
        for token in tokens:
            end = start + len(token)
            offset_mapping.append([start, end])
            start = end + 1

        result = {
            "input_ids": [[cls_token_id] + token_ids + [sep_token_id]],
            "attention_mask": [[1] * (len(token_ids) + 2)],
            "offset_mapping": [offset_mapping],
        }

        if kwargs.get("return_tensors") == "pt":
            try:
                import torch
                result = {k: torch.tensor(v) for k, v in result.items()}
            except ImportError:
                pass

        return result

    tokenizer.side_effect = tokenize
    return tokenizer


@pytest.fixture
def mock_slow_tokenizer():
    """Mock a slow tokenizer without offset mapping support. Settings loaded from YAML if available."""
    tokenizer = MagicMock()
    tokenizer.is_fast = False

    if get_tokenizer_mock_settings is not None:
        try:
            settings = get_tokenizer_mock_settings("slow")
            cls_token_id = settings.get("cls_token_id", 101)
            sep_token_id = settings.get("sep_token_id", 102)
        except Exception:
            cls_token_id = 101
            sep_token_id = 102
    else:
        cls_token_id = 101
        sep_token_id = 102

    def tokenize(text, **kwargs):
        tokens = text.split()
        token_ids = list(range(len(tokens)))

        result = {
            "input_ids": [[cls_token_id] + token_ids + [sep_token_id]],
            "attention_mask": [[1] * (len(token_ids) + 2)],
        }

        if kwargs.get("return_tensors") == "pt":
            try:
                import torch
                result = {k: torch.tensor(v) for k, v in result.items()}
            except ImportError:
                pass

        return result

    tokenizer.side_effect = tokenize
    return tokenizer


@pytest.fixture
def label2id() -> Dict[str, int]:
    """Standard label to ID mapping for testing. Loaded from YAML config if available."""
    if get_label_mapping is not None:
        try:
            return get_label_mapping("standard")
        except Exception:
            pass

    return {
        "O": 0,
        "PERSON": 1,
        "ORG": 2,
        "JOB_TITLE": 3,
        "SKILL": 4,
        "DATE": 5,
        "EXPERIENCE": 6,
    }


# Pytest configuration hooks to integrate YAML test configurations
def pytest_configure(config):
    """Configure pytest using YAML test configurations."""
    if not YAML_CONFIG_AVAILABLE:
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
                existing_markers = config.getini("markers")
                if not any(marker_name in str(m) for m in existing_markers):
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
                config.addinivalue_line(
                    "markers", f"{marker_name}: {description}")

        # Store settings in config for access by other hooks
        config._yaml_execution_settings = execution_settings
        config._yaml_env_settings = env_settings

    except Exception as e:
        # Don't fail if config loading fails - tests should still run
        import warnings
        warnings.warn(
            f"Failed to load YAML test configuration: {e}. Tests will run with defaults.")


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
