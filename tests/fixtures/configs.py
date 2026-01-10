"""Shared config fixtures for tests."""

import pytest


@pytest.fixture
def hpo_config_smoke():
    """Load and return smoke.yaml HPO config structure."""
    return {
        "search_space": {
            "backbone": {"type": "choice", "values": ["distilbert"]},
            "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 5e-5},
            "batch_size": {"type": "choice", "values": [4]},
            "dropout": {"type": "uniform", "min": 0.1, "max": 0.3},
            "weight_decay": {"type": "loguniform", "min": 0.001, "max": 0.1},
        },
        "sampling": {"algorithm": "random", "max_trials": 1, "timeout_minutes": 20},
        "checkpoint": {
            "enabled": True,
            "study_name": "hpo_distilbert_smoke_test",
            "storage_path": "{study_name}/study.db",
            "auto_resume": True,
            "save_only_best": True,
        },
        "mlflow": {"log_best_checkpoint": True},
        "early_termination": {
            "policy": "bandit",
            "evaluation_interval": 1,
            "slack_factor": 0.2,
            "delay_evaluation": 2,
        },
        "objective": {"metric": "macro-f1", "goal": "maximize"},
        "selection": {
            "accuracy_threshold": 0.015,
            "use_relative_threshold": True,
            "min_accuracy_gain": 0.02,
        },
        "k_fold": {
            "enabled": True,
            "n_splits": 2,
            "random_seed": 42,
            "shuffle": True,
            "stratified": True,
        },
        "refit": {"enabled": True},
        "cleanup": {
            "disable_auto_cleanup": False,
            "disable_auto_optuna_mark": False,
        },
    }


@pytest.fixture
def hpo_config_minimal():
    """Minimal HPO config for simple tests."""
    return {
        "search_space": {
            "backbone": {"type": "choice", "values": ["distilbert"]},
            "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 5e-5},
            "batch_size": {"type": "choice", "values": [4]},
        },
        "sampling": {"algorithm": "random", "max_trials": 1, "timeout_minutes": 20},
        "checkpoint": {
            "enabled": True,
            "study_name": "hpo_test",
            "storage_path": "{study_name}/study.db",
            "auto_resume": True,
        },
        "objective": {"metric": "macro-f1", "goal": "maximize"},
        "k_fold": {"enabled": False},
        "refit": {"enabled": False},
    }


@pytest.fixture
def selection_config_default():
    """Default best_model_selection.yaml configuration."""
    return {
        "run": {
            "mode": "force_new"
        },
        "objective": {
            "metric": "macro-f1",
            "goal": "maximize"
        },
        "scoring": {
            "f1_weight": 0.7,
            "latency_weight": 0.3,
            "normalize_weights": True
        },
        "benchmark": {
            "required_metrics": ["latency_batch_1_ms"]
        }
    }


@pytest.fixture
def acquisition_config_default():
    """Default artifact_acquisition.yaml configuration."""
    return {
        "priority": ["local", "drive", "mlflow"],
        "local": {
            "match_strategy": "tags",
            "require_exact_match": True,
            "validate": True
        },
        "drive": {
            "enabled": True,
            "folder_path": "resume-ner-checkpoints",
            "validate": True
        },
        "mlflow": {
            "enabled": True,
            "validate": True,
            "download_timeout": 300
        }
    }


@pytest.fixture
def conversion_config_default():
    """Default conversion.yaml configuration."""
    return {
        "onnx": {
            "opset_version": 18,
            "quantize_int8": False
        },
        "smoke_test": {
            "enabled": True,
            "max_samples": 10
        }
    }






