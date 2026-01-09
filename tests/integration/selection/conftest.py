"""Shared fixtures for best model selection tests."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Optional


@pytest.fixture
def sample_selection_config():
    """Sample best_model_selection.yaml configuration matching the actual file."""
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
def custom_selection_config():
    """Custom selection configuration with non-default values."""
    return {
        "run": {
            "mode": "reuse_if_exists"
        },
        "objective": {
            "metric": "accuracy",
            "goal": "minimize"
        },
        "scoring": {
            "f1_weight": 0.5,
            "latency_weight": 0.5,
            "normalize_weights": False
        },
        "benchmark": {
            "required_metrics": ["latency_batch_1_ms", "throughput_samples_per_sec"]
        }
    }


@pytest.fixture
def minimal_selection_config():
    """Minimal selection configuration (only required fields)."""
    return {
        "objective": {
            "metric": "macro-f1",
            "goal": "maximize"
        },
        "scoring": {
            "f1_weight": 0.7,
            "latency_weight": 0.3
        },
        "benchmark": {
            "required_metrics": ["latency_batch_1_ms"]
        }
    }


@pytest.fixture
def mock_tags_config():
    """Mock tags configuration (TagsRegistry or dict)."""
    return {
        "grouping": {
            "study_key_hash": "tags.grouping.study_key_hash",
            "trial_key_hash": "tags.grouping.trial_key_hash"
        },
        "process": {
            "stage": "tags.process.stage",
            "backbone": "tags.process.backbone"
        }
    }


@pytest.fixture
def mock_benchmark_experiment():
    """Mock benchmark experiment dictionary."""
    return {
        "name": "test_experiment-benchmark",
        "id": "benchmark_experiment_id_123"
    }


@pytest.fixture
def mock_hpo_experiments():
    """Mock HPO experiments dictionary."""
    return {
        "distilbert": {
            "name": "test_experiment-hpo-distilbert",
            "id": "hpo_experiment_id_distilbert"
        },
        "deberta": {
            "name": "test_experiment-hpo-deberta",
            "id": "hpo_experiment_id_deberta"
        }
    }


@pytest.fixture
def mock_mlflow_run():
    """Create a mock MLflow run with required tags and metrics."""
    run = Mock()
    run.info.run_id = "test_run_id_123"
    run.info.experiment_id = "test_experiment_id"
    run.info.status = "FINISHED"
    run.info.start_time = 1234567890
    
    # Mock tags
    run.data.tags = {
        "tags.grouping.study_key_hash": "study_hash_123",
        "tags.grouping.trial_key_hash": "trial_hash_456",
        "tags.process.stage": "hpo",
        "tags.process.backbone": "distilbert"
    }
    
    # Mock metrics
    run.data.metrics = {
        "macro-f1": 0.75,
        "latency_batch_1_ms": 5.0
    }
    
    # Mock params
    run.data.params = {
        "backbone": "distilbert",
        "learning_rate": "2e-5"
    }
    
    return run


@pytest.fixture
def mock_benchmark_run(mock_mlflow_run):
    """Create a mock benchmark MLflow run."""
    benchmark_run = Mock()
    benchmark_run.info.run_id = "benchmark_run_id_123"
    benchmark_run.info.experiment_id = "benchmark_experiment_id"
    benchmark_run.info.status = "FINISHED"
    benchmark_run.info.start_time = 1234567890
    
    benchmark_run.data.tags = {
        "tags.grouping.study_key_hash": "study_hash_123",
        "tags.grouping.trial_key_hash": "trial_hash_456",
        "tags.process.backbone": "distilbert"
    }
    
    benchmark_run.data.metrics = {
        "latency_batch_1_ms": 5.0,
        "throughput_samples_per_sec": 200.0
    }
    
    return benchmark_run


@pytest.fixture
def mock_trial_run(mock_mlflow_run):
    """Create a mock trial run (has macro-f1 metric)."""
    trial_run = Mock()
    trial_run.info.run_id = "trial_run_id_123"
    trial_run.info.experiment_id = "hpo_experiment_id"
    trial_run.info.status = "FINISHED"
    trial_run.info.start_time = 1234567890
    
    trial_run.data.tags = {
        "tags.grouping.study_key_hash": "study_hash_123",
        "tags.grouping.trial_key_hash": "trial_hash_456",
        "tags.process.stage": "hpo",
        "tags.process.backbone": "distilbert"
    }
    
    trial_run.data.metrics = {
        "macro-f1": 0.75
    }
    
    trial_run.data.params = {
        "backbone": "distilbert"
    }
    
    return trial_run


@pytest.fixture
def mock_refit_run(mock_mlflow_run):
    """Create a mock refit run (has checkpoint artifacts)."""
    refit_run = Mock()
    refit_run.info.run_id = "refit_run_id_123"
    refit_run.info.experiment_id = "hpo_experiment_id"
    refit_run.info.status = "FINISHED"
    refit_run.info.start_time = 1234567890
    
    refit_run.data.tags = {
        "tags.grouping.study_key_hash": "study_hash_123",
        "tags.grouping.trial_key_hash": "trial_hash_456",
        "tags.process.stage": "hpo_refit",
        "tags.process.backbone": "distilbert"
    }
    
    refit_run.data.metrics = {}  # Refit runs don't have macro-f1
    
    refit_run.data.params = {
        "backbone": "distilbert"
    }
    
    return refit_run


@pytest.fixture
def mock_cache_data():
    """Create mock cache data for best model selection."""
    return {
        "cache_key": "test_cache_key_12345678",
        "schema_version": 1,
        "timestamp": "2026-01-08T20:00:00Z",
        "best_model": {
            "run_id": "test_run_id_123",
            "trial_run_id": "trial_run_id_123",
            "experiment_name": "test_experiment-hpo-distilbert",
            "experiment_id": "hpo_experiment_id",
            "backbone": "distilbert",
            "study_key_hash": "study_hash_123",
            "trial_key_hash": "trial_hash_456",
            "f1_score": 0.75,
            "latency_ms": 5.0,
            "composite_score": 0.85,
            "has_refit_run": True
        },
        "inputs_summary": {
            "experiment_name": "test_experiment",
            "benchmark_experiment_id": "benchmark_experiment_id_123"
        }
    }


@pytest.fixture
def mock_mlflow_client():
    """Create a mock MLflow client."""
    client = Mock()
    return client


@pytest.fixture
def sample_acquisition_config():
    """Sample artifact_acquisition.yaml configuration matching the actual file."""
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
def custom_acquisition_config():
    """Custom acquisition configuration with non-default values."""
    return {
        "priority": ["mlflow", "local"],
        "local": {
            "match_strategy": "metadata_run_id",
            "require_exact_match": False,
            "validate": False
        },
        "drive": {
            "enabled": False,
            "folder_path": "custom-checkpoints",
            "validate": False
        },
        "mlflow": {
            "enabled": True,
            "validate": False,
            "download_timeout": 600
        }
    }


@pytest.fixture
def mock_best_run_info():
    """Mock best run info dictionary for acquisition."""
    return {
        "run_id": "test_run_id_123",
        "study_key_hash": "study_hash_123",
        "trial_key_hash": "trial_hash_456",
        "backbone": "distilbert",
        "experiment_name": "test_experiment-hpo-distilbert"
    }


@pytest.fixture
def mock_checkpoint_path(tmp_path):
    """Create a mock valid checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    
    # Create essential checkpoint files
    (checkpoint_dir / "config.json").write_text('{"model_type": "bert"}')
    (checkpoint_dir / "pytorch_model.bin").write_bytes(b"fake_model_data")
    
    return checkpoint_dir
