"""Shared fixtures for benchmarking tests."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Optional


@pytest.fixture
def sample_benchmark_config():
    """Sample benchmark.yaml configuration matching the actual file."""
    return {
        "benchmarking": {
            "batch_sizes": [1, 8, 16],
            "iterations": 100,
            "warmup_iterations": 10,
            "max_length": 512,
            "device": None,
            "test_data": None
        },
        "output": {
            "filename": "benchmark.json"
        }
    }


@pytest.fixture
def custom_benchmark_config():
    """Custom benchmark configuration with non-default values."""
    return {
        "benchmarking": {
            "batch_sizes": [2, 4, 32],
            "iterations": 200,
            "warmup_iterations": 20,
            "max_length": 256,
            "device": "cuda",
            "test_data": "/custom/path/test.json"
        },
        "output": {
            "filename": "custom_benchmark.json"
        }
    }


@pytest.fixture
def minimal_benchmark_config():
    """Minimal benchmark configuration (only required fields)."""
    return {
        "benchmarking": {
            "batch_sizes": [1]
        },
        "output": {
            "filename": "benchmark.json"
        }
    }


@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    """Create a mock checkpoint directory with model files."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    
    # Create minimal model files (tokenizer and model config)
    (checkpoint_dir / "tokenizer_config.json").write_text('{"vocab_size": 1000}')
    (checkpoint_dir / "config.json").write_text('{"model_type": "distilbert"}')
    (checkpoint_dir / "model.safetensors").write_bytes(b"fake_model_data")
    
    return checkpoint_dir


@pytest.fixture
def mock_test_data_file(tmp_path):
    """Create a mock test data JSON file."""
    test_data = tmp_path / "test.json"
    test_data.write_text(json.dumps([
        {"text": "Sample text 1", "labels": ["O", "O", "O"]},
        {"text": "Sample text 2", "labels": ["O", "B-PER", "I-PER"]},
    ]))
    return test_data


@pytest.fixture
def mock_best_trial_info(mock_checkpoint_dir):
    """Create mock best trial information dict."""
    return {
        "backbone": "distilbert",
        "trial_id": "0",
        "trial_name": "trial_0",
        "checkpoint_dir": str(mock_checkpoint_dir),
        "trial_dir": str(mock_checkpoint_dir.parent),
        "hyperparameters": {
            "learning_rate": 2e-5,
            "batch_size": 4,
            "dropout": 0.1
        },
        "metrics": {
            "macro-f1": 0.75
        }
    }


@pytest.fixture
def mock_best_trials(mock_best_trial_info):
    """Create mock best_trials dictionary."""
    return {
        "distilbert": mock_best_trial_info
    }


@pytest.fixture
def mock_mlflow_tracker():
    """Create a mock MLflow benchmark tracker."""
    tracker = Mock()
    tracker.experiment_name = "test_experiment-benchmark"
    
    # Mock context manager for start_benchmark_run
    mock_run = Mock()
    mock_run.run_id = "test_run_id_123"
    mock_context = MagicMock()
    mock_context.__enter__ = Mock(return_value=mock_run)
    mock_context.__exit__ = Mock(return_value=None)
    tracker.start_benchmark_run = Mock(return_value=mock_context)
    tracker.log_benchmark_results = Mock()
    
    return tracker


@pytest.fixture
def mock_data_config():
    """Create mock data configuration."""
    return {
        "dataset_name": "test_dataset",
        "dataset_version": "v1.0",
        "dataset_path": "/path/to/dataset"
    }


@pytest.fixture
def mock_hpo_config():
    """Create mock HPO configuration."""
    return {
        "sampling": {
            "algorithm": "random",
            "max_trials": 10
        },
        "objective": {
            "metric": "macro-f1",
            "goal": "maximize"
        }
    }


@pytest.fixture
def mock_benchmark_output(tmp_path):
    """Create a mock benchmark output JSON file."""
    output_file = tmp_path / "benchmark.json"
    output_file.write_text(json.dumps({
        "batch_size_1": {
            "mean_latency_ms": 5.0,
            "std_latency_ms": 0.5,
            "throughput_samples_per_sec": 200.0
        },
        "batch_size_8": {
            "mean_latency_ms": 20.0,
            "std_latency_ms": 2.0,
            "throughput_samples_per_sec": 400.0
        }
    }))
    return output_file

