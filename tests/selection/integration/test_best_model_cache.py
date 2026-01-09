"""Component tests for best model selection cache dual-file strategy."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from selection.cache import (
    load_cached_best_model,
    save_best_model_cache,
    compute_selection_cache_key,
)
from paths import get_cache_file_path


@pytest.fixture
def cache_dir(tmp_path):
    """Create cache directory structure."""
    cache_base = tmp_path / "outputs" / "cache" / "best_model_selection"
    cache_base.mkdir(parents=True)
    return cache_base


def test_cache_dual_file_strategy_creates_all_files(tmp_path, cache_dir):
    """Test that saving cache creates timestamped, latest, and index files."""
    root_dir = tmp_path
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    best_model = {
        "backbone": "distilbert-base-uncased",
        "run_id": "run-123",
        "params": {"learning_rate": 2e-5},
        "tags": {"code.stage": "hpo_refit"},
    }

    tags_config = {}
    selection_config = {}
    benchmark_experiment = {"id": "bench-123", "name": "bench-exp"}
    hpo_experiments = {"distilbert": {"id": "hpo-123", "name": "hpo-exp"}}

    # Save cache
    timestamped_file, latest_file, index_file = save_best_model_cache(
        root_dir=root_dir,
        config_dir=config_dir,
        best_model=best_model,
        experiment_name="test_experiment",
        selection_config=selection_config,
        tags_config=tags_config,
        benchmark_experiment=benchmark_experiment,
        hpo_experiments=hpo_experiments,
        tracking_uri="file:///tmp/mlflow",
        inputs_summary={},
    )

    # All three files should exist
    assert timestamped_file.exists()
    assert latest_file.exists()
    assert index_file.exists()

    # Latest file should contain the same data as timestamped
    latest_data = json.loads(latest_file.read_text())
    timestamped_data = json.loads(timestamped_file.read_text())
    assert latest_data["best_model"]["backbone"] == timestamped_data["best_model"]["backbone"]

    # Index file should contain entry
    index_data = json.loads(index_file.read_text())
    assert isinstance(index_data, (list, dict))
    if isinstance(index_data, list):
        assert len(index_data) > 0
    elif isinstance(index_data, dict) and "entries" in index_data:
        assert len(index_data["entries"]) > 0


def test_cache_load_valid_cache_with_mlflow_validation(tmp_path, cache_dir):
    """Test that load_cached_best_model validates cache and MLflow run."""
    root_dir = tmp_path
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Compute cache key for consistent test
    selection_config = {}
    tags_config = {}
    cache_key = compute_selection_cache_key(
        "test_experiment", selection_config, tags_config, "bench-123", "file:///tmp/mlflow"
    )

    # Create valid cache file with matching cache_key
    # Use get_cache_file_path to get the correct filename from config
    latest_file = get_cache_file_path(
        root_dir, config_dir, "best_model_selection", file_type="latest"
    )
    cache_data = {
        "schema_version": 1,
        "cache_key": cache_key,
        "best_model": {
            "backbone": "distilbert-base-uncased",
            "run_id": "run-123",
            "params": {"learning_rate": 2e-5},
        },
        "experiment_name": "test_experiment",
    }
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(json.dumps(cache_data))

    # Mock MLflow client to return valid run
    mock_client = Mock()
    mock_run = Mock()
    mock_run.info.status = "FINISHED"
    mock_client.get_run.return_value = mock_run

    with patch("orchestration.jobs.selection.cache.MlflowClient", return_value=mock_client):
        result = load_cached_best_model(
            root_dir=root_dir,
            config_dir=config_dir,
            experiment_name="test_experiment",
            selection_config=selection_config,
            tags_config=tags_config,
            benchmark_experiment_id="bench-123",
            tracking_uri="file:///tmp/mlflow",
        )

        assert result is not None
        assert result["best_model"]["backbone"] == "distilbert-base-uncased"


def test_cache_load_cache_key_mismatch_returns_none(tmp_path, cache_dir):
    """Test that cache_key mismatch causes cache to be ignored."""
    root_dir = tmp_path
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create cache with different cache_key
    latest_file = get_cache_file_path(
        root_dir, config_dir, "best_model_selection", file_type="latest"
    )
    cache_data = {
        "schema_version": 1,
        "cache_key": "old_cache_key_1234",  # Different from current
        "best_model": {"backbone": "distilbert", "run_id": "run-123"},
    }
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(json.dumps(cache_data))

    # Current config will produce different cache_key
    result = load_cached_best_model(
        root_dir=root_dir,
        config_dir=config_dir,
        experiment_name="test_experiment",
        selection_config={},
        tags_config={},
        benchmark_experiment_id="bench-123",
        tracking_uri="file:///tmp/mlflow",
    )

    # Should return None due to cache_key mismatch
    assert result is None


def test_cache_partial_write_recovery(tmp_path, cache_dir):
    """Test that cache handles partial writes gracefully."""
    root_dir = tmp_path
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Simulate partial write: index updated but latest missing
    index_file = cache_dir / "best_model_selection_index.json"
    index_data = {
        "entries": [
            {
                "timestamp": "20260108_120000",
                "backbone": "distilbert",
                "file": "best_model_selection_distilbert_20260108_120000.json",
            }
        ]
    }
    index_file.write_text(json.dumps(index_data))

    # Latest file is missing (partial write scenario)
    latest_file = cache_dir / "latest_best_model_selection_cache.json"
    assert not latest_file.exists()

    # Timestamped file exists
    timestamped_file = cache_dir / index_data["entries"][0]["file"]
    cache_key = compute_selection_cache_key(
        "test_experiment", {}, {}, "bench-123", "file:///tmp/mlflow"
    )
    timestamped_data = {
        "schema_version": 1,
        "cache_key": cache_key,
        "best_model": {"backbone": "distilbert", "run_id": "run-123"},
        "experiment_name": "test_experiment",
    }
    timestamped_file.write_text(json.dumps(timestamped_data))

    # Load should handle this gracefully (either use timestamped or return None)
    # This tests that the system doesn't crash on partial writes
    # Since latest is missing, load should return None (it only checks latest file)
    mock_client = Mock()
    mock_run = Mock()
    mock_run.info.status = "FINISHED"
    mock_client.get_run.return_value = mock_run

    with patch("orchestration.jobs.selection.cache.MlflowClient", return_value=mock_client):
        result = load_cached_best_model(
            root_dir=root_dir,
            config_dir=config_dir,
            experiment_name="test_experiment",
            selection_config={},
            tags_config={},
            benchmark_experiment_id="bench-123",
            tracking_uri="file:///tmp/mlflow",
        )
        # Should return None when latest file is missing (partial write scenario)
        assert result is None


def test_cache_missing_metrics_handling(tmp_path, cache_dir):
    """Test cache behavior when best_model has missing/NaN metrics."""
    root_dir = tmp_path
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Compute cache key
    selection_config = {}
    tags_config = {}
    cache_key = compute_selection_cache_key(
        "test_experiment", selection_config, tags_config, "bench-123", "file:///tmp/mlflow"
    )

    # Create cache with missing metrics
    best_model_missing_metrics = {
        "backbone": "distilbert-base-uncased",
        "run_id": "run-123",
        "params": {"learning_rate": 2e-5},
        "tags": {"code.stage": "hpo_refit"},
        # Missing metrics field
    }

    cache_data = {
        "schema_version": 1,
        "cache_key": cache_key,
        "best_model": best_model_missing_metrics,
        "experiment_name": "test_experiment",
    }

    latest_file = get_cache_file_path(
        root_dir, config_dir, "best_model_selection", file_type="latest"
    )
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(json.dumps(cache_data))

    # Mock MLflow client
    mock_client = Mock()
    mock_run = Mock()
    mock_run.info.status = "FINISHED"
    mock_client.get_run.return_value = mock_run

    # Load should handle missing metrics gracefully
    with patch("orchestration.jobs.selection.cache.MlflowClient", return_value=mock_client):
        result = load_cached_best_model(
            root_dir=root_dir,
            config_dir=config_dir,
            experiment_name="test_experiment",
            selection_config=selection_config,
            tags_config=tags_config,
            benchmark_experiment_id="bench-123",
            tracking_uri="file:///tmp/mlflow",
        )

        # Should return data even with missing metrics
        assert result is not None
        assert "metrics" not in result.get("best_model", {}) or result["best_model"].get("metrics") is None

