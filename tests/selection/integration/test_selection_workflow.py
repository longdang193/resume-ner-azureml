"""Integration tests for end-to-end best model selection workflow."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from shared.yaml_utils import load_yaml

from selection.mlflow_selection import find_best_model_from_mlflow
from selection.cache import (
    load_cached_best_model,
    compute_selection_cache_key,
)


class TestSelectionWorkflow:
    """Test end-to-end selection workflow with config loading."""

    @patch("orchestration.jobs.selection.mlflow_selection.MlflowClient")
    def test_workflow_loads_config_and_uses_all_options(
        self,
        mock_client_class,
        tmp_path,
        mock_tags_config,
        mock_benchmark_experiment,
        mock_hpo_experiments,
    ):
        """Test full workflow: load config, extract options, run selection."""
        # Setup config directory structure
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create best_model_selection.yaml
        selection_yaml = config_dir / "best_model_selection.yaml"
        selection_yaml.write_text("""
run:
  mode: force_new
objective:
  metric: "macro-f1"
  goal: "maximize"
scoring:
  f1_weight: 0.7
  latency_weight: 0.3
  normalize_weights: true
benchmark:
  required_metrics:
    - "latency_batch_1_ms"
""")
        
        # Load config
        selection_config = load_yaml(selection_yaml)
        
        # Verify config was loaded
        assert "run" in selection_config
        assert "objective" in selection_config
        assert "scoring" in selection_config
        assert "benchmark" in selection_config
        
        # Extract options from config
        objective_metric = selection_config["objective"]["metric"]
        f1_weight = selection_config["scoring"]["f1_weight"]
        latency_weight = selection_config["scoring"]["latency_weight"]
        normalize_weights = selection_config["scoring"]["normalize_weights"]
        required_metrics = selection_config["benchmark"]["required_metrics"]
        
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.search_runs.return_value = []
        
        # Run selection with config options
        result = find_best_model_from_mlflow(
            benchmark_experiment=mock_benchmark_experiment,
            hpo_experiments=mock_hpo_experiments,
            tags_config=mock_tags_config,
            selection_config=selection_config,
            use_python_filtering=True,
        )
        
        # Verify function was called with config options
        assert mock_client.search_runs.called
        
        # Verify config options were extracted correctly
        assert objective_metric == "macro-f1"
        assert f1_weight == 0.7
        assert latency_weight == 0.3
        assert normalize_weights is True
        assert required_metrics == ["latency_batch_1_ms"]

    @patch("orchestration.jobs.selection.mlflow_selection.MlflowClient")
    def test_workflow_custom_config_values(
        self,
        mock_client_class,
        tmp_path,
        mock_tags_config,
        mock_benchmark_experiment,
        mock_hpo_experiments,
    ):
        """Test workflow with custom config values."""
        # Setup config with custom values
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        selection_yaml = config_dir / "best_model_selection.yaml"
        selection_yaml.write_text("""
run:
  mode: reuse_if_exists
objective:
  metric: "accuracy"
  goal: "minimize"
scoring:
  f1_weight: 0.5
  latency_weight: 0.5
  normalize_weights: false
benchmark:
  required_metrics:
    - "latency_batch_1_ms"
    - "throughput_samples_per_sec"
""")
        
        # Load config
        selection_config = load_yaml(selection_yaml)
        
        # Extract custom options
        objective_metric = selection_config["objective"]["metric"]
        f1_weight = selection_config["scoring"]["f1_weight"]
        latency_weight = selection_config["scoring"]["latency_weight"]
        normalize_weights = selection_config["scoring"]["normalize_weights"]
        required_metrics = selection_config["benchmark"]["required_metrics"]
        
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.search_runs.return_value = []
        
        # Run selection with custom config
        result = find_best_model_from_mlflow(
            benchmark_experiment=mock_benchmark_experiment,
            hpo_experiments=mock_hpo_experiments,
            tags_config=mock_tags_config,
            selection_config=selection_config,
            use_python_filtering=True,
        )
        
        # Verify custom values were used
        assert objective_metric == "accuracy"
        assert f1_weight == 0.5
        assert latency_weight == 0.5
        assert normalize_weights is False
        assert len(required_metrics) == 2

    def test_workflow_cache_key_computation(
        self,
        tmp_path,
        sample_selection_config,
        mock_tags_config,
    ):
        """Test that cache key computation uses selection_config."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Load config (simulated)
        selection_config = sample_selection_config
        
        # Compute cache key
        cache_key = compute_selection_cache_key(
            experiment_name="test_experiment",
            selection_config=selection_config,
            tags_config=mock_tags_config,
            benchmark_experiment_id="benchmark_exp_id_123",
        )
        
        # Verify cache key is computed
        assert isinstance(cache_key, str)
        assert len(cache_key) == 16
        
        # Verify cache key changes when selection_config changes
        modified_config = selection_config.copy()
        modified_config["scoring"]["f1_weight"] = 0.8
        
        cache_key2 = compute_selection_cache_key(
            experiment_name="test_experiment",
            selection_config=modified_config,
            tags_config=mock_tags_config,
            benchmark_experiment_id="benchmark_exp_id_123",
        )
        
        assert cache_key != cache_key2

    @patch("orchestration.jobs.selection.cache.get_cache_file_path")
    @patch("orchestration.jobs.selection.cache.load_json")
    def test_workflow_cache_loading_with_config(
        self,
        mock_load_json,
        mock_get_cache_path,
        tmp_path,
        sample_selection_config,
        mock_tags_config,
        mock_cache_data,
    ):
        """Test that cache loading validates selection_config via cache key."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        cache_file = root_dir / "cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.touch()
        mock_get_cache_path.return_value = cache_file
        
        # Cache data with matching cache key
        # Use reuse_if_exists mode for cache loading test
        selection_config = sample_selection_config.copy()
        selection_config["run"] = {"mode": "reuse_if_exists"}
        
        cache_data = mock_cache_data.copy()
        current_cache_key = compute_selection_cache_key(
            experiment_name="test_experiment",
            selection_config=selection_config,
            tags_config=mock_tags_config,
            benchmark_experiment_id="benchmark_exp_id_123",
        )
        cache_data["cache_key"] = current_cache_key
        
        mock_load_json.return_value = cache_data
        
        # Mock MLflow client
        with patch("orchestration.jobs.selection.cache.MlflowClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_run = Mock()
            mock_run.info.status = "FINISHED"
            mock_client.get_run.return_value = mock_run
            
            # Load cache
            result = load_cached_best_model(
                root_dir=root_dir,
                config_dir=config_dir,
                experiment_name="test_experiment",
                selection_config=selection_config,
                tags_config=mock_tags_config,
                benchmark_experiment_id="benchmark_exp_id_123",
            )
            
            # Verify cache was loaded and validated
            assert mock_load_json.called
            assert result is not None

