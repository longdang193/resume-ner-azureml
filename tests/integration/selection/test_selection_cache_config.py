"""Component tests for selection cache behavior with best_model_selection.yaml config."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from orchestration.jobs.selection.cache import (
    load_cached_best_model,
    compute_selection_cache_key,
    save_best_model_cache,
)


class TestSelectionCacheConfig:
    """Test that cache functions use selection_config correctly."""

    def test_compute_selection_cache_key_includes_selection_config(
        self,
        sample_selection_config,
        mock_tags_config,
    ):
        """Test that compute_selection_cache_key includes selection_config."""
        experiment_name = "test_experiment"
        benchmark_experiment_id = "benchmark_exp_id_123"
        
        # Compute cache key
        cache_key = compute_selection_cache_key(
            experiment_name=experiment_name,
            selection_config=sample_selection_config,
            tags_config=mock_tags_config,
            benchmark_experiment_id=benchmark_experiment_id,
        )
        
        # Verify cache key is computed (non-empty string)
        assert isinstance(cache_key, str)
        assert len(cache_key) == 16  # SHA256 hex digest truncated to 16 chars
        
        # Verify cache key changes when selection_config changes
        modified_config = sample_selection_config.copy()
        modified_config["scoring"]["f1_weight"] = 0.8
        
        cache_key2 = compute_selection_cache_key(
            experiment_name=experiment_name,
            selection_config=modified_config,
            tags_config=mock_tags_config,
            benchmark_experiment_id=benchmark_experiment_id,
        )
        
        # Cache keys should be different
        assert cache_key != cache_key2

    def test_compute_selection_cache_key_deterministic(
        self,
        sample_selection_config,
        mock_tags_config,
    ):
        """Test that compute_selection_cache_key is deterministic."""
        experiment_name = "test_experiment"
        benchmark_experiment_id = "benchmark_exp_id_123"
        
        # Compute cache key twice
        cache_key1 = compute_selection_cache_key(
            experiment_name=experiment_name,
            selection_config=sample_selection_config,
            tags_config=mock_tags_config,
            benchmark_experiment_id=benchmark_experiment_id,
        )
        
        cache_key2 = compute_selection_cache_key(
            experiment_name=experiment_name,
            selection_config=sample_selection_config,
            tags_config=mock_tags_config,
            benchmark_experiment_id=benchmark_experiment_id,
        )
        
        # Cache keys should be identical
        assert cache_key1 == cache_key2

    @patch("orchestration.jobs.selection.cache.MlflowClient")
    @patch("orchestration.jobs.selection.cache.get_cache_file_path")
    @patch("orchestration.jobs.selection.cache.load_json")
    def test_load_cached_best_model_validates_cache_key(
        self,
        mock_load_json,
        mock_get_cache_path,
        mock_client_class,
        tmp_path,
        sample_selection_config,
        mock_tags_config,
        mock_cache_data,
    ):
        """Test that load_cached_best_model validates cache key from selection_config."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        cache_file = root_dir / "cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.touch()  # Create file so it exists
        mock_get_cache_path.return_value = cache_file
        
        # Use reuse_if_exists mode for cache loading test
        selection_config = sample_selection_config.copy()
        selection_config["run"] = {"mode": "reuse_if_exists"}
        
        # Cache data with matching cache key
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
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_run = Mock()
        mock_run.info.status = "FINISHED"
        mock_client.get_run.return_value = mock_run
        
        # Call function
        result = load_cached_best_model(
            root_dir=root_dir,
            config_dir=config_dir,
            experiment_name="test_experiment",
            selection_config=selection_config,
            tags_config=mock_tags_config,
            benchmark_experiment_id="benchmark_exp_id_123",
        )
        
        # Verify cache key was validated (load_json was called)
        assert mock_load_json.called
        
        # If cache key matches, result should be returned
        # (validation happens at line 116 in cache.py)
        assert result is not None

    @patch("orchestration.jobs.selection.cache.get_cache_file_path")
    @patch("orchestration.jobs.selection.cache.load_json")
    def test_load_cached_best_model_cache_key_mismatch(
        self,
        mock_load_json,
        mock_get_cache_path,
        tmp_path,
        sample_selection_config,
        mock_tags_config,
        mock_cache_data,
    ):
        """Test that load_cached_best_model returns None when cache key mismatches."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Setup mocks
        cache_file = root_dir / "cache.json"
        mock_get_cache_path.return_value = cache_file
        
        # Cache data with mismatched cache key
        cache_data = mock_cache_data.copy()
        cache_data["cache_key"] = "different_cache_key_123"
        
        mock_load_json.return_value = cache_data
        
        # Call function
        result = load_cached_best_model(
            root_dir=root_dir,
            config_dir=config_dir,
            experiment_name="test_experiment",
            selection_config=sample_selection_config,
            tags_config=mock_tags_config,
            benchmark_experiment_id="benchmark_exp_id_123",
        )
        
        # Should return None due to cache key mismatch
        assert result is None

    @patch("orchestration.jobs.selection.cache.save_cache_with_dual_strategy")
    def test_save_best_model_cache_includes_selection_config(
        self,
        mock_save_cache,
        tmp_path,
        sample_selection_config,
        mock_tags_config,
        mock_benchmark_experiment,
        mock_hpo_experiments,
    ):
        """Test that save_best_model_cache includes selection_config in cache data."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        best_model = {
            "run_id": "test_run_id_123",
            "backbone": "distilbert"
        }
        
        # Mock save_cache_with_dual_strategy to return tuple
        mock_save_cache.return_value = (
            Path("timestamped.json"),
            Path("latest.json"),
            Path("index.json")
        )
        
        # Call function
        save_best_model_cache(
            root_dir=root_dir,
            config_dir=config_dir,
            best_model=best_model,
            experiment_name="test_experiment",
            selection_config=sample_selection_config,
            tags_config=mock_tags_config,
            benchmark_experiment=mock_benchmark_experiment,
            hpo_experiments=mock_hpo_experiments,
        )
        
        # Verify save_cache_with_dual_strategy was called
        assert mock_save_cache.called
        
        # Verify selection_config is included in cache data
        call_args = mock_save_cache.call_args
        cache_data = call_args.kwargs.get("data", {})
        
        # Cache key computation includes selection_config (line 50 in cache.py)
        # So the cache data should have a cache_key that depends on selection_config
        assert "cache_key" in cache_data

    @patch("orchestration.jobs.selection.cache.get_cache_file_path")
    @patch("orchestration.jobs.selection.cache.load_json")
    def test_run_mode_reuse_if_exists_behavior(
        self,
        mock_load_json,
        mock_get_cache_path,
        tmp_path,
        sample_selection_config,
        mock_tags_config,
        mock_cache_data,
    ):
        """Test that run.mode=reuse_if_exists allows cache reuse."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Set mode to reuse_if_exists (sample_selection_config has force_new by default)
        selection_config = sample_selection_config.copy()
        selection_config["run"] = {"mode": "reuse_if_exists"}
        
        # Setup mocks
        cache_file = root_dir / "cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.touch()
        mock_get_cache_path.return_value = cache_file
        
        # Cache data with matching cache key
        cache_data = mock_cache_data.copy()
        from orchestration.jobs.selection.cache import compute_selection_cache_key
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
            
            # Call function
            result = load_cached_best_model(
                root_dir=root_dir,
                config_dir=config_dir,
                experiment_name="test_experiment",
                selection_config=selection_config,
                tags_config=mock_tags_config,
                benchmark_experiment_id="benchmark_exp_id_123",
            )
            
            # Should load cache when mode is reuse_if_exists
            assert mock_load_json.called
            assert result is not None

    @patch("orchestration.jobs.selection.cache.get_cache_file_path")
    @patch("orchestration.jobs.selection.cache.load_json")
    def test_run_mode_force_new_behavior(
        self,
        mock_load_json,
        mock_get_cache_path,
        tmp_path,
        sample_selection_config,
        mock_tags_config,
        mock_cache_data,
    ):
        """Test that run.mode=force_new skips cache and returns None immediately."""
        root_dir = tmp_path / "outputs"
        config_dir = tmp_path / "config"
        root_dir.mkdir()
        config_dir.mkdir()
        
        # Set mode to force_new
        selection_config = sample_selection_config.copy()
        selection_config["run"]["mode"] = "force_new"
        
        # Setup mocks (even if cache exists, it should be skipped)
        cache_file = root_dir / "cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.touch()
        mock_get_cache_path.return_value = cache_file
        
        # Call function
        result = load_cached_best_model(
            root_dir=root_dir,
            config_dir=config_dir,
            experiment_name="test_experiment",
            selection_config=selection_config,
            tags_config=mock_tags_config,
            benchmark_experiment_id="benchmark_exp_id_123",
        )
        
        # Should return None immediately without checking cache
        assert result is None
        # load_json should not be called when mode is force_new
        assert not mock_load_json.called

