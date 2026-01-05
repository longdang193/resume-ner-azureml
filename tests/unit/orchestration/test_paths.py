"""Unit tests for path resolution module."""

import json
import tempfile
from pathlib import Path
import pytest

from orchestration.paths import (
    load_paths_config,
    resolve_output_path,
    get_cache_file_path,
    get_timestamped_cache_filename,
    get_cache_strategy_config,
    save_cache_with_dual_strategy,
    load_cache_file,
)


class TestLoadPathsConfig:
    """Test paths configuration loading."""
    
    def test_load_paths_config_with_file(self, tmp_path):
        """Test loading paths config from existing file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
cache:
  best_configurations: "best_configurations"
""")
        
        config = load_paths_config(config_dir)
        
        assert config["base"]["outputs"] == "outputs"
        assert config["outputs"]["hpo"] == "hpo"
    
    def test_load_paths_config_without_file(self, tmp_path):
        """Test loading paths config with defaults when file doesn't exist."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config = load_paths_config(config_dir)
        
        # Should return defaults
        assert "base" in config
        assert "outputs" in config
        assert config["base"]["outputs"] == "outputs"


class TestResolveOutputPath:
    """Test output path resolution."""
    
    def test_resolve_simple_path(self, tmp_path):
        """Test resolving simple output path."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        path = resolve_output_path(tmp_path, config_dir, "hpo")
        
        assert path == tmp_path / "outputs" / "hpo"
    
    def test_resolve_cache_subdirectory(self, tmp_path):
        """Test resolving cache subdirectory."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        path = resolve_output_path(
            tmp_path, config_dir, "cache", subcategory="best_configurations"
        )
        
        assert path == tmp_path / "outputs" / "cache" / "best_configurations"
    
    def test_resolve_path_with_pattern(self, tmp_path):
        """Test resolving path with pattern replacement."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create minimal config with pattern
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  final_training: "final_training"
patterns:
  final_training: "{backbone}_{run_id}"
""")
        
        path = resolve_output_path(
            tmp_path, config_dir, "final_training",
            backbone="distilbert", run_id="20251227_220407"
        )
        
        assert path == tmp_path / "outputs" / "final_training" / "distilbert_20251227_220407"


class TestGetCacheFilePath:
    """Test cache file path resolution."""
    
    def test_get_latest_cache_file(self, tmp_path):
        """Test getting latest cache file path."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        path = get_cache_file_path(
            tmp_path, config_dir, "best_configurations", file_type="latest"
        )
        
        assert path.name == "latest_best_configuration.json"
        assert "best_configurations" in str(path)
    
    def test_get_index_cache_file(self, tmp_path):
        """Test getting index cache file path."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        path = get_cache_file_path(
            tmp_path, config_dir, "best_configurations", file_type="index"
        )
        
        assert path.name == "index.json"


class TestGetTimestampedCacheFilename:
    """Test timestamped cache filename generation."""
    
    def test_generate_best_config_filename(self, tmp_path):
        """Test generating best config timestamped filename."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        filename = get_timestamped_cache_filename(
            config_dir,
            "best_configurations",
            "distilbert-base-uncased",
            "trial_2",
            "20251227_220407"
        )
        
        assert filename.startswith("best_config_")
        assert "distilbert" in filename
        assert "trial_2" in filename
        assert "20251227_220407" in filename
        assert filename.endswith(".json")
    
    def test_generate_final_training_filename(self, tmp_path):
        """Test generating final training timestamped filename."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        filename = get_timestamped_cache_filename(
            config_dir,
            "final_training",
            "distilbert",
            "20251227_220407",
            "20251227_220500"
        )
        
        assert filename.startswith("final_training_")
        assert "distilbert" in filename
        assert "20251227_220407" in filename or "20251227_220500" in filename
        assert filename.endswith(".json")


class TestGetCacheStrategyConfig:
    """Test cache strategy config loading."""
    
    def test_get_strategy_config(self, tmp_path):
        """Test getting cache strategy configuration."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        strategy = get_cache_strategy_config(config_dir, "best_configurations")
        
        assert "strategy" in strategy
        assert "timestamped" in strategy
        assert "latest" in strategy
        assert "index" in strategy


class TestSaveCacheWithDualStrategy:
    """Test dual file strategy saving."""
    
    def test_save_cache_creates_all_files(self, tmp_path):
        """Test that saving cache creates timestamped, latest, and index files."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        data = {
            "backbone": "distilbert",
            "trial_name": "trial_2",
            "selection_criteria": {"best_value": 0.5},
        }
        
        timestamped_file, latest_file, index_file = save_cache_with_dual_strategy(
            root_dir=tmp_path,
            config_dir=config_dir,
            cache_type="best_configurations",
            data=data,
            backbone="distilbert",
            identifier="trial_2",
            timestamp="20251227_220407",
        )
        
        # Check all files exist
        assert timestamped_file.exists()
        assert latest_file.exists()
        assert index_file.exists()
        
        # Check timestamped file content
        timestamped_data = json.loads(timestamped_file.read_text())
        assert timestamped_data["backbone"] == "distilbert"
        assert "cache_metadata" in timestamped_data
        
        # Check latest file content
        latest_data = json.loads(latest_file.read_text())
        assert latest_data["backbone"] == "distilbert"
        assert latest_data["cache_metadata"]["timestamped_file"] == timestamped_file.name
        
        # Check index file content
        index_data = json.loads(index_file.read_text())
        assert "entries" in index_data
        assert len(index_data["entries"]) == 1
        assert index_data["entries"][0]["timestamp"] == "20251227_220407"


class TestLoadCacheFile:
    """Test cache file loading."""
    
    def test_load_latest_cache(self, tmp_path):
        """Test loading latest cache file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create cache directory and files
        cache_dir = tmp_path / "outputs" / "cache" / "best_configurations"
        cache_dir.mkdir(parents=True)
        
        latest_file = cache_dir / "latest_best_configuration.json"
        latest_file.write_text(json.dumps({"backbone": "distilbert", "trial": "trial_2"}))
        
        data = load_cache_file(
            tmp_path, config_dir, "best_configurations", use_latest=True
        )
        
        assert data is not None
        assert data["backbone"] == "distilbert"
    
    def test_load_specific_timestamp(self, tmp_path):
        """Test loading cache by specific timestamp."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        cache_dir = tmp_path / "outputs" / "cache" / "best_configurations"
        cache_dir.mkdir(parents=True)
        
        timestamped_file = cache_dir / "best_config_distilbert_trial_2_20251227_220407.json"
        timestamped_file.write_text(json.dumps({"backbone": "distilbert", "timestamp": "20251227_220407"}))
        
        data = load_cache_file(
            tmp_path,
            config_dir,
            "best_configurations",
            use_latest=False,
            specific_timestamp="20251227_220407"
        )
        
        assert data is not None
        assert data["timestamp"] == "20251227_220407"
    
    def test_load_returns_none_when_not_found(self, tmp_path):
        """Test loading returns None when cache not found."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        data = load_cache_file(
            tmp_path, config_dir, "best_configurations", use_latest=True
        )
        
        assert data is None

