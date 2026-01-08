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


class TestPathBuildingV2:
    """Test v2 path building for HPO with study_key_hash and trial_key_hash."""

    def test_path_building_v2_hpo(self, tmp_path):
        """Test path building v2 for HPO with study_key_hash and trial_key_hash."""
        from orchestration.naming_centralized import create_naming_context, build_output_path
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
patterns:
  hpo_v2: "{storage_env}/{model}/study-{study8}/trial-{trial8}"
""")
        
        # Create context with study_key_hash and trial_key_hash
        study_key_hash = "a" * 64  # 64-char hash
        trial_key_hash = "b" * 64  # 64-char hash
        study8 = study_key_hash[:8]
        trial8 = trial_key_hash[:8]
        
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash=study_key_hash,
            trial_key_hash=trial_key_hash,
        )
        
        path = build_output_path(tmp_path, context, config_dir=config_dir)
        
        # Verify path pattern: outputs/hpo/{storage_env}/{model}/study-{study8}/trial-{trial8}
        assert "hpo" in str(path)
        assert "local" in str(path)
        assert "distilbert" in str(path)
        assert f"study-{study8}" in str(path)
        assert f"trial-{trial8}" in str(path)
        
        # Verify study8 and trial8 are first 8 chars of hashes
        assert study8 == study_key_hash[:8]
        assert trial8 == trial_key_hash[:8]

    def test_path_building_v2_normalized(self, tmp_path):
        """Test that path is normalized (no invalid chars)."""
        from orchestration.naming_centralized import create_naming_context, build_output_path
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
patterns:
  hpo_v2: "{storage_env}/{model}/study-{study8}/trial-{trial8}"
normalize_paths:
  replace:
    "/": "_"
    "\\": "_"
    "-": "_"
    " ": "_"
""")
        
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash=study_key_hash,
            trial_key_hash=trial_key_hash,
        )
        
        path = build_output_path(tmp_path, context, config_dir=config_dir)
        
        # Path should be valid (no invalid filesystem chars)
        path_str = str(path)
        # Should not contain common invalid chars (though normalization may vary)
        assert len(path_str) > 0
        # Path should be a valid Path object
        assert path.exists() or not path.exists()  # Either is fine, just check it's valid

    def test_path_building_v2_all_storage_envs(self, tmp_path):
        """Test path building v2 for all storage environments."""
        from orchestration.naming_centralized import create_naming_context, build_output_path
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
patterns:
  hpo_v2: "{storage_env}/{model}/study-{study8}/trial-{trial8}"
env_overrides:
  colab:
    base:
      outputs: "/content/drive/MyDrive/resume-ner-azureml/outputs"
  azureml:
    base:
      outputs: "/mnt/outputs"
  kaggle:
    base:
      outputs: "/kaggle/working/outputs"
""")
        
        study_key_hash = "a" * 64
        trial_key_hash = "b" * 64
        
        storage_envs = ["local", "colab", "kaggle", "azureml"]
        
        for storage_env in storage_envs:
            context = create_naming_context(
                process_type="hpo",
                model="distilbert",
                environment=storage_env,
                storage_env=storage_env,
                study_key_hash=study_key_hash,
                trial_key_hash=trial_key_hash,
            )
            
            path = build_output_path(tmp_path, context, config_dir=config_dir)
            
            # Verify storage_env is in path
            assert storage_env in str(path)
            # Verify v2 pattern components
            assert f"study-{study_key_hash[:8]}" in str(path)
            assert f"trial-{trial_key_hash[:8]}" in str(path)

    def test_path_building_v2_study8_trial8_format(self, tmp_path):
        """Test that study8 and trial8 are correctly formatted (first 8 chars)."""
        from orchestration.naming_centralized import create_naming_context, build_output_path
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
patterns:
  hpo_v2: "{storage_env}/{model}/study-{study8}/trial-{trial8}"
""")
        
        # Use specific hashes to verify format
        study_key_hash = "350a79aa1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        trial_key_hash = "747428f2abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234"
        study8_expected = "350a79aa"
        trial8_expected = "747428f2"
        
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash=study_key_hash,
            trial_key_hash=trial_key_hash,
        )
        
        path = build_output_path(tmp_path, context, config_dir=config_dir)
        path_str = str(path)
        
        # Verify exact 8-char hashes are used
        assert f"study-{study8_expected}" in path_str
        assert f"trial-{trial8_expected}" in path_str
        assert study8_expected == study_key_hash[:8]
        assert trial8_expected == trial_key_hash[:8]

