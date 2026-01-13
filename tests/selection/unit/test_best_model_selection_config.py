"""Unit tests for best_model_selection.yaml config loading."""

import pytest
from pathlib import Path
from common.shared.yaml_utils import load_yaml


class TestBestModelSelectionConfigLoading:
    """Test loading best_model_selection.yaml configuration."""

    def test_load_best_model_selection_config(self, tmp_path):
        """Test loading best_model_selection.yaml from config directory."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "best_model_selection.yaml"
        config_file.write_text("""
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
        
        config = load_yaml(config_file)
        
        # Verify all sections are present
        assert "run" in config
        assert "objective" in config
        assert "scoring" in config
        assert "benchmark" in config
        
        # Verify run section
        assert config["run"]["mode"] == "force_new"
        
        # Verify objective section
        assert config["objective"]["metric"] == "macro-f1"
        assert config["objective"]["goal"] == "maximize"
        
        # Verify scoring section
        assert config["scoring"]["f1_weight"] == 0.7
        assert config["scoring"]["latency_weight"] == 0.3
        assert config["scoring"]["normalize_weights"] is True
        
        # Verify benchmark section
        assert config["benchmark"]["required_metrics"] == ["latency_batch_1_ms"]

    def test_load_config_with_custom_values(self, tmp_path):
        """Test loading config with custom non-default values."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "best_model_selection.yaml"
        config_file.write_text("""
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
        
        config = load_yaml(config_file)
        
        # Verify custom values
        assert config["run"]["mode"] == "reuse_if_exists"
        assert config["objective"]["metric"] == "accuracy"
        assert config["objective"]["goal"] == "minimize"
        assert config["scoring"]["f1_weight"] == 0.5
        assert config["scoring"]["latency_weight"] == 0.5
        assert config["scoring"]["normalize_weights"] is False
        assert len(config["benchmark"]["required_metrics"]) == 2

    def test_load_config_structure_validation(self, tmp_path):
        """Test that loaded config has correct structure."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "best_model_selection.yaml"
        config_file.write_text("""
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
        
        config = load_yaml(config_file)
        
        # Verify structure types
        assert isinstance(config, dict)
        assert isinstance(config["run"], dict)
        assert isinstance(config["objective"], dict)
        assert isinstance(config["scoring"], dict)
        assert isinstance(config["benchmark"], dict)
        
        # Verify run section structure
        assert "mode" in config["run"]
        assert isinstance(config["run"]["mode"], str)
        
        # Verify objective section structure
        assert "metric" in config["objective"]
        assert "goal" in config["objective"]
        assert isinstance(config["objective"]["metric"], str)
        assert isinstance(config["objective"]["goal"], str)
        
        # Verify scoring section structure
        assert "f1_weight" in config["scoring"]
        assert "latency_weight" in config["scoring"]
        assert "normalize_weights" in config["scoring"]
        assert isinstance(config["scoring"]["f1_weight"], (int, float))
        assert isinstance(config["scoring"]["latency_weight"], (int, float))
        assert isinstance(config["scoring"]["normalize_weights"], bool)
        
        # Verify benchmark section structure
        assert "required_metrics" in config["benchmark"]
        assert isinstance(config["benchmark"]["required_metrics"], list)

    def test_load_config_matches_actual_file(self, tmp_path):
        """Test that loaded config matches the structure of the actual config file."""
        # Load actual config file
        actual_config_path = Path(__file__).parent.parent.parent / "config" / "best_model_selection.yaml"
        
        if actual_config_path.exists():
            actual_config = load_yaml(actual_config_path)
            
            # Verify all expected sections exist
            assert "run" in actual_config
            assert "objective" in actual_config
            assert "scoring" in actual_config
            assert "benchmark" in actual_config
            
            # Verify run section
            assert "mode" in actual_config["run"]
            assert actual_config["run"]["mode"] in ["reuse_if_exists", "force_new"]
            
            # Verify objective section
            assert "metric" in actual_config["objective"]
            assert "goal" in actual_config["objective"]
            assert actual_config["objective"]["goal"] in ["maximize", "minimize"]
            
            # Verify scoring section
            assert "f1_weight" in actual_config["scoring"]
            assert "latency_weight" in actual_config["scoring"]
            assert "normalize_weights" in actual_config["scoring"]
            
            # Verify benchmark section
            assert "required_metrics" in actual_config["benchmark"]

            # Verify champion_selection section (Phase 2) if present
            if "champion_selection" in actual_config:
                champion_config = actual_config["champion_selection"]
                assert "min_trials_per_group" in champion_config
                assert "top_k_for_stable_score" in champion_config
                assert "require_artifact_available" in champion_config
                assert "artifact_check_source" in champion_config
                assert "prefer_schema_version" in champion_config
                assert "allow_mixed_schema_groups" in champion_config
                
                # Verify types
                assert isinstance(champion_config["min_trials_per_group"], int)
                assert isinstance(champion_config["top_k_for_stable_score"], int)
                assert isinstance(champion_config["require_artifact_available"], bool)
                assert isinstance(champion_config["artifact_check_source"], str)
                assert isinstance(champion_config["prefer_schema_version"], str)
                assert isinstance(champion_config["allow_mixed_schema_groups"], bool)

    def test_load_config_with_champion_selection_section(self, tmp_path):
        """Test loading config with champion_selection section (Phase 2)."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "best_model_selection.yaml"
        config_file.write_text("""
run:
  mode: force_new
objective:
  metric: "macro-f1"
  direction: "maximize"
champion_selection:
  min_trials_per_group: 3
  top_k_for_stable_score: 3
  require_artifact_available: true
  artifact_check_source: "tag"
  prefer_schema_version: "auto"
  allow_mixed_schema_groups: false
scoring:
  f1_weight: 0.7
  latency_weight: 0.3
  normalize_weights: true
benchmark:
  required_metrics:
    - "latency_batch_1_ms"
""")
        
        config = load_yaml(config_file)
        
        # Verify champion_selection section is present
        assert "champion_selection" in config
        
        champion_config = config["champion_selection"]
        
        # Verify all champion_selection fields
        assert champion_config["min_trials_per_group"] == 3
        assert champion_config["top_k_for_stable_score"] == 3
        assert champion_config["require_artifact_available"] is True
        assert champion_config["artifact_check_source"] == "tag"
        assert champion_config["prefer_schema_version"] == "auto"
        assert champion_config["allow_mixed_schema_groups"] is False

    def test_load_config_champion_selection_custom_values(self, tmp_path):
        """Test loading config with custom champion_selection values."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "best_model_selection.yaml"
        config_file.write_text("""
run:
  mode: force_new
objective:
  metric: "macro-f1"
  direction: "maximize"
champion_selection:
  min_trials_per_group: 5
  top_k_for_stable_score: 2
  require_artifact_available: false
  artifact_check_source: "disk"
  prefer_schema_version: "2.0"
  allow_mixed_schema_groups: true
scoring:
  f1_weight: 0.7
  latency_weight: 0.3
benchmark:
  required_metrics:
    - "latency_batch_1_ms"
""")
        
        config = load_yaml(config_file)
        
        champion_config = config["champion_selection"]
        
        # Verify custom values
        assert champion_config["min_trials_per_group"] == 5
        assert champion_config["top_k_for_stable_score"] == 2
        assert champion_config["require_artifact_available"] is False
        assert champion_config["artifact_check_source"] == "disk"
        assert champion_config["prefer_schema_version"] == "2.0"
        assert champion_config["allow_mixed_schema_groups"] is True

    def test_load_config_champion_selection_structure_validation(self, tmp_path):
        """Test that champion_selection section has correct structure."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "best_model_selection.yaml"
        config_file.write_text("""
run:
  mode: force_new
objective:
  metric: "macro-f1"
  direction: "maximize"
champion_selection:
  min_trials_per_group: 3
  top_k_for_stable_score: 3
  require_artifact_available: true
  artifact_check_source: "tag"
  prefer_schema_version: "auto"
  allow_mixed_schema_groups: false
scoring:
  f1_weight: 0.7
benchmark:
  required_metrics: []
""")
        
        config = load_yaml(config_file)
        
        # Verify champion_selection structure types
        assert isinstance(config["champion_selection"], dict)
        champion_config = config["champion_selection"]
        
        assert isinstance(champion_config["min_trials_per_group"], int)
        assert isinstance(champion_config["top_k_for_stable_score"], int)
        assert isinstance(champion_config["require_artifact_available"], bool)
        assert isinstance(champion_config["artifact_check_source"], str)
        assert isinstance(champion_config["prefer_schema_version"], str)
        assert isinstance(champion_config["allow_mixed_schema_groups"], bool)
        
        # Verify valid values
        assert champion_config["min_trials_per_group"] > 0
        assert champion_config["top_k_for_stable_score"] > 0
        assert champion_config["artifact_check_source"] in ["tag", "disk"]
        assert champion_config["prefer_schema_version"] in ["1.0", "2.0", "auto"]

    def test_load_config_champion_selection_with_objective_direction(self, tmp_path):
        """Test that objective.direction (new) and objective.goal (legacy) both work."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Test with new 'direction' key
        config_file = config_dir / "best_model_selection.yaml"
        config_file.write_text("""
run:
  mode: force_new
objective:
  metric: "macro-f1"
  direction: "maximize"
champion_selection:
  min_trials_per_group: 3
  top_k_for_stable_score: 3
scoring:
  f1_weight: 0.7
benchmark:
  required_metrics: []
""")
        
        config = load_yaml(config_file)
        assert config["objective"]["direction"] == "maximize"
        
        # Test with legacy 'goal' key
        config_file.write_text("""
run:
  mode: force_new
objective:
  metric: "macro-f1"
  goal: "minimize"
champion_selection:
  min_trials_per_group: 3
  top_k_for_stable_score: 3
scoring:
  f1_weight: 0.7
benchmark:
  required_metrics: []
""")
        
        config = load_yaml(config_file)
        assert config["objective"]["goal"] == "minimize"

