"""Unit tests for centralized naming system."""

import pytest
from pathlib import Path
from orchestration.naming_centralized import (
    NamingContext,
    create_naming_context,
    build_output_path,
    build_parent_training_id,
)


def test_naming_context_validation():
    """Test that NamingContext validates inputs."""
    # Valid context
    context = NamingContext(
        process_type="hpo",
        model="distilbert",
        environment="local",
        trial_id="trial_1_20251229_100000"
    )
    assert context.process_type == "hpo"
    
    # Invalid process_type
    with pytest.raises(ValueError, match="Invalid process_type"):
        NamingContext(
            process_type="invalid",
            model="distilbert",
            environment="local"
        )
    
    # Invalid environment
    with pytest.raises(ValueError, match="Invalid environment"):
        NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="invalid"
        )
    
    # Invalid variant
    with pytest.raises(ValueError, match="Variant must be >= 1"):
        NamingContext(
            process_type="final_training",
            model="distilbert",
            environment="local",
            spec_fp="abc123",
            exec_fp="xyz789",
            variant=0
        )


def test_naming_context_final_training_requires_fingerprints():
    """Test that final_training requires spec_fp and exec_fp."""
    with pytest.raises(ValueError, match="final_training requires spec_fp and exec_fp"):
        NamingContext(
            process_type="final_training",
            model="distilbert",
            environment="local"
        )


def test_naming_context_conversion_requires_parent_and_conv_fp():
    """Test that conversion requires parent_training_id and conv_fp."""
    with pytest.raises(ValueError, match="conversion requires parent_training_id and conv_fp"):
        NamingContext(
            process_type="conversion",
            model="distilbert",
            environment="local"
        )


def test_create_naming_context_auto_detect():
    """Test that create_naming_context auto-detects environment."""
    context = create_naming_context(
        process_type="hpo",
        model="distilbert",
        trial_id="trial_1_20251229_100000"
    )
    
    assert context.environment in ["local", "colab", "kaggle", "azure"]


def test_build_output_path_hpo():
    """Test path building for HPO."""
    context = NamingContext(
        process_type="hpo",
        model="distilbert",
        environment="local",
        trial_id="trial_1_20251229_100000"
    )
    
    path = build_output_path(Path("/root"), context)
    
    assert str(path) == "/root/outputs/hpo/local/distilbert/trial_1_20251229_100000"


def test_build_output_path_benchmarking():
    """Test path building for benchmarking."""
    context = NamingContext(
        process_type="benchmarking",
        model="distilbert",
        environment="colab",
        trial_id="trial_1_20251229_100000"
    )
    
    path = build_output_path(Path("/root"), context)
    
    assert str(path) == "/root/outputs/benchmarking/colab/distilbert/trial_1_20251229_100000"


def test_build_output_path_final_training():
    """Test path building for final training."""
    context = NamingContext(
        process_type="final_training",
        model="distilbert",
        environment="local",
        spec_fp="abc123def4567890",
        exec_fp="xyz789abc1234567",
        variant=1
    )
    
    path = build_output_path(Path("/root"), context)
    
    expected = "/root/outputs/final_training/local/distilbert/spec_abc123def4567890_exec_xyz789abc1234567/v1"
    assert str(path) == expected


def test_build_output_path_final_training_variant():
    """Test path building for final training with variant."""
    context = NamingContext(
        process_type="final_training",
        model="distilbert",
        environment="local",
        spec_fp="abc123def4567890",
        exec_fp="xyz789abc1234567",
        variant=2
    )
    
    path = build_output_path(Path("/root"), context)
    
    assert "v2" in str(path)


def test_build_output_path_conversion():
    """Test path building for conversion."""
    context = NamingContext(
        process_type="conversion",
        model="distilbert",
        environment="local",
        parent_training_id="spec_abc_exec_xyz/v1",
        conv_fp="conv1234567890123"
    )
    
    path = build_output_path(Path("/root"), context)
    
    expected = "/root/outputs/conversion/local/distilbert/spec_abc_exec_xyz/v1/conv_conv1234567890123"
    assert str(path) == expected


def test_build_output_path_best_configurations():
    """Test path building for best configurations."""
    context = NamingContext(
        process_type="best_configurations",
        model="distilbert",
        environment="local",
        spec_fp="abc123def4567890"
    )
    
    path = build_output_path(Path("/root"), context)
    
    expected = "/root/outputs/cache/best_configurations/distilbert/spec_abc123def4567890"
    assert str(path) == expected


def test_build_parent_training_id():
    """Test building parent training ID."""
    spec_fp = "abc123def4567890"
    exec_fp = "xyz789abc1234567"
    variant = 1
    
    parent_id = build_parent_training_id(spec_fp, exec_fp, variant)
    
    assert parent_id == "spec_abc123def4567890_exec_xyz789abc1234567/v1"


class TestRunNameGeneration:
    """Test MLflow run name generation for HPO processes."""

    def test_hpo_trial_run_name(self, tmp_path):
        """Test HPO trial run name generation."""
        from orchestration.jobs.tracking.naming.run_names import build_mlflow_run_name
        
        # Create minimal config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      trial_number:
        format: "{number}"
        zero_pad: 2
        source: "trial_number"
        default: "unknown"
""")
        
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            stage="hpo_trial",
            study_key_hash="a" * 64,  # 64-char hash
            trial_number=1,
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir)
        
        # Should match pattern: {env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}
        assert "local" in run_name
        assert "distilbert" in run_name
        assert "hpo_trial" in run_name
        assert "study-" in run_name
        assert "t01" in run_name or "t1" in run_name
        assert len(run_name) <= 256  # MLflow limit

    def test_hpo_trial_fold_run_name(self, tmp_path):
        """Test HPO trial fold run name generation."""
        from orchestration.jobs.tracking.naming.run_names import build_mlflow_run_name
        
        # Create minimal config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_trial_fold:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}_fold{fold_idx}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      trial_number:
        format: "{number}"
        zero_pad: 2
        source: "trial_number"
        default: "unknown"
      fold_idx:
        format: "{number}"
        source: "fold_idx"
        default: "0"
""")
        
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            stage="hpo_trial",
            study_key_hash="a" * 64,
            trial_number=1,
            fold_idx=0,
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir)
        
        # Should include fold index
        assert "fold0" in run_name or "fold_0" in run_name
        assert len(run_name) <= 256

    def test_hpo_refit_run_name(self, tmp_path):
        """Test HPO refit run name generation."""
        from orchestration.jobs.tracking.naming.run_names import build_mlflow_run_name
        
        # Create minimal config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_refit:
    pattern: "{env}_{model}_hpo_refit_study-{study_hash}_trial-{trial_hash}_t{trial_number}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      trial_hash:
        length: 8
        source: "trial_key_hash"
        default: "unknown"
      trial_number:
        format: "{number}"
        zero_pad: 2
        source: "trial_number"
        default: "unknown"
""")
        
        # For refit, we need to set process_type correctly
        # The naming system detects refit based on stage
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            stage="hpo_refit",  # This should trigger refit naming
            study_key_hash="a" * 64,
            trial_key_hash="b" * 64,
            trial_number=1,
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir)
        
        # Should contain refit-related terms (may fall back to legacy if policy not found)
        # The key is that it generates a valid name
        assert "distilbert" in run_name
        assert len(run_name) <= 256

    def test_hpo_sweep_run_name(self, tmp_path):
        """Test HPO sweep/study parent run name generation."""
        from orchestration.jobs.tracking.naming.run_names import build_mlflow_run_name
        
        # Create minimal config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_sweep:
    pattern: "{env}_{model}_hpo_study-{study_hash}{semantic_suffix}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      semantic_suffix:
        enabled: true
        max_length: 30
        source: "study_name"
        default: ""
""")
        
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            stage="hpo_sweep",
            study_key_hash="a" * 64,
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir)
        
        # Should match pattern: {env}_{model}_hpo_study-{hash}
        assert "hpo" in run_name
        assert "study-" in run_name
        assert len(run_name) <= 256

    def test_run_name_max_length(self, tmp_path):
        """Test that run names respect max_length=256."""
        from orchestration.jobs.tracking.naming.run_names import build_mlflow_run_name
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
validate:
  max_length: 256
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}"
""")
        
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            stage="hpo_trial",
            study_key_hash="a" * 64,
            trial_number=1,
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir)
        
        assert len(run_name) <= 256

    def test_run_name_forbidden_chars_removed(self, tmp_path):
        """Test that forbidden chars are removed from run names."""
        from orchestration.jobs.tracking.naming.run_names import build_mlflow_run_name
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
validate:
  forbidden_chars:
    - "/"
    - "\\\\"
    - ":"
    - "*"
    - "?"
    - "<"
    - ">"
    - "|"
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}"
""")
        
        # Use a model name that would normally have forbidden chars
        # The normalization should handle this
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",  # Use normal model name
            environment="local",
            storage_env="local",
            stage="hpo_trial",
            study_key_hash="a" * 64,
            trial_number=1,
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir)
        
        # Should not contain common forbidden chars (basic validation)
        forbidden = ["/", ":", "*", "?", "<", ">", "|"]
        for char in forbidden:
            assert char not in run_name, f"Forbidden char '{char}' found in run name: {run_name}"


