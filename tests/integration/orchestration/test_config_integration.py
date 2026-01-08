"""Integration tests for configuration consistency."""

import yaml
from pathlib import Path
import pytest

from orchestration.naming_centralized import NamingContext, build_output_path
from orchestration.jobs.tracking.naming.run_names import build_mlflow_run_name
from orchestration.jobs.tracking.naming.tags import build_mlflow_tags
from orchestration.paths import resolve_output_path


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory with all config files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create paths.yaml
    paths_yaml = config_dir / "paths.yaml"
    paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
  benchmarking: "benchmarking"
  final_training: "final_training"
  conversion: "conversion"
patterns:
  hpo_v2: "{storage_env}/{model}/study-{study8}/trial-{trial8}"
  benchmarking_v2: "{storage_env}/{model}/study-{study8}/trial-{trial8}/bench-{bench8}"
  final_training_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}"
  conversion_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}/conv-{conv8}"
""")
    
    # Create naming.yaml
    naming_yaml = config_dir / "naming.yaml"
    naming_yaml.write_text("""
schema_version: 1
separators:
  field: "_"
  component: "-"
  version: "_"
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}{version}"
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
  final_training:
    pattern: "{env}_{model}_final_training_spec-{spec_hash}_exec-{exec_hash}_v{variant}{version}"
    components:
      spec_hash:
        length: 8
        source: "spec_fp"
        default: "unknown"
      exec_hash:
        length: 8
        source: "exec_fp"
        default: "unknown"
      variant:
        format: "{number}"
        source: "variant"
        default: "1"
version:
  format: "{separator}{number}"
  separator: "_"
""")
    
    # Create mlflow.yaml
    mlflow_yaml = config_dir / "mlflow.yaml"
    mlflow_yaml.write_text("""
naming:
  project_name: "resume-ner"
  tags:
    max_length: 250
    sanitize: true
  run_name:
    max_length: 100
    shorten_fingerprints: true
    auto_increment:
      enabled: true
      processes:
        hpo: true
        benchmarking: true
      format: "{base}.{version}"
""")
    
    # Create tags.yaml
    tags_yaml = config_dir / "tags.yaml"
    tags_yaml.write_text("""
schema_version: 1
grouping:
  study_key_hash: "code.study_key_hash"
  trial_key_hash: "code.trial_key_hash"
process:
  stage: "code.stage"
  model: "code.model"
  project: "code.project"
""")
    
    return config_dir


@pytest.fixture
def root_dir(tmp_path):
    """Create a temporary root directory."""
    return tmp_path


class TestEndToEndScenarios:
    """Test complete workflows (7.1)."""

    def test_hpo_trial_workflow(self, root_dir, config_dir):
        """Test HPO trial: context → path → name → tags."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890",
            trial_number=5,
            stage="hpo_trial"
        )
        
        # Build path
        path = build_output_path(root_dir, context, config_dir=config_dir)
        assert "hpo" in str(path)
        assert "study-350a79aa" in str(path)
        assert "trial-747428f2" in str(path)
        
        # Build run name
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        assert "local" in run_name
        assert "distilbert" in run_name
        assert "hpo_trial" in run_name
        
        # Build tags
        tags = build_mlflow_tags(
            context=context,
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890",
            config_dir=config_dir
        )
        assert tags["code.stage"] == "hpo"
        assert tags["code.model"] == "distilbert"
        assert tags["code.study_key_hash"] == "350a79aa1234567890abcdef"
    
    def test_final_training_workflow(self, root_dir, config_dir):
        """Test final training: context → path → name → tags."""
        context = NamingContext(
            process_type="final_training",
            model="distilbert",
            environment="local",
            storage_env="local",
            spec_fp="abc123def4567890",
            exec_fp="xyz789abc1234567",
            variant=1
        )
        
        # Build path
        path = build_output_path(root_dir, context, config_dir=config_dir)
        assert "final_training" in str(path)
        assert "spec" in str(path)
        assert "exec" in str(path)
        assert "v1" in str(path) or "/1" in str(path)
        
        # Build run name
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        assert "final_training" in run_name
        assert "distilbert" in run_name
        
        # Build tags
        tags = build_mlflow_tags(context=context, config_dir=config_dir)
        assert tags["code.stage"] == "final_training"
        assert tags["code.spec_fp"] == "abc123def4567890"
        assert tags["code.exec_fp"] == "xyz789abc1234567"
        assert tags["code.variant"] == "1"
    
    def test_benchmarking_workflow(self, root_dir, config_dir):
        """Test benchmarking: context → path → name → tags."""
        context = NamingContext(
            process_type="benchmarking",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890",
            benchmark_config_hash="abc12345abcdef1234567890"
        )
        
        # Build path
        path = build_output_path(root_dir, context, config_dir=config_dir)
        assert "benchmarking" in str(path)
        assert "study-350a79aa" in str(path)
        assert "trial-747428f2" in str(path)
        
        # Build run name
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        assert "benchmark" in run_name
        
        # Build tags
        tags = build_mlflow_tags(
            context=context,
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890",
            config_dir=config_dir
        )
        assert tags["code.stage"] == "benchmarking"
        assert tags["code.benchmark_config_hash"] == "abc12345abcdef1234567890"


class TestConfigurationConsistency:
    """Test config file interactions (7.2)."""

    def test_paths_match_naming_patterns(self, root_dir, config_dir):
        """Test paths from paths.yaml match patterns in naming.yaml."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890"
        )
        
        # Path should use v2 pattern from paths.yaml
        path = build_output_path(root_dir, context, config_dir=config_dir)
        assert "study-350a79aa" in str(path)
        assert "trial-747428f2" in str(path)
        
        # Run name should use pattern from naming.yaml
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        assert "study-350a79aa" in run_name
    
    def test_tag_keys_from_tags_yaml(self, config_dir):
        """Test tag keys from tags.yaml used in build_mlflow_tags()."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local"
        )
        
        tags = build_mlflow_tags(
            context=context,
            study_key_hash="350a79aa1234567890abcdef",
            config_dir=config_dir
        )
        
        # Should use tag keys from tags.yaml
        assert "code.stage" in tags
        assert "code.model" in tags
        assert "code.study_key_hash" in tags
    
    def test_naming_patterns_from_naming_yaml(self, root_dir, config_dir):
        """Test naming patterns from naming.yaml used in build_mlflow_run_name()."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_number=5,
            stage="hpo_trial"
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        
        # Should use pattern from naming.yaml
        assert "local" in run_name
        assert "distilbert" in run_name
        assert "hpo_trial" in run_name
        assert "study-350a79aa" in run_name


class TestCrossProcessConsistency:
    """Test consistency across processes (7.3)."""

    def test_same_model_environment_consistent_paths(self, root_dir, config_dir):
        """Test same model/environment produces consistent paths."""
        model = "distilbert"
        env = "local"
        
        # HPO context
        hpo_context = NamingContext(
            process_type="hpo",
            model=model,
            environment=env,
            storage_env=env,
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890"
        )
        hpo_path = build_output_path(root_dir, hpo_context, config_dir=config_dir)
        
        # Benchmarking context
        bench_context = NamingContext(
            process_type="benchmarking",
            model=model,
            environment=env,
            storage_env=env,
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890"
        )
        bench_path = build_output_path(root_dir, bench_context, config_dir=config_dir)
        
        # Both should have same model and environment
        assert model in str(hpo_path)
        assert env in str(hpo_path)
        assert model in str(bench_path)
        assert env in str(bench_path)
    
    def test_tag_keys_consistent_across_processes(self, config_dir):
        """Test tag keys consistent across all processes."""
        processes = ["hpo", "benchmarking", "final_training", "conversion"]
        
        for process_type in processes:
            if process_type == "final_training":
                context = NamingContext(
                    process_type=process_type,
                    model="distilbert",
                    environment="local",
                    spec_fp="abc123",
                    exec_fp="xyz789"
                )
            elif process_type == "conversion":
                context = NamingContext(
                    process_type=process_type,
                    model="distilbert",
                    environment="local",
                    parent_training_id="spec-abc_exec-xyz/v1",
                    conv_fp="conv123"
                )
            else:
                context = NamingContext(
                    process_type=process_type,
                    model="distilbert",
                    environment="local"
                )
            
            tags = build_mlflow_tags(context=context, config_dir=config_dir)
            
            # All should have consistent tag keys
            assert "code.stage" in tags
            assert "code.model" in tags
            assert "code.env" in tags
    
    def test_naming_conventions_consistent(self, root_dir, config_dir):
        """Test naming conventions consistent (separators, normalization)."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_number=5,
            stage="hpo_trial"
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        
        # Should use consistent separators (field: "_", component: "-")
        # Pattern: {env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}
        parts = run_name.split("_")
        assert len(parts) >= 3  # Should have multiple parts separated by "_"
        
        # Should have component separator "-" in study hash part
        assert "study-350a79aa" in run_name or "study_350a79aa" in run_name

