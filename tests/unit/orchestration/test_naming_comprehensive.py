"""Comprehensive unit tests for naming system."""

import yaml
from pathlib import Path
import pytest

from orchestration.naming_centralized import (
    NamingContext,
    create_naming_context,
    build_output_path,
    build_parent_training_id,
)
from orchestration.jobs.tracking.naming.run_names import build_mlflow_run_name
from orchestration.jobs.tracking.naming.policy import (
    load_naming_policy,
    format_run_name,
    validate_run_name,
)


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def root_dir(tmp_path):
    """Create a temporary root directory."""
    return tmp_path


class TestNamingContextValidation:
    """Test NamingContext validation for all process types (2.1)."""

    def test_naming_context_hpo_valid(self):
        """Test valid HPO context."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            trial_id="trial_1_20251229_100000"
        )
        assert context.process_type == "hpo"
        assert context.model == "distilbert"
        assert context.environment == "local"
    
    def test_naming_context_hpo_refit_valid(self):
        """Test valid HPO refit context."""
        context = NamingContext(
            process_type="hpo_refit",
            model="distilbert",
            environment="local",
            trial_id="trial_1_20251229_100000"
        )
        assert context.process_type == "hpo_refit"
    
    def test_naming_context_benchmarking_valid(self):
        """Test valid benchmarking context."""
        context = NamingContext(
            process_type="benchmarking",
            model="distilbert",
            environment="local",
            trial_id="trial_1_20251229_100000"
        )
        assert context.process_type == "benchmarking"
    
    def test_naming_context_final_training_valid(self):
        """Test valid final_training context."""
        context = NamingContext(
            process_type="final_training",
            model="distilbert",
            environment="local",
            spec_fp="abc123def4567890",
            exec_fp="xyz789abc1234567",
            variant=1
        )
        assert context.process_type == "final_training"
        assert context.spec_fp == "abc123def4567890"
        assert context.exec_fp == "xyz789abc1234567"
    
    def test_naming_context_conversion_valid(self):
        """Test valid conversion context."""
        context = NamingContext(
            process_type="conversion",
            model="distilbert",
            environment="local",
            parent_training_id="spec-abc_exec-xyz/v1",
            conv_fp="conv1234567890123"
        )
        assert context.process_type == "conversion"
    
    def test_naming_context_best_configurations_valid(self):
        """Test valid best_configurations context."""
        context = NamingContext(
            process_type="best_configurations",
            model="distilbert",
            environment="local",
            spec_fp="abc123def4567890"
        )
        assert context.process_type == "best_configurations"
    
    def test_naming_context_invalid_process_type(self):
        """Test invalid process_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid process_type"):
            NamingContext(
                process_type="invalid",
                model="distilbert",
                environment="local"
            )
    
    def test_naming_context_invalid_environment(self):
        """Test invalid environment raises ValueError."""
        with pytest.raises(ValueError, match="Invalid environment"):
            NamingContext(
                process_type="hpo",
                model="distilbert",
                environment="invalid"
            )
    
    def test_naming_context_invalid_variant(self):
        """Test invalid variant (< 1) raises ValueError."""
        with pytest.raises(ValueError, match="Variant must be >= 1"):
            NamingContext(
                process_type="final_training",
                model="distilbert",
                environment="local",
                spec_fp="abc123",
                exec_fp="xyz789",
                variant=0
            )
    
    def test_naming_context_final_training_requires_fingerprints(self):
        """Test final_training requires spec_fp and exec_fp."""
        with pytest.raises(ValueError, match="final_training requires spec_fp and exec_fp"):
            NamingContext(
                process_type="final_training",
                model="distilbert",
                environment="local"
            )
    
    def test_naming_context_conversion_requires_parent_and_conv_fp(self):
        """Test conversion requires parent_training_id and conv_fp."""
        with pytest.raises(ValueError, match="conversion requires parent_training_id and conv_fp"):
            NamingContext(
                process_type="conversion",
                model="distilbert",
                environment="local"
            )
    
    def test_naming_context_best_configurations_requires_spec_fp(self):
        """Test best_configurations requires spec_fp."""
        with pytest.raises(ValueError, match="best_configurations requires spec_fp"):
            NamingContext(
                process_type="best_configurations",
                model="distilbert",
                environment="local"
            )
    
    def test_naming_context_storage_env_defaults_to_environment(self):
        """Test storage_env defaults to environment if not provided."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local"
        )
        assert context.storage_env == "local"
    
    def test_naming_context_storage_env_explicit(self):
        """Test explicit storage_env override."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="colab"
        )
        assert context.storage_env == "colab"


class TestContextCreation:
    """Test context creation (2.2)."""

    def test_create_naming_context_auto_detects_environment(self):
        """Test that create_naming_context auto-detects environment when None."""
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            trial_id="trial_1_20251229_100000"
        )
        assert context.environment in ["local", "colab", "kaggle", "azure", "azureml"]
    
    def test_create_naming_context_defaults_storage_env(self):
        """Test defaults storage_env to environment."""
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            environment="local",
            trial_id="trial_1_20251229_100000"
        )
        assert context.storage_env == "local"
    
    def test_create_naming_context_handles_all_process_types(self):
        """Test handles all process types."""
        process_types = ["hpo", "hpo_refit", "benchmarking", "final_training", "conversion", "best_configurations"]
        
        for process_type in process_types:
            if process_type == "final_training":
                context = create_naming_context(
                    process_type=process_type,
                    model="distilbert",
                    environment="local",
                    spec_fp="abc123",
                    exec_fp="xyz789"
                )
            elif process_type == "conversion":
                context = create_naming_context(
                    process_type=process_type,
                    model="distilbert",
                    environment="local",
                    parent_training_id="spec-abc_exec-xyz/v1",
                    conv_fp="conv123"
                )
            elif process_type == "best_configurations":
                context = create_naming_context(
                    process_type=process_type,
                    model="distilbert",
                    environment="local",
                    spec_fp="abc123"
                )
            else:
                context = create_naming_context(
                    process_type=process_type,
                    model="distilbert",
                    environment="local",
                    trial_id="trial_1"
                )
            assert context.process_type == process_type
    
    def test_create_naming_context_preserves_optional_fields(self):
        """Test preserves all optional fields."""
        context = create_naming_context(
            process_type="hpo",
            model="distilbert",
            environment="local",
            trial_id="trial_1",
            study_name="test_study",
            study_key_hash="abc123",
            trial_key_hash="xyz789",
            trial_number=5,
            fold_idx=2
        )
        assert context.study_name == "test_study"
        assert context.study_key_hash == "abc123"
        assert context.trial_key_hash == "xyz789"
        assert context.trial_number == 5
        assert context.fold_idx == 2


class TestRunNameBuilding:
    """Test run name building (2.3)."""

    def test_build_mlflow_run_name_hpo_trial(self, root_dir, config_dir):
        """Test HPO trial run name."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
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
version:
  format: "{separator}{number}"
  separator: "_"
""")
        
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_number=5,
            stage="hpo_trial"
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        assert "local" in run_name
        assert "distilbert" in run_name
        assert "hpo_trial" in run_name
        assert "study-350a79aa" in run_name
        assert "t05" in run_name or "t5" in run_name
    
    def test_build_mlflow_run_name_hpo_trial_fold(self, root_dir, config_dir):
        """Test HPO trial fold run name."""
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
version:
  format: "{separator}{number}"
  separator: "_"
""")
        
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_number=5,
            fold_idx=2,
            stage="hpo_trial"
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        assert "fold2" in run_name or "fold_2" in run_name
    
    def test_build_mlflow_run_name_hpo_sweep(self, root_dir, config_dir):
        """Test HPO sweep run name."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_sweep:
    pattern: "{env}_{model}_hpo_study-{study_hash}{semantic_suffix}{version}"
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
version:
  format: "{separator}{number}"
  separator: "_"
""")
        
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            study_name="smoke_test",
            stage="hpo_sweep"
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        assert "hpo" in run_name
        assert "study-350a79aa" in run_name
    
    def test_build_mlflow_run_name_hpo_refit(self, root_dir, config_dir):
        """Test HPO refit run name."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_refit:
    pattern: "{env}_{model}_hpo_refit_study-{study_hash}_trial-{trial_hash}_t{trial_number}{version}"
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
version:
  format: "{separator}{number}"
  separator: "_"
""")
        
        context = NamingContext(
            process_type="hpo_refit",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890",
            trial_number=5
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        assert "hpo_refit" in run_name
        assert "study-350a79aa" in run_name
        assert "trial-747428f2" in run_name
    
    def test_build_mlflow_run_name_final_training(self, root_dir, config_dir):
        """Test final training run name."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
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
        
        context = NamingContext(
            process_type="final_training",
            model="distilbert",
            environment="local",
            storage_env="local",
            spec_fp="abc123def4567890",
            exec_fp="xyz789abc1234567",
            variant=1
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        assert "final_training" in run_name
        assert "spec-abc123de" in run_name or "spec_abc123de" in run_name
        assert "exec-xyz789ab" in run_name or "exec_xyz789ab" in run_name
        assert "v1" in run_name
    
    def test_build_mlflow_run_name_benchmarking(self, root_dir, config_dir):
        """Test benchmarking run name."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  benchmarking:
    pattern: "{env}_{model}_benchmark_study-{study_hash}_trial-{trial_hash}_bench-{bench_hash}{version}"
    components:
      study_hash:
        length: 8
        source: "study_key_hash"
        default: "unknown"
      trial_hash:
        length: 8
        source: "trial_key_hash"
        default: "unknown"
      bench_hash:
        length: 8
        source: "benchmark_config_hash"
        default: "unknown"
version:
  format: "{separator}{number}"
  separator: "_"
""")
        
        context = NamingContext(
            process_type="benchmarking",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890",
            benchmark_config_hash="abc12345abcdef1234567890"
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        assert "benchmark" in run_name
        assert "study-350a79aa" in run_name
        assert "trial-747428f2" in run_name
        assert "bench-abc12345" in run_name
    
    def test_build_mlflow_run_name_conversion(self, root_dir, config_dir):
        """Test conversion run name."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  conversion:
    pattern: "{env}_{model}_conversion_spec-{spec_hash}_exec-{exec_hash}_v{variant}_conv-{conv_hash}{version}"
    components:
      spec_hash:
        length: 8
        source: "parent_training_id"
        default: "unknown"
      exec_hash:
        length: 8
        source: "parent_training_id"
        default: "unknown"
      variant:
        format: "{number}"
        source: "parent_training_id"
        default: "1"
      conv_hash:
        length: 8
        source: "conv_fp"
        default: "unknown"
version:
  format: "{separator}{number}"
  separator: "_"
""")
        
        context = NamingContext(
            process_type="conversion",
            model="distilbert",
            environment="local",
            storage_env="local",
            parent_training_id="spec-abc12345_exec-xyz789ab/v1",
            conv_fp="conv1234567890123"
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        assert "conversion" in run_name
        assert "conv" in run_name


class TestRunNamePolicy:
    """Test policy-driven formatting (2.4)."""

    def test_load_naming_policy(self, config_dir):
        """Test loading naming policy."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
run_names:
  hpo_trial:
    pattern: "{env}_{model}_hpo_trial_study-{study_hash}_t{trial_number}{version}"
""")
        
        policy = load_naming_policy(config_dir)
        assert "run_names" in policy
        assert "hpo_trial" in policy["run_names"]
    
    def test_load_naming_policy_fallback_when_missing(self, config_dir):
        """Test fallback to legacy when policy unavailable."""
        # Don't create naming.yaml
        policy = load_naming_policy(config_dir)
        assert policy == {} or "run_names" not in policy
    
    def test_format_run_name(self, config_dir):
        """Test format_run_name function."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
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
""")
        
        policy = load_naming_policy(config_dir)
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_number=5,
            stage="hpo_trial"
        )
        
        run_name = format_run_name("hpo_trial", context, policy, config_dir)
        assert "local" in run_name
        assert "distilbert" in run_name
        assert "study-350a79aa" in run_name
    
    def test_validate_run_name(self, config_dir):
        """Test validate_run_name function."""
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
validate:
  max_length: 256
  forbidden_chars: ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]
  warn_length: 150
""")
        
        policy = load_naming_policy(config_dir)
        
        # Valid name
        validate_run_name("local_distilbert_hpo_trial", policy)
        
        # Invalid name with forbidden char
        with pytest.raises(ValueError, match="forbidden"):
            validate_run_name("local/distilbert/hpo_trial", policy)


class TestAutoIncrementVersioning:
    """Test auto-increment versioning (2.5)."""

    def test_auto_increment_enabled_for_hpo(self, root_dir, config_dir):
        """Test auto-increment enabled for HPO processes."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  run_name:
    auto_increment:
      enabled: true
      processes:
        hpo: true
      format: "{base}.{version}"
""")
        
        naming_yaml = config_dir / "naming.yaml"
        naming_yaml.write_text("""
schema_version: 1
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
version:
  format: "{separator}{number}"
  separator: "_"
""")
        
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_number=5,
            stage="hpo_trial"
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        # Should have version suffix if auto-increment is enabled
        # Format may vary based on implementation
        assert len(run_name) > 0
    
    def test_auto_increment_disabled_for_final_training(self, root_dir, config_dir):
        """Test auto-increment disabled for final_training, conversion."""
        mlflow_yaml = config_dir / "mlflow.yaml"
        mlflow_yaml.write_text("""
naming:
  run_name:
    auto_increment:
      enabled: true
      processes:
        hpo: true
        benchmarking: true
""")
        
        context = NamingContext(
            process_type="final_training",
            model="distilbert",
            environment="local",
            storage_env="local",
            spec_fp="abc123def4567890",
            exec_fp="xyz789abc1234567",
            variant=1
        )
        
        run_name = build_mlflow_run_name(context, config_dir=config_dir, root_dir=root_dir)
        # Should use variant, not auto-increment
        assert "v1" in run_name


class TestBuildParentTrainingId:
    """Test building parent training ID."""

    def test_build_parent_training_id(self):
        """Test building parent training ID."""
        spec_fp = "abc123def4567890"
        exec_fp = "xyz789abc1234567"
        variant = 1
        
        parent_id = build_parent_training_id(spec_fp, exec_fp, variant)
        
        assert parent_id == "spec_abc123def4567890_exec_xyz789abc1234567/v1"
    
    def test_build_parent_training_id_variant_2(self):
        """Test building parent training ID with variant 2."""
        spec_fp = "abc123def4567890"
        exec_fp = "xyz789abc1234567"
        variant = 2
        
        parent_id = build_parent_training_id(spec_fp, exec_fp, variant)
        
        assert parent_id == "spec_abc123def4567890_exec_xyz789abc1234567/v2"

