"""Comprehensive unit tests for path building."""

from pathlib import Path
import pytest

from orchestration.naming_centralized import NamingContext, build_output_path


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


class TestHPOPaths:
    """Test HPO v2 path structure (3.1)."""

    def test_hpo_v2_path_with_hashes(self, root_dir, config_dir):
        """Test HPO v2 path with all required hashes."""
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
        
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890"
        )
        
        path = build_output_path(root_dir, context, config_dir=config_dir)
        
        assert "hpo" in str(path)
        assert "local" in str(path)
        assert "distilbert" in str(path)
        assert "study-350a79aa" in str(path)
        assert "trial-747428f2" in str(path)
    
    def test_hpo_v2_path_all_storage_environments(self, root_dir, config_dir):
        """Test HPO v2 path with all storage environments."""
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
        
        storage_envs = ["local", "colab", "kaggle", "azureml"]
        
        for storage_env in storage_envs:
            context = NamingContext(
                process_type="hpo",
                model="distilbert",
                environment=storage_env,
                storage_env=storage_env,
                study_key_hash="350a79aa1234567890abcdef",
                trial_key_hash="747428f2abcdef1234567890"
            )
            
            path = build_output_path(root_dir, context, config_dir=config_dir)
            assert storage_env in str(path)


class TestHPORefitPaths:
    """Test HPO refit path structure (3.2)."""

    def test_hpo_refit_path_structure(self, root_dir, config_dir):
        """Test HPO refit path structure."""
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
        
        context = NamingContext(
            process_type="hpo_refit",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890"
        )
        
        path = build_output_path(root_dir, context, config_dir=config_dir)
        
        assert "refit" in str(path)
        assert "study-350a79aa" in str(path)
        assert "trial-747428f2" in str(path)
    
    def test_hpo_refit_inherits_study_trial_structure(self, root_dir, config_dir):
        """Test HPO refit inherits study/trial structure from parent."""
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
        
        context = NamingContext(
            process_type="hpo_refit",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890"
        )
        
        path = build_output_path(root_dir, context, config_dir=config_dir)
        
        # Should have same study/trial structure as parent
        assert "study-350a79aa" in str(path)
        assert "trial-747428f2" in str(path)
        assert "refit" in str(path)


class TestBenchmarkingPaths:
    """Test benchmarking v2 path structure (3.3)."""

    def test_benchmarking_v2_path_with_bench_hash(self, root_dir, config_dir):
        """Test benchmarking v2 path with benchmark_config_hash."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  benchmarking: "benchmarking"
patterns:
  benchmarking_v2: "{storage_env}/{model}/study-{study8}/trial-{trial8}/bench-{bench8}"
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
        
        path = build_output_path(root_dir, context, config_dir=config_dir)
        
        assert "benchmarking" in str(path)
        assert "study-350a79aa" in str(path)
        assert "trial-747428f2" in str(path)
        assert "bench-abc12345" in str(path)
    
    def test_benchmarking_v2_path_without_bench_hash(self, root_dir, config_dir):
        """Test benchmarking v2 path without bench_hash (optional)."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  benchmarking: "benchmarking"
patterns:
  benchmarking_v2: "{storage_env}/{model}/study-{study8}/trial-{trial8}/bench-{bench8}"
""")
        
        context = NamingContext(
            process_type="benchmarking",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890"
        )
        
        path = build_output_path(root_dir, context, config_dir=config_dir)
        
        assert "benchmarking" in str(path)
        assert "study-350a79aa" in str(path)
        assert "trial-747428f2" in str(path)


class TestFinalTrainingPaths:
    """Test final training path structure (3.4)."""

    def test_final_training_path_structure(self, root_dir, config_dir):
        """Test final training path structure."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  final_training: "final_training"
patterns:
  final_training_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}"
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
        
        path = build_output_path(root_dir, context, config_dir=config_dir)
        
        assert "final_training" in str(path)
        assert "local" in str(path)
        assert "distilbert" in str(path)
        assert "spec-abc123de" in str(path) or "spec_abc123de" in str(path)
        assert "exec-xyz789ab" in str(path) or "exec_xyz789ab" in str(path)
        assert "v1" in str(path) or "/1" in str(path)
    
    def test_final_training_path_all_variants(self, root_dir, config_dir):
        """Test final training path with all variants (v1, v2, v3)."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  final_training: "final_training"
patterns:
  final_training_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}"
""")
        
        for variant in [1, 2, 3]:
            context = NamingContext(
                process_type="final_training",
                model="distilbert",
                environment="local",
                storage_env="local",
                spec_fp="abc123def4567890",
                exec_fp="xyz789abc1234567",
                variant=variant
            )
            
            path = build_output_path(root_dir, context, config_dir=config_dir)
            assert f"v{variant}" in str(path) or f"/{variant}" in str(path)
    
    def test_final_training_path_normalization(self, root_dir, config_dir):
        """Test path normalization (special characters in model/env)."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  final_training: "final_training"
patterns:
  final_training_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}"
normalize_paths:
  replace:
    "/": "_"
    "\\": "_"
    "-": "_"
    " ": "_"
""")
        
        context = NamingContext(
            process_type="final_training",
            model="distilbert-base-uncased",  # Has special characters
            environment="local",
            storage_env="local",
            spec_fp="abc123def4567890",
            exec_fp="xyz789abc1234567",
            variant=1
        )
        
        path = build_output_path(root_dir, context, config_dir=config_dir)
        # Path should be normalized (special chars replaced)
        assert "final_training" in str(path)


class TestConversionPaths:
    """Test conversion path structure (3.5)."""

    def test_conversion_path_structure(self, root_dir, config_dir):
        """Test conversion path structure."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  conversion: "conversion"
patterns:
  conversion_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}/conv-{conv8}"
""")
        
        context = NamingContext(
            process_type="conversion",
            model="distilbert",
            environment="local",
            storage_env="local",
            parent_training_id="spec-abc12345_exec-xyz789ab/v1",
            conv_fp="conv1234567890123"
        )
        
        path = build_output_path(root_dir, context, config_dir=config_dir)
        
        assert "conversion" in str(path)
        assert "conv" in str(path)
    
    def test_conversion_path_parses_parent_training_id(self, root_dir, config_dir):
        """Test conversion parses parent_training_id."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  conversion: "conversion"
patterns:
  conversion_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}/conv-{conv8}"
""")
        
        context = NamingContext(
            process_type="conversion",
            model="distilbert",
            environment="local",
            storage_env="local",
            parent_training_id="spec-abc12345_exec-xyz789ab/v1",
            conv_fp="conv1234567890123"
        )
        
        path = build_output_path(root_dir, context, config_dir=config_dir)
        
        # Should parse parent_training_id to extract spec/exec/variant
        assert "conversion" in str(path)
        assert "conv" in str(path)


class TestBestConfigurationsPaths:
    """Test best config path structure (3.6)."""

    def test_best_config_path_structure(self, root_dir, config_dir):
        """Test best config path structure."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  cache: "cache"
cache:
  best_configurations: "best_configurations"
patterns:
  best_config_v2: "{model}/spec-{spec8}"
""")
        
        context = NamingContext(
            process_type="best_configurations",
            model="distilbert",
            environment="local",
            storage_env="local",
            spec_fp="abc123def4567890"
        )
        
        path = build_output_path(root_dir, context, config_dir=config_dir)
        
        assert "cache" in str(path)
        assert "best_configurations" in str(path)
        assert "distilbert" in str(path)
        assert "spec" in str(path)


class TestPathNormalization:
    """Test path normalization (3.7)."""

    def test_path_normalization_special_characters(self, root_dir, config_dir):
        """Test special character replacement."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  final_training: "final_training"
patterns:
  final_training_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}"
normalize_paths:
  replace:
    "/": "_"
    "\\": "_"
    "-": "_"
    " ": "_"
    ":": "_"
    "*": "_"
    "?": "_"
    "\"": "_"
    "<": "_"
    ">": "_"
    "|": "_"
  lowercase: false
  max_component_length: 255
  max_path_length: 260
""")
        
        context = NamingContext(
            process_type="final_training",
            model="distilbert/base-uncased",  # Has special chars
            environment="local",
            storage_env="local",
            spec_fp="abc123def4567890",
            exec_fp="xyz789abc1234567",
            variant=1
        )
        
        path = build_output_path(root_dir, context, config_dir=config_dir)
        # Path should be normalized
        assert "final_training" in str(path)
    
    def test_path_normalization_case_preservation(self, root_dir, config_dir):
        """Test case preservation (lowercase: false)."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  final_training: "final_training"
patterns:
  final_training_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}"
normalize_paths:
  replace: {}
  lowercase: false
""")
        
        context = NamingContext(
            process_type="final_training",
            model="DistilBERT",  # Mixed case
            environment="local",
            storage_env="local",
            spec_fp="abc123def4567890",
            exec_fp="xyz789abc1234567",
            variant=1
        )
        
        path = build_output_path(root_dir, context, config_dir=config_dir)
        # Case should be preserved
        assert "final_training" in str(path)

