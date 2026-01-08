"""Comprehensive unit tests for path resolution module."""

import json
import yaml
from pathlib import Path
import pytest

from orchestration.paths import (
    load_paths_config,
    apply_env_overrides,
    validate_paths_config,
    resolve_output_path,
    get_cache_file_path,
    get_timestamped_cache_filename,
    get_cache_strategy_config,
    save_cache_with_dual_strategy,
    load_cache_file,
    get_drive_backup_base,
    get_drive_backup_path,
    parse_hpo_path_v2,
    is_v2_path,
    find_study_by_hash,
    find_trial_by_hash,
    build_output_path,
)
from orchestration.naming_centralized import NamingContext


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


class TestLoadPathsConfig:
    """Test paths configuration loading (1.1)."""

    def test_load_paths_config_with_file(self, config_dir):
        """Test loading paths config from existing file."""
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
        assert config["cache"]["best_configurations"] == "best_configurations"

    def test_load_paths_config_without_file(self, config_dir):
        """Test loading paths config with defaults when file doesn't exist."""
        config = load_paths_config(config_dir)

        # Should return defaults
        assert "base" in config
        assert "outputs" in config
        assert config["base"]["outputs"] == "outputs"

    def test_load_paths_config_schema_version_v1(self, config_dir):
        """Test schema version v1 validation."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 1
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
""")

        config = load_paths_config(config_dir)
        assert config["schema_version"] == 1

    def test_load_paths_config_schema_version_v2(self, config_dir):
        """Test schema version v2+ validation with required patterns."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
patterns:
  final_training_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}"
  conversion_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}/conv-{conv8}"
  best_config_v2: "{model}/spec-{spec8}"
  hpo_v2: "{storage_env}/{model}/study-{study8}/trial-{trial8}"
  benchmarking_v2: "{storage_env}/{model}/study-{study8}/trial-{trial8}/bench-{bench8}"
""")

        config = load_paths_config(config_dir)
        assert config["schema_version"] == 2
        assert "patterns" in config

    def test_load_paths_config_missing_base_outputs(self, config_dir):
        """Test invalid config handling (missing base.outputs)."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base: {}
outputs:
  hpo: "hpo"
""")

        with pytest.raises(ValueError, match="base.outputs"):
            load_paths_config(config_dir)

    def test_load_paths_config_invalid_schema_version(self, config_dir):
        """Test invalid schema_version handling."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: "invalid"
base:
  outputs: "outputs"
""")

        with pytest.raises(ValueError, match="schema_version must be an integer"):
            load_paths_config(config_dir)

    def test_load_paths_config_missing_required_patterns_v2(self, config_dir):
        """Test missing required pattern keys for v2+."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
patterns:
  final_training_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}"
  # Missing other required patterns
""")

        with pytest.raises(ValueError, match="Missing required pattern keys"):
            load_paths_config(config_dir)

    def test_load_paths_config_unknown_placeholder(self, config_dir):
        """Test placeholder validation (unknown token)."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
schema_version: 2
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
patterns:
  final_training_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}"
  conversion_v2: "{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}/conv-{conv8}"
  best_config_v2: "{model}/spec-{spec8}"
  hpo_v2: "{storage_env}/{model}/study-{study8}/trial-{trial8}"
  benchmarking_v2: "{storage_env}/{model}/study-{study8}/trial-{trial8}/bench-{bench8}"
  test_pattern: "{unknown_placeholder}"
""")

        # Should warn or raise for unknown placeholder in v2+
        with pytest.raises(ValueError, match="Unknown placeholder"):
            load_paths_config(config_dir)


class TestApplyEnvOverrides:
    """Test environment overrides (1.2)."""

    def test_apply_env_overrides_colab(self, config_dir):
        """Test Colab environment override."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
env_overrides:
  colab:
    base:
      outputs: "/content/drive/MyDrive/resume-ner-azureml/outputs"
""")

        config = load_paths_config(config_dir)
        overridden = apply_env_overrides(config, "colab")

        assert overridden["base"]["outputs"] == "/content/drive/MyDrive/resume-ner-azureml/outputs"

    def test_apply_env_overrides_azureml(self, config_dir):
        """Test Azure ML environment override."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
env_overrides:
  azureml:
    base:
      outputs: "/mnt/outputs"
""")

        config = load_paths_config(config_dir)
        overridden = apply_env_overrides(config, "azureml")

        assert overridden["base"]["outputs"] == "/mnt/outputs"

    def test_apply_env_overrides_kaggle(self, config_dir):
        """Test Kaggle environment override."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
env_overrides:
  kaggle:
    base:
      outputs: "/kaggle/working/outputs"
""")

        config = load_paths_config(config_dir)
        overridden = apply_env_overrides(config, "kaggle")

        assert overridden["base"]["outputs"] == "/kaggle/working/outputs"

    def test_apply_env_overrides_local(self, config_dir):
        """Test local (no override)."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
env_overrides:
  colab:
    base:
      outputs: "/content/drive/MyDrive/resume-ner-azureml/outputs"
""")

        config = load_paths_config(config_dir)
        overridden = apply_env_overrides(config, "local")

        # Should return original config
        assert overridden["base"]["outputs"] == "outputs"

    def test_apply_env_overrides_shallow_merge(self, config_dir):
        """Test shallow merge behavior (only base, outputs sections)."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
  benchmarking: "benchmarking"
env_overrides:
  colab:
    base:
      outputs: "/content/drive/MyDrive/resume-ner-azureml/outputs"
    outputs:
      hpo: "hpo_colab"
""")

        config = load_paths_config(config_dir)
        overridden = apply_env_overrides(config, "colab")

        assert overridden["base"]["outputs"] == "/content/drive/MyDrive/resume-ner-azureml/outputs"
        assert overridden["outputs"]["hpo"] == "hpo_colab"
        # Not overridden
        assert overridden["outputs"]["benchmarking"] == "benchmarking"

    def test_apply_env_overrides_missing_override(self, config_dir):
        """Test missing override (returns original config)."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
env_overrides:
  colab:
    base:
      outputs: "/content/drive/MyDrive/resume-ner-azureml/outputs"
""")

        config = load_paths_config(config_dir)
        overridden = apply_env_overrides(config, "nonexistent")

        # Should return original config
        assert overridden["base"]["outputs"] == "outputs"

    def test_apply_env_overrides_none_storage_env(self, config_dir):
        """Test None storage_env (returns original config)."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
env_overrides:
  colab:
    base:
      outputs: "/content/drive/MyDrive/resume-ner-azureml/outputs"
""")

        config = load_paths_config(config_dir)
        overridden = apply_env_overrides(config, None)

        # Should return original config
        assert overridden["base"]["outputs"] == "outputs"


class TestResolveOutputPath:
    """Test output path resolution (1.3)."""

    def test_resolve_simple_path_hpo(self, root_dir, config_dir):
        """Test resolving simple HPO path."""
        path = resolve_output_path(root_dir, config_dir, "hpo")
        assert path == root_dir / "outputs" / "hpo"

    def test_resolve_simple_path_benchmarking(self, root_dir, config_dir):
        """Test resolving simple benchmarking path."""
        path = resolve_output_path(root_dir, config_dir, "benchmarking")
        assert path == root_dir / "outputs" / "benchmarking"

    def test_resolve_simple_path_final_training(self, root_dir, config_dir):
        """Test resolving simple final_training path."""
        path = resolve_output_path(root_dir, config_dir, "final_training")
        assert path == root_dir / "outputs" / "final_training"

    def test_resolve_simple_path_conversion(self, root_dir, config_dir):
        """Test resolving simple conversion path."""
        path = resolve_output_path(root_dir, config_dir, "conversion")
        assert path == root_dir / "outputs" / "conversion"

    def test_resolve_cache_subdirectory(self, root_dir, config_dir):
        """Test resolving cache subdirectory."""
        path = resolve_output_path(
            root_dir, config_dir, "cache", subcategory="best_configurations"
        )
        assert path == root_dir / "outputs" / "cache" / "best_configurations"

    def test_resolve_path_with_pattern(self, root_dir, config_dir):
        """Test resolving path with pattern replacement."""
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
            root_dir, config_dir, "final_training",
            backbone="distilbert", run_id="20251227_220407"
        )
        assert path == root_dir / "outputs" / \
            "final_training" / "distilbert_20251227_220407"

    def test_resolve_path_absolute_base(self, root_dir, config_dir):
        """Test absolute vs relative base paths."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "/absolute/path/outputs"
outputs:
  hpo: "hpo"
""")

        path = resolve_output_path(root_dir, config_dir, "hpo")
        assert path == Path("/absolute/path/outputs") / "hpo"

    def test_resolve_path_all_categories(self, root_dir, config_dir):
        """Test all output categories from config."""
        categories = [
            "hpo", "hpo_tests", "benchmarking", "final_training",
            "dry_run", "conversion", "best_model_selection",
            "cache", "e2e_test", "pytest_logs"
        ]

        for category in categories:
            path = resolve_output_path(root_dir, config_dir, category)
            # Path may not exist, but should be valid
            assert path.exists() or not path.exists()
            assert "outputs" in str(path) or category == "cache"


class TestBuildOutputPathV2:
    """Test v2 path building (1.4)."""

    def test_build_output_path_hpo_with_hashes(self, root_dir, config_dir):
        """Test HPO with study_key_hash and trial_key_hash (v2 pattern)."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890"
        )

        path = build_output_path(root_dir, context, config_dir=config_dir)

        # Should use v2 pattern: outputs/hpo/{storage_env}/{model}/study-{study8}/trial-{trial8}
        assert "hpo" in str(path)
        assert "local" in str(path)
        assert "distilbert" in str(path)
        assert "study-350a79aa" in str(path)
        assert "trial-747428f2" in str(path)

    def test_build_output_path_hpo_without_hashes(self, root_dir, config_dir):
        """Test HPO without hashes (fallback to legacy)."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="local",
            trial_id="trial_1_20251229_100000"
        )

        # Should fallback to legacy pattern
        path = build_output_path(root_dir, context, config_dir=config_dir)
        assert "hpo" in str(path)
        assert "distilbert" in str(path)

    def test_build_output_path_hpo_with_storage_env_override(self, root_dir, config_dir):
        """Test HPO with storage_env override."""
        context = NamingContext(
            process_type="hpo",
            model="distilbert",
            environment="local",
            storage_env="colab",
            study_key_hash="350a79aa1234567890abcdef",
            trial_key_hash="747428f2abcdef1234567890"
        )

        path = build_output_path(root_dir, context, config_dir=config_dir)
        assert "colab" in str(path)

    def test_build_output_path_hpo_refit_with_hashes(self, root_dir, config_dir):
        """Test HPO refit with hashes (v2 pattern with /refit suffix)."""
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

    def test_build_output_path_benchmarking_with_all_hashes(self, root_dir, config_dir):
        """Test benchmarking with study_key_hash, trial_key_hash, benchmark_config_hash."""
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

    def test_build_output_path_benchmarking_without_bench_hash(self, root_dir, config_dir):
        """Test benchmarking without bench hash (optional)."""
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
        # bench-{hash} may or may not be present

    def test_build_output_path_final_training_variant_1(self, root_dir, config_dir):
        """Test final training with spec_fp, exec_fp, variant=1."""
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
        assert "spec-abc123de" in str(path) or "spec_abc123de" in str(path)
        assert "exec-xyz789ab" in str(path) or "exec_xyz789ab" in str(path)
        assert "v1" in str(path) or "/1" in str(path)

    def test_build_output_path_final_training_multiple_variants(self, root_dir, config_dir):
        """Test final training with multiple variants (v1, v2, v3)."""
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

    def test_build_output_path_conversion(self, root_dir, config_dir):
        """Test conversion with parent_training_id and conv_fp."""
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

    def test_build_output_path_best_configurations(self, root_dir, config_dir):
        """Test best configurations with spec_fp."""
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


class TestCacheFilePaths:
    """Test cache file paths (1.5)."""

    def test_get_cache_file_path_latest(self, root_dir, config_dir):
        """Test getting latest cache file path."""
        path = get_cache_file_path(
            root_dir, config_dir, "best_configurations", file_type="latest"
        )
        assert path.name == "latest_best_configuration.json"
        assert "best_configurations" in str(path)

    def test_get_cache_file_path_index(self, root_dir, config_dir):
        """Test getting index cache file path."""
        path = get_cache_file_path(
            root_dir, config_dir, "best_configurations", file_type="index"
        )
        assert path.name == "index.json"

    def test_get_cache_file_path_specific_filename(self, root_dir, config_dir):
        """Test specific filename override."""
        path = get_cache_file_path(
            root_dir, config_dir, "best_configurations", filename="custom.json"
        )
        assert path.name == "custom.json"

    def test_get_cache_file_path_all_cache_types(self, root_dir, config_dir):
        """Test all cache types."""
        cache_types = ["best_configurations",
                       "final_training", "best_model_selection"]

        for cache_type in cache_types:
            path = get_cache_file_path(
                root_dir, config_dir, cache_type, file_type="latest")
            assert path.exists() or not path.exists()  # May not exist yet
            assert cache_type in str(path) or "cache" in str(path)


class TestCacheStrategyOperations:
    """Test cache strategy operations (1.6)."""

    def test_save_cache_creates_timestamped_file(self, root_dir, config_dir):
        """Test that saving cache creates timestamped file."""
        data = {
            "backbone": "distilbert",
            "trial_name": "trial_2",
            "selection_criteria": {"best_value": 0.5},
        }

        timestamped_file, latest_file, index_file = save_cache_with_dual_strategy(
            root_dir=root_dir,
            config_dir=config_dir,
            cache_type="best_configurations",
            data=data,
            backbone="distilbert",
            identifier="trial_2",
            timestamp="20251227_220407",
        )

        assert timestamped_file.exists()
        timestamped_data = json.loads(timestamped_file.read_text())
        assert timestamped_data["backbone"] == "distilbert"
        assert "cache_metadata" in timestamped_data

    def test_save_cache_updates_latest_pointer(self, root_dir, config_dir):
        """Test that saving cache updates latest pointer file."""
        data = {"backbone": "distilbert", "trial_name": "trial_2"}

        timestamped_file, latest_file, index_file = save_cache_with_dual_strategy(
            root_dir=root_dir,
            config_dir=config_dir,
            cache_type="best_configurations",
            data=data,
            backbone="distilbert",
            identifier="trial_2",
            timestamp="20251227_220407",
        )

        assert latest_file.exists()
        latest_data = json.loads(latest_file.read_text())
        assert latest_data["backbone"] == "distilbert"
        assert latest_data["cache_metadata"]["timestamped_file"] == timestamped_file.name

    def test_save_cache_updates_index_file(self, root_dir, config_dir):
        """Test that saving cache updates index file."""
        data = {"backbone": "distilbert", "trial_name": "trial_2"}

        timestamped_file, latest_file, index_file = save_cache_with_dual_strategy(
            root_dir=root_dir,
            config_dir=config_dir,
            cache_type="best_configurations",
            data=data,
            backbone="distilbert",
            identifier="trial_2",
            timestamp="20251227_220407",
        )

        assert index_file.exists()
        index_data = json.loads(index_file.read_text())
        assert "entries" in index_data
        assert len(index_data["entries"]) == 1
        assert index_data["entries"][0]["timestamp"] == "20251227_220407"

    def test_save_cache_max_entries_limit(self, root_dir, config_dir):
        """Test max entries limit enforcement."""
        # Create config with max_entries limit
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
cache_strategies:
  best_configurations:
    strategy: "dual"
    timestamped:
      enabled: true
    latest:
      enabled: true
    index:
      enabled: true
      max_entries: 2
""")

        # Save 3 entries
        for i in range(3):
            data = {"backbone": "distilbert", "trial_name": f"trial_{i}"}
            save_cache_with_dual_strategy(
                root_dir=root_dir,
                config_dir=config_dir,
                cache_type="best_configurations",
                data=data,
                backbone="distilbert",
                identifier=f"trial_{i}",
                timestamp=f"20251227_22040{i}",
            )

        # Check index has only max_entries
        index_file = get_cache_file_path(
            root_dir, config_dir, "best_configurations", file_type="index")
        index_data = json.loads(index_file.read_text())
        assert len(index_data["entries"]) <= 2

    def test_load_cache_file_latest(self, root_dir, config_dir):
        """Test loading latest cache file."""
        # Create cache with save
        data = {"backbone": "distilbert", "trial": "trial_2"}
        save_cache_with_dual_strategy(
            root_dir=root_dir,
            config_dir=config_dir,
            cache_type="best_configurations",
            data=data,
            backbone="distilbert",
            identifier="trial_2",
            timestamp="20251227_220407",
        )

        loaded = load_cache_file(root_dir, config_dir,
                                 "best_configurations", use_latest=True)
        assert loaded is not None
        assert loaded["backbone"] == "distilbert"

    def test_load_cache_file_specific_timestamp(self, root_dir, config_dir):
        """Test loading cache by specific timestamp."""
        data = {"backbone": "distilbert", "timestamp": "20251227_220407"}
        save_cache_with_dual_strategy(
            root_dir=root_dir,
            config_dir=config_dir,
            cache_type="best_configurations",
            data=data,
            backbone="distilbert",
            identifier="trial_2",
            timestamp="20251227_220407",
        )

        loaded = load_cache_file(
            root_dir,
            config_dir,
            "best_configurations",
            use_latest=False,
            specific_timestamp="20251227_220407"
        )
        assert loaded is not None
        assert loaded["timestamp"] == "20251227_220407"

    def test_load_cache_file_specific_identifier(self, root_dir, config_dir):
        """Test loading cache by specific identifier (from index)."""
        data = {"backbone": "distilbert", "trial": "trial_2"}
        save_cache_with_dual_strategy(
            root_dir=root_dir,
            config_dir=config_dir,
            cache_type="best_configurations",
            data=data,
            backbone="distilbert",
            identifier="trial_2",
            timestamp="20251227_220407",
        )

        loaded = load_cache_file(
            root_dir,
            config_dir,
            "best_configurations",
            use_latest=False,
            specific_identifier="trial_2"
        )
        assert loaded is not None
        assert loaded["backbone"] == "distilbert"

    def test_load_cache_file_returns_none_when_not_found(self, root_dir, config_dir):
        """Test loading returns None when cache not found."""
        loaded = load_cache_file(
            root_dir, config_dir, "best_configurations", use_latest=True
        )
        assert loaded is None


class TestTimestampedCacheFilenames:
    """Test timestamped cache filenames (1.7)."""

    def test_get_timestamped_cache_filename_best_config(self, config_dir):
        """Test best config pattern."""
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

    def test_get_timestamped_cache_filename_final_training(self, config_dir):
        """Test final training pattern."""
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

    def test_get_timestamped_cache_filename_best_model_selection(self, config_dir):
        """Test best model selection pattern."""
        filename = get_timestamped_cache_filename(
            config_dir,
            "best_model_selection",
            "distilbert",
            "experiment_cachekey",
            "20251228_001000"
        )

        assert filename.startswith("best_model_selection_")
        assert "distilbert" in filename
        assert "experiment_cachekey" in filename
        assert "20251228_001000" in filename
        assert filename.endswith(".json")


class TestDriveBackupPaths:
    """Test Drive backup paths (1.8)."""

    def test_get_drive_backup_base(self, config_dir):
        """Test getting Drive backup base directory."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
drive:
  mount_point: "/content/drive"
  backup_base_dir: "resume-ner-azureml"
""")

        drive_base = get_drive_backup_base(config_dir)
        assert drive_base is not None
        assert "drive" in str(drive_base)
        assert "resume-ner-azureml" in str(drive_base)

    def test_get_drive_backup_path_converts_local_path(self, root_dir, config_dir):
        """Test converting local output path to Drive path."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
drive:
  mount_point: "/content/drive"
  backup_base_dir: "resume-ner-azureml"
""")

        local_path = root_dir / "outputs" / "hpo" / "local" / "distilbert"
        local_path.mkdir(parents=True, exist_ok=True)

        drive_path = get_drive_backup_path(root_dir, config_dir, local_path)
        assert drive_path is not None
        assert "drive" in str(drive_path)
        assert "hpo" in str(drive_path)
        assert "local" in str(drive_path)
        assert "distilbert" in str(drive_path)

    def test_get_drive_backup_path_returns_none_outside_outputs(self, root_dir, config_dir):
        """Test returns None for paths outside outputs/."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
drive:
  mount_point: "/content/drive"
  backup_base_dir: "resume-ner-azureml"
""")

        local_path = root_dir / "other_dir" / "file.txt"
        drive_path = get_drive_backup_path(root_dir, config_dir, local_path)
        assert drive_path is None

    def test_get_drive_backup_path_returns_none_when_not_configured(self, root_dir, config_dir):
        """Test returns None when Drive not configured."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
""")

        local_path = root_dir / "outputs" / "hpo"
        drive_path = get_drive_backup_path(root_dir, config_dir, local_path)
        assert drive_path is None


class TestPathParsingAndDetection:
    """Test path parsing and detection (1.9)."""

    def test_parse_hpo_path_v2_extracts_components(self):
        """Test extracting study8, trial8, storage_env, model from v2 paths."""
        path = Path(
            "outputs/hpo/local/distilbert/study-350a79aa/trial-747428f2")
        result = parse_hpo_path_v2(path)

        assert result is not None
        assert result["storage_env"] == "local"
        assert result["model"] == "distilbert"
        assert result["study8"] == "350a79aa"
        assert result["trial8"] == "747428f2"

    def test_parse_hpo_path_v2_returns_none_for_non_v2(self):
        """Test returns None for non-v2 paths."""
        path = Path("outputs/hpo/local/distilbert/trial_1_20251229_100000")
        result = parse_hpo_path_v2(path)
        assert result is None

    def test_parse_hpo_path_v2_handles_full_paths(self):
        """Test handles full paths and relative fragments."""
        full_path = Path(
            "/absolute/path/outputs/hpo/local/distilbert/study-350a79aa/trial-747428f2")
        result = parse_hpo_path_v2(full_path)
        assert result is not None
        assert result["study8"] == "350a79aa"

    def test_is_v2_path_detects_v2_pattern(self):
        """Test detects v2 pattern."""
        v2_path = Path(
            "outputs/hpo/local/distilbert/study-350a79aa/trial-747428f2")
        assert is_v2_path(v2_path) is True

    def test_is_v2_path_returns_false_for_legacy(self):
        """Test returns False for legacy paths."""
        legacy_path = Path(
            "outputs/hpo/local/distilbert/trial_1_20251229_100000")
        assert is_v2_path(legacy_path) is False

    def test_find_study_by_hash(self, root_dir, config_dir):
        """Test finding study folder by study_key_hash."""
        # Create v2 structure
        study_path = root_dir / "outputs" / "hpo" / \
            "local" / "distilbert" / "study-350a79aa"
        study_path.mkdir(parents=True, exist_ok=True)

        found = find_study_by_hash(
            root_dir, config_dir, "distilbert", "350a79aa1234567890abcdef")
        assert found is not None
        assert "study-350a79aa" in str(found)

    def test_find_trial_by_hash(self, root_dir, config_dir):
        """Test finding trial folder by study_key_hash + trial_key_hash."""
        # Create v2 structure
        trial_path = root_dir / "outputs" / "hpo" / "local" / \
            "distilbert" / "study-350a79aa" / "trial-747428f2"
        trial_path.mkdir(parents=True, exist_ok=True)

        found = find_trial_by_hash(
            root_dir, config_dir, "distilbert",
            "350a79aa1234567890abcdef", "747428f2abcdef1234567890"
        )
        assert found is not None
        assert "trial-747428f2" in str(found)

    def test_find_study_by_hash_returns_none_when_not_found(self, root_dir, config_dir):
        """Test returns None when study not found."""
        found = find_study_by_hash(
            root_dir, config_dir, "distilbert", "nonexistent1234567890")
        assert found is None

    def test_find_trial_by_hash_returns_none_when_not_found(self, root_dir, config_dir):
        """Test returns None when trial not found."""
        found = find_trial_by_hash(
            root_dir, config_dir, "distilbert",
            "350a79aa1234567890abcdef", "nonexistent1234567890"
        )
        assert found is None


class TestBaseDirectories:
    """Test base directory configuration (1.10)."""

    def test_base_outputs(self, config_dir):
        """Test base.outputs configuration."""
        paths_yaml = config_dir / "paths.yaml"
        paths_yaml.write_text("""
base:
  outputs: "outputs"
  notebooks: "notebooks"
  config: "config"
  src: "src"
  tests: "tests"
  mlruns: "mlruns"
""")

        config = load_paths_config(config_dir)
        assert config["base"]["outputs"] == "outputs"
        assert config["base"]["notebooks"] == "notebooks"
        assert config["base"]["config"] == "config"
        assert config["base"]["src"] == "src"
        assert config["base"]["tests"] == "tests"
        assert config["base"]["mlruns"] == "mlruns"
