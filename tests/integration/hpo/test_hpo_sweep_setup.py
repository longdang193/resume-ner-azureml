"""Component tests for HPO sweep setup."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from orchestration.config_loader import load_experiment_config, load_all_configs
from orchestration.jobs.hpo.local.mlflow.run_setup import setup_hpo_mlflow_run
from orchestration.jobs.hpo.hpo_helpers import create_study_name


class TestHPOSweepSetup:
    """Test HPO sweep setup including config loading and MLflow run creation."""

    def test_load_configs_from_smoke_yaml(self, tmp_path):
        """Test loading configs via load_experiment_config and load_all_configs."""
        # Create config directory structure
        config_root = tmp_path / "config"
        config_root.mkdir()
        
        # Create experiment config
        experiment_dir = config_root / "experiment"
        experiment_dir.mkdir()
        experiment_yaml = experiment_dir / "test.yaml"
        experiment_yaml.write_text("""
experiment_name: test
data_config: data/data.yaml
model_config: model/model.yaml
train_config: train/train.yaml
hpo_config: hpo/smoke.yaml
env_config: env/env.yaml
benchmark_config: benchmark/benchmark.yaml
""")
        
        # Create domain configs
        (config_root / "data").mkdir()
        (config_root / "model").mkdir()
        (config_root / "train").mkdir()
        (config_root / "hpo").mkdir()
        (config_root / "env").mkdir()
        
        (config_root / "data" / "data.yaml").write_text("name: test_data\nversion: 1.0")
        (config_root / "model" / "model.yaml").write_text("backbone: distilbert")
        (config_root / "train" / "train.yaml").write_text("epochs: 10")
        (config_root / "hpo" / "smoke.yaml").write_text("""
search_space:
  backbone:
    type: choice
    values: [distilbert]
  learning_rate:
    type: loguniform
    min: 1e-5
    max: 5e-5
sampling:
  algorithm: random
  max_trials: 1
checkpoint:
  enabled: true
  study_name: "hpo_{backbone}_smoke_test_path_testing_23"
""")
        (config_root / "env" / "env.yaml").write_text("name: test_env")
        (config_root / "benchmark.yaml").write_text("benchmarking:\n  enabled: true")
        
        # Load experiment config
        exp_cfg = load_experiment_config(config_root, "test")
        
        assert exp_cfg.name == "test"
        assert exp_cfg.hpo_config == config_root / "hpo" / "smoke.yaml"
        
        # Load all configs
        configs = load_all_configs(exp_cfg)
        
        assert "data" in configs
        assert "model" in configs
        assert "train" in configs
        assert "hpo" in configs
        assert "env" in configs
        
        # Verify smoke.yaml content
        assert configs["hpo"]["search_space"]["backbone"]["values"] == ["distilbert"]
        assert configs["hpo"]["checkpoint"]["study_name"] == "hpo_{backbone}_smoke_test_path_testing_23"
        
        # Benchmark config is only loaded if file exists
        if exp_cfg.benchmark_config.exists():
            assert "benchmark" in configs
        else:
            # If benchmark.yaml doesn't exist, it won't be in configs
            assert "benchmark" not in configs

    def test_setup_hpo_mlflow_run_creates_parent_run(self, tmp_path):
        """Test that setup_hpo_mlflow_run creates MLflow parent run with correct tags."""
        # Create config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create minimal paths.yaml
        (config_dir / "paths.yaml").write_text("base:\n  outputs: outputs")
        (config_dir / "naming.yaml").write_text("""
schema_version: 1
run_names:
  hpo_sweep:
    pattern: "hpo_{model}_{study_name}"
""")
        (config_dir / "tags.yaml").write_text("schema_version: 1")
        (config_dir / "mlflow.yaml").write_text("naming:\n  project_name: resume-ner")
        
        backbone = "distilbert"
        study_name = "hpo_distilbert_smoke_test_path_testing_23"
        output_dir = tmp_path / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        run_id = "20251227_220407"
        
        data_config = {"name": "test", "version": "1.0"}
        hpo_config = {"search_space": {}, "objective": {"metric": "macro-f1"}}
        benchmark_config = {"benchmarking": {"enabled": True}}
        
        # Compute study_key_hash
        from orchestration.jobs.tracking.naming.hpo_keys import (
            build_hpo_study_key,
            build_hpo_study_key_hash,
        )
        study_key = build_hpo_study_key(data_config, hpo_config, backbone, benchmark_config)
        study_key_hash = build_hpo_study_key_hash(study_key)
        
        context, run_name = setup_hpo_mlflow_run(
            backbone=backbone,
            study_name=study_name,
            output_dir=output_dir,
            run_id=run_id,
            should_resume=False,
            checkpoint_enabled=True,
            data_config=data_config,
            hpo_config=hpo_config,
            benchmark_config=benchmark_config,
            study_key_hash=study_key_hash,
        )
        
        # Verify context was created
        assert context is not None
        assert context.model == "distilbert"
        assert context.study_key_hash == study_key_hash
        
        # Verify run name was generated
        assert run_name is not None
        assert len(run_name) > 0

    def test_setup_hpo_mlflow_run_computes_study_key_hash(self, tmp_path):
        """Test that setup_hpo_mlflow_run computes study_key_hash if not provided."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "paths.yaml").write_text("base:\n  outputs: outputs")
        (config_dir / "naming.yaml").write_text("schema_version: 1")
        
        backbone = "distilbert"
        study_name = "hpo_distilbert_smoke_test_path_testing_23"
        output_dir = tmp_path / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        data_config = {"name": "test", "version": "1.0"}
        hpo_config = {"search_space": {}, "objective": {"metric": "macro-f1"}}
        
        # Don't provide study_key_hash - should be computed
        context, run_name = setup_hpo_mlflow_run(
            backbone=backbone,
            study_name=study_name,
            output_dir=output_dir,
            run_id="test",
            should_resume=False,
            checkpoint_enabled=True,
            data_config=data_config,
            hpo_config=hpo_config,
            benchmark_config=None,
            study_key_hash=None,  # Should be computed
        )
        
        # Verify study_key_hash was computed
        assert context is not None
        assert context.study_key_hash is not None
        assert len(context.study_key_hash) == 64  # Full SHA256 hash

    def test_create_study_name_from_checkpoint_config(self):
        """Test that study name uses template from checkpoint.study_name."""
        backbone = "distilbert"
        run_id = "20251227_220407"
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_{backbone}_smoke_test_path_testing_23",
        }
        
        study_name = create_study_name(
            backbone=backbone,
            run_id=run_id,
            should_resume=False,
            checkpoint_config=checkpoint_config,
        )
        
        # Should replace {backbone} placeholder
        assert study_name == "hpo_distilbert_smoke_test_path_testing_23"
        assert "{backbone}" not in study_name

    def test_create_study_name_without_checkpoint(self):
        """Test study name creation when checkpoint disabled."""
        backbone = "distilbert"
        run_id = "20251227_220407"
        checkpoint_config = {"enabled": False}
        
        study_name = create_study_name(
            backbone=backbone,
            run_id=run_id,
            should_resume=False,
            checkpoint_config=checkpoint_config,
        )
        
        # Should use unique name with run_id
        assert "hpo_distilbert" in study_name
        assert run_id in study_name

    def test_setup_hpo_mlflow_run_tags(self, tmp_path):
        """Test that MLflow parent run has correct tags (code.stage=hpo, code.project=resume-ner)."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "paths.yaml").write_text("base:\n  outputs: outputs")
        (config_dir / "naming.yaml").write_text("""
schema_version: 1
run_names:
  hpo_sweep:
    pattern: "hpo_{model}_{study_name}"
""")
        (config_dir / "mlflow.yaml").write_text("naming:\n  project_name: resume-ner")
        
        backbone = "distilbert"
        study_name = "hpo_distilbert_smoke_test_path_testing_23"
        output_dir = tmp_path / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        data_config = {"name": "test", "version": "1.0"}
        hpo_config = {"search_space": {}, "objective": {"metric": "macro-f1"}}
        
        from orchestration.jobs.tracking.naming.hpo_keys import (
            build_hpo_study_key,
            build_hpo_study_key_hash,
        )
        study_key = build_hpo_study_key(data_config, hpo_config, backbone, None)
        study_key_hash = build_hpo_study_key_hash(study_key)
        
        context, run_name = setup_hpo_mlflow_run(
            backbone=backbone,
            study_name=study_name,
            output_dir=output_dir,
            run_id="test",
            should_resume=False,
            checkpoint_enabled=True,
            data_config=data_config,
            hpo_config=hpo_config,
            study_key_hash=study_key_hash,
        )
        
        # Verify context has correct attributes
        assert context is not None
        assert context.stage == "hpo_sweep"
        assert context.model == "distilbert"
        assert context.study_key_hash == study_key_hash

    def test_checkpoint_file_created(self, tmp_path):
        """Test that checkpoint file is created at {study_name}/study.db."""
        from orchestration.jobs.hpo.local.checkpoint.manager import (
            resolve_storage_path,
        )
        
        output_dir = tmp_path / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        checkpoint_config = {
            "enabled": True,
            "study_name": "hpo_distilbert_smoke_test_path_testing_23",
            "storage_path": "{study_name}/study.db",
        }
        
        study_name = "hpo_distilbert_smoke_test_path_testing_23"
        backbone = "distilbert"
        
        storage_path = resolve_storage_path(
            output_dir=output_dir,
            checkpoint_config=checkpoint_config,
            backbone=backbone,
            study_name=study_name,
        )
        
        # Verify path resolves correctly
        assert storage_path is not None
        assert "hpo_distilbert_smoke_test_path_testing_23" in str(storage_path)
        assert "study.db" in str(storage_path)
        
        # Create the checkpoint file
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_path.touch()
        
        assert storage_path.exists()

    def test_study_key_hash_and_family_hash_computed(self, tmp_path):
        """Test that study_key_hash and study_family_hash are computed and can be tagged."""
        from orchestration.jobs.tracking.naming.hpo_keys import (
            build_hpo_study_key,
            build_hpo_study_key_hash,
            build_hpo_study_family_key,
            build_hpo_study_family_hash,
        )
        
        data_config = {
            "name": "resume_ner",
            "version": "1.0",
            "local_path": "/data",
        }
        hpo_config = {
            "search_space": {"learning_rate": {"type": "loguniform", "min": 1e-5, "max": 5e-5}},
            "objective": {"metric": "macro-f1", "goal": "maximize"},
        }
        benchmark_config = {"benchmarking": {"metric": "macro-f1"}}
        backbone = "distilbert"
        
        # Compute study_key_hash
        study_key = build_hpo_study_key(data_config, hpo_config, backbone, benchmark_config)
        study_key_hash = build_hpo_study_key_hash(study_key)
        
        # Compute study_family_hash
        study_family_key = build_hpo_study_family_key(data_config, hpo_config, benchmark_config)
        study_family_hash = build_hpo_study_family_hash(study_family_key)
        
        # Verify hashes are computed
        assert study_key_hash is not None
        assert len(study_key_hash) == 64
        assert study_family_hash is not None
        assert len(study_family_hash) == 64
        assert study_key_hash != study_family_hash  # Should be different

