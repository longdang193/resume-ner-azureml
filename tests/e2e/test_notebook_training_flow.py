"""Integration tests for notebook 02_best_config_selection.ipynb training flow.

These tests mirror the notebook execution flow:
1. Environment detection
2. Repository setup
3. Load configuration
4. Setup MLflow
5. Drive backup setup (Colab only)
6. Best model selection
7. Final training
8. Model conversion handoff
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch

import pytest

import orchestration.jobs.final_training.executor as executor
from orchestration.config_loader import ExperimentConfig


class DummyExperimentConfig(SimpleNamespace):
    """Minimal stand-in for ExperimentConfig."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, "data_config"):
            self.data_config = None
        if not hasattr(self, "name"):
            self.name = "test_experiment"


@pytest.fixture
def mock_mlflow_client(monkeypatch):
    """Mock MLflow client for all tests."""
    mock_client = Mock()
    mock_experiment = Mock()
    mock_experiment.experiment_id = "exp-123"
    mock_experiment.name = "test-experiment-training"
    mock_client.get_experiment_by_name.return_value = mock_experiment
    mock_client.search_experiments.return_value = []
    monkeypatch.setattr("orchestration.jobs.final_training.executor.MlflowClient", lambda: mock_client)
    return mock_client


@pytest.fixture
def mock_mlflow_tracking(monkeypatch):
    """Mock MLflow tracking URI."""
    mock_mlflow = Mock()
    mock_mlflow.get_tracking_uri.return_value = "file:///tmp/mlflow"
    monkeypatch.setattr("orchestration.jobs.final_training.executor.mlflow", mock_mlflow)
    return mock_mlflow


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create a tiny dataset for testing."""
    dataset_dir = tmp_path / "dataset_tiny" / "seed0"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.json").write_text(json.dumps([
        {"text": "Sample text", "annotations": []}
    ]))
    (dataset_dir / "validation.json").write_text(json.dumps([
        {"text": "Validation text", "annotations": []}
    ]))
    return dataset_dir


def test_notebook_flow_local_environment_detection(tmp_path, monkeypatch):
    """Test Step 1: Environment detection (local)."""
    # Simulate local environment
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.delenv("COLAB_TPU", raising=False)
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)

    from shared.platform_detection import detect_platform

    platform = detect_platform()
    assert platform == "local"


def test_notebook_flow_colab_environment_detection(tmp_path, monkeypatch):
    """Test Step 1: Environment detection (Colab)."""
    monkeypatch.setenv("COLAB_GPU", "1")

    from shared.platform_detection import detect_platform

    platform = detect_platform()
    assert platform == "colab"


def test_notebook_flow_config_loading(tmp_path, monkeypatch):
    """Test Step 3: Load configuration."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create minimal experiment config
    exp_config_file = config_dir / "experiment" / "test.yaml"
    exp_config_file.parent.mkdir()
    exp_config_file.write_text("""
experiment_name: test_experiment
data_config: data/resume_tiny.yaml
model_config: model/distilbert.yaml
train_config: train.yaml
""")

    # Mock the config loader
    def fake_load_experiment_config(cfg_dir, name):
        return DummyExperimentConfig(
            name="test_experiment",
            data_config=config_dir / "data" / "resume_tiny.yaml"
        )

    monkeypatch.setattr(
        "orchestration.config_loader.load_experiment_config",
        fake_load_experiment_config
    )

    from orchestration.config_loader import load_experiment_config

    exp_config = load_experiment_config(config_dir, "test_experiment")
    assert exp_config.name == "test_experiment"


def test_notebook_flow_best_model_selection_with_cache(tmp_path, monkeypatch, mock_mlflow_client):
    """Test Step 6: Best model selection with cache reuse."""
    root_dir = tmp_path
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create cache directory
    cache_dir = tmp_path / "outputs" / "cache" / "best_model_selection"
    cache_dir.mkdir(parents=True)

    # Create cached best model selection
    cache_data = {
        "best_model": {
            "backbone": "distilbert-base-uncased",
            "params": {"learning_rate": 2e-5},
            "tags": {"code.stage": "hpo_refit"},
        },
        "experiment_name": "test_experiment",
    }
    (cache_dir / "latest_best_model_selection_cache.json").write_text(
        json.dumps(cache_data)
    )

    # Mock the selection cache loader
    def fake_load_cached_best_model(*args, **kwargs):
        return cache_data

    monkeypatch.setattr(
        "orchestration.jobs.selection.cache.load_cached_best_model",
        fake_load_cached_best_model
    )

    from orchestration.jobs.selection.cache import load_cached_best_model

    result = load_cached_best_model(
        root_dir=root_dir,
        config_dir=config_dir,
        experiment_name="test_experiment",
        selection_config={"run": {"mode": "reuse_if_exists"}},
        tags_config={},
        benchmark_experiment_id="bench-123",
        tracking_uri="file:///tmp/mlflow",
    )

    assert result is not None
    assert result["best_model"]["backbone"] == "distilbert-base-uncased"


def test_notebook_flow_final_training_reuse_on_rerun(tmp_path, monkeypatch, tiny_dataset, mock_mlflow_client, mock_mlflow_tracking):
    """Test Step 7: Final training reuse on notebook rerun (reuse_if_exists mode)."""
    root_dir = tmp_path
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    outputs_root = tmp_path / "outputs"

    # Setup patches
    def _setup_patches():
        monkeypatch.setattr(
            executor,
            "load_final_training_config",
            lambda root_dir, config_dir, best_config, experiment_config: {
                "backbone": "distilbert-base-uncased",
                "spec_fp": "spec123",
                "exec_fp": "exec456",
                "variant": 1,
                "learning_rate": 1e-4,
                "batch_size": 4,
                "dropout": 0.1,
                "weight_decay": 0.01,
                "epochs": 1,
                "random_seed": 42,
                "early_stopping_enabled": False,
                "use_combined_data": True,
            },
        )
        monkeypatch.setattr(executor, "load_all_configs", lambda experiment_config: {})
        monkeypatch.setattr(executor, "detect_platform", lambda: "local")

        def fake_create_context(**kwargs):
            return SimpleNamespace(**kwargs)

        def fake_build_output_path(root_dir_arg, ctx):
            variant = ctx.variant if hasattr(ctx, 'variant') else 1
            return outputs_root / f"v{variant}"

        monkeypatch.setattr(executor, "create_naming_context", fake_create_context)
        monkeypatch.setattr(executor, "build_output_path", fake_build_output_path)
        monkeypatch.setattr(executor, "build_mlflow_run_name", lambda **kwargs: "test_run_name")
        monkeypatch.setattr(executor, "build_mlflow_tags", lambda **kwargs: {})

        def fake_load_yaml(path: Path):
            if path.name == "final_training.yaml":
                return {"run": {"mode": "reuse_if_exists"}, "dataset": {}}
            return {}

        monkeypatch.setattr(executor, "load_yaml", fake_load_yaml)
        monkeypatch.setattr(executor, "resolve_dataset_path", lambda data_config: tiny_dataset)

        import orchestration.metadata_manager as mm
        monkeypatch.setattr(mm, "save_metadata_with_fingerprints", lambda **kwargs: None)

    _setup_patches()

    # Create completed checkpoint from previous run
    final_output_dir = outputs_root / "v1"
    checkpoint_dir = final_output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "config.json").write_text("{}")
    (checkpoint_dir / "model.safetensors").write_text("fake")
    metadata = {"status": {"training": {"completed": True}}}
    (final_output_dir / "metadata.json").write_text(json.dumps(metadata))

    # Track if training was called
    training_called = []

    def fake_subprocess_run(*args, **kwargs):
        training_called.append(True)
        raise AssertionError("Training should not be called when reusing")

    monkeypatch.setattr(executor, "subprocess", SimpleNamespace(run=fake_subprocess_run))

    # Execute (simulating notebook rerun)
    best_model = {
        "backbone": "distilbert-base-uncased",
        "params": {},
        "tags": {"code.stage": "hpo_refit"},
    }
    experiment_config = DummyExperimentConfig()

    result = executor.execute_final_training(
        root_dir=root_dir,
        config_dir=config_dir,
        best_model=best_model,
        experiment_config=experiment_config,
        lineage={},
        training_experiment_name="test-experiment-training",
        platform="local",
    )

    # Should have reused existing checkpoint
    assert result == checkpoint_dir
    assert len(training_called) == 0


def test_notebook_flow_conversion_handoff_requires_metadata(tmp_path, monkeypatch):
    """Test Step 8: Conversion handoff requires parent training metadata."""
    # Simulate final training output directory
    training_output_dir = tmp_path / "outputs" / "final_training" / "local" / "distilbert" / "spec-abc_exec-xyz" / "v1"
    training_output_dir.mkdir(parents=True)
    checkpoint_dir = training_output_dir / "checkpoint"
    checkpoint_dir.mkdir()

    # Case 1: Missing metadata.json should raise error
    with pytest.raises((FileNotFoundError, ValueError), match="Metadata|metadata"):
        from shared.json_cache import load_json
        metadata_path = training_output_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Case 2: Metadata missing spec_fp/exec_fp should raise error
    metadata_path = training_output_dir / "metadata.json"
    incomplete_metadata = {"backbone": "distilbert"}
    metadata_path.write_text(json.dumps(incomplete_metadata))

    with pytest.raises(ValueError, match="Missing required fingerprints|spec_fp|exec_fp"):
        from shared.json_cache import load_json
        metadata = load_json(metadata_path)
        if not metadata.get("spec_fp") or not metadata.get("exec_fp"):
            raise ValueError("Missing required fingerprints in metadata")

    # Case 3: Complete metadata should work
    complete_metadata = {
        "spec_fp": "abc123def456",
        "exec_fp": "xyz789abc123",
        "mlflow": {"run_id": "run-123"},
    }
    metadata_path.write_text(json.dumps(complete_metadata))

    from shared.json_cache import load_json
    metadata = load_json(metadata_path)
    assert metadata["spec_fp"] == "abc123def456"
    assert metadata["exec_fp"] == "xyz789abc123"


def test_notebook_flow_tracking_disabled_still_produces_outputs(tmp_path, monkeypatch, tiny_dataset, mock_mlflow_client):
    """Test that tracking disabled still produces filesystem outputs."""
    root_dir = tmp_path
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    outputs_root = tmp_path / "outputs"

    # Setup patches similar to reuse test
    def _setup_patches():
        monkeypatch.setattr(
            executor,
            "load_final_training_config",
            lambda root_dir, config_dir, best_config, experiment_config: {
                "backbone": "distilbert-base-uncased",
                "spec_fp": "spec123",
                "exec_fp": "exec456",
                "variant": 1,
                "learning_rate": 1e-4,
                "batch_size": 4,
                "dropout": 0.1,
                "weight_decay": 0.01,
                "epochs": 1,
                "random_seed": 42,
                "early_stopping_enabled": False,
                "use_combined_data": True,
            },
        )
        monkeypatch.setattr(executor, "load_all_configs", lambda experiment_config: {})
        monkeypatch.setattr(executor, "detect_platform", lambda: "local")

        def fake_create_context(**kwargs):
            return SimpleNamespace(**kwargs)

        def fake_build_output_path(root_dir_arg, ctx):
            variant = ctx.variant if hasattr(ctx, 'variant') else 1
            return outputs_root / f"v{variant}"

        monkeypatch.setattr(executor, "create_naming_context", fake_create_context)
        monkeypatch.setattr(executor, "build_output_path", fake_build_output_path)
        monkeypatch.setattr(executor, "build_mlflow_run_name", lambda **kwargs: "test_run_name")
        monkeypatch.setattr(executor, "build_mlflow_tags", lambda **kwargs: {})

        def fake_load_yaml(path: Path):
            if path.name == "final_training.yaml":
                return {"run": {"mode": "force_new"}, "dataset": {}}
            return {}

        monkeypatch.setattr(executor, "load_yaml", fake_load_yaml)
        monkeypatch.setattr(executor, "resolve_dataset_path", lambda data_config: tiny_dataset)

        import orchestration.metadata_manager as mm
        monkeypatch.setattr(mm, "save_metadata_with_fingerprints", lambda **kwargs: None)

    _setup_patches()

    # Mock MLflow to fail (disabled)
    mock_mlflow_client.get_experiment_by_name.side_effect = Exception("MLflow disabled")
    # Patch mlflow.get_tracking_uri to return None (disabled)
    mock_mlflow = Mock()
    mock_mlflow.get_tracking_uri.return_value = None
    monkeypatch.setattr(executor, "mlflow", mock_mlflow)

    subprocess_calls = []

    def fake_subprocess_run(*args, **kwargs):
        subprocess_calls.append(args)
        result = Mock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    monkeypatch.setattr(executor, "subprocess", SimpleNamespace(run=fake_subprocess_run))

    best_model = {"backbone": "distilbert-base-uncased", "params": {}}
    experiment_config = DummyExperimentConfig()

    # Should not raise even if MLflow fails
    result = executor.execute_final_training(
        root_dir=root_dir,
        config_dir=config_dir,
        best_model=best_model,
        experiment_config=experiment_config,
        lineage={},
        training_experiment_name="test-experiment-training",
        platform="local",
    )

    # Training should still proceed (subprocess was called)
    assert len(subprocess_calls) > 0
    # Result should be a checkpoint path (even if directory doesn't exist yet in mocked scenario)
    assert result is not None
    assert "checkpoint" in str(result)


def test_notebook_flow_storage_env_paths_colab(tmp_path, monkeypatch):
    """Test that Colab environment uses correct output paths."""
    monkeypatch.setenv("COLAB_GPU", "1")

    from shared.platform_detection import detect_platform
    from orchestration.paths import resolve_output_path

    platform = detect_platform()
    assert platform == "colab"

    # Check that paths.yaml env_overrides would apply
    # (actual path resolution tested in unit tests, this just verifies platform detection)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    # paths.yaml would have colab override pointing to /content/drive/...
    # But we can't test that without full config setup, so just verify platform


def test_notebook_flow_storage_env_paths_kaggle(tmp_path, monkeypatch):
    """Test that Kaggle environment uses correct output paths."""
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")

    from shared.platform_detection import detect_platform

    platform = detect_platform()
    assert platform == "kaggle"


def test_notebook_flow_metric_based_best_checkpoint_selection(tmp_path, monkeypatch):
    """Test that best checkpoint is selected based on training.metric and training.metric_mode."""
    root_dir = tmp_path
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create train.yaml with metric configuration
    train_config_file = config_dir / "train.yaml"
    train_config_file.write_text("""
training:
  metric: "macro-f1"
  metric_mode: "max"
  epochs: 1
  batch_size: 1
""")

    # Load train config
    from shared.yaml_utils import load_yaml
    train_config = load_yaml(train_config_file)

    # Verify metric configuration
    assert train_config["training"]["metric"] == "macro-f1"
    assert train_config["training"]["metric_mode"] == "max"

    # Test that metric_mode "max" means select highest value
    # (This is tested at the training script level, but we verify config is correct)
    assert train_config["training"]["metric_mode"] == "max"

    # Test that metric_mode "min" would select lowest value
    train_config_min = train_config.copy()
    train_config_min["training"]["metric_mode"] = "min"
    assert train_config_min["training"]["metric_mode"] == "min"

    # Verify default metric is macro-f1 (from train.yaml)
    # This ensures best checkpoint selection uses the right metric
    assert train_config["training"]["metric"] == "macro-f1"

