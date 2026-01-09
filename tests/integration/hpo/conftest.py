"""Shared pytest fixtures for HPO integration tests."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock
from typing import Dict, Any


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create temporary config directory with all required YAML files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create minimal config files required for HPO tests
    (config_dir / "data.yaml").write_text("dataset_name: test_data\ndataset_version: v1")
    (config_dir / "mlflow.yaml").write_text("experiment_name: test_exp")
    (config_dir / "paths.yaml").write_text("""schema_version: 2
base:
  outputs: "outputs"
outputs:
  hpo: "hpo"
patterns:
  hpo_v2: '{storage_env}/{model}/study-{study8}'
  final_training_v2: '{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}'
  conversion_v2: '{storage_env}/{model}/spec-{spec8}_exec-{exec8}/v{variant}/conv-{conv8}'
  best_config_v2: '{model}/spec-{spec8}'
  benchmarking_v2: '{storage_env}/{model}/study-{study8}/trial-{trial8}/bench-{bench8}'""")
    (config_dir / "naming.yaml").write_text("run_name_templates:\n  hpo: 'hpo_{model}_{stage}'")
    (config_dir / "tags.yaml").write_text("project_name: test_project")
    
    return config_dir


@pytest.fixture
def tmp_project_structure(tmp_path, tmp_config_dir):
    """Create temporary project structure with src/training module."""
    # Create src/training module structure (required for trial execution)
    src_dir = tmp_path / "src" / "training"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text("# Training module")
    
    return tmp_path


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create minimal dataset (10 samples) in JSON format."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    # Create train.json (dataset loader expects .json, not .jsonl)
    (dataset_dir / "train.json").write_text(json.dumps([
        {"text": f"Sample {i}", "label": "POS" if i % 2 == 0 else "NEG"}
        for i in range(10)
    ]))
    
    # Create val.json
    (dataset_dir / "val.json").write_text(json.dumps([
        {"text": f"Val {i}", "label": "POS" if i % 2 == 0 else "NEG"}
        for i in range(5)
    ]))
    
    return dataset_dir


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary output directory for HPO."""
    output_dir = tmp_path / "outputs" / "hpo" / "local" / "distilbert"
    output_dir.mkdir(parents=True)
    return output_dir


@pytest.fixture
def mock_mlflow_client():
    """Provide mocked MLflow client with common operations."""
    mock_parent_run = Mock()
    mock_parent_run.info.run_id = "hpo_parent_123"
    mock_parent_run.info.experiment_id = "exp_123"
    mock_parent_run.info.status = "RUNNING"
    
    def get_run_side_effect(run_id):
        if run_id == "hpo_parent_123" or isinstance(run_id, str):
            # Set up tags with string values (not Mock objects)
            mock_parent_run.data.tags = {
                "code.study_key_hash": "a" * 64,
                "code.study_family_hash": "b" * 64,
            }
            return mock_parent_run
        return mock_parent_run
    
    mock_client = Mock()
    mock_client.get_run.side_effect = get_run_side_effect
    mock_client.create_run = Mock()
    mock_client.set_tag = Mock()
    mock_client.log_metric = Mock()
    mock_client.log_param = Mock()
    mock_client.set_terminated = Mock()
    
    return mock_client, mock_parent_run


@pytest.fixture
def mock_mlflow_setup(mock_mlflow_client):
    """Set up MLflow mocks for HPO tests."""
    mock_client, mock_parent_run = mock_mlflow_client
    
    # This fixture can be used with @patch to set up MLflow
    # Return the mocks so tests can use them
    return {
        "client": mock_client,
        "parent_run": mock_parent_run,
    }


@pytest.fixture
def hpo_config_smoke():
    """Load and return smoke.yaml HPO config structure."""
    return {
        "search_space": {
            "backbone": {"type": "choice", "values": ["distilbert"]},
            "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 5e-5},
            "batch_size": {"type": "choice", "values": [4]},
            "dropout": {"type": "uniform", "min": 0.1, "max": 0.3},
            "weight_decay": {"type": "loguniform", "min": 0.001, "max": 0.1},
        },
        "sampling": {"algorithm": "random", "max_trials": 1, "timeout_minutes": 20},
        "checkpoint": {
            "enabled": True,
            "study_name": "hpo_distilbert_smoke_test",
            "storage_path": "{study_name}/study.db",
            "auto_resume": True,
            "save_only_best": True,
        },
        "mlflow": {"log_best_checkpoint": True},
        "early_termination": {
            "policy": "bandit",
            "evaluation_interval": 1,
            "slack_factor": 0.2,
            "delay_evaluation": 2,
        },
        "objective": {"metric": "macro-f1", "goal": "maximize"},
        "selection": {
            "accuracy_threshold": 0.015,
            "use_relative_threshold": True,
            "min_accuracy_gain": 0.02,
        },
        "k_fold": {
            "enabled": True,
            "n_splits": 2,
            "random_seed": 42,
            "shuffle": True,
            "stratified": True,
        },
        "refit": {"enabled": True},
        "cleanup": {
            "disable_auto_cleanup": False,
            "disable_auto_optuna_mark": False,
        },
    }


@pytest.fixture
def hpo_config_minimal():
    """Minimal HPO config for simple tests."""
    return {
        "search_space": {
            "backbone": {"type": "choice", "values": ["distilbert"]},
            "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 5e-5},
            "batch_size": {"type": "choice", "values": [4]},
        },
        "sampling": {"algorithm": "random", "max_trials": 1, "timeout_minutes": 20},
        "checkpoint": {
            "enabled": True,
            "study_name": "hpo_test",
            "storage_path": "{study_name}/study.db",
            "auto_resume": True,
        },
        "objective": {"metric": "macro-f1", "goal": "maximize"},
        "k_fold": {"enabled": False},
        "refit": {"enabled": False},
    }


@pytest.fixture
def train_config_minimal():
    """Minimal training config for HPO tests."""
    return {"training": {"epochs": 1}}


@pytest.fixture
def data_config_minimal():
    """Minimal data config for HPO tests."""
    return {"dataset_name": "test_data", "dataset_version": "v1"}


@pytest.fixture
def mock_training_subprocess(tmp_output_dir):
    """Mock training subprocess to return success and create metrics.json."""
    from orchestration.constants import METRICS_FILENAME
    
    def subprocess_side_effect(*args, **kwargs):
        # Extract output_dir from environment variable (AZURE_ML_OUTPUT_CHECKPOINT)
        output_path = None
        if "env" in kwargs:
            env = kwargs["env"]
            if "AZURE_ML_OUTPUT_CHECKPOINT" in env:
                output_path = Path(env["AZURE_ML_OUTPUT_CHECKPOINT"])
            elif "AZURE_ML_OUTPUT_checkpoint" in env:
                output_path = Path(env["AZURE_ML_OUTPUT_checkpoint"])
        
        # Fallback: try to extract from command args
        if not output_path:
            cmd = args[0] if args else []
            for i, arg in enumerate(cmd):
                if isinstance(arg, str) and arg == "--output-dir" and i + 1 < len(cmd):
                    output_path = Path(cmd[i + 1])
                    break
        
        # If we found an output path, create metrics there
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            metrics_file = output_path / METRICS_FILENAME
            metrics_file.write_text(json.dumps({"macro-f1": 0.75}))
        
        # Also proactively create metrics in any existing CV fold folders
        study_folders = list(tmp_output_dir.glob("study-*"))
        for study_folder in study_folders:
            for trial_folder in study_folder.glob("trial-*"):
                cv_folder = trial_folder / "cv"
                if cv_folder.exists():
                    for fold_folder in cv_folder.glob("fold*"):
                        fold_folder.mkdir(parents=True, exist_ok=True)
                        metrics_file = fold_folder / METRICS_FILENAME
                        if not metrics_file.exists():
                            metrics_file.write_text(json.dumps({"macro-f1": 0.75}))
                else:
                    trial_folder.mkdir(parents=True, exist_ok=True)
                    metrics_file = trial_folder / METRICS_FILENAME
                    if not metrics_file.exists():
                        metrics_file.write_text(json.dumps({"macro-f1": 0.75}))
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Training completed"
        mock_result.stderr = ""
        return mock_result
    
    return subprocess_side_effect


