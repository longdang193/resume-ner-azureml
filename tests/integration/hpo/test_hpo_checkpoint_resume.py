"""Integration tests for HPO checkpoint and resume functionality."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
import sys
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from orchestration.jobs.local_sweeps import run_local_hpo_sweep
from shared.platform_detection import detect_platform, resolve_checkpoint_path
from orchestration.jobs.checkpoint_manager import resolve_storage_path, get_storage_uri


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def hpo_config():
    """Minimal HPO config for testing."""
    return {
        "search_space": {
            "learning_rate": {
                "type": "loguniform",
                "min": 1e-5,
                "max": 5e-5,
            },
            "batch_size": {
                "type": "choice",
                "values": [4],
            },
        },
        "sampling": {
            "algorithm": "random",
            "max_trials": 3,
            "timeout_minutes": 10,
        },
        "objective": {
            "metric": "macro-f1",
            "goal": "maximize",
        },
    }


@pytest.fixture
def train_config():
    """Minimal train config for testing."""
    return {
        "training": {
            "epochs": 1,
        },
    }


def test_checkpoint_creation(temp_output_dir, hpo_config, train_config, tmp_path):
    """Test that checkpoint file is created when checkpointing is enabled."""
    checkpoint_config = {
        "enabled": True,
        "storage_path": "study.db",
        "auto_resume": True,
    }
    
    # Create a minimal dataset structure
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "train.jsonl").write_text('{"text": "test", "entities": []}\n')
    (dataset_path / "val.jsonl").write_text('{"text": "test", "entities": []}\n')
    
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    output_dir = temp_output_dir / "hpo"
    output_dir.mkdir(parents=True)
    
    # Mock the training subprocess to avoid actual training
    with patch("orchestration.jobs.local_sweeps.subprocess.run") as mock_run:
        # Mock successful training with metrics
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        
        # Create metrics file
        trial_output = output_dir / "trial_0"
        trial_output.mkdir(parents=True)
        metrics_file = trial_output / "metrics.json"
        metrics_file.write_text(json.dumps({"macro-f1": 0.5}))
        
        study = run_local_hpo_sweep(
            dataset_path=str(dataset_path),
            config_dir=config_dir,
            backbone="distilbert",
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            mlflow_experiment_name="test-checkpoint",
            checkpoint_config=checkpoint_config,
        )
        
        # Verify checkpoint file was created
        checkpoint_file = output_dir / "study.db"
        assert checkpoint_file.exists(), "Checkpoint file should be created"
        
        # Verify study has trials
        assert len(study.trials) > 0, "Study should have trials"


def test_resume_from_checkpoint(temp_output_dir, hpo_config, train_config, tmp_path):
    """Test that HPO can resume from an existing checkpoint."""
    checkpoint_config = {
        "enabled": True,
        "storage_path": "study.db",
        "auto_resume": True,
    }
    
    # Create a minimal dataset structure
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "train.jsonl").write_text('{"text": "test", "entities": []}\n')
    (dataset_path / "val.jsonl").write_text('{"text": "test", "entities": []}\n')
    
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    output_dir = temp_output_dir / "hpo"
    output_dir.mkdir(parents=True)
    
    # First run: complete 2 trials
    hpo_config["sampling"]["max_trials"] = 2
    
    with patch("orchestration.jobs.local_sweeps.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        
        # Create metrics files for first 2 trials
        for trial_num in range(2):
            trial_output = output_dir / f"trial_{trial_num}"
            trial_output.mkdir(parents=True)
            metrics_file = trial_output / "metrics.json"
            metrics_file.write_text(json.dumps({"macro-f1": 0.5 + trial_num * 0.1}))
        
        study1 = run_local_hpo_sweep(
            dataset_path=str(dataset_path),
            config_dir=config_dir,
            backbone="distilbert",
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            mlflow_experiment_name="test-resume",
            checkpoint_config=checkpoint_config,
        )
        
        # Import optuna for TrialState enum
        import optuna
        completed_trials_1 = len([
            t for t in study1.trials 
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        assert completed_trials_1 == 2, f"First run should complete 2 trials, got {completed_trials_1}"
    
    # Second run: resume and complete remaining trials (max_trials=3, already completed 2)
    hpo_config["sampling"]["max_trials"] = 3
    
    with patch("orchestration.jobs.local_sweeps.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        
        # Create metrics file for third trial
        trial_output = output_dir / "trial_2"
        trial_output.mkdir(parents=True)
        metrics_file = trial_output / "metrics.json"
        metrics_file.write_text(json.dumps({"macro-f1": 0.7}))
        
        study2 = run_local_hpo_sweep(
            dataset_path=str(dataset_path),
            config_dir=config_dir,
            backbone="distilbert",
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            mlflow_experiment_name="test-resume",
            checkpoint_config=checkpoint_config,
        )
        
        # Import optuna for TrialState enum
        import optuna
        completed_trials_2 = len([
            t for t in study2.trials 
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        # Should have 3 completed trials total (2 from first run + 1 from second)
        assert completed_trials_2 == 3, f"Resumed run should have 3 total completed trials, got {completed_trials_2}"


def test_checkpoint_disabled(temp_output_dir, hpo_config, train_config, tmp_path):
    """Test that checkpoint file is NOT created when checkpointing is disabled."""
    checkpoint_config = {
        "enabled": False,
    }
    
    # Create a minimal dataset structure
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "train.jsonl").write_text('{"text": "test", "entities": []}\n')
    (dataset_path / "val.jsonl").write_text('{"text": "test", "entities": []}\n')
    
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    output_dir = temp_output_dir / "hpo"
    output_dir.mkdir(parents=True)
    
    with patch("orchestration.jobs.local_sweeps.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        
        trial_output = output_dir / "trial_0"
        trial_output.mkdir(parents=True)
        metrics_file = trial_output / "metrics.json"
        metrics_file.write_text(json.dumps({"macro-f1": 0.5}))
        
        study = run_local_hpo_sweep(
            dataset_path=str(dataset_path),
            config_dir=config_dir,
            backbone="distilbert",
            hpo_config=hpo_config,
            train_config=train_config,
            output_dir=output_dir,
            mlflow_experiment_name="test-no-checkpoint",
            checkpoint_config=checkpoint_config,
        )
        
        # Verify checkpoint file was NOT created
        checkpoint_file = output_dir / "study.db"
        assert not checkpoint_file.exists(), "Checkpoint file should NOT be created when disabled"


def test_platform_detection():
    """Test platform detection utility."""
    # Test local platform (default)
    with patch.dict("os.environ", {}, clear=True):
        platform = detect_platform()
        assert platform == "local", f"Expected 'local', got '{platform}'"
    
    # Test Colab detection
    with patch.dict("os.environ", {"COLAB_GPU": "1"}, clear=True):
        platform = detect_platform()
        assert platform == "colab", f"Expected 'colab', got '{platform}'"
    
    # Test Kaggle detection
    with patch.dict("os.environ", {"KAGGLE_KERNEL_RUN_TYPE": "something"}, clear=True):
        platform = detect_platform()
        assert platform == "kaggle", f"Expected 'kaggle', got '{platform}'"


def test_resolve_storage_path(temp_output_dir):
    """Test storage path resolution."""
    checkpoint_config = {
        "enabled": True,
        "storage_path": "{backbone}/study.db",
    }
    
    storage_path = resolve_storage_path(
        output_dir=temp_output_dir,
        checkpoint_config=checkpoint_config,
        backbone="distilbert",
    )
    
    assert storage_path is not None, "Storage path should be resolved"
    assert storage_path.name == "study.db", "Storage path should end with study.db"
    assert "distilbert" in str(storage_path), "Storage path should contain backbone name"
    assert storage_path.parent.exists(), "Parent directory should be created"


def test_get_storage_uri():
    """Test storage URI conversion."""
    # Test with path
    storage_path = Path("/tmp/test/study.db")
    uri = get_storage_uri(storage_path)
    assert uri == f"sqlite:///{storage_path.resolve()}", f"Unexpected URI: {uri}"
    assert uri.startswith("sqlite:///"), "URI should start with sqlite:///"
    
    # Test with None
    uri = get_storage_uri(None)
    assert uri is None, "URI should be None for None path"

