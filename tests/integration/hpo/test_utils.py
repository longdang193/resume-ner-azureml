"""Test utility functions for HPO integration tests."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock
import optuna


def create_mock_trial(trial_number: int, params: Dict[str, Any], metric_value: float) -> Mock:
    """
    Create mock Optuna trial object.
    
    Args:
        trial_number: Trial number.
        params: Trial hyperparameters.
        metric_value: Trial objective metric value.
    
    Returns:
        Mock trial with attributes.
    """
    mock_trial = Mock(spec=optuna.trial.FrozenTrial)
    mock_trial.number = trial_number
    mock_trial.params = params
    mock_trial.value = metric_value
    mock_trial.state = optuna.trial.TrialState.COMPLETE
    mock_trial.user_attrs = {}
    mock_trial.system_attrs = {}
    return mock_trial


def create_mock_mlflow_run(
    run_id: str,
    tags: Optional[Dict[str, str]] = None,
    metrics: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Mock:
    """
    Create mock MLflow run object.
    
    Args:
        run_id: MLflow run ID.
        tags: Optional run tags.
        metrics: Optional run metrics.
        params: Optional run parameters.
    
    Returns:
        Mock run with data.
    """
    mock_run = Mock()
    mock_run.info.run_id = run_id
    mock_run.info.experiment_id = "exp_123"
    mock_run.info.status = "RUNNING"
    mock_run.data.tags = tags or {}
    mock_run.data.metrics = metrics or {}
    mock_run.data.params = params or {}
    return mock_run


def assert_path_structure_v2(output_dir: Path, expected_structure: Dict[str, Any]) -> None:
    """
    Validate v2 path structure.
    
    Args:
        output_dir: Base output directory.
        expected_structure: Dictionary with expected structure:
            - study_folders: Expected number of study folders
            - trial_folders: Expected number of trial folders per study
            - cv_folds: Expected number of CV fold folders per trial (if CV enabled)
            - refit_folders: Expected number of refit folders (if refit enabled)
    
    Raises:
        AssertionError: If structure doesn't match expectations.
    """
    study_folders = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("study-")]
    
    if "study_folders" in expected_structure:
        assert len(study_folders) == expected_structure["study_folders"], \
            f"Expected {expected_structure['study_folders']} study folders, found {len(study_folders)}"
    
    if study_folders and "trial_folders" in expected_structure:
        study_folder = study_folders[0]
        trial_folders = [d for d in study_folder.iterdir() if d.is_dir() and d.name.startswith("trial-")]
        assert len(trial_folders) == expected_structure["trial_folders"], \
            f"Expected {expected_structure['trial_folders']} trial folders, found {len(trial_folders)}"
        
        if "cv_folds" in expected_structure and expected_structure["cv_folds"] > 0:
            for trial_folder in trial_folders:
                cv_folder = trial_folder / "cv"
                assert cv_folder.exists(), f"CV folder should exist in {trial_folder}"
                fold_folders = [d for d in cv_folder.iterdir() if d.is_dir() and d.name.startswith("fold")]
                assert len(fold_folders) == expected_structure["cv_folds"], \
                    f"Expected {expected_structure['cv_folds']} CV fold folders, found {len(fold_folders)}"
        
        if "refit_folders" in expected_structure and expected_structure["refit_folders"] > 0:
            refit_count = 0
            for trial_folder in trial_folders:
                refit_folder = trial_folder / "refit"
                if refit_folder.exists():
                    refit_count += 1
            assert refit_count == expected_structure["refit_folders"], \
                f"Expected {expected_structure['refit_folders']} refit folders, found {refit_count}"


def assert_mlflow_tags(run: Mock, expected_tags: Dict[str, str]) -> None:
    """
    Validate MLflow tags present and correct.
    
    Args:
        run: Mock MLflow run object.
        expected_tags: Dictionary of expected tag key-value pairs.
    
    Raises:
        AssertionError: If tags don't match expectations.
    """
    actual_tags = run.data.tags if hasattr(run.data, 'tags') else {}
    
    for key, expected_value in expected_tags.items():
        assert key in actual_tags, f"Tag '{key}' not found in run tags"
        assert actual_tags[key] == expected_value, \
            f"Tag '{key}' has value '{actual_tags[key]}', expected '{expected_value}'"


def assert_study_trials_preserved(
    study1: optuna.Study,
    study2: optuna.Study,
    tolerance: float = 1e-6,
) -> None:
    """
    Assert that trials from study1 are preserved in study2.
    
    Args:
        study1: First study (original).
        study2: Second study (resumed).
        tolerance: Tolerance for float comparisons.
    
    Raises:
        AssertionError: If trials are not preserved.
    """
    study1_values = [t.value for t in study1.trials if t.value is not None]
    study2_values = [t.value for t in study2.trials if t.value is not None]
    
    # All study1 values should be in study2
    for value1 in study1_values:
        found = any(abs(value1 - value2) < tolerance for value2 in study2_values)
        assert found, f"Trial value {value1} from study1 not found in study2"


def create_hpo_config(
    max_trials: int = 1,
    k_fold_enabled: bool = False,
    refit_enabled: bool = False,
    checkpoint_enabled: bool = True,
    study_name: str = "hpo_test",
) -> Dict[str, Any]:
    """
    Create HPO config dictionary with specified options.
    
    Args:
        max_trials: Maximum number of trials.
        k_fold_enabled: Whether k-fold CV is enabled.
        refit_enabled: Whether refit training is enabled.
        checkpoint_enabled: Whether checkpointing is enabled.
        study_name: Study name for checkpointing.
    
    Returns:
        HPO configuration dictionary.
    """
    config = {
        "search_space": {
            "backbone": {"type": "choice", "values": ["distilbert"]},
            "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 5e-5},
            "batch_size": {"type": "choice", "values": [4]},
        },
        "sampling": {"algorithm": "random", "max_trials": max_trials, "timeout_minutes": 20},
        "objective": {"metric": "macro-f1", "goal": "maximize"},
        "k_fold": {
            "enabled": k_fold_enabled,
            "n_splits": 2 if k_fold_enabled else None,
            "random_seed": 42,
            "shuffle": True,
            "stratified": True,
        },
        "refit": {"enabled": refit_enabled},
    }
    
    if checkpoint_enabled:
        config["checkpoint"] = {
            "enabled": True,
            "study_name": study_name,
            "storage_path": "{study_name}/study.db",
            "auto_resume": True,
        }
    
    return config


def setup_mlflow_mocks():
    """
    Set up MLflow mocks for testing.
    
    Returns:
        Tuple of (mock_client, mock_parent_run, mock_mlflow_module).
    """
    mock_parent_run = Mock()
    mock_parent_run.info.run_id = "hpo_parent_123"
    mock_parent_run.info.experiment_id = "exp_123"
    mock_parent_run.info.status = "RUNNING"
    
    def get_run_side_effect(run_id):
        if isinstance(run_id, str):
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
    
    mock_mlflow = Mock()
    mock_mlflow.tracking.MlflowClient.return_value = mock_client
    mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_parent_run)
    mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
    mock_mlflow.active_run.return_value = mock_parent_run
    mock_mlflow.set_experiment = Mock()
    mock_mlflow.get_tracking_uri.return_value = "file:///mlruns"
    
    return mock_client, mock_parent_run, mock_mlflow


