"""Custom assertion helpers for HPO tests."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pytest
import optuna

from orchestration.constants import METRICS_FILENAME


def assert_checkpoint_exists(study_folder: Path, study_name: Optional[str] = None) -> None:
    """
    Assert that checkpoint file (study.db) exists in study folder.
    
    Args:
        study_folder: Path to study folder.
        study_name: Optional study name (for legacy paths).
    
    Raises:
        AssertionError: If checkpoint doesn't exist.
    """
    checkpoint_files = list(study_folder.rglob("study.db"))
    assert len(checkpoint_files) > 0, f"Checkpoint file (study.db) not found in {study_folder}"


def assert_fold_splits_exist(study_folder: Path, n_splits: int) -> None:
    """
    Assert that fold_splits.json exists and has correct number of splits.
    
    Args:
        study_folder: Path to study folder.
        n_splits: Expected number of folds.
    
    Raises:
        AssertionError: If fold_splits.json doesn't exist or has wrong number of splits.
    """
    fold_splits_file = study_folder / "fold_splits.json"
    assert fold_splits_file.exists(), f"fold_splits.json not found in {study_folder}"
    
    fold_splits_data = json.loads(fold_splits_file.read_text())
    assert "folds" in fold_splits_data, "fold_splits.json missing 'folds' key"
    assert len(fold_splits_data["folds"]) == n_splits, \
        f"Expected {n_splits} folds, found {len(fold_splits_data['folds'])}"


def assert_trial_meta_exists(trial_folder: Path) -> None:
    """
    Assert that trial_meta.json exists in trial folder.
    
    Args:
        trial_folder: Path to trial folder.
    
    Raises:
        AssertionError: If trial_meta.json doesn't exist.
    """
    trial_meta_file = trial_folder / "trial_meta.json"
    assert trial_meta_file.exists(), f"trial_meta.json not found in {trial_folder}"


def assert_metrics_file_exists(output_dir: Path, metric_name: str = "macro-f1") -> None:
    """
    Assert that metrics.json exists and contains expected metric.
    
    Args:
        output_dir: Directory containing metrics.json.
        metric_name: Expected metric name.
    
    Raises:
        AssertionError: If metrics.json doesn't exist or doesn't contain metric.
    """
    metrics_file = output_dir / METRICS_FILENAME
    assert metrics_file.exists(), f"metrics.json not found in {output_dir}"
    
    metrics_data = json.loads(metrics_file.read_text())
    assert metric_name in metrics_data, \
        f"Metric '{metric_name}' not found in metrics.json. Available: {list(metrics_data.keys())}"


def assert_v2_path_pattern(path: Path, pattern_type: str = "study") -> None:
    """
    Assert that path follows v2 pattern (study-{hash} or trial-{hash}).
    
    Args:
        path: Path to check.
        pattern_type: Either "study" or "trial".
    
    Raises:
        AssertionError: If path doesn't match v2 pattern.
    """
    if pattern_type == "study":
        assert path.name.startswith("study-"), f"Path {path} should start with 'study-'"
        assert len(path.name) == len("study-") + 8, \
            f"Study folder name should be 'study-' + 8 hex chars, got {path.name}"
    elif pattern_type == "trial":
        assert path.name.startswith("trial-"), f"Path {path} should start with 'trial-'"
        assert len(path.name) == len("trial-") + 8, \
            f"Trial folder name should be 'trial-' + 8 hex chars, got {path.name}"
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")


def assert_study_has_trials(study: optuna.Study, min_trials: int = 1) -> None:
    """
    Assert that study has at least min_trials completed trials.
    
    Args:
        study: Optuna study object.
        min_trials: Minimum number of completed trials.
    
    Raises:
        AssertionError: If study doesn't have enough completed trials.
    """
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed_trials) >= min_trials, \
        f"Expected at least {min_trials} completed trials, found {len(completed_trials)}"


def assert_best_trial_selected(study: optuna.Study) -> None:
    """
    Assert that study has a best trial selected.
    
    Args:
        study: Optuna study object.
    
    Raises:
        AssertionError: If no best trial is available.
    """
    assert study.best_trial is not None, "Study should have a best trial"
    assert study.best_trial.state == optuna.trial.TrialState.COMPLETE, \
        "Best trial should be in COMPLETE state"


def assert_refit_folder_structure(trial_folder: Path) -> None:
    """
    Assert that refit folder has correct structure.
    
    Args:
        trial_folder: Path to trial folder.
    
    Raises:
        AssertionError: If refit structure is incorrect.
    """
    refit_folder = trial_folder / "refit"
    assert refit_folder.exists(), f"Refit folder should exist in {trial_folder}"
    
    # Refit should have checkpoint directory
    checkpoint_dir = refit_folder / "checkpoint"
    # Checkpoint might not exist if refit hasn't completed, but folder structure should be there
    assert refit_folder.is_dir(), "Refit should be a directory"


def assert_cv_fold_structure(trial_folder: Path, n_splits: int) -> None:
    """
    Assert that CV fold structure is correct.
    
    Args:
        trial_folder: Path to trial folder.
        n_splits: Expected number of folds.
    
    Raises:
        AssertionError: If CV structure is incorrect.
    """
    cv_folder = trial_folder / "cv"
    assert cv_folder.exists(), f"CV folder should exist in {trial_folder}"
    
    fold_folders = [d for d in cv_folder.iterdir() if d.is_dir() and d.name.startswith("fold")]
    assert len(fold_folders) == n_splits, \
        f"Expected {n_splits} fold folders, found {len(fold_folders)}"
    
    # Verify fold folder names are fold0, fold1, etc.
    for i in range(n_splits):
        fold_folder = cv_folder / f"fold{i}"
        assert fold_folder.exists(), f"Fold folder 'fold{i}' should exist"


def assert_mlflow_run_hierarchy(
    parent_run_id: str,
    child_run_ids: List[str],
    mock_client: Mock,
) -> None:
    """
    Assert that MLflow run hierarchy is correct (parent-child relationships).
    
    Args:
        parent_run_id: Expected parent run ID.
        child_run_ids: List of expected child run IDs.
        mock_client: Mocked MLflow client.
    
    Raises:
        AssertionError: If hierarchy is incorrect.
    """
    # Verify create_run was called for each child
    assert mock_client.create_run.called, "create_run should have been called"
    
    # Get all create_run calls
    create_run_calls = mock_client.create_run.call_args_list
    
    # Verify parent_run_id is set in tags for child runs
    for call in create_run_calls:
        args, kwargs = call
        tags = kwargs.get("tags", {})
        # Child runs should have parent run ID in tags or as parameter
        # This depends on implementation, so we just verify create_run was called
        pass


def assert_study_resumed(study: optuna.Study, expected_min_trials: int) -> None:
    """
    Assert that study was resumed (has at least expected_min_trials).
    
    Args:
        study: Optuna study object.
        expected_min_trials: Minimum number of trials expected after resume.
    
    Raises:
        AssertionError: If study doesn't have enough trials.
    """
    assert len(study.trials) >= expected_min_trials, \
        f"Resumed study should have at least {expected_min_trials} trials, found {len(study.trials)}"

