"""Component tests for trial execution with and without CV."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from orchestration.jobs.hpo.local.trial.execution import TrialExecutor
from orchestration.jobs.hpo.local.cv.orchestrator import run_training_trial_with_cv


class TestTrialExecutionNoCV:
    """Test trial execution without CV (k_fold.enabled=false)."""

    @patch("orchestration.jobs.hpo.local.trial.execution.subprocess.run")
    @patch("orchestration.jobs.hpo.local.trial.execution.Path.exists")
    def test_trial_execution_no_cv_creates_single_run(self, mock_exists, mock_subprocess, tmp_path):
        """Test that trial execution without CV creates single trial run (no fold runs)."""
        # Setup
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        output_dir = tmp_path / "outputs" / "hpo" / "trial_0"
        output_dir.mkdir(parents=True)
        
        # Mock Path.exists to return True for training module check
        mock_exists.return_value = True
        
        # Mock subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # Create metrics file
        metrics_file = output_dir / "metrics.json"
        metrics_file.write_text(json.dumps({"macro-f1": 0.75}))
        
        executor = TrialExecutor(
            config_dir=config_dir,
            mlflow_experiment_name="test",
        )
        
        trial_params = {
            "learning_rate": 3e-5,
            "batch_size": 4,
            "backbone": "distilbert",
            "trial_number": 0,
        }
        
        # Execute trial
        metric_value = executor.execute(
            trial_params=trial_params,
            dataset_path=str(tmp_path / "dataset"),
            backbone="distilbert",
            output_dir=output_dir,
            train_config={"epochs": 1},
            objective_metric="macro-f1",
        )
        
        # Verify metric was returned
        assert metric_value == 0.75
        assert mock_subprocess.called

    @patch("orchestration.jobs.hpo.local.trial.execution.subprocess.run")
    @patch("orchestration.jobs.hpo.local.trial.execution.Path.exists")
    def test_trial_execution_no_cv_output_path(self, mock_exists, mock_subprocess, tmp_path):
        """Test that trial execution without CV uses correct output path."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create v2 path structure
        output_dir = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / "study-abc12345" / "trial-def67890"
        output_dir.mkdir(parents=True)
        
        mock_exists.return_value = True
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        metrics_file = output_dir / "metrics.json"
        metrics_file.write_text(json.dumps({"macro-f1": 0.80}))
        
        executor = TrialExecutor(
            config_dir=config_dir,
            mlflow_experiment_name="test",
        )
        
        metric_value = executor.execute(
            trial_params={"learning_rate": 3e-5, "backbone": "distilbert", "trial_number": 0},
            dataset_path=str(tmp_path / "dataset"),
            backbone="distilbert",
            output_dir=output_dir,
            train_config={},
            objective_metric="macro-f1",
        )
        
        assert metric_value == 0.80
        # Verify output directory structure
        assert "study-abc12345" in str(output_dir)
        assert "trial-def67890" in str(output_dir)

    @patch("orchestration.jobs.hpo.local.trial.execution.subprocess.run")
    @patch("orchestration.jobs.hpo.local.trial.execution.Path.exists")
    def test_trial_execution_no_cv_metrics_file(self, mock_exists, mock_subprocess, tmp_path):
        """Test that metrics.json is read from trial output directory."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        output_dir = tmp_path / "outputs" / "hpo" / "trial_0"
        output_dir.mkdir(parents=True)
        
        mock_exists.return_value = True
        mock_result = Mock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        # Create metrics file with multiple metrics
        metrics = {
            "macro-f1": 0.75,
            "micro-f1": 0.78,
            "accuracy": 0.82,
        }
        metrics_file = output_dir / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))
        
        executor = TrialExecutor(
            config_dir=config_dir,
            mlflow_experiment_name="test",
        )
        
        metric_value = executor.execute(
            trial_params={"backbone": "distilbert", "trial_number": 0},
            dataset_path=str(tmp_path / "dataset"),
            backbone="distilbert",
            output_dir=output_dir,
            train_config={},
            objective_metric="macro-f1",
        )
        
        assert metric_value == 0.75


class TestTrialExecutionWithCV:
    """Test trial execution with CV (k_fold.enabled=true, n_splits=2)."""

    @patch("orchestration.jobs.hpo.local.cv.orchestrator.run_training_trial")
    @patch("orchestration.jobs.hpo.local.cv.orchestrator.mlflow")
    def test_trial_execution_with_cv_creates_nested_runs(self, mock_mlflow, mock_run_trial, tmp_path):
        """Test that trial execution with CV creates trial run and fold runs."""
        # Setup
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        output_dir = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / "study-abc12345"
        output_dir.mkdir(parents=True)
        
        # Mock MLflow - provide actual string hash values
        study_key_hash = "a" * 64
        study_family_hash = "b" * 64
        mock_parent_run = Mock()
        mock_parent_run.data.tags.get = Mock(side_effect=lambda key, default=None: {
            "code.study_key_hash": study_key_hash,
            "code.study_family_hash": study_family_hash,
        }.get(key, default))
        
        mock_trial_run = Mock()
        mock_trial_run.info.run_id = "trial_run_123"
        mock_client = Mock()
        mock_client.create_run.return_value = mock_trial_run
        mock_client.get_run.return_value = mock_parent_run
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_trial_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        
        # Mock fold executions
        mock_run_trial.side_effect = [0.75, 0.80]  # Two folds
        
        # Create fold splits
        fold_splits = [
            ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),  # Fold 0
            ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4]),  # Fold 1
        ]
        
        fold_splits_file = output_dir / "fold_splits.json"
        from training.cv_utils import save_fold_splits
        save_fold_splits(fold_splits, fold_splits_file, {"k": 2, "random_seed": 42})
        
        trial_params = {
            "learning_rate": 3e-5,
            "batch_size": 4,
            "backbone": "distilbert",
            "trial_number": 0,
        }
        
        # Execute trial with CV
        avg_metric, fold_metrics = run_training_trial_with_cv(
            trial_params=trial_params,
            dataset_path=str(tmp_path / "dataset"),
            config_dir=config_dir,
            backbone="distilbert",
            output_dir=output_dir,
            train_config={"epochs": 1},
            mlflow_experiment_name="test",
            objective_metric="macro-f1",
            fold_splits=fold_splits,
            fold_splits_file=fold_splits_file,
            hpo_parent_run_id="parent_123",
        )
        
        # Verify average metric
        assert avg_metric == pytest.approx(0.775)  # (0.75 + 0.80) / 2
        assert len(fold_metrics) == 2
        assert fold_metrics == [0.75, 0.80]
        
        # Verify trial run was created
        assert mock_client.create_run.called

    @patch("orchestration.jobs.hpo.local.cv.orchestrator.run_training_trial")
    @patch("orchestration.jobs.hpo.local.cv.orchestrator.mlflow")
    def test_trial_execution_with_cv_creates_fold_runs(self, mock_mlflow, mock_run_trial, tmp_path):
        """Test that each fold creates a fold-level MLflow run (child of trial run)."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        output_dir = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / "study-abc12345"
        output_dir.mkdir(parents=True)
        
        # Provide actual hash values
        study_key_hash = "a" * 64
        study_family_hash = "b" * 64
        mock_parent_run = Mock()
        mock_parent_run.data.tags.get = Mock(side_effect=lambda key, default=None: {
            "code.study_key_hash": study_key_hash,
            "code.study_family_hash": study_family_hash,
        }.get(key, default))
        
        # Mock MLflow runs
        mock_trial_run = Mock()
        mock_trial_run.info.run_id = "trial_run_123"
        mock_fold_run_0 = Mock()
        mock_fold_run_0.info.run_id = "fold_run_0"
        mock_fold_run_1 = Mock()
        mock_fold_run_1.info.run_id = "fold_run_1"
        
        mock_client = Mock()
        mock_client.create_run.return_value = mock_trial_run
        mock_client.get_run.return_value = mock_parent_run
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        # Mock start_run to return different runs for trial and folds
        run_contexts = [mock_trial_run, mock_fold_run_0, mock_fold_run_1]
        mock_mlflow.start_run.side_effect = [
            MagicMock(__enter__=Mock(return_value=run_contexts[i]), __exit__=Mock(return_value=None))
            for i in range(3)
        ]
        
        mock_run_trial.side_effect = [0.75, 0.80]
        
        fold_splits = [
            ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
            ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4]),
        ]
        
        fold_splits_file = output_dir / "fold_splits.json"
        from training.cv_utils import save_fold_splits
        save_fold_splits(fold_splits, fold_splits_file, {"k": 2})
        
        avg_metric, fold_metrics = run_training_trial_with_cv(
            trial_params={"learning_rate": 3e-5, "backbone": "distilbert", "trial_number": 0},
            dataset_path=str(tmp_path / "dataset"),
            config_dir=config_dir,
            backbone="distilbert",
            output_dir=output_dir,
            train_config={},
            mlflow_experiment_name="test",
            objective_metric="macro-f1",
            fold_splits=fold_splits,
            fold_splits_file=fold_splits_file,
            hpo_parent_run_id="parent_123",
            study_key_hash=study_key_hash,
            study_family_hash=study_family_hash,
        )
        
        # Verify trial was executed for each fold
        assert mock_run_trial.call_count == 2  # Two folds
        # Verify trial run was created
        assert mock_client.create_run.called

    @patch("orchestration.jobs.hpo.local.cv.orchestrator.run_training_trial")
    @patch("orchestration.jobs.hpo.local.cv.orchestrator.mlflow")
    def test_trial_execution_with_cv_aggregates_metrics(self, mock_mlflow, mock_run_trial, tmp_path):
        """Test that CV trial aggregates metrics across folds (average)."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        output_dir = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / "study-abc12345"
        output_dir.mkdir(parents=True)
        
        # Provide actual hash values
        study_key_hash = "a" * 64
        study_family_hash = "b" * 64
        mock_parent_run = Mock()
        mock_parent_run.data.tags.get = Mock(side_effect=lambda key, default=None: {
            "code.study_key_hash": study_key_hash,
            "code.study_family_hash": study_family_hash,
        }.get(key, default))
        
        # Mock MLflow
        mock_trial_run = Mock()
        mock_trial_run.info.run_id = "trial_run_123"
        mock_client = Mock()
        mock_client.create_run.return_value = mock_trial_run
        mock_client.get_run.return_value = mock_parent_run
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_trial_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        
        # Mock different metrics for each fold
        mock_run_trial.side_effect = [0.70, 0.85]  # Fold 0: 0.70, Fold 1: 0.85
        
        fold_splits = [
            ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
            ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4]),
        ]
        
        fold_splits_file = output_dir / "fold_splits.json"
        from training.cv_utils import save_fold_splits
        save_fold_splits(fold_splits, fold_splits_file, {"k": 2})
        
        avg_metric, fold_metrics = run_training_trial_with_cv(
            trial_params={"backbone": "distilbert", "trial_number": 0},
            dataset_path=str(tmp_path / "dataset"),
            config_dir=config_dir,
            backbone="distilbert",
            output_dir=output_dir,
            train_config={},
            mlflow_experiment_name="test",
            objective_metric="macro-f1",
            fold_splits=fold_splits,
            fold_splits_file=fold_splits_file,
            hpo_parent_run_id="parent_123",
            study_key_hash=study_key_hash,
            study_family_hash=study_family_hash,
        )
        
        # Verify aggregation
        assert avg_metric == pytest.approx(0.775)  # (0.70 + 0.85) / 2
        assert fold_metrics == [0.70, 0.85]

    @patch("orchestration.jobs.hpo.local.cv.orchestrator.run_training_trial")
    @patch("orchestration.jobs.hpo.local.cv.orchestrator.mlflow")
    def test_trial_execution_with_cv_output_paths(self, mock_mlflow, mock_run_trial, tmp_path):
        """Test that CV trial creates fold-specific output directories."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Create v2 path structure
        study_dir = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / "study-abc12345"
        study_dir.mkdir(parents=True)
        
        # Provide actual hash values
        study_key_hash = "a" * 64
        study_family_hash = "b" * 64
        mock_parent_run = Mock()
        mock_parent_run.data.tags.get = Mock(side_effect=lambda key, default=None: {
            "code.study_key_hash": study_key_hash,
            "code.study_family_hash": study_family_hash,
        }.get(key, default))
        
        # Mock MLflow
        mock_trial_run = Mock()
        mock_trial_run.info.run_id = "trial_run_123"
        mock_client = Mock()
        mock_client.create_run.return_value = mock_trial_run
        mock_client.get_run.return_value = mock_parent_run
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_trial_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        
        mock_run_trial.side_effect = [0.75, 0.80]
        
        fold_splits = [
            ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
            ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4]),
        ]
        
        fold_splits_file = study_dir / "fold_splits.json"
        from training.cv_utils import save_fold_splits
        save_fold_splits(fold_splits, fold_splits_file, {"k": 2})
        
        # Capture output_dir passed to run_training_trial
        captured_output_dirs = []
        original_run_trial = mock_run_trial
        
        def capture_output_dir(*args, **kwargs):
            if "output_dir" in kwargs:
                captured_output_dirs.append(kwargs["output_dir"])
            return original_run_trial.side_effect.pop(0) if original_run_trial.side_effect else 0.75
        
        mock_run_trial.side_effect = [0.75, 0.80]
        mock_run_trial.side_effect = capture_output_dir
        
        # Reset side_effect properly
        mock_run_trial.side_effect = [0.75, 0.80]
        
        avg_metric, fold_metrics = run_training_trial_with_cv(
            trial_params={"backbone": "distilbert", "trial_number": 0},
            dataset_path=str(tmp_path / "dataset"),
            config_dir=config_dir,
            backbone="distilbert",
            output_dir=study_dir,
            train_config={},
            mlflow_experiment_name="test",
            objective_metric="macro-f1",
            fold_splits=fold_splits,
            fold_splits_file=fold_splits_file,
            hpo_parent_run_id="parent_123",
            study_key_hash=study_key_hash,
            study_family_hash=study_family_hash,
        )
        
        # Verify trial was executed for each fold
        assert mock_run_trial.call_count == 2
        # Each call should have fold_idx parameter
        for call in mock_run_trial.call_args_list:
            assert "fold_idx" in call.kwargs or len(call.args) > 0

    @patch("orchestration.jobs.hpo.local.cv.orchestrator.run_training_trial")
    @patch("orchestration.jobs.hpo.local.cv.orchestrator.mlflow")
    def test_trial_execution_with_cv_smoke_yaml_params(self, mock_mlflow, mock_run_trial, tmp_path):
        """Test CV trial execution with smoke.yaml parameters (n_splits=2, random_seed=42)."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        output_dir = tmp_path / "outputs" / "hpo" / "local" / "distilbert" / "study-abc12345"
        output_dir.mkdir(parents=True)
        
        # Provide actual hash values
        study_key_hash = "a" * 64
        study_family_hash = "b" * 64
        mock_parent_run = Mock()
        mock_parent_run.data.tags.get = Mock(side_effect=lambda key, default=None: {
            "code.study_key_hash": study_key_hash,
            "code.study_family_hash": study_family_hash,
        }.get(key, default))
        
        # Mock MLflow
        mock_trial_run = Mock()
        mock_trial_run.info.run_id = "trial_run_123"
        mock_client = Mock()
        mock_client.create_run.return_value = mock_trial_run
        mock_client.get_run.return_value = mock_parent_run
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_trial_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        
        # smoke.yaml: n_splits=2
        fold_splits = [
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),  # Fold 0
            ([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),  # Fold 1
        ]
        
        fold_splits_file = output_dir / "fold_splits.json"
        from training.cv_utils import save_fold_splits
        save_fold_splits(fold_splits, fold_splits_file, {
            "k": 2,
            "random_seed": 42,  # smoke.yaml value
            "shuffle": True,
            "stratified": True,
        })
        
        mock_run_trial.side_effect = [0.75, 0.80]
        
        avg_metric, fold_metrics = run_training_trial_with_cv(
            trial_params={"backbone": "distilbert", "trial_number": 0},
            dataset_path=str(tmp_path / "dataset"),
            config_dir=config_dir,
            backbone="distilbert",
            output_dir=output_dir,
            train_config={},
            mlflow_experiment_name="test",
            objective_metric="macro-f1",
            fold_splits=fold_splits,
            fold_splits_file=fold_splits_file,
            hpo_parent_run_id="parent_123",
            study_key_hash=study_key_hash,
            study_family_hash=study_family_hash,
        )
        
        # Verify 2 folds were executed
        assert len(fold_metrics) == 2
        assert mock_run_trial.call_count == 2
        assert avg_metric == pytest.approx(0.775)

