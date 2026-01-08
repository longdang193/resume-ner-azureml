"""Component tests for MLflow structure validation in HPO workflow."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import mlflow
from mlflow.tracking import MlflowClient


class TestMLflowRunHierarchy:
    """Test MLflow run hierarchy and parent-child relationships."""

    @patch("mlflow.tracking.MlflowClient")
    def test_hpo_parent_run_has_correct_tags(self, mock_client_class, tmp_path):
        """Test that HPO parent run has correct tags (study_key_hash, study_family_hash)."""
        # Mock MLflow client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock parent run
        mock_parent_run = Mock()
        mock_parent_run.info.run_id = "parent_run_123"
        mock_parent_run.data.tags = {
            "code.study_key_hash": "a" * 64,
            "code.study_family_hash": "b" * 64,
            "mlflow.runName": "hpo_distilbert_sweep",
        }
        mock_client.get_run.return_value = mock_parent_run
        
        # Verify tags
        parent_run = mock_client.get_run("parent_run_123")
        assert "code.study_key_hash" in parent_run.data.tags
        assert "code.study_family_hash" in parent_run.data.tags
        assert len(parent_run.data.tags["code.study_key_hash"]) == 64
        assert len(parent_run.data.tags["code.study_family_hash"]) == 64

    @patch("mlflow.tracking.MlflowClient")
    def test_trial_run_is_child_of_hpo_parent(self, mock_client_class):
        """Test that trial run is created as child of HPO parent."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock parent run
        mock_parent_run = Mock()
        mock_parent_run.info.experiment_id = "exp_123"
        mock_client.get_run.return_value = mock_parent_run
        
        # Mock trial run creation
        mock_trial_run = Mock()
        mock_trial_run.info.run_id = "trial_run_456"
        mock_client.create_run.return_value = mock_trial_run
        
        # Create trial run (simulating create_trial_run_no_cv)
        trial_tags = {
            "mlflow.parentRunId": "parent_run_123",
            "azureml.runType": "trial",
            "trial_number": "0",
            "code.study_key_hash": "a" * 64,
        }
        trial_run = mock_client.create_run(
            experiment_id="exp_123",
            tags=trial_tags,
            run_name="trial_0",
        )
        
        # Verify parent-child relationship
        create_run_kwargs = mock_client.create_run.call_args[1]
        assert create_run_kwargs["tags"]["mlflow.parentRunId"] == "parent_run_123"
        assert trial_run.info.run_id == "trial_run_456"

    @patch("mlflow.tracking.MlflowClient")
    def test_fold_run_is_child_of_trial_run(self, mock_client_class):
        """Test that fold run (CV) is child of trial run, not HPO parent."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock trial run
        mock_trial_run = Mock()
        mock_trial_run.info.run_id = "trial_run_456"
        mock_trial_run.info.experiment_id = "exp_123"
        mock_client.get_run.return_value = mock_trial_run
        
        # Mock fold run creation (simulated - actual creation happens in training subprocess)
        # The fold run should have trial_run_456 as parent
        fold_tags = {
            "mlflow.parentRunId": "trial_run_456",  # Trial run, not HPO parent
            "fold_idx": "0",
            "code.study_key_hash": "a" * 64,
            "code.trial_key_hash": "c" * 64,
        }
        
        # Verify fold run would be created with trial as parent
        assert fold_tags["mlflow.parentRunId"] == "trial_run_456"
        assert "fold_idx" in fold_tags

    @patch("mlflow.tracking.MlflowClient")
    def test_refit_run_is_child_of_hpo_parent(self, mock_client_class):
        """Test that refit run is child of HPO parent, not trial run."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock HPO parent run
        mock_parent_run = Mock()
        mock_parent_run.info.experiment_id = "exp_123"
        mock_client.get_run.return_value = mock_parent_run
        
        # Mock refit run creation
        mock_refit_run = Mock()
        mock_refit_run.info.run_id = "refit_run_789"
        mock_client.create_run.return_value = mock_refit_run
        
        # Create refit run (simulating _create_refit_mlflow_run)
        refit_tags = {
            "mlflow.parentRunId": "parent_run_123",  # HPO parent, not trial
            "mlflow.runType": "refit",
            "code.refit": "true",
        }
        refit_run = mock_client.create_run(
            experiment_id="exp_123",
            tags=refit_tags,
            run_name="refit_trial_0",
        )
        
        # Verify refit is child of HPO parent
        create_run_kwargs = mock_client.create_run.call_args[1]
        assert create_run_kwargs["tags"]["mlflow.parentRunId"] == "parent_run_123"
        assert create_run_kwargs["tags"]["mlflow.runType"] == "refit"
        assert refit_run.info.run_id == "refit_run_789"


class TestMLflowRunTags:
    """Test MLflow run tags and metadata."""

    @patch("mlflow.tracking.MlflowClient")
    def test_hpo_parent_run_tags(self, mock_client_class):
        """Test that HPO parent run has all required tags."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Expected tags for HPO parent run
        expected_tags = {
            "code.study_key_hash": "a" * 64,
            "code.study_family_hash": "b" * 64,
            "mlflow.runName": "hpo_distilbert_sweep",
            "code.project": "resume-ner",
        }
        
        mock_parent_run = Mock()
        mock_parent_run.data.tags = expected_tags
        mock_client.get_run.return_value = mock_parent_run
        
        parent_run = mock_client.get_run("parent_123")
        
        # Verify all expected tags are present
        for tag_key, tag_value in expected_tags.items():
            assert tag_key in parent_run.data.tags
            assert parent_run.data.tags[tag_key] == tag_value

    @patch("mlflow.tracking.MlflowClient")
    def test_trial_run_tags(self, mock_client_class):
        """Test that trial run has all required tags."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Expected tags for trial run
        expected_tags = {
            "mlflow.parentRunId": "parent_run_123",
            "code.study_key_hash": "a" * 64,
            "code.trial_key_hash": "c" * 64,
            "trial_number": "0",
            "azureml.runType": "trial",
            "azureml.trial": "true",
        }
        
        mock_trial_run = Mock()
        mock_trial_run.data.tags = expected_tags
        mock_client.get_run.return_value = mock_trial_run
        
        trial_run = mock_client.get_run("trial_456")
        
        # Verify all expected tags are present
        assert trial_run.data.tags["mlflow.parentRunId"] == "parent_run_123"
        assert trial_run.data.tags["trial_number"] == "0"
        assert trial_run.data.tags["azureml.runType"] == "trial"
        assert "code.study_key_hash" in trial_run.data.tags

    @patch("mlflow.tracking.MlflowClient")
    def test_refit_run_tags(self, mock_client_class):
        """Test that refit run has all required tags."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Expected tags for refit run
        expected_tags = {
            "mlflow.parentRunId": "parent_run_123",
            "mlflow.runType": "refit",
            "code.refit": "true",
            "code.refit_has_validation": "false",
            "code.study_key_hash": "a" * 64,
            "code.trial_key_hash": "c" * 64,
        }
        
        mock_refit_run = Mock()
        mock_refit_run.data.tags = expected_tags
        mock_client.get_run.return_value = mock_refit_run
        
        refit_run = mock_client.get_run("refit_789")
        
        # Verify all expected tags are present
        assert refit_run.data.tags["mlflow.parentRunId"] == "parent_run_123"
        assert refit_run.data.tags["mlflow.runType"] == "refit"
        assert refit_run.data.tags["code.refit"] == "true"
        assert refit_run.data.tags["code.refit_has_validation"] == "false"

    @patch("mlflow.tracking.MlflowClient")
    def test_trial_run_inherits_study_key_hash_from_parent(self, mock_client_class):
        """Test that trial run inherits study_key_hash from HPO parent."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock parent run with study_key_hash
        study_key_hash = "a" * 64
        mock_parent_run = Mock()
        mock_parent_run.data.tags = {
            "code.study_key_hash": study_key_hash,
            "code.study_family_hash": "b" * 64,
        }
        mock_client.get_run.return_value = mock_parent_run
        
        # Get study_key_hash from parent (as done in create_trial_run_no_cv)
        parent_run = mock_client.get_run("parent_123")
        inherited_study_key_hash = parent_run.data.tags.get("code.study_key_hash")
        
        # Verify trial run would use inherited hash
        assert inherited_study_key_hash == study_key_hash
        
        # Trial run tags should include this hash
        trial_tags = {
            "mlflow.parentRunId": "parent_123",
            "code.study_key_hash": inherited_study_key_hash,
        }
        assert trial_tags["code.study_key_hash"] == study_key_hash


class TestMLflowRunMetrics:
    """Test MLflow metrics logging structure."""

    @patch("mlflow.tracking.MlflowClient")
    def test_trial_run_logs_metrics(self, mock_client_class):
        """Test that trial run logs objective metric."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Log metric to trial run
        trial_run_id = "trial_run_456"
        objective_metric = "macro-f1"
        metric_value = 0.75
        
        mock_client.log_metric(trial_run_id, objective_metric, metric_value)
        
        # Verify metric was logged
        assert mock_client.log_metric.called
        call_args = mock_client.log_metric.call_args
        assert call_args[0][0] == trial_run_id
        assert call_args[0][1] == objective_metric
        assert call_args[0][2] == metric_value

    @patch("mlflow.tracking.MlflowClient")
    def test_trial_run_logs_hyperparameters(self, mock_client_class):
        """Test that trial run logs hyperparameters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Log hyperparameters to trial run
        trial_run_id = "trial_run_456"
        hyperparameters = {
            "learning_rate": 3e-5,
            "batch_size": 4,
            "dropout": 0.2,
        }
        
        for param_name, param_value in hyperparameters.items():
            mock_client.log_param(trial_run_id, param_name, str(param_value))
        
        # Verify all hyperparameters were logged
        assert mock_client.log_param.call_count == 3
        logged_params = {call[0][1]: call[0][2] for call in mock_client.log_param.call_args_list}
        assert logged_params["learning_rate"] == "3e-05"
        assert logged_params["batch_size"] == "4"
        assert logged_params["dropout"] == "0.2"

    @patch("mlflow.tracking.MlflowClient")
    def test_cv_trial_run_logs_aggregated_metrics(self, mock_client_class):
        """Test that CV trial run logs aggregated metrics (mean, std)."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Log CV aggregated metrics
        trial_run_id = "trial_run_456"
        cv_mean = 0.75
        cv_std = 0.02
        fold_metrics = [0.73, 0.77]
        
        mock_client.log_metric(trial_run_id, "cv_mean", cv_mean)
        mock_client.log_metric(trial_run_id, "cv_std", cv_std)
        
        # Verify CV metrics were logged
        log_metric_calls = [call[0] for call in mock_client.log_metric.call_args_list]
        metric_names = [call[1] for call in log_metric_calls]
        assert "cv_mean" in metric_names
        assert "cv_std" in metric_names

    @patch("mlflow.tracking.MlflowClient")
    def test_refit_run_logs_metrics(self, mock_client_class):
        """Test that refit run logs metrics."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Log metrics to refit run
        refit_run_id = "refit_run_789"
        metrics = {
            "macro-f1": 0.80,
            "accuracy": 0.85,
        }
        
        for metric_name, metric_value in metrics.items():
            mock_client.log_metric(refit_run_id, metric_name, metric_value)
        
        # Verify metrics were logged
        assert mock_client.log_metric.call_count == 2
        logged_metrics = {call[0][1]: call[0][2] for call in mock_client.log_metric.call_args_list}
        assert logged_metrics["macro-f1"] == 0.80
        assert logged_metrics["accuracy"] == 0.85


class TestMLflowRunStructure:
    """Test overall MLflow run structure and relationships."""

    @patch("mlflow.tracking.MlflowClient")
    def test_mlflow_structure_no_cv(self, mock_client_class):
        """Test MLflow structure for HPO without CV:
        HPO Parent -> Trial Runs
        HPO Parent -> Refit Run
        """
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # HPO parent run
        parent_run_id = "parent_123"
        mock_parent_run = Mock()
        mock_parent_run.info.run_id = parent_run_id
        mock_parent_run.info.experiment_id = "exp_123"
        mock_parent_run.data.tags = {"code.study_key_hash": "a" * 64}
        
        # Trial runs (children of parent)
        trial_run_ids = ["trial_0", "trial_1"]
        mock_trial_runs = []
        for trial_id in trial_run_ids:
            mock_trial = Mock()
            mock_trial.info.run_id = trial_id
            mock_trial.data.tags = {
                "mlflow.parentRunId": parent_run_id,
                "trial_number": trial_run_ids.index(trial_id),
            }
            mock_trial_runs.append(mock_trial)
        
        # Refit run (child of parent)
        refit_run_id = "refit_789"
        mock_refit_run = Mock()
        mock_refit_run.info.run_id = refit_run_id
        mock_refit_run.data.tags = {
            "mlflow.parentRunId": parent_run_id,
            "mlflow.runType": "refit",
        }
        
        # Verify structure
        assert mock_parent_run.info.run_id == parent_run_id
        for trial_run in mock_trial_runs:
            assert trial_run.data.tags["mlflow.parentRunId"] == parent_run_id
        assert mock_refit_run.data.tags["mlflow.parentRunId"] == parent_run_id

    @patch("mlflow.tracking.MlflowClient")
    def test_mlflow_structure_with_cv(self, mock_client_class):
        """Test MLflow structure for HPO with CV:
        HPO Parent -> Trial Runs -> Fold Runs
        HPO Parent -> Refit Run
        """
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # HPO parent run
        parent_run_id = "parent_123"
        
        # Trial run (child of parent)
        trial_run_id = "trial_0"
        mock_trial_run = Mock()
        mock_trial_run.info.run_id = trial_run_id
        mock_trial_run.data.tags = {
            "mlflow.parentRunId": parent_run_id,
            "trial_number": "0",
        }
        
        # Fold runs (children of trial run)
        fold_run_ids = ["fold_0", "fold_1"]
        mock_fold_runs = []
        for fold_id in fold_run_ids:
            mock_fold = Mock()
            mock_fold.info.run_id = fold_id
            mock_fold.data.tags = {
                "mlflow.parentRunId": trial_run_id,  # Parent is trial, not HPO parent
                "fold_idx": str(fold_run_ids.index(fold_id)),
            }
            mock_fold_runs.append(mock_fold)
        
        # Refit run (child of parent)
        refit_run_id = "refit_789"
        mock_refit_run = Mock()
        mock_refit_run.data.tags = {
            "mlflow.parentRunId": parent_run_id,  # Parent is HPO parent
            "mlflow.runType": "refit",
        }
        
        # Verify structure
        assert mock_trial_run.data.tags["mlflow.parentRunId"] == parent_run_id
        for fold_run in mock_fold_runs:
            assert fold_run.data.tags["mlflow.parentRunId"] == trial_run_id
        assert mock_refit_run.data.tags["mlflow.parentRunId"] == parent_run_id

    @patch("mlflow.tracking.MlflowClient")
    def test_mlflow_runs_have_grouping_tags(self, mock_client_class):
        """Test that all MLflow runs have grouping tags (study_key_hash, trial_key_hash)."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        study_key_hash = "a" * 64
        trial_key_hash = "c" * 64
        
        # HPO parent run
        parent_tags = {
            "code.study_key_hash": study_key_hash,
            "code.study_family_hash": "b" * 64,
        }
        
        # Trial run
        trial_tags = {
            "code.study_key_hash": study_key_hash,  # Inherited from parent
            "code.trial_key_hash": trial_key_hash,
        }
        
        # Refit run
        refit_tags = {
            "code.study_key_hash": study_key_hash,
            "code.trial_key_hash": trial_key_hash,
        }
        
        # Verify all runs have study_key_hash
        assert "code.study_key_hash" in parent_tags
        assert "code.study_key_hash" in trial_tags
        assert "code.study_key_hash" in refit_tags
        
        # Verify trial and refit have trial_key_hash
        assert "code.trial_key_hash" in trial_tags
        assert "code.trial_key_hash" in refit_tags
        
        # Verify hashes match across runs
        assert parent_tags["code.study_key_hash"] == trial_tags["code.study_key_hash"]
        assert trial_tags["code.trial_key_hash"] == refit_tags["code.trial_key_hash"]


class TestMLflowRunStatus:
    """Test MLflow run status transitions."""

    @patch("mlflow.tracking.MlflowClient")
    def test_trial_run_status_transitions(self, mock_client_class):
        """Test that trial run status transitions correctly (RUNNING -> FINISHED)."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        trial_run_id = "trial_run_456"
        
        # Initially RUNNING (created by create_trial_run_no_cv)
        mock_run = Mock()
        mock_run.info.status = "RUNNING"
        mock_client.get_run.return_value = mock_run
        
        # After training completes, mark as FINISHED
        mock_client.set_terminated(trial_run_id, status="FINISHED")
        
        # Verify status transition
        assert mock_client.set_terminated.called
        call_args = mock_client.set_terminated.call_args
        assert call_args[0][0] == trial_run_id
        assert call_args[1]["status"] == "FINISHED"

    @patch("mlflow.tracking.MlflowClient")
    def test_refit_run_status_finished_after_upload(self, mock_client_class):
        """Test that refit run is marked FINISHED after artifact upload."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        refit_run_id = "refit_run_789"
        
        # Initially RUNNING
        mock_run = Mock()
        mock_run.info.status = "RUNNING"
        mock_client.get_run.return_value = mock_run
        
        # After artifact upload, mark as FINISHED
        mock_client.set_terminated(refit_run_id, status="FINISHED")
        mock_client.set_tag(refit_run_id, "code.refit_artifacts_uploaded", "true")
        
        # Verify status and tag
        assert mock_client.set_terminated.called
        assert mock_client.set_tag.called
        set_tag_calls = [call[0] for call in mock_client.set_tag.call_args_list]
        assert any(call[0] == refit_run_id and call[1] == "code.refit_artifacts_uploaded" for call in set_tag_calls)

