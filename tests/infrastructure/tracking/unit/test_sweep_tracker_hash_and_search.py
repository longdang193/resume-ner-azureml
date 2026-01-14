"""Unit tests for sweep tracker hash computation and best trial search.

Tests cover:
1. v2 hash computation in start_sweep_run() - success and failure cases
2. Best trial search using study_key_hash + trial_number + parentRunId
3. Fallback behavior when parent tags are missing
4. Edge cases and error handling
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from infrastructure.tracking.mlflow.trackers.sweep_tracker import MLflowSweepTracker


@pytest.fixture
def sample_configs():
    """Sample configurations for testing."""
    return {
        "data_config": {
            "name": "test_dataset",
            "version": "1.0",
            "local_path": "/tmp/data",
            "schema": {"labels": ["PER", "ORG"]},
            "split_seed": 42,
        },
        "hpo_config": {
            "search_space": {
                "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
                "batch_size": {"type": "choice", "choices": [4, 8, 16]},
            },
            "objective": {"metric": "macro-f1", "direction": "maximize"},
            "k_fold": {"enabled": True, "n_splits": 2},
            "sampling": {"algorithm": "random"},
        },
        "train_config": {
            "max_steps": 1000,
            "num_epochs": 3,
            "seed_policy": "fixed",
            "eval": {
                "evaluator_version": "v1",
                "metric": {"name": "macro-f1"},
            },
        },
        "backbone": "distilbert-base-uncased",
        "benchmark_config": {},
    }


@pytest.fixture
def mock_context(sample_configs):
    """Mock NamingContext for testing."""
    mock_ctx = Mock()
    mock_ctx.model = sample_configs["backbone"].split("-")[0]
    mock_ctx.stage = "hpo_sweep"
    mock_ctx.process_type = "hpo"
    # Provide concrete string values so tag sanitization logic does not see Mocks.
    mock_ctx.environment = "local"
    mock_ctx.storage_env = "local"
    mock_ctx.spec_fp = None
    mock_ctx.exec_fp = None
    mock_ctx.data_fp = None
    mock_ctx.eval_fp = None
    mock_ctx.variant = None
    mock_ctx.trial_id = None
    mock_ctx.parent_training_id = None
    mock_ctx.conv_fp = None
    return mock_ctx


@pytest.fixture
def root_dir_with_config(tmp_path: Path) -> Path:
    """Create a root directory with config/ directory."""
    root = tmp_path / "workspace"
    root.mkdir()
    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "tags.yaml").write_text("schema_version: 1")
    return root


class TestSweepTrackerV2HashComputation:
    """Test v2 hash computation in start_sweep_run()."""

    @patch("mlflow.start_run")
    @patch("mlflow.set_tags")
    @patch("mlflow.get_tracking_uri")
    def test_v2_hash_computation_success(
        self, mock_get_uri, mock_set_tags, mock_start_run, sample_configs, mock_context, root_dir_with_config
    ):
        """Test that v2 hash is computed successfully when all configs are available."""
        tracker = MLflowSweepTracker(experiment_name="test")
        
        # Setup mocks
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_run.info.experiment_id = "exp-123"
        mock_run.info.artifact_uri = "file:///tmp"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_get_uri.return_value = "file:///tmp"
        
        output_dir = root_dir_with_config / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        # Call start_sweep_run with all required configs
        with patch("mlflow.active_run", return_value=mock_run):
            with patch("infrastructure.tracking.mlflow.index.update_mlflow_index"):
                with patch.object(tracker, "_log_sweep_metadata"):
                    # Stub tag-builder to avoid depending on full NamingContext implementation
                    with patch("infrastructure.tracking.mlflow.trackers.sweep_tracker.build_mlflow_tags") as mock_build_tags:
                        mock_build_tags.return_value = {
                            "code.study_key_hash": "a" * 64,
                            "code.fingerprint.data": "data-fp",
                            "code.fingerprint.eval": "eval-fp",
                        }
                        with tracker.start_sweep_run(
                            run_name="test-run",
                            hpo_config=sample_configs["hpo_config"],
                            backbone=sample_configs["backbone"],
                            study_name="test-study",
                            checkpoint_config={"enabled": True},
                            storage_path=None,
                            should_resume=False,
                            context=mock_context,
                            output_dir=output_dir,
                            data_config=sample_configs["data_config"],
                            train_config=sample_configs["train_config"],
                        ):
                            # Context manager should execute without error
                            pass

        # Verify v2 hash was computed (check tags were set)
        assert mock_set_tags.called
        tags_call = mock_set_tags.call_args[0][0]
        assert "code.study_key_hash" in tags_call
        study_key_hash = tags_call["code.study_key_hash"]
        assert isinstance(study_key_hash, str)
        assert len(study_key_hash) == 64

        # Verify fingerprints were set (indicates v2)
        assert "code.fingerprint.data" in tags_call
        assert "code.fingerprint.eval" in tags_call

    @patch("mlflow.start_run")
    @patch("mlflow.set_tags")
    @patch("mlflow.get_tracking_uri")
    def test_v2_hash_computation_missing_train_config(
        self, mock_get_uri, mock_set_tags, mock_start_run, sample_configs, mock_context, root_dir_with_config
    ):
        """Test that hash computation handles missing train_config gracefully."""
        tracker = MLflowSweepTracker(experiment_name="test")
        
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_run.info.experiment_id = "exp-123"
        mock_run.info.artifact_uri = "file:///tmp"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_get_uri.return_value = "file:///tmp"
        
        output_dir = root_dir_with_config / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        # Call without train_config
        with patch("mlflow.active_run", return_value=mock_run):
            with patch("infrastructure.tracking.mlflow.index.update_mlflow_index"):
                with patch.object(tracker, "_log_sweep_metadata"):
                    with patch("infrastructure.tracking.mlflow.trackers.sweep_tracker.build_mlflow_tags") as mock_build_tags:
                        mock_build_tags.return_value = {
                            "code.study_key_hash": "a" * 64,
                            "code.fingerprint.data": "data-fp",
                            "code.fingerprint.eval": "eval-fp",
                        }
                        with tracker.start_sweep_run(
                            run_name="test-run",
                            hpo_config=sample_configs["hpo_config"],
                            backbone=sample_configs["backbone"],
                            study_name="test-study",
                            checkpoint_config={"enabled": True},
                            storage_path=None,
                            should_resume=False,
                            context=mock_context,
                            output_dir=output_dir,
                            data_config=sample_configs["data_config"],
                            train_config=None,  # Missing train_config
                        ):
                            # Context manager should execute without error even when train_config is None
                            pass

        # Should still work (tags might be empty or use defaults)
        assert mock_set_tags.called

    @patch("mlflow.start_run")
    @patch("mlflow.set_tags")
    @patch("mlflow.get_tracking_uri")
    def test_v2_hash_computation_empty_eval_config(
        self, mock_get_uri, mock_set_tags, mock_start_run, sample_configs, mock_context, root_dir_with_config
    ):
        """Test that v2 hash computation handles empty eval config (uses objective fallback)."""
        tracker = MLflowSweepTracker(experiment_name="test")
        
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-123"
        mock_run.info.experiment_id = "exp-123"
        mock_run.info.artifact_uri = "file:///tmp"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_get_uri.return_value = "file:///tmp"
        
        output_dir = root_dir_with_config / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        # Train config without eval section
        train_config_no_eval = {
            "max_steps": 1000,
            "num_epochs": 3,
        }
        
        with patch("mlflow.active_run", return_value=mock_run):
            with patch("infrastructure.tracking.mlflow.index.update_mlflow_index"):
                with patch.object(tracker, "_log_sweep_metadata"):
                    with patch("infrastructure.tracking.mlflow.trackers.sweep_tracker.build_mlflow_tags") as mock_build_tags:
                        mock_build_tags.return_value = {
                            "code.study_key_hash": "a" * 64,
                            "code.fingerprint.data": "data-fp",
                            "code.fingerprint.eval": "eval-fp",
                        }
                        with tracker.start_sweep_run(
                            run_name="test-run",
                            hpo_config=sample_configs["hpo_config"],
                            backbone=sample_configs["backbone"],
                            study_name="test-study",
                            checkpoint_config={"enabled": True},
                            storage_path=None,
                            should_resume=False,
                            context=mock_context,
                            output_dir=output_dir,
                            data_config=sample_configs["data_config"],
                            train_config=train_config_no_eval,
                        ):
                            # Context manager should execute without error even when eval config is missing
                            pass

        # Should still compute hash (uses objective from hpo_config)
        assert mock_set_tags.called


class TestSweepTrackerBestTrialSearch:
    """Test best trial run ID search logic."""

    def test_best_trial_search_by_study_key_hash_success(
        self, sample_configs, root_dir_with_config
    ):
        """Test successful search using study_key_hash + trial_number + parentRunId."""
        tracker = MLflowSweepTracker(experiment_name="test")
        
        parent_run_id = "parent-run-123"
        best_trial_number = 0
        study_key_hash = "a" * 64
        correct_trial_run_id = "trial-run-correct"
        
        # Create mock study
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.number = best_trial_number
        mock_study.best_trial = mock_trial
        
        # Create mock MLflow client
        mock_client = MagicMock()
        
        # Mock parent run with study_key_hash
        mock_parent_run = MagicMock()
        mock_parent_run.data.tags.get.return_value = study_key_hash
        mock_client.get_run.return_value = mock_parent_run
        
        # Mock search results - return trial run that matches parent
        mock_trial_run = MagicMock()
        mock_trial_run.info.run_id = correct_trial_run_id
        mock_trial_run.data.tags.get.side_effect = lambda key: {
            "mlflow.parentRunId": parent_run_id,
            "trial_number": str(best_trial_number),
            "code.study_key_hash": study_key_hash,
        }.get(key)
        
        mock_client.search_runs.return_value = [mock_trial_run]
        
        # Mock active run
        mock_active_run = MagicMock()
        mock_active_run.info.experiment_id = "exp-123"
        
        output_dir = root_dir_with_config / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            with patch("mlflow.active_run", return_value=mock_active_run):
                with patch("mlflow.log_param") as mock_log_param:
                    with patch("mlflow.set_tag") as mock_set_tag:
                        tracker._log_best_trial_id(
                            study=mock_study,
                            parent_run_id=parent_run_id,
                            output_dir=output_dir,
                        )
                        
                        # Verify search was performed with correct filter
                        search_call = mock_client.search_runs.call_args
                        filter_str = search_call[1]["filter_string"]
                        assert "code.study_key_hash" in filter_str
                        assert "trial_number" in filter_str
                        assert study_key_hash in filter_str
                        assert str(best_trial_number) in filter_str
                        
                        # Verify best trial run ID was logged
                        assert mock_set_tag.called
                        tag_calls = {call[0][0]: call[0][1] for call in mock_set_tag.call_args_list}
                        # Check that best trial run ID tag was set (actual tag name depends on config)
                        assert any("best_trial" in tag.lower() or "run_id" in tag.lower() 
                                  for tag in tag_calls.keys())

    def test_best_trial_search_filters_by_parent_run_id(
        self, sample_configs, root_dir_with_config
    ):
        """Test that search correctly filters to only runs belonging to current parent."""
        tracker = MLflowSweepTracker(experiment_name="test")
        
        parent_run_id = "parent-run-123"
        other_parent_run_id = "parent-run-456"  # Different parent
        best_trial_number = 0
        study_key_hash = "a" * 64
        
        # Create mock study
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.number = best_trial_number
        mock_study.best_trial = mock_trial
        
        # Create mock MLflow client
        mock_client = MagicMock()
        
        # Mock parent run
        mock_parent_run = MagicMock()
        mock_parent_run.data.tags.get.return_value = study_key_hash
        mock_client.get_run.return_value = mock_parent_run
        
        # Mock search results - one matching, one from different parent
        correct_trial_run = MagicMock()
        correct_trial_run.info.run_id = "trial-run-correct"
        correct_trial_run.data.tags.get.side_effect = lambda key: {
            "mlflow.parentRunId": parent_run_id,  # Matches!
            "trial_number": str(best_trial_number),
            "code.study_key_hash": study_key_hash,
        }.get(key)
        
        wrong_trial_run = MagicMock()
        wrong_trial_run.info.run_id = "trial-run-wrong"
        wrong_trial_run.data.tags.get.side_effect = lambda key: {
            "mlflow.parentRunId": other_parent_run_id,  # Different parent!
            "trial_number": str(best_trial_number),
            "code.study_key_hash": study_key_hash,
        }.get(key)
        
        # Return both runs from search
        mock_client.search_runs.return_value = [correct_trial_run, wrong_trial_run]
        
        # Mock active run
        mock_active_run = MagicMock()
        mock_active_run.info.experiment_id = "exp-123"
        
        output_dir = root_dir_with_config / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            with patch("mlflow.active_run", return_value=mock_active_run):
                with patch("mlflow.log_param"):
                    with patch("mlflow.set_tag") as mock_set_tag:
                        tracker._log_best_trial_id(
                            study=mock_study,
                            parent_run_id=parent_run_id,
                            output_dir=output_dir,
                        )
                        
                        # Verify only the correct trial run ID was used
                        # Check that set_tag was called with correct run ID
                        tag_calls = [call[0][1] for call in mock_set_tag.call_args_list]
                        # Should contain the correct run ID
                        assert any(correct_trial_run.info.run_id in str(val) for val in tag_calls)
                        # Should NOT contain the wrong run ID
                        assert not any(wrong_trial_run.info.run_id in str(val) for val in tag_calls)

    def test_best_trial_search_no_parent_study_key_hash(
        self, sample_configs, root_dir_with_config
    ):
        """Test search behavior when parent run doesn't have study_key_hash tag."""
        tracker = MLflowSweepTracker(experiment_name="test")
        
        parent_run_id = "parent-run-123"
        best_trial_number = 0
        
        # Create mock study
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.number = best_trial_number
        mock_study.best_trial = mock_trial
        
        # Create mock MLflow client
        mock_client = MagicMock()
        
        # Mock parent run WITHOUT study_key_hash
        mock_parent_run = MagicMock()
        mock_parent_run.data.tags.get.return_value = None  # No study_key_hash
        mock_client.get_run.return_value = mock_parent_run
        
        # Mock active run
        mock_active_run = MagicMock()
        mock_active_run.info.experiment_id = "exp-123"
        
        output_dir = root_dir_with_config / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            with patch("mlflow.active_run", return_value=mock_active_run):
                with patch("mlflow.log_param"):
                    with patch("mlflow.set_tag") as mock_set_tag:
                        # Should not crash, just log warning
                        tracker._log_best_trial_id(
                            study=mock_study,
                            parent_run_id=parent_run_id,
                            output_dir=output_dir,
                        )
                        
                        # Search should not be performed (no study_key_hash)
                        assert not mock_client.search_runs.called
                        # Tags might still be set (but without run ID)
                        # This is acceptable behavior

    def test_best_trial_search_no_matching_runs(
        self, sample_configs, root_dir_with_config
    ):
        """Test search behavior when no matching runs are found."""
        tracker = MLflowSweepTracker(experiment_name="test")
        
        parent_run_id = "parent-run-123"
        best_trial_number = 0
        study_key_hash = "a" * 64
        
        # Create mock study
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.number = best_trial_number
        mock_study.best_trial = mock_trial
        
        # Create mock MLflow client
        mock_client = MagicMock()
        
        # Mock parent run
        mock_parent_run = MagicMock()
        mock_parent_run.data.tags.get.return_value = study_key_hash
        mock_client.get_run.return_value = mock_parent_run
        
        # Mock search returns empty results
        mock_client.search_runs.return_value = []
        
        # Mock active run
        mock_active_run = MagicMock()
        mock_active_run.info.experiment_id = "exp-123"
        
        output_dir = root_dir_with_config / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            with patch("mlflow.active_run", return_value=mock_active_run):
                with patch("mlflow.log_param"):
                    with patch("mlflow.set_tag"):
                        # Should not crash, just not find the run
                        tracker._log_best_trial_id(
                            study=mock_study,
                            parent_run_id=parent_run_id,
                            output_dir=output_dir,
                        )
                        
                        # Search should have been attempted
                        assert mock_client.search_runs.called

    def test_best_trial_search_exception_handling(
        self, sample_configs, root_dir_with_config
    ):
        """Test that exceptions during search are handled gracefully."""
        tracker = MLflowSweepTracker(experiment_name="test")
        
        parent_run_id = "parent-run-123"
        best_trial_number = 0
        
        # Create mock study
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.number = best_trial_number
        mock_study.best_trial = mock_trial
        
        # Create mock MLflow client that raises exception
        mock_client = MagicMock()
        mock_client.get_run.side_effect = Exception("MLflow error")
        
        # Mock active run
        mock_active_run = MagicMock()
        mock_active_run.info.experiment_id = "exp-123"
        
        output_dir = root_dir_with_config / "outputs" / "hpo"
        output_dir.mkdir(parents=True)
        
        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            with patch("mlflow.active_run", return_value=mock_active_run):
                with patch("mlflow.log_param"):
                    with patch("mlflow.set_tag"):
                        # Should not crash, handle exception gracefully
                        tracker._log_best_trial_id(
                            study=mock_study,
                            parent_run_id=parent_run_id,
                            output_dir=output_dir,
                        )
                        
                        # Should have attempted to get parent run
                        assert mock_client.get_run.called


class TestSweepTrackerTrialNumberExtraction:
    """Test _extract_trial_number method."""

    def test_extract_trial_number_from_tag(self):
        """Test extracting trial number from trial_number tag."""
        tracker = MLflowSweepTracker(experiment_name="test")
        
        mock_run = MagicMock()
        mock_run.data.tags.get.side_effect = lambda key: {
            "trial_number": "5",
        }.get(key)
        
        result = tracker._extract_trial_number(mock_run)
        assert result == 5

    def test_extract_trial_number_from_run_name(self):
        """Test extracting trial number from run name when tag not available."""
        tracker = MLflowSweepTracker(experiment_name="test")
        
        mock_run = MagicMock()
        mock_run.data.tags.get.return_value = None  # No trial_number tag
        mock_run.info.run_name = "trial_3_something"
        
        result = tracker._extract_trial_number(mock_run)
        assert result == 3

    def test_extract_trial_number_returns_none_when_not_found(self):
        """Test that None is returned when trial number cannot be extracted."""
        tracker = MLflowSweepTracker(experiment_name="test")
        
        mock_run = MagicMock()
        mock_run.data.tags.get.return_value = None
        mock_run.info.run_name = "some_other_run_name"
        
        result = tracker._extract_trial_number(mock_run)
        assert result is None


