"""Tests for consolidated MLflow utilities in tracking.mlflow module.

This module tests the new consolidated utilities:
1. Artifact upload utilities (log_artifact_safe, log_artifacts_safe, upload_checkpoint_archive)
2. Run lifecycle utilities (terminate_run_safe, ensure_run_terminated, terminate_run_with_tags)
3. Run creation utilities (create_child_run, create_run_safe, get_or_create_experiment, resolve_experiment_id)
4. URL utilities (get_mlflow_run_url)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import mlflow
from mlflow.tracking import MlflowClient


class TestArtifactUploadUtilities:
    """Test artifact upload utilities."""

    @pytest.fixture
    def mock_mlflow_client(self):
        """Create a mock MLflow client."""
        with patch('mlflow.tracking.MlflowClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def mock_active_run(self):
        """Create a mock active MLflow run."""
        with patch('mlflow.active_run') as mock_active:
            mock_run = MagicMock()
            mock_run.info.run_id = "test-run-id-123"
            mock_active.return_value = mock_run
            yield mock_run

    def test_log_artifact_safe_with_active_run(self, mock_active_run):
        """Test log_artifact_safe with active run."""
        from tracking.mlflow import log_artifact_safe
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            test_file = Path(tmp_file.name)
            test_file.write_text("test content")
        
        try:
            with patch('mlflow.log_artifact') as mock_log:
                result = log_artifact_safe(
                    local_path=test_file,
                    artifact_path="test_artifact",
                    run_id=None,  # Use active run
                )
                
                assert result is True, "Should succeed"
                mock_log.assert_called_once()
        finally:
            test_file.unlink(missing_ok=True)

    def test_log_artifact_safe_with_explicit_run_id(self, mock_mlflow_client):
        """Test log_artifact_safe with explicit run_id."""
        from tracking.mlflow import log_artifact_safe
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            test_file = Path(tmp_file.name)
            test_file.write_text("test content")
        
        try:
            run_id = "explicit-run-id-456"
            result = log_artifact_safe(
                local_path=test_file,
                artifact_path="test_artifact",
                run_id=run_id,
            )
            
            assert result is True, "Should succeed"
            mock_mlflow_client.log_artifact.assert_called_once()
            call_kwargs = mock_mlflow_client.log_artifact.call_args[1]
            assert call_kwargs.get('run_id') == run_id
        finally:
            test_file.unlink(missing_ok=True)

    def test_log_artifact_safe_handles_errors(self, mock_active_run):
        """Test log_artifact_safe handles errors gracefully."""
        from tracking.mlflow import log_artifact_safe
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            test_file = Path(tmp_file.name)
            test_file.write_text("test content")
        
        try:
            with patch('mlflow.log_artifact', side_effect=Exception("Upload failed")):
                result = log_artifact_safe(
                    local_path=test_file,
                    artifact_path="test_artifact",
                    run_id=None,
                )
                
                assert result is False, "Should return False on error"
        finally:
            test_file.unlink(missing_ok=True)

    def test_log_artifacts_safe_with_directory(self, mock_active_run):
        """Test log_artifacts_safe with directory."""
        from tracking.mlflow import log_artifacts_safe
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "file1.txt").write_text("content1")
            (test_dir / "file2.txt").write_text("content2")
            
            with patch('mlflow.log_artifacts') as mock_log:
                result = log_artifacts_safe(
                    local_dir=test_dir,
                    artifact_path="test_dir",
                    run_id=None,
                )
                
                assert result is True, "Should succeed"
                mock_log.assert_called_once()

    def test_upload_checkpoint_archive(self, mock_mlflow_client):
        """Test upload_checkpoint_archive utility."""
        from tracking.mlflow import upload_checkpoint_archive
        
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
            archive_path = Path(tmp_file.name)
            archive_path.write_bytes(b"test archive")
        
        try:
            manifest = {"file_count": 2, "total_size": 200}
            run_id = "test-run-id-789"
            
            # Mock MlflowClient to verify the function works correctly
            with patch('mlflow.tracking.MlflowClient') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client
                result = upload_checkpoint_archive(
                    archive_path=archive_path,
                    manifest=manifest,
                    artifact_path="checkpoint",
                    run_id=run_id,
                )
                
                assert result is True, "Should succeed"
                # Verify that MlflowClient was used (indirectly via log_artifact_safe)
                assert mock_client_class.called, "MlflowClient should be instantiated"
        finally:
            archive_path.unlink(missing_ok=True)


class TestRunLifecycleUtilities:
    """Test run lifecycle management utilities."""

    @pytest.fixture
    def mock_mlflow_client(self):
        """Create a mock MLflow client."""
        with patch('mlflow.tracking.MlflowClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    def test_terminate_run_safe_with_running_run(self, mock_mlflow_client):
        """Test terminate_run_safe with RUNNING run."""
        from tracking.mlflow import terminate_run_safe
        
        run_id = "test-run-id-123"
        mock_run = MagicMock()
        mock_run.info.status = "RUNNING"
        mock_mlflow_client.get_run.return_value = mock_run
        
        result = terminate_run_safe(
            run_id=run_id,
            status="FINISHED",
            check_status=True,
        )
        
        assert result is True, "Should succeed"
        mock_mlflow_client.set_terminated.assert_called_once_with(
            run_id, status="FINISHED"
        )

    def test_terminate_run_safe_with_already_terminated(self, mock_mlflow_client):
        """Test terminate_run_safe with already-terminated run."""
        from tracking.mlflow import terminate_run_safe
        
        run_id = "test-run-id-123"
        mock_run = MagicMock()
        mock_run.info.status = "FINISHED"
        mock_mlflow_client.get_run.return_value = mock_run
        
        result = terminate_run_safe(
            run_id=run_id,
            status="FINISHED",
            check_status=True,
        )
        
        assert result is True, "Should succeed (already terminated)"
        mock_mlflow_client.set_terminated.assert_not_called()

    def test_terminate_run_safe_with_tags(self, mock_mlflow_client):
        """Test terminate_run_safe with tags."""
        from tracking.mlflow import terminate_run_safe
        
        run_id = "test-run-id-123"
        tags = {"tag1": "value1", "tag2": "value2"}
        mock_run = MagicMock()
        mock_run.info.status = "RUNNING"
        mock_mlflow_client.get_run.return_value = mock_run
        
        result = terminate_run_safe(
            run_id=run_id,
            status="FINISHED",
            tags=tags,
            check_status=True,
        )
        
        assert result is True, "Should succeed"
        assert mock_mlflow_client.set_tag.call_count == len(tags), "Should set all tags"
        mock_mlflow_client.set_terminated.assert_called_once()

    def test_terminate_run_with_tags(self, mock_mlflow_client):
        """Test terminate_run_with_tags convenience function."""
        from tracking.mlflow import terminate_run_with_tags
        
        run_id = "test-run-id-123"
        tags = {"tag1": "value1"}
        mock_run = MagicMock()
        mock_run.info.status = "RUNNING"
        mock_mlflow_client.get_run.return_value = mock_run
        
        result = terminate_run_with_tags(
            run_id=run_id,
            status="FINISHED",
            tags=tags,
        )
        
        assert result is True, "Should succeed"
        mock_mlflow_client.set_tag.assert_called()
        mock_mlflow_client.set_terminated.assert_called_once()

    def test_ensure_run_terminated(self, mock_mlflow_client):
        """Test ensure_run_terminated utility."""
        from tracking.mlflow import ensure_run_terminated
        
        run_id = "test-run-id-123"
        mock_run = MagicMock()
        mock_run.info.status = "RUNNING"
        mock_mlflow_client.get_run.return_value = mock_run
        
        result = ensure_run_terminated(
            run_id=run_id,
            expected_status="FINISHED",
        )
        
        assert result is True, "Should succeed"
        mock_mlflow_client.set_terminated.assert_called_once()

    def test_ensure_run_terminated_already_finished(self, mock_mlflow_client):
        """Test ensure_run_terminated with already-finished run."""
        from tracking.mlflow import ensure_run_terminated
        
        run_id = "test-run-id-123"
        mock_run = MagicMock()
        mock_run.info.status = "FINISHED"
        mock_mlflow_client.get_run.return_value = mock_run
        
        result = ensure_run_terminated(
            run_id=run_id,
            expected_status="FINISHED",
        )
        
        assert result is True, "Should succeed (already terminated)"
        mock_mlflow_client.set_terminated.assert_not_called()


class TestRunCreationUtilities:
    """Test run creation utilities."""

    @pytest.fixture
    def mock_mlflow_client(self):
        """Create a mock MLflow client."""
        with patch('mlflow.tracking.MlflowClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    def test_get_or_create_experiment_existing(self):
        """Test get_or_create_experiment with existing experiment."""
        from tracking.mlflow import get_or_create_experiment
        
        experiment_name = "test-experiment"
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        
        with patch('mlflow.get_experiment_by_name', return_value=mock_experiment):
            result = get_or_create_experiment(experiment_name)
            
            assert result == "exp-123", "Should return existing experiment ID"

    def test_get_or_create_experiment_new(self):
        """Test get_or_create_experiment with new experiment."""
        from tracking.mlflow import get_or_create_experiment
        
        experiment_name = "new-experiment"
        
        with patch('mlflow.get_experiment_by_name', return_value=None):
            with patch('mlflow.create_experiment', return_value="exp-456"):
                result = get_or_create_experiment(experiment_name)
                
                assert result == "exp-456", "Should create and return new experiment ID"

    def test_resolve_experiment_id_from_parent(self, mock_mlflow_client):
        """Test resolve_experiment_id from parent run."""
        from tracking.mlflow import resolve_experiment_id
        
        parent_run_id = "parent-run-123"
        mock_run = MagicMock()
        mock_run.info.experiment_id = "exp-789"
        mock_mlflow_client.get_run.return_value = mock_run
        
        result = resolve_experiment_id(parent_run_id=parent_run_id)
        
        assert result == "exp-789", "Should resolve from parent run"

    def test_resolve_experiment_id_from_name(self):
        """Test resolve_experiment_id from experiment name."""
        from tracking.mlflow import resolve_experiment_id
        
        experiment_name = "test-exp"
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-999"
        
        with patch('mlflow.get_experiment_by_name', return_value=mock_experiment):
            result = resolve_experiment_id(experiment_name=experiment_name)
            
            assert result == "exp-999", "Should resolve from experiment name"


class TestURLUtilities:
    """Test URL generation utilities."""

    def test_get_mlflow_run_url_azureml(self):
        """Test get_mlflow_run_url with Azure ML tracking URI."""
        from tracking.mlflow import get_mlflow_run_url
        
        experiment_id = "exp-123"
        run_id = "run-456"
        
        with patch('mlflow.get_tracking_uri', return_value="azureml://eastus.api.azureml.ms/mlflow/v2.0/subscriptions/sub/resourceGroups/rg/providers/Microsoft.MachineLearningServices/workspaces/ws"):
            url = get_mlflow_run_url(experiment_id, run_id)
            
            assert "https://" in url, "Should convert azureml:// to https://"
            assert experiment_id in url, "Should include experiment ID"
            assert run_id in url, "Should include run ID"

    def test_get_mlflow_run_url_standard(self):
        """Test get_mlflow_run_url with standard MLflow tracking URI."""
        from tracking.mlflow import get_mlflow_run_url
        
        experiment_id = "exp-123"
        run_id = "run-456"
        
        with patch('mlflow.get_tracking_uri', return_value="http://localhost:5000"):
            url = get_mlflow_run_url(experiment_id, run_id)
            
            assert "http://localhost:5000" in url, "Should include tracking URI"
            assert experiment_id in url, "Should include experiment ID"
            assert run_id in url, "Should include run ID"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

